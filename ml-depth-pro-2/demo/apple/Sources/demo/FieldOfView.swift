//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import CoreML
import CoreVideo
import Metal
import Synchronization

let fieldOfViewShader = """

    // torch's antialiased bilinear resize of the frame's centred 16:9 crop: a triangle filter widened by the scale.
    kernel void resize_planar_antialiased(texture2d<half, access::read> rgb [[texture(0)]],
                                          texture2d<half, access::write> planar [[texture(1)]],
                                          uint2 gid [[thread_position_in_grid]]) {
        uint2 size = uint2(planar.get_width(), planar.get_height() / 3);
        if (gid.x >= size.x || gid.y >= 3 * size.y) return;
        uint channel = gid.y / size.y;
        int2 frame = int2(rgb.get_width(), rgb.get_height());
        int2 crop = min(frame, int2(frame.y * 16 / 9, frame.x * 9 / 16));
        int2 origin = (frame - crop) / 2;
        float2 scale = float2(crop) / float2(size);
        float2 support = max(scale, 1.0);
        float2 centre = scale * (float2(gid.x, gid.y % size.y) + 0.5);
        int2 low = max(int2(centre - support + 0.5), int2(0));
        int2 high = min(int2(centre + support + 0.5), crop);
        float sum = 0.0, total = 0.0;
        for (int y = low.y; y < high.y; y++) {
            float wy = max(1.0 - abs((float(y) - centre.y + 0.5) / support.y), 0.0);
            for (int x = low.x; x < high.x; x++) {
                float w = wy * max(1.0 - abs((float(x) - centre.x + 0.5) / support.x), 0.0);
                sum += w * float(rgb.read(uint2(origin + int2(x, y)))[channel]);
                total += w;
            }
        }
        planar.write(half4(sum / total), gid);
    }

    """

/// A camera's field of view from the `python -m demo.export fov` package: the per-axis median over `sampleCount`
/// frames' centred 16:9 crops.
final class FieldOfView: @unchecked Sendable {
    static let sampleCount = 16
    static let packageName = "depthpro2-fov_512x896.mlpackage"
    private static let blackLevel: UInt8 = 8

    struct Estimate: Sendable {
        private(set) var samples: [SIMD2<Double>] = []
        /// Horizontal and vertical field of view in degrees, once `sampleCount` frames are in.
        private(set) var degrees: SIMD2<Double>?

        mutating func add(_ fov: SIMD2<Double>) {
            guard degrees == nil else { return }
            samples.append(fov)
            guard samples.count == FieldOfView.sampleCount else { return }
            // The lower middle value of an even count, as torch.median takes.
            func median(_ values: [Double]) -> Double { values.sorted()[(values.count - 1) / 2] }
            degrees = SIMD2(median(samples.map(\.x)), median(samples.map(\.y)))
        }

        /// The focal length in pixels of a `width` x `height` frame, from its 16:9 crop; a crop keeps it.
        func focalPixels(width: Int, height: Int) -> Double? {
            degrees.map { 0.5 * Double(min(height, width * 9 / 16)) / tan($0.y * .pi / 360) }
        }
    }

    let inputSize: (width: Int, height: Int)
    private let model: MLModel
    private let gpu: GPU
    private let pipeline: MTLComputePipelineState
    private let textureCache: CVMetalTextureCache
    private let planar: MTLTexture
    private let features: MLFeatureProvider
    private let queue = DispatchQueue(label: "demo.fov", qos: .userInitiated)
    private let busy = Atomic<Bool>(false)
    private let estimates = Mutex<[String: Estimate]>([:])

    /// The FoV package in the directory of `package`, if there is one.
    static func beside(_ package: URL, gpu: GPU) throws -> FieldOfView? {
        let fov = package.deletingLastPathComponent().appendingPathComponent(packageName)
        return FileManager.default.fileExists(atPath: fov.path) ? try FieldOfView(package: fov, gpu: gpu) : nil
    }

    /// Loads `package` on the CPU and GPU, whatever `--compute-units` says of the depth model.
    init(package: URL, gpu: GPU) throws {
        let compiled = try CompiledPackage(package, computeUnits: .gpu)
        model = try compiled.load()
        compiled.discard()
        guard let shape = model.modelDescription.inputDescriptionsByName["frame"]?.multiArrayConstraint?.shape,
            shape.count == 4
        else { throw DemoError("\(package.lastPathComponent) has no frame input of shape (1, 3, H, W)") }
        let (height, width) = (shape[2].intValue, shape[3].intValue)
        inputSize = (width, height)
        self.gpu = gpu
        guard let function = gpu.library.makeFunction(name: "resize_planar_antialiased") else {
            throw DemoError("missing resize_planar_antialiased kernel")
        }
        pipeline = try gpu.device.makeComputePipelineState(function: function)
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, gpu.device, nil, &cache)
        guard let cache else { throw DemoError("CVMetalTextureCacheCreate failed") }
        textureCache = cache
        let input = try makeHalfSurface(width: width, height: 3 * height)
        planar = try gpu.texture(input, format: .r16Float, usage: .shaderWrite)
        let frame = MLMultiArray(pixelBuffer: input, shape: [1, 3, height, width].map { NSNumber(value: $0) })
        features = try MLDictionaryFeatureProvider(dictionary: ["frame": MLFeatureValue(multiArray: frame)])
    }

    /// The estimate for camera `id`, kept for the session; until complete, `frame` (BGRA) may add a sample.
    func estimate(for id: String, from frame: CVPixelBuffer) -> Estimate {
        let estimate = estimates.withLock { $0[id, default: Estimate()] }
        guard estimate.degrees == nil, !Self.isBlack(frame),
            busy.compareExchange(expected: false, desired: true, ordering: .acquiring).exchanged
        else { return estimate }
        guard (try? resize(frame)) != nil else {
            busy.store(false, ordering: .releasing)
            return estimate
        }
        queue.async { [self] in
            if let fov = try? predict() { estimates.withLock { $0[id, default: Estimate()].add(fov) } }
            busy.store(false, ordering: .releasing)
        }
        return estimate
    }

    /// Drops the estimate for camera `id`, so its next frames take a new one.
    func forget(_ id: String) { estimates.withLock { $0[id] = nil } }

    /// Whether `frame` (BGRA) is black, as from a covered or starting camera.
    private static func isBlack(_ frame: CVPixelBuffer) -> Bool {
        CVPixelBufferLockBaseAddress(frame, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(frame, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(frame) else { return false }
        let rowBytes = CVPixelBufferGetBytesPerRow(frame)
        for y in stride(from: 0, to: CVPixelBufferGetHeight(frame), by: 8) {
            let row = (base + y * rowBytes).assumingMemoryBound(to: UInt8.self)
            for x in stride(from: 0, to: 4 * CVPixelBufferGetWidth(frame), by: 32)
            where max(row[x], row[x + 1], row[x + 2]) > blackLevel {
                return false
            }
        }
        return true
    }

    private func resize(_ frame: CVPixelBuffer) throws {
        var cvTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, frame, nil, .bgra8Unorm, CVPixelBufferGetWidth(frame),
            CVPixelBufferGetHeight(frame), 0, &cvTexture)
        guard let cvTexture, let source = CVMetalTextureGetTexture(cvTexture),
            let commands = gpu.queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder()
        else { throw DemoError("cannot resize the frame for the FoV package") }
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(planar, index: 1)
        encoder.dispatchThreads(
            MTLSize(width: planar.width, height: planar.height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        CVMetalTextureCacheFlush(textureCache, 0)
        if let error = commands.error { throw error }
    }

    private func predict() throws -> SIMD2<Double> {
        guard let fov = try model.prediction(from: features).featureValue(for: "fov")?.multiArrayValue,
            fov.count == 2, fov[0].doubleValue.isFinite, fov[1].doubleValue.isFinite
        else { throw DemoError("the FoV package gave no field of view") }
        return SIMD2(fov[0].doubleValue, fov[1].doubleValue)
    }
}
