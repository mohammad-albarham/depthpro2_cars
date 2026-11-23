//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import CoreVideo
import Metal
import MetalPerformanceShaders

let preprocessShader = """
    kernel void split_planar(texture2d<half, access::read> rgb [[texture(0)]],
                             texture2d<half, access::write> planar [[texture(1)]],
                             texture2d<half, access::write> display [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
        uint h = rgb.get_height();
        if (gid.x >= rgb.get_width() || gid.y >= h) return;
        half4 c = clamp(rgb.read(gid), 0.0h, 1.0h);
        planar.write(half4(c.r), gid);
        planar.write(half4(c.g), uint2(gid.x, gid.y + h));
        planar.write(half4(c.b), uint2(gid.x, gid.y + 2 * h));
        display.write(half4(c.rgb, 1.0h), gid);
    }

    """

func fillScale(from source: (width: Int, height: Int), to target: (width: Int, height: Int)) -> Double {
    max(Double(target.width) / Double(source.width), Double(target.height) / Double(source.height))
}

final class Preprocess {
    private let gpu: GPU
    private let pipeline: MTLComputePipelineState
    private let scaler: MPSImageLanczosScale
    private let textureCache: CVMetalTextureCache
    private let resized: MTLTexture
    private let planar: MTLTexture

    init(gpu: GPU, input: CVPixelBuffer, height: Int, width: Int) throws {
        self.gpu = gpu
        guard let function = gpu.library.makeFunction(name: "split_planar") else {
            throw DemoError("missing split_planar kernel")
        }
        pipeline = try gpu.device.makeComputePipelineState(function: function)
        scaler = MPSImageLanczosScale(device: gpu.device)
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, gpu.device, nil, &cache)
        guard let cache else { throw DemoError("CVMetalTextureCacheCreate failed") }
        textureCache = cache
        resized = try gpu.privateTexture(.rgba16Float, width: width, height: height)
        planar = try gpu.texture(input, format: .r16Float, usage: .shaderWrite)
    }

    func run(_ image: CVPixelBuffer, display: MTLTexture) throws -> (start: Double, end: Double) {
        let (srcWidth, srcHeight) = (CVPixelBufferGetWidth(image), CVPixelBufferGetHeight(image))
        var cvTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, image, nil, .bgra8Unorm, srcWidth, srcHeight, 0,
            &cvTexture)
        guard let cvTexture, let source = CVMetalTextureGetTexture(cvTexture),
            let commands = gpu.queue.makeCommandBuffer()
        else { throw DemoError("cannot wrap the camera frame as a texture") }

        let (width, height) = (Double(resized.width), Double(resized.height))
        let scale = fillScale(from: (srcWidth, srcHeight), to: (resized.width, resized.height))
        var transform = MPSScaleTransform(
            scaleX: scale, scaleY: scale,
            translateX: (width - Double(srcWidth) * scale) / 2,
            translateY: (height - Double(srcHeight) * scale) / 2)
        withUnsafePointer(to: &transform) { pointer in
            scaler.scaleTransform = pointer
            scaler.encode(commandBuffer: commands, sourceTexture: source, destinationTexture: resized)
            scaler.scaleTransform = nil
        }

        guard let encoder = commands.makeComputeCommandEncoder() else {
            throw DemoError("cannot make a compute encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(resized, index: 0)
        encoder.setTexture(planar, index: 1)
        encoder.setTexture(display, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: resized.width, height: resized.height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        CVMetalTextureCacheFlush(textureCache, 0)
        if let error = commands.error { throw error }
        return (commands.gpuStartTime, commands.gpuEndTime)
    }
}
