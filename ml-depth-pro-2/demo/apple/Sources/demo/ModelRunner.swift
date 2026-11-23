//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import CoreML
import CoreVideo
import Metal
import QuartzCore
import Synchronization

struct FrameTimes: Sendable {
    var seq = 0
    var captureTime = 0.0
    var preprocessStart = 0.0
    var preprocessEnd = 0.0
    var stepStart = 0.0
    var stepEnd = 0.0
}

final class ResultSlot: @unchecked Sendable {
    let inverseDepth: CVPixelBuffer
    let inverseDepthTexture: MTLTexture
    let camera: MTLTexture
    let predictionOptions = MLPredictionOptions()
    var times = FrameTimes()
    var metricScale: Float = 0
    var depthRange = SIMD2<Float>.zero
    fileprivate var readers = 0

    init(gpu: GPU, spec: ModelSpec) throws {
        inverseDepth = try makeHalfSurface(width: spec.width, height: spec.height)
        inverseDepthTexture = try gpu.texture(inverseDepth, format: .r16Float, usage: .shaderRead)
        camera = try gpu.privateTexture(.bgra8Unorm, width: spec.width, height: spec.height)
        let backing = MLMultiArray(
            pixelBuffer: inverseDepth, shape: [1, spec.height, spec.width].map { NSNumber(value: $0) })
        predictionOptions.outputBackings = ["inverse_depth": backing]
    }
}

final class ResultPool: @unchecked Sendable {
    private let slots: [ResultSlot]
    private let condition = NSCondition()
    private var latest: ResultSlot?

    init(gpu: GPU, spec: ModelSpec, count: Int) throws {
        slots = try (0..<count).map { _ in try ResultSlot(gpu: gpu, spec: spec) }
    }

    func acquireWritable() -> ResultSlot {
        condition.lock()
        defer { condition.unlock() }
        while true {
            if let slot = slots.first(where: { $0 !== latest && $0.readers == 0 }) { return slot }
            condition.wait()
        }
    }

    func publish(_ slot: ResultSlot) {
        condition.lock()
        latest = slot
        condition.unlock()
    }

    func acquireLatest() -> ResultSlot? {
        condition.lock()
        defer { condition.unlock() }
        latest?.readers += 1
        return latest
    }

    func release(_ slot: ResultSlot) {
        condition.lock()
        slot.readers -= 1
        condition.signal()
        condition.unlock()
    }
}

final class ModelRunner: @unchecked Sendable {
    private static let slotCount = 4

    private static let count = Atomic<Int>(0)

    /// Tells this runner from another, whatever becomes of the other.
    let id = count.add(1, ordering: .relaxed).newValue
    let stream: DepthStream
    let pool: ResultPool
    private let preprocess: Preprocess

    init(gpu: GPU, model: MLModel, spec: ModelSpec) throws {
        stream = try DepthStream(model: model, spec: spec)
        preprocess = try Preprocess(gpu: gpu, input: stream.input, height: spec.height, width: spec.width)
        pool = try ResultPool(gpu: gpu, spec: spec, count: Self.slotCount)
    }

    func process(_ image: CVPixelBuffer, into slot: ResultSlot) throws {
        var times = FrameTimes()
        (times.preprocessStart, times.preprocessEnd) = try preprocess.run(image, display: slot.camera)
        times.stepStart = CACurrentMediaTime()
        try stream.step(predictionOptions: slot.predictionOptions)
        times.stepEnd = CACurrentMediaTime()
        slot.times = times
    }
}
