//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import CoreVideo
import Foundation
import Synchronization

final class Worker: @unchecked Sendable {
    private static let calibrationSeconds = 1.5

    struct Status: Sendable {
        var generation = 0
        var error: String?
        var focalPixels: Double?
        var focalSource = FocalSource.camera
        var frameSize: (width: Int, height: Int)?
        var depthRange: ClosedRange<Double>?
    }

    private let status = Mutex(Status())
    private let recalibration = Mutex(false)
    private let runnerBox: Mutex<ModelRunner?>
    private let processing = NSLock()
    private let mailbox: Mailbox
    private let metrics: Metrics
    private let minDepth: Double?
    private let maxDepth: Double?

    init(runner: ModelRunner, mailbox: Mailbox, metrics: Metrics, minDepth: Double?, maxDepth: Double?) {
        runnerBox = Mutex(runner)
        self.mailbox = mailbox
        self.metrics = metrics
        self.minDepth = minDepth
        self.maxDepth = maxDepth
    }

    func start(onResult: @escaping @Sendable () -> Void) {
        let thread = Thread { [self] in run(onResult: onResult) }
        thread.name = "demo.inference"
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    /// The results of the model running, none while it is replaced.
    var pool: ResultPool? { runnerBox.withLock { $0?.pool } }

    /// Runs the next frames on `runner`, a model of another size; its stream starts anew.
    func replace(runner: ModelRunner) { runnerBox.withLock { $0 = runner } }

    /// Lets go of the model once the frame in progress is done, so that its memory is free before the next one is
    /// built; frames are dropped until `replace`.
    func suspend() {
        runnerBox.withLock { $0 = nil }
        processing.lock()
        processing.unlock()
    }

    func recalibrate() { recalibration.withLock { $0 = true } }

    /// Drops other generations' frames; `generation`'s first frame restarts the stream and the color bar.
    func follow(generation: Int) { status.withLock { $0 = Status(generation: generation) } }

    func currentStatus() -> Status { status.withLock { $0 } }

    private func run(onResult: @Sendable () -> Void) {
        var runnerID: Int?
        var seq = 0
        var streamingGeneration = 0
        var smoother = DepthRangeSmoother()
        var calibratingUntil = -Double.infinity
        while true {
            let frame = mailbox.take()
            processing.lock()
            defer { processing.unlock() }
            // Held only for this frame, so that a model that is replaced is freed, not kept for the next frame.
            guard let runner = runnerBox.withLock({ $0 }) else { continue }
            if runner.id != runnerID {
                runnerID = runner.id
                streamingGeneration = 0
            }
            guard frame.generation == status.withLock({ $0.generation }) else { continue }
            if frame.generation != streamingGeneration {
                runner.stream.reset()
                smoother = DepthRangeSmoother()
                streamingGeneration = frame.generation
            }
            if takeRecalibrationRequest() || smoother.range == nil {
                calibratingUntil = frame.captureTime + Self.calibrationSeconds
            }
            do {
                let slot = runner.pool.acquireWritable()
                try runner.process(frame.image, into: slot)
                slot.times.seq = seq
                slot.times.captureTime = frame.captureTime
                metrics.processed(slot.times)
                let scale = metricScale(focalPixels: frame.focalPixels, frame: frame.image, spec: runner.stream.spec)
                if frame.captureTime < calibratingUntil, let measured = measuredDepthRange(in: slot.inverseDepth, scale: scale) {
                    smoother.update(with: measured, at: frame.captureTime)
                }
                let range = try colorRange(following: smoother.range)
                slot.metricScale = Float(scale)
                slot.depthRange = SIMD2(Float(range.lowerBound), Float(range.upperBound))
                runner.pool.publish(slot)
                status.withLock {
                    guard $0.generation == frame.generation else { return }
                    $0.error = nil
                    $0.focalPixels = frame.focalPixels
                    $0.focalSource = frame.focalSource
                    $0.frameSize = (CVPixelBufferGetWidth(frame.image), CVPixelBufferGetHeight(frame.image))
                    $0.depthRange = range
                }
                onResult()
                seq += 1
            } catch {
                status.withLock { $0.error = error.localizedDescription }
            }
        }
    }

    private func takeRecalibrationRequest() -> Bool {
        recalibration.withLock { requested in
            defer { requested = false }
            return requested
        }
    }

    private func colorRange(following tracked: ClosedRange<Double>?) throws -> ClosedRange<Double> {
        guard let near = minDepth ?? tracked?.lowerBound, let far = maxDepth ?? tracked?.upperBound else {
            throw DemoError("no depth in this frame; pass --min-depth and --max-depth")
        }
        guard near < far else {
            throw DemoError(
                String(format: "the color range %.3g to %.3g m is empty; pass --min-depth and --max-depth together", near, far))
        }
        return near...far
    }
}
