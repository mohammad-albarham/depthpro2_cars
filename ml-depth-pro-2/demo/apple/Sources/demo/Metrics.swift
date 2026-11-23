//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import Foundation
import QuartzCore
import Synchronization

final class Metrics: Sendable {
    struct Snapshot {
        var cameraFPS = 0.0
        var processedFPS = 0.0
        var displayedFPS = 0.0
        var droppedPercent = 0.0
        var preprocessMs = 0.0
        var stepMs = 0.0
        var latencyP50Ms = 0.0
        var latencyP95Ms = 0.0
    }

    private struct State {
        var cameraTimes: [Double] = []
        var droppedTimes: [Double] = []
        var processedTimes: [Double] = []
        var displayed: [(time: Double, latencyMs: Double)] = []
        var preprocessMs = 0.0
        var stepMs = 0.0
    }

    private let state = Mutex(State())
    private static let span = 2.0
    private static let smoothing = 0.1

    func camera(at time: Double, dropped: Bool) {
        state.withLock {
            $0.cameraTimes.append(time)
            if dropped { $0.droppedTimes.append(time) }
            $0.cameraTimes.removeAll { $0 < time - Self.span }
            $0.droppedTimes.removeAll { $0 < time - Self.span }
        }
    }

    func processed(_ times: FrameTimes) {
        state.withLock {
            $0.processedTimes.append(times.stepEnd)
            $0.processedTimes.removeAll { $0 < times.stepEnd - Self.span }
            Self.smooth(&$0.preprocessMs, 1000 * (times.preprocessEnd - times.preprocessStart))
            Self.smooth(&$0.stepMs, 1000 * (times.stepEnd - times.stepStart))
        }
    }

    func displayed(_ times: FrameTimes, at display: Double) {
        state.withLock {
            $0.displayed.append((display, 1000 * (display - times.captureTime)))
            $0.displayed.removeAll { $0.time < display - Self.span }
        }
    }

    func snapshot() -> Snapshot {
        let start = CACurrentMediaTime() - Self.span
        return state.withLock {
            let camera = $0.cameraTimes.filter { $0 >= start }
            let dropped = $0.droppedTimes.filter { $0 >= start }
            let displayed = $0.displayed.filter { $0.time >= start }
            let latency = displayed.map(\.latencyMs).sorted()
            return Snapshot(
                cameraFPS: Self.rate(camera), processedFPS: Self.rate($0.processedTimes.filter { $0 >= start }),
                displayedFPS: Self.rate(displayed.map(\.time)),
                droppedPercent: camera.isEmpty ? 0 : 100 * Double(dropped.count) / Double(camera.count),
                preprocessMs: $0.preprocessMs, stepMs: $0.stepMs,
                latencyP50Ms: Self.quantile(latency, 0.5), latencyP95Ms: Self.quantile(latency, 0.95))
        }
    }

    private static func rate(_ times: [Double]) -> Double {
        guard let first = times.first, let last = times.last, last > first else { return 0 }
        return Double(times.count - 1) / (last - first)
    }

    private static func quantile(_ sorted: [Double], _ q: Double) -> Double {
        sorted.isEmpty ? 0 : sorted[Int(q * Double(sorted.count - 1))]
    }

    private static func smooth(_ value: inout Double, _ sample: Double) {
        value = value == 0 ? sample : value + smoothing * (sample - value)
    }
}
