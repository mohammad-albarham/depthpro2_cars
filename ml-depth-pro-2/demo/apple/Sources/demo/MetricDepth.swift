//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import CoreVideo

func metricScale(focalPixels: Double, frame: CVPixelBuffer, spec: ModelSpec) -> Double {
    let source = (CVPixelBufferGetWidth(frame), CVPixelBufferGetHeight(frame))
    let focalInModelPixels = focalPixels * fillScale(from: source, to: (spec.width, spec.height))
    return focalInModelPixels / Double(spec.width)
}

func measuredDepthRange(in inverseDepth: CVPixelBuffer, scale: Double) -> ClosedRange<Double>? {
    let (step, lowQuantile, highQuantile) = (8, 0.05, 0.95)
    CVPixelBufferLockBaseAddress(inverseDepth, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(inverseDepth, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(inverseDepth) else { return nil }
    let rowBytes = CVPixelBufferGetBytesPerRow(inverseDepth)
    var meters: [Double] = []
    for y in stride(from: 0, to: CVPixelBufferGetHeight(inverseDepth), by: step) {
        let row = (base + y * rowBytes).assumingMemoryBound(to: Float16.self)
        for x in stride(from: 0, to: CVPixelBufferGetWidth(inverseDepth), by: step) where row[x].isFinite && row[x] > 0 {
            meters.append(scale / Double(row[x]))
        }
    }
    guard !meters.isEmpty else { return nil }
    meters.sort()
    let last = Double(meters.count - 1)
    return meters[Int(lowQuantile * last)]...meters[Int(highQuantile * last)]
}

struct DepthRangeSmoother {
    private static let timeConstant = 0.5
    private static let longestStep = 0.1

    private var smoothed: SIMD2<Double>?
    private var time = 0.0

    var range: ClosedRange<Double>? { smoothed.map { exp($0.x)...exp($0.y) } }

    mutating func update(with measured: ClosedRange<Double>, at newTime: Double) {
        let target = SIMD2(log(measured.lowerBound), log(measured.upperBound))
        let previous = smoothed ?? target
        let step = min(max(0, newTime - time), Self.longestStep)
        let weight = smoothed == nil ? 1 : 1 - exp(-step / Self.timeConstant)
        smoothed = previous + weight * (target - previous)
        time = newTime
    }
}
