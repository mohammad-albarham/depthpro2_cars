//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import CoreML
import Foundation

/// Metal objects are safe to use from any thread, which the build of a new resolution needs.
extension GPU: @unchecked Sendable {}

/// One input size of a multi-resolution package: its function `res_<width>x<height>`.
struct Resolution: Equatable, Sendable {
    let function: String
    let width: Int
    let height: Int

    var title: String { "\(width)×\(height)" }

    /// The resolutions among `functions`, the names of a package's functions, smallest first; none unless it has
    /// several.
    static func list(functions: [String]) -> [Resolution] {
        let all = functions.compactMap { name -> Resolution? in
            let size = name.dropFirst(4).split(separator: "x").compactMap { Int($0) }
            guard name.hasPrefix("res_"), size.count == 2 else { return nil }
            return Resolution(function: name, width: size[0], height: size[1])
        }
        return all.count < 2 ? [] : all.sorted { ($0.width * $0.height, $0.width) < ($1.width * $1.height, $1.width) }
    }
}

/// What depends on the depth model's input size: the model with its buffers.
struct DepthSetup: @unchecked Sendable {
    let spec: ModelSpec
    let runner: ModelRunner
    let resolutions: [Resolution]

    /// Loads `function` of `compiled` (nil: its default) and everything built on its size.
    static func make(compiled: CompiledPackage, function: String?, gpu: GPU) throws -> DepthSetup {
        let (model, spec) = try loadModel(from: compiled, function: function)
        let runner = try ModelRunner(gpu: gpu, model: model, spec: spec)
        return DepthSetup(spec: spec, runner: runner, resolutions: Resolution.list(functions: compiled.functionNames))
    }
}
