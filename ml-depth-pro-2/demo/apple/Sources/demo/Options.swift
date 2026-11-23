//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import Foundation

struct Options {
    private static let usage = """
        usage: demo PACKAGE [options]

        PACKAGE is a Core ML package from `python -m demo.export stream`, for example depthpro2-s_w4.mlpackage. Put
        depthpro2-fov.mlpackage, from `python -m demo.export fov`, next to it: a camera that reports no focal length
        then gets one estimated from its frames.

        options:
          --resolution WxH     one of the package's input sizes, which the menu in the window switches between; default: the package's default
          --compute-units U    where Core ML runs the model: gpu (default) or ane
          --focal-px PIXELS    focal length in pixels, for a camera that reports none
          --min-depth METERS   near end of the color bar
          --max-depth METERS   far end of the color bar
        """

    let package: URL
    /// The function of a multi-resolution package to run, `res_<W>x<H>`; nil runs its default.
    let function: String?
    let computeUnits: ComputeUnits
    let focalPixels: Double?
    let minDepth: Double?
    let maxDepth: Double?

    init(arguments: [String]) throws {
        var package: URL?
        var function: String?
        var computeUnits = ComputeUnits.gpu
        var focalPixels: Double?
        var minDepth: Double?
        var maxDepth: Double?
        var rest = arguments.dropFirst()
        while let argument = rest.popFirst() {
            switch argument {
            case "-h", "--help":
                print(Self.usage)
                exit(0)
            case "--resolution":
                guard let value = rest.popFirst(), value.wholeMatch(of: /[1-9][0-9]*x[1-9][0-9]*/) != nil else {
                    throw DemoError("--resolution takes WxH, one of the package's input sizes")
                }
                function = "res_\(value)"
            case "--compute-units":
                guard let value = rest.popFirst().flatMap(ComputeUnits.init(rawValue:)) else {
                    let choices = ComputeUnits.allCases.map(\.rawValue).joined(separator: ", ")
                    throw DemoError("--compute-units takes \(choices)")
                }
                computeUnits = value
            case "--focal-px": focalPixels = try Self.number(after: argument, in: &rest)
            case "--min-depth": minDepth = try Self.number(after: argument, in: &rest)
            case "--max-depth": maxDepth = try Self.number(after: argument, in: &rest)
            default:
                guard !argument.hasPrefix("-"), package == nil else { throw DemoError(Self.usage) }
                package = URL(fileURLWithPath: argument)
            }
        }
        guard let package else { throw DemoError(Self.usage) }
        guard FileManager.default.fileExists(atPath: package.path) else {
            throw DemoError("no such package: \(package.path)")
        }
        if let minDepth, let maxDepth, minDepth >= maxDepth {
            throw DemoError("--min-depth must be less than --max-depth")
        }
        (self.package, self.function, self.computeUnits) = (package, function, computeUnits)
        (self.focalPixels, self.minDepth, self.maxDepth) = (focalPixels, minDepth, maxDepth)
    }

    private static func number(after option: String, in rest: inout ArraySlice<String>) throws -> Double {
        guard let value = rest.popFirst().flatMap(Double.init), value.isFinite, value > 0 else {
            throw DemoError("\(option) takes a positive number")
        }
        return value
    }
}
