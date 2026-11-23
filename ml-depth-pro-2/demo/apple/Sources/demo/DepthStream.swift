//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import CoreML
import CoreVideo
import Foundation
import Synchronization

struct ModelSpec {
    let preset: String
    let height: Int
    let width: Int
    let window: Int

    /// The preset and window from the package's metadata, the size from the loaded function's
    /// input: a multi-resolution package's metadata describes only its default function.
    init(model: MLModel, name: String) throws {
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        func entry(_ key: String) throws -> String {
            guard let value = metadata["demo.\(key)"] else { throw DemoError("\(name) has no metadata demo.\(key)") }
            return value
        }
        func integer(_ key: String) throws -> Int {
            guard let value = Int(try entry(key)) else { throw DemoError("\(name): demo.\(key) is not an integer") }
            return value
        }
        preset = try entry("preset")
        window = try integer("window")
        guard let shape = model.modelDescription.inputDescriptionsByName["frame"]?.multiArrayConstraint?.shape,
              shape.count == 4
        else { throw DemoError("\(name) has no input frame (1, 3, H, W)") }
        (height, width) = (shape[2].intValue, shape[3].intValue)
        guard model.modelDescription.outputDescriptionsByName["inverse_depth"] != nil else {
            throw DemoError("\(name) has no output inverse_depth: it is not a package from `python -m demo.export stream`")
        }
    }

    var label: String { "\(preset) \(width)x\(height) W\(window)" }  // width first, as the camera's size
}

final class DepthStream {
    let spec: ModelSpec
    let input: CVPixelBuffer
    private let model: MLModel
    private let features: MLFeatureProvider
    private var state: MLState?

    init(model: MLModel, spec: ModelSpec) throws {
        self.model = model
        self.spec = spec
        input = try makeHalfSurface(width: spec.width, height: 3 * spec.height)
        let frame = MLMultiArray(
            pixelBuffer: input, shape: [1, 3, spec.height, spec.width].map { NSNumber(value: $0) })
        features = try MLDictionaryFeatureProvider(dictionary: ["frame": MLFeatureValue(multiArray: frame)])
        reset()
    }

    /// Forgets the earlier frames, so the next step starts a new window.
    func reset() {
        state = spec.window > 1 ? model.makeState() : nil
    }

    func step(predictionOptions: MLPredictionOptions) throws {
        if let state {
            _ = try model.prediction(from: features, using: state, options: predictionOptions)
        } else {
            _ = try model.prediction(from: features, options: predictionOptions)
        }
    }
}

/// A package compiled once, so that each of its functions loads without compiling again.
final class CompiledPackage: @unchecked Sendable {
    private static let all = Mutex<[CompiledPackage]>([])

    let name: String
    let computeUnits: ComputeUnits
    /// The functions the package has, as Core ML reports them; one, `main`, for a package of one size.
    let functionNames: [String]
    private let url: URL

    /// Where the compiled packages of every run live, one folder per process, so that the next run can tell which
    /// a crash or a force quit left behind.
    private static let folder = FileManager.default.temporaryDirectory.appendingPathComponent("depthpro2-demo")
    private static var runFolder: URL { folder.appendingPathComponent(String(getpid())) }

    init(_ package: URL, computeUnits: ComputeUnits) throws {
        name = package.lastPathComponent
        self.computeUnits = computeUnits
        let compiled = try MLModel.compileModel(at: package)
        let home = Self.runFolder.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        url = home.appendingPathComponent(compiled.lastPathComponent)
        try FileManager.default.moveItem(at: compiled, to: url)
        functionNames = Self.listFunctions(of: url)
        Self.all.withLock { $0.append(self) }
    }

    /// Removes what runs that are no longer there left in the temporary folder.
    static func removeStale() {
        let runs = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        for run in runs where Int32(run).map({ kill($0, 0) == -1 && errno == ESRCH }) ?? true {
            try? FileManager.default.removeItem(at: folder.appendingPathComponent(run))
        }
    }

    /// Removes the compiled copy once the model is loaded, for a package that is not loaded again.
    func discard() {
        Self.all.withLock { $0.removeAll { $0 === self } }
        remove()
    }

    private final class Names: @unchecked Sendable { var value: [String] = [] }

    /// The functions of the compiled package at `url`, which Core ML only lists asynchronously.
    private static func listFunctions(of url: URL) -> [String] {
        let names = Names()
        let done = DispatchSemaphore(value: 0)
        Task.detached {
            do {
                names.value = try await MLModelAsset(url: url).functionNames
            } catch {
                note("cannot list the package's functions: \(error.localizedDescription)")
            }
            done.signal()
        }
        done.wait()
        return names.value
    }

    /// Removes every compiled package, as the app quits or fails.
    static func removeAll() {
        all.withLock { $0.forEach { $0.remove() } }
        try? FileManager.default.removeItem(at: runFolder)
    }

    nonisolated(unsafe) private static var signalSources: [DispatchSourceSignal] = []

    /// Removes the compiled packages when the app is interrupted (control-C) or terminated, which otherwise leaves
    /// them in the temporary folder, as large as the model. The handler runs on its own queue, so that it works
    /// while the main thread is busy, loading the model, or stuck.
    static func removeOnInterrupt() {
        for number in [SIGINT, SIGTERM] {
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .global(qos: .userInteractive))
            source.setEventHandler {
                removeAll()
                _exit(128 + number)
            }
            source.resume()
            signalSources.append(source)
        }
    }

    /// `function` of the package (nil: its default) on the compute units it was compiled for.
    func load(function: String? = nil) throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits.mlUnits
        configuration.functionName = function
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    private func remove() { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
}

func loadModel(from compiled: CompiledPackage, function: String?) throws -> (model: MLModel, spec: ModelSpec) {
    let model: MLModel
    do {
        model = try compiled.load(function: function)
    } catch {
        // Only a function the package lacks is a matter of its resolutions; any other failure says what it was.
        if let function, !compiled.functionNames.contains(function) {
            throw DemoError("\(compiled.name) has no \(function): \(resolutions(of: compiled))")
        }
        throw error
    }
    return (model, try ModelSpec(model: model, name: compiled.name))
}

/// The resolutions a package has, for an error message.
private func resolutions(of compiled: CompiledPackage) -> String {
    let list = Resolution.list(functions: compiled.functionNames)
    guard list.isEmpty else { return "its resolutions are " + list.map { "\($0.width)x\($0.height)" }.joined(separator: ", ") }
    guard let model = try? compiled.load() else { return "it does not load" }
    let spec = try? ModelSpec(model: model, name: compiled.name)
    return "it has one resolution" + (spec.map { ", \($0.width)x\($0.height)" } ?? "")
}

/// Where Core ML may run a model, as `--compute-units` names it.
enum ComputeUnits: String, CaseIterable, Sendable {
    case gpu, ane

    var mlUnits: MLComputeUnits {
        switch self {
        case .gpu: .cpuAndGPU
        case .ane: .cpuAndNeuralEngine
        }
    }

    /// For the startup note.
    var label: String {
        switch self {
        case .gpu: "the CPU and GPU"
        case .ane: "the CPU and Neural Engine"
        }
    }
}
