//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

@preconcurrency import AVFoundation
import AppKit
import MetalKit

@MainActor
final class LiveApp: NSObject, NSApplicationDelegate {
    private static let refreshInterval = 0.25
    private static let inset: CGFloat = 12
    private static let maxScreenFraction = 0.95

    private let options: Options
    private let gpu: GPU
    private let compiled: CompiledPackage
    private let resolutions: [Resolution]
    private var spec: ModelSpec
    private let worker: Worker
    private let fieldOfView: FieldOfView?
    private let mailbox = Mailbox()
    private let metrics = Metrics()
    private let window: NSWindow
    private let view: InputView
    private let renderer: Renderer
    private let statsPill = Pill()
    private let focalPill = Pill()
    private let statusPill = Pill()
    private let colorBar = ColorBar()
    private let picker = Picker()
    private let resolutionPicker = Picker()
    private var switching: String?
    private var memoryWarned = false
    private var deviceToResume: AVCaptureDevice?
    private lazy var discovery = AVCaptureDevice.DiscoverySession(
        deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera], mediaType: .video, position: .unspecified)
    private var discoveryObservation: NSKeyValueObservation?
    private var cameras: [AVCaptureDevice] = []
    private var device: AVCaptureDevice?
    private var camera: Camera?
    private var generation = 0
    private var announcedGeneration = 0

    init(options: Options) throws {
        self.options = options
        let gpu = try GPU()
        self.gpu = gpu
        note("loading \(options.package.lastPathComponent)")
        let loadStart = Date()
        let compiled = try CompiledPackage(options.package, computeUnits: options.computeUnits)
        let setup = try DepthSetup.make(compiled: compiled, function: options.function, gpu: gpu)
        (self.compiled, resolutions) = (compiled, setup.resolutions)
        if resolutions.isEmpty { compiled.discard() }  // nothing to switch to: no need to keep the compiled copy
        let spec = setup.spec
        self.spec = spec
        note(String(format: "%@ loaded in %.1f s, Core ML on %@", spec.label, Date().timeIntervalSince(loadStart), options.computeUnits.label))
        worker = Worker(
            runner: setup.runner, mailbox: mailbox, metrics: metrics, minDepth: options.minDepth, maxDepth: options.maxDepth)
        fieldOfView = options.focalPixels == nil ? try FieldOfView.beside(options.package, gpu: gpu) : nil
        if fieldOfView != nil { note("a camera that reports no focal length gets one estimated by the FoV package") }

        let aspect = NSSize(width: 2 * spec.width, height: spec.height)
        let width = min(aspect.width, (NSScreen.main?.visibleFrame.width ?? aspect.width) * Self.maxScreenFraction)
        view = InputView(
            frame: NSRect(x: 0, y: 0, width: width, height: width * aspect.height / aspect.width), device: gpu.device)
        view.colorPixelFormat = .bgra8Unorm
        view.isPaused = true
        view.enableSetNeedsDisplay = false
        renderer = try Renderer(gpu: gpu, view: view, worker: worker, metrics: metrics)
        view.delegate = renderer

        for pill in [statsPill, focalPill, statusPill, colorBar, picker, resolutionPicker] as [NSView] {
            view.addSubview(pill)
        }
        let depthHalf = NSLayoutGuide()
        view.addLayoutGuide(depthHalf)
        let inset = Self.inset
        NSLayoutConstraint.activate([
            depthHalf.leadingAnchor.constraint(equalTo: view.centerXAnchor),
            depthHalf.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            statsPill.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: inset),
            statsPill.topAnchor.constraint(equalTo: view.topAnchor, constant: inset),
            focalPill.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -inset),
            focalPill.topAnchor.constraint(equalTo: view.topAnchor, constant: inset),
            statusPill.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            statusPill.topAnchor.constraint(equalTo: view.topAnchor, constant: inset),
            colorBar.centerXAnchor.constraint(equalTo: depthHalf.centerXAnchor),
            colorBar.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -inset),
            colorBar.widthAnchor.constraint(equalTo: depthHalf.widthAnchor, multiplier: 0.5),
            picker.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: inset),
            picker.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -inset),
            resolutionPicker.leadingAnchor.constraint(equalTo: picker.leadingAnchor),
            resolutionPicker.bottomAnchor.constraint(equalTo: picker.topAnchor, constant: -6),
        ])

        window = NSWindow(
            contentRect: view.frame, styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered, defer: false)
        window.title = "Depth Pro 2 demo"
        window.appearance = NSAppearance(named: .darkAqua)
        window.titlebarAppearsTransparent = true
        window.backgroundColor = .black
        window.contentAspectRatio = aspect
        window.contentView = view
        super.init()
        view.onKey = { [unowned self] in handle($0) }
        picker.onPick = { [unowned self] in use(cameras[$0]) }
        resolutionPicker.onPick = { [unowned self] in switchResolution(to: $0) }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate()
        window.makeFirstResponder(view)
        showResolutions()
        worker.start { [view] in
            DispatchQueue.main.async { MainActor.assumeIsolated { view.draw() } }
        }
        AVCaptureDevice.requestAccess(for: .video) { granted in
            Task { @MainActor in self.startCamera(granted: granted) }
        }
        Timer.scheduledTimer(withTimeInterval: Self.refreshInterval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.refresh() }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    // Skips exit's teardown, which breaks Core ML's in-flight GPU work and logs "mutex lock failed".
    func applicationWillTerminate(_ notification: Notification) {
        CompiledPackage.removeAll()
        _exit(0)
    }

    private func startCamera(granted: Bool) {
        guard granted else {
            fail(DemoError("camera access denied; allow it in System Settings > Privacy & Security > Camera"))
        }
        discoveryObservation = discovery.observe(\.devices) { @Sendable [weak self] _, _ in
            Task { @MainActor in self?.listCameras() }
        }
        listCameras()
    }

    /// The size a camera format must cover for no frame to be upscaled: the model's input, and the FoV package's if loaded.
    private func inputSize(width: Int, height: Int) -> CMVideoDimensions {
        let fov = fieldOfView?.inputSize ?? (width: 0, height: 0)
        return CMVideoDimensions(width: Int32(max(width, fov.width)), height: Int32(max(height, fov.height)))
    }

    private var inputSize: CMVideoDimensions { inputSize(width: spec.width, height: spec.height) }

    /// Lists the usable cameras in the picker, the computer's own first, then those built into a connected display, and
    /// starts on `preferred` if it is usable, else on the first.
    private func listCameras(preferring preferred: AVCaptureDevice? = nil) {
        cameras = discovery.devices.filter { Camera.isUsable($0, covering: inputSize) }.sorted { a, b in
            let (aRank, bRank) = (a.rank, b.rank)
            return aRank != bRank ? aRank < bRank : a.localizedName.localizedStandardCompare(b.localizedName) == .orderedAscending
        }
        if device == nil, switching == nil {
            guard let first = preferred.flatMap({ cameras.contains($0) ? $0 : nil }) ?? cameras.first else {
                fail(DemoError("no camera with a \(Camera.requirement(covering: inputSize))"))
            }
            use(first)
        }
        picker.show(cameras.map(\.localizedName), selected: device.flatMap(cameras.firstIndex))
    }

    /// Replaces the camera; the worker drops the old camera's frames.
    private func use(_ device: AVCaptureDevice) {
        guard device != self.device, switching == nil else { return }
        camera?.stop()
        self.device = device
        generation += 1
        worker.follow(generation: generation)
        do {
            let camera = try Camera(
                device: device, aspect: Double(spec.width) / Double(spec.height), covering: inputSize, generation: generation,
                mailbox: mailbox, metrics: metrics, focalPixels: options.focalPixels, fieldOfView: fieldOfView)
            note("camera \(camera.summary)")
            camera.start()
            self.camera = camera
        } catch {
            fail(error)
        }
    }

    /// Lists the package's resolutions in their menu, with the one running selected; a package of one size has none.
    private func showResolutions() {
        let current = resolutions.firstIndex { ($0.width, $0.height) == (spec.width, spec.height) }
        resolutionPicker.show(resolutions.map(\.title), selected: current)
        resolutionPicker.isHidden = resolutions.count < 2
    }

    /// What a switch of size came to.
    private enum Switch {
        case switched(DepthSetup)
        case stayed(DepthSetup, because: Error)
    }

    /// Runs the model at `resolutions[index]`. The camera stops and the model is let go of first, so that the memory
    /// of one model is enough, then the new one is built on another queue, and the camera restarts so that the
    /// stream, the focal length and the color bar start anew. If the new size does not load, the old one does again.
    private func switchResolution(to index: Int) {
        let resolution = resolutions[index]
        guard switching == nil, (resolution.width, resolution.height) != (spec.width, spec.height) else {
            showResolutions()
            return
        }
        // A camera that cannot cover the new size would leave the app with none, so it stays as it is.
        let needed = inputSize(width: resolution.width, height: resolution.height)
        if let device, !Camera.isUsable(device, covering: needed) {
            note("cannot switch to \(resolution.title): \(device.localizedName) has no \(Camera.requirement(covering: needed))")
            showResolutions()
            return
        }
        switching = "loading \(resolution.title)"
        let previous = resolutions.first { ($0.width, $0.height) == (spec.width, spec.height) }
        deviceToResume = device
        camera?.stop()
        camera = nil
        device = nil
        let (compiled, worker, gpu) = (compiled, worker, gpu)
        let start = Date()
        DispatchQueue.global(qos: .userInitiated).async {
            worker.suspend()
            let outcome: Result<Switch, Error>
            do {
                outcome = .success(.switched(try DepthSetup.make(compiled: compiled, function: resolution.function, gpu: gpu)))
            } catch {
                // The old size is built again; if that fails too there is no model to run.
                outcome = Result { .stayed(try DepthSetup.make(compiled: compiled, function: previous?.function, gpu: gpu), because: error) }
            }
            DispatchQueue.main.async {
                MainActor.assumeIsolated { self.finishSwitch(outcome, to: resolution, since: start) }
            }
        }
    }

    private func finishSwitch(_ outcome: Result<Switch, Error>, to resolution: Resolution, since start: Date) {
        switching = nil
        switch outcome {
        case .failure(let error):
            fail(error)
        case .success(.stayed(let setup, let error)):
            worker.replace(runner: setup.runner)
            note("cannot switch to \(resolution.title): \(error.localizedDescription); still at \(spec.label)")
        case .success(.switched(let setup)):
            spec = setup.spec
            worker.replace(runner: setup.runner)
            let aspect = NSSize(width: 2 * spec.width, height: spec.height)
            window.contentAspectRatio = aspect
            let width = view.frame.width
            window.setContentSize(NSSize(width: width, height: width * aspect.height / aspect.width))
            note(String(format: "%@ loaded in %.1f s", spec.label, Date().timeIntervalSince(start)))
        }
        if deviceToResume != nil { listCameras(preferring: deviceToResume) }
        deviceToResume = nil
        showResolutions()
    }

    private func handle(_ event: NSEvent) {
        switch event.charactersIgnoringModifiers?.lowercased() {
        case "c": worker.recalibrate()
        case "f":
            guard let device, let fieldOfView else { break }
            fieldOfView.forget(device.uniqueID)
            worker.recalibrate()
        case "q": NSApp.terminate(nil)
        default: break
        }
    }

    private func refresh() {
        let current = worker.currentStatus()
        showStats(metrics.snapshot())
        showFocal(of: current)
        colorBar.show(current.depthRange)
        showStatus(of: current)
        announceStart(of: current)
    }

    private func announceStart(of status: Worker.Status) {
        guard status.generation != announcedGeneration, let focal = status.focalPixels, let size = status.frameSize,
            let range = status.depthRange
        else { return }
        let first = announcedGeneration == 0
        announcedGeneration = status.generation
        let fixedColors = options.minDepth != nil && options.maxDepth != nil
        let colorBar = fixedColors
            ? String(format: "%.3g to %.3g m", range.lowerBound, range.upperBound) : "calibrated on the scene"
        note(
            String(
                format: "focal length %.0f px at %dx%d (%@), color bar %@", focal, size.width, size.height,
                status.focalSource.rawValue, colorBar))
        guard first else { return }
        let keys = [
            fixedColors ? nil : "c to recalibrate the color bar",
            fieldOfView == nil ? nil : "f to re-estimate the focal length",
            "q to quit",
        ]
        note(keys.compactMap { $0 }.joined(separator: ", "))
    }

    private func showFocal(of status: Worker.Status) {
        focalPill.isHidden = status.focalPixels == nil || status.frameSize == nil
        if let focal = status.focalPixels, let size = status.frameSize {
            focalPill.show(
                String(format: "f %.0f px", focal),
                String(format: "%@ · %d×%d", status.focalSource.rawValue, size.width, size.height))
        }
    }

    /// A message once the app holds more than this share of the Mac's memory: past it, the Mac starts to swap.
    private static let memoryWarningShare = 0.5

    private var memoryWarning: String? {
        let total = Double(ProcessInfo.processInfo.physicalMemory)
        guard let used = physicalFootprint().map({ Double($0) }), used > Self.memoryWarningShare * total else { return nil }
        return String(format: "using %.0f GB of this Mac's %.0f GB; a smaller size or model runs better", used / 1e9, total / 1e9)
    }

    private func showStatus(of status: Worker.Status) {
        let memory = memoryWarning
        if let memory, !memoryWarned { note(memory) }
        memoryWarned = memory != nil
        let messages = [status.error, switching, camera?.message, memory].compactMap { $0 }
        statusPill.isHidden = messages.isEmpty
        statusPill.show(messages.joined(separator: "\n"))
    }

    private func showStats(_ stats: Metrics.Snapshot) {
        statsPill.show(
            spec.label,
            String(
                format: """
                    camera   %5.1f fps   dropped %4.1f%%
                    model    %5.1f fps   preprocess %4.1f ms   step %5.1f ms
                    display  %5.1f fps   latency p50 %4.0f ms   p95 %4.0f ms
                    """,
                stats.cameraFPS, stats.droppedPercent, stats.processedFPS, stats.preprocessMs, stats.stepMs,
                stats.displayedFPS, stats.latencyP50Ms, stats.latencyP95Ms))
    }
}

final class InputView: MTKView {
    var onKey: ((NSEvent) -> Void)?

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with event: NSEvent) {
        onKey?(event)
    }
}

@MainActor
func runLive(options: Options) throws {
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)
    let delegate = try LiveApp(options: options)
    app.delegate = delegate
    app.run()
}

extension AVCaptureDevice {
    /// 0 for the computer's own camera, 1 for the other built-in kind (a connected display's, which also reports
    /// `.builtInWideAngleCamera` but is a USB video class device), 2 for the rest.
    fileprivate var rank: Int {
        guard deviceType == .builtInWideAngleCamera else { return 2 }
        return modelID.hasPrefix("UVC") ? 1 : 0
    }
}

/// The memory the app holds, GPU buffers included, as Activity Monitor's Memory column shows it.
private func physicalFootprint() -> UInt64? {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? info.phys_footprint : nil
}
