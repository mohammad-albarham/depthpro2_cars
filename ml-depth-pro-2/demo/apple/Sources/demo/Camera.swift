//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

@preconcurrency import AVFoundation
import CoreMedia
import Synchronization

/// Where a frame's focal length comes from, in order of precedence.
enum FocalSource: String, Sendable {
    case camera
    case option = "--focal-px"
    case estimate = "estimated"
}

struct CameraFrame: @unchecked Sendable {
    let generation: Int
    let image: CVPixelBuffer
    let captureTime: Double
    let focalPixels: Double
    let focalSource: FocalSource
}

final class Mailbox: Sendable {
    private let frame = Mutex<CameraFrame?>(nil)
    private let signal = DispatchSemaphore(value: 0)

    func putDroppingUnread(_ newFrame: CameraFrame) -> Bool {
        let replaced = frame.withLock { slot in
            defer { slot = newFrame }
            return slot != nil
        }
        if !replaced { signal.signal() }
        return replaced
    }

    func take() -> CameraFrame {
        while true {
            signal.wait()
            if let next = frame.withLock({ slot in defer { slot = nil }; return slot }) {
                return next
            }
        }
    }
}

final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private static let frameRate = 30.0
    private static let frameRateTolerance = 0.1
    private static let framesBeforeWarning = 30
    private static let framesBeforeFallback = 2  // the built-in camera's first frame has no focal length
    private static let focalLengthKey = "CinematicFramingVirtualCameraFocalLength" as CFString

    private let session = AVCaptureSession()
    private let device: AVCaptureDevice
    private let generation: Int
    private let format: AVCaptureDevice.Format
    private let frameSize: CMVideoDimensions
    private let frameDuration: CMTime
    private let queue = DispatchQueue(label: "demo.capture", qos: .userInteractive)
    private let mailbox: Mailbox
    private let metrics: Metrics
    private let fallbackFocalPixels: Double?
    private let fieldOfView: FieldOfView?
    private let lockedMessage: Mutex<String?>
    private var observers: [NSObjectProtocol] = []
    private var framesWithoutFocalLength = 0

    /// `aspect` is the model's width over height, `input` the size the format must cover; `generation` tells this
    /// camera's frames from earlier cameras'.
    init(
        device: AVCaptureDevice, aspect: Double, covering input: CMVideoDimensions, generation: Int, mailbox: Mailbox,
        metrics: Metrics, focalPixels: Double?, fieldOfView: FieldOfView?
    ) throws {
        self.generation = generation
        self.mailbox = mailbox
        self.metrics = metrics
        fallbackFocalPixels = focalPixels
        self.fieldOfView = fieldOfView
        guard let (format, rate) = Self.closestFormat(of: device, covering: input, aspect: aspect) else {
            throw DemoError("\(device.localizedName): no \(Self.requirement(covering: input))")
        }
        self.device = device
        self.format = format
        lockedMessage = Mutex("starting \(device.localizedName)")
        frameSize = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        frameDuration = CMTimeMaximum(
            rate.minFrameDuration,
            CMTimeMinimum(CMTime(value: 1, timescale: Int32(Self.frameRate)), rate.maxFrameDuration))
        super.init()
        AVCaptureDevice.centerStageControlMode = .app
        AVCaptureDevice.isCenterStageEnabled = false
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input), session.canAddOutput(output) else {
            throw DemoError("cannot attach \(device.localizedName) to the capture session")
        }
        session.addInput(input)
        session.addOutput(output)
        observeFailures()
    }

    var message: String? { lockedMessage.withLock { $0 } }

    var summary: String {
        String(format: "%@, %dx%d at %.0f fps", device.localizedName, frameSize.width, frameSize.height, 1 / frameDuration.seconds)
    }

    func start() {
        queue.async { self.startSession() }
    }

    func stop() {
        observers.forEach(NotificationCenter.default.removeObserver)
        observers = []
        queue.async { self.session.stopRunning() }
    }

    private func observeFailures() {
        let center = NotificationCenter.default
        func on(_ name: Notification.Name, _ object: Any, _ react: @escaping @Sendable (Notification) -> Void) {
            observers.append(center.addObserver(forName: name, object: object, queue: nil, using: react))
        }
        on(AVCaptureSession.runtimeErrorNotification, session) { [weak self] note in
            guard let self else { return }
            let error = note.userInfo?[AVCaptureSessionErrorKey] as? Error
            lockedMessage.withLock { $0 = "camera error: \(error?.localizedDescription ?? "unknown"); restarting" }
            queue.async { if !self.session.isRunning { self.startSession() } }
        }
        on(AVCaptureSession.wasInterruptedNotification, session) { [weak self] _ in
            self?.lockedMessage.withLock { $0 = "camera interrupted" }
        }
    }

    private func startSession() {
        session.startRunning()
        // On macOS, startRunning resets the format to the session preset's.
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.activeFormat = format
            device.activeVideoMinFrameDuration = frameDuration
            device.activeVideoMaxFrameDuration = frameDuration
        } catch {
            note("cannot set the camera format: \(error.localizedDescription)")
        }
    }

    static func requirement(covering input: CMVideoDimensions) -> String {
        "format covering \(input.width)x\(input.height) at \(Int(frameRate)) fps"
    }

    static func isUsable(_ device: AVCaptureDevice, covering input: CMVideoDimensions) -> Bool {
        closestFormat(of: device, covering: input, aspect: 1) != nil  // any aspect: the requirement is on size and rate
    }

    /// The format covering `input` whose aspect is closest to `aspect`, then the smallest.
    private static func closestFormat(
        of device: AVCaptureDevice, covering input: CMVideoDimensions, aspect: Double
    ) -> (format: AVCaptureDevice.Format, rate: AVFrameRateRange)? {
        let candidates = device.formats.compactMap {
            format -> (format: AVCaptureDevice.Format, rate: AVFrameRateRange, mismatch: Double, width: Int32)? in
            let size = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            guard size.width >= input.width, size.height >= input.height,
                let rate = format.videoSupportedFrameRateRanges
                    .filter({ $0.maxFrameRate >= frameRate - frameRateTolerance })
                    .min(by: { abs($0.maxFrameRate - frameRate) < abs($1.maxFrameRate - frameRate) })
            else { return nil }
            return (format, rate, abs(Double(size.width) / Double(size.height) - aspect), size.width)
        }
        return candidates.min { ($0.mismatch, $0.width) < ($1.mismatch, $1.width) }.map { ($0.format, $0.rate) }
    }

    func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let image = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let reported = reportedFocalPixels(of: sampleBuffer)
        framesWithoutFocalLength = reported == nil ? framesWithoutFocalLength + 1 : 0
        guard let (focalPixels, focalSource) = reported.map({ ($0, FocalSource.camera) }) ?? unreportedFocal(of: image)
        else {
            if fieldOfView == nil, framesWithoutFocalLength == Self.framesBeforeWarning {
                let message =
                    "\(device.localizedName) reports no focal length; relaunch with --focal-px, "
                    + "or put \(FieldOfView.packageName) next to the model to estimate it"
                note(message)
                lockedMessage.withLock { $0 = message }
            }
            return
        }
        let captureTime = hostTime(sampleBuffer)
        let dropped = mailbox.putDroppingUnread(
            CameraFrame(
                generation: generation, image: image, captureTime: captureTime, focalPixels: focalPixels,
                focalSource: focalSource))
        metrics.camera(at: captureTime, dropped: dropped)
        lockedMessage.withLock { $0 = nil }
    }

    func captureOutput(
        _ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        metrics.camera(at: hostTime(sampleBuffer), dropped: true)
    }

    /// `--focal-px`, else the estimate, once the camera has sent frames without a focal length.
    private func unreportedFocal(of image: CVPixelBuffer) -> (Double, FocalSource)? {
        guard framesWithoutFocalLength >= Self.framesBeforeFallback else { return nil }
        let width = Double(CVPixelBufferGetWidth(image))
        return fallbackFocalPixels.map { ($0 * width / Double(frameSize.width), .option) }
            ?? estimatedFocalPixels(of: image).map { ($0, .estimate) }
    }

    /// The focal length `fieldOfView` estimated for this camera; until it has, `image` may add a frame to it.
    private func estimatedFocalPixels(of image: CVPixelBuffer) -> Double? {
        guard let fieldOfView else { return nil }
        let estimate = fieldOfView.estimate(for: device.uniqueID, from: image)
        let (width, height) = (CVPixelBufferGetWidth(image), CVPixelBufferGetHeight(image))
        if let focal = estimate.focalPixels(width: width, height: height) { return focal }
        lockedMessage.withLock { $0 = "estimating focal length \(estimate.samples.count)/\(FieldOfView.sampleCount)" }
        return nil
    }

    // Undocumented attachment: the built-in camera sets it to its focal length in pixels.
    private func reportedFocalPixels(of sampleBuffer: CMSampleBuffer) -> Double? {
        let value = CMGetAttachment(sampleBuffer, key: Self.focalLengthKey, attachmentModeOut: nil)
        let focal = (value as? NSNumber)?.doubleValue ?? (value as? String).flatMap(Double.init)
        return focal.flatMap { $0 > 0 ? $0 : nil }
    }

    private func hostTime(_ sampleBuffer: CMSampleBuffer) -> Double {
        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        guard let clock = session.synchronizationClock else { return time.seconds }
        return CMSyncConvertTime(time, from: clock, to: CMClockGetHostTimeClock()).seconds
    }
}
