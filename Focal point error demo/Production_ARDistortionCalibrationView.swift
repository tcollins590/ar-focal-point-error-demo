//
//  ARDistortionCalibrationView.swift  (clean rebuild)
//
//  Guided camera calibration, stance-major flow:
//    1. Print/tape the QR code.
//    2. Auto-lock the target (point at the code, no taps).
//    3. For each DISTANCE (chips: 4 / 6 / 8 / 10 ft):
//         stand there, then PAN the phone so the code visits each target
//         position (5 down the middle, 3 left, 3 right, + far corners at the
//         far stances). Each position's ring fills like a dial; when the
//         sweep is complete, step to the next distance.
//    4. Solve (health-gated). Failure reopens the grid slightly and resumes.
//
//  State machine (single source of truth — Phase):
//    acquiring → moveToStance(s) → sweeping(s, p) → … → solving → done
//  Cell = (stance, position); progression is gated ONLY by cell credits, so
//  UI state can never disagree with coverage state.
//

import SwiftUI
import RealityKit
import ARKit
import Vision
import CoreImage.CIFilterBuiltins

// MARK: - Engine

final class ARDistortionCalibrationEngine: NSObject, ObservableObject, ARSessionDelegate {
    static let qrPayload = "ARCAL-TARGET"

    enum Phase: Equatable {
        case acquiring
        case moveToStance
        case sweeping
        case solving
        case done
    }

    // ---- protocol definition ----
    // 4/7/10 ft build the radius-distance grid; the 4th pass stands at 8 ft
    // with FOCUS LOCKED at infinity — measuring k1 at the capture-relevant
    // lens position (~0.81) directly. The code is mildly defocused there;
    // the low-density full-page code keeps detection alive.
    static let stances: [Double] = [1.2, 2.1, 3.0]
    static let farFocusStance = 2
    static let farFocusLensPosition: Float = 0.81
    static func stanceLabel(_ i: Int) -> String {
        i == farFocusStance ? "\(stanceFt(i)) ft ∞" : "\(stanceFt(i)) ft"
    }
    static func stanceFt(_ i: Int) -> Int { Int((stances[i] * 3.28084).rounded()) }
    // Tight enough that "4 ft" really samples the 4 ft lens state: +0.35 m
    // let users hover at 5+ ft and the solve lost its low-lens leg entirely
    // (measured lensLo 0.753 vs 0.702 on a proper run).
    static let stanceTolerance = 0.20                             // meters

    /// Pan path per sweep (snake order to minimize tilt travel):
    /// center column top→bottom, right column bottom→top, left column
    /// top→bottom. The two extreme corners join at the far stances only
    /// (the code physically fits near screen corners only from a distance).
    static let basePositions: [(pos: CGPoint, say: String)] = [
        (CGPoint(x: 0.50, y: 0.06), "top"),
        (CGPoint(x: 0.50, y: 0.28), "upper middle"),
        (CGPoint(x: 0.50, y: 0.50), "center"),
        (CGPoint(x: 0.50, y: 0.72), "lower middle"),
        (CGPoint(x: 0.50, y: 0.94), "bottom"),
        (CGPoint(x: 0.85, y: 0.90), "lower right"),
        (CGPoint(x: 0.85, y: 0.50), "right side"),
        (CGPoint(x: 0.85, y: 0.10), "upper right"),
        (CGPoint(x: 0.15, y: 0.10), "upper left"),
        (CGPoint(x: 0.15, y: 0.50), "left side"),
        (CGPoint(x: 0.15, y: 0.90), "lower left"),
    ]
    static let cornerPositions: [(pos: CGPoint, say: String)] = [
        (CGPoint(x: 0.91, y: 0.045), "far top-right corner"),
        (CGPoint(x: 0.09, y: 0.955), "far bottom-left corner"),
    ]
    static func positions(forStance s: Int) -> [(pos: CGPoint, say: String)] {
        s >= 1 ? basePositions + cornerPositions : basePositions
    }
    static let baseCellQuota = 10

    // ---- published state (all mutated on main) ----
    @Published var phase: Phase = .acquiring
    @Published var stanceIndex = 0
    @Published var positionIndex = 0
    @Published var cellFill: Double = 0            // current cell 0…1
    @Published var stanceDone: [Bool] = ARDistortionCalibrationEngine.stances.map { _ in false }
    @Published var sweepProgressText = ""          // "Position 3 of 11"
    @Published var positionDoneFlags: [Bool] = []
    @Published var preLockHint = "Point the camera at the printed code"
    @Published var qrScreenPoint: CGPoint?
    @Published var qrHalfSizePt: Double = 0
    @Published var distanceM: Double = 0
    @Published var motionOK = true
    @Published var acceptingNow = false
    @Published var failureBanner: String?
    @Published var finished: ARDistortionUserCalibration?
    @Published var trackingReady = false
    @Published var settleFill: Double = 0        // 0…1 while arming a position

    weak var arView: ARView?
    let calibrator = ARDistortionSelfCalibrator()

    // ---- internals ----
    private let visionQueue = DispatchQueue(label: "ar.cal.vision")
    private var visionBusy = false
    private var lastVisionTime: TimeInterval = 0
    private var target: simd_float3?
    private var imageAnchor: ARImageAnchor?
    private var autoLockCounter = 0
    private var lastAutoPoint: CGPoint?
    private var visionStableCount = 0
    private var lastVisionPoint: CGPoint?
    private var predRejectStreak = 0
    private var lockVerifyErrs: [Double] = []
    private var lockVerified = false
    private var prevTransform: simd_float4x4?
    private var prevTimestamp: TimeInterval = 0
    private var angularVelocity: Double = 0
    private var translationSpeed: Double = 0
    private var lastAcceptTime: CFAbsoluteTime = 0
    /// Vision-measured correction applied to the 60 fps projected dot.
    private var visionOffset = CGSize.zero
    private var estimatedQRDistance: Double = 0
    private var hudFrameCounter = 0
    private var inStanceFrames = 0
    private var trackingNormalFrames = 0
    private var inRingSince: CFAbsoluteTime?
    private var advancePending = false
    private var cellQuota = ARDistortionCalibrationEngine.baseCellQuota
    /// cells[stance][position] = credited samples
    private var cells: [[Int]] = ARDistortionCalibrationEngine.stances.indices.map {
        [Int](repeating: 0, count: ARDistortionCalibrationEngine.positions(forStance: $0).count)
    }
    private var bannerGeneration = 0
    private var hiResBusy = false
    private var hiResSampleCount = 0
    private var flowLogHandle: FileHandle?
    private var flowLogStart: CFAbsoluteTime = 0

    // The view reports the active ring's screen center each frame so cell
    // credit and ring color use the SAME geometry the user sees.
    var currentRingCenter: CGPoint = .zero

    func flog(_ msg: String) {
        let line = String(format: "t+%7.1fs | %@\n", CFAbsoluteTimeGetCurrent() - flowLogStart, msg)
        visionQueue.async { [weak self] in
            if let d = line.data(using: .utf8) { try? self?.flowLogHandle?.write(contentsOf: d) }
        }
    }

    // ---- QR assets ----
    static func makeReferenceImage() -> ARReferenceImage? {
        guard let cg = makeQRCGImage() else { return nil }
        let ref = ARReferenceImage(cg, orientation: .up, physicalWidth: 0.19)
        ref.name = "ar_cal_target"
        return ref
    }

    static func makeQRCGImage() -> CGImage? {
        let filter = CIFilter.qrCodeGenerator()
        filter.message = qrPayload.data(using: .ascii)!
        // Lowest density (fewest, largest modules) maximizes detection range —
        // a full-page print reaches ~15 ft, covering capture-focus lens values.
        filter.correctionLevel = "L"
        guard let out = filter.outputImage?.transformed(by: CGAffineTransform(scaleX: 30, y: 30)) else { return nil }
        return CIContext().createCGImage(out, from: out.extent)
    }

    // ---- lifecycle ----
    func start(arView: ARView) {
        self.arView = arView
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        // Enable high-res still capture: each completed circle also samples
        // the PHOTO pipeline, so the calibration measures the pipeline that
        // production measurements actually use.
        if let fmt = ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing {
            config.videoFormat = fmt
        }
        if let ref = Self.makeReferenceImage() {
            config.detectionImages = [ref]
            config.maximumNumberOfTrackedImages = 1
        }
        arView.session.delegate = self
        arView.session.run(config)
        calibrator.reset()
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("guided_cal_flow.log")
        FileManager.default.createFile(atPath: url.path, contents: Data())
        flowLogHandle = try? FileHandle(forWritingTo: url)
        flowLogStart = CFAbsoluteTimeGetCurrent()
        flog("session start")
    }

    func stop() {
        setFocusLocked(false)
        flog("session stop phase=\(phase)")
        visionQueue.async { [weak self] in try? self?.flowLogHandle?.close() }
        arView?.session.pause()
    }

    // ---- grid helpers ----
    private func cellDone(_ s: Int, _ p: Int) -> Bool { cells[s][p] >= cellQuota }

    private func firstIncomplete(stance s: Int) -> Int? {
        cells[s].indices.first { !cellDone(s, $0) }
    }

    private func nextIncompleteStance() -> Int? {
        Self.stances.indices.first { firstIncomplete(stance: $0) != nil }
    }

    private func refreshDerived() {
        stanceDone = Self.stances.indices.map { firstIncomplete(stance: $0) == nil }
        if phase == .sweeping {
            let total = Self.positions(forStance: stanceIndex).count
            let done = cells[stanceIndex].filter { $0 >= cellQuota }.count
            sweepProgressText = "Position \(min(done + 1, total)) of \(total)"
            cellFill = min(1, Double(cells[stanceIndex][positionIndex]) / Double(cellQuota))
            positionDoneFlags = cells[stanceIndex].map { $0 >= cellQuota }
        }
    }

    static let trackRange: ClosedRange<Double> = 0.9...3.4   // meters shown on the gauge
    static func trackFrac(_ meters: Double) -> Double {
        max(0, min(1, (meters - trackRange.lowerBound) / (trackRange.upperBound - trackRange.lowerBound)))
    }
    var stanceDistance: Double { Self.stances[stanceIndex] }
    var inStance: Bool { abs(distanceM - stanceDistance) <= Self.stanceTolerance }
    var currentSay: String { Self.positions(forStance: stanceIndex)[positionIndex].say }
    var currentPos: CGPoint { Self.positions(forStance: stanceIndex)[positionIndex].pos }

    private func setFocusLocked(_ locked: Bool) {
        guard let d = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera else { return }
        do {
            try d.lockForConfiguration()
            if locked {
                d.setFocusModeLocked(lensPosition: Self.farFocusLensPosition)
            } else if d.isFocusModeSupported(.continuousAutoFocus) {
                d.focusMode = .continuousAutoFocus
            }
            d.unlockForConfiguration()
            flog(locked ? "focus LOCKED at \(Self.farFocusLensPosition)" : "focus unlocked (auto)")
        } catch {
            flog("focus lock failed: \(error)")
        }
    }

    // ---- transitions (all on main) ----
    private func enterMoveToStance(_ s: Int) {
        stanceIndex = s
        inStanceFrames = 0
        phase = .moveToStance
        setFocusLocked(s == Self.farFocusStance)
        refreshDerived()
        flog("phase moveToStance -> \(Self.stanceLabel(s))")
    }

    private func enterSweeping(position p: Int) {
        positionIndex = p
        inRingSince = nil
        settleFill = 0
        phase = .sweeping
        refreshDerived()
        flog("phase sweeping stance=\(Self.stanceFt(stanceIndex))ft pos=\(p) '\(currentSay)' cell=\(cells[stanceIndex][p])/\(cellQuota)")
    }

    private func advanceAfterCell() {
        if let p = firstIncomplete(stance: stanceIndex) {
            enterSweeping(position: p)
            return
        }
        if let s = nextIncompleteStance() {
            enterMoveToStance(s)
            return
        }
        beginSolve()
    }

    /// Manual "skip this position": defers the cell, moves to the next.
    func skipPosition() {
        guard phase == .sweeping else { return }
        let n = cells[stanceIndex].count
        for step in 1...n {
            let idx = (positionIndex + step) % n
            if !cellDone(stanceIndex, idx) {
                enterSweeping(position: idx)
                return
            }
        }
        advanceAfterCell()
    }

    private func beginSolve() {
        guard phase != .solving else { return }
        guard calibrator.samples.count >= 250 else {
            // Grid complete but sample floor unmet (heavy rejection): reopen.
            cellQuota += 2
            refreshDerived()
            if let s = nextIncompleteStance() { enterMoveToStance(s) }
            return
        }
        setFocusLocked(false)
        phase = .solving
        flog("SOLVE start n=\(calibrator.samples.count)")
        let model = ARProjectionCorrector.deviceModelIdentifier
        visionQueue.async { [weak self] in
            guard let self else { return }
            let status = self.calibrator.solve(deviceModel: model)
            var diag = self.calibrator.lastSolveDiagnostics
            diag["solved"] = { if case .solved = status { return 1 } else { return 0 } }()
            if let data = try? JSONSerialization.data(withJSONObject: diag, options: [.prettyPrinted]) {
                let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                    .appendingPathComponent("guided_cal_debug.json")
                try? data.write(to: url)
            }
            if case .solved(let cal) = status { cal.save() }
            DispatchQueue.main.async {
                switch status {
                case .solved(let cal):
                    self.flog("SOLVE ok rms=\(String(format: "%.2f", cal.rmsPx))")
                    // Hand the fresh calibration to the live corrector — the
                    // settings UI and all projections read from its cache.
                    ARProjectionCorrector.shared.applyUserCalibration(cal)
                    self.phase = .done
                    self.finished = cal
                case .failed(let why):
                    self.flog("SOLVE failed: \(why)")
                    self.calibrator.resumeCollecting()
                    self.failureBanner = why
                    self.bannerGeneration += 1
                    let gen = self.bannerGeneration
                    DispatchQueue.main.asyncAfter(deadline: .now() + 6) { [weak self] in
                        if self?.bannerGeneration == gen { self?.failureBanner = nil }
                    }
                    // Reopen the grid a little everywhere and continue.
                    self.cellQuota += 4
                    self.refreshDerived()
                    if let s = self.nextIncompleteStance() {
                        self.enterMoveToStance(s)
                    } else {
                        self.phase = .sweeping
                    }
                case .collecting:
                    self.phase = .sweeping
                }
            }
        }
    }

    /// One full-resolution sample per completed circle: detects the code in a
    /// captureHighResolutionFrame and feeds it to the same solver (per-sample
    /// intrinsics make mixed resolutions first-class).
    private func captureHiResSample() {
        guard !hiResBusy, let session = arView?.session, let tw = target else { return }
        hiResBusy = true
        let av = angularVelocity
        let ts = translationSpeed
        session.captureHighResolutionFrame { [weak self] frame, _ in
            guard let self else { return }
            guard let frame else {
                DispatchQueue.main.async { self.hiResBusy = false; self.flog("hi-res capture returned nil frame") }
                return
            }
            let camera = frame.camera
            let K = camera.intrinsics
            let res = camera.imageResolution
            let raw = camera.projectPoint(tw, orientation: .landscapeRight, viewportSize: res)
            let camT = camera.transform
            let lens = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera?.lensPosition ?? 0.7
            let pixelBuffer = frame.capturedImage
            self.visionQueue.async {
                let request = VNDetectBarcodesRequest()
                request.symbologies = [.qr]
                try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
                var obs: CGPoint?
                if let qr = request.results?.first(where: { $0.payloadStringValue == Self.qrPayload }) {
                    let p1 = qr.topLeft, p2 = qr.bottomRight, p3 = qr.topRight, p4 = qr.bottomLeft
                    let d1 = CGPoint(x: p2.x - p1.x, y: p2.y - p1.y)
                    let d2 = CGPoint(x: p4.x - p3.x, y: p4.y - p3.y)
                    let den = d1.x * d2.y - d1.y * d2.x
                    if abs(den) > 1e-9 {
                        let s3 = CGPoint(x: p3.x - p1.x, y: p3.y - p1.y)
                        let uu = (s3.x * d2.y - s3.y * d2.x) / den
                        let n = CGPoint(x: p1.x + uu * d1.x, y: p1.y + uu * d1.y)
                        obs = CGPoint(x: n.x * res.width, y: (1 - n.y) * res.height)
                    }
                }
                DispatchQueue.main.async {
                    self.hiResBusy = false
                    guard self.phase == .sweeping || self.phase == .moveToStance else { return }
                    guard let obs else { self.flog("hi-res: no QR detection"); return }
                    guard hypot(obs.x - raw.x, obs.y - raw.y) < 260 else {
                        self.flog(String(format: "hi-res: gate reject d=%.0f", hypot(obs.x - raw.x, obs.y - raw.y)))
                        return
                    }
                    let before = self.calibrator.samples.count
                    self.calibrator.add(obs: obs, camera: camT,
                                        fx: Double(K[0][0]), fy: Double(K[1][1]),
                                        cx: Double(K[2][0]), cy: Double(K[2][1]),
                                        lens: lens, targetWorld: tw,
                                        angularVelocity: av, translationSpeed: ts)
                    if self.calibrator.samples.count > before {
                        self.hiResSampleCount += 1
                        self.flog(String(format: "hi-res sample #%d fx=%.0f r=%.0f",
                                         self.hiResSampleCount, K[0][0],
                                         hypot(obs.x - CGFloat(K[2][0]), obs.y - CGFloat(K[2][1]))))
                    }
                }
            }
        }
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for a in anchors where a is ARImageAnchor { imageAnchor = a as? ARImageAnchor }
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for a in anchors where a is ARImageAnchor { imageAnchor = a as? ARImageAnchor }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let arView, phase != .done else { return }
        let viewportSize = arView.bounds.size
        guard viewportSize.width > 0 else { return }
        let camera = frame.camera

        // Motion estimate
        if let pt = prevTransform, frame.timestamp > prevTimestamp {
            let dt = frame.timestamp - prevTimestamp
            let r1 = simd_float3x3(simd_float3(pt.columns.0.x, pt.columns.0.y, pt.columns.0.z),
                                   simd_float3(pt.columns.1.x, pt.columns.1.y, pt.columns.1.z),
                                   simd_float3(pt.columns.2.x, pt.columns.2.y, pt.columns.2.z))
            let c2 = camera.transform
            let r2 = simd_float3x3(simd_float3(c2.columns.0.x, c2.columns.0.y, c2.columns.0.z),
                                   simd_float3(c2.columns.1.x, c2.columns.1.y, c2.columns.1.z),
                                   simd_float3(c2.columns.2.x, c2.columns.2.y, c2.columns.2.z))
            let rel = r1.transpose * r2
            let tr = rel.columns.0.x + rel.columns.1.y + rel.columns.2.z
            angularVelocity = Double(acos(max(-1, min(1, (tr - 1) / 2)))) / dt
            let dp = simd_float3(c2.columns.3.x - pt.columns.3.x,
                                 c2.columns.3.y - pt.columns.3.y,
                                 c2.columns.3.z - pt.columns.3.z)
            translationSpeed = Double(simd_length(dp)) / dt
        }
        prevTransform = camera.transform
        prevTimestamp = frame.timestamp

        // AR must be well-initialized before anything is collected.
        if case .normal = camera.trackingState {
            trackingNormalFrames += 1
        } else {
            trackingNormalFrames = 0
        }
        let ready = trackingNormalFrames >= 45
        if ready != trackingReady { trackingReady = ready }

        // HUD-rate state (10 Hz)
        hudFrameCounter += 1
        if hudFrameCounter % 6 == 0 {
            motionOK = angularVelocity < 0.25 && translationSpeed < 0.20
            acceptingNow = CFAbsoluteTimeGetCurrent() - lastAcceptTime < 0.8
            if let t = target {
                let c = camera.transform.columns.3
                distanceM = Double(simd_length(simd_float3(c.x - t.x, c.y - t.y, c.z - t.z)))
            }
            // moveToStance -> sweeping when the user settles at the distance
            if phase == .moveToStance {
                inStanceFrames = inStance ? inStanceFrames + 1 : 0
                if inStanceFrames >= 5 {
                    enterSweeping(position: firstIncomplete(stance: stanceIndex) ?? 0)
                }
            }
            // Far-focus pass bailout: if the defocused code can't be detected
            // at all for 15 s, waive the pass (the slope extrapolation covers
            // it) rather than wedging the flow.
            if phase == .sweeping, stanceIndex == Self.farFocusStance,
               cells[stanceIndex].allSatisfy({ $0 == 0 }),
               CFAbsoluteTimeGetCurrent() - lastAcceptTime > 15,
               lastAcceptTime > 0 {
                flog("far-focus pass WAIVED: no detections under defocus")
                for i in cells[stanceIndex].indices { cells[stanceIndex][i] = cellQuota }
                refreshDerived()
                advanceAfterCell()
            }
        }

        // Smooth tracking dot: project at frame rate, corrected by the
        // latest Vision fix (Vision alone is ~10-20 Hz and looks laggy).
        if let t = target {
            let proj = camera.projectPoint(t, orientation: .portrait, viewportSize: viewportSize)
            if proj.x.isFinite, proj.y.isFinite {
                qrScreenPoint = CGPoint(x: proj.x + visionOffset.width,
                                        y: proj.y + visionOffset.height)
            }
        } else if let ia = imageAnchor, ia.isTracked {
            let c = ia.transform.columns.3
            let proj = camera.projectPoint(simd_float3(c.x, c.y, c.z),
                                           orientation: .portrait, viewportSize: viewportSize)
            if proj.x.isFinite, proj.y.isFinite { qrScreenPoint = proj }
        }

        // Auto-lock via ARKit image tracking
        if trackingReady, target == nil, let ia = imageAnchor, ia.isTracked {
            let c = ia.transform.columns.3
            let sp = camera.projectPoint(simd_float3(c.x, c.y, c.z),
                                         orientation: .portrait, viewportSize: viewportSize)
            let ctr = CGPoint(x: viewportSize.width / 2, y: viewportSize.height / 2)
            // Center-locked: rays through the principal point are immune to
            // the very distortion being calibrated.
            let near = hypot(sp.x - ctr.x, sp.y - ctr.y) < min(viewportSize.width, viewportSize.height) * 0.15
            let stable = lastAutoPoint.map { hypot(sp.x - $0.x, sp.y - $0.y) < 6 } ?? false
            lastAutoPoint = sp
            autoLockCounter = (near && stable) ? autoLockCounter + 1 : 0
            if autoLockCounter >= 45 {
                lockTarget(at: sp, fallback: simd_float3(c.x, c.y, c.z), camera: camera)
            }
        }

        // Vision QR detection (~10 Hz)
        if !visionBusy, frame.timestamp - lastVisionTime > 0.05 {
            visionBusy = true
            lastVisionTime = frame.timestamp
            runVision(frame: frame, viewportSize: viewportSize)
        }
    }

    private func lockTarget(at screenPoint: CGPoint, fallback: simd_float3, camera: ARCamera) {
        guard let arView else { return }
        let world: simd_float3
        if let hit = arView.raycast(from: screenPoint, allowing: .estimatedPlane, alignment: .any).first {
            world = simd_float3(hit.worldTransform.columns.3.x,
                                hit.worldTransform.columns.3.y,
                                hit.worldTransform.columns.3.z)
        } else {
            world = fallback
        }
        target = world
        lockVerifyErrs = []
        lockVerified = false
        let cam = camera.transform.columns.3
        distanceM = Double(simd_length(simd_float3(cam.x - world.x, cam.y - world.y, cam.z - world.z)))
        flog(String(format: "target locked dist=%.2fm (verifying)", distanceM))
        enterMoveToStance(nextIncompleteStance() ?? 0)
    }

    private func runVision(frame: ARFrame, viewportSize: CGSize) {
        let pixelBuffer = frame.capturedImage
        let camera = frame.camera
        let K = camera.intrinsics
        let res = camera.imageResolution
        let displayT = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
        let tw = target
        let raw = tw.map { camera.projectPoint($0, orientation: .landscapeRight, viewportSize: res) }
        let portraitPred = tw.map { camera.projectPoint($0, orientation: .portrait, viewportSize: viewportSize) }
        let camTransform = camera.transform
        let lens = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera?.lensPosition ?? 0.7
        let av = angularVelocity
        let ts = translationSpeed

        visionQueue.async { [weak self] in
            guard let self else { return }
            let request = VNDetectBarcodesRequest()
            request.symbologies = [.qr]
            try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
            var centerPx: CGPoint?
            var centerNorm: CGPoint?
            var estDist = 0.0
            var halfSizePt = 0.0
            if let qr = request.results?.first(where: { $0.payloadStringValue == Self.qrPayload }) {
                let p1 = qr.topLeft, p2 = qr.bottomRight, p3 = qr.topRight, p4 = qr.bottomLeft
                let d1 = CGPoint(x: p2.x - p1.x, y: p2.y - p1.y)
                let d2 = CGPoint(x: p4.x - p3.x, y: p4.y - p3.y)
                let den = d1.x * d2.y - d1.y * d2.x
                if abs(den) > 1e-9 {
                    let s3 = CGPoint(x: p3.x - p1.x, y: p3.y - p1.y)
                    let uu = (s3.x * d2.y - s3.y * d2.x) / den
                    let n = CGPoint(x: p1.x + uu * d1.x, y: p1.y + uu * d1.y)
                    centerNorm = n
                    centerPx = CGPoint(x: n.x * res.width, y: (1 - n.y) * res.height)
                }
                let a = CGPoint(x: qr.topLeft.x * res.width, y: qr.topLeft.y * res.height)
                let b = CGPoint(x: qr.topRight.x * res.width, y: qr.topRight.y * res.height)
                let side = Double(hypot(a.x - b.x, a.y - b.y))
                if side > 1 {
                    estDist = 0.19 * Double(K[0][0]) / side
                    halfSizePt = 0.5 * side * Double(viewportSize.height) / Double(res.width)
                }
            }
            DispatchQueue.main.async {
                self.handleVision(centerPx: centerPx, centerNorm: centerNorm,
                                  estDist: estDist, halfSizePt: halfSizePt,
                                  raw: raw, portraitPred: portraitPred, tw: tw, K: K,
                                  camTransform: camTransform, lens: lens,
                                  av: av, ts: ts,
                                  displayT: displayT, viewportSize: viewportSize)
            }
        }
    }

    private func handleVision(centerPx: CGPoint?, centerNorm: CGPoint?,
                              estDist: Double, halfSizePt: Double,
                              raw: CGPoint?, portraitPred: CGPoint?, tw: simd_float3?, K: simd_float3x3,
                              camTransform: simd_float4x4, lens: Float,
                              av: Double, ts: Double,
                              displayT: CGAffineTransform, viewportSize: CGSize) {
        visionBusy = false
        estimatedQRDistance = estDist
        if halfSizePt > 0 { qrHalfSizePt = halfSizePt }

        guard let n = centerNorm else {
            qrScreenPoint = nil
            visionStableCount = 0
            if target == nil { preLockHint = "Point the camera at the printed code" }
            return
        }
        let tl = CGPoint(x: n.x, y: 1 - n.y).applying(displayT)
        let sp = CGPoint(x: tl.x * viewportSize.width, y: tl.y * viewportSize.height)

        // Pre-lock: distance coaching + Vision fallback lock
        if target == nil {
            let stable = lastVisionPoint.map { hypot(sp.x - $0.x, sp.y - $0.y) < 10 } ?? false
            visionStableCount = stable ? visionStableCount + 1 : 0
            let ctr = CGPoint(x: viewportSize.width / 2, y: viewportSize.height / 2)
            let centered = hypot(sp.x - ctr.x, sp.y - ctr.y) < min(viewportSize.width, viewportSize.height) * 0.15
            if estDist > 1.8 {
                preLockHint = "Walk closer to the code"
            } else if estDist > 0, estDist < 0.5 {
                preLockHint = "Step back a little"
            } else if !centered {
                preLockHint = "Place the code in the box"
            } else {
                preLockHint = "Hold steady — locking…"
            }
            if visionStableCount >= 8, centered, estDist > 0.45, estDist < 2.0,
               let av = arView, let cam = av.session.currentFrame?.camera,
               let hit = av.raycast(from: sp, allowing: .estimatedPlane, alignment: .any).first {
                // No raycast hit -> no lock. (A bad fallback once anchored at
                // the world origin; verification caught it, but don't try.)
                lockTarget(at: sp,
                           fallback: simd_float3(hit.worldTransform.columns.3.x,
                                                 hit.worldTransform.columns.3.y,
                                                 hit.worldTransform.columns.3.z),
                           camera: cam)
            }
        }
        lastVisionPoint = sp
        if target == nil {
            qrScreenPoint = sp                       // pre-lock: Vision-direct
        } else if let pp = portraitPred, pp.x.isFinite {
            // Post-lock: refine the 60 fps dot's correction offset (lerped).
            let nx = sp.x - pp.x, ny = sp.y - pp.y
            visionOffset = CGSize(width: 0.6 * nx + 0.4 * visionOffset.width,
                                  height: 0.6 * ny + 0.4 * visionOffset.height)
        }

        // Post-lock verification: the anchor's pinhole projection must sit on
        // the detected code (within the known distortion envelope) before any
        // calibration data is trusted. Bad raycasts re-lock immediately.
        if target != nil, !lockVerified, let obs = centerPx, let raw {
            lockVerifyErrs.append(Double(hypot(obs.x - raw.x, obs.y - raw.y)))
            if lockVerifyErrs.count >= 8 {
                let mean = lockVerifyErrs.reduce(0, +) / Double(lockVerifyErrs.count)
                if mean < 25 {
                    lockVerified = true
                    flog(String(format: "lock verified: mean proj-vs-detect %.1f px", mean))
                } else {
                    flog(String(format: "lock REJECTED: mean proj-vs-detect %.1f px — re-locking", mean))
                    target = nil
                    phase = .acquiring
                    visionStableCount = 0
                    preLockHint = "Re-locking — place the code in the box and hold steady…"
                }
                lockVerifyErrs = []
            }
            if !lockVerified { return }
        }

        // Ingestion + cell credit
        guard phase == .sweeping, lockVerified, let obs = centerPx, let tw, let raw else { return }
        guard hypot(obs.x - raw.x, obs.y - raw.y) < 120 else {
            predRejectStreak += 1
            if predRejectStreak > 20 {
                predRejectStreak = 0
                target = nil
                flog("re-lock: prediction gate streak")
                preLockHint = "Re-locking the target — hold steady…"
                phase = .acquiring
            }
            return
        }
        predRejectStreak = 0
        // Never let the sample cap silently stop collection: shed the oldest
        // samples and keep the newest (this wedged a run at 10 ft pos 10).
        if calibrator.samples.count >= calibrator.maxSamples - 5 {
            calibrator.pruneOldest(400)
            flog("sample cap reached — pruned oldest 400 (n=\(calibrator.samples.count))")
        }
        let before = calibrator.samples.count
        calibrator.add(obs: obs, camera: camTransform,
                       fx: Double(K[0][0]), fy: Double(K[1][1]),
                       cx: Double(K[2][0]), cy: Double(K[2][1]),
                       lens: lens, targetWorld: tw,
                       angularVelocity: av, translationSpeed: ts)
        guard calibrator.samples.count > before else { return }
        lastAcceptTime = CFAbsoluteTimeGetCurrent()
        // Credit the active cell only when: tracking is solid, the code has
        // SETTLED in the ring (0.5 s), and the user is at the stance
        // distance — the same conditions the UI shows.
        let inRing = hypot(sp.x - currentRingCenter.x, sp.y - currentRingCenter.y) < 70
        let now = CFAbsoluteTimeGetCurrent()
        if inRing && inStance && trackingReady {
            if inRingSince == nil { inRingSince = now }
            let settled = now - (inRingSince ?? now)
            settleFill = min(1, settled / 0.5)
            guard settled >= 0.5, !advancePending else { return }
            cells[stanceIndex][positionIndex] += 1
            refreshDerived()
            // Photo-pipeline sample fires MID-hold (credit 3 of 10), while the
            // user is still steady on the ring — firing at cell-done caught
            // the swing to the next position and 24/37 got motion-trimmed.
            if cells[stanceIndex][positionIndex] == 3 { captureHiResSample() }
            if cellDone(stanceIndex, positionIndex) {
                flog("cell done stance=\(Self.stanceFt(stanceIndex))ft pos='\(currentSay)'")
                // A visible completion beat before the ring moves on.
                advancePending = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) { [weak self] in
                    guard let self else { return }
                    self.advancePending = false
                    self.inRingSince = nil
                    self.settleFill = 0
                    self.advanceAfterCell()
                }
            }
        } else {
            inRingSince = nil
            settleFill = 0
        }
    }
}

// MARK: - View

public struct ARDistortionCalibrationView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var engine = ARDistortionCalibrationEngine()
    @State private var step = 0
    @State private var qrFileURL: URL?

    public init() {}

    public var body: some View {
        switch step {
        case 0: qrStep
        case 1: arStep
        default: doneStep
        }
    }

    // Step 1 — get the code
    private var qrStep: some View {
        VStack(spacing: 20) {
            HStack {
                Button("Cancel") { dismiss() }
                Spacer()
            }
            Text("Camera Calibration").font(.title2).bold()
            Text("Step 1 of 2 — print the calibration code")
                .foregroundColor(.secondary)
            if let cg = ARDistortionCalibrationEngine.makeQRCGImage() {
                Image(decorative: cg, scale: 1)
                    .resizable().interpolation(.none)
                    .frame(width: 220, height: 220)
            }
            VStack(alignment: .leading, spacing: 10) {
                Label("Print this code as LARGE as fits the page (about 8 inches wide).", systemImage: "printer")
                Label("Tape it flat on a wall — not on glass or a screen.", systemImage: "rectangle.portrait")
                Label("Pick a spot with good light and about 15 ft of open floor.", systemImage: "lightbulb")
            }
            .font(.callout)
            .padding(.horizontal)
            if let url = qrFileURL {
                ShareLink(item: url) {
                    Label("Print or send this code", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.accentColor.opacity(0.15))
                        .cornerRadius(12)
                }
            }
            Spacer()
            Button {
                step = 1
            } label: {
                Text("The code is on the wall — start")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.accentColor)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }
        }
        .padding()
        .onAppear { prepareShareFile() }
    }

    private func prepareShareFile() {
        guard qrFileURL == nil else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            guard let cg = ARDistortionCalibrationEngine.makeQRCGImage() else { return }
            let img = UIImage(cgImage: cg)
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("camera-calibration-code.png")
            try? img.pngData()?.write(to: url)
            DispatchQueue.main.async { qrFileURL = url }
        }
    }

    // Step 2 — the guided flow
    private var arStep: some View {
        ZStack {
            CalibrationARViewContainer(engine: engine)
                .edgesIgnoringSafeArea(.all)

            GeometryReader { geo in
                // Acquire reticle: the box where the code must sit to lock.
                if engine.phase == .acquiring {
                    let boxSize: CGFloat = 190
                    let center = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                    let inBox = engine.qrScreenPoint.map {
                        abs($0.x - center.x) < boxSize / 2 && abs($0.y - center.y) < boxSize / 2
                    } ?? false
                    ScanBrackets()
                        .stroke(inBox ? Color.green : Color.cyan,
                                style: StrokeStyle(lineWidth: 4, lineCap: .round))
                        .frame(width: boxSize, height: boxSize)
                        .position(center)
                        .animation(.easeInOut(duration: 0.2), value: inBox)
                }

                // Active ring (sweeping only): clamped so the code always fits.
                if engine.phase == .sweeping {
                    let inset = CGFloat((engine.qrHalfSizePt > 0 ? engine.qrHalfSizePt : 55) + 26)
                    let raw = CGPoint(x: engine.currentPos.x * geo.size.width,
                                      y: engine.currentPos.y * geo.size.height)
                    let center = CGPoint(x: min(max(raw.x, inset), geo.size.width - inset),
                                         y: min(max(raw.y, inset), geo.size.height - inset))
                    let _ = { engine.currentRingCenter = center }()
                    let inRing = engine.qrScreenPoint.map {
                        hypot($0.x - center.x, $0.y - center.y) < 70
                    } ?? false
                    let recording = inRing && engine.inStance && engine.motionOK && engine.acceptingNow

                    ZStack {
                        Circle()
                            .stroke(Color.white.opacity(0.3), lineWidth: 5)
                            .frame(width: 110, height: 110)
                        // settle arc (thin, inside) while arming the position
                        if engine.settleFill > 0, engine.settleFill < 1 {
                            Circle()
                                .trim(from: 0, to: engine.settleFill)
                                .stroke(Color.white.opacity(0.8),
                                        style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                                .rotationEffect(.degrees(-90))
                                .frame(width: 96, height: 96)
                        }
                        Circle()
                            .trim(from: 0, to: engine.cellFill)
                            .stroke(recording ? Color.green : Color.cyan,
                                    style: StrokeStyle(lineWidth: 5, lineCap: .round))
                            .rotationEffect(.degrees(-90))
                            .frame(width: 110, height: 110)
                            .animation(.easeOut(duration: 0.2), value: engine.cellFill)
                        if !inRing {
                            Circle()
                                .stroke(style: StrokeStyle(lineWidth: 2, dash: [8, 7]))
                                .foregroundColor(.cyan)
                                .frame(width: 126, height: 126)
                        }
                    }
                    .position(center)
                    .animation(.easeInOut(duration: 0.45), value: engine.positionIndex)
                }

                // Live QR tracking marker
                if let qr = engine.qrScreenPoint {
                    ZStack {
                        Circle().stroke(Color.white, lineWidth: 2)
                            .frame(width: 34, height: 34)
                        Circle().fill(Color.white).frame(width: 7, height: 7)
                    }
                    .position(qr)
                }
            }
            .ignoresSafeArea()

            VStack(spacing: 10) {
                HStack {
                    Button("Cancel") {
                        engine.stop()
                        dismiss()
                    }
                    .padding(10)
                    .background(.black.opacity(0.6))
                    .foregroundColor(.white)
                    .cornerRadius(8)
                    Spacer()
                    if engine.phase != .acquiring {
                        Label(String(format: "%.1f ft", engine.distanceM * 3.28084),
                              systemImage: "ruler")
                            .font(.system(size: 15, weight: .semibold, design: .rounded))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Capsule().fill(Color.black.opacity(0.7)))
                            .foregroundColor(.white)
                    }
                }
                .padding(.horizontal)
                .padding(.top)

                instructionBanner

                if engine.phase == .moveToStance || engine.phase == .sweeping {
                    DistanceGauge(engine: engine)
                        .padding(.horizontal)
                }

                Spacer()

                if engine.phase == .sweeping {
                    VStack(spacing: 8) {
                        HStack(spacing: 5) {
                            ForEach(0..<ARDistortionCalibrationEngine.positions(forStance: engine.stanceIndex).count, id: \.self) { i in
                                Circle()
                                    .fill(engine.positionDoneFlags.indices.contains(i) && engine.positionDoneFlags[i]
                                          ? Color.green
                                          : (i == engine.positionIndex ? Color.cyan : Color.white.opacity(0.25)))
                                    .frame(width: 7, height: 7)
                            }
                        }
                        HStack {
                            Text(engine.sweepProgressText)
                                .font(.system(size: 14, weight: .semibold, design: .rounded))
                                .foregroundColor(.white)
                            Spacer()
                            Button("Skip this position") { engine.skipPosition() }
                                .font(.system(size: 13, weight: .semibold, design: .rounded))
                                .foregroundColor(.cyan)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Color.black.opacity(0.75)))
                    .padding()
                }
            }
        }
        .onChange(of: engine.finished != nil) { done in
            if done {
                engine.stop()
                step = 2
            }
        }
    }

    /// The single instruction — one message, one place.
    private var instructionBanner: some View {
        let (text, color): (String, Color) = {
            if let fail = engine.failureBanner { return (fail, .red) }
            switch engine.phase {
            case .acquiring:
                if !engine.trackingReady {
                    return ("Getting ready — move the phone slowly", .cyan)
                }
                return (engine.preLockHint, .cyan)
            case .moveToStance:
                let label = ARDistortionCalibrationEngine.stanceLabel(engine.stanceIndex)
                let dir = engine.distanceM < engine.stanceDistance ? "back" : "forward"
                if engine.stanceIndex == ARDistortionCalibrationEngine.farFocusStance {
                    return ("Far-focus pass: walk \(dir) to about 8 ft", .cyan)
                }
                return ("Walk \(dir) to the \(label) mark", .cyan)
            case .sweeping:
                if engine.qrScreenPoint == nil { return ("Point back at the code", .orange) }
                if !engine.inStance {
                    let ft = ARDistortionCalibrationEngine.stanceFt(engine.stanceIndex)
                    let dir = engine.distanceM < engine.stanceDistance ? "back" : "forward"
                    return ("Step \(dir) to the \(ft) ft mark", .orange)
                }
                if !engine.motionOK { return ("Hold still", .orange) }
                return ("Pan slowly — place the code in the ring (\(engine.currentSay))", .cyan)
            case .solving:
                return ("Hold still — computing…", .cyan)
            case .done:
                return ("Done", .green)
            }
        }()
        return Text(text)
            .font(.system(size: 17, weight: .bold, design: .rounded))
            .multilineTextAlignment(.center)
            .foregroundColor(.white)
            .padding(.horizontal, 18)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity)
            .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Color.black.opacity(0.8)))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(color.opacity(0.8), lineWidth: 2))
            .padding(.horizontal)
    }

    // Step 3 — done
    private var doneStep: some View {
        VStack(spacing: 24) {
            Spacer()
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 64))
                .foregroundColor(.green)
            Text("Camera calibrated").font(.title2).bold()
            if let cal = engine.finished {
                Text("Accuracy verified to ±\(String(format: "%.1f", cal.rmsPx)) pixels on this device. Measurements will now use this calibration automatically.")
                    .multilineTextAlignment(.center)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)
            }
            Spacer()
            Button {
                dismiss()
            } label: {
                Text("Done")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.accentColor)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }
        }
        .padding()
    }
}

/// Classic scanner corner brackets.
struct ScanBrackets: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        let l: CGFloat = min(rect.width, rect.height) * 0.22
        let r: CGFloat = 14
        // top-left
        p.move(to: CGPoint(x: rect.minX, y: rect.minY + l))
        p.addLine(to: CGPoint(x: rect.minX, y: rect.minY + r))
        p.addQuadCurve(to: CGPoint(x: rect.minX + r, y: rect.minY),
                       control: CGPoint(x: rect.minX, y: rect.minY))
        p.addLine(to: CGPoint(x: rect.minX + l, y: rect.minY))
        // top-right
        p.move(to: CGPoint(x: rect.maxX - l, y: rect.minY))
        p.addLine(to: CGPoint(x: rect.maxX - r, y: rect.minY))
        p.addQuadCurve(to: CGPoint(x: rect.maxX, y: rect.minY + r),
                       control: CGPoint(x: rect.maxX, y: rect.minY))
        p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + l))
        // bottom-right
        p.move(to: CGPoint(x: rect.maxX, y: rect.maxY - l))
        p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - r))
        p.addQuadCurve(to: CGPoint(x: rect.maxX - r, y: rect.maxY),
                       control: CGPoint(x: rect.maxX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.maxX - l, y: rect.maxY))
        // bottom-left
        p.move(to: CGPoint(x: rect.minX + l, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX + r, y: rect.maxY))
        p.addQuadCurve(to: CGPoint(x: rect.minX, y: rect.maxY - r),
                       control: CGPoint(x: rect.minX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - l))
        return p
    }
}

/// Distance gauge: a track bracketed by the stance marks (4/6/8/10 ft) with
/// a live dot showing where the user is standing.
struct DistanceGauge: View {
    @ObservedObject var engine: ARDistortionCalibrationEngine

    private func frac(_ m: Double) -> CGFloat {
        CGFloat(ARDistortionCalibrationEngine.trackFrac(m))
    }

    var body: some View {
        GeometryReader { g in
            let w: CGFloat = g.size.width
            ZStack(alignment: .topLeading) {
                Capsule()
                    .fill(Color.white.opacity(0.18))
                    .frame(width: w, height: 4)
                    .offset(y: 11)
                ForEach(ARDistortionCalibrationEngine.stances.indices, id: \.self) { i in
                    tick(i, x: frac(ARDistortionCalibrationEngine.stances[i]) * w)
                }
                userDot
                    .position(x: frac(engine.distanceM) * w, y: 13)
                    .animation(.easeOut(duration: 0.25), value: engine.distanceM)
            }
        }
        .frame(height: 44)
        .padding(.horizontal, 18)
        .padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Color.black.opacity(0.75)))
    }

    private var userDot: some View {
        Circle()
            .fill(Color.white)
            .frame(width: 12, height: 12)
            .shadow(color: Color.black.opacity(0.5), radius: 2)
    }

    @ViewBuilder
    private func tick(_ i: Int, x: CGFloat) -> some View {
        let done: Bool = engine.stanceDone[i]
        let active: Bool = i == engine.stanceIndex && !done
        let color: Color = done ? .green : (active ? .cyan : Color.white.opacity(0.5))
        Circle()
            .fill(color)
            .frame(width: active ? 14 : 10, height: active ? 14 : 10)
            .position(x: x, y: 13)
        Text(ARDistortionCalibrationEngine.stanceLabel(i))
            .font(.system(size: 11, weight: .bold, design: .rounded))
            .foregroundColor(color)
            .position(x: x, y: 34)
    }
}

struct CalibrationARViewContainer: UIViewRepresentable {
    let engine: ARDistortionCalibrationEngine

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        context.coordinator.engine = engine
        engine.start(arView: arView)
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        weak var engine: ARDistortionCalibrationEngine?
    }

    static func dismantleUIView(_ uiView: ARView, coordinator: Coordinator) {
        coordinator.engine?.stop()
        uiView.session.pause()
    }
}
