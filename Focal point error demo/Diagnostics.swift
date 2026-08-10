//
//  Diagnostics.swift
//  Focal point error demo
//
//  Diagnostic engine for isolating the source of ARKit's systematic
//  projection error. Data streams, all logged to CSV:
//
//   F rows (every frame):  intrinsics, lens position (focus), camera pose.
//   V rows (~10 Hz):       Vision QR detection in the RAW CAPTURED IMAGE vs
//                          intrinsics-based reprojection of the world target —
//                          bypasses the rendering path entirely. Also logs the
//                          projected center of ARKit's own tracked image anchor.
//   S rows (~0.5 Hz):      QR detected in the RENDERED ARView snapshot (what
//                          the user actually sees) vs the displayTransform-
//                          mapped Vision point (orange) and projectPoint
//                          (green). Separates display-path error from
//                          camera-model error numerically. Snapshot JPEGs are
//                          saved to Documents/snaps for offline inspection.
//   E rows:                user event marks.
//
//  Recording auto-starts with the AR session.
//

import ARKit
import RealityKit
import Vision
import SwiftUI
import AVFoundation
import CoreImage.CIFilterBuiltins

// MARK: - HUD state

struct DiagnosticsHUD {
    var fx: Float = 0
    var fy: Float = 0
    var cx: Float = 0
    var cy: Float = 0
    var imageWidth: Int = 0
    var imageHeight: Int = 0
    var lensPosition: Float = -1
    var pitchDeg: Float = 0
    var consistencyDeltaPx: Float = 0   // ARKit projectPoint vs manual intrinsics math
    var qrDetected: Bool = false
    var errU: Float = 0                 // observed - predicted, captured-image px
    var errV: Float = 0
    var errMag: Float = 0
    var radiusPx: Float = 0             // predicted point's distance from principal point
    var corrErrMag: Float = -1          // error after fitted correction, image px
    var imgAnchorErrPx: Float = -1      // ARKit image-anchor reprojection vs Vision obs
    var imgTracked: Bool = false
    var fitSx: Double = 0               // fitted scale, image x axis (landscape) -> portrait FY
    var fitSy: Double = 0               // fitted scale, image y axis (landscape) -> portrait FX
    var fitOffsetX: Double = 0
    var fitOffsetY: Double = 0
    var fitCount: Int = 0
    var fitValid: Bool = false
    var targetSource: String = "none"   // none | manual | image
    var targetDistM: Float = -1
    var afLocked: Bool = false
    var recording: Bool = false
    var snapCount: Int = 0
    var displayPathErrPx: Float = -1    // snapshot QR vs orange (displayTransform) in view pts
    var screenTotalErrPx: Float = -1    // snapshot QR vs green (projectPoint) in view pts
    var calibrating: Bool = false
    var calStatus: String = ""
}

// MARK: - Anchor refinement

/// Raycast anchor placement carries cm-scale depth error (estimatedPlane at
/// 2 m was measured 15 cm off along the view ray) which shows up as a
/// constant px offset from offset viewpoints — dwarfing the lens residual the
/// rig is trying to judge. This solves the anchor's true 3D position from the
/// live QR detections (k1 fixed from the adopted calibration), so photo stats
/// can be reported against ground truth instead of the raycast guess.
final class AnchorRefiner {
    private struct S {
        let obs: SIMD2<Double>
        let cam: simd_float4x4
        let fx: Double, cx: Double, cy: Double, k1px: Double
    }
    private var samples: [S] = []
    private var base: simd_float3?
    private(set) var refined: simd_float3?
    var count: Int { samples.count }

    func add(obs: CGPoint, cam: simd_float4x4, fx: Double, cx: Double, cy: Double,
             k1px: Double, target: simd_float3) {
        if let b = base, simd_length(b - target) > 0.05 {
            samples.removeAll(); refined = nil
        }
        base = target
        samples.append(S(obs: SIMD2(Double(obs.x), Double(obs.y)), cam: cam,
                         fx: fx, cx: cx, cy: cy, k1px: k1px))
        if samples.count > 800 { samples.removeFirst(200) }
        if samples.count >= 40, samples.count % 20 == 0 { solve() }
    }

    private func project(_ X: simd_float3, _ s: S) -> SIMD2<Double>? {
        let inv = s.cam.inverse
        let pc = inv * SIMD4<Float>(X.x, X.y, X.z, 1)
        guard pc.z < -0.01 else { return nil }
        let u = s.cx + s.fx * Double(pc.x / -pc.z)
        let v = s.cy - s.fx * Double(pc.y / -pc.z)
        let dx = u - s.cx, dy = v - s.cy
        let g = s.k1px * (dx * dx + dy * dy)
        return SIMD2(u + dx * g, v + dy * g)
    }

    private func solve() {
        guard var X = refined ?? base else { return }
        let eps: Float = 1e-3
        for _ in 0..<8 {
            var res: [(SIMD2<Double>, S)] = []
            for s in samples { if let p = project(X, s) { res.append((p - s.obs, s)) } }
            guard res.count > 20 else { return }
            let med = res.map { simd_length($0.0) }.sorted()[res.count / 2]
            var A = [[Double]](repeating: [Double](repeating: 0, count: 3), count: 3)
            var b = [Double](repeating: 0, count: 3)
            for (r, s) in res {
                let n = simd_length(r)
                let w = 1.0 / (1 + pow(n / max(3 * med, 1e-6), 2))
                var J = [[Double]](repeating: [0, 0], count: 3)
                for a in 0..<3 {
                    var Xp = X
                    Xp[a] += eps
                    guard let pp = project(Xp, s), let p0 = project(X, s) else { continue }
                    J[a] = [(pp.x - p0.x) / Double(eps), (pp.y - p0.y) / Double(eps)]
                }
                for i in 0..<3 {
                    for j in 0..<3 { A[i][j] += w * (J[i][0] * J[j][0] + J[i][1] * J[j][1]) }
                    b[i] -= w * (J[i][0] * r.x + J[i][1] * r.y)
                }
            }
            // Solve 3x3 via Cramer
            let d = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0])
            guard abs(d) > 1e-12 else { return }
            func rep(_ c: Int) -> Double {
                var M = A
                for i in 0..<3 { M[i][c] = b[i] }
                return M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1]) - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0]) + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0])
            }
            let step = SIMD3<Float>(Float(rep(0) / d), Float(rep(1) / d), Float(rep(2) / d))
            X += step
            if simd_length(step) < 1e-5 { break }
        }
        if let b = base, simd_length(X - b) < 0.5 { refined = X }
    }
}

// MARK: - Online scale/offset fit

/// Fits observed = s * predicted + offset per image axis, over recent samples.
/// A significant s deviation from 1.0 with small offset means focal length
/// scale error; a significant offset with s ~= 1.0 means principal point error.
final class ProjectionFitter {
    private var samples: [(pred: SIMD2<Double>, obs: SIMD2<Double>)] = []
    private let maxSamples = 900

    func add(pred: SIMD2<Double>, obs: SIMD2<Double>) {
        samples.append((pred, obs))
        if samples.count > maxSamples {
            samples.removeFirst(samples.count - maxSamples)
        }
    }

    func reset() { samples.removeAll() }

    var count: Int { samples.count }

    func fit() -> (sx: Double, sy: Double, offset: SIMD2<Double>, valid: Bool)? {
        let n = samples.count
        guard n >= 30 else { return nil }
        var meanP = SIMD2<Double>.zero
        var meanO = SIMD2<Double>.zero
        for s in samples { meanP += s.pred; meanO += s.obs }
        meanP /= Double(n)
        meanO /= Double(n)

        var sxx = 0.0, sxo = 0.0, syy = 0.0, syo = 0.0
        for s in samples {
            let p = s.pred - meanP
            let o = s.obs - meanO
            sxx += p.x * p.x
            sxo += p.x * o.x
            syy += p.y * p.y
            syo += p.y * o.y
        }
        guard sxx > 1e-6, syy > 1e-6 else { return nil }
        let sx = sxo / sxx
        let sy = syo / syy
        let offset = meanO - SIMD2<Double>(sx * meanP.x, sy * meanP.y)
        // Scale is only observable if predicted positions spanned enough of the
        // image (~30 px std dev on each axis).
        let valid = (sxx / Double(n)).squareRoot() > 30 && (syy / Double(n)).squareRoot() > 30
        return (sx, sy, offset, valid)
    }
}

// MARK: - CSV logging

final class CSVLogger {
    private(set) var url: URL?
    private var handle: FileHandle?
    private var buffer: [String] = []

    static let columns = [
        "type", "t", "fx", "fy", "cx", "cy", "res_w", "res_h", "lens", "af_locked",
        "pitch", "roll", "yaw", "cam_x", "cam_y", "cam_z",
        "qw", "qx", "qy", "qz",
        "target_x", "target_y", "target_z", "target_src",
        "pred_u", "pred_v", "manual_du", "manual_dv",
        "obs_u", "obs_v", "err_u", "err_v", "radius_px",
        "imgA_u", "imgA_v", "img_tracked",
        "fit_sx", "fit_sy", "fit_n", "fx_slider", "fy_slider", "corr_err",
        "snap_qr_u", "snap_qr_v", "orange_u", "orange_v", "green_u", "green_v", "snap_file",
        "note",
    ]

    var isRecording: Bool { handle != nil }

    func start() {
        stop()
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let name = "ardiag_\(formatter.string(from: Date())).csv"
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let fileURL = dir.appendingPathComponent(name)
        FileManager.default.createFile(atPath: fileURL.path,
                                       contents: (Self.columns.joined(separator: ",") + "\n").data(using: .utf8))
        handle = try? FileHandle(forWritingTo: fileURL)
        _ = try? handle?.seekToEnd()
        url = fileURL
    }

    /// Writes a row given a sparse field dictionary; unknown keys are ignored.
    func append(_ fields: [String: String]) {
        guard handle != nil else { return }
        buffer.append(Self.columns.map { fields[$0] ?? "" }.joined(separator: ","))
        if buffer.count >= 60 { flush() }
    }

    private func flush() {
        guard let handle, !buffer.isEmpty else { return }
        let data = (buffer.joined(separator: "\n") + "\n").data(using: .utf8)!
        try? handle.write(contentsOf: data)
        buffer.removeAll()
    }

    func stop() {
        flush()
        try? handle?.close()
        handle = nil
    }
}

// MARK: - Engine

final class DiagnosticsEngine: ObservableObject {
    @Published var hud = DiagnosticsHUD()
    /// Vision-detected QR center mapped to screen coords via displayTransform.
    @Published var visionScreenPoint: CGPoint?
    /// Corrected projection (magenta): predicted point with the fitted
    /// camera-model correction applied, mapped to screen. If the correction is
    /// right, this stays pinned to the QR at all tilts.
    @Published var correctedScreenPoint: CGPoint?

    // Correction constants fitted from session ardiag_20260808_205027
    // (iPhone 17 Pro Max): obs = pred + offset + pred_c*(scale + k1*r^2)
    // Correction: pure residual radial distortion (see DemoProjectionCorrector).
    // Loads persisted self-calibration if present, else measured default.
    let corrector = DemoProjectionCorrector()
    let calibrator = DemoSelfCalibrator()
    private var solving = false
    private var samplesAtLastSolve = 0
    private var photoBusy = false
    private lazy var photoIndex = photoURLs.count
    private var lastTargetWorld: simd_float3?
    private var lastLiveFx: Double = 0
    private var prevTransform: simd_float4x4?
    private var prevTimestamp: TimeInterval = 0
    private var angularVelocity: Double = 0
    private var translationSpeed: Double = 0

    let logger = CSVLogger()
    weak var arView: ARView?

    private let fitter = ProjectionFitter()
    let anchorRefiner = AnchorRefiner()
    private let visionQueue = DispatchQueue(label: "diag.vision")
    private var visionBusy = false
    private var lastVisionTime: TimeInterval = 0
    private var snapBusy = false
    private var lastSnapTime: TimeInterval = 0
    private var snapIndex = 0
    private let maxSnaps = 150
    private var frameCounter = 0

    var fxSlider: Float = 1.0
    var fySlider: Float = 1.0

    private var captureDevice: AVCaptureDevice? {
        ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera
    }

    static var snapsDirectory: URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("snaps")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Runtime-generated QR reference image for ARKit image tracking. Must be
    /// visually identical to the printed/displayed target (content "ARCAL-TARGET").
    static func makeQRReferenceImage() -> ARReferenceImage? {
        let filter = CIFilter.qrCodeGenerator()
        filter.message = "ARCAL-TARGET".data(using: .ascii)!
        filter.correctionLevel = "H"
        guard let out = filter.outputImage?.transformed(by: CGAffineTransform(scaleX: 30, y: 30)) else { return nil }
        let ctx = CIContext()
        guard let cg = ctx.createCGImage(out, from: out.extent) else { return nil }
        // Physical width is a guess; automaticImageScaleEstimation refines it.
        let ref = ARReferenceImage(cg, orientation: .up, physicalWidth: 0.15)
        ref.name = "qr_target"
        return ref
    }

    // MARK: Focus lock

    func toggleAFLock() {
        guard let device = captureDevice else { return }
        do {
            try device.lockForConfiguration()
            if hud.afLocked {
                if device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusMode = .continuousAutoFocus
                }
            } else {
                device.setFocusModeLocked(lensPosition: device.lensPosition)
            }
            device.unlockForConfiguration()
            hud.afLocked.toggle()
            markEvent(hud.afLocked ? "af_locked" : "af_unlocked")
        } catch {
            print("AF lock failed: \(error)")
        }
    }

    // MARK: Target / recording / events

    func targetChanged(distance: Float = -1) {
        fitter.reset()
        if distance > 0 {
            hud.calStatus = String(format: "anchor placed  %.1f m", distance)
            markEvent(String(format: "target_set dist=%.2f", distance))
        } else {
            markEvent("target_set")
        }
    }

    func placementFailed() {
        hud.calStatus = "⚠️ raycast failed — anchor NOT placed (aim at a surface, move closer)"
        markEvent("target_place_failed")
    }

    func placementTooClose(distance: Float) {
        hud.calStatus = String(format: "⚠️ too close (%.1f m) — step back to 5+ ft, then place the anchor", distance)
        markEvent(String(format: "target_place_too_close dist=%.2f", distance))
    }

    func resetFit() {
        fitter.reset()
        markEvent("fit_reset")
    }

    func startRecordingIfNeeded() {
        guard !logger.isRecording else { return }
        logger.start()
        hud.recording = true
    }

    func toggleRecording() {
        if logger.isRecording {
            logger.stop()
        } else {
            logger.start()
        }
        hud.recording = logger.isRecording
    }

    func markEvent(_ note: String) {
        guard logger.isRecording else { return }
        let t = prevTimestamp > 0 ? prevTimestamp : Date().timeIntervalSince1970
        logger.append(["type": "E", "t": String(format: "%.4f", t), "note": note.replacingOccurrences(of: ",", with: ";")])
    }

    // MARK: Per-frame processing (call from ARSessionDelegate on main)

    func process(frame: ARFrame,
                 targetWorld: simd_float3?,
                 targetSource: String,
                 imageAnchor: ARImageAnchor?,
                 viewportSize: CGSize) {
        let camera = frame.camera
        let K = camera.intrinsics
        let res = camera.imageResolution
        let lens = captureDevice?.lensPosition ?? -1
        corrector.currentLens = lens
        lastTargetWorld = targetWorld
        lastLiveFx = Double(K[0][0])
        let euler = camera.eulerAngles

        // Motion estimate for calibration sample gating.
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
            let ang = acos(max(-1, min(1, (tr - 1) / 2)))
            angularVelocity = Double(ang) / dt
            let dp = simd_float3(c2.columns.3.x - pt.columns.3.x,
                                 c2.columns.3.y - pt.columns.3.y,
                                 c2.columns.3.z - pt.columns.3.z)
            translationSpeed = Double(simd_length(dp)) / dt
        }
        prevTransform = camera.transform
        prevTimestamp = frame.timestamp

        var predicted: CGPoint?
        var greenScreen: CGPoint?
        var manualDelta = SIMD2<Float>.zero
        if let w = targetWorld {
            // ARKit's own projection into captured-image pixel space.
            let p = camera.projectPoint(w, orientation: .landscapeRight, viewportSize: res)
            predicted = p
            greenScreen = camera.projectPoint(w, orientation: .portrait, viewportSize: viewportSize)
            // Independent projection straight from the intrinsics matrix; the
            // delta validates that projectPoint == K * [R|t] (it should be ~0).
            if let m = Self.manualProjectToImage(w, camera: camera) {
                manualDelta = SIMD2<Float>(Float(m.x - p.x), Float(m.y - p.y))
            }
        }

        // ARKit's own image-tracking estimate of the QR center, reprojected
        // into captured-image space.
        var imgAnchorPx: CGPoint?
        var imgTracked = false
        if let ia = imageAnchor {
            imgTracked = ia.isTracked
            let c = ia.transform.columns.3
            imgAnchorPx = camera.projectPoint(simd_float3(c.x, c.y, c.z),
                                              orientation: .landscapeRight, viewportSize: res)
        }

        frameCounter += 1
        if frameCounter % 6 == 0 {
            hud.fx = K[0][0]; hud.fy = K[1][1]
            hud.cx = K[2][0]; hud.cy = K[2][1]
            hud.imageWidth = Int(res.width); hud.imageHeight = Int(res.height)
            hud.lensPosition = lens
            hud.pitchDeg = euler.x * 180 / .pi
            hud.consistencyDeltaPx = simd_length(manualDelta)
            hud.targetSource = targetSource
            hud.imgTracked = imgTracked
            if let w = targetWorld {
                let c = camera.transform.columns.3
                hud.targetDistM = simd_length(simd_float3(c.x - w.x, c.y - w.y, c.z - w.z))
            } else {
                hud.targetDistM = -1
            }
        }

        if logger.isRecording {
            let cam = camera.transform.columns.3
            let q = simd_quaternion(simd_float3x3(
                simd_float3(camera.transform.columns.0.x, camera.transform.columns.0.y, camera.transform.columns.0.z),
                simd_float3(camera.transform.columns.1.x, camera.transform.columns.1.y, camera.transform.columns.1.z),
                simd_float3(camera.transform.columns.2.x, camera.transform.columns.2.y, camera.transform.columns.2.z)))
            var row: [String: String] = [
                "type": "F",
                "t": String(format: "%.4f", frame.timestamp),
                "fx": String(format: "%.3f", K[0][0]), "fy": String(format: "%.3f", K[1][1]),
                "cx": String(format: "%.3f", K[2][0]), "cy": String(format: "%.3f", K[2][1]),
                "res_w": String(format: "%.0f", res.width), "res_h": String(format: "%.0f", res.height),
                "lens": String(format: "%.4f", lens),
                "af_locked": hud.afLocked ? "1" : "0",
                "pitch": String(format: "%.4f", euler.x),
                "roll": String(format: "%.4f", euler.z),
                "yaw": String(format: "%.4f", euler.y),
                "cam_x": String(format: "%.4f", cam.x),
                "cam_y": String(format: "%.4f", cam.y),
                "cam_z": String(format: "%.4f", cam.z),
                "qw": String(format: "%.6f", q.real),
                "qx": String(format: "%.6f", q.imag.x),
                "qy": String(format: "%.6f", q.imag.y),
                "qz": String(format: "%.6f", q.imag.z),
                "target_src": targetSource,
                "manual_du": String(format: "%.3f", manualDelta.x),
                "manual_dv": String(format: "%.3f", manualDelta.y),
                "fx_slider": String(format: "%.4f", fxSlider),
                "fy_slider": String(format: "%.4f", fySlider),
            ]
            if let w = targetWorld {
                row["target_x"] = String(format: "%.4f", w.x)
                row["target_y"] = String(format: "%.4f", w.y)
                row["target_z"] = String(format: "%.4f", w.z)
            }
            if let p = predicted {
                row["pred_u"] = String(format: "%.2f", p.x)
                row["pred_v"] = String(format: "%.2f", p.y)
            }
            if let ip = imgAnchorPx {
                row["imgA_u"] = String(format: "%.2f", ip.x)
                row["imgA_v"] = String(format: "%.2f", ip.y)
                row["img_tracked"] = imgTracked ? "1" : "0"
            }
            logger.append(row)
        }

        // Corrected projection mapped to screen (magenta marker).
        if let p = predicted {
            let principal = CGPoint(x: CGFloat(K[2][0]), y: CGFloat(K[2][1]))
            let cp = correctImagePoint(p, principal: principal)
            let dT = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
            let norm = CGPoint(x: cp.x / res.width, y: cp.y / res.height).applying(dT)
            correctedScreenPoint = CGPoint(x: norm.x * viewportSize.width,
                                           y: norm.y * viewportSize.height)
        } else {
            correctedScreenPoint = nil
        }

        // Throttled Vision QR detection on the raw captured image.
        if !visionBusy, frame.timestamp - lastVisionTime > 0.1 {
            visionBusy = true
            lastVisionTime = frame.timestamp
            let pixelBuffer = frame.capturedImage
            let displayT = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
            let snapshot = FrameSnapshot(
                timestamp: frame.timestamp,
                predicted: predicted,
                imgAnchorPx: imgAnchorPx,
                imgTracked: imgTracked,
                principal: CGPoint(x: CGFloat(K[2][0]), y: CGFloat(K[2][1])),
                imageResolution: res,
                displayTransform: displayT,
                viewportSize: viewportSize,
                lens: lens, pitch: euler.x, roll: euler.z, yaw: euler.y,
                camTransform: camera.transform,
                fx: Double(K[0][0]), fy: Double(K[1][1]),
                targetWorld: targetWorld, targetSource: targetSource,
                angularVelocity: angularVelocity, translationSpeed: translationSpeed
            )
            visionQueue.async { [weak self] in
                self?.detectQR(in: pixelBuffer, snapshot: snapshot)
            }
        }

        // Periodic rendered-screen snapshot: QR position in what the user
        // actually sees, vs orange (displayTransform) and green (projectPoint).
        if targetWorld != nil, !snapBusy, frame.timestamp - lastSnapTime > 2.0,
           snapIndex < maxSnaps, let arView {
            snapBusy = true
            lastSnapTime = frame.timestamp
            let green = greenScreen
            let orange = visionScreenPoint
            let t = frame.timestamp
            let viewSize = viewportSize
            arView.snapshot(saveToHDR: false) { [weak self] image in
                guard let self, let cg = image?.cgImage else {
                    DispatchQueue.main.async { self?.snapBusy = false }
                    return
                }
                self.visionQueue.async {
                    self.processScreenSnapshot(cg, green: green, orange: orange,
                                               t: t, viewSize: viewSize)
                }
            }
        }
    }

    private struct FrameSnapshot {
        let timestamp: TimeInterval
        let predicted: CGPoint?
        let imgAnchorPx: CGPoint?
        let imgTracked: Bool
        let principal: CGPoint
        let imageResolution: CGSize
        let displayTransform: CGAffineTransform
        let viewportSize: CGSize
        let lens: Float
        let pitch: Float
        let roll: Float
        let yaw: Float
        let camTransform: simd_float4x4
        let fx: Double
        let fy: Double
        let targetWorld: simd_float3?
        let targetSource: String
        let angularVelocity: Double
        let translationSpeed: Double
    }

    // MARK: Vision on captured image

    static let targetPayload = "ARCAL-TARGET"

    /// Chooses the right QR when several are visible: exact payload match
    /// first, else the one nearest an expected position (if provided).
    private static func pickQR(_ results: [VNBarcodeObservation]?,
                               near expected: CGPoint?, // normalized, bottom-left origin
                               maxDistance: CGFloat = 0.25) -> VNBarcodeObservation? {
        guard let results, !results.isEmpty else { return nil }
        if let match = results.first(where: { $0.payloadStringValue == targetPayload }) {
            return match
        }
        guard let expected else { return results.count == 1 ? results[0] : nil }
        let scored = results.map { qr -> (VNBarcodeObservation, CGFloat) in
            let c = CGPoint(x: (qr.topLeft.x + qr.topRight.x + qr.bottomLeft.x + qr.bottomRight.x) / 4,
                            y: (qr.topLeft.y + qr.topRight.y + qr.bottomLeft.y + qr.bottomRight.y) / 4)
            return (qr, hypot(c.x - expected.x, c.y - expected.y))
        }
        let best = scored.min { $0.1 < $1.1 }!
        return best.1 <= maxDistance ? best.0 : nil
    }

    private static func qrCenter(of qr: VNBarcodeObservation?) -> CGPoint? {
        guard let qr else { return nil }
        // Intersection of the quad's diagonals: the projection of the square's
        // physical center (projective invariant). Corner averaging is biased
        // under perspective at oblique angles.
        let p1 = qr.topLeft, p2 = qr.bottomRight, p3 = qr.topRight, p4 = qr.bottomLeft
        let d1 = CGPoint(x: p2.x - p1.x, y: p2.y - p1.y)
        let d2 = CGPoint(x: p4.x - p3.x, y: p4.y - p3.y)
        let denom = d1.x * d2.y - d1.y * d2.x
        if abs(denom) > 1e-9 {
            let s3 = CGPoint(x: p3.x - p1.x, y: p3.y - p1.y)
            let u = (s3.x * d2.y - s3.y * d2.x) / denom
            return CGPoint(x: p1.x + u * d1.x, y: p1.y + u * d1.y)
        }
        let pts = [qr.topLeft, qr.topRight, qr.bottomLeft, qr.bottomRight]
        return pts.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x / 4, y: $0.y + $1.y / 4) }
    }

    private func detectQR(in pixelBuffer: CVPixelBuffer, snapshot: FrameSnapshot) {
        let request = VNDetectBarcodesRequest()
        request.symbologies = [.qr]
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
        // Expected position: predicted target in Vision-normalized coords.
        var expected: CGPoint?
        if let p = snapshot.predicted {
            expected = CGPoint(x: p.x / snapshot.imageResolution.width,
                               y: 1 - p.y / snapshot.imageResolution.height)
        }
        var center: CGPoint?
        if let c = Self.qrCenter(of: Self.pickQR(request.results, near: expected)) {
            // Vision normalized coords have origin at bottom-left; convert to
            // top-left-origin pixel coords.
            center = CGPoint(x: c.x * snapshot.imageResolution.width,
                             y: (1 - c.y) * snapshot.imageResolution.height)
        }
        DispatchQueue.main.async { [weak self] in
            self?.handleVisionResult(center: center, snapshot: snapshot)
        }
    }

    private func handleVisionResult(center: CGPoint?, snapshot: FrameSnapshot) {
        visionBusy = false
        hud.qrDetected = center != nil

        guard let obs = center else {
            visionScreenPoint = nil
            return
        }

        // Orange dot: where the camera video shows the QR on screen, assuming
        // the background is drawn per displayTransform.
        let norm = CGPoint(x: obs.x / snapshot.imageResolution.width,
                           y: obs.y / snapshot.imageResolution.height)
        let viewNorm = norm.applying(snapshot.displayTransform)
        visionScreenPoint = CGPoint(x: viewNorm.x * snapshot.viewportSize.width,
                                    y: viewNorm.y * snapshot.viewportSize.height)

        if let ip = snapshot.imgAnchorPx {
            hud.imgAnchorErrPx = Float(hypot(ip.x - obs.x, ip.y - obs.y))
        }

        if hud.calibrating, ["manual", "auto"].contains(snapshot.targetSource), let tw = snapshot.targetWorld,
           let pd = snapshot.predicted, hypot(obs.x - pd.x, obs.y - pd.y) < 120 {
            calibrator.add(obs: obs, camera: snapshot.camTransform,
                           fx: snapshot.fx, fy: snapshot.fy,
                           cx: Double(snapshot.principal.x), cy: Double(snapshot.principal.y),
                           lens: snapshot.lens, targetWorld: tw,
                           angularVelocity: snapshot.angularVelocity,
                           translationSpeed: snapshot.translationSpeed)
            hud.calStatus = "cal " + calibrator.guidance
            maybeSolveCalibration()
        }

        guard let pred = snapshot.predicted else { return }

        if ["manual", "auto"].contains(snapshot.targetSource), let tw = snapshot.targetWorld,
           hypot(obs.x - pred.x, obs.y - pred.y) < 120 {
            anchorRefiner.add(obs: obs, cam: snapshot.camTransform,
                              fx: snapshot.fx,
                              cx: Double(snapshot.principal.x), cy: Double(snapshot.principal.y),
                              k1px: corrector.k1, target: tw)
        }

        let err = SIMD2<Double>(obs.x - pred.x, obs.y - pred.y)
        let radial = SIMD2<Double>(Double(pred.x - snapshot.principal.x),
                                   Double(pred.y - snapshot.principal.y))
        let radius = simd_length(radial)

        fitter.add(pred: SIMD2<Double>(pred.x, pred.y),
                   obs: SIMD2<Double>(obs.x, obs.y))
        let fit = fitter.fit()

        hud.errU = Float(err.x)
        hud.errV = Float(err.y)
        hud.errMag = Float(simd_length(err))
        hud.radiusPx = Float(radius)
        let corrected = correctImagePoint(pred, principal: snapshot.principal)
        hud.corrErrMag = Float(hypot(obs.x - corrected.x, obs.y - corrected.y))
        if let fit {
            hud.fitSx = fit.sx
            hud.fitSy = fit.sy
            hud.fitOffsetX = fit.offset.x
            hud.fitOffsetY = fit.offset.y
            hud.fitValid = fit.valid
        }
        hud.fitCount = fitter.count

        if logger.isRecording {
            var row: [String: String] = [
                "type": "V",
                "t": String(format: "%.4f", snapshot.timestamp),
                "lens": String(format: "%.4f", snapshot.lens),
                "af_locked": hud.afLocked ? "1" : "0",
                "pitch": String(format: "%.4f", snapshot.pitch),
                "roll": String(format: "%.4f", snapshot.roll),
                "yaw": String(format: "%.4f", snapshot.yaw),
                "pred_u": String(format: "%.2f", pred.x),
                "pred_v": String(format: "%.2f", pred.y),
                "obs_u": String(format: "%.2f", obs.x),
                "obs_v": String(format: "%.2f", obs.y),
                "err_u": String(format: "%.2f", err.x),
                "err_v": String(format: "%.2f", err.y),
                "radius_px": String(format: "%.1f", radius),
                "corr_err": String(format: "%.2f", hud.corrErrMag),
                "fit_n": String(fitter.count),
                "fx_slider": String(format: "%.4f", fxSlider),
                "fy_slider": String(format: "%.4f", fySlider),
            ]
            if let fit {
                row["fit_sx"] = String(format: "%.5f", fit.sx)
                row["fit_sy"] = String(format: "%.5f", fit.sy)
            }
            if let ip = snapshot.imgAnchorPx {
                row["imgA_u"] = String(format: "%.2f", ip.x)
                row["imgA_v"] = String(format: "%.2f", ip.y)
                row["img_tracked"] = snapshot.imgTracked ? "1" : "0"
            }
            logger.append(row)
        }
    }

    // MARK: Vision on rendered snapshot

    private func processScreenSnapshot(_ cg: CGImage, green: CGPoint?, orange: CGPoint?,
                                       t: TimeInterval, viewSize: CGSize) {
        let request = VNDetectBarcodesRequest()
        request.symbologies = [.qr]
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        try? handler.perform([request])
        var expected: CGPoint?
        if let g = green {
            expected = CGPoint(x: g.x / viewSize.width, y: 1 - g.y / viewSize.height)
        }
        let qrNorm = Self.qrCenter(of: Self.pickQR(request.results, near: expected))
        // Snapshot has the same orientation/aspect as the view; normalized
        // (bottom-left origin) -> view points.
        let qrView = qrNorm.map { CGPoint(x: $0.x * viewSize.width,
                                          y: (1 - $0.y) * viewSize.height) }

        // Save a downscaled JPEG for offline visual inspection.
        var fileName = ""
        if qrView != nil || snapIndex % 5 == 0 {  // always keep detected frames
            snapIndex += 1
            fileName = String(format: "snap_%03d_t%.1f.jpg", snapIndex, t)
            let scale = 750.0 / CGFloat(cg.width)
            let size = CGSize(width: 750, height: CGFloat(cg.height) * scale)
            let renderer = UIGraphicsImageRenderer(size: size)
            let img = renderer.image { _ in
                UIImage(cgImage: cg).draw(in: CGRect(origin: .zero, size: size))
            }
            if let data = img.jpegData(compressionQuality: 0.6) {
                try? data.write(to: Self.snapsDirectory.appendingPathComponent(fileName))
            }
        }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.snapBusy = false
            self.hud.snapCount = self.snapIndex
            if let q = qrView {
                if let o = orange {
                    self.hud.displayPathErrPx = Float(hypot(q.x - o.x, q.y - o.y))
                }
                if let g = green {
                    self.hud.screenTotalErrPx = Float(hypot(q.x - g.x, q.y - g.y))
                }
            }
            if self.logger.isRecording {
                var row: [String: String] = [
                    "type": "S",
                    "t": String(format: "%.4f", t),
                    "snap_file": fileName,
                ]
                if let q = qrView {
                    row["snap_qr_u"] = String(format: "%.2f", q.x)
                    row["snap_qr_v"] = String(format: "%.2f", q.y)
                }
                if let o = orange {
                    row["orange_u"] = String(format: "%.2f", o.x)
                    row["orange_v"] = String(format: "%.2f", o.y)
                }
                if let g = green {
                    row["green_u"] = String(format: "%.2f", g.x)
                    row["green_v"] = String(format: "%.2f", g.y)
                }
                self.logger.append(row)
            }
        }
    }

    /// Applies the fitted correction to a predicted captured-image point.
    func correctImagePoint(_ p: CGPoint, principal: CGPoint) -> CGPoint {
        corrector.distort(p, principal: principal)
    }

    /// Adopts the production guided flow's saved calibration into this demo's
    /// corrector so the magenta overlay validates the PRODUCTION solve
    /// against live QR ground truth. k1n = k1_px·fx² -> k1_px via video fx.
    func adoptGuidedCalibration() {
        guard let user = ARDistortionUserCalibration.load() else {
            hud.calStatus = "no guided calibration found"
            return
        }
        let fx = lastLiveFx > 100 ? lastLiveFx : 1338
        for pt in user.points {
            corrector.apply(RadialCalibration(
                k1: pt.k1n / (fx * fx), scaleDiag: 0,
                principalOffsetX: 0, principalOffsetY: 0,
                rmsPx: user.rmsPx, sampleCount: user.sampleCount,
                lensMin: pt.lens, lensMax: pt.lens,
                deviceModel: user.deviceModel, date: user.date))
        }
        let pts = user.points.map { String(format: "%.2f→%.2e", $0.lens, $0.k1n / (fx * fx)) }
            .joined(separator: "  ")
        hud.calStatus = String(format: "GUIDED CAL ADOPTED  rms %.1f px  n=%d  %@",
                               user.rmsPx, user.sampleCount, pts)
        markEvent("guided_cal_adopted")
    }

    // MARK: Self-calibration

    /// Guided station sequence: screen-fraction position + label. Ordered to
    /// cover center -> mid-ring -> outer-ring -> corners, both sides.
    static let calStations: [(pos: CGPoint, name: String)] = [
        (CGPoint(x: 0.50, y: 0.50), "center"),
        (CGPoint(x: 0.50, y: 0.28), "upper middle"),
        (CGPoint(x: 0.50, y: 0.72), "lower middle"),
        (CGPoint(x: 0.74, y: 0.16), "upper right"),
        (CGPoint(x: 0.26, y: 0.84), "lower left"),
        (CGPoint(x: 0.82, y: 0.08), "top-right corner"),
        (CGPoint(x: 0.18, y: 0.92), "bottom-left corner"),
    ]
    @Published var calStationIndex = 0
    @Published var photoURLs: [URL] = (try? FileManager.default.contentsOfDirectory(
        at: DiagnosticsEngine.photosDirectory, includingPropertiesForKeys: nil))?
        .filter { $0.pathExtension == "jpg" }.sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []

    var currentStation: (pos: CGPoint, name: String)? {
        guard hud.calibrating else { return nil }
        return Self.calStations[calStationIndex % Self.calStations.count]
    }

    func nextStation() {
        calStationIndex += 1
        let st = Self.calStations[calStationIndex % Self.calStations.count]
        markEvent("station_\(st.name)")
    }

    func startCalibration() {
        samplesAtLastSolve = 0
        guard hud.targetDistM > 0.2, hud.targetDistM < 4 else {
            hud.calStatus = hud.targetDistM < 0
                ? "no anchor — Place Anchor on the QR first"
                : String(format: "target %.1f m away — re-place anchor within 3 m of the QR", hud.targetDistM)
            return
        }
        calibrator.reset()
        calStationIndex = 0
        hud.calibrating = true
        hud.calStatus = "hold the QR in the ring, walk in ↔ out, then Next Zone"
        markEvent("cal_start")
    }

    func cancelCalibration() {
        hud.calibrating = false
        hud.calStatus = "calibration cancelled"
        markEvent("cal_cancel")
    }

    private func maybeSolveCalibration() {
        guard hud.calibrating, !solving, calibrator.samples.count >= 150,
              calibrator.samples.count >= samplesAtLastSolve + 80,
              calibrator.readyToSolve || calibrator.samples.count >= calibrator.maxSamples else { return }
        solving = true
        samplesAtLastSolve = calibrator.samples.count
        hud.calibrating = false  // stop feeding samples before background solve reads them
        hud.calStatus = "solving…"
        var un = utsname(); uname(&un)
        let model = withUnsafeBytes(of: &un.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
        visionQueue.async { [weak self] in
            guard let self else { return }
            let status = self.calibrator.solve(deviceModel: model)
            DispatchQueue.main.async {
                self.solving = false
                switch status {
                case .solved(let cals):
                    for cal in cals { self.corrector.apply(cal) }
                    let pts = self.corrector.calibrations.points
                        .map { String(format: "%.2f→%.1e", ($0.lensMin + $0.lensMax) / 2, $0.k1) }
                        .joined(separator: "  ")
                    self.hud.calStatus = String(format: "CAL OK  rms %.1f px  n=%d  curve: %@",
                                                cals[0].rmsPx, cals[0].sampleCount, pts)
                    self.markEvent("cal_solved " + pts)
                case .failed(let why):
                    // Keep the collected samples and resume: more data at the
                    // named weak spot usually fixes it.
                    self.calibrator.resumeCollecting()
                    self.hud.calibrating = true
                    self.hud.calStatus = "RETRY: \(why) — keep going, will re-solve"
                    self.markEvent("cal_failed: \(why)")
                case .collecting:
                    break
                }
            }
        }
    }

    // MARK: High-res photo prototype (mirrors Cloneable standard-accuracy capture)

    /// Captures a still via ARSession.captureHighResolutionFrame — the same
    /// API Cloneable's standard accuracy uses — projects the target into it
    /// raw (green) and k1-corrected (magenta, rescaled to the photo's
    /// intrinsics), detects the QR in the photo (orange), and saves an
    /// annotated JPEG to Documents/photos. Answers whether the video-frame
    /// calibration transfers to the high-res capture path.
    func capturePhoto() {
        guard let arView else { return }
        guard let tw = lastTargetWorld else {
            hud.calStatus = "place an anchor on the QR before Photo"
            return
        }
        guard !photoBusy else { return }
        photoBusy = true
        hud.calStatus = "capturing high-res frame…"
        let fxLive = lastLiveFx
        let k1Live = corrector.k1
        let refined = anchorRefiner.refined
        let refN = anchorRefiner.count
        arView.session.captureHighResolutionFrame { [weak self] frame, error in
            guard let self else { return }
            guard let frame else {
                DispatchQueue.main.async {
                    self.photoBusy = false
                    self.hud.calStatus = "photo failed: \(error?.localizedDescription ?? "unknown")"
                }
                return
            }
            self.visionQueue.async {
                self.processPhoto(frame: frame, target: tw, fxLive: fxLive, k1Live: k1Live,
                                  refined: refined, refN: refN)
            }
        }
    }

    static var photosDirectory: URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("photos")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func processPhoto(frame: ARFrame, target: simd_float3, fxLive: Double, k1Live: Double,
                              refined: simd_float3?, refN: Int) {
        let camera = frame.camera
        let res = camera.imageResolution
        let K = camera.intrinsics
        let principal = CGPoint(x: CGFloat(K[2][0]), y: CGFloat(K[2][1]))
        let fxHi = Double(K[0][0])
        let raw = camera.projectPoint(target, orientation: .landscapeRight, viewportSize: res)

        // Rescale k1 from calibration (video-frame px units) to photo px units:
        // same lens, so k1 scales with 1/fx².
        let m = fxHi > 0 && fxLive > 0 ? fxLive / fxHi : 1
        let k1Hi = k1Live * m * m
        let dx = Double(raw.x - principal.x), dy = Double(raw.y - principal.y)
        let g = k1Hi * (dx * dx + dy * dy)
        let corrected = CGPoint(x: Double(raw.x) + dx * g, y: Double(raw.y) + dy * g)
        let radius = (dx * dx + dy * dy).squareRoot()

        // QR ground truth in the photo itself.
        let request = VNDetectBarcodesRequest()
        request.symbologies = [.qr]
        let handler = VNImageRequestHandler(cvPixelBuffer: frame.capturedImage, options: [:])
        try? handler.perform([request])
        let expected = CGPoint(x: raw.x / res.width, y: 1 - raw.y / res.height)
        var obs: CGPoint?
        if let c = Self.qrCenter(of: Self.pickQR(request.results, near: expected)) {
            obs = CGPoint(x: c.x * res.width, y: (1 - c.y) * res.height)
        }
        let rawErr = obs.map { hypot($0.x - raw.x, $0.y - raw.y) }
        let corrErr = obs.map { hypot($0.x - corrected.x, $0.y - corrected.y) }

        // Same stats against the sweep-refined anchor: isolates the lens
        // residual from raycast anchor error (the dominant verification bias).
        var refCorrected: CGPoint?
        var refCorrErr: CGFloat?
        var anchorShiftCm: Double = 0
        if let ra = refined {
            anchorShiftCm = Double(simd_length(ra - target)) * 100
            let rawR = camera.projectPoint(ra, orientation: .landscapeRight, viewportSize: res)
            let dxr = Double(rawR.x - principal.x), dyr = Double(rawR.y - principal.y)
            let gr = k1Hi * (dxr * dxr + dyr * dyr)
            let cR = CGPoint(x: Double(rawR.x) + dxr * gr, y: Double(rawR.y) + dyr * gr)
            refCorrected = cR
            refCorrErr = obs.map { hypot($0.x - cR.x, $0.y - cR.y) }
        }

        // Annotated JPEG (raw landscape orientation).
        let ci = CIImage(cvPixelBuffer: frame.capturedImage)
        var fileName = ""
        if let cg = CIContext().createCGImage(ci, from: ci.extent) {
            let format = UIGraphicsImageRendererFormat()
            format.scale = 1
            let img = UIGraphicsImageRenderer(size: res, format: format).image { ctx in
                UIImage(cgImage: cg).draw(at: .zero)
                let g2 = ctx.cgContext
                func ring(_ p: CGPoint, _ color: UIColor, _ r: CGFloat) {
                    g2.setStrokeColor(color.cgColor)
                    g2.setLineWidth(6)
                    g2.strokeEllipse(in: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
                }
                ring(raw, .green, 34)
                ring(corrected, .magenta, 24)
                if let refCorrected { ring(refCorrected, .cyan, 16) }
                if let obs { ring(obs, .orange, 46) }
                var stats = String(format: "res %.0fx%.0f  fxHi %.0f (video fx %.0f)  k1hi %.2e  r %.0f px\nraw err %@  corrected err %@",
                                   res.width, res.height, fxHi, fxLive, k1Hi, radius,
                                   rawErr.map { String(format: "%.1f px", $0) } ?? "QR not found",
                                   corrErr.map { String(format: "%.1f px", $0) } ?? "-")
                if let refCorrErr {
                    stats += String(format: "\nrefined anchor (n=%d, shift %.1f cm): corrected err %.1f px",
                                    refN, anchorShiftCm, refCorrErr)
                } else {
                    stats += "\nno refined anchor yet — sweep the QR live before photos"
                }
                (stats as NSString).draw(at: CGPoint(x: 40, y: 40), withAttributes: [
                    .font: UIFont.boldSystemFont(ofSize: 44),
                    .foregroundColor: UIColor.yellow,
                ])
            }
            photoIndex += 1
            fileName = String(format: "photo_%02d.jpg", photoIndex)
            try? img.jpegData(compressionQuality: 0.7)?
                .write(to: Self.photosDirectory.appendingPathComponent(fileName))
        }

        let savedURL = fileName.isEmpty ? nil : Self.photosDirectory.appendingPathComponent(fileName)
        DispatchQueue.main.async {
            if let savedURL { self.photoURLs.append(savedURL) }
            self.photoBusy = false
            if let rawErr, let corrErr {
                self.hud.calStatus = String(format: "PHOTO %@ (%.0fx%.0f): raw %.1f px → corrected %.1f px @ r=%.0f",
                                            fileName, res.width, res.height, rawErr, corrErr, radius)
            } else {
                self.hud.calStatus = String(format: "PHOTO %@ (%.0fx%.0f): QR not found in photo", fileName, res.width, res.height)
            }
            self.markEvent(String(format: "photo %@ raw=%.1f corr=%.1f refCorr=%.1f shiftCm=%.1f refN=%d r=%.0f fxHi=%.0f resW=%.0f",
                                  fileName, rawErr ?? -1, corrErr ?? -1, refCorrErr ?? -1,
                                  anchorShiftCm, refN, radius, fxHi, res.width))
        }
    }

    // MARK: Projection math

    /// Projects a world point into captured-image pixel coordinates using the
    /// intrinsics matrix directly (top-left origin, landscape orientation).
    static func manualProjectToImage(_ w: simd_float3, camera: ARCamera) -> CGPoint? {
        let pc = camera.transform.inverse * simd_float4(w.x, w.y, w.z, 1)
        let zForward = -pc.z
        guard zForward > 0.001 else { return nil }
        let K = camera.intrinsics
        let u = K[0][0] * (pc.x / zForward) + K[2][0]
        let v = K[1][1] * (-pc.y / zForward) + K[2][1]
        return CGPoint(x: CGFloat(u), y: CGFloat(v))
    }
}

// MARK: - HUD view

struct DiagnosticsPanel: View {
    @ObservedObject var engine: DiagnosticsEngine

    private var hud: DiagnosticsHUD { engine.hud }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Group {
                Text(String(format: "K  fx %.1f  fy %.1f  cx %.1f  cy %.1f",
                            hud.fx, hud.fy, hud.cx, hud.cy))
                Text(String(format: "img %dx%d   lens %.3f %@   pitch %.1f°",
                            hud.imageWidth, hud.imageHeight, hud.lensPosition,
                            hud.afLocked ? "LOCKED" : "af", hud.pitchDeg))
                Text(String(format: "projectPoint vs manual K: %.2f px   %@",
                            hud.consistencyDeltaPx,
                            hud.recording ? "REC" : "not recording"))
                    .foregroundColor(hud.recording ? .white : .red)
            }
            Divider().background(Color.white.opacity(0.3))
            Text("correction: \(engine.corrector.calibrationSource)")
                .foregroundColor(engine.corrector.isUserCalibrated ? .green : .orange)
            if !hud.calStatus.isEmpty {
                Text(hud.calStatus)
                    .foregroundColor(hud.calibrating ? .cyan : (hud.calStatus.hasPrefix("CAL OK") ? .green : .red))
                    .lineLimit(3)
            }
            if hud.targetSource != "none" {
                Text("target: \(hud.targetSource)"
                     + (hud.targetDistM > 0 ? String(format: "  %.1f m", hud.targetDistM) : "")
                     + (hud.imgTracked ? "  (img tracked)" : ""))
                    .foregroundColor(hud.targetDistM > 4 ? .red : .white)
                Text(hud.qrDetected
                     ? String(format: "QR err (%+.1f, %+.1f) px  |%.1f|  r=%.0f",
                              hud.errU, hud.errV, hud.errMag, hud.radiusPx)
                     : "QR not detected")
                    .foregroundColor(hud.qrDetected ? .orange : .gray)
                if hud.corrErrMag >= 0 {
                    Text(String(format: "corrected err: %.1f px", hud.corrErrMag))
                        .foregroundColor(Color(red: 1, green: 0.3, blue: 1))
                }
                if hud.imgAnchorErrPx >= 0 {
                    Text(String(format: "ARKit img-track vs QR: %.1f px", hud.imgAnchorErrPx))
                        .foregroundColor(.yellow)
                }
                if hud.displayPathErrPx >= 0 {
                    Text(String(format: "display-path Δ %.1f px  on-screen err %.1f px  [%d snaps]",
                                hud.displayPathErrPx, hud.screenTotalErrPx, hud.snapCount))
                        .foregroundColor(.cyan)
                }
                if hud.fitCount >= 30 {
                    Text(String(format: "fit n=%d %@", hud.fitCount,
                                hud.fitValid ? "" : "(tilt for more spread)"))
                    Text(String(format: "  scale img-x %.4f → FY slider", hud.fitSx))
                    Text(String(format: "  scale img-y %.4f → FX slider", hud.fitSy))
                    Text(String(format: "  offset (%+.1f, %+.1f) px", hud.fitOffsetX, hud.fitOffsetY))
                }
            } else {
                Text("Aim crosshair at the QR center, tap Place Anchor")
                    .foregroundColor(.yellow)
            }
        }
        .font(.system(size: 11, design: .monospaced))
        .foregroundColor(.white)
        .padding(8)
        .background(Color.black.opacity(0.65))
        .cornerRadius(8)
    }
}


// MARK: - Photo viewer (zoom + pan)

struct PhotoViewerSheet: View {
    @ObservedObject var engine: DiagnosticsEngine
    @Environment(\.dismiss) private var dismiss
    @State private var index = 0

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button("Close") { dismiss() }
                    .foregroundColor(.white)
                Spacer()
                Text(engine.photoURLs.isEmpty ? "no photos" :
                        "\(engine.photoURLs[min(index, engine.photoURLs.count - 1)].lastPathComponent)  (\(index + 1)/\(engine.photoURLs.count))")
                    .foregroundColor(.white)
                    .font(.caption)
                Spacer()
                Button("Delete All") {
                    for u in engine.photoURLs { try? FileManager.default.removeItem(at: u) }
                    engine.photoURLs = []
                    dismiss()
                }
                .foregroundColor(.red)
            }
            .padding()
            if !engine.photoURLs.isEmpty {
                TabView(selection: $index) {
                    ForEach(engine.photoURLs.indices, id: \.self) { i in
                        ZoomableImageView(url: engine.photoURLs[i]).tag(i)
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: .always))
                .onAppear { index = engine.photoURLs.count - 1 }
            } else {
                Spacer()
                Text("Take a photo first").foregroundColor(.gray)
                Spacer()
            }
        }
        .background(Color.black.ignoresSafeArea())
    }
}

/// Pinch-zoom / pan / double-tap image view. Marker rings are baked into the
/// image pixels, so they stay exactly registered at any zoom.
struct ZoomableImageView: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> UIScrollView {
        let sv = UIScrollView()
        sv.maximumZoomScale = 15
        sv.minimumZoomScale = 1
        sv.bouncesZoom = true
        sv.showsHorizontalScrollIndicator = false
        sv.showsVerticalScrollIndicator = false
        sv.delegate = context.coordinator
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFit
        if let raw = UIImage(contentsOfFile: url.path), let cg = raw.cgImage {
            // Rotate the landscape sensor image upright for viewing.
            iv.image = UIImage(cgImage: cg, scale: 1, orientation: .right)
        }
        iv.frame = sv.bounds
        iv.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        iv.tag = 99
        sv.addSubview(iv)
        let dt = UITapGestureRecognizer(target: context.coordinator,
                                        action: #selector(Coordinator.doubleTap(_:)))
        dt.numberOfTapsRequired = 2
        sv.addGestureRecognizer(dt)
        return sv
    }

    func updateUIView(_ uiView: UIScrollView, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, UIScrollViewDelegate {
        func viewForZooming(in scrollView: UIScrollView) -> UIView? {
            scrollView.viewWithTag(99)
        }
        @objc func doubleTap(_ g: UITapGestureRecognizer) {
            guard let sv = g.view as? UIScrollView else { return }
            if sv.zoomScale > 1.5 {
                sv.setZoomScale(1, animated: true)
            } else {
                let p = g.location(in: sv.viewWithTag(99))
                let size = CGSize(width: sv.bounds.width / 6, height: sv.bounds.height / 6)
                sv.zoom(to: CGRect(x: p.x - size.width / 2, y: p.y - size.height / 2,
                                   width: size.width, height: size.height), animated: true)
            }
        }
    }
}
