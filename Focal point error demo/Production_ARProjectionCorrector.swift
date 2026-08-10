//
//  ARProjectionCorrector.swift
//
//  Corrects ARKit's residual radial lens distortion in 3D→pixel projections.
//
//  Background (measured Aug 2026, iPhone 17 Pro Max, QR ground-truth rig):
//  ARKit's per-frame intrinsics are accurate (fx tracks focus breathing, cx/cy
//  tracks OIS), but its camera model leaves ~1.5% radial distortion
//  uncorrected at the image edges:
//
//      observed = pinhole + (pinhole − principal) · k1 · r²
//
//  with k1 mildly increasing with lensPosition. In standard-accuracy AR
//  measurement this biases pole heights TALL by ~1–2% (the image shows the
//  pole ends outside where the pinhole pixel array believes they are; the
//  error grows with how much of the frame the pole fills). Applying this
//  correction reduces that systematic bias to the ~±3 px calibration floor.
//
//  Units: curves store k1n = k1_px · fx² (dimensionless, resolution
//  independent), so the same calibration serves the 1920×1440 video stream
//  and the ~4032×3024 captureHighResolutionFrame stills (verified: the still
//  is the same optical geometry, intrinsics scale exactly with resolution).
//
//  All distortion math runs in RAW captured-image space around the frame's
//  own principal point; results map into the caller's (orientation,
//  viewportSize) space via the same rotation + aspect-fill convention
//  projectPoint/displayTransform use.
//

import ARKit
import UIKit

public final class ARProjectionCorrector {
    public static let shared = ARProjectionCorrector()

    /// User-facing toggle: apply the lens calibration, or run the legacy
    /// zero-correction path. Persisted.
    private static let enabledKey = "arDistortionCorrectionEnabled"
    public var isEnabled: Bool {
        get { UserDefaults.standard.object(forKey: Self.enabledKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: Self.enabledKey) }
    }

    /// Lens position context for post-capture (measure-stage) projections,
    /// where the live capture device no longer reflects the photo's focus.
    /// Set from the photo's stored arLensPosition before generating arrays.
    public var lensOverride: Float?

    struct CurvePoint {
        let lens: Float   // AVCaptureDevice.lensPosition at calibration
        let k1n: Double   // k1_px · fx² at that lens position
    }

    /// Factory curves per device model (station self-calibration, in-house).
    /// TODO(calibration): replace/augment with the on-device self-calibration
    /// flow; a user calibration should take precedence over these.
    private static let factoryCurves: [String: [CurvePoint]] = [
        // iPhone 17 Pro Max — calibrated 2026-08-08 (fx_ref 1338 @1920×1440)
        "iPhone18,2": [
            CurvePoint(lens: 0.698, k1n: 0.03034),
            CurvePoint(lens: 0.773, k1n: 0.04977),
        ],
    ]
    /// Generic fallback for unmeasured models (mid-curve of measured devices).
    private static let genericCurve = [CurvePoint(lens: 0.75, k1n: 0.0394)]

    // Curve + source are read from projection loops (any thread) and written
    // when calibration completes — guard with a lock.
    private let stateLock = NSLock()
    private var _curve: [CurvePoint] = []
    private var _calibrationSource = "none"
    private var curve: [CurvePoint] {
        stateLock.lock(); defer { stateLock.unlock() }
        return _curve
    }
    /// "user" | "factory:<model>" | "generic" — surface in debug UI / telemetry.
    public var calibrationSource: String {
        stateLock.lock(); defer { stateLock.unlock() }
        return _calibrationSource
    }

    /// Live-lens cache: avoid an AVFoundation capture-device query per
    /// projection call inside 20k–40k-step pixel-array loops (review F9).
    private var cachedLiveLens: Float = 0.77
    private var cachedLiveLensTime: CFAbsoluteTime = 0
    private var liveLens: Float {
        stateLock.lock(); defer { stateLock.unlock() }
        let now = CFAbsoluteTimeGetCurrent()
        if now - cachedLiveLensTime > 0.2 {
            cachedLiveLensTime = now
            cachedLiveLens = ARWorldTrackingConfiguration
                .configurableCaptureDeviceForPrimaryCamera?.lensPosition ?? 0.77
        }
        return cachedLiveLens
    }

    public static var deviceModelIdentifier: String {
        var un = utsname()
        uname(&un)
        return withUnsafeBytes(of: &un.machine) {
            String(decoding: $0.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }

    /// Metadata of the applied user calibration, for the settings UI
    /// (no file I/O on the main thread — cached here).
    public private(set) var userCalibration: ARDistortionUserCalibration?

    init() {
        // No baked defaults: with no user calibration, the correction is
        // identity (legacy behavior). The stored calibration loads off the
        // main thread (CLAUDE.md: no sync file reads from view bodies).
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let user = ARDistortionUserCalibration.load() else { return }
            DispatchQueue.main.async { self?.applyUserCalibration(user) }
        }
    }

    /// Applies a user self-calibration if it belongs to THIS device model —
    /// a calibration migrated from another phone via backup restore must not
    /// silently apply the wrong optics.
    public func applyUserCalibration(_ user: ARDistortionUserCalibration) {
        guard !user.points.isEmpty, user.deviceModel == Self.deviceModelIdentifier else { return }
        stateLock.lock()
        _curve = user.points.map { CurvePoint(lens: $0.lens, k1n: $0.k1n) }
            .sorted { $0.lens < $1.lens }
        _calibrationSource = "user"
        stateLock.unlock()
        userCalibration = user
    }

    /// Removes the stored calibration and returns to identity (no correction).
    public func deleteUserCalibration() {
        DispatchQueue.global(qos: .utility).async { ARDistortionUserCalibration.clear() }
        stateLock.lock()
        _curve = []
        _calibrationSource = "none"
        stateLock.unlock()
        userCalibration = nil
    }

    /// k1n interpolated over lens position (clamped at curve ends). When the
    /// caller has no stored lens (older captures), falls back to the live
    /// capture device, then to far focus — poles are shot at 5–15 m where the
    /// lens sits near the far end anyway.
    func k1n(forLens lens: Float?) -> Double {
        guard !curve.isEmpty else { return 0 }
        let l = lens ?? lensOverride ?? liveLens
        guard let first = curve.first, let last = curve.last else { return 0 }
        // Real captures run at longer range (lens ~0.78-0.82) than the
        // calibration stances can reach with a 15 cm code (~0.72-0.78).
        // Clamping there under-corrects by 30-40%; instead extrapolate along
        // the curve's own measured slope, bounded (±0.06 lens, 0.5-2x k1n).
        if curve.count >= 2 {
            if l > last.lens {
                let a = curve[curve.count - 2]
                let slope = (last.k1n - a.k1n) / Double(last.lens - a.lens)
                let dl = Double(min(l - last.lens, 0.06))
                return max(last.k1n * 0.5, min(last.k1n * 2, last.k1n + slope * dl))
            }
            if l < first.lens {
                let b = curve[1]
                let slope = (b.k1n - first.k1n) / Double(b.lens - first.lens)
                let dl = Double(max(l - first.lens, -0.06))
                return max(first.k1n * 0.5, min(first.k1n * 2, first.k1n + slope * dl))
            }
        }
        if l <= first.lens { return first.k1n }
        if l >= last.lens { return last.k1n }
        for i in 1..<curve.count where l <= curve[i].lens {
            let a = curve[i - 1], b = curve[i]
            let f = Double((l - a.lens) / (b.lens - a.lens))
            return a.k1n + f * (b.k1n - a.k1n)
        }
        return last.k1n
    }

    // MARK: - Capture-path validation

    /// The calibration transfers between the video stream and high-res still
    /// captures because k1 is normalized by fx² — which assumes the still's
    /// intrinsics scale EXACTLY with its resolution (no hidden crop). This
    /// verifies that assumption on every capture; a mismatch means the device
    /// crops its still path and per-path calibration would be required.
    public private(set) var lastCaptureScalingOK: Bool?

    @discardableResult
    public func validateIntrinsicScaling(photo: ARCamera, video: ARCamera) -> Bool {
        let pr = Double(photo.intrinsics[0][0]) / Double(photo.imageResolution.width)
        let vr = Double(video.intrinsics[0][0]) / Double(video.imageResolution.width)
        guard pr.isFinite, vr.isFinite, vr > 0 else { return true }
        let ok = abs(pr / vr - 1) < 0.01
        lastCaptureScalingOK = ok
        if !ok {
            print(String(format: "⚠️ ARDistortion: photo/video intrinsic scaling mismatch (fx/W %.4f vs %.4f) — correction transfer invalid on this device",
                         pr, vr))
        }
        return ok
    }

    // MARK: - Forward correction

    /// Drop-in replacement for ARCamera.projectPoint for IMAGE-SPACE
    /// projections (pixel arrays, guy wires, tilted lines). Returns where the
    /// captured image actually shows the world point.
    public func correctedProjectPoint(_ point: SIMD3<Float>,
                                      camera: ARCamera,
                                      orientation: UIInterfaceOrientation,
                                      viewportSize: CGSize,
                                      lensPosition: Float? = nil) -> CGPoint {
        let base = camera.projectPoint(point, orientation: orientation, viewportSize: viewportSize)
        guard isEnabled, k1n(forLens: lensPosition) != 0,
              let dRaw = rawDelta(point, camera: camera, lens: lensPosition) else {
            return base
        }
        let d = mapVector(dRaw, camera: camera, orientation: orientation, viewportSize: viewportSize)
        #if DEBUG
        logOncePerSecond(camera: camera, orientation: orientation,
                         viewportSize: viewportSize, lens: lensPosition, delta: d)
        #endif
        return CGPoint(x: base.x + d.dx, y: base.y + d.dy)
    }

    #if DEBUG
    private var lastLogTime: CFAbsoluteTime = 0
    private var maxDeltaSinceLog: Double = 0
    private let telemetryQueue = DispatchQueue(label: "ar.distortion.telemetry", qos: .utility)
    private static var telemetryURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ar_correction.log")
    }

    /// Persist the throttled telemetry so field sessions are inspectable
    /// after the fact (rolling: truncates past ~200 KB).
    private func persistTelemetry(_ line: String) {
        telemetryQueue.async {
            let url = Self.telemetryURL
            let fm = FileManager.default
            if let attrs = try? fm.attributesOfItem(atPath: url.path),
               let size = attrs[.size] as? Int, size > 200_000 {
                try? fm.removeItem(at: url)
            }
            if !fm.fileExists(atPath: url.path) {
                fm.createFile(atPath: url.path, contents: Data())
            }
            if let h = try? FileHandle(forWritingTo: url) {
                _ = try? h.seekToEnd()
                let stamp = ISO8601DateFormatter().string(from: Date())
                try? h.write(contentsOf: Data("\(stamp) \(line)\n".utf8))
                try? h.close()
            }
        }
    }
    /// Debug telemetry: throttled summary of what the correction is doing —
    /// verifies the video↔photo focal rescale on-device.
    private func logOncePerSecond(camera: ARCamera, orientation: UIInterfaceOrientation,
                                  viewportSize: CGSize, lens: Float?, delta: CGVector) {
        let mag = hypot(delta.dx, delta.dy)
        maxDeltaSinceLog = max(maxDeltaSinceLog, mag)
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastLogTime > 1 else { return }
        lastLogTime = now
        let K = camera.intrinsics
        let res = camera.imageResolution
        let l = lens ?? lensOverride
            ?? ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera?.lensPosition ?? -1
        let k1px = k1n(forLens: lens) / Double(K[0][0] * K[0][0])
        let line = String(format: "ARDistortion[%@]: res=%.0fx%.0f fx=%.1f lens=%.3f k1px=%.3e viewport=%.0fx%.0f maxΔ=%.1fpx",
                          calibrationSource, res.width, res.height, K[0][0], l, k1px,
                          viewportSize.width, viewportSize.height, maxDeltaSinceLog)
        print("📐 " + line)
        persistTelemetry(line)
        maxDeltaSinceLog = 0
    }
    #endif

    /// Distortion displacement in raw landscape captured-image pixels.
    private func rawDelta(_ point: SIMD3<Float>, camera: ARCamera, lens: Float?) -> CGVector? {
        let res = camera.imageResolution
        let raw = camera.projectPoint(point, orientation: .landscapeRight, viewportSize: res)
        guard raw.x.isFinite, raw.y.isFinite else { return nil }
        let K = camera.intrinsics
        let fx = Double(K[0][0])
        guard fx > 1 else { return nil }
        let dx = Double(raw.x) - Double(K[2][0])
        let dy = Double(raw.y) - Double(K[2][1])
        let r2 = dx * dx + dy * dy
        // The cubic model is only valid within the image domain; the array
        // builders extrapolate world points far off-frame, where k1·r³
        // explodes (observed 21k px) and could even fold an out-of-frame
        // point back inside. No correction beyond ~1.25× the half-diagonal.
        let maxR = 0.78 * Double(res.width)
        guard r2 < maxR * maxR else { return nil }
        let g = (k1n(forLens: lens) / (fx * fx)) * r2
        return CGVector(dx: dx * g, dy: dy * g)
    }

    /// Rotates + fill-scales a raw-image-space vector into the caller's
    /// (orientation, viewportSize) space. displayTransform is rotation +
    /// uniform aspect-fill scale + translation; vectors need only the linear
    /// part, so this is exact regardless of crop offsets.
    private func mapVector(_ v: CGVector, camera: ARCamera,
                           orientation: UIInterfaceOrientation,
                           viewportSize: CGSize) -> CGVector {
        let res = camera.imageResolution
        let rotatedVector: CGVector
        let rotatedSize: CGSize
        switch orientation {
        case .portrait:
            rotatedVector = CGVector(dx: -v.dy, dy: v.dx)
            rotatedSize = CGSize(width: res.height, height: res.width)
        case .portraitUpsideDown:
            rotatedVector = CGVector(dx: v.dy, dy: -v.dx)
            rotatedSize = CGSize(width: res.height, height: res.width)
        case .landscapeLeft:
            rotatedVector = CGVector(dx: -v.dx, dy: -v.dy)
            rotatedSize = res
        default: // .landscapeRight = raw buffer orientation
            rotatedVector = v
            rotatedSize = res
        }
        let f = max(viewportSize.width / rotatedSize.width,
                    viewportSize.height / rotatedSize.height)
        return CGVector(dx: rotatedVector.dx * f, dy: rotatedVector.dy * f)
    }

    // MARK: - Inverse correction (observed pixel → pinhole pixel)

    /// The user tapped/dragged on the (distorted) image at `observed`. Returns
    /// the pinhole-equivalent pixel — feed THIS to pixel→ray math
    /// (vectorFromPixelLocation etc.) so rays pass through the physical
    /// feature the user actually indicated.
    public func undistortedPixel(_ observed: CGPoint,
                                 camera: ARCamera,
                                 orientation: UIInterfaceOrientation,
                                 viewportSize: CGSize,
                                 lensPosition: Float? = nil) -> CGPoint {
        guard isEnabled, k1n(forLens: lensPosition) != 0 else { return observed }
        let res = camera.imageResolution
        let K = camera.intrinsics
        let fx = Double(K[0][0])
        guard fx > 1 else { return observed }
        let raw = targetToRaw(observed, camera: camera, orientation: orientation, viewportSize: viewportSize)
        let cx = Double(K[2][0]), cy = Double(K[2][1])
        let dx = Double(raw.x) - cx, dy = Double(raw.y) - cy
        let rd = (dx * dx + dy * dy).squareRoot()
        guard rd > 1e-9 else { return observed }
        let k1 = k1n(forLens: lensPosition) / (fx * fx)
        // Solve r·(1 + k1·r²) = rd for the pinhole radius (Newton).
        var r = rd
        for _ in 0..<4 {
            let f = r * (1 + k1 * r * r) - rd
            let df = 1 + 3 * k1 * r * r
            r -= f / df
        }
        let s = r / rd
        let undistorted = CGPoint(x: cx + dx * s, y: cy + dy * s)
        return rawToTarget(undistorted, camera: camera, orientation: orientation, viewportSize: viewportSize)
    }

    // Full affine map raw-buffer pixel ↔ (orientation, viewport) pixel:
    // rotate, then uniform aspect-fill with centered crop offset — the same
    // convention projectPoint(orientation:viewportSize:) uses.
    private func rotatedDims(_ res: CGSize, _ o: UIInterfaceOrientation) -> CGSize {
        (o == .portrait || o == .portraitUpsideDown)
            ? CGSize(width: res.height, height: res.width) : res
    }

    private func rawToTarget(_ p: CGPoint, camera: ARCamera,
                             orientation: UIInterfaceOrientation,
                             viewportSize: CGSize) -> CGPoint {
        let res = camera.imageResolution
        let rot: CGPoint
        switch orientation {
        case .portrait:           rot = CGPoint(x: res.height - p.y, y: p.x)
        case .portraitUpsideDown: rot = CGPoint(x: p.y, y: res.width - p.x)
        case .landscapeLeft:      rot = CGPoint(x: res.width - p.x, y: res.height - p.y)
        default:                  rot = p
        }
        let rs = rotatedDims(res, orientation)
        let f = max(viewportSize.width / rs.width, viewportSize.height / rs.height)
        let ox = (viewportSize.width - f * rs.width) / 2
        let oy = (viewportSize.height - f * rs.height) / 2
        return CGPoint(x: ox + f * rot.x, y: oy + f * rot.y)
    }

    private func targetToRaw(_ q: CGPoint, camera: ARCamera,
                             orientation: UIInterfaceOrientation,
                             viewportSize: CGSize) -> CGPoint {
        let res = camera.imageResolution
        let rs = rotatedDims(res, orientation)
        let f = max(viewportSize.width / rs.width, viewportSize.height / rs.height)
        let ox = (viewportSize.width - f * rs.width) / 2
        let oy = (viewportSize.height - f * rs.height) / 2
        let rot = CGPoint(x: (q.x - ox) / f, y: (q.y - oy) / f)
        switch orientation {
        case .portrait:           return CGPoint(x: rot.y, y: res.height - rot.x)
        case .portraitUpsideDown: return CGPoint(x: res.width - rot.y, y: rot.x)
        case .landscapeLeft:      return CGPoint(x: res.width - rot.x, y: res.height - rot.y)
        default:                  return rot
        }
    }
}

public extension ARCamera {
    /// Distortion-corrected projectPoint for image-space projections.
    func correctedProjectPoint(_ point: SIMD3<Float>,
                               orientation: UIInterfaceOrientation,
                               viewportSize: CGSize,
                               lensPosition: Float? = nil) -> CGPoint {
        ARProjectionCorrector.shared.correctedProjectPoint(
            point, camera: self, orientation: orientation,
            viewportSize: viewportSize, lensPosition: lensPosition)
    }
}
