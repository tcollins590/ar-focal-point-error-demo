//
//  DemoProjectionCorrector.swift
//
//  Drop-in correction for ARKit's residual radial distortion, plus an
//  embedded self-calibrator.
//
//  Findings this implements (measured on iPhone 17 Pro Max, Aug 2026):
//   - ARKit's per-frame intrinsics (fx, fy, cx, cy) are accurate, including
//     focus-breathing compensation and per-frame OIS principal-point motion.
//   - The residual error is uncorrected radial distortion:
//        observed = pinhole + (pinhole - principal) * k1 * r^2
//     with k1 ~ +2.2e-8 px^-2 (≈15 px in far corners, 0 at center).
//
//  Design rules ("be smart" checklist):
//   - All distortion math happens in CAPTURED-IMAGE space around the
//     PER-FRAME principal point from ARKit (never a cached center, never
//     screen space — the principal point moves with OIS frame to frame, and
//     its on-screen position additionally moves with crop/orientation).
//   - The calibrator solves the target's 3D position jointly with k1, so
//     anchor-depth errors (e.g. LiDAR failing on glossy surfaces) cannot
//     masquerade as camera-model error.
//   - A focal-scale term and principal-offset term are fitted as HEALTH
//     CHECKS and must come out ≈ 0; otherwise calibration is rejected
//     (contaminated data) rather than silently absorbed.
//   - Samples are motion-gated and coverage-gated (must span inner and outer
//     radii), fitted robustly (outlier trim), and validated split-half.
//

import ARKit
import simd

// MARK: - Calibration result

struct RadialCalibration: Codable, Equatable {
    var k1: Double                    // px^-2, captured-image space
    var scaleDiag: Double             // fitted focal scale minus 1 (health, ≈0)
    var principalOffsetX: Double      // fitted extra principal offset (health, ≈0)
    var principalOffsetY: Double
    var rmsPx: Double
    var sampleCount: Int
    var lensMin: Float
    var lensMax: Float
    var deviceModel: String
    var date: Date

    static var storageURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ar_calibration.json")
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) {
            try? data.write(to: Self.storageURL)
        }
    }

    static func load() -> RadialCalibration? {
        guard let data = try? Data(contentsOf: storageURL) else { return nil }
        return try? JSONDecoder().decode(RadialCalibration.self, from: data)
    }
}

// MARK: - Calibration set (k1 as a function of lens position)

/// Distortion is focus-dependent (measured: k1 2.6e-8 at lens ~0.69 vs
/// 1.5e-8 at lens ~0.76). Store calibrations at multiple focus states and
/// interpolate k1 over lensPosition, clamped at the ends.
struct RadialCalibrationSet: Codable {
    var points: [RadialCalibration] = []

    static var storageURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ar_calibration_set.json")
    }

    mutating func insert(_ cal: RadialCalibration) {
        let lens = (cal.lensMin + cal.lensMax) / 2
        points.removeAll { abs(($0.lensMin + $0.lensMax) / 2 - lens) < 0.03 }
        points.append(cal)
        points.sort { ($0.lensMin + $0.lensMax) < ($1.lensMin + $1.lensMax) }
    }

    func k1(forLens lens: Float) -> Double? {
        guard !points.isEmpty else { return nil }
        let xs = points.map { Double(($0.lensMin + $0.lensMax) / 2) }
        let ys = points.map { $0.k1 }
        let x = Double(lens)
        if x <= xs.first! { return ys.first! }
        if x >= xs.last! { return ys.last! }
        for i in 1..<xs.count where x <= xs[i] {
            let f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
        }
        return ys.last!
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) {
            try? data.write(to: Self.storageURL)
        }
    }

    static func load() -> RadialCalibrationSet? {
        guard let data = try? Data(contentsOf: storageURL) else { return nil }
        return try? JSONDecoder().decode(RadialCalibrationSet.self, from: data)
    }
}

// MARK: - Corrector (production runtime)

final class DemoProjectionCorrector {
    /// Factory-baked k1(lens) curves per device model, measured in-house.
    /// Used when the device has never run self-calibration. Units: k1 in
    /// 1920x1440 captured-image px^-2 at the given lensPosition.
    static let factoryCurves: [String: [(lens: Float, k1: Double)]] = [
        // iPhone 17 Pro Max — Tyler's device, station-calibrated 2026-08-08
        "iPhone18,2": [(0.698, 1.695e-8), (0.773, 2.780e-8)],
    ]
    static let genericK1 = 2.2e-8

    static var deviceModelIdentifier: String {
        var un = utsname(); uname(&un)
        return withUnsafeBytes(of: &un.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }

    /// Fallback radial distortion coefficient when no calibration for the
    /// current lens state exists.
    var defaultK1: Double
    private(set) var calibrations = RadialCalibrationSet()
    /// True when THIS device has run self-calibration (vs riding factory or
    /// generic defaults) — drive the "please calibrate" prompt off this.
    private(set) var isUserCalibrated = false
    /// Set per frame from the capture device; selects k1 from the curve.
    var currentLens: Float = -1
    /// Fitted radial-linear term (ARKit fx underestimate) from the adopted
    /// guided calibration; applied with k1. Resolution independent.
    var appliedScale: Double = 0

    var k1: Double {
        if currentLens >= 0, let k = calibrations.k1(forLens: currentLens) { return k }
        if let k = calibrations.k1(forLens: 0.7) { return k }
        return defaultK1
    }

    /// "user" | "factory:<model>" | "generic" — for debug display/telemetry.
    var calibrationSource: String {
        if isUserCalibrated { return "user" }
        return Self.factoryCurves[Self.deviceModelIdentifier] != nil
            ? "factory:\(Self.deviceModelIdentifier)" : "generic"
    }

    init(k1: Double = DemoProjectionCorrector.genericK1) {
        defaultK1 = k1
        if let set = RadialCalibrationSet.load(), !set.points.isEmpty {
            calibrations = set
            isUserCalibrated = true
            appliedScale = set.points.first?.scaleDiag ?? 0
        } else if let single = RadialCalibration.load() {
            calibrations.insert(single)   // migrate old single-point file
            isUserCalibrated = true
        } else if let curve = Self.factoryCurves[Self.deviceModelIdentifier] {
            for (lens, k1v) in curve {
                calibrations.insert(RadialCalibration(
                    k1: k1v, scaleDiag: 0, principalOffsetX: 0, principalOffsetY: 0,
                    rmsPx: 0, sampleCount: 0, lensMin: lens, lensMax: lens,
                    deviceModel: Self.deviceModelIdentifier, date: .distantPast))
            }
        }
    }

    func apply(_ cal: RadialCalibration) {
        appliedScale = cal.scaleDiag
        if !isUserCalibrated {
            // First user calibration replaces the factory/generic curve
            // entirely rather than mixing provenances.
            calibrations = RadialCalibrationSet()
            isUserCalibrated = true
        }
        calibrations.insert(cal)
        calibrations.save()
    }

    /// Pinhole image point -> where the (distorted) camera image actually
    /// shows that feature. Principal point must be the same frame's.
    func distort(_ p: CGPoint, principal c: CGPoint) -> CGPoint {
        let dx = Double(p.x - c.x), dy = Double(p.y - c.y)
        let g = appliedScale + k1 * (dx * dx + dy * dy)
        return CGPoint(x: Double(p.x) + dx * g, y: Double(p.y) + dy * g)
    }

    /// Observed (distorted) image point -> pinhole point, for unprojection /
    /// building rays. Newton iteration on the radial magnitude.
    func undistort(_ p: CGPoint, principal c: CGPoint) -> CGPoint {
        let dx = Double(p.x - c.x), dy = Double(p.y - c.y)
        let rd = (dx * dx + dy * dy).squareRoot()
        guard rd > 1e-9 else { return p }
        var r = rd  // solve r * (1 + scale + k1 r^2) = rd
        for _ in 0..<4 {
            let f = r * (1 + appliedScale + k1 * r * r) - rd
            let df = 1 + appliedScale + 3 * k1 * r * r
            r -= f / df
        }
        let s = r / rd
        return CGPoint(x: Double(c.x) + dx * s, y: Double(c.y) + dy * s)
    }

    /// World point -> corrected captured-image pixel (top-left origin,
    /// landscape). This is where the camera image actually shows the point.
    func projectedImagePoint(_ world: simd_float3, camera: ARCamera) -> CGPoint? {
        let res = camera.imageResolution
        let p = camera.projectPoint(world, orientation: .landscapeRight, viewportSize: res)
        guard p.x.isFinite, p.y.isFinite else { return nil }
        let K = camera.intrinsics
        return distort(p, principal: CGPoint(x: CGFloat(K[2][0]), y: CGFloat(K[2][1])))
    }

    /// World point -> corrected screen point for the given orientation and
    /// viewport (matches the live camera background).
    func projectedScreenPoint(_ world: simd_float3, frame: ARFrame,
                              orientation: UIInterfaceOrientation,
                              viewportSize: CGSize) -> CGPoint? {
        guard let img = projectedImagePoint(world, camera: frame.camera) else { return nil }
        let res = frame.camera.imageResolution
        let t = frame.displayTransform(for: orientation, viewportSize: viewportSize)
        let n = CGPoint(x: img.x / res.width, y: img.y / res.height).applying(t)
        return CGPoint(x: n.x * viewportSize.width, y: n.y * viewportSize.height)
    }

    /// The user tapped `screen` on the (distorted) video. Returns the screen
    /// point whose pinhole ray actually passes through that physical feature —
    /// feed THIS to ARView.raycast / camera.unprojectPoint instead of the tap.
    func raycastCompensatedScreenPoint(_ screen: CGPoint, frame: ARFrame,
                                       orientation: UIInterfaceOrientation,
                                       viewportSize: CGSize) -> CGPoint {
        let res = frame.camera.imageResolution
        let t = frame.displayTransform(for: orientation, viewportSize: viewportSize)
        let tInv = t.inverted()
        let nView = CGPoint(x: screen.x / viewportSize.width, y: screen.y / viewportSize.height)
        let nImg = nView.applying(tInv)
        let img = CGPoint(x: nImg.x * res.width, y: nImg.y * res.height)
        let K = frame.camera.intrinsics
        let und = undistort(img, principal: CGPoint(x: CGFloat(K[2][0]), y: CGFloat(K[2][1])))
        let n2 = CGPoint(x: und.x / res.width, y: und.y / res.height).applying(t)
        return CGPoint(x: n2.x * viewportSize.width, y: n2.y * viewportSize.height)
    }
}

// MARK: - Self-calibrator

/// Station-based self-calibration: the user holds the QR at a screen zone
/// (center, mid-ring, outer, corner) and walks backward/forward, then moves
/// to the next zone. Each zone is therefore sampled across a range of
/// distances — i.e. lens positions — which makes k1's focus-dependence
/// observable in a single run (fitted as k1 = k1_0 + slope*(lens - 0.7)),
/// and simultaneously breaks the depth↔scale and lateral↔principal-point
/// degeneracies via the distance variation.
final class DemoSelfCalibrator {

    struct Sample {
        let worldToCam: simd_double3x3   // rotation, world -> camera
        let camPos: SIMD3<Double>
        let fx: Double, fy: Double, cx: Double, cy: Double
        let obs: SIMD2<Double>           // detected pixel, captured-image space
        let lens: Float
    }

    enum Status: Equatable {
        case collecting
        case solved([RadialCalibration])
        case failed(String)
    }

    private(set) var samples: [Sample] = []
    private(set) var status: Status = .collecting

    /// Re-arm collection after a failed solve (samples are kept).
    func resumeCollecting() { status = .collecting }
    private var anchor0: SIMD3<Double> = .zero
    private var haveAnchor = false

    // Station zones by radius from the principal point, with per-zone sample
    // quotas AND per-zone distance-spread quotas (the walk in/out at each
    // zone is what makes the solve well-conditioned).
    static let radiusBins: [(Double, Double)] = [(0, 300), (300, 500), (500, 700), (700, 1e9)]
    static let zoneNames = ["center", "mid-ring", "outer-ring", "corners"]
    static let binQuota = [20, 25, 25, 15]
    private(set) var binCounts = [0, 0, 0, 0]
    private var binDists: [[Double]] = [[], [], [], []]
    let maxSamples = 800

    /// Median camera-to-anchor distance across samples.
    var targetDistance: Double {
        guard haveAnchor, !samples.isEmpty else { return 1 }
        let ds = samples.map { simd_length($0.camPos - anchor0) }.sorted()
        return ds[ds.count / 2]
    }

    /// Required in/out walk range per zone (center zone exempt).
    var minZoneSpread: Double { max(0.25, 0.3 * targetDistance) }

    private func zoneSpread(_ i: Int) -> Double {
        let d = binDists[i].sorted()
        guard d.count > 10 else { return 0 }
        return d[(d.count * 9) / 10] - d[d.count / 10]
    }

    var coverageMet: Bool {
        for i in 0..<4 {
            if binCounts[i] < Self.binQuota[i] { return false }
            if i > 0 && zoneSpread(i) < minZoneSpread { return false }
        }
        return true
    }

    var readyToSolve: Bool { coverageMet }

    /// One instruction at a time: the first unmet requirement.
    var guidance: String {
        for i in 0..<4 {
            if binCounts[i] < Self.binQuota[i] {
                return String(format: "hold QR at %@ + walk in/out  (%d/%d)",
                              Self.zoneNames[i], binCounts[i], Self.binQuota[i])
            }
            if i > 0 && zoneSpread(i) < minZoneSpread {
                return String(format: "hold QR at %@, walk back & forward  (range %.0f/%.0f cm)",
                              Self.zoneNames[i], zoneSpread(i) * 100, minZoneSpread * 100)
            }
        }
        return "coverage complete — solving"
    }

    func reset() {
        samples.removeAll()
        binCounts = [0, 0, 0, 0]
        binDists = [[], [], [], []]
        haveAnchor = false
        status = .collecting
    }

    /// Feed one motion-gated observation against a STATIC manual anchor.
    func add(obs: CGPoint, camera cam: simd_float4x4,
             fx: Double, fy: Double, cx: Double, cy: Double,
             lens: Float, targetWorld: simd_float3,
             angularVelocity: Double, translationSpeed: Double) {
        guard status == .collecting, samples.count < maxSamples else { return }
        guard angularVelocity < 0.25, translationSpeed < 0.20 else { return }
        // Reject AF-hunting excursions: near the focus limits the linear
        // k1(lens) model doesn't hold and a few samples get huge leverage.
        guard lens > 0.35, lens < 0.95 else { return }
        if !haveAnchor {
            anchor0 = SIMD3<Double>(Double(targetWorld.x), Double(targetWorld.y), Double(targetWorld.z))
            haveAnchor = true
        }
        let r = simd_double3x3(
            SIMD3<Double>(Double(cam.columns.0.x), Double(cam.columns.0.y), Double(cam.columns.0.z)),
            SIMD3<Double>(Double(cam.columns.1.x), Double(cam.columns.1.y), Double(cam.columns.1.z)),
            SIMD3<Double>(Double(cam.columns.2.x), Double(cam.columns.2.y), Double(cam.columns.2.z)))
        let s = Sample(worldToCam: r.transpose,
                       camPos: SIMD3<Double>(Double(cam.columns.3.x), Double(cam.columns.3.y), Double(cam.columns.3.z)),
                       fx: fx, fy: fy, cx: cx, cy: cy,
                       obs: SIMD2<Double>(Double(obs.x), Double(obs.y)),
                       lens: lens)
        let rad = simd_length(s.obs - SIMD2<Double>(cx, cy))
        let dist = simd_length(s.camPos - anchor0)
        for (i, bin) in Self.radiusBins.enumerated() where rad >= bin.0 && rad < bin.1 {
            binCounts[i] += 1
            binDists[i].append(dist)
        }
        samples.append(s)
    }

    // Model params th = [dX, dY, dZ, k1_0 (1e-8 @lens 0.70), k1_slope (1e-8
    // per lens unit), scale, dcx, dcy]
    static let lensRef = 0.70
    private func residual(_ s: Sample, _ th: [Double]) -> SIMD2<Double>? {
        let X = anchor0 + SIMD3<Double>(th[0], th[1], th[2])
        let Xc = s.worldToCam * (X - s.camPos)
        let z = -Xc.z
        guard z > 0.05 else { return nil }
        let cx = s.cx + th[6], cy = s.cy + th[7]
        let dx = s.fx * (1 + th[5]) * Xc.x / z
        let dy = s.fy * (1 + th[5]) * (-Xc.y) / z
        let k1 = (th[3] + th[4] * (Double(s.lens) - Self.lensRef)) * 1e-8
        let g = k1 * (dx * dx + dy * dy)
        return SIMD2<Double>(s.obs.x - (cx + dx * (1 + g)),
                             s.obs.y - (cy + dy * (1 + g)))
    }

    private func gaussNewton(_ subset: [Sample], start: [Double]) -> ([Double], Double)? {
        var th = start
        let n = 8
        let steps = [1e-4, 1e-4, 1e-4, 1e-2, 1e-1, 1e-4, 1e-2, 1e-2]
        // Priors: anchor sigma 15 cm, slope sigma 30 units (weak), scale
        // sigma 1%, principal offset sigma 5 px. k1_0 unregularized.
        let priorW: [Double] = [1 / 0.0225, 1 / 0.0225, 1 / 0.0225, 0, 1.0 / 400, 1e4, 0.04, 0.04]
        for _ in 0..<25 {
            var jtj = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
            var jtr = [Double](repeating: 0, count: n)
            var used = 0
            for s in subset {
                guard let r0 = residual(s, th) else { continue }
                var J = [[Double]](repeating: [0, 0], count: n)
                var ok = true
                for p in 0..<n {
                    var tp = th; tp[p] += steps[p]
                    guard let rp = residual(s, tp) else { ok = false; break }
                    J[p][0] = (r0.x - rp.x) / steps[p]
                    J[p][1] = (r0.y - rp.y) / steps[p]
                }
                guard ok else { continue }
                for a in 0..<n {
                    for b in 0..<n {
                        jtj[a][b] += J[a][0] * J[b][0] + J[a][1] * J[b][1]
                    }
                    jtr[a] += J[a][0] * r0.x + J[a][1] * r0.y
                }
                used += 1
            }
            for p in 0..<n {
                jtj[p][p] += priorW[p] + 1e-9
                jtr[p] += -th[p] * priorW[p]
            }
            guard used > 20, let d = Self.solveN(jtj, jtr) else { return nil }
            for p in 0..<n { th[p] += d[p] }
            if d.map({ abs($0) }).max()! < 1e-7 { break }
        }
        var sse = 0.0; var cnt = 0
        for s in subset {
            if let r = residual(s, th) { sse += r.x * r.x + r.y * r.y; cnt += 1 }
        }
        guard cnt > 0 else { return nil }
        return (th, (sse / Double(cnt)).squareRoot())
    }

    private static func solveN(_ A: [[Double]], _ b: [Double]) -> [Double]? {
        var a = A, x = b
        let n = b.count
        for col in 0..<n {
            var piv = col
            for r in (col + 1)..<n where abs(a[r][col]) > abs(a[piv][col]) { piv = r }
            if abs(a[piv][col]) < 1e-12 { return nil }
            a.swapAt(col, piv); x.swapAt(col, piv)
            for r in (col + 1)..<n {
                let f = a[r][col] / a[col][col]
                for c in col..<n { a[r][c] -= f * a[col][c] }
                x[r] -= f * x[col]
            }
        }
        var out = [Double](repeating: 0, count: n)
        for r in stride(from: n - 1, through: 0, by: -1) {
            var s = x[r]
            for c in (r + 1)..<n { s -= a[r][c] * out[c] }
            out[r] = s / a[r][r]
        }
        return out
    }

    @discardableResult
    func solve(deviceModel: String) -> Status {
        guard haveAnchor, samples.count >= 120 else {
            status = .failed("need more samples (\(samples.count))")
            return status
        }
        guard coverageMet else {
            status = .failed("coverage unmet — " + guidance)
            return status
        }
        let start: [Double] = [0, 0, 0, 2.0, 0, 0, 0, 0]
        guard let (th1, _) = gaussNewton(samples, start: start) else {
            status = .failed("solver did not converge")
            return status
        }
        let norms = samples.compactMap { s in residual(s, th1).map { simd_length($0) } }
        let med = norms.sorted()[norms.count / 2]
        let kept = samples.filter { s in
            residual(s, th1).map { simd_length($0) < max(3 * med, 3.0) } ?? false
        }
        guard kept.count >= 100, let (th, rms) = gaussNewton(kept, start: th1) else {
            status = .failed("too many outliers")
            return status
        }
        let even = kept.enumerated().filter { $0.offset % 2 == 0 }.map { $0.element }
        let odd = kept.enumerated().filter { $0.offset % 2 == 1 }.map { $0.element }
        guard let (te, _) = gaussNewton(even, start: th),
              let (to, _) = gaussNewton(odd, start: th) else {
            status = .failed("split-half solve failed")
            return status
        }
        let lensVals = kept.map { Double($0.lens) }.sorted()
        let lensLo = lensVals[lensVals.count / 10]
        let lensHi = lensVals[(lensVals.count * 9) / 10]
        func k1At(_ p: [Double], _ lens: Double) -> Double {
            (p[3] + p[4] * (lens - Self.lensRef)) * 1e-8
        }
        let lensMid = lensVals[lensVals.count / 2]
        let k1m = k1At(th, lensMid), k1e = k1At(te, lensMid), k1o = k1At(to, lensMid)
        let ref = max(abs(k1m), 5e-9)
        if abs(k1e - k1o) > 0.5 * ref {
            status = .failed(String(format: "k1 unstable: %.1e vs %.1e — collect more / move slower", k1e, k1o))
        } else if abs(th[5]) > 0.008 {
            status = .failed(String(format: "scale %.3f ≠ 1: contaminated — walk in/out more, matte target", 1 + th[5]))
        } else if abs(th[6]) > 10 || abs(th[7]) > 10 {
            status = .failed(String(format: "principal offset (%.1f,%.1f) too large — bad data", th[6], th[7]))
        } else if rms > 5.0 {
            status = .failed(String(format: "residual %.1f px too high", rms))
        } else if abs(th[4]) > 30 {
            status = .failed(String(format: "k1 lens-slope %.0f implausible", th[4]))
        } else {
            // Emit curve points at the sampled lens extremes (or one point if
            // the lens range was too narrow to constrain the slope).
            var cals: [RadialCalibration] = []
            let make: (Double, Double) -> RadialCalibration = { lens, k1 in
                RadialCalibration(
                    k1: k1, scaleDiag: th[5],
                    principalOffsetX: th[6], principalOffsetY: th[7],
                    rmsPx: rms, sampleCount: kept.count,
                    lensMin: Float(lens), lensMax: Float(lens),
                    deviceModel: deviceModel, date: Date())
            }
            if lensHi - lensLo > 0.03 {
                cals = [make(lensLo, k1At(th, lensLo)), make(lensHi, k1At(th, lensHi))]
            } else {
                cals = [make(lensMid, k1m)]
            }
            status = .solved(cals)
        }
        return status
    }
}
