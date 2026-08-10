//
//  ARDistortionSelfCalibrator.swift
//
//  On-device self-calibration of ARKit's residual radial distortion (see
//  ARProjectionCorrector.swift for background). The user points the camera at
//  a printed QR target and follows guided "stations" (hold the QR at a screen
//  zone, walk in/out); the solver jointly fits the target's 3D position and
//  the distortion curve, with health gates that reject contaminated data
//  instead of absorbing it.
//
//  Solver model, per sample (Gauss–Newton, numeric Jacobian):
//    params th = [ΔX, ΔY, ΔZ, k1_0 (1e-8 @lens 0.70), k1_slope (1e-8/lens),
//                 scale, Δcx, Δcy]
//    scale / Δc are HEALTH CHECKS and must come out ≈ 0 — otherwise the data
//    was contaminated (anchor depth, glossy target, motion) and we reject.
//

import ARKit
import simd

/// A stored user calibration: k1n = k1_px · fx² (resolution independent).
public struct ARDistortionUserCalibration: Codable {
    public struct Point: Codable {
        public let lens: Float
        public let k1n: Double
    }
    public let points: [Point]
    public let rmsPx: Double
    public let sampleCount: Int
    public let deviceModel: String
    public let date: Date
    /// Fitted radial-linear term (ARKit fx underestimate, ~+0.6% measured).
    /// Must be APPLIED with k1, not just health-checked: at photo radius
    /// r=1600 a discarded 0.66% is ~11 px of uniform under-correction.
    public let scale: Double

    public init(points: [Point], rmsPx: Double, sampleCount: Int,
                deviceModel: String, date: Date, scale: Double) {
        self.points = points
        self.rmsPx = rmsPx
        self.sampleCount = sampleCount
        self.deviceModel = deviceModel
        self.date = date
        self.scale = scale
    }

    enum CodingKeys: String, CodingKey { case points, rmsPx, sampleCount, deviceModel, date, scale }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        points = try c.decode([Point].self, forKey: .points)
        rmsPx = try c.decode(Double.self, forKey: .rmsPx)
        sampleCount = try c.decode(Int.self, forKey: .sampleCount)
        deviceModel = try c.decode(String.self, forKey: .deviceModel)
        date = try c.decode(Date.self, forKey: .date)
        // Pre-scale calibrations decode with 0 (k1-only, legacy behavior).
        scale = try c.decodeIfPresent(Double.self, forKey: .scale) ?? 0
    }

    static var storageURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ar_distortion_calibration.json")
    }

    public func save() {
        if let data = try? JSONEncoder().encode(self) {
            try? data.write(to: Self.storageURL)
        }
    }

    public static func load() -> ARDistortionUserCalibration? {
        guard let data = try? Data(contentsOf: storageURL) else { return nil }
        return try? JSONDecoder().decode(ARDistortionUserCalibration.self, from: data)
    }

    public static func clear() {
        try? FileManager.default.removeItem(at: storageURL)
    }
}

final class ARDistortionSelfCalibrator {

    struct Sample {
        let worldToCam: simd_double3x3
        let camPos: SIMD3<Double>
        let fx: Double, fy: Double, cx: Double, cy: Double
        let obs: SIMD2<Double>
        let lens: Float
    }

    enum Status {
        case collecting
        case solved(ARDistortionUserCalibration)
        case failed(String)
    }

    private(set) var samples: [Sample] = []
    private(set) var status: Status = .collecting
    /// Numeric detail of the last solve attempt (gate values) — persisted by
    /// the flow for field diagnosis of "something's off" failures.
    private(set) var lastSolveDiagnostics: [String: Double] = [:]
    private var anchor0: SIMD3<Double> = .zero
    private var haveAnchor = false

    // Five zones out to the true screen corners (~980-1030 px image radius).
    // The captured photo's corners reach ~1200 px; sampling the edge band
    // minimizes extrapolation and gives k1 its highest-leverage data (∝ r³).
    // Radius bins in NORMALIZED units (radius / fx) so samples from the
    // video stream and full-resolution photo captures bin identically.
    // Values are the old 1920-px thresholds divided by fx≈1338.
    static let radiusBins: [(Double, Double)] = [(0, 0.224), (0.224, 0.374), (0.374, 0.523), (0.523, 0.673), (0.673, 1e9)]
    // Methodical coverage: enough data for a k1 good to ~1-2%, not a sprint.
    static let binQuota = [40, 60, 60, 45, 30]
    static let binCount = 5
    private(set) var binCounts = [Int](repeating: 0, count: 5)
    // Image-x hemisphere tallies (portrait top vs bottom of screen). With
    // only one side sampled at high radius, radial distortion is degenerate
    // with a principal-point offset — both sides are REQUIRED.
    private(set) var binTop = [Int](repeating: 0, count: 5)
    private(set) var binBottom = [Int](repeating: 0, count: 5)
    private var binDists: [[Double]] = Array(repeating: [], count: 5)
    let maxSamples = 2000

    func resumeCollecting() { status = .collecting }

    /// Drops the oldest samples (rebuilding zone tallies) so a failed solve
    /// at the sample cap can keep collecting instead of wedging (review F6).
    func pruneOldest(_ n: Int) {
        guard n > 0, !samples.isEmpty else { return }
        samples.removeFirst(min(n, samples.count))
        binCounts = [Int](repeating: 0, count: Self.binCount)
        binTop = [Int](repeating: 0, count: Self.binCount)
        binBottom = [Int](repeating: 0, count: Self.binCount)
        binDists = Array(repeating: [], count: Self.binCount)
        binSegs = [[Int]](repeating: [Int](repeating: 0, count: 4), count: Self.binCount)
        for s in samples {
            let rad = simd_length(s.obs - SIMD2<Double>(s.cx, s.cy)) / s.fx
            let dist = simd_length(s.camPos - anchor0)
            for (i, bin) in Self.radiusBins.enumerated() where rad >= bin.0 && rad < bin.1 {
                binCounts[i] += 1
                binDists[i].append(dist)
                if s.obs.x >= s.cx { binBottom[i] += 1 } else { binTop[i] += 1 }
                let band = Self.band(forBin: i)
                let frac = (dist - band.lo) / (band.hi - band.lo)
                if frac >= 0, frac < 1 {
                    binSegs[i][min(Self.bandSegments - 1, Int(frac * Double(Self.bandSegments)))] += 1
                }
            }
        }
    }

    var targetDistance: Double {
        guard haveAnchor, !samples.isEmpty else { return 1 }
        let ds = samples.map { simd_length($0.camPos - anchor0) }.sorted()
        return ds[ds.count / 2]
    }

    // Explicit per-zone DISTANCE-BAND coverage: every zone must be sampled
    // across its full working band, in segments the user can see. This is
    // what decorrelates radius from distance/focus (the degeneracy behind
    // the failed solves). Edge zone's band starts farther out (the QR must
    // fit inside near-edge rings).
    static let bandSegments = 4
    static func band(forBin i: Int) -> (lo: Double, hi: Double) {
        // Bands never demand a distance the edge-clipping guard forbids:
        // bin 3 starts at 1.5 m (guard 1.6 m rounds to the same coached ft),
        // bin 4 at 2.0 m.
        switch i {
        case binCount - 1: return (2.0, 3.5)
        case binCount - 2: return (1.5, 3.0)
        default: return (1.0, 3.0)
        }
    }
    static func segmentQuota(forBin i: Int) -> Int {
        i == binCount - 1 ? 4 : 6
    }
    private(set) var binSegs = [[Int]](repeating: [Int](repeating: 0, count: 4), count: 5)

    func bandCovered(_ i: Int) -> Bool {
        binSegs[i].allSatisfy { $0 >= Self.segmentQuota(forBin: i) }
    }

    func segmentFills(forBin i: Int) -> [Double] {
        let q = Double(Self.segmentQuota(forBin: i))
        return binSegs[i].map { min(1, Double($0) / q) }
    }

    /// Each outer zone needs at least a quarter of its quota on EACH screen
    /// hemisphere (top/bottom) — one-sided high-radius data makes k1
    /// degenerate with a principal offset.
    func hemisphereBalanced(_ i: Int) -> Bool {
        i < 2 || min(binTop[i], binBottom[i]) * 4 >= Self.binQuota[i]
    }

    var coverageMet: Bool {
        for i in 0..<Self.binCount {
            if binCounts[i] < Self.binQuota[i] { return false }
            if !bandCovered(i) { return false }
            if !hemisphereBalanced(i) { return false }
        }
        return true
    }

    /// 0…1 overall progress across counts and walk-range quotas.
    var progress: Double {
        var parts: [Double] = []
        for i in 0..<Self.binCount {
            parts.append(min(1, Double(binCounts[i]) / Double(Self.binQuota[i])))
            parts.append(segmentFills(forBin: i).reduce(0, +) / Double(Self.bandSegments))
        }
        return parts.reduce(0, +) / Double(parts.count)
    }

    func reset() {
        samples.removeAll()
        binCounts = [Int](repeating: 0, count: Self.binCount)
        binTop = [Int](repeating: 0, count: Self.binCount)
        binBottom = [Int](repeating: 0, count: Self.binCount)
        binDists = Array(repeating: [], count: Self.binCount)
        binSegs = [[Int]](repeating: [Int](repeating: 0, count: 4), count: Self.binCount)
        haveAnchor = false
        status = .collecting
    }

    func add(obs: CGPoint, camera cam: simd_float4x4,
             fx: Double, fy: Double, cx: Double, cy: Double,
             lens: Float, targetWorld: simd_float3,
             angularVelocity: Double, translationSpeed: Double) {
        guard case .collecting = status, samples.count < maxSamples else { return }
        guard angularVelocity < 0.25, translationSpeed < 0.20 else { return }
        guard lens > 0.35, lens < 0.95 else { return }   // AF-hunt rejection
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
        let rad = simd_length(s.obs - SIMD2<Double>(cx, cy)) / fx
        let dist = simd_length(s.camPos - anchor0)
        for (i, bin) in Self.radiusBins.enumerated() where rad >= bin.0 && rad < bin.1 {
            binCounts[i] += 1
            binDists[i].append(dist)
            if s.obs.x >= cx { binBottom[i] += 1 } else { binTop[i] += 1 }
            let band = Self.band(forBin: i)
            let frac = (dist - band.lo) / (band.hi - band.lo)
            if frac >= 0, frac < 1 {
                binSegs[i][min(Self.bandSegments - 1, Int(frac * Double(Self.bandSegments)))] += 1
            }
        }
        samples.append(s)
    }

    // MARK: - Solver

    static let lensRef = 0.70

    static let refFx = 1338.0

    private func residual(_ s: Sample, _ th: [Double]) -> SIMD2<Double>? {
        let X = anchor0 + SIMD3<Double>(th[0], th[1], th[2])
        let Xc = s.worldToCam * (X - s.camPos)
        let z = -Xc.z
        guard z > 0.05 else { return nil }
        let cx = s.cx + th[6], cy = s.cy + th[7]
        let dx = s.fx * (1 + th[5]) * Xc.x / z
        let dy = s.fy * (1 + th[5]) * (-Xc.y) / z
        // th[3]/th[4] define k1 in pixel units AT refFx; scale by (refFx/fx)^2
        // so samples from the video stream and hi-res photo captures fit one
        // physical curve (k1_px scales as 1/fx^2 for the same lens).
        let k1 = (th[3] + th[4] * (Double(s.lens) - Self.lensRef)) * 1e-8
            * (Self.refFx * Self.refFx) / (s.fx * s.fx)
        let g = k1 * (dx * dx + dy * dy)
        return SIMD2<Double>(s.obs.x - (cx + dx * (1 + g)),
                             s.obs.y - (cy + dy * (1 + g)))
    }

    private func gaussNewton(_ subset: [Sample], start: [Double]) -> ([Double], Double)? {
        var th = start
        let n = 8
        let steps = [1e-4, 1e-4, 1e-4, 1e-2, 1e-1, 1e-4, 1e-2, 1e-2]
        // Priors: anchor sigma 50 cm (the global walk-range requirement makes
        // anchor position well observed; a tight prior would leak raycast
        // depth error into other params), lens-slope sigma 20 units, scale
        // 1%, principal offset 5 px. k1_0 unregularized.
        let priorW: [Double] = [4.0, 4.0, 4.0, 0, 1.0 / 400, 1e4, 0.04, 0.04]
        for _ in 0..<25 {
            var jtj = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
            var jtr = [Double](repeating: 0, count: n)
            var used = 0
            for s in subset {
                guard let r0 = residual(s, th) else { continue }
                var jac = [[Double]](repeating: [0, 0], count: n)
                var ok = true
                for p in 0..<n {
                    var tp = th; tp[p] += steps[p]
                    guard let rp = residual(s, tp) else { ok = false; break }
                    jac[p][0] = (r0.x - rp.x) / steps[p]
                    jac[p][1] = (r0.y - rp.y) / steps[p]
                }
                guard ok else { continue }
                for a in 0..<n {
                    for b in 0..<n {
                        jtj[a][b] += jac[a][0] * jac[b][0] + jac[a][1] * jac[b][1]
                    }
                    jtr[a] += jac[a][0] * r0.x + jac[a][1] * r0.y
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
        var sse = 0.0
        var cnt = 0
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
            a.swapAt(col, piv)
            x.swapAt(col, piv)
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
        guard haveAnchor, samples.count >= 250 else {
            status = .failed("Keep going — more samples needed")
            return status
        }
        let start: [Double] = [0, 0, 0, 2.0, 0, 0, 0, 0]
        guard let (th1, _) = gaussNewton(samples, start: start) else {
            status = .failed("Couldn't compute — try again in better light")
            return status
        }
        let norms = samples.compactMap { s in residual(s, th1).map { simd_length($0) } }
        guard !norms.isEmpty else {
            status = .failed("Couldn't compute — try again in better light")
            return status
        }
        let med = norms.sorted()[norms.count / 2]
        let kept = samples.filter { s in
            residual(s, th1).map { simd_length($0) < max(3 * med, 3.0) } ?? false
        }
        guard kept.count >= 100, let (th, rms) = gaussNewton(kept, start: th1) else {
            status = .failed("Too much noise — move slower, avoid glare")
            return status
        }
        // Half-fits restart from the neutral prior: starting them at the full
        // fit would sit both halves in the same flat spot under exactly the
        // degeneracies this gate exists to catch.
        let even = kept.enumerated().filter { $0.offset % 2 == 0 }.map { $0.element }
        let odd = kept.enumerated().filter { $0.offset % 2 == 1 }.map { $0.element }
        guard let (te, _) = gaussNewton(even, start: start),
              let (to, _) = gaussNewton(odd, start: start) else {
            status = .failed("Couldn't verify — collect a bit more")
            return status
        }
        let lensVals = kept.map { Double($0.lens) }.sorted()
        let lensLo = lensVals[lensVals.count / 10]
        let lensHi = lensVals[(lensVals.count * 9) / 10]
        let lensMid = lensVals[lensVals.count / 2]
        func k1At(_ p: [Double], _ lens: Double) -> Double {
            (p[3] + p[4] * (lens - Self.lensRef)) * 1e-8
        }
        let k1m = k1At(th, lensMid), k1e = k1At(te, lensMid), k1o = k1At(to, lensMid)
        // Residual rms of the hi-res photo samples alone (fx>2000): verifies
        // the PHOTO pipeline with the anchor solved jointly — free of the
        // raycast/drift contamination that biases external photo tests.
        var hiSse = 0.0
        var hiCnt = 0
        for smp in kept where smp.fx > 2000 {
            if let r = residual(smp, th) { hiSse += r.x * r.x + r.y * r.y; hiCnt += 1 }
        }
        let rmsHiRes = hiCnt > 0 ? (hiSse / Double(hiCnt)).squareRoot() : -1
        lastSolveDiagnostics = [
            "rms": rms, "n": Double(kept.count),
            "rmsHiRes": rmsHiRes, "nHiRes": Double(hiCnt),
            "scale": th[5], "dcx": th[6], "dcy": th[7],
            "k1_mid": k1m, "k1_even": k1e, "k1_odd": k1o,
            "k1_slope_1e8": th[4],
            "dX": th[0], "dY": th[1], "dZ": th[2],
            "lensLo": lensLo, "lensHi": lensHi,
        ]
        let ref = max(abs(k1m), 5e-9)
        if abs(k1e - k1o) > 0.5 * ref {
            status = .failed("Results unstable — walk the zones once more, slowly")
        } else if abs(th[5]) > 0.012 {
            status = .failed("Something's off — make sure the code is on a flat wall (not glass) and walk in/out more")
        } else if abs(th[6]) > 10 || abs(th[7]) > 10 {
            status = .failed("Something's off — re-point at the code and continue")
        } else if rms > 5.0 {
            status = .failed("Too much noise — better light, slower movement")
        } else if abs(th[4]) > 30 {
            status = .failed("Results unstable — continue the walking pattern")
        } else {
            // Emit k1n (k1 · fx²) so the calibration is resolution independent.
            func point(_ lens: Double) -> ARDistortionUserCalibration.Point {
                // The solver measures k1 on the (1+scale)-scaled pinhole
                // vector; the corrector applies g = scale + k1·r² on the
                // unscaled one. (1+s)³ converts between the conventions
                // (review F4; ~2% of k1 at s=0.006).
                let conv = pow(1 + th[5], 3)
                return ARDistortionUserCalibration.Point(lens: Float(lens),
                                                         k1n: k1At(th, lens) * conv * Self.refFx * Self.refFx)
            }
            let pts = (lensHi - lensLo > 0.03) ? [point(lensLo), point(lensHi)] : [point(lensMid)]
            let cal = ARDistortionUserCalibration(points: pts, rmsPx: rms,
                                                  sampleCount: kept.count,
                                                  deviceModel: deviceModel, date: Date(),
                                                  scale: th[5])
            status = .solved(cal)
        }
        return status
    }
}
