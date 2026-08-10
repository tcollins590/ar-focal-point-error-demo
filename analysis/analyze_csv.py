#!/usr/bin/env python3
"""Analyze a CSV recorded by the AR Focal Point Error Demo diagnostics.

Usage: python3 analyze_csv.py ardiag_YYYYMMDD_HHMMSS.csv

Fits several camera-model corrections to the (predicted -> observed) QR
reprojection samples and reports which hypothesis the data supports:

  1. Per-axis scale + offset:      obs = s * pred + c
  2. Scale + principal offset + radial distortion:
                                   obs = pred_c*(1 + s + k1*r^2) + principal + c
  3. Correlation of the error with lens position (focus breathing) and pitch.

Interpretation guide:
  - s significantly != 1, k1 ~ 0, stable across lens positions
        -> static intrinsics miscalibration. Constant per-device fix.
  - s varies with lens position
        -> focus breathing not tracked by ARKit intrinsics. Fix must be a
           function of lensPosition (or lock focus).
  - k1 significant
        -> residual lens distortion. Fix needs a radial term, not just scale.
  - offset large, s ~ 1
        -> principal point error.
Axis mapping: image x (landscape) corresponds to the portrait FY slider,
image y corresponds to the portrait FX slider.
"""

import csv
import sys

import numpy as np


def load(path):
    frames, vision, snaps = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["type"] == "V" and row.get("obs_u"):
                vision.append(row)
            elif row["type"] == "F":
                frames.append(row)
            elif row["type"] == "S":
                snaps.append(row)
    return frames, vision, snaps


def f(row, key, default=np.nan):
    try:
        return float(row[key])
    except (ValueError, TypeError, KeyError):
        return default


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    frames, vision, snaps = load(sys.argv[1])
    if not vision:
        sys.exit("No Vision (V) rows in this CSV - record with a QR target placed.")

    # Principal point from frame rows (assume stable; report if not).
    cx = np.array([f(r, "cx") for r in frames])
    cy = np.array([f(r, "cy") for r in frames])
    fx = np.array([f(r, "fx") for r in frames])
    fy = np.array([f(r, "fy") for r in frames])
    lens_f = np.array([f(r, "lens") for r in frames])

    print(f"Frames: {len(frames)}, Vision samples: {len(vision)}")
    print(f"Reported intrinsics over session:")
    print(f"  fx {np.nanmean(fx):9.2f}  (std {np.nanstd(fx):.3f}, range {np.nanmax(fx)-np.nanmin(fx):.3f})")
    print(f"  fy {np.nanmean(fy):9.2f}  (std {np.nanstd(fy):.3f}, range {np.nanmax(fy)-np.nanmin(fy):.3f})")
    print(f"  cx {np.nanmean(cx):9.2f}  cy {np.nanmean(cy):9.2f}")
    print(f"  lensPosition range: {np.nanmin(lens_f):.3f} .. {np.nanmax(lens_f):.3f}")
    if np.nanstd(lens_f) > 0.01 and np.nanstd(fx) < 0.5:
        print("  !! lensPosition moved but reported fx barely changed ->")
        print("     ARKit intrinsics do NOT track focus breathing.")

    pred = np.array([[f(r, "pred_u"), f(r, "pred_v")] for r in vision])
    obs = np.array([[f(r, "obs_u"), f(r, "obs_v")] for r in vision])
    lens = np.array([f(r, "lens") for r in vision])
    pitch = np.array([f(r, "pitch") for r in vision])
    c = np.array([np.nanmean(cx), np.nanmean(cy)])

    err = obs - pred
    print(f"\nRaw reprojection error (obs - pred), captured-image px:")
    print(f"  mean ({np.mean(err[:,0]):+.2f}, {np.mean(err[:,1]):+.2f})  rms {np.sqrt((err**2).sum(1).mean()):.2f}")

    # Model 1: per-axis scale + offset
    print("\nModel 1: obs = s * pred + offset (per axis)")
    for axis, name, slider in [(0, "img-x", "FY"), (1, "img-y", "FX")]:
        p = pred[:, axis] - pred[:, axis].mean()
        o = obs[:, axis] - obs[:, axis].mean()
        denom = (p * p).sum()
        spread = np.sqrt(denom / len(p))
        if spread < 30:
            print(f"  {name}: insufficient spread ({spread:.0f} px std) - tilt more during capture")
            continue
        s = (p * o).sum() / denom
        off = obs[:, axis].mean() - s * pred[:, axis].mean()
        resid = obs[:, axis] - (s * pred[:, axis] + off)
        print(f"  {name}: scale {s:.5f}  offset {off:+7.2f} px  resid rms {resid.std():.2f} px"
              f"   -> {slider} slider = {s:.4f}")

    # Model 2: isotropic scale + k1 radial + principal offset
    # obs - pred = c_off + pred_c * s + pred_c * r^2 * k1
    pc = pred - c
    r2 = (pc ** 2).sum(1)
    n = len(pred)
    A = np.zeros((2 * n, 4))
    b = np.zeros(2 * n)
    A[0::2, 0] = 1;  A[1::2, 1] = 1          # principal/const offset
    A[0::2, 2] = pc[:, 0]; A[1::2, 2] = pc[:, 1]           # scale
    A[0::2, 3] = pc[:, 0] * r2; A[1::2, 3] = pc[:, 1] * r2  # k1
    b[0::2] = err[:, 0]; b[1::2] = err[:, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    ox, oy, s_iso, k1 = sol
    resid = b - A @ sol
    rmax2 = r2.max()
    print("\nModel 2: isotropic scale + radial k1 + offset")
    print(f"  offset ({ox:+.2f}, {oy:+.2f}) px   scale 1{s_iso:+.5f}   k1 {k1:.3e}")
    print(f"  radial term at max radius: {k1 * rmax2 * np.sqrt(rmax2):+.2f} px "
          f"vs scale term: {s_iso * np.sqrt(rmax2):+.2f} px")
    print(f"  resid rms {resid.std():.2f} px")

    # ARKit image-tracking stream: how far is ARKit's own vision-based
    # estimate of the QR from the raw Vision detection?
    ia = np.array([[f(r, "imgA_u"), f(r, "imgA_v")] for r in vision])
    have_ia = ~np.isnan(ia[:, 0])
    if have_ia.sum() > 10:
        d = np.sqrt(((ia[have_ia] - obs[have_ia]) ** 2).sum(1))
        print(f"\nARKit image-track vs Vision QR (captured-image px):")
        print(f"  median {np.median(d):.1f}  p90 {np.percentile(d, 90):.1f}  n={have_ia.sum()}")
        print("  (small = ARKit's vision pipeline internally sees the QR where")
        print("   Vision does; its geometric reprojection error is then a pose-")
        print("   vs-model inconsistency, not a detection issue)")

    # S rows: rendered-screen truth. display-path delta ~0 means the camera
    # background is drawn faithfully per displayTransform (H4 eliminated).
    if snaps:
        sq = np.array([[f(r, "snap_qr_u"), f(r, "snap_qr_v")] for r in snaps])
        orange = np.array([[f(r, "orange_u"), f(r, "orange_v")] for r in snaps])
        green = np.array([[f(r, "green_u"), f(r, "green_v")] for r in snaps])
        ok = ~np.isnan(sq[:, 0])
        print(f"\nRendered-screen snapshots: {len(snaps)} rows, QR found in {ok.sum()}")
        for name, ref in [("display-path (snapQR vs orange)", orange),
                          ("on-screen total (snapQR vs green)", green)]:
            m = ok & ~np.isnan(ref[:, 0])
            if m.sum() > 3:
                d = np.sqrt(((sq[m] - ref[m]) ** 2).sum(1))
                print(f"  {name}: median {np.median(d):.1f} px  p90 {np.percentile(d, 90):.1f} px  n={m.sum()}")

    # Model 3: does the error scale depend on focus / pitch?
    print("\nModel 3: per-sample scale estimate vs lens position / pitch")
    rad = np.sqrt(r2)
    good = rad > 200  # need leverage to estimate scale from a single sample
    if good.sum() > 10:
        s_i = (pc[good] * err[good]).sum(1) / r2[good] + 1.0
        for name, x in [("lensPosition", lens[good]), ("pitch(rad)", pitch[good])]:
            if np.nanstd(x) < 1e-4:
                print(f"  {name}: no variation in session")
                continue
            cor = np.corrcoef(x, s_i)[0, 1]
            print(f"  corr(scale, {name}) = {cor:+.3f}"
                  + ("   <- strong dependence!" if abs(cor) > 0.5 else ""))
        lo, hi = np.percentile(s_i, [10, 90])
        print(f"  per-sample scale p10..p90: {lo:.4f} .. {hi:.4f}")
    else:
        print("  not enough samples far from image center (tilt more).")


if __name__ == "__main__":
    main()
