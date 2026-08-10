# Diagnosing and Correcting the ARKit Projection Error

## Goal

Apple will not fix this, so we will. The goal, in three steps:

1. **Localize** the error to a pipeline stage — camera model (intrinsics /
   distortion), world tracking (geometry), or rendering (background display
   path) — using objective per-frame measurements, not eyeballing.
2. **Characterize** it — constant per device, or a function of focus (lens
   position), tilt, or radius from image center.
3. **Correct** it — produce a calibration procedure any of our apps can run
   (or ship per-device constants) so that projection, raycasting math, and
   rendering are pixel-accurate.

## Why the demo behaves the way it does (frame of reference)

- The RealityKit cube and `ARCamera.projectPoint()` (green dot) agree with
  each other, but both drift off the real-world feature in the video. So the
  inconsistency is between ARKit's *camera model* and the *actual optics +
  display path* — not between two ARKit APIs.
- The FY slider scales the portrait projection's vertical focal length.
  Portrait-vertical is the captured image's landscape-x axis, so
  "FY ≈ 1.020 fixes it" means: **the true fx is ~2% larger than ARKit's
  reported intrinsics** (or something masquerading as that).
- A pure focal-scale error produces a shift proportional to distance from the
  image center — which grows as tilting sweeps the anchor toward the screen
  edge. That's exactly the reported symptom, but principal-point error,
  radial distortion, and focus breathing produce similar-looking drift.
  The diagnostics below distinguish them by their different signatures.

## Hypotheses and their fingerprints

| # | Hypothesis | Fingerprint in the data |
|---|-----------|------------------------|
| H1 | **Focus breathing** — lens focal length changes 1–4% with focus; ARKit intrinsics don't track it | Fitted scale varies with `lensPosition`; error changes when AF is locked vs auto; `fx` column static while `lens` moves |
| H2 | **Static intrinsics miscalibration** | Fitted scale constant (~1.02) regardless of focus, tilt, distance |
| H3 | **Residual lens distortion** | Error grows with r³, not r — Model 2's `k1` term dominant |
| H4 | **Display-path mismatch** (RealityKit background crop ≠ projection assumption) | Captured-image-space error (QR test) is small, but on-screen misalignment is large; orange dot off the QR in the video |
| H5 | **World tracking / geometry error** | Error appears/changes under camera *translation* but not pure rotation; or QR world position unstable |

## How the diagnostics work

- **Ground truth**: aim the crosshair at the center of any QR code (paper or
  on a screen) and tap **Place Anchor**. The center-screen raycast direction
  passes through the principal point, so it is immune to focal-length error —
  the resulting world point is trustworthy.
- Every ~100 ms, Vision detects the QR center in the **raw captured image**
  and compares it to the intrinsics-based reprojection of the anchor. This
  comparison lives entirely in captured-image pixel space — the rendering
  path cannot contaminate it.
- The **orange ring** is the Vision detection mapped to screen via
  `displayTransform`. If it sits exactly on the QR in the camera video, the
  display path is faithful (H4 eliminated) and all error is camera-model.
- The HUD shows live intrinsics, `lensPosition` (focus), an AF lock toggle,
  and an online least-squares fit of scale + offset — the objective version
  of the "1.020" number.
- `projectPoint vs manual K` verifies ARKit's `projectPoint` equals
  `K·[R|t]` math (internal consistency; expected ≈ 0).

## Test protocol (on device)

Print or display a QR code (any content, the bigger and flatter the better).

1. **Baseline**: start AR, wait for tracking to settle. Tap **● Record**.
2. **Lock target**: aim crosshair at QR center from ~1 m, tap
   **Place Anchor**. Green dot and orange ring should coincide near center.
3. **Rotation sweep (the key test)**: keeping the phone *in place* (rotate,
   don't translate — pivot around the phone, not your body), slowly tilt so
   the QR sweeps from center to each screen edge and back. 20–30 s.
   Watch the HUD fit converge. Tap **Mark**.
4. **Focus sweep**: point at the QR, then at something very close (~15 cm) so
   AF refocuses, then back. Repeat a few times. Watch whether `lens` moves
   and whether `fx` moves with it, and whether the QR error changes. **Mark**.
5. **AF locked repeat**: tap **AF: Auto** → locked, redo step 3. **Mark**.
6. **Translation test**: walk half a meter left/right and re-aim. If error
   jumps only here, geometry (H5) is implicated. **Mark**.
7. **Distance sweep**: repeat step 3 from ~0.5 m and ~3 m (changes both focus
   and scale). **Mark**, then **■ Stop Rec** and **Share CSV** (AirDrop) or
   grab it from the Files app (On My iPhone → Focal point error demo).
8. Run `python3 analysis/analyze_csv.py <file.csv>` — it prints which
   hypothesis the data supports and the exact correction constants.

Quick visual checks along the way:

- Orange ring off the QR in the video → display-path problem (H4).
- Orange ring on the QR, green dot drifting → camera-model problem (H1/H2/H3).
- HUD `fx` static while `lens` moves → intrinsics don't track focus (H1).

## Correction strategy (once characterized)

- **H2 (static)**: one-time per-device-model calibration; apply corrected
  K′ = diag(s_x, s_y, 1)·K + principal offset in all our projection /
  unprojection / raycast math. For rendering: SceneKit and Metal accept a
  custom projection matrix; RealityKit's `ARView` does not, so either render
  with a corrected pipeline or scale the background video layer by the
  inverse factor.
- **H1 (focus)**: same correction but as a function of `lensPosition`
  (calibrate at 3–4 focus distances, interpolate), or lock AF where the app
  can tolerate it.
- **H3 (distortion)**: add the fitted k1 radial term to our projection math
  (cheap) — full undistortion of the video only if visual alignment at the
  extreme edges matters.
- **H4 (display)**: compensate the background transform ourselves (draw the
  camera image with our own corrected transform, or offset rendered content
  by the display-path delta).
- The QR procedure itself is app-embeddable: any of our apps can run a
  30-second self-calibration with a printed marker and persist the constants.
