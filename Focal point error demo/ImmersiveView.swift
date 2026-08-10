//
//  ImmersiveView.swift
//  Focal point error demo
//
//  Created by Tyler Collins on 8/4/25.
//

import SwiftUI
import RealityKit
import ARKit

struct ImmersiveView: View {
    @StateObject private var arState = ARState()
    @State private var fxScale: Float = 1.0
    @State private var fyScale: Float = 1.0
    @State private var showDiagnostics = true
    @State private var showPhotos = false
    @State private var showGuidedCal = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            ARViewContainer(arState: arState)
                .edgesIgnoringSafeArea(.all)

            // Overlay for markers
            GeometryReader { geometry in
                ZStack {
                    // Center crosshair
                    Image(systemName: "plus")
                        .font(.system(size: 30))
                        .foregroundColor(.white)
                        .background(Circle().fill(Color.black.opacity(0.3)).frame(width: 40, height: 40))
                        .position(x: geometry.size.width / 2, y: geometry.size.height / 2)

                    // OEM projection marker (green)
                    if let oemPoint = arState.oemProjectedPoint {
                        Circle()
                            .fill(Color.green)
                            .frame(width: 10, height: 10)
                            .position(x: CGFloat(oemPoint.x), y: CGFloat(oemPoint.y))
                    }

                    // Custom projection marker (red)
                    if let customPoint = arState.customProjectedPoint {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 10, height: 10)
                            .position(x: CGFloat(customPoint.x), y: CGFloat(customPoint.y))
                    }

                    // Vision QR detection mapped to screen (orange ring).
                    // Should sit exactly on the QR center in the camera video if
                    // the background rendering path matches displayTransform.
                    if let visionPoint = arState.engine.visionScreenPoint {
                        Circle()
                            .stroke(Color.orange, lineWidth: 2)
                            .frame(width: 16, height: 16)
                            .position(x: visionPoint.x, y: visionPoint.y)
                    }

                    // Calibration station ring: park the QR inside this ring
                    if let station = arState.engine.currentStation {
                        let px = station.pos.x * geometry.size.width
                        let py = station.pos.y * geometry.size.height
                        ZStack {
                            Circle()
                                .stroke(style: StrokeStyle(lineWidth: 3, dash: [8, 6]))
                                .foregroundColor(.cyan)
                                .frame(width: 90, height: 90)
                            Text("QR here\nwalk in↔out")
                                .font(.system(size: 11, weight: .bold))
                                .multilineTextAlignment(.center)
                                .foregroundColor(.cyan)
                                .shadow(color: .black, radius: 2)
                                .offset(y: 70)
                        }
                        .position(x: px, y: py)
                    }

                    // Corrected projection (magenta): fitted camera-model fix
                    if let corrPoint = arState.engine.correctedScreenPoint {
                        Circle()
                            .fill(Color(red: 1, green: 0.3, blue: 1))
                            .frame(width: 10, height: 10)
                            .position(x: corrPoint.x, y: corrPoint.y)
                    }

                    // ARKit's own image-tracking estimate of the QR center (yellow)
                    if let imgPoint = arState.imageAnchorScreenPoint {
                        Circle()
                            .stroke(Color.yellow, lineWidth: 2)
                            .frame(width: 22, height: 22)
                            .position(x: imgPoint.x, y: imgPoint.y)
                    }
                }
            }
            .ignoresSafeArea()

            VStack {
                HStack {
                    Button("Close") {
                        dismiss()
                    }
                    .padding(10)
                    .background(Color.black.opacity(0.7))
                    .foregroundColor(.white)
                    .cornerRadius(8)

                    Spacer()

                    Button(showDiagnostics ? "Hide Diag" : "Diag") {
                        showDiagnostics.toggle()
                    }
                    .padding(10)
                    .background(Color.black.opacity(0.7))
                    .foregroundColor(.white)
                    .cornerRadius(8)

                    Spacer()

                    Button("Place Anchor") {
                        arState.placeAnchor()
                    }
                    .padding(10)
                    .background(Color.blue.opacity(0.7))
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .padding()

                if showDiagnostics {
                    HStack {
                        DiagnosticsPanel(engine: arState.engine)
                        Spacer()
                    }
                    .padding(.horizontal)
                }

                Spacer()

                if showDiagnostics {
                    diagnosticsControls
                }

                // Color legend
                HStack(spacing: 14) {
                    legendDot(.green, "ARKit")
                    legendDot(.red, "Adjusted")
                    legendDot(.orange, "Vision QR")
                    legendDot(.yellow, "Img track")
                    legendDot(Color(red: 1, green: 0.3, blue: 1), "Corrected")
                }
                .padding(.horizontal)
                .padding(.vertical, 6)
                .background(Color.black.opacity(0.7))
                .cornerRadius(8)
                .padding(.bottom, 6)

                // FX/FY adjustment sliders
                VStack(spacing: 12) {
                    VStack(spacing: 2) {
                        Text("FX Scale: \(String(format: "%.3f", fxScale))")
                            .foregroundColor(.white)
                            .font(.caption)
                        Slider(value: $fxScale, in: 0.95...1.05)
                            .accentColor(.red)
                            .onChange(of: fxScale) { newValue in
                                arState.fxScale = newValue
                                arState.engine.fxSlider = newValue
                            }
                    }

                    VStack(spacing: 2) {
                        Text("FY Scale: \(String(format: "%.3f", fyScale))")
                            .foregroundColor(.white)
                            .font(.caption)
                        Slider(value: $fyScale, in: 0.95...1.05)
                            .accentColor(.red)
                            .onChange(of: fyScale) { newValue in
                                arState.fyScale = newValue
                                arState.engine.fySlider = newValue
                            }
                    }
                }
                .padding(10)
                .background(Color.black.opacity(0.7))
                .cornerRadius(10)
                .padding([.horizontal, .bottom])
            }
        }
        .sheet(isPresented: $showPhotos) {
            PhotoViewerSheet(engine: arState.engine)
        }
        .fullScreenCover(isPresented: $showGuidedCal, onDismiss: {
            arState.engine.adoptGuidedCalibration()
        }) {
            ARDistortionCalibrationView()
        }
        .onAppear {
            arState.fxScale = fxScale
            arState.fyScale = fyScale
            arState.engine.fxSlider = fxScale
            arState.engine.fySlider = fyScale
        }
    }

    private var diagnosticsControls: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                diagButton(arState.engine.hud.afLocked ? "AF: Locked" : "AF: Auto",
                           color: arState.engine.hud.afLocked ? .orange : .gray) {
                    arState.engine.toggleAFLock()
                }
                diagButton(arState.engine.hud.recording ? "■ Stop Rec" : "● Record",
                           color: arState.engine.hud.recording ? .red : .gray) {
                    arState.engine.toggleRecording()
                }
                diagButton("Mark", color: .gray) {
                    arState.engine.markEvent("mark")
                }
                diagButton("Reset Fit", color: .gray) {
                    arState.engine.resetFit()
                }
                diagButton("Photo", color: .blue) {
                    arState.engine.capturePhoto()
                }
                diagButton("View", color: .blue) {
                    showPhotos = true
                }
                diagButton("Guided Cal", color: .purple) {
                    showGuidedCal = true
                }
            }
            HStack(spacing: 8) {
                diagButton(arState.engine.hud.calibrating ? "Cancel Cal" : "Calibrate",
                           color: arState.engine.hud.calibrating ? .red : .purple) {
                    if arState.engine.hud.calibrating {
                        arState.engine.cancelCalibration()
                    } else {
                        arState.engine.startCalibration()
                    }
                }
                if arState.engine.hud.calibrating {
                    diagButton("Next Zone (\((arState.engine.calStationIndex % DiagnosticsEngine.calStations.count) + 1)/\(DiagnosticsEngine.calStations.count))",
                               color: .cyan) {
                        arState.engine.nextStation()
                    }
                }
                if !arState.engine.hud.recording, let url = arState.engine.logger.url {
                    ShareLink(item: url) {
                        Text("Share CSV")
                            .font(.caption)
                            .padding(8)
                            .background(Color.blue.opacity(0.7))
                            .foregroundColor(.white)
                            .cornerRadius(6)
                    }
                }
            }
        }
        .padding(.horizontal)
    }

    private func diagButton(_ title: String, color: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.caption)
                .padding(8)
                .background(color.opacity(0.7))
                .foregroundColor(.white)
                .cornerRadius(6)
        }
    }

    private func legendDot(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 5) {
            Circle()
                .fill(color)
                .frame(width: 10, height: 10)
            Text(label)
                .foregroundColor(.white)
                .font(.caption)
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    let arState: ARState

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)

        // Configure AR session
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]
        // Same as Cloneable standard accuracy: enable high-res still capture.
        if let fmt = ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing {
            config.videoFormat = fmt
        }
        // Track our QR target as a reference image: gives ARKit's own
        // vision-based world estimate of the QR as an independent stream.
        if let ref = DiagnosticsEngine.makeQRReferenceImage() {
            config.detectionImages = [ref]
            config.maximumNumberOfTrackedImages = 1
        }
        arView.session.delegate = arState
        arView.session.run(config)

        arState.arView = arView
        arState.engine.arView = arView

        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}

class ARState: NSObject, ObservableObject, ARSessionDelegate {
    @Published var oemProjectedPoint: CGPoint?
    @Published var customProjectedPoint: CGPoint?
    @Published var imageAnchorScreenPoint: CGPoint?

    var fxScale: Float = 1.0
    var fyScale: Float = 1.0

    let engine = DiagnosticsEngine()

    weak var arView: ARView?
    private var anchorEntity: AnchorEntity?
    private var targetWorld: simd_float3?
    private var imageAnchor: ARImageAnchor?
    private var recordingStarted = false
    private var autoTarget: simd_float3?
    // ARKit-pinned backing anchor for the target, mirroring production's
    // DriftCorrectionTracker: reading its live transform keeps the frozen
    // world point aligned as ARKit re-optimizes its map (drift corrections).
    private var targetAnchor: ARAnchor?
    private var autoLockCounter = 0
    private var lastAutoPoint: CGPoint?

    func placeAnchor() {
        guard let arView = arView else { return }

        // Remove existing anchor if any
        if let existing = anchorEntity {
            arView.scene.removeAnchor(existing)
        }

        // Cast ray from center of screen. The center ray passes through the
        // principal point, so its direction is immune to focal length error —
        // making the resulting world point trustworthy ground truth.
        let screenCenter = CGPoint(x: arView.bounds.width / 2, y: arView.bounds.height / 2)

        guard let raycastResult = arView.raycast(
            from: screenCenter,
            allowing: .estimatedPlane,
            alignment: .any
        ).first else {
            engine.placementFailed()
            return
        }

        // Create anchor at hit location
        let anchor = AnchorEntity(world: raycastResult.worldTransform)

        // Create small transparent cube
        var material = SimpleMaterial(color: .systemBlue, isMetallic: false)
        material.color = SimpleMaterial.BaseColor(tint: UIColor.systemBlue.withAlphaComponent(0.15))
        let cube = ModelEntity(
            mesh: .generateBox(size: 0.02),
            materials: [material]
        )

        anchor.addChild(cube)
        arView.scene.addAnchor(anchor)

        let world = simd_float3(raycastResult.worldTransform.columns.3.x,
                                raycastResult.worldTransform.columns.3.y,
                                raycastResult.worldTransform.columns.3.z)
        // Anchor depth error from a close-range raycast turns into parallax
        // that dwarfs the lens residual when verifying from offset viewpoints
        // (~1.5 cm at 0.5 m ≈ 10-15 px in photos). Require a 5 ft standoff.
        let camPos = arView.cameraTransform.translation
        let dist = simd_length(world - camPos)
        if dist < 1.5 {
            arView.scene.removeAnchor(anchor)
            engine.placementTooClose(distance: dist)
            return
        }
        self.anchorEntity = anchor
        self.targetWorld = world
        self.autoTarget = nil
        if let old = targetAnchor { arView.session.remove(anchor: old) }
        let pin = ARAnchor(name: "verify_target", transform: raycastResult.worldTransform)
        arView.session.add(anchor: pin)
        targetAnchor = pin
        engine.targetChanged(distance: dist)
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for a in anchors where a is ARImageAnchor {
            imageAnchor = a as? ARImageAnchor
            engine.markEvent("image_anchor_detected")
        }
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for a in anchors where a is ARImageAnchor {
            imageAnchor = a as? ARImageAnchor
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let arView = arView else { return }
        let viewportSize = arView.bounds.size
        guard viewportSize.width > 0, viewportSize.height > 0 else { return }

        if !recordingStarted {
            recordingStarted = true
            engine.startRecordingIfNeeded()
        }

        // Auto-anchor, calibration-style: when the Vision-detected QR center
        // sits stably at screen center (within 15% of the short side) for
        // ~0.75 s, raycast through the detection and freeze the target — no
        // manual placement. Raycast must hit (no image-anchor fallback: a
        // miss would anchor at the wrong depth) and must be 5+ ft away.
        if targetWorld == nil, autoTarget == nil {
            if let vp = engine.visionScreenPoint {
                let center = CGPoint(x: viewportSize.width / 2, y: viewportSize.height / 2)
                let near = hypot(vp.x - center.x, vp.y - center.y) < min(viewportSize.width, viewportSize.height) * 0.15
                let stable = lastAutoPoint.map { hypot(vp.x - $0.x, vp.y - $0.y) < 8 } ?? false
                lastAutoPoint = vp
                autoLockCounter = (near && stable) ? autoLockCounter + 1 : 0
                if !near, engine.hud.calStatus.isEmpty || engine.hud.calStatus.hasPrefix("center the QR") {
                    engine.hud.calStatus = "center the QR at screen center (5+ ft away) to auto-place the anchor"
                }
                if autoLockCounter >= 45 {
                    autoLockCounter = 0
                    if let hit = arView.raycast(from: vp, allowing: .estimatedPlane, alignment: .any).first {
                        let world = simd_float3(hit.worldTransform.columns.3.x,
                                                hit.worldTransform.columns.3.y,
                                                hit.worldTransform.columns.3.z)
                        let camPos = arView.cameraTransform.translation
                        let dist = simd_length(world - camPos)
                        if dist < 1.5 {
                            engine.placementTooClose(distance: dist)
                        } else {
                            autoTarget = world
                            if let old = targetAnchor { arView.session.remove(anchor: old) }
                            let pin = ARAnchor(name: "verify_target", transform: hit.worldTransform)
                            arView.session.add(anchor: pin)
                            targetAnchor = pin
                            engine.targetChanged(distance: dist)
                            engine.markEvent("auto_target_locked")
                            engine.hud.calStatus = String(format: "anchor placed %.1f m — now STEP left & right while panning the QR to the corners", dist)
                        }
                    } else {
                        engine.hud.calStatus = "no surface behind QR — move slightly or scan the wall first"
                    }
                }
            }
        }

        var targetSource = "none"
        // Prefer the ARKit-pinned anchor's live transform: it receives the
        // session's drift/relocalization corrections that a frozen coordinate
        // would silently miss.
        let pinned = targetAnchor.map { a in
            simd_float3(a.transform.columns.3.x, a.transform.columns.3.y, a.transform.columns.3.z)
        }
        var effectiveTarget = targetWorld
        if targetWorld != nil {
            effectiveTarget = pinned ?? targetWorld
            targetSource = "manual"
        } else if let at = autoTarget {
            effectiveTarget = pinned ?? at
            targetSource = "auto"
        } else if let ia = imageAnchor, ia.isTracked {
            let c = ia.transform.columns.3
            effectiveTarget = simd_float3(c.x, c.y, c.z)
            targetSource = "image"
        }

        // Yellow marker: ARKit's image-track estimate on screen.
        if let ia = imageAnchor, ia.isTracked {
            let c = ia.transform.columns.3
            imageAnchorScreenPoint = frame.camera.projectPoint(
                simd_float3(c.x, c.y, c.z),
                orientation: .portrait,
                viewportSize: viewportSize
            )
        } else {
            imageAnchorScreenPoint = nil
        }

        if let world = effectiveTarget {
            // OEM projection using ARCamera.projectPoint
            oemProjectedPoint = frame.camera.projectPoint(
                world,
                orientation: .portrait,
                viewportSize: viewportSize
            )

            // Custom projection with FX/FY scaling
            customProjectedPoint = customProjectPoint(
                worldPosition: world,
                camera: frame.camera,
                viewportSize: viewportSize
            )
        } else {
            oemProjectedPoint = nil
            customProjectedPoint = nil
        }

        engine.process(frame: frame,
                       targetWorld: effectiveTarget,
                       targetSource: targetSource,
                       imageAnchor: imageAnchor,
                       viewportSize: viewportSize)
    }

    private func customProjectPoint(worldPosition: simd_float3, camera: ARCamera, viewportSize: CGSize) -> CGPoint {
        // Get camera matrices
        let viewMatrix = camera.viewMatrix(for: .portrait)
        let projectionMatrix = camera.projectionMatrix(for: .portrait, viewportSize: viewportSize, zNear: 0.001, zFar: 1000)

        // Apply FX/FY scaling to projection matrix
        var scaledProjection = projectionMatrix
        scaledProjection[0][0] *= fxScale  // Scale fx
        scaledProjection[1][1] *= fyScale  // Scale fy

        // Transform to clip space
        let worldPoint = simd_float4(worldPosition.x, worldPosition.y, worldPosition.z, 1.0)
        let clipPoint = scaledProjection * (viewMatrix * worldPoint)

        // Perspective divide to NDC
        let ndcPoint = simd_float3(clipPoint.x, clipPoint.y, clipPoint.z) / clipPoint.w

        // Convert to screen coordinates
        let screenX = (ndcPoint.x + 1.0) * 0.5 * Float(viewportSize.width)
        let screenY = (1.0 - ndcPoint.y) * 0.5 * Float(viewportSize.height)

        return CGPoint(x: CGFloat(screenX), y: CGFloat(screenY))
    }
}
