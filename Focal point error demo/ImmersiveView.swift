//
//  ImmersiveView.swift
//  Focal point error demo
//
//  Created by Tyler Collins on 8/4/25.
//

import SwiftUI
import RealityKit
import ARKit
import Combine

struct ImmersiveView: View {
    @StateObject private var arState = ARState()
    @State private var fxScale: Float = 1.0
    @State private var fyScale: Float = 1.020
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
                }
            }
            .ignoresSafeArea()
            
            VStack {
                HStack {
                    Button("Close") {
                        dismiss()
                    }
                    .padding()
                    .background(Color.black.opacity(0.7))
                    .foregroundColor(.white)
                    .cornerRadius(8)
                    
                    Spacer()
                    
                    Button("Place Anchor") {
                        arState.placeAnchor()
                    }
                    .padding()
                    .background(Color.blue.opacity(0.7))
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .padding()
                
                Spacer()
                
                // Color legend
                HStack(spacing: 20) {
                    HStack(spacing: 5) {
                        Circle()
                            .fill(Color.green)
                            .frame(width: 10, height: 10)
                        Text("ARKit")
                            .foregroundColor(.white)
                            .font(.caption)
                    }
                    HStack(spacing: 5) {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 10, height: 10)
                        Text("Adjusted focal point")
                            .foregroundColor(.white)
                            .font(.caption)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.7))
                .cornerRadius(8)
                .padding(.bottom, 10)
                
                // FX/FY adjustment sliders
                VStack(spacing: 20) {
                    VStack {
                        Text("FX Scale: \(String(format: "%.3f", fxScale))")
                            .foregroundColor(.white)
                            .font(.caption)
                        Slider(value: $fxScale, in: 0.95...1.05)
                            .accentColor(.red)
                            .onChange(of: fxScale) { newValue in
                                arState.fxScale = newValue
                            }
                    }
                    
                    VStack {
                        Text("FY Scale: \(String(format: "%.3f", fyScale))")
                            .foregroundColor(.white)
                            .font(.caption)
                        Slider(value: $fyScale, in: 0.95...1.05)
                            .accentColor(.red)
                            .onChange(of: fyScale) { newValue in
                                arState.fyScale = newValue
                            }
                    }
                }
                .padding()
                .background(Color.black.opacity(0.7))
                .cornerRadius(10)
                .padding()
            }
        }
        .onAppear {
            arState.fxScale = fxScale
            arState.fyScale = fyScale
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
        arView.session.run(config)
        
        arState.arView = arView
        
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {}
}

class ARState: ObservableObject {
    @Published var oemProjectedPoint: CGPoint?
    @Published var customProjectedPoint: CGPoint?
    @Published var fxScale: Float = 1.0
    @Published var fyScale: Float = 1.0
    
    weak var arView: ARView?
    private var anchorEntity: AnchorEntity?
    private var cubeEntity: ModelEntity?
    private var updateTimer: AnyCancellable?
    
    init() {
        // Start update timer for projections
        updateTimer = Timer.publish(every: 1.0/30.0, on: .main, in: .common)
            .autoconnect()
            .sink { _ in
                self.updateProjections()
            }
    }
    
    func placeAnchor() {
        guard let arView = arView else { return }
        
        // Remove existing anchor if any
        if let existing = anchorEntity {
            arView.scene.removeAnchor(existing)
        }
        
        // Cast ray from center of screen
        let screenCenter = CGPoint(x: arView.bounds.width / 2, y: arView.bounds.height / 2)
        
        guard let raycastResult = arView.raycast(
            from: screenCenter,
            allowing: .estimatedPlane,
            alignment: .any
        ).first else { return }
        
        // Create anchor at hit location
        let anchor = AnchorEntity(world: raycastResult.worldTransform)
        
        // Create small transparent cube
        var material = SimpleMaterial(color: .systemBlue, isMetallic: false)
        material.color = SimpleMaterial.BaseColor(tint: UIColor.systemBlue.withAlphaComponent(0.3))
        let cube = ModelEntity(
            mesh: .generateBox(size: 0.05),
            materials: [material]
        )
        
        anchor.addChild(cube)
        arView.scene.addAnchor(anchor)
        
        self.anchorEntity = anchor
        self.cubeEntity = cube
    }
    
    private func updateProjections() {
        guard let arView = arView,
              let anchor = anchorEntity,
              let frame = arView.session.currentFrame else {
            oemProjectedPoint = nil
            customProjectedPoint = nil
            return
        }
        
        let worldPosition = anchor.position(relativeTo: nil)
        
        // OEM projection using ARCamera.projectPoint
        let oemPoint2D = frame.camera.projectPoint(
            simd_float3(worldPosition.x, worldPosition.y, worldPosition.z),
            orientation: .portrait,
            viewportSize: arView.bounds.size
        )
        oemProjectedPoint = oemPoint2D
        
        // Custom projection with FX/FY scaling
        let customPoint2D = customProjectPoint(
            worldPosition: simd_float3(worldPosition.x, worldPosition.y, worldPosition.z),
            camera: frame.camera,
            viewportSize: arView.bounds.size
        )
        customProjectedPoint = customPoint2D
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