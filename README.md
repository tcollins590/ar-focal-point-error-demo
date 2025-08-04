# AR Focal Point Error Demo

A minimal iOS app demonstrating projection discrepancies in ARKit when using adjusted focal length parameters. This project was created to report a potential issue with ARKit's camera projection accuracy to Apple.

## Problem Statement

ARKit's `ARCamera.projectPoint()` method appears to use focal length values that don't perfectly match the physical camera characteristics of iOS devices. This results in systematic projection errors that become more pronounced when objects are further from the screen center or when the camera is tilted.

## What This App Demonstrates

The app visualizes the difference between:
- **Green dot**: ARKit's native `ARCamera.projectPoint()` projection
- **Red dot**: Custom projection using ARKit's view/projection matrices with adjustable focal length scaling

When placing an anchor in AR space, both dots should theoretically overlap perfectly. However, adjusting the focal length scaling factors (FX/FY) can achieve better alignment, suggesting that ARKit's internal camera model may benefit from device-specific calibration.

## Technical Implementation

### Custom Projection Method

The custom projection implementation uses ARKit's own transformation matrices but applies focal length scaling:

```swift
// Get camera matrices from ARKit
let viewMatrix = camera.viewMatrix(for: .portrait)
let projectionMatrix = camera.projectionMatrix(for: .portrait, viewportSize: viewportSize, zNear: 0.001, zFar: 1000)

// Apply focal length scaling
var scaledProjection = projectionMatrix
scaledProjection[0][0] *= fxScale  // Scale fx (horizontal focal length)
scaledProjection[1][1] *= fyScale  // Scale fy (vertical focal length)

// Transform world point to screen coordinates
let worldPoint = simd_float4(worldPosition.x, worldPosition.y, worldPosition.z, 1.0)
let clipPoint = scaledProjection * (viewMatrix * worldPoint)
let ndcPoint = simd_float3(clipPoint.x, clipPoint.y, clipPoint.z) / clipPoint.w

// Convert NDC to screen coordinates
let screenX = (ndcPoint.x + 1.0) * 0.5 * Float(viewportSize.width)
let screenY = (1.0 - ndcPoint.y) * 0.5 * Float(viewportSize.height)
```

### Key Observations

1. **Default Behavior**: With FX/FY scale at 1.0, the red dot (custom) should match the green dot (ARKit) exactly, as we're using ARKit's own matrices.

2. **Empirical Adjustment**: On tested devices, an FY scale of ~1.020 (2% increase) often provides better alignment, especially when the camera is tilted up or down.

3. **Tilt-Dependent Error**: The projection error becomes more pronounced when the device is tilted, suggesting the focal length discrepancy compounds with viewing angle.

## How to Use

1. Build and run on a physical iOS device (ARKit doesn't work in simulator)
2. Tap "Enter AR" to start the AR session
3. Point the camera at a surface and tap "Place Anchor" to place a semi-transparent blue cube
4. Observe the green (ARKit) and red (adjusted) projection dots
5. Adjust the FX/FY sliders to see how focal length scaling affects the red dot's position
6. Try tilting the device up and down to observe how the error changes with viewing angle

## Implications

This demonstration suggests that:
- ARKit's camera intrinsics may not perfectly match the physical camera properties
- Device-specific calibration could improve AR projection accuracy
- Applications requiring precise AR alignment (measurements, annotations, etc.) may benefit from custom projection calibration

## Building the Project

1. Open `Focal point error demo.xcodeproj` in Xcode
2. Select your development team in the project settings
3. Build and run on a physical iOS device with ARKit support

## Requirements

- iOS 15.0+
- Xcode 14.0+
- Physical iOS device (iPhone/iPad with ARKit support)

## License

This demo is provided as-is for bug reporting and educational purposes.