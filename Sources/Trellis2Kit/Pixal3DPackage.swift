import Foundation
import MLXToolKit
import TRELLIS2

/// Pixal3D as its own `imageTo3D` lane (AB-T-0485 P3; mlx-engine AB-A-0284 / AB-D-0230, contract 1.64.0).
///
/// The same runtime as `Trellis2Package` — every request runs Pixal3D's projection-conditioned flows —
/// behind its own manifest: provenance `xocialize/pixal3d-mlx`, the Pixal3D res1536 lane's footprint, a
/// surface that advertises only the tiers Pixal3D honors (`res1024`, `res1536`) and declares the camera
/// controls (`ImageTo3DControls`: fov, pose, worldFrame, cameraReport). Register it beside the stock
/// lanes with an explicit id, e.g. `engine.register(Pixal3DPackage.registration,
/// configuration: Trellis2Configuration.pixal3dLane(ImageTo3DContract.res1024), id: "pixal3d")` —
/// without one its id derives from the provenance, which is distinct from the stock package's but
/// explicit is what mlx-engine asked for.
///
/// Requests take `ImageTo3DRequest.cameras` (one per view; `fovXDegrees` alone for a single photo,
/// `cameraToWorld` on every view for posed multi-view) and `.frame`; the v0.13.0 `metaData` keys stay as
/// the fallback for older hosts. The response reports view 0's camera in the returned mesh's frame.
@InferenceActor
public final class Pixal3DPackage: ModelPackage {
    public typealias Configuration = Trellis2Configuration

    /// The camera controls Pixal3D honors. fov: the single-view horizontal field of view (the package
    /// accepts (0, 120); declared as the closed range it is tested on). pose: posed multi-view (needs
    /// the `_mv` weights locally). worldFrame: the cameras' frame. cameraReport: `ImageTo3DResponse.camera`.
    public nonisolated static let controls = ImageTo3DControls(features: [.fov, .pose, .worldFrame, .cameraReport],
                                                               fovXDegreesRange: 1 ... 119)

    public nonisolated static var manifest: PackageManifest {
        let t = Trellis2Tier.res1536.hrTokenCeiling ?? 49_152
        return Trellis2Package.makeManifest(
            sourceRepo: Trellis2Configuration.pixal3dRepo,
            footprint: QuantFootprint(quant: .bf16, residentBytes: Trellis2Footprint.pixal3dResidentCascade,
                                      peakActivationBytes: Trellis2Footprint.pixal3dCascadeActivation(tokens: t)),
            surface: ImageTo3DContract.descriptor(
                name: "pixal3d-image-to-3d",
                summary: "Pixal3D (TRELLIS.2 with pixel-aligned, camera-projected conditioning): a UV-textured "
                       + "3D mesh (GLB) that follows the photo pixel for pixel. Takes the photo's horizontal "
                       + "field of view (default 20°), or posed views with camera-to-world matrices; reports the "
                       + "camera. Modes res1024/res1536.",
                modes: [ImageTo3DContract.res1024, ImageTo3DContract.res1536],
                controls: controls))
    }

    public nonisolated static var registration: PackageRegistration { .of(Pixal3DPackage.self) }
    /// The id mlx-engine asked Pixal3D lanes to register under (AB-A-0284).
    public nonisolated static let suggestedID = "pixal3d"

    private let inner: Trellis2Package

    /// Any configuration becomes a Pixal3D one (`pixal3dOnly`); use `Trellis2Configuration.pixal3dLane`.
    public nonisolated init(configuration: Configuration) {
        var c = configuration
        c.pixal3dOnly = true
        // Pixal3D has no res512 path: a configuration left on the stock default tier runs res1024
        if !Trellis2Tier(mode: c.defaultMode).isCascade { c.defaultMode = ImageTo3DContract.res1024 }
        if let ceiling = c.tierCeiling, !Trellis2Tier(mode: ceiling).isCascade { c.tierCeiling = ImageTo3DContract.res1024 }
        inner = Trellis2Package(configuration: c)
    }

    public func load() async throws { try await inner.load() }
    public func unload() async { await inner.unload() }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()
        if let req = request as? ImageTo3DRequest {
            if let why = ImageTo3DControls.refusal(for: req, declared: Self.controls) {
                throw PackageError.unsupportedRequestFeature(why)
            }
            if let mode = req.mode, !Self.manifest.surfaces[0].supportedModes.contains(mode) {
                throw PackageError.unsupportedRequestFeature(
                    "mode \(mode.rawValue): Pixal3D runs res1024 / res1536 (no Pixal3D res512 path)")
            }
        }
        return try await inner.run(request)
    }
}
