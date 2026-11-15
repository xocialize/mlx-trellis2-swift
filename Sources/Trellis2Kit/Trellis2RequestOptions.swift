import Foundation
import MLXToolKit
import TRELLIS2

/// Per-request overrides read from `ImageTo3DRequest.metaData` (AB-A-0263, MES Mesh 0.1). A host that
/// keeps one `Trellis2Package` resident per lane and runs many jobs on it sets these per job; every
/// key that is absent (or `null`) falls back to the registration's `Trellis2Configuration`, so a
/// request without them runs exactly the configured job.
///
/// | key             | type                         | falls back to                          |
/// |-----------------|------------------------------|----------------------------------------|
/// | `seed`          | non-negative integer (UInt64; a decimal string carries values past 2^53) | `seed` |
/// | `steps`         | integer 1...50               | `steps`                                |
/// | `texture`       | bool                         | `texture`                              |
/// | `decimateFaces` | integer 1,000...2,000,000    | `decimateFaces` (nil = 120,000)        |
/// | `atlasSize`     | 512, 1024, 2048 or 4096, ≤ `maxAtlasSize` | 1024                      |
/// | `yUp`           | bool                         | `yUpOutput`                            |
/// | `pbrMaps`       | bool                         | false                                  |
/// | `pixal3d`       | bool                         | false (stock TRELLIS.2)                |
/// | `pixal3dFovDegrees` | number, (0, 120)         | 20 (Pixal3DCamera.defaultFovDegrees)   |
/// | `pixal3dFrame`  | "front" or "world"           | "front"                                |
/// | `pixal3dCameras`| array of 16-number arrays (row-major camera-to-world, one per view: image first, then `additionalViews`) | nil |
/// | `normalMap`     | bool                         | false                                  |
/// | `normalMapSize` | 512, 1024, 2048 or 4096, ≤ `maxAtlasSize` | the atlas size            |
/// | `aoMap`         | bool                         | false                                  |
/// | `aoMapSize`     | 512, 1024, 2048 or 4096, ≤ `maxAtlasSize` | the atlas size            |
/// | `aoRays`        | integer 1...256              | 64                                     |
/// | `aoDistance`    | number, (0, 2] (× bbox diagonal) | 0.25                               |
///
/// Detail maps: `normalMap` bakes the raw decoded surface's normals into a tangent-space map on the
/// decimated mesh (glTF `normalTexture`, with a TANGENT attribute). `aoMap` ray-casts ambient
/// occlusion on the output mesh (glTF `occlusionTexture`); with `pbrMaps` on and the AO at the atlas
/// size, it rides the metallic-roughness texture's R channel (ORM). Neither needs `texture`. A size
/// or ray key without its map switched on is accepted and ignored.
///
/// Pixal3D mode (AB-T-0485): projection-conditioned flows on the cascade tiers. Single view takes a
/// horizontal FOV (distance follows from it). Posed multi-view takes one camera-to-world matrix per
/// view (Z-up world, camera looks down −Z with +Y up; view 0 is the canonical front); without
/// cameras, `additionalViews` are refused in Pixal3D mode. The mesh comes back in Pixal3D's world
/// frame (Z-up, front camera at −Y); `"world"` maps it into the cameras' frame (multi-view only).
///
/// A present but malformed value throws `PackageError.unsupportedRequestFeature` before any weights
/// are touched; nothing is silently clamped.
public struct Trellis2RequestOptions: Sendable, Equatable {
    public var seed: UInt64
    public var steps: Int
    public var texture: Bool
    public var decimateFaces: Int
    public var atlasSize: Int
    public var yUp: Bool
    /// Also bake the metallic-roughness atlas (glTF: G = roughness, B = metallic) into the GLB.
    /// Ignored when `texture` is false (there is no tex decode to read it from).
    public var pbrMaps: Bool
    /// Pixal3D projection-conditioned generation instead of stock TRELLIS.2.
    public var pixal3d: Bool
    /// Pixal3D single-view horizontal field of view, degrees.
    public var pixal3dFovDegrees: Double
    /// Pixal3D output frame: "front" (canonical front camera frame) or "world" (multi-view: the c2w frame).
    public var pixal3dFrame: String
    /// Pixal3D posed multi-view cameras: one row-major 4×4 camera-to-world per view.
    public var pixal3dCameras: [[Double]]?
    /// Bake a tangent-space normal map (glTF `normalTexture`) from the raw decoded surface.
    public var normalMap: Bool
    /// Normal-map edge; nil = the atlas size.
    public var normalMapSize: Int?
    /// Bake an ambient-occlusion map (glTF `occlusionTexture`).
    public var aoMap: Bool
    /// AO-map edge; nil = the atlas size.
    public var aoMapSize: Int?
    /// AO rays per texel.
    public var aoRays: Int
    /// AO ray length, × the mesh's bounding-box diagonal.
    public var aoDistance: Double

    public enum Key {
        public static let seed = "seed"
        public static let steps = "steps"
        public static let texture = "texture"
        public static let decimateFaces = "decimateFaces"
        public static let atlasSize = "atlasSize"
        public static let yUp = "yUp"
        public static let pbrMaps = "pbrMaps"
        public static let pixal3d = "pixal3d"
        public static let pixal3dFovDegrees = "pixal3dFovDegrees"
        public static let pixal3dFrame = "pixal3dFrame"
        public static let pixal3dCameras = "pixal3dCameras"
        public static let normalMap = "normalMap"
        public static let normalMapSize = "normalMapSize"
        public static let aoMap = "aoMap"
        public static let aoMapSize = "aoMapSize"
        public static let aoRays = "aoRays"
        public static let aoDistance = "aoDistance"
    }

    public static let atlasSizes = [512, 1024, 2048, 4096]
    public static let stepsRange = 1 ... 50
    public static let decimateFacesRange = 1_000 ... 2_000_000
    /// The face target MeshBake uses when neither the request nor the configuration names one.
    public static let defaultDecimateFaces = 120_000
    public static let aoRaysRange = 1 ... 256

    /// The configuration's job, with nothing overridden.
    public init(configuration c: Trellis2Configuration) {
        seed = c.seed
        steps = c.steps
        texture = c.texture
        decimateFaces = c.decimateFaces ?? Self.defaultDecimateFaces
        atlasSize = 1024
        yUp = c.yUpOutput
        pbrMaps = false
        pixal3d = c.pixal3dOnly ?? false
        pixal3dFovDegrees = 20
        pixal3dFrame = "front"
        pixal3dCameras = nil
        normalMap = false
        normalMapSize = nil
        aoMap = false
        aoMapSize = nil
        aoRays = 64
        aoDistance = 0.25
    }

    /// The bake's detail-map request (`.none` unless `normalMap` or `aoMap` is on).
    public var detailMaps: DetailMapOptions {
        DetailMapOptions(normalMapSize: normalMap ? (normalMapSize ?? atlasSize) : nil,
                         aoMapSize: aoMap ? (aoMapSize ?? atlasSize) : nil,
                         aoRays: aoRays, aoMaxDistance: Float(aoDistance))
    }

    /// The configuration's job with `metaData`'s keys applied.
    public init(metaData: MetaData, configuration: Trellis2Configuration) throws {
        self.init(configuration: configuration)
        if let v = try Self.uint64(metaData, Key.seed) { seed = v }
        if let v = try Self.int(metaData, Key.steps, in: Self.stepsRange) { steps = v }
        if let v = try Self.bool(metaData, Key.texture) { texture = v }
        if let v = try Self.int(metaData, Key.decimateFaces, in: Self.decimateFacesRange) { decimateFaces = v }
        if let v = try Self.int(metaData, Key.atlasSize, in: 512 ... 4096) {
            guard Self.atlasSizes.contains(v) else {
                throw PackageError.unsupportedRequestFeature(
                    "metaData[\"atlasSize\"] = \(v): one of \(Self.atlasSizes.map(String.init).joined(separator: ", "))")
            }
            guard v <= configuration.maxAtlasSize else {   // the lane's charge assumes at most this
                throw PackageError.unsupportedRequestFeature(
                    "metaData[\"atlasSize\"] = \(v) is above this registration's maxAtlasSize \(configuration.maxAtlasSize)")
            }
            atlasSize = v
        }
        if let v = try Self.textureSize(metaData, Key.normalMapSize, max: configuration.maxAtlasSize) { normalMapSize = v }
        if let v = try Self.textureSize(metaData, Key.aoMapSize, max: configuration.maxAtlasSize) { aoMapSize = v }
        if let v = try Self.bool(metaData, Key.normalMap) { normalMap = v }
        if let v = try Self.bool(metaData, Key.aoMap) { aoMap = v }
        if let v = try Self.int(metaData, Key.aoRays, in: Self.aoRaysRange) { aoRays = v }
        if let v = try Self.number(metaData, Key.aoDistance) {
            guard v > 0, v <= 2 else {
                throw PackageError.unsupportedRequestFeature("metaData[\"aoDistance\"] = \(v): must be in (0, 2]")
            }
            aoDistance = v
        }
        if let v = try Self.bool(metaData, Key.yUp) { yUp = v }
        if let v = try Self.bool(metaData, Key.pbrMaps) { pbrMaps = v }
        if let v = try Self.bool(metaData, Key.pixal3d) { pixal3d = v }
        if let v = try Self.number(metaData, Key.pixal3dFovDegrees) {
            guard v > 0, v < 120 else {
                throw PackageError.unsupportedRequestFeature("metaData[\"pixal3dFovDegrees\"] = \(v): must be in (0, 120)")
            }
            pixal3dFovDegrees = v
        }
        if let v = Self.value(metaData, Key.pixal3dFrame) {
            guard case .string(let f) = v, f == "front" || f == "world" else {
                throw Self.refuse(Key.pixal3dFrame, "\"front\" or \"world\"", v)
            }
            pixal3dFrame = f
        }
        if let v = Self.value(metaData, Key.pixal3dCameras) {
            guard case .array(let views) = v, !views.isEmpty else {
                throw Self.refuse(Key.pixal3dCameras, "a non-empty array of 16-number arrays", v)
            }
            pixal3dCameras = try views.map { m -> [Double] in
                guard case .array(let xs) = m, xs.count == 16 else {
                    throw Self.refuse(Key.pixal3dCameras, "16-number row-major 4×4 matrices", m)
                }
                return try xs.map { x -> Double in
                    switch x {
                    case .int(let i): return Double(i)
                    case .double(let d) where d.isFinite: return d
                    default: throw Self.refuse(Key.pixal3dCameras, "finite numbers", x)
                    }
                }
            }
        }
    }

    // MARK: parsing

    private static func value(_ m: MetaData, _ key: String) -> MetaValue? {
        switch m[key] {
        case nil, .null?: nil
        case let v?: v
        }
    }

    private static func refuse(_ key: String, _ want: String, _ got: MetaValue) -> PackageError {
        .unsupportedRequestFeature("metaData[\"\(key)\"] must be \(want) (got \(got))")
    }

    private static func number(_ m: MetaData, _ key: String) throws -> Double? {
        guard let v = value(m, key) else { return nil }
        switch v {
        case .int(let i): return Double(i)
        case .double(let d) where d.isFinite: return d
        default: throw refuse(key, "a number", v)
        }
    }

    /// A texture edge: one of `atlasSizes`, at most the registration's `maxAtlasSize` (its charge
    /// assumes no larger 2×-supersampled grid).
    private static func textureSize(_ m: MetaData, _ key: String, max cap: Int) throws -> Int? {
        guard let v = try int(m, key, in: 512 ... 4096) else { return nil }
        guard atlasSizes.contains(v) else {
            throw PackageError.unsupportedRequestFeature(
                "metaData[\"\(key)\"] = \(v): one of \(atlasSizes.map(String.init).joined(separator: ", "))")
        }
        guard v <= cap else {
            throw PackageError.unsupportedRequestFeature(
                "metaData[\"\(key)\"] = \(v) is above this registration's maxAtlasSize \(cap)")
        }
        return v
    }

    private static func bool(_ m: MetaData, _ key: String) throws -> Bool? {
        guard let v = value(m, key) else { return nil }
        if case .bool(let b) = v { return b }
        throw refuse(key, "true or false", v)
    }

    /// An integer: `.int`, or a `.double` with no fractional part (JSON bridges sometimes send 2048.0).
    private static func int(_ m: MetaData, _ key: String, in range: ClosedRange<Int>) throws -> Int? {
        guard let v = value(m, key) else { return nil }
        let n: Int
        switch v {
        case .int(let i): n = i
        case .double(let d) where d == d.rounded() && abs(d) <= 1e15: n = Int(d)
        default: throw refuse(key, "an integer in \(range.lowerBound)...\(range.upperBound)", v)
        }
        guard range.contains(n) else {
            throw refuse(key, "an integer in \(range.lowerBound)...\(range.upperBound)", v)
        }
        return n
    }

    /// A non-negative integer seed: `.int`, an integral `.double` below 2^53, or a decimal `.string`
    /// (the only lossless way to carry a UInt64 above Int64.max through JSON).
    private static func uint64(_ m: MetaData, _ key: String) throws -> UInt64? {
        guard let v = value(m, key) else { return nil }
        switch v {
        case .int(let i) where i >= 0: return UInt64(i)
        case .double(let d) where d >= 0 && d == d.rounded() && d < 9_007_199_254_740_992: return UInt64(d)
        case .string(let s) where UInt64(s) != nil: return UInt64(s)!
        default: throw refuse(key, "a non-negative integer (or its decimal string)", v)
        }
    }
}
