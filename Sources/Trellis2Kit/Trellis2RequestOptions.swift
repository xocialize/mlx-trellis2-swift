import Foundation
import MLXToolKit

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

    public enum Key {
        public static let seed = "seed"
        public static let steps = "steps"
        public static let texture = "texture"
        public static let decimateFaces = "decimateFaces"
        public static let atlasSize = "atlasSize"
        public static let yUp = "yUp"
        public static let pbrMaps = "pbrMaps"
    }

    public static let atlasSizes = [512, 1024, 2048, 4096]
    public static let stepsRange = 1 ... 50
    public static let decimateFacesRange = 1_000 ... 2_000_000
    /// The face target MeshBake uses when neither the request nor the configuration names one.
    public static let defaultDecimateFaces = 120_000

    /// The configuration's job, with nothing overridden.
    public init(configuration c: Trellis2Configuration) {
        seed = c.seed
        steps = c.steps
        texture = c.texture
        decimateFaces = c.decimateFaces ?? Self.defaultDecimateFaces
        atlasSize = 1024
        yUp = c.yUpOutput
        pbrMaps = false
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
        if let v = try Self.bool(metaData, Key.yUp) { yUp = v }
        if let v = try Self.bool(metaData, Key.pbrMaps) { pbrMaps = v }
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
