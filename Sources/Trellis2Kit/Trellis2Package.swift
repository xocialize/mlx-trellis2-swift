import Foundation
import MLX
import MLXToolKit
import TRELLIS2

// MARK: - Configuration (C9)

/// Init-time configuration for the TRELLIS.2 image→3D package.
///
/// `defaultMode` selects the resolution tier when a request omits `mode`. When
/// `weightsRootOverride` is nil and the engine stamped `modelsRootDirectory`, `load()`
/// auto-materializes the declared `weightSources` (the consolidated `xocialize/trellis2-mlx`
/// snapshot) into the ModelStore layout and loads from there (engine ≥0.19.0 MAT contract).
/// An explicit `weightsRootOverride` is the dev escape hatch and never touches the network.
public struct Trellis2Configuration: PackageConfiguration, ModelStorable, QuantConfigured {
    public var quant: Quant
    public var defaultMode: Mode

    /// Pin this registration to one tier: a lane (AB-A-0262). When set, the footprint hints are this
    /// tier's and `run()` refuses a request whose `mode` is ABOVE it (a lower tier fits inside the
    /// lane's charge and runs). nil (default) = the pre-0.11 behaviour: charged for `defaultMode`,
    /// any request mode accepted — an app that roams tiers should then configure the highest one.
    /// A host that schedules against one memory budget (Gaea) registers one lane per tier.
    public var tierCeiling: Mode?

    /// The largest `metaData["atlasSize"]` this registration accepts (AB-A-0263); a larger request is
    /// refused. It is part of the footprint: the bake supersamples 2×, so a 4096 atlas rasterizes
    /// 8192², and on res512 — where the bake, not the HR flow, is the peak — that moves the charge
    /// (see `Trellis2Footprint.activationRes512(maxAtlasSize:)`). One of 512, 1024, 2048, 4096.
    /// Default 1024 — v0.11.0's atlas and charges; a host offering larger textures raises it.
    public var maxAtlasSize: Int
    public var modelsRootDirectory: URL?   // ModelStorable — the engine stamps the store root

    /// Explicit weights root (out-of-sandbox snapshot). When set, `load()` resolves the consolidated
    /// snapshot under this directory FIRST — supporting both a flat `<root>/<file>` layout and the
    /// `<root>/xocialize/trellis2-mlx/<file>` store layout. A sandboxed app sets this to a
    /// security-scoped folder; nil → the engine store / auto-materialization path.
    public var weightsRootOverride: URL?

    /// Diffusion steps per sampler stage (SS / shape / tex). 12 is the HF default.
    public var steps: Int

    /// Base RNG seed. SS noise + the shape/tex SLat noise draws derive from it deterministically.
    public var seed: UInt64

    /// Run the texture path (tex SLat flow + tex decoder → base-color UV texture). Default on;
    /// false = geometry-only flat-gray mesh (skips a 2nd flow + 2nd decode, ~2× faster).
    public var texture: Bool

    /// Decimate the extracted mesh to this face target before GLB encode (matches the HF `to_glb`
    /// tail). nil = MeshBake's default (~120k).
    public var decimateFaces: Int?
    /// UV unwrap backend for the bake: "xatlas" (default) or "provenance"
    /// (UV-UNWRAP-METAL-PLAN.md — ~8x faster bake, near-parity quality).
    public var unwrapBackend: String?

    /// Opt-in full-workflow profiling (BENCHMARKS.md). When set, `run()` writes one flat
    /// per-generation stage-timing record (JSONSerialization, sortedKeys — the unwrapbench/bakeab
    /// house convention) to this path AND logs a `JSON {…}` line. nil (default) = no profiling,
    /// zero overhead. Set from env `METRICS_JSON` by `trellis2-run-engine`.
    public var metricsPath: String?

    /// SDPA precision for the CFG shape SLat flows: "fp32" (default — parity-safe),
    /// "fp16", or "bf16". Same fused kernel either way; this is only the Q/K/V cast.
    /// The cascade's HR shape flow is ~54% of res1536 e2e, so this is THE speed lever —
    /// but CFG amplifies SDPA rounding noise (speckle), hence the conservative default.
    /// SS always runs fp32 regardless. Unrecognized values fall back to fp32.
    public var slatCfgAttention: String?
    /// EXPERIMENTAL CFG-cost knobs for the shape SLat flows (both upstream-exact when nil/default;
    /// verdicts in BENCHMARKS.md before changing either in production):
    /// `slatCfgInterval` — "lo,hi" guidance interval override (upstream "0.6,1.0"). Narrowing
    /// trims double-forward steps at the schedule edges.
    /// `slatCfgNegEvery` — CFG cache stride: recompute vNeg only every N-th CFG step, reuse
    /// between (fresh vPos + CFG combination every step). 1/nil = off. N=2 ≈ −19% flow forwards.
    public var slatCfgInterval: String?
    public var slatCfgNegEvery: Int?

    /// Cascade HR token cap override. nil (default) = budget-derived: measured GPU peak fits
    /// ≈ 28 GB + 1.55 MB/token (resolution-independent), so the pipeline caps tokens to keep
    /// predicted peak under 75 % of physical RAM − 5 GB and lets the resolution back-off absorb
    /// it (a 43k-token asset hit 96.6 % of this basis under the upstream 49,152 cap alone).
    /// Set explicitly when the engine/app knows the true available budget.
    public var maxHRTokens: Int?

    /// SDPA precision for the CFG-free tex SLat flow. nil (default) = "bf16": the tex flow
    /// runs guidance 1.0 (single forward, no CFG cancellation), and the controlled A/B
    /// (2026-07-20, seeded, mode-split visual review) showed bf16 tex is output-identical
    /// on the shape path (bit-equal decoded voxels) and visually indistinguishable, at
    /// ~2.5× on the stage (−27% res1024 e2e). Set "fp32" to restore the old behavior.
    public var texAttention: String?

    /// Which stages `additionalViews` condition: "all" (nil, default — SS, shape and tex, the
    /// oracle's multi-image get_cond) or "texture" (SS and shape see the primary image only; the
    /// tex flow sees every view). The views carry no pose, so under "all" a feature from one view
    /// can land on another side of the object (skirt closure copied front and back, AB-R-0722);
    /// "texture" keeps their colour evidence without letting them steer geometry.
    public var additionalViewsScope: String?

    /// Emit the GLB Y-up (glTF/VRM convention) instead of TRELLIS's native Z-up: bakes the
    /// (x,y,z)→(x,z,−y) rotation into the vertices/normals before encode. REQUIRED for the
    /// rig→USDZ→RealityKit path (a post-rig reorientation mis-composes with clip playback).
    /// Default false (Z-up) — existing Z-up consumers apply their own reorientation.
    public var yUpOutput: Bool

    /// Optional app-injected foreground matting for RAW (unmasked) inputs — T0.4. When set, `run()`
    /// mattes every view that lacks a usable alpha (opaque RGB/JPEG, or an RGBA whose alpha is all
    /// 255 — upstream `preprocess_image`'s `has_alpha` test) BEFORE DINOv3 conditioning, so a raw
    /// phone photo gets background-removed instead of the full-frame-resize fallback. Typed on the
    /// canonical MLXToolKit artifacts only (`Image` → `Matte`), so Trellis2Kit carries no
    /// cross-package dependency (C13); the app composes the shipped BiRefNet `.matting` capability
    /// here, e.g. `{ try await engine.run(MattingRequest(image: $0), package: birefnetID) … }`.
    /// nil (default) = prior behavior: honor an existing alpha, else full-frame resize.
    /// NOT part of the Codable surface (in-process wiring, not persisted configuration).
    public var matting: Trellis2Matting? = nil

    /// Pixal3D (AB-T-0485): where the converted `pixal3d_*.safetensors` + `naf.safetensors` live.
    /// nil = the consolidated snapshot directory. Loaded lazily on the first Pixal3D request.
    /// (Publishing the Pixal3D weights and their lanes is P3; until then this points at a local dir.)
    public var pixal3dWeightsDir: URL?
    /// Pixal3D-only registration (P2 measurement shape; the real lanes are P3): skip the stock SS
    /// and SLat flows, preload the single-view Pixal3D set at prepare, and default every request to
    /// `pixal3d: true`. nil/false = a stock registration (Pixal3D still available per request, lazily).
    public var pixal3dOnly: Bool?
    /// With `pixal3dOnly`: preload the posed multi-view (`_mv`) set instead of the single-view one.
    public var pixal3dMultiView: Bool?
    /// Flow weights resident in their stored bf16 instead of fp32 (`Trellis2Pipeline.flowWeightsBF16`):
    /// byte-identical output (exact promotion in each matmul), 11 GB less resident, 8–15 % slower. Opt-in.
    /// A Pixal3D lane with it is charged `Trellis2Footprint.pixal3dResidentCascadeBF16` (21 GB); stock lanes
    /// keep their fp32 charge.
    public var flowWeightsBF16: Bool?

    /// Explicit Codable surface: everything except the `matting` closure (which decodes to nil).
    enum CodingKeys: String, CodingKey {
        case quant, defaultMode, tierCeiling, maxAtlasSize, modelsRootDirectory, weightsRootOverride
        case steps, seed, texture, decimateFaces, yUpOutput, unwrapBackend, metricsPath
        case slatCfgAttention, texAttention, slatCfgInterval, slatCfgNegEvery, maxHRTokens
        case additionalViewsScope, pixal3dWeightsDir, pixal3dOnly, pixal3dMultiView, flowWeightsBF16
    }

    public init(quant: Quant = .bf16,
                defaultMode: Mode = ImageTo3DContract.res512,
                tierCeiling: Mode? = nil,
                maxAtlasSize: Int = 1024,
                modelsRootDirectory: URL? = nil,
                weightsRootOverride: URL? = nil,
                steps: Int = 12,
                seed: UInt64 = 0,
                texture: Bool = true,
                decimateFaces: Int? = 300_000,
                unwrapBackend: String? = nil,
                yUpOutput: Bool = false,
                metricsPath: String? = nil,
                slatCfgAttention: String? = nil,
                texAttention: String? = nil,
                slatCfgInterval: String? = nil,
                slatCfgNegEvery: Int? = nil,
                maxHRTokens: Int? = nil,
                additionalViewsScope: String? = nil,
                matting: Trellis2Matting? = nil,
                pixal3dWeightsDir: URL? = nil,
                pixal3dOnly: Bool? = nil,
                pixal3dMultiView: Bool? = nil,
                flowWeightsBF16: Bool? = nil) {
        self.quant = quant
        self.defaultMode = defaultMode
        self.tierCeiling = tierCeiling
        self.maxAtlasSize = maxAtlasSize
        self.modelsRootDirectory = modelsRootDirectory
        self.weightsRootOverride = weightsRootOverride
        self.steps = steps
        self.seed = seed
        self.texture = texture
        self.decimateFaces = decimateFaces
        self.unwrapBackend = unwrapBackend
        self.yUpOutput = yUpOutput
        self.metricsPath = metricsPath
        self.slatCfgAttention = slatCfgAttention
        self.texAttention = texAttention
        self.slatCfgInterval = slatCfgInterval
        self.slatCfgNegEvery = slatCfgNegEvery
        self.maxHRTokens = maxHRTokens
        self.additionalViewsScope = additionalViewsScope
        self.matting = matting
        self.pixal3dWeightsDir = pixal3dWeightsDir
        self.pixal3dOnly = pixal3dOnly
        self.pixal3dMultiView = pixal3dMultiView
        self.flowWeightsBF16 = flowWeightsBF16
    }
}

/// The injected foreground-matting hook: canonical `Image` in, canonical `Matte` out (soft alpha at
/// the input image's dimensions, per `MattingContract`). Async so the app can route it through
/// `MLXServeEngine.run(.matting)`; throws propagate out of `Trellis2Package.run()` (a broken matter
/// is a failed generation, not a silent full-frame fallback).
public typealias Trellis2Matting = @Sendable (Image) async throws -> Matte

// MARK: - Per-tier footprints (contract 1.13/1.14 FootprintConfigured; AB-A-0262)

extension Trellis2Configuration {
    /// A lane pinned to `tier` (`defaultMode` and `tierCeiling` both set): the shape a host that
    /// charges per lane registers, one per tier. Every other knob keeps its default.
    /// `maxAtlasSize` bounds the per-request atlas; on res512 it sets the charge (34 / 38 / 48 GB at
    /// 1024 / 2048 / 4096), on the cascade lanes it does not move it.
    public static func lane(_ tier: Mode, maxAtlasSize: Int = 1024, modelsRootDirectory: URL? = nil,
                            weightsRootOverride: URL? = nil) -> Trellis2Configuration {
        Trellis2Configuration(defaultMode: tier, tierCeiling: tier, maxAtlasSize: maxAtlasSize,
                              modelsRootDirectory: modelsRootDirectory, weightsRootOverride: weightsRootOverride)
    }

    /// A Pixal3D lane (AB-T-0485 P3): `pixal3dOnly`, pinned to a cascade `tier` (res1024 or res1536 —
    /// res512 has no Pixal3D path), charged Pixal3D's footprint (`Trellis2Footprint.pixal3d…`). Every
    /// request on it runs Pixal3D; the stock flows are never loaded. Weights: the stock snapshot
    /// (decoders, DINOv3) + `xocialize/pixal3d-mlx`.
    public static func pixal3dLane(_ tier: Mode, maxAtlasSize: Int = 1024, modelsRootDirectory: URL? = nil,
                                   weightsRootOverride: URL? = nil, pixal3dWeightsDir: URL? = nil) -> Trellis2Configuration {
        Trellis2Configuration(defaultMode: tier, tierCeiling: tier, maxAtlasSize: maxAtlasSize,
                              modelsRootDirectory: modelsRootDirectory, weightsRootOverride: weightsRootOverride,
                              pixal3dWeightsDir: pixal3dWeightsDir, pixal3dOnly: true)
    }

    /// True for a Pixal3D-only registration.
    public var isPixal3DLane: Bool { pixal3dOnly ?? false }

    /// The tier the footprint is declared for: the lane's ceiling, else the default mode.
    public var chargedTier: Trellis2Tier { Trellis2Tier(mode: tierCeiling ?? defaultMode) }

    /// The HR token bound the charged tier runs under: an explicit `maxHRTokens`, else the tier's
    /// `hrTokenCeiling` (nil for res512, which has no HR pass). The pipeline enforces it as a hard
    /// cap, so the footprint below is a bound.
    public var hrTokenBound: Int? {
        guard chargedTier.isCascade else { return nil }
        return maxHRTokens.map { min(49_152, $0) } ?? chargedTier.hrTokenCeiling
    }
}

extension Trellis2Tier {
    /// The tier a request or configuration `Mode` names (anything unrecognised is res512, as in run()).
    public init(mode: Mode) {
        switch mode {
        case ImageTo3DContract.res1536: self = .res1536
        case ImageTo3DContract.res1024: self = .res1024
        default: self = .res512
        }
    }
    /// Order for the lane check: res512 < res1024 < res1536.
    var rank: Int { switch self { case .res512: 0; case .res1024: 1; case .res1536: 2 } }
}

/// Measured per-tier footprints (AB-A-0262): Release, M5 Max 128 GB, mlx-engine-swift 0.74.1 +
/// xocialize/mlx-swift 0.32.4, BiRefNet matting registered in the same process, phys_footprint
/// sampled every 200 ms through `trellis2-run-engine` (one fresh process per run). README
/// "Footprint per tier" has the table and the receipt.
///
///   tier     asset   HR tokens (res)    peak phys   post-run resident phys
///   res512   dress    4,856 (512)        30.54 GB    24.78 GB
///   res1024  dress   22,644 (1024)       57.50 GB    30.83 GB
///   res1024  hair    25,977 (1024)       60.74 GB    30.85 GB
///   res1536  shoes   21,189 (1536)       59.62 GB    30.77 GB
///   res1536  dress   29,437 (1152)       65.23 GB    30.73 GB
///
/// Resident is phys after a run with MLX's cache cleared (the tier's flows cached, BiRefNet's
/// ~1 GB included, which keeps it on the safe side). The cascade activation is a line in HR tokens
/// through the measured cascade peaks plus ~7 % headroom, charged at the tier's token bound.
public enum Trellis2Footprint {
    /// res512 resident: SS DiT + SS decoder + shape/tex decoders + DINOv3 + shape_512 + tex_512.
    public static let residentRes512: UInt64 = 25_000_000_000
    /// Cascade resident (res1024 and res1536 alike): tex_512 swapped for shape_1024 + tex_1024 —
    /// generate() keeps only the tier's flows cached.
    public static let residentCascade: UInt64 = 31_000_000_000
    /// res512 activation over resident at a 1024 atlas: 30.54 − 24.78 = 5.8 GB measured on the
    /// largest shape SLat in the corpus; declared 9 so the tier totals 34 GB.
    public static let activationRes512: UInt64 = 9_000_000_000

    /// res512 activation for a registration that accepts atlases up to `maxAtlasSize` (AB-A-0263).
    /// On res512 the 2×-supersampled bake is the peak once the atlas grows (dress, pbrMaps on, peak
    /// phys): 2048 → 34.73 GB, 4096 → 43.59 GB; declared 13 / 23 GB so the lane totals 38 / 48 GB.
    /// The cascade lanes' HR stage stays above the bake (res1536 dress at 4096: 65.23 GB, the same
    /// peak as at 1024), so their activation does not depend on the atlas.
    public static func activationRes512(maxAtlasSize: Int) -> UInt64 {
        switch maxAtlasSize {
        case ...1024: activationRes512
        case ...2048: 13_000_000_000
        default: 23_000_000_000
        }
    }
    /// Cascade activation over resident, linear in HR tokens up to `measuredTokenCeiling`:
    /// 38 GB at 28,672 tokens (res1024's bound) and 41 GB at 32,768 (res1536's).
    public static let cascadeActivationBase: UInt64 = 17_000_000_000
    public static let activationPerHRToken: Double = 3_000_000_000 / 4_096
    /// Past the largest token bound that was measured toward, an explicit `maxHRTokens` is charged
    /// the pre-0.73-engine slope (1.55 MB/token, BENCHMARKS.md) — the conservative one.
    public static let measuredTokenCeiling = 32_768
    public static let activationPerHRTokenBeyond: Double = 1_550_000

    // MARK: Pixal3D lanes (AB-T-0485 P3) — measured 2026-11-14, README "Pixal3D mode"
    // Release, M5 Max 128 GB, mlx-engine-swift 0.74.1, `pixal3dOnly` lane through trellis2-run-engine
    // (phys_footprint every 200 ms, one fresh process per run, matted inputs, no BiRefNet):
    //   tier     asset   HR tokens (res)    peak phys   post-run resident
    //   res1024  dress    8,500 (1024)       41.63 GB    31.21 GB
    //   res1024  hair    10,729 (1024)       43.81 GB    30.98 GB
    //   res1024  skirt   12,742 (1024)       43.37 GB    31.26 GB
    //   res1024  shoes   15,242 (1024)       51.58 GB    31.11 GB
    //   res1024  top      8,851 (1024)       41.02 GB    31.33 GB
    //   res1536  dress   20,289 (1536)       56.36 GB    31.69 GB
    //   res1536  hair    28,243 (1536)       66.90 GB    31.42 GB
    //   res1536  skirt   29,804 (1536)       62.28 GB    31.34 GB
    //   res1536  shoes   26,167 (1280)       65.14 GB    31.62 GB   (token back-off)
    //   res1536  top     20,712 (1536)       56.54 GB    31.75 GB
    // Activation over resident is bounded by 4 GB + 1.2 MB/token (every point ≥ 5.6 % under it).
    /// Pixal3D cascade resident: Pixal3D SS + shape_512 + shape_1024 + tex_1024 (fp32-cast) + NAF + the
    /// stock decoders + DINOv3.
    public static let pixal3dResidentCascade: UInt64 = 32_000_000_000
    /// Pixal3D cascade resident with `flowWeightsBF16` (v0.16.0): the four flows stay bf16. Measured
    /// 20.28 / 20.00 / 20.36 GB post-run (res1024 dress, res1024 hair, res1536 hair) against 31.2–31.4 fp32,
    /// with byte-identical GLBs and the same activation (peaks 30.74 / 32.42 / 55.91 GB); 8–15 % slower.
    /// Only the Pixal3D lanes are re-charged; a stock lane with bf16 flows keeps its fp32 charge (unmeasured).
    public static let pixal3dResidentCascadeBF16: UInt64 = 21_000_000_000
    /// Pixal3D cascade activation over resident: base + slope × HR tokens (fit through the measured
    /// peaks plus headroom), charged at the tier's token bound; past `measuredTokenCeiling` the
    /// conservative 1.55 MB/token.
    public static let pixal3dCascadeActivationBase: UInt64 = 4_000_000_000
    public static let pixal3dActivationPerHRToken: Double = 1_200_000
    public static func pixal3dCascadeActivation(tokens: Int) -> UInt64 {
        let inRange = min(tokens, measuredTokenCeiling), beyond = max(0, tokens - measuredTokenCeiling)
        let bytes = Double(pixal3dCascadeActivationBase) + pixal3dActivationPerHRToken * Double(inRange)
            + activationPerHRTokenBeyond * Double(beyond)
        return UInt64(bytes.rounded(.up))
    }

    /// The manifest's footprint: the larger res1536 lane, stock or Pixal3D.
    public static var worstLane: QuantFootprint {
        let t = Trellis2Tier.res1536.hrTokenCeiling ?? 49_152
        let stock = (residentCascade, cascadeActivation(tokens: t))
        let pix = (pixal3dResidentCascade, pixal3dCascadeActivation(tokens: t))
        let w = stock.0 + stock.1 >= pix.0 + pix.1 ? stock : pix
        return QuantFootprint(quant: .bf16, residentBytes: w.0, peakActivationBytes: w.1)
    }

    /// Activation for a cascade run of at most `tokens` HR tokens.
    public static func cascadeActivation(tokens: Int) -> UInt64 {
        let inRange = min(tokens, measuredTokenCeiling), beyond = max(0, tokens - measuredTokenCeiling)
        let bytes = Double(cascadeActivationBase) + activationPerHRToken * Double(inRange)
            + activationPerHRTokenBeyond * Double(beyond)
        return UInt64(bytes.rounded(.up))
    }
}

extension Trellis2Configuration: FootprintConfigured {
    /// Per-tier charging, keyed on `chargedTier` (the lane's `tierCeiling`, else `defaultMode`).
    /// Without a `tierCeiling` a request above the configured tier is not re-charged; with one it is
    /// refused, so a lane's charge holds. Numbers: `Trellis2Footprint`.
    public var residentBytesHint: UInt64? {
        if isPixal3DLane && chargedTier.isCascade {
            return (flowWeightsBF16 ?? false) ? Trellis2Footprint.pixal3dResidentCascadeBF16 : Trellis2Footprint.pixal3dResidentCascade
        }
        return chargedTier.isCascade ? Trellis2Footprint.residentCascade : Trellis2Footprint.residentRes512
    }
    public var peakActivationBytesHint: UInt64? {
        if isPixal3DLane, let tokens = hrTokenBound { return Trellis2Footprint.pixal3dCascadeActivation(tokens: tokens) }
        guard let tokens = hrTokenBound else { return Trellis2Footprint.activationRes512(maxAtlasSize: maxAtlasSize) }
        return Trellis2Footprint.cascadeActivation(tokens: tokens)
    }
}

// MARK: - Weight sources (auto-materialization, engine ≥0.19.0 MAT gate)

extension Trellis2Configuration: WeightSourcing {
    /// The consolidated, remap-free weights repo (one download): every component in the Swift
    /// module-key layout (incl. DINOv3, redistributed under the DINOv3 License §1.b) +
    /// `normalization.json` + the license files. The repo is UNGATED (token-less first-run
    /// materialization) — anti-mirror is by card/license framing (pipeline artifact, not a DINOv3
    /// mirror; standalone-DINOv3 → Meta's gated repo; "Built with DINOv3").
    public static let consolidatedRepo = "xocialize/trellis2-mlx"

    /// Every weight/normalization file of the published snapshot the probe requires. A
    /// half-finished download must re-report missing rather than silently degrade (an absent tex
    /// pair would drop textures; an absent cascade DiT would silently cap res1024/res1536 at 512).
    /// See the SW6 report for exactly which upstream file each name is assembled from.
    public static let probeFiles = [
        "normalization.json",
        "dino.safetensors",
        "struct_flow.safetensors",
        "struct_dec.safetensors",
        "shape_flow_512.safetensors",
        "shape_flow_1024.safetensors",
        "shape_dec.safetensors",
        "tex_flow_512.safetensors",
        "tex_flow_1024.safetensors",
        "tex_dec.safetensors",
    ]

    /// A canonical ORIGINAL-upstream tensor key. The snapshot must carry it; the OLD broken port
    /// shipped renamed keys (`adaLN.weight`/`b0.`) under the SAME filenames, which a filename-only
    /// probe served as "present" → the 2026-07-14 nil-unwrap crash. `snapshotPresent` reads the
    /// struct_flow safetensors header and re-reports the snapshot missing (→ re-materialize) if this
    /// key is absent, so a content swap under a stable filename can't silently degrade.
    public static let canonicalKey = "adaLN_modulation.1.weight"

    /// ONE consolidated snapshot source. The upstream 3-repo chain stays a load()-time fallback for
    /// pre-staged dev machines and is deliberately NOT declared (a fresh machine materializes the
    /// consolidated repo; DINOv3's upstream repo is Meta-gated anyway).
    public var weightSources: [WeightSource] {
        [WeightSource(role: "snapshot", repo: Self.consolidatedRepo,
                      matching: ["*.safetensors", "*.json", "LICENSE", "NOTICE", "DINOv3_LICENSE.md"])]
            + (isPixal3DLane ? [Self.pixal3dSource] : [])
    }

    // Pixal3D weights (AB-T-0485 P3): a second snapshot, declared only by a Pixal3D lane.
    public static let pixal3dRepo = "xocialize/pixal3d-mlx"
    static let pixal3dSource = WeightSource(role: "pixal3d", repo: pixal3dRepo,
                                            matching: ["*.safetensors", "*.json", "LICENSE", "NOTICE", "NAF_LICENSE"])
    /// The single-view set the repo publishes. The posed multi-view `_mv` set is not published: a
    /// multi-view lane needs it in `pixal3dWeightsDir` (`Tools/pixal3d/convert_weights.py --mv`).
    public static func pixal3dProbeFiles(multiView: Bool) -> [String] {
        let sfx = multiView ? "_mv" : ""
        return ["pixal3d_struct_flow", "pixal3d_shape_flow_512", "pixal3d_shape_flow_1024", "pixal3d_tex_flow_1024"]
            .map { "\($0)\(sfx).safetensors" } + ["naf.safetensors"]
    }
    /// Content guard: a Pixal3D flow carries `proj_linear` (a stock flow under the same name would not).
    public static let pixal3dCanonicalKey = "blocks.0.cross_attn.proj_linear.weight"

    static func pixal3dPresent(at dir: URL, multiView: Bool) -> Bool {
        let fm = FileManager.default
        guard pixal3dProbeFiles(multiView: multiView).allSatisfy({ fm.fileExists(atPath: dir.appending(path: $0).path) })
        else { return false }
        return safetensorsHeaderContains(dir.appending(path: "pixal3d_struct_flow\(multiView ? "_mv" : "").safetensors"),
                                         key: pixal3dCanonicalKey)
    }

    /// Where the Pixal3D weights are: `pixal3dWeightsDir` (flat or `<dir>/xocialize/pixal3d-mlx`), the
    /// ModelStore copy of `xocialize/pixal3d-mlx`, then the stock snapshot dir (a local all-in-one
    /// folder). nil when none holds the set this configuration runs.
    public func pixal3dSnapshotDir(storeRoot: URL?, stockDir: URL? = nil) -> URL? {
        let mv = pixal3dMultiView ?? false
        var candidates = [URL]()
        if let d = pixal3dWeightsDir { candidates += [d, d.appending(path: Self.pixal3dRepo)] }
        if let d = ModelStore(root: storeRoot).directory(for: Self.pixal3dRepo) { candidates.append(d) }
        if let d = stockDir { candidates.append(d) }
        return candidates.first { Self.pixal3dPresent(at: $0, multiView: mv) }
    }

    /// Explicit `weightsRootOverride` first (dev escape hatch — flat `<dir>/<file>` or
    /// `<dir>/<org>/<name>/<file>`), then the ModelStore layout (`<root>/xocialize/trellis2-mlx`).
    /// nil store + no override ⇒ everything missing (fresh-machine posture, MAT-4).
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let stock = missingStockSources(storeRoot: storeRoot)
        guard isPixal3DLane else { return stock }
        let stockDir = weightsRootOverride.flatMap { d in
            Self.snapshotPresent(at: d) ? d : (Self.snapshotPresent(at: d.appending(path: Self.consolidatedRepo))
                                               ? d.appending(path: Self.consolidatedRepo) : nil)
        }
        return stock + (pixal3dSnapshotDir(storeRoot: storeRoot, stockDir: stockDir) == nil ? [Self.pixal3dSource] : [])
    }

    func missingStockSources(storeRoot: URL?) -> [WeightSource] {
        if let dir = weightsRootOverride,
           Self.snapshotPresent(at: dir)
            || Self.snapshotPresent(at: dir.appending(path: Self.consolidatedRepo)) {
            return []
        }
        if let dir = ModelStore(root: storeRoot).directory(for: Self.consolidatedRepo),
           Self.snapshotPresent(at: dir) {
            return []
        }
        return [weightSources[0]]
    }

    static func snapshotPresent(at dir: URL) -> Bool {
        let fm = FileManager.default
        guard probeFiles.allSatisfy({ fm.fileExists(atPath: dir.appending(path: $0).path) }) else { return false }
        // Content guard (not filename-only): the struct_flow snapshot must carry the ORIGINAL keys.
        return safetensorsHeaderContains(dir.appending(path: "struct_flow.safetensors"), key: canonicalKey)
    }

    /// Cheap content probe: a safetensors file starts with an 8-byte little-endian header length, then
    /// that many bytes of JSON whose keys are the tensor names. Read only the header (≪ the multi-GB
    /// file) and substring-match `key`. Returns false on any read error (⇒ treat as missing/re-fetch).
    static func safetensorsHeaderContains(_ url: URL, key: String) -> Bool {
        guard let fh = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? fh.close() }
        guard let lenBytes = try? fh.read(upToCount: 8), lenBytes.count == 8 else { return false }
        let n = lenBytes.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard n > 0, n < 100_000_000, let header = try? fh.read(upToCount: Int(n)) else { return false }
        return header.range(of: Data(key.utf8)) != nil
    }
}

// MARK: - Cold-start prewarm

extension Trellis2Configuration: WeightPrewarming {
    /// The files the charged tier loads, not the whole snapshot (AB-A-0263 follow-up). The folder is
    /// ~35 GB — encoders and the other tier's flows included — and paging all of it competed for the
    /// volume with the first job's own weight reads. res512: ~the 512 set; cascade: the 1024 set.
    public var prewarmPaths: [URL] {
        let root = weightsRootOverride
            ?? ModelStore(root: modelsRootDirectory).directory(for: Self.consolidatedRepo)
        guard let root else { return [] }
        // An override may hold the snapshot flat or under <org>/<name>, as load() resolves it.
        let dir = Self.snapshotPresent(at: root) ? root : root.appending(path: Self.consolidatedRepo)
        if isPixal3DLane {
            // the Pixal3D lane loads the stock decoders + DINOv3 and its own flows; never the stock flows
            let stock = ["normalization.json"] + ["struct_dec", "shape_dec", "tex_dec", "dino"].map { "\($0).safetensors" }
            var paths = stock.map { dir.appending(path: $0) }
            if let pdir = pixal3dSnapshotDir(storeRoot: modelsRootDirectory, stockDir: dir) {
                let sfx = (pixal3dMultiView ?? false) ? "_mv" : ""
                let flows = ["pixal3d_struct_flow", "pixal3d_shape_flow_512", "pixal3d_shape_flow_1024", "pixal3d_tex_flow_1024"]
                paths += flows.map { pdir.appending(path: "\($0)\(sfx).safetensors") } + [pdir.appending(path: "naf.safetensors")]
            }
            return paths
        }
        let flows = chargedTier.isCascade ? ["shape_flow_512", "shape_flow_1024", "tex_flow_1024"]
                                          : ["shape_flow_512", "tex_flow_512"]
        let files = ["normalization.json"]
            + (["struct_flow", "struct_dec", "shape_dec", "tex_dec", "dino"] + flows).map { "\($0).safetensors" }
        return files.map { dir.appending(path: $0) }
    }
}

/// Errors specific to the TRELLIS.2 runtime.
public enum Trellis2Error: Error, Sendable {
    case imageDecodeFailed
    case matteDecodeFailed
    case weightsNotFound(String)
    case normalizationMissing
}

// MARK: - ModelPackage conformer

/// MLXToolKit `ModelPackage` conformer for TRELLIS.2 image→3D (UV-textured GLB output).
///
/// C7 weight license: the DINOv3 conditioner license is non-SPDX but functionally permissive and
/// allowlisted as `SPDXLicense.dinov3`, so the default `.permissiveOnly` gate ADMITS the package.
/// TRELLIS.2 itself is MIT; the most-restrictive component (DINOv3) governs, hence
/// `weightLicense = .dinov3`. **Product obligation:** display "Built with DINOv3" wherever shipped.
///
/// The neural core (`Trellis2Pipeline`) is the parity-verified Stage-1 port; this conformer is the
/// thin declaration + dispatch layer (manifest, license gate, weight resolution, load()/run()).
@InferenceActor
public final class Trellis2Package: ModelPackage {
    public typealias Configuration = Trellis2Configuration

    public nonisolated static var manifest: PackageManifest {
        makeManifest(sourceRepo: "microsoft/TRELLIS.2-4B", footprint: Trellis2Footprint.worstLane,
                     surface: ImageTo3DContract.descriptor(
                        name: "trellis2-image-to-3d",
                        summary: "Generate a UV-textured 3D triangle mesh (GLB) from a single image. Invents "
                               + "geometry + a base-color texture atlas from one view; pick detail via mode "
                               + "res512/res1024/res1536.",
                        modes: [ImageTo3DContract.res512, ImageTo3DContract.res1024, ImageTo3DContract.res1536]))
    }

    /// The manifest shape shared by the stock package and `Pixal3DPackage` (licence, backends, OS, chip,
    /// specialty); each passes its own provenance, worst-lane footprint and surface.
    nonisolated static func makeManifest(sourceRepo: String, footprint: QuantFootprint,
                                         surface: ToolDescriptor) -> PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(
                weightLicense: .dinov3,    // C7 — DINOv3 governs; allowlisted. "Built with DINOv3".
                portCodeLicense: .mit),    // C8 — our Swift port
            provenance: Provenance(sourceRepo: sourceRepo, revision: "main", tier: 3),
            requirements: RequirementsManifest(
                // Split footprint (contract 1.14), T0.3 per-tier re-baseline (2026-07-15, engine-governed
                // trellis2-run-engine phys_footprint peaks, single view). The QuantFootprint is the
                // WORST-mode (res1536) declaration — per-tier charging comes from the FootprintConfigured
                // hints below (keyed on the configured defaultMode), the BiRefNet same-quant pattern.
                //   residentBytes = 23 GB — the cascade working set, all fp32-cast (parity dtype), no
                //     per-stage eviction: SS DiT + shape_512(LR) + shape_1024 + tex_1024 (4×~5.2 GB) +
                //     DINOv3 (~1.2) + decoders (~0.5) ≈ 22.5; generate() prunes cached flows to the
                //     tier's set, so res512-only sessions never hold the 1024 pair (see hints).
                //   peakActivationBytes = 56 GB — res1536 MEASURED peak phys 76.97 GB (1850 s run,
                //     token back-off active) − ~22.5 resident ⇒ ~54.5 transient; declare 56. res1024
                //     cascade measured 46.76 GB peak phys ⇒ ~24 transient (hint declares 26). The
                //     transient is dominated by the HR shape/tex decode (7.8M voxels @1024, ~2.2× @1536)
                //     + the LR full-decode upsample (1.7M voxels) + O(T²) SLat self-attention (≤49,152
                //     HR tokens). Bare-CLI (ungoverned buffer pool) ratchets to ~112 GB — the engine's
                //     bounded pool is the admission-basis environment.
                // AB-A-0262 (v0.11.0): the res1536 numbers are now its lane's — resident cascade + the
                // activation at the tier's hard HR token ceiling (32,768) — from `Trellis2Footprint`.
                // A host's largest-footprint fallback is therefore still the worst tier.
                // P3 (AB-T-0485): the Pixal3D res1536 lane is a backer of the same capability; the manifest
                // carries whichever res1536 lane is larger, so the largest-footprint fallback still holds.
                footprints: [footprint],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .pro),
            specialties: [SpecialtyWeight(Specialty(rawValue: "3d-generation"), strength: 1.0)],
            surfaces: [surface])
    }

    private let configuration: Configuration
    private var pipeline: Trellis2Pipeline?          // host-owned lifecycle; no global singleton (C13)
    private var shapeMean: [Float] = []
    private var shapeStd: [Float] = []
    private var texMean: [Float] = []
    private var texStd: [Float] = []

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    /// Harness convenience mirroring the other `-swift` packages: the engine-side registration value.
    public nonisolated static var registration: PackageRegistration { .of(Trellis2Package.self) }

    public static let consolidatedRepo = Trellis2Configuration.consolidatedRepo

    // MARK: load

    public func load() async throws {
        guard pipeline == nil else { return }   // idempotent (C13)
        let root = configuration.modelsRootDirectory
        let override = configuration.weightsRootOverride

        // First-run auto-materialization (MAT gate): a dir-less config with an engine-stamped store
        // root downloads the consolidated snapshot into the ModelStore layout, progress forwarded via
        // WeightDownloadProgress so the engine surfaces `.downloading`. The explicit override never
        // touches the network.
        if override == nil, let root {
            let missing = configuration.missingWeightSources(storeRoot: root)
            if !missing.isEmpty {
                try await WeightMaterializer.materialize(missing, into: root)
            }
        }

        let dir = try Self.resolveSnapshotDir(root: root, override: override)
        // The consolidated snapshot is a flat directory of module-key safetensors — the pipeline's
        // ckptDir loader expects `<name>.safetensors`, which the consolidated names satisfy directly.
        // struct_dec / shape_flow_512 / tex_flow_512 / etc. map onto the pipeline's expected files.
        let ckptDir = dir.path
        let pixal3dOnly = configuration.pixal3dOnly ?? false
        pipeline = try Trellis2Pipeline(consolidatedDir: ckptDir, loadStockStructFlow: !pixal3dOnly)
        pipeline?.flowWeightsBF16 = configuration.flowWeightsBF16 ?? false
        // Pixal3D weights: pixal3dWeightsDir, the ModelStore copy of xocialize/pixal3d-mlx, else the stock dir
        pipeline?.pixal3dDir = configuration.pixal3dSnapshotDir(storeRoot: root, stockDir: dir)?.path
            ?? configuration.pixal3dWeightsDir?.path
        if pixal3dOnly {
            guard configuration.chargedTier.isCascade else {
                throw PackageError.unsupportedRequestFeature(
                    "pixal3dOnly on \(configuration.chargedTier.rawValue): Pixal3D runs on res1024 / res1536 only (AB-T-0485 P3)")
            }
            try pipeline?.preloadPixal3D(multiView: configuration.pixal3dMultiView ?? false, tier: configuration.chargedTier)
        }
        // A lane knows its tier: load its flows now (prepare), so a job reads nothing from disk and its
        // resident set is the declared one from the start. Without a tierCeiling they stay lazy.
        if configuration.tierCeiling != nil && !pixal3dOnly {
            try pipeline?.preloadFlows(for: configuration.chargedTier)
        }

        let norm = try Self.loadNormalization(dir)
        shapeMean = norm.shapeMean; shapeStd = norm.shapeStd
        texMean = norm.texMean; texStd = norm.texStd
    }

    public func unload() async {
        pipeline = nil; shapeMean = []; shapeStd = []; texMean = []; texStd = []
        MLX.Memory.clearCache()   // release pooled weight/activation buffers so phys_footprint falls (C13/eff)
    }

    // MARK: run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN gate: the entry checkpoint is the FIRST act of run() — before capability validation and
        // the notLoaded guard — so a pre-cancelled run refuses before weights.
        try Task.checkCancellation()
        guard request.capability == .imageTo3D, let req = request as? ImageTo3DRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        // Resolution tier (voxel grid): res512 = direct 512 flows; res1024/res1536 = the T0.3
        // cascade (LR 512 pass → decoder-upsample re-quantization → HR 1024 flows, token
        // back-off floor 1024).
        let mode = req.mode ?? configuration.defaultMode
        let tier = Trellis2Tier(mode: mode)
        // A lane (`tierCeiling`) is charged for its own tier; a higher one would out-spend that charge.
        if let ceiling = configuration.tierCeiling, tier.rank > Trellis2Tier(mode: ceiling).rank {
            throw PackageError.unsupportedRequestFeature(
                "mode \(mode.rawValue) is above this registration's tier \(ceiling.rawValue) (tierCeiling); "
                    + "send it to the \(mode.rawValue) lane")
        }

        // Per-request overrides (AB-A-0263): metaData keys over the configuration; malformed → refused here.
        let options = try Trellis2RequestOptions(metaData: req.metaData, configuration: configuration)
        // Contract 1.64.0 (AB-A-0284): typed cameras / frame. The stock surface declares no camera controls,
        // so it refuses them (what the engine's pre-flight does); the Pixal3D path takes them.
        if !options.pixal3d, let why = ImageTo3DControls.refusal(for: req, declared: nil) {
            throw PackageError.unsupportedRequestFeature(why)
        }
        if options.pixal3d, let why = ImageTo3DControls.refusal(for: req, declared: Pixal3DPackage.controls) {
            throw PackageError.unsupportedRequestFeature(why)
        }
        // P3 res512 study (AB-T-0485): neither Pixal3D res512 texture passed its gate; res512 stays stock.
        if options.pixal3d && !tier.isCascade {
            throw PackageError.unsupportedRequestFeature(
                "pixal3d: mode \(tier.rawValue) has no Pixal3D path (upstream ships no 512 mode; the P3 res512 "
                + "study kept res512 stock); use res1024 or res1536")
        }

        guard let pipeline else { throw PackageError.notLoaded }

        // Preprocess each input to DINOv3's NCHW [1,3,512,512]. RAW (no-usable-alpha) views are
        // matted first through the app-injected `configuration.matting` hook (T0.4 — e.g. the shipped
        // BiRefNet `.matting` package composed via the engine); each view is matted ONCE, then
        // conditioned at every needed resolution. Without the hook, a raw view falls back to the
        // full-frame resize (background leaks into the mesh).
        // Multi-view: the front image plus any `additionalViews` (front/side/back turnaround of the SAME
        // subject) are each preprocessed and conditioned together — DINOv3 per view, tokens concatenated
        // (see encodeImages), matching the oracle's multi-image get_cond.
        // Dual-resolution conditioning (upstream cascade tiers): SS + the LR pass use cond_512;
        // the HR shape + tex passes use cond_1024. DINOv3 is resolution-agnostic (patch grid derived
        // from input dims), so 512² and 1024² inputs both work; each view is preprocessed at both
        // sizes and its tokens concatenated per size. res512 skips the 1024² pass entirely
        // (upstream cond_1024=None for the '512' pipeline type).
        var views = [req.image] + (req.additionalViews ?? [])
        let tMat = Date()
        if let matting = configuration.matting {
            for i in views.indices {
                views[i] = try await ImagePreprocess.mattedIfNeeded(views[i], using: matting)
                try Task.checkCancellation()   // the matter may itself be a model run
            }
        }
        let mattingS = configuration.matting != nil ? -tMat.timeIntervalSinceNow : 0
        if options.pixal3d {
            return try runPixal3D(pipeline: pipeline, views: views, tier: tier, options: options, mattingS: mattingS,
                                  typedCameras: req.cameras, typedFrame: req.frame)
        }
        // Profiling bookends (metricsPath set): preprocessing (CPU) is timed separately from the
        // DINOv3 encode (GPU) — each encode is eval-barriered so the delta is real wall-clock.
        let tPre = Date()
        let px512 = try views.map { try ImagePreprocess.dinoPixels(from: $0, size: 512) }
        let px1024: [MLXArray]? = tier.isCascade ? try views.map { try ImagePreprocess.dinoPixels(from: $0, size: 1024) } : nil
        let preprocessS = -tPre.timeIntervalSinceNow
        try Task.checkCancellation()

        // additionalViewsScope "texture": SS + shape condition on the primary view alone, and the
        // tex flow on every view (its tier's resolution: 1024 for the cascade, else 512).
        let textureOnlyViews = configuration.additionalViewsScope == "texture" && views.count > 1
        let shapeViews = textureOnlyViews ? 1 : views.count

        let tDino512 = Date()
        let (cond, negCond) = pipeline.encodeImages(Array(px512.prefix(shapeViews)))
        MLX.eval(cond, negCond)
        let dinoEncode512S = -tDino512.timeIntervalSinceNow
        let (cond1024, negCond1024): (MLXArray, MLXArray)
        var dinoEncode1024S = 0.0
        if let px1024 {
            let tDino1024 = Date()
            (cond1024, negCond1024) = pipeline.encodeImages(Array(px1024.prefix(shapeViews)))
            MLX.eval(cond1024, negCond1024)
            dinoEncode1024S = -tDino1024.timeIntervalSinceNow
        } else {
            (cond1024, negCond1024) = (cond, negCond)
        }
        var texConds: (cond: MLXArray, negCond: MLXArray)? = nil
        var dinoEncodeTexS = 0.0
        if textureOnlyViews {
            let tDinoTex = Date()
            let tc = pipeline.encodeImages(px1024 ?? px512)
            MLX.eval(tc.cond, tc.negCond)
            texConds = tc
            dinoEncodeTexS = -tDinoTex.timeIntervalSinceNow
        }
        try Task.checkCancellation()

        // In-package SS constants (no goldens dir): the deterministic 16³ RoPE grid + seeded noise.
        let ssPhases = SSGrid.ropePhases(resolution: 16)
        let ssNoise = SSGrid.noise(resolution: 16, seed: options.seed)

        // The pipeline's samplers/decoders honor cooperative cancellation at their step/layer seams
        // (they read the ambient Task); generate() is the long-running body.
        let (baked, _, stageMetrics) = try pipeline.generate(
            tier: tier,
            cond: cond, negCond: negCond, cond1024: cond1024, negCond1024: negCond1024,
            texCond: texConds?.cond, texNegCond: texConds?.negCond,
            ssNoise: ssNoise, ssPhases: ssPhases,
            shapeMean: MLXArray(shapeMean), shapeStd: MLXArray(shapeStd),
            texMean: MLXArray(texMean), texStd: MLXArray(texStd),
            texture: options.texture, targetFaces: options.decimateFaces,
            yUp: options.yUp, atlasSize: options.atlasSize, pbrMaps: options.pbrMaps,
            detailMaps: options.detailMaps,
            unwrapBackend: configuration.unwrapBackend == "provenance" ? .provenance : .xatlas,
            slatCfgSDPA: configuration.slatCfgAttention.flatMap { SDPAPrecision(rawValue: $0) } ?? .fp32,
            texSDPA: configuration.texAttention.flatMap { SDPAPrecision(rawValue: $0) } ?? .bf16,
            steps: options.steps,
            slatCfgInterval: configuration.slatCfgInterval.flatMap { s -> (Float, Float)? in
                let p = s.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
                return p.count == 2 && p[0] < p[1] ? (p[0], p[1]) : nil   // malformed -> upstream default
            },
            slatNegEvery: max(1, configuration.slatCfgNegEvery ?? 1),
            maxHRTokens: configuration.maxHRTokens,
            seed: options.seed, log: { _ in })
        try Task.checkCancellation()

        let tGlb = Date()
        let glb = try Self.glbData(baked, options: options)
        let glbExportS = -tGlb.timeIntervalSinceNow
        let mesh = Mesh(format: .glb, data: glb,
                        vertexCount: baked.vertices.dim(0), faceCount: baked.faces.dim(0),
                        hasVertexColors: false)   // UV-textured, not vertex colors

        // Full-workflow profile record (opt-in): generate()-scope stage timers + the DINO/GLB
        // bookends this method owns. One flat JSON line (sortedKeys) — unwrapbench/bakeab convention.
        if let path = configuration.metricsPath {
            var rec = stageMetrics.flatRecord()
            rec["matting_hook"] = configuration.matting != nil
            rec["matting_s"] = mattingS
            rec["preprocess_s"] = preprocessS
            rec["dino_encode_512_s"] = dinoEncode512S
            rec["dino_encode_1024_s"] = dinoEncode1024S
            rec["glb_export_s"] = glbExportS
            rec["views"] = views.count
            rec["views_scope"] = textureOnlyViews ? "texture" : "all"
            rec["dino_encode_tex_s"] = dinoEncodeTexS
            rec["mesh_verts"] = baked.vertices.dim(0)
            rec["mesh_faces"] = baked.faces.dim(0)
            rec["atlas_size"] = baked.atlasSize
            rec["pbr_maps"] = baked.mrRGBA != nil && options.pbrMaps
            if !options.detailMaps.isEmpty { Self.recordDetailMaps(&rec, baked) }
            rec["y_up"] = options.yUp
            rec["texture"] = options.texture
            rec["target_faces"] = options.decimateFaces
            rec["glb_bytes"] = glb.count
            rec["gpu_peak_bytes"] = Int(Memory.peakMemory)
            // SDPA kernel lane (M5/NAX): MLX_ENABLE_TF32 defaults ON, so "fp32" SDPA runs
            // the NAX kernel with TF32 (10-bit-mantissa) matmuls on gen>=17 GPUs unless
            // explicitly disabled. State the lane with every timing.
            rec["mlx_tf32_env"] = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "unset(=on)"
            rec["e2e_total_s"] = mattingS + preprocessS + dinoEncode512S + dinoEncode1024S + dinoEncodeTexS
                + stageMetrics.generateTotalS + glbExportS
            if let data = try? JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys]),
               let str = String(data: data, encoding: .utf8) {
                print("JSON \(str)")
                try? str.write(toFile: path, atomically: true, encoding: .utf8)
            }
        }
        return ImageTo3DResponse(mesh: mesh)
    }

    // MARK: Pixal3D (AB-T-0485)

    /// Pixal3D projection-conditioned generation. Single view: crop ×1.1 and a FOV; posed multi-view:
    /// one camera-to-world per view, no crop, the `_mv` weights. Cascade tiers only.
    private func runPixal3D(pipeline: Trellis2Pipeline, views: [Image], tier: Trellis2Tier,
                            options: Trellis2RequestOptions, mattingS: Double,
                            typedCameras: [ImageTo3DCamera]? = nil, typedFrame: ImageTo3DFrame? = nil) throws -> ImageTo3DResponse {
        // Typed contract fields (1.64.0) win over the v0.13.0 metaData keys (kept as the fallback).
        let posedTyped = typedCameras.map { cs in !cs.isEmpty && cs.allSatisfy { $0.cameraToWorld != nil } } ?? false
        let frameRaw = typedFrame?.rawValue ?? options.pixal3dFrame
        let posedMV = (posedTyped || options.pixal3dCameras != nil) && views.count > 1
        guard let pdir = pipeline.pixal3dDir.map({ URL(fileURLWithPath: $0) }),
              Trellis2Configuration.pixal3dPresent(at: pdir, multiView: posedMV) else {
            throw Trellis2Error.weightsNotFound(Trellis2Configuration.pixal3dRepo
                + (posedMV ? " (+ the unpublished _mv set in pixal3dWeightsDir)" : "")
                + " — register a Pixal3D lane (Trellis2Configuration.pixal3dLane) or set pixal3dWeightsDir")
        }
        let fov = (typedCameras?.first?.fovXDegrees ?? options.pixal3dFovDegrees) * .pi / 180
        var cams: [Pixal3DCamera]
        if let tc = typedCameras, posedTyped {
            cams = tc.map { c in Pixal3DCamera(fovX: (c.fovXDegrees.map { $0 * .pi / 180 }) ?? fov, meshScale: 1.0,
                                               c2w: Mat4(flat: c.cameraToWorld!)) }
        } else if typedCameras != nil {
            guard views.count == 1 else {
                throw PackageError.unsupportedRequestFeature(
                    "pixal3d with additionalViews needs a cameraToWorld on every camera (posed multi-view)")
            }
            cams = [Pixal3DCamera(fovX: fov, meshScale: 1.0)]
        } else if let mats = options.pixal3dCameras {
            guard mats.count == views.count else {
                throw PackageError.unsupportedRequestFeature(
                    "pixal3dCameras has \(mats.count) matrices for \(views.count) views (image + additionalViews)")
            }
            cams = mats.map { Pixal3DCamera(fovX: fov, meshScale: 1.0, c2w: Mat4(flat: $0)) }
        } else {
            guard views.count == 1 else {
                throw PackageError.unsupportedRequestFeature(
                    "pixal3d with additionalViews needs pixal3dCameras (posed multi-view); pose-free views are not supported")
            }
            cams = [Pixal3DCamera(fovX: fov, meshScale: 1.0)]
        }
        let posed = posedTyped || options.pixal3dCameras != nil
        let tPre = Date()
        let pviews: [Pixal3DView] = try views.enumerated().map { i, img in
            let p512 = posed ? try ImagePreprocess.pixal3dViewPixels(from: img, size: 512)
                             : try ImagePreprocess.pixal3dPixels(from: img, size: 512)
            let p1024 = posed ? try ImagePreprocess.pixal3dViewPixels(from: img, size: 1024)
                              : try ImagePreprocess.pixal3dPixels(from: img, size: 1024)
            return Pixal3DView(normalized512: p512.normalized, guide512: p512.guide,
                               normalized1024: p1024.normalized, guide1024: p1024.guide, camera: cams[i])
        }
        let preprocessS = -tPre.timeIntervalSinceNow
        try Task.checkCancellation()
        let (bakedOpt, _, stageMetrics, camera) = try pipeline.generatePixal3D(
            tier: tier, views: pviews, multiViewWeights: posed && views.count > 1,
            ssPhases: SSGrid.ropePhases(resolution: 16),
            shapeMean: MLXArray(shapeMean), shapeStd: MLXArray(shapeStd),
            texMean: MLXArray(texMean), texStd: MLXArray(texStd),
            texture: options.texture, targetFaces: options.decimateFaces,
            yUp: options.yUp, atlasSize: options.atlasSize, pbrMaps: options.pbrMaps,
            detailMaps: options.detailMaps,
            unwrapBackend: configuration.unwrapBackend == "provenance" ? .provenance : .xatlas,
            slatCfgSDPA: configuration.slatCfgAttention.flatMap { SDPAPrecision(rawValue: $0) } ?? .fp32,
            texSDPA: configuration.texAttention.flatMap { SDPAPrecision(rawValue: $0) } ?? .bf16,
            steps: options.steps, maxHRTokens: configuration.maxHRTokens, seed: options.seed,
            frame: Pixal3DFrame(rawValue: frameRaw) ?? .front, log: { _ in })
        guard let baked = bakedOpt else { throw PackageError.notLoaded }
        try Task.checkCancellation()
        let tGlb = Date()
        let glb = try Self.glbData(baked, options: options)
        let glbExportS = -tGlb.timeIntervalSinceNow
        let mesh = Mesh(format: .glb, data: glb, vertexCount: baked.vertices.dim(0), faceCount: baked.faces.dim(0),
                        hasVertexColors: false)
        if let path = configuration.metricsPath {
            var rec = stageMetrics.flatRecord()
            rec["pixal3d"] = true
            rec["pixal3d_views"] = camera.views
            rec["pixal3d_fov_deg"] = camera.fovXRadians * 180 / .pi
            rec["pixal3d_distance"] = camera.distance
            rec["pixal3d_frame"] = camera.frame.rawValue
            rec["pixal3d_camera0_to_mesh"] = camera.camera0ToMesh
            rec["matting_s"] = mattingS
            rec["preprocess_s"] = preprocessS
            rec["glb_export_s"] = glbExportS
            rec["mesh_verts"] = baked.vertices.dim(0)
            rec["mesh_faces"] = baked.faces.dim(0)
            rec["y_up"] = options.yUp
            if !options.detailMaps.isEmpty { Self.recordDetailMaps(&rec, baked) }
            rec["glb_bytes"] = glb.count
            rec["gpu_peak_bytes"] = Int(Memory.peakMemory)
            rec["e2e_total_s"] = mattingS + preprocessS + stageMetrics.generateTotalS + glbExportS
            if let data = try? JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys]),
               let str = String(data: data, encoding: .utf8) {
                print("JSON \(str)")
                try? str.write(toFile: path, atomically: true, encoding: .utf8)
            }
        }
        // contract 1.64.0 camera report: view 0's camera in the returned mesh's frame, and that frame
        return ImageTo3DResponse(mesh: mesh,
                                 camera: ImageTo3DCamera(fovXDegrees: camera.fovXRadians * 180 / .pi,
                                                         cameraToWorld: camera.camera0ToMesh),
                                 frame: ImageTo3DFrame(rawValue: camera.frame.rawValue))
    }

    // MARK: GLB

    /// The response GLB: base colour, plus whichever opt-in maps the request switched on.
    static func glbData(_ baked: BakedMesh, options: Trellis2RequestOptions) throws -> Data {
        let d = baked.detail
        let occlusion: GLTFExport.Occlusion? =
            d.occlusionInMetallicRoughness && options.pbrMaps ? .metallicRoughnessRed
            : d.occlusionRGBA.map { .texture(pixels: $0, width: d.occlusionSize, height: d.occlusionSize) }
        return try GLTFExport.glbData(
            positions: baked.vertices, indices: baked.faces,
            normals: baked.normals, uvs: baked.uvs,
            baseColorRGBA: (baked.texRGBA, baked.atlasSize, baked.atlasSize),
            metallicRoughnessRGBA: options.pbrMaps ? baked.mrRGBA.map { ($0, baked.atlasSize, baked.atlasSize) } : nil,
            tangents: d.normalRGBA != nil ? d.tangents : nil,
            normalRGBA: d.normalRGBA.map { ($0, d.normalSize, d.normalSize) },
            occlusion: occlusion)
    }

    /// Detail-map facts for the metrics record (only written when a map was asked for).
    static func recordDetailMaps(_ rec: inout [String: Any], _ baked: BakedMesh) {
        let d = baked.detail
        rec["normal_map_size"] = d.normalRGBA != nil ? d.normalSize : 0
        rec["ao_map_size"] = d.occlusionSize
        rec["ao_in_metallic_roughness"] = d.occlusionInMetallicRoughness
    }

    // MARK: weight resolution

    /// Resolve the consolidated-snapshot directory: explicit override (flat or `<dir>/<org>/<name>`)
    /// first, then the ModelStore layout under `root`. Throws if neither holds a complete snapshot.
    private static func resolveSnapshotDir(root: URL?, override: URL?) throws -> URL {
        if let override {
            if Trellis2Configuration.snapshotPresent(at: override) { return override }
            let sub = override.appending(path: Trellis2Configuration.consolidatedRepo)
            if Trellis2Configuration.snapshotPresent(at: sub) { return sub }
        }
        if let dir = ModelStore(root: root).directory(for: Trellis2Configuration.consolidatedRepo),
           Trellis2Configuration.snapshotPresent(at: dir) {
            return dir
        }
        throw Trellis2Error.weightsNotFound(Trellis2Configuration.consolidatedRepo)
    }

    /// Parse shape + tex slat mean/std from the consolidated `normalization.json`
    /// (`{shape_slat_normalization:{mean,std}, tex_slat_normalization:{mean,std}}`).
    private static func loadNormalization(_ dir: URL) throws
        -> (shapeMean: [Float], shapeStd: [Float], texMean: [Float], texStd: [Float]) {
        let path = dir.appending(path: "normalization.json")
        guard let data = FileManager.default.contents(atPath: path.path),
              let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Trellis2Error.normalizationMissing
        }
        func vec(_ k1: String, _ k2: String) -> [Float] {
            ((obj[k1] as? [String: Any])?[k2] as? [Double])?.map { Float($0) } ?? []
        }
        let sm = vec("shape_slat_normalization", "mean"), ss = vec("shape_slat_normalization", "std")
        guard !sm.isEmpty, !ss.isEmpty else { throw Trellis2Error.normalizationMissing }
        return (sm, ss, vec("tex_slat_normalization", "mean"), vec("tex_slat_normalization", "std"))
    }
}

// Registration (call site, when integrating):
//   try await engine.register(PackageRegistration.of(Trellis2Package.self),
//                             configuration: Trellis2Configuration())
