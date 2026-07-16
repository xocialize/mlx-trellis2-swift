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

    /// Explicit Codable surface: everything except the `matting` closure (which decodes to nil).
    enum CodingKeys: String, CodingKey {
        case quant, defaultMode, modelsRootDirectory, weightsRootOverride
        case steps, seed, texture, decimateFaces, yUpOutput
    }

    public init(quant: Quant = .bf16,
                defaultMode: Mode = ImageTo3DContract.res512,
                modelsRootDirectory: URL? = nil,
                weightsRootOverride: URL? = nil,
                steps: Int = 12,
                seed: UInt64 = 0,
                texture: Bool = true,
                decimateFaces: Int? = 300_000,
                yUpOutput: Bool = false,
                matting: Trellis2Matting? = nil) {
        self.quant = quant
        self.defaultMode = defaultMode
        self.modelsRootDirectory = modelsRootDirectory
        self.weightsRootOverride = weightsRootOverride
        self.steps = steps
        self.seed = seed
        self.texture = texture
        self.decimateFaces = decimateFaces
        self.yUpOutput = yUpOutput
        self.matting = matting
    }
}

/// The injected foreground-matting hook: canonical `Image` in, canonical `Matte` out (soft alpha at
/// the input image's dimensions, per `MattingContract`). Async so the app can route it through
/// `MLXServeEngine.run(.matting)`; throws propagate out of `Trellis2Package.run()` (a broken matter
/// is a failed generation, not a silent full-frame fallback).
public typealias Trellis2Matting = @Sendable (Image) async throws -> Matte

// MARK: - Per-tier footprint hints (contract 1.13/1.14 FootprintConfigured)

extension Trellis2Configuration: FootprintConfigured {
    /// Per-tier charging (same quant, very different working sets — the BiRefNet pattern).
    /// Keyed on the CONFIGURED defaultMode: an app that pins a tier is charged that tier.
    /// A per-request mode override above the configured tier is not re-charged (same gap as
    /// BiRefNet); apps that roam tiers should configure the highest one they use.
    ///   res512  — resident 18 GB (SS + shape_512 + tex_512 fp32 + DINOv3 + decoders ≈ 17.3),
    ///             activation 18 GB (conservative carry-over of the pre-T0.3 measured transient;
    ///             true res512 decodes at 512, well under it).
    ///   res1024 — resident 23 GB (cascade set), activation 26 GB (measured peak phys 46.76 GB).
    ///   res1536 — nil ⇒ falls through to the manifest QuantFootprint (23 + 56, measured 76.97).
    public var residentBytesHint: UInt64? {
        switch defaultMode {
        case ImageTo3DContract.res512: 18_000_000_000
        case ImageTo3DContract.res1024: 23_000_000_000
        default: nil
        }
    }
    public var peakActivationBytesHint: UInt64? {
        switch defaultMode {
        case ImageTo3DContract.res512: 18_000_000_000
        case ImageTo3DContract.res1024: 26_000_000_000
        default: nil
        }
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
    }

    /// Explicit `weightsRootOverride` first (dev escape hatch — flat `<dir>/<file>` or
    /// `<dir>/<org>/<name>/<file>`), then the ModelStore layout (`<root>/xocialize/trellis2-mlx`).
    /// nil store + no override ⇒ everything missing (fresh-machine posture, MAT-4).
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let dir = weightsRootOverride,
           Self.snapshotPresent(at: dir)
            || Self.snapshotPresent(at: dir.appending(path: Self.consolidatedRepo)) {
            return []
        }
        if let dir = ModelStore(root: storeRoot).directory(for: Self.consolidatedRepo),
           Self.snapshotPresent(at: dir) {
            return []
        }
        return weightSources
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
    public var prewarmPaths: [URL] {
        let dir = weightsRootOverride
            ?? ModelStore(root: modelsRootDirectory).directory(for: Self.consolidatedRepo)
        guard let dir else { return [] }
        return [dir.appending(path: "normalization.json"), dir]
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
        PackageManifest(
            license: LicenseDeclaration(
                weightLicense: .dinov3,    // C7 — DINOv3 governs; allowlisted. "Built with DINOv3".
                portCodeLicense: .mit),    // C8 — our Swift port
            provenance: Provenance(sourceRepo: "microsoft/TRELLIS.2-4B", revision: "main", tier: 3),
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
                footprints: [QuantFootprint(quant: .bf16,
                                            residentBytes: 23_000_000_000,
                                            peakActivationBytes: 56_000_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .pro),
            specialties: [SpecialtyWeight(Specialty(rawValue: "3d-generation"), strength: 1.0)],
            surfaces: [ImageTo3DContract.descriptor(
                name: "trellis2-image-to-3d",
                summary: "Generate a UV-textured 3D triangle mesh (GLB) from a single image. Invents "
                       + "geometry + a base-color texture atlas from one view; pick detail via mode "
                       + "res512/res1024/res1536.",
                modes: [ImageTo3DContract.res512, ImageTo3DContract.res1024, ImageTo3DContract.res1536])])
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
        pipeline = try Trellis2Pipeline(consolidatedDir: ckptDir)

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
        guard let pipeline else { throw PackageError.notLoaded }

        // Resolution tier (voxel grid): res512 = direct 512 flows; res1024/res1536 = the T0.3
        // cascade (LR 512 pass → decoder-upsample re-quantization → HR 1024 flows, token
        // back-off floor 1024).
        let mode = req.mode ?? configuration.defaultMode
        let tier: Trellis2Tier = switch mode {
        case ImageTo3DContract.res1536: .res1536
        case ImageTo3DContract.res1024: .res1024
        default: .res512
        }

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
        if let matting = configuration.matting {
            for i in views.indices {
                views[i] = try await ImagePreprocess.mattedIfNeeded(views[i], using: matting)
                try Task.checkCancellation()   // the matter may itself be a model run
            }
        }
        let px512 = try views.map { try ImagePreprocess.dinoPixels(from: $0, size: 512) }
        try Task.checkCancellation()

        let (cond, negCond) = pipeline.encodeImages(px512)
        let (cond1024, negCond1024): (MLXArray, MLXArray)
        if tier.isCascade {
            let px1024 = try views.map { try ImagePreprocess.dinoPixels(from: $0, size: 1024) }
            (cond1024, negCond1024) = pipeline.encodeImages(px1024)
        } else {
            (cond1024, negCond1024) = (cond, negCond)
        }
        try Task.checkCancellation()

        // In-package SS constants (no goldens dir): the deterministic 16³ RoPE grid + seeded noise.
        let ssPhases = SSGrid.ropePhases(resolution: 16)
        let ssNoise = SSGrid.noise(resolution: 16, seed: configuration.seed)

        // The pipeline's samplers/decoders honor cooperative cancellation at their step/layer seams
        // (they read the ambient Task); generate() is the long-running body.
        let (baked, _) = try pipeline.generate(
            tier: tier,
            cond: cond, negCond: negCond, cond1024: cond1024, negCond1024: negCond1024,
            ssNoise: ssNoise, ssPhases: ssPhases,
            shapeMean: MLXArray(shapeMean), shapeStd: MLXArray(shapeStd),
            texMean: MLXArray(texMean), texStd: MLXArray(texStd),
            texture: configuration.texture, targetFaces: configuration.decimateFaces ?? 120_000,
            yUp: configuration.yUpOutput, seed: configuration.seed, log: { _ in })
        try Task.checkCancellation()

        let glb = try GLTFExport.glbData(
            positions: baked.vertices, indices: baked.faces,
            normals: baked.normals, uvs: baked.uvs,
            baseColorRGBA: (baked.texRGBA, baked.atlasSize, baked.atlasSize))
        let mesh = Mesh(format: .glb, data: glb,
                        vertexCount: baked.vertices.dim(0), faceCount: baked.faces.dim(0),
                        hasVertexColors: false)   // UV-textured, not vertex colors
        return ImageTo3DResponse(mesh: mesh)
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
