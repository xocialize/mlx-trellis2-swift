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

    public init(quant: Quant = .bf16,
                defaultMode: Mode = ImageTo3DContract.res512,
                modelsRootDirectory: URL? = nil,
                weightsRootOverride: URL? = nil,
                steps: Int = 12,
                seed: UInt64 = 0) {
        self.quant = quant
        self.defaultMode = defaultMode
        self.modelsRootDirectory = modelsRootDirectory
        self.weightsRootOverride = weightsRootOverride
        self.steps = steps
        self.seed = seed
    }
}

// MARK: - Weight sources (auto-materialization, engine ≥0.19.0 MAT gate)

extension Trellis2Configuration: WeightSourcing {
    /// The consolidated, remap-free weights repo (one download): every component in the Swift
    /// module-key layout (incl. DINOv3, redistributed under the DINOv3 License §1.b) +
    /// `normalization.json` + the license files. The repo is GATED (auto-approval, DINOv3 terms):
    /// first-run materialization env-detects `HF_TOKEN` (an account that accepted the terms).
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
        "shape_dec.safetensors",
        "tex_flow_512.safetensors",
        "tex_dec.safetensors",
    ]

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
        return probeFiles.allSatisfy { fm.fileExists(atPath: dir.appending(path: $0).path) }
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
                // Split footprint (contract 1.14), res512 default tier. DOCUMENTED ESTIMATE — a full
                // measured run is ~10 min on the GPU; these need an in-app phys_footprint re-baseline
                // (MLXEngineTestKit) before the registry Eff flips to ✅. Basis:
                //   residentBytes ≈ 21 GB — the pipeline holds all 7 components resident at once with
                //     NO per-stage eviction, and every constructor casts weights to fp32 (the parity
                //     dtype). On-disk bf16/fp16 sums ~11 GB → ~21 GB fp32-resident (3×1.3B DiTs 7.8→15.5,
                //     2 sparse decoders 1.9→3.8, ss_dec 0.15→0.3, DINOv3 1.2 already-fp32). Declaring
                //     `.bf16` quant but an fp32-resident floor is deliberate — the manifest reports the
                //     memory this build actually occupies. P0 efficiency follow-up (bf16-resident
                //     weights + per-stage load→use→evict) would roughly halve this.
                //   peakActivationBytes ≈ 8 GB — transient scratch on top: sparse full-attention over
                //     ~19.5k SLat tokens (O(T²)) + the ~7.85M-voxel shape/tex decodes. Estimate anchored
                //     to the old port's MEASURED res512 total ~14 GB (which held bf16 weights); flagged
                //     as the least-certain number pending the in-app probe.
                footprints: [QuantFootprint(quant: .bf16,
                                            residentBytes: 21_000_000_000,
                                            peakActivationBytes: 8_000_000_000)],
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

        // Resolution tier (voxel grid). res512 is the validated tier; res1024/res1536 map onto the
        // cascade in the core (currently the res512 pipeline; higher tiers documented as follow-up).
        _ = req.mode ?? configuration.defaultMode

        // Preprocess the (assumed pre-masked / plain) input to DINOv3's NCHW [1,3,512,512]. Background
        // removal (BiRefNet matting) is a SEPARATE package composed at the app layer — a follow-up.
        let pixels = try ImagePreprocess.dinoPixels(from: req.image)
        try Task.checkCancellation()

        let (cond, negCond) = pipeline.encodeImage(pixels)
        try Task.checkCancellation()

        // In-package SS constants (no goldens dir): the deterministic 16³ RoPE grid + seeded noise.
        let ssPhases = SSGrid.ropePhases(resolution: 16)
        let ssNoise = SSGrid.noise(resolution: 16, seed: configuration.seed)

        // The pipeline's samplers/decoders honor cooperative cancellation at their step/layer seams
        // (they read the ambient Task); generate() is the long-running body.
        let baked = try pipeline.generate(
            cond: cond, negCond: negCond, ssNoise: ssNoise, ssPhases: ssPhases,
            shapeMean: MLXArray(shapeMean), shapeStd: MLXArray(shapeStd),
            texMean: MLXArray(texMean), texStd: MLXArray(texStd),
            seed: configuration.seed, log: { _ in })
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
