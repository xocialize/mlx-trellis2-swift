import Foundation

/// Structured per-stage timing for one `Trellis2Pipeline.generate` call — the
/// full-workflow metrics harness (BENCHMARKS.md; UV-UNWRAP-METAL-PLAN.md).
///
/// All `*S` fields are wall-clock SECONDS, measured with `Date()` deltas whose
/// windows each END on an `MLX.eval(...)` barrier — MLX is lazy, so a delta only
/// measures a stage if completion is forced inside its window. Every timer here
/// brackets an existing eval in `generate()`/`MeshBake.run`; do NOT add a timer
/// without a matching barrier or its cost silently migrates to the next eval.
///
/// This is generate()-SCOPE only: DINOv3 conditioning and GLB export happen in
/// the package around generate() and are added to the flat record by the caller.
public struct BakeStageMetrics: Codable, Sendable {
    // sub-stage seconds (MeshBake.run)
    public var rawDualgridS = 0.0      // DualGridMesh.extract + raw-shell BVH build
    public var remeshS = 0.0           // remeshDualContouring
    public var simplifyS = 0.0         // QEM simplify to face budget (0 if under budget)
    public var unwrapS = 0.0           // uvUnwrapParallel / provenanceUnwrap
    public var rasterizeS = 0.0        // 5-pass jittered rasterize @2× supersample
    public var bakeSampleS = 0.0       // BVH closest-point remap + GridSample3d trilinear
    public var writeDownsampleS = 0.0  // supersample write + box-average down
    public var inpaintS = 0.0          // dilateInpaint to completion
    // counts
    public var rawFaces = 0
    public var remeshFaces = 0
    public var simplifyFaces = 0       // == remeshFaces when no simplify ran
    public var finalFaces = 0
    public var charts = 0
    public var coveredTexelWrites = 0  // K (5-pass conservative)
    public var coverage: Float = 0     // texels filled before inpaint

    public init() {}
}

/// Per-stage timing for one full generation (generate() scope).
public struct Trellis2StageMetrics: Codable, Sendable {
    public var tier = ""
    public var hrResolution = 0
    public var seed: UInt64 = 0
    public var backend = ""
    public var xatlasLane = "swift-vendored-parallel"   // BENCHMARKS.md hard rule: always state the lane
    public var stepsSS = 12            // FlowEulerSampler.sampleSS default
    public var stepsSLat = 12          // FlowEulerSampler.sampleSLat default
    public var slatCfgSdpa = "fp32"    // SDPA precision of the CFG shape flows (state-the-lane)
    public var texSdpa = "fp32"        // SDPA precision of the CFG-free tex flow

    // stage seconds (see generate() [1]..[7])
    public var ssSampleS = 0.0         // [1a] SS flow Euler sampler (fp32 CFG)
    public var ssDecodeS = 0.0         // [1b] SS decoder → occupancy coords @32³
    public var shapeSlatS = 0.0        // [2] shape SLat flow (LR pass on cascade, final on res512)
    public var upsampleRequantizeS: Double? = nil   // [2b] cascade only
    public var shapeSlatHrS: Double? = nil          // [2c] cascade only (HR shape flow)
    public var shapeDecodeS = 0.0      // [3] shape decoder → dual-grid voxels
    public var texSlatS: Double? = nil // [4] tex SLat flow (nil when texture off)
    public var texDecodeS: Double? = nil            // [5] tex decoder → base color
    public var meshbakeS = 0.0         // [6] MeshBake.run total
    public var yupS = 0.0              // [7] Y-up reorientation (negligible)

    // counts
    public var voxels32 = 0            // occupancy voxels @32³ (SS output)
    public var shapeTokens = 0         // LR SLat tokens
    public var hrTokens = 0            // HR SLat tokens (== shapeTokens on res512)
    public var decodedVoxels = 0       // shape decoder output voxel count

    public var bake = BakeStageMetrics()

    public init() {}

    /// Sum of the generate()-scope stage timers (excludes DINO/GLB, added by the caller).
    public var generateTotalS: Double {
        ssSampleS + ssDecodeS + shapeSlatS + (upsampleRequantizeS ?? 0) + (shapeSlatHrS ?? 0)
            + shapeDecodeS + (texSlatS ?? 0) + (texDecodeS ?? 0) + meshbakeS + yupS
    }

    /// Flat `[String: Any]` record for JSONSerialization (sortedKeys) — the
    /// unwrapbench/bakeab house convention. The package merges DINO/GLB/total
    /// bookends + mesh/memory context on top before emitting the `JSON ` line.
    public func flatRecord() -> [String: Any] {
        var r: [String: Any] = [
            "tier": tier, "hr_resolution": hrResolution, "seed": seed, "backend": backend,
            "xatlas_lane": xatlasLane, "steps_ss": stepsSS, "steps_slat": stepsSLat,
            "slat_cfg_sdpa": slatCfgSdpa, "tex_sdpa": texSdpa,
            "ss_sample_s": ssSampleS, "ss_decode_s": ssDecodeS,
            "shape_slat_s": shapeSlatS, "shape_decode_s": shapeDecodeS,
            "meshbake_s": meshbakeS, "yup_s": yupS,
            "generate_total_s": generateTotalS,
            "voxels_32": voxels32, "shape_tokens": shapeTokens, "hr_tokens": hrTokens,
            "decoded_voxels": decodedVoxels,
            // bake sub-stages
            "bake_raw_dualgrid_s": bake.rawDualgridS, "bake_remesh_s": bake.remeshS,
            "bake_simplify_s": bake.simplifyS, "bake_unwrap_s": bake.unwrapS,
            "bake_rasterize_s": bake.rasterizeS, "bake_sample_s": bake.bakeSampleS,
            "bake_write_downsample_s": bake.writeDownsampleS, "bake_inpaint_s": bake.inpaintS,
            "bake_raw_faces": bake.rawFaces, "bake_remesh_faces": bake.remeshFaces,
            "bake_simplify_faces": bake.simplifyFaces, "bake_final_faces": bake.finalFaces,
            "bake_charts": bake.charts, "bake_covered_texel_writes": bake.coveredTexelWrites,
            "bake_coverage": bake.coverage,
        ]
        if let v = upsampleRequantizeS { r["upsample_requantize_s"] = v }
        if let v = shapeSlatHrS { r["shape_slat_hr_s"] = v }
        if let v = texSlatS { r["tex_slat_s"] = v }
        if let v = texDecodeS { r["tex_decode_s"] = v }
        return r
    }
}
