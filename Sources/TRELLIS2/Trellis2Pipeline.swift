import Foundation
import MLX
import MLXRandom

/// End-to-end TRELLIS.2 generation, all native Swift-MLX: image conditioning →
/// sparse structure → shape SLat → shape mesh + tex SLat → PBR → textured GLB.
/// Every stage is a parity-gated component; this chains them.
///
/// Tiers (T0.3) mirror upstream `pipeline_type`: `res512` runs the 512 flows directly on the
/// 32³ structure grid; `res1024`/`res1536` run the cascade — LR 512-flow pass, shape-decoder
/// upsample to the 512 dual-grid, host re-quantization onto the hr/16 SLat grid (token
/// back-off to ≥1024), then the 1024 flows at hr resolution. SLat flow DiTs load lazily per
/// tier (each is ~5.2 GB fp32-resident); `generate` prunes cached flows to the tier's set.
public final class Trellis2Pipeline {
    let ssDit: SparseStructureFlowModel
    let ssDec: SparseStructureDecoder
    let shapeDec: ShapeSlatDecoder
    let texDec: ShapeSlatDecoder     // FlexiDualGrid decoder w/ 6-ch (PBR) output weights
    let dino: DINOv3                 // image conditioning (ViT-L/16)

    /// The four SLat flow DiTs, keyed by stage×resolution. Lazily loaded (see `flow(_:)`).
    public enum FlowKind: String, CaseIterable, Sendable {
        case shape512, shape1024, tex512, tex1024
    }
    private let flowURL: (FlowKind) -> URL
    private var flows: [FlowKind: SLatFlowModel] = [:]

    /// `ckptDir` = the TRELLIS.2-4B ckpts dir; `ssDecPath` = the TRELLIS-image-large SS
    /// decoder; `dinoPath` = the facebook DINOv3 ViT-L/16 model.safetensors.
    public init(ckptDir: String, ssDecPath: String, dinoPath: String) throws {
        func load(_ f: String) throws -> [String: MLXArray] {
            try loadArrays(url: URL(fileURLWithPath: "\(ckptDir)/\(f).safetensors"))
        }
        ssDit = SparseStructureFlowModel(weights: try load("ss_flow_img_dit_1_3B_64_bf16"))
        ssDec = SparseStructureDecoder(weights: try loadArrays(url: URL(fileURLWithPath: ssDecPath)))
        shapeDec = ShapeSlatDecoder(weights: try load("shape_dec_next_dc_f16c32_fp16"))
        texDec = ShapeSlatDecoder(weights: try load("tex_dec_next_dc_f16c32_fp16"))
        dino = DINOv3(weights: try loadArrays(url: URL(fileURLWithPath: dinoPath)))
        flowURL = { kind in
            let f: String
            switch kind {
            case .shape512: f = "slat_flow_img2shape_dit_1_3B_512_bf16"
            case .shape1024: f = "slat_flow_img2shape_dit_1_3B_1024_bf16"
            case .tex512: f = "slat_flow_imgshape2tex_dit_1_3B_512_bf16"
            case .tex1024: f = "slat_flow_imgshape2tex_dit_1_3B_1024_bf16"
            }
            return URL(fileURLWithPath: "\(ckptDir)/\(f).safetensors")
        }
    }

    /// Load from a single consolidated snapshot directory whose files carry the module-key names
    /// (`struct_flow`/`struct_dec`/`shape_flow_512`/`shape_flow_1024`/`shape_dec`/`tex_flow_512`/
    /// `tex_flow_1024`/`tex_dec`/`dino` `.safetensors`) — the layout the `xocialize/trellis2-mlx`
    /// repo publishes and the `trellis2-consolidate` tool assembles. No key remap: the published
    /// keys ARE the Swift module keys (each component self-verifies via its parity gate).
    public init(consolidatedDir dir: String) throws {
        func load(_ f: String) throws -> [String: MLXArray] {
            try loadArrays(url: URL(fileURLWithPath: "\(dir)/\(f).safetensors"))
        }
        ssDit = SparseStructureFlowModel(weights: try load("struct_flow"))
        ssDec = SparseStructureDecoder(weights: try load("struct_dec"))
        shapeDec = ShapeSlatDecoder(weights: try load("shape_dec"))
        texDec = ShapeSlatDecoder(weights: try load("tex_dec"))
        dino = DINOv3(weights: try load("dino"))
        flowURL = { kind in
            let f: String
            switch kind {
            case .shape512: f = "shape_flow_512"
            case .shape1024: f = "shape_flow_1024"
            case .tex512: f = "tex_flow_512"
            case .tex1024: f = "tex_flow_1024"
            }
            return URL(fileURLWithPath: "\(dir)/\(f).safetensors")
        }
    }

    /// Lazily load + cache a SLat flow DiT (fp32-cast, ~5.2 GB resident each).
    public func flow(_ kind: FlowKind) throws -> SLatFlowModel {
        if let m = flows[kind] { return m }
        let m = SLatFlowModel(weights: try loadArrays(url: flowURL(kind)))
        flows[kind] = m
        return m
    }

    /// Drop cached flow DiTs (all, or all but `keeping`) so their buffers can be reclaimed.
    public func releaseFlows(keeping: Set<FlowKind> = []) {
        flows = flows.filter { keeping.contains($0.key) }
    }

    /// Single-view image conditioning: preprocessed pixels [1,3,S,S] → DINOv3 tokens.
    public func encodeImage(_ pixels: MLXArray) -> (cond: MLXArray, negCond: MLXArray) {
        encodeImages([pixels])
    }

    /// Multi-view image conditioning. Each element is one view's preprocessed pixels [1,3,S,S]
    /// (front + additional views of the SAME subject). DINOv3 runs per view (its patch-embed assumes
    /// batch 1, so we don't batch), then the per-view token sets are CONCATENATED along the sequence
    /// into a single [1, K·N, D] context — matching the oracle's `get_cond`
    /// (`cond = image_cond_model(images); cond.reshape(1, -1, D)`). K=1 is the single-view identity.
    /// neg_cond is zeros (matches get_cond's `torch.zeros_like`).
    public func encodeImages(_ pixelsPerView: [MLXArray]) -> (cond: MLXArray, negCond: MLXArray) {
        precondition(!pixelsPerView.isEmpty, "encodeImages requires at least one view")
        let perView = pixelsPerView.map { dino($0) }                 // each [1, N, D]
        let cond = perView.count == 1 ? perView[0]
                                      : MLX.concatenated(perView, axis: 1)   // [1, K·N, D]
        return (cond, MLXArray.zeros(cond.shape, dtype: cond.dtype))
    }

    /// cond/negCond = SS + res512 conditioning (DINOv3 at image_size 512, the oracle's cond_512 —
    /// SS always uses the 512 tokens). `cond1024`/`negCond1024` = the cascade tiers' shape+tex
    /// conditioning (DINOv3 at image_size 1024); unused by `.res512`, so pass the 512 pair there.
    /// ssNoise = [1,8,16,16,16]; ssPhases = SS-DiT rope phases for the 16³ grid.
    /// `shapeMean/Std`, `texMean/Std` = the pipeline's per-channel SLat normalization [32].
    /// The sampler emits a NORMALIZED latent (std≈1); the decoder needs it DENORMALIZED
    /// (× std + mean). The tex concat_cond, by contrast, wants the raw normalized shape slat.
    /// `texture` false → geometry-only fast path (skip the tex flow + decoder; flat-gray mesh,
    /// ~2× faster). `targetFaces` → decimation target for MeshBake. `yUp` → emit Y-up
    /// (x,y,z)→(x,z,−y), the glTF/VRM convention required BEFORE rigging (a post-rig
    /// reorientation mis-composes with clip playback).
    /// `remeshRes` nil → per-tier default (256; 384 at res1536).
    /// Returns the baked mesh and the EFFECTIVE hr resolution (== tier target unless the
    /// cascade token back-off reduced it).
    public func generate(tier: Trellis2Tier = .res1024,
                         cond: MLXArray, negCond: MLXArray,
                         cond1024: MLXArray, negCond1024: MLXArray, ssNoise: MLXArray, ssPhases: MLXArray,
                         shapeMean: MLXArray, shapeStd: MLXArray, texMean: MLXArray, texStd: MLXArray,
                         texture: Bool = true, targetFaces: Int = 120_000, yUp: Bool = false,
                         unwrapBackend: UnwrapBackend = .xatlas,
                         slatCfgSDPA: SDPAPrecision = .fp32, texSDPA: SDPAPrecision? = nil,
                         steps: Int = 12,
                         slatCfgInterval: (Float, Float)? = nil,   // nil = upstream (0.6, 1.0)
                         slatNegEvery: Int = 1,                    // CFG cache: 1 = off (upstream)
                         remeshRes: Int? = nil, seed: UInt64 = 0,
                         log: (String) -> Void = { print($0) }) throws -> (mesh: BakedMesh, hrResolution: Int, metrics: Trellis2StageMetrics) {
        // Bound flow residency to this tier's working set.
        releaseFlows(keeping: tier.isCascade ? [.shape512, .shape1024, .tex1024] : [.shape512, .tex512])

        // Per-stage timing (BENCHMARKS.md full-workflow harness). Each `-t.timeIntervalSinceNow`
        // read below follows the stage's MLX.eval barrier, so deltas are real wall-clock.
        var metrics = Trellis2StageMetrics()
        metrics.tier = tier.rawValue
        metrics.seed = seed
        metrics.backend = unwrapBackend.rawValue

        // High-guidance CFG (SS r0.7, shape r0.5) is fp-sensitive — the g·vPos−(g−1)·vNeg
        // near-cancellation amplifies per-forward noise, so bf16-SDPA noise there shows up
        // as surface speckle. Per-stage SDPA precision (same fused kernel, cast only):
        //   SS         — ALWAYS fp32 (highest rescale 0.7, knife-edge schedule; ~5% of cascade e2e)
        //   shape SLat — `slatCfgSDPA` (default fp32; fp16 is the A/B candidate — 10-bit
        //                mantissa vs bf16's 7, so ~8× less rounding noise under CFG)
        //   tex SLat   — `texSDPA`; nil = legacy (the ambient fastAttention request). The tex
        //                flow is CFG-free (guidance 1.0, single forward), the documented bf16-safe case.
        let savedCast = TRELLIS2Config.sdpaCastDtype
        defer { TRELLIS2Config.sdpaCastDtype = savedCast }
        metrics.slatCfgSdpa = slatCfgSDPA.rawValue
        metrics.texSdpa = texSDPA?.rawValue ?? (savedCast == .bfloat16 ? "bf16" : "fp32")
        metrics.stepsSS = steps
        metrics.stepsSLat = steps
        let cfgInterval = slatCfgInterval ?? (0.6, 1.0)
        metrics.slatCfgInterval = "\(cfgInterval.0),\(cfgInterval.1)"
        metrics.slatNegEvery = slatNegEvery

        // 1) sparse structure: SS sampler → z_s → decode → occupancy coords on the 32³
        //    SLat grid (all tiers: upstream ss_res=32 for '512' and both cascades).
        var t = Date()
        TRELLIS2Config.sdpaCastDtype = nil   // SS: fp32 SDPA, unconditionally
        let zs = FlowEulerSampler.sampleSS(model: ssDit, noise: ssNoise, cond: cond, negCond: negCond, phases: ssPhases,
                                           steps: steps, guidanceStrength: 7.5, guidanceRescale: 0.7,
                                           guidanceInterval: (0.6, 1.0), rescaleT: 5.0)
        MLX.eval(zs); metrics.ssSampleS = -t.timeIntervalSinceNow   // [1a] SS flow sampler
        t = Date()
        let coords = ssDec.coords(ssDec(zs), resolution: 32); MLX.eval(coords)
        metrics.ssDecodeS = -t.timeIntervalSinceNow                 // [1b] SS decoder → coords
        metrics.voxels32 = coords.dim(0)
        log("  [1] sparse structure: \(coords.dim(0)) voxels @32³ (\(String(format: "%.1f", metrics.ssSampleS + metrics.ssDecodeS))s)")

        // 2) shape SLat (SDPA per `slatCfgSDPA`; default fp32 — CFG quality). res512: the 512
        //    flow IS the final pass. Cascade: the 512 flow is the LR pass, conditioned like res512.
        TRELLIS2Config.sdpaCastDtype = slatCfgSDPA.castDtype
        t = Date(); MLXRandom.seed(seed)
        var shapeSlat = FlowEulerSampler.sampleSLat(
            model: try flow(.shape512), noiseFeats: MLXRandom.normal([coords.dim(0), 32]), coords: coords,
            cond: cond, negCond: negCond,
            steps: steps, guidanceStrength: 7.5, guidanceRescale: 0.5, guidanceInterval: cfgInterval,
            rescaleT: 3.0, negEvery: slatNegEvery)
        MLX.eval(shapeSlat)
        metrics.shapeSlatS = -t.timeIntervalSinceNow
        metrics.shapeTokens = coords.dim(0)
        metrics.hrTokens = coords.dim(0)   // overwritten below on the cascade tiers
        log("  [2] \(tier.isCascade ? "LR " : "")shape SLat sampled: \(coords.dim(0)) tokens (\(String(format: "%.1f", metrics.shapeSlatS))s)")

        // 2b) cascade: "upsample" = the shape DECODER's forward-output coords on the LR slat
        //     (upstream `upsample(slat, 4)` is the decoder run to full depth), re-quantized onto
        //     the hr/16 SLat grid with token back-off, then the HR pass with the 1024 flow +
        //     1024² DINOv3 conditioning.
        var hrCoords = coords
        var hrResolution = tier.targetResolution
        if tier.isCascade {
            t = Date()
            let lrDenorm = shapeSlat * shapeStd + shapeMean
            let upCoords = shapeDec(SparseTensor(feats: lrDenorm, coords: coords)).coords
            MLX.eval(upCoords)
            (hrCoords, hrResolution) = CascadeQuantize.requantize(
                upCoords: upCoords, lrResolution: 512, targetResolution: tier.targetResolution)
            MLX.eval(hrCoords)
            metrics.upsampleRequantizeS = -t.timeIntervalSinceNow
            metrics.hrTokens = hrCoords.dim(0)
            if hrResolution != tier.targetResolution {
                log("  [2b] token back-off: resolution reduced to \(hrResolution)")
            }
            log("  [2b] upsample \(upCoords.dim(0)) → re-quantized \(hrCoords.dim(0)) tokens @\(hrResolution/16)³ (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

            t = Date(); MLXRandom.seed(seed &+ 2)
            shapeSlat = FlowEulerSampler.sampleSLat(
                model: try flow(.shape1024), noiseFeats: MLXRandom.normal([hrCoords.dim(0), 32]), coords: hrCoords,
                cond: cond1024, negCond: negCond1024,
                steps: steps, guidanceStrength: 7.5, guidanceRescale: 0.5, guidanceInterval: cfgInterval,
                rescaleT: 3.0, negEvery: slatNegEvery)
            MLX.eval(shapeSlat)
            metrics.shapeSlatHrS = -t.timeIntervalSinceNow
            log("  [2c] HR shape SLat sampled: \(hrCoords.dim(0)) tokens (\(String(format: "%.1f", metrics.shapeSlatHrS ?? 0))s)")
        }
        // CFG samplers done; tex (CFG-free, single forward) runs at `texSDPA`,
        // falling back to the ambient request (bf16 when the CLI set fastAttention).
        TRELLIS2Config.sdpaCastDtype = texSDPA?.castDtype ?? savedCast

        // 3) shape decode → 7-ch dual-grid + subdivision masks. Denormalize the latent
        //    (sampler emits normalized; decoder expects std·x+mean).
        t = Date()
        let shapeSlatDenorm = shapeSlat * shapeStd + shapeMean
        let (shapeOut, subs) = shapeDec.decodeCapturingSubs(SparseTensor(feats: shapeSlatDenorm, coords: hrCoords))
        MLX.eval(shapeOut.feats, shapeOut.coords)
        metrics.shapeDecodeS = -t.timeIntervalSinceNow
        metrics.decodedVoxels = shapeOut.count
        log("  [3] shape decoded: \(shapeOut.count) voxels (\(String(format: "%.1f", metrics.shapeDecodeS))s)")

        // 4-5) texture: tex SLat sampler (concat_cond=shape_slat, no CFG) → tex decode → base
        //      color. `texture=false` skips both stages for a flat-gray geometry-only mesh.
        //      Cascade mirrors upstream: tex 1024 flow, HR shape slat concat, same HR coords,
        //      cond_1024; res512 keeps the 512 flow + cond_512.
        let baseColor: MLXArray
        if texture {
            t = Date(); MLXRandom.seed(seed &+ 1)
            let texSlat = FlowEulerSampler.sampleSLat(
                model: try flow(tier.isCascade ? .tex1024 : .tex512),
                noiseFeats: MLXRandom.normal([hrCoords.dim(0), 32]), coords: hrCoords,
                cond: tier.isCascade ? cond1024 : cond, negCond: tier.isCascade ? negCond1024 : negCond,
                concatCond: shapeSlat, steps: steps, guidanceStrength: 1.0, guidanceRescale: 0.0, guidanceInterval: (0.6, 0.9), rescaleT: 3.0)
            MLX.eval(texSlat)
            metrics.texSlatS = -t.timeIntervalSinceNow
            log("  [4] tex SLat sampled (\(String(format: "%.1f", metrics.texSlatS ?? 0))s)")
            t = Date()
            let texSlatDenorm = texSlat * texStd + texMean
            let texOut = texDec(SparseTensor(feats: texSlatDenorm, coords: hrCoords), guidedMasks: subs)
            baseColor = MLX.clip(texOut.feats[0..., 0..<3] * 0.5 + 0.5, min: 0, max: 1)
            MLX.eval(baseColor)
            metrics.texDecodeS = -t.timeIntervalSinceNow
            log("  [5] tex decoded: \(texOut.count) voxels (\(String(format: "%.1f", metrics.texDecodeS ?? 0))s)")
        } else {
            baseColor = MLXArray.zeros([shapeOut.count, 3]) + 0.5   // flat gray, geometry-only
            log("  [4-5] texture off — geometry-only (flat gray)")
        }

        // 6) mesh + texture bake → clean mesh, at the tier's (possibly backed-off) resolution.
        t = Date()
        let remesh = remeshRes ?? (hrResolution >= 1536 ? 384 : 256)
        let (bakedInit, bakeMetrics) = try MeshBake.run(shapeFeats: shapeOut.feats, coords: shapeOut.coords, texBaseColor: baseColor,
                                     fineRes: Float(hrResolution), remeshRes: remesh,
                                     targetFaces: targetFaces, atlasSize: 1024,
                                     backend: unwrapBackend, log: log)
        metrics.meshbakeS = -t.timeIntervalSinceNow
        metrics.bake = bakeMetrics
        var baked = bakedInit

        // 7) optional Y-up reorientation (x,y,z)→(x,z,−y): a proper rotation, winding preserved.
        t = Date()
        if yUp {
            func toYUp(_ a: MLXArray) -> MLXArray {
                MLX.stacked([a[0..., 0], a[0..., 2], -a[0..., 1]], axis: 1)
            }
            baked = BakedMesh(vertices: toYUp(baked.vertices), faces: baked.faces, normals: toYUp(baked.normals),
                              uvs: baked.uvs, texRGBA: baked.texRGBA, atlasSize: baked.atlasSize, coverage: baked.coverage,
                              mrRGBA: baked.mrRGBA,
                              filledBeforeInpaint: baked.filledBeforeInpaint, wallFlipped: baked.wallFlipped)
            log("  [7] reoriented Y-up")
        }
        metrics.yupS = -t.timeIntervalSinceNow
        metrics.hrResolution = hrResolution
        return (baked, hrResolution, metrics)
    }
}
