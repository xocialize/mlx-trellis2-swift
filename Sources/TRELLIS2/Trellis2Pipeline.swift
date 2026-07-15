import Foundation
import MLX
import MLXRandom

/// End-to-end TRELLIS.2 generation, all native Swift-MLX: image conditioning →
/// sparse structure → shape SLat → shape mesh + tex SLat → PBR → textured GLB.
/// Every stage is a parity-gated component; this chains them.
public final class Trellis2Pipeline {
    let ssDit: SparseStructureFlowModel
    let ssDec: SparseStructureDecoder
    let shapeDit: SLatFlowModel
    let shapeDec: ShapeSlatDecoder
    let texDit: SLatFlowModel
    let texDec: ShapeSlatDecoder     // FlexiDualGrid decoder w/ 6-ch (PBR) output weights
    let dino: DINOv3                 // image conditioning (ViT-L/16)

    /// `ckptDir` = the TRELLIS.2-4B ckpts dir; `ssDecPath` = the TRELLIS-image-large SS
    /// decoder; `dinoPath` = the facebook DINOv3 ViT-L/16 model.safetensors.
    public init(ckptDir: String, ssDecPath: String, dinoPath: String) throws {
        func load(_ f: String) throws -> [String: MLXArray] {
            try loadArrays(url: URL(fileURLWithPath: "\(ckptDir)/\(f).safetensors"))
        }
        ssDit = SparseStructureFlowModel(weights: try load("ss_flow_img_dit_1_3B_64_bf16"))
        ssDec = SparseStructureDecoder(weights: try loadArrays(url: URL(fileURLWithPath: ssDecPath)))
        shapeDit = SLatFlowModel(weights: try load("slat_flow_img2shape_dit_1_3B_512_bf16"))
        shapeDec = ShapeSlatDecoder(weights: try load("shape_dec_next_dc_f16c32_fp16"))
        texDit = SLatFlowModel(weights: try load("slat_flow_imgshape2tex_dit_1_3B_512_bf16"))
        texDec = ShapeSlatDecoder(weights: try load("tex_dec_next_dc_f16c32_fp16"))
        dino = DINOv3(weights: try loadArrays(url: URL(fileURLWithPath: dinoPath)))
    }

    /// Load from a single consolidated snapshot directory whose files carry the module-key names
    /// (`struct_flow`/`struct_dec`/`shape_flow_512`/`shape_dec`/`tex_flow_512`/`tex_dec`/`dino`
    /// `.safetensors`) — the layout the `xocialize/trellis2-mlx` repo publishes and the
    /// `trellis2-consolidate` tool assembles. No key remap: the published keys ARE the Swift module
    /// keys (each component self-verifies via its parity gate).
    public init(consolidatedDir dir: String) throws {
        func load(_ f: String) throws -> [String: MLXArray] {
            try loadArrays(url: URL(fileURLWithPath: "\(dir)/\(f).safetensors"))
        }
        ssDit = SparseStructureFlowModel(weights: try load("struct_flow"))
        ssDec = SparseStructureDecoder(weights: try load("struct_dec"))
        // The SS decoder emits 64³ occupancy, so shape/tex SLat run on a 64³ coord grid (= the
        // oracle '1024' tier); load the resolution-matched 1024 flow models, NOT the 512 ones
        // (running shape_flow_512/tex_flow_512 on 64³ coords was off-distribution). Conditioned on
        // cond_1024 (DINOv3 at image_size 1024) in generate(); SS still uses cond_512.
        shapeDit = SLatFlowModel(weights: try load("shape_flow_1024"))
        shapeDec = ShapeSlatDecoder(weights: try load("shape_dec"))
        texDit = SLatFlowModel(weights: try load("tex_flow_1024"))
        texDec = ShapeSlatDecoder(weights: try load("tex_dec"))
        dino = DINOv3(weights: try load("dino"))
    }

    /// Single-view image conditioning: preprocessed pixels [1,3,512,512] → DINOv3 tokens.
    public func encodeImage(_ pixels: MLXArray) -> (cond: MLXArray, negCond: MLXArray) {
        encodeImages([pixels])
    }

    /// Multi-view image conditioning. Each element is one view's preprocessed pixels [1,3,512,512]
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

    /// cond/negCond = DINOv3 tokens [1,N,D]; ssNoise = [1,8,16,16,16]; ssPhases = SS-DiT
    /// rope phases for the 16³ grid. Returns a clean textured mesh.
    /// `shapeMean/Std`, `texMean/Std` = the pipeline's per-channel SLat normalization [32].
    /// The sampler emits a NORMALIZED latent (std≈1); the decoder needs it DENORMALIZED
    /// (× std + mean). The tex concat_cond, by contrast, wants the raw normalized shape_slat.
    /// `texture` false → geometry-only fast path (skip the tex flow + decoder; flat-gray mesh,
    /// ~2× faster). `targetFaces` → decimation target for MeshBake. `yUp` → emit Y-up
    /// (x,y,z)→(x,z,−y), the glTF/VRM convention required BEFORE rigging (a post-rig
    /// reorientation mis-composes with clip playback).
    /// `cond`/`negCond` = SS conditioning (DINOv3 at image_size 512, the oracle's cond_512 — SS
    /// always uses the 512 tokens). `cond1024`/`negCond1024` = shape+tex conditioning (DINOv3 at
    /// image_size 1024). Pass the same array for both to run single-resolution.
    public func generate(cond: MLXArray, negCond: MLXArray,
                         cond1024: MLXArray, negCond1024: MLXArray, ssNoise: MLXArray, ssPhases: MLXArray,
                         shapeMean: MLXArray, shapeStd: MLXArray, texMean: MLXArray, texStd: MLXArray,
                         texture: Bool = true, targetFaces: Int = 120_000, yUp: Bool = false,
                         seed: UInt64 = 0, log: (String) -> Void = { print($0) }) throws -> BakedMesh {
        // High-guidance CFG (SS r0.7, shape r0.5) is fp-sensitive — the g·vPos−(g−1)·vNeg
        // near-cancellation amplifies per-forward noise, so bf16-SDPA noise there shows up
        // as surface speckle. Run the CFG samplers in fp32; keep bf16 for the CFG-free tex.
        let fastRequested = TRELLIS2Config.fastAttention

        // 1) sparse structure: SS sampler → z_s → decode → occupancy coords
        var t = Date()
        TRELLIS2Config.fastAttention = false
        let zs = FlowEulerSampler.sampleSS(model: ssDit, noise: ssNoise, cond: cond, negCond: negCond, phases: ssPhases,
                                           steps: 12, guidanceStrength: 7.5, guidanceRescale: 0.7,
                                           guidanceInterval: (0.6, 1.0), rescaleT: 5.0)
        let coords = ssDec.coords(ssDec(zs)); MLX.eval(coords)
        log("  [1] sparse structure: \(coords.dim(0)) voxels (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 2) shape SLat sampler on those coords (fp32 SDPA — CFG quality)
        t = Date(); MLXRandom.seed(seed)
        let shapeSlat = FlowEulerSampler.sampleSLat(
            model: shapeDit, noiseFeats: MLXRandom.normal([coords.dim(0), 32]), coords: coords, cond: cond1024, negCond: negCond1024,
            guidanceStrength: 7.5, guidanceRescale: 0.5, guidanceInterval: (0.6, 1.0), rescaleT: 3.0)
        MLX.eval(shapeSlat)
        TRELLIS2Config.fastAttention = fastRequested
        log("  [2] shape SLat sampled (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 3) shape decode → 7-ch dual-grid + subdivision masks. Denormalize the latent
        //    (sampler emits normalized; decoder expects std·x+mean).
        t = Date()
        let shapeSlatDenorm = shapeSlat * shapeStd + shapeMean
        let (shapeOut, subs) = shapeDec.decodeCapturingSubs(SparseTensor(feats: shapeSlatDenorm, coords: coords))
        MLX.eval(shapeOut.feats, shapeOut.coords)
        log("  [3] shape decoded: \(shapeOut.count) voxels (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 4-5) texture: tex SLat sampler (concat_cond=shape_slat, no CFG) → tex decode → base
        //      color. `texture=false` skips both stages for a flat-gray geometry-only mesh.
        let baseColor: MLXArray
        if texture {
            t = Date(); MLXRandom.seed(seed &+ 1)
            let texSlat = FlowEulerSampler.sampleSLat(
                model: texDit, noiseFeats: MLXRandom.normal([coords.dim(0), 32]), coords: coords, cond: cond1024, negCond: negCond1024,
                concatCond: shapeSlat, guidanceStrength: 1.0, guidanceRescale: 0.0, guidanceInterval: (0.6, 0.9), rescaleT: 3.0)
            MLX.eval(texSlat)
            log("  [4] tex SLat sampled (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")
            t = Date()
            let texSlatDenorm = texSlat * texStd + texMean
            let texOut = texDec(SparseTensor(feats: texSlatDenorm, coords: coords), guidedMasks: subs)
            baseColor = MLX.clip(texOut.feats[0..., 0..<3] * 0.5 + 0.5, min: 0, max: 1)
            MLX.eval(baseColor)
            log("  [5] tex decoded: \(texOut.count) voxels (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")
        } else {
            baseColor = MLXArray.zeros([shapeOut.count, 3]) + 0.5   // flat gray, geometry-only
            log("  [4-5] texture off — geometry-only (flat gray)")
        }

        // 6) mesh + texture bake → clean mesh
        var baked = try MeshBake.run(shapeFeats: shapeOut.feats, coords: shapeOut.coords, texBaseColor: baseColor,
                                     fineRes: 1024, remeshRes: 256, targetFaces: targetFaces, atlasSize: 1024, log: log)

        // 7) optional Y-up reorientation (x,y,z)→(x,z,−y): a proper rotation, winding preserved.
        if yUp {
            func toYUp(_ a: MLXArray) -> MLXArray {
                MLX.stacked([a[0..., 0], a[0..., 2], -a[0..., 1]], axis: 1)
            }
            baked = BakedMesh(vertices: toYUp(baked.vertices), faces: baked.faces, normals: toYUp(baked.normals),
                              uvs: baked.uvs, texRGBA: baked.texRGBA, atlasSize: baked.atlasSize, coverage: baked.coverage)
            log("  [7] reoriented Y-up")
        }
        return baked
    }
}
