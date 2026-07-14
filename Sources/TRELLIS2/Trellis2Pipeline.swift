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

    /// Image conditioning: preprocessed pixels [1,3,512,512] → DINOv3 tokens.
    /// neg_cond is zeros (matches get_cond's `torch.zeros_like`).
    public func encodeImage(_ pixels: MLXArray) -> (cond: MLXArray, negCond: MLXArray) {
        let cond = dino(pixels)
        return (cond, MLXArray.zeros(cond.shape, dtype: cond.dtype))
    }

    /// cond/negCond = DINOv3 tokens [1,N,D]; ssNoise = [1,8,16,16,16]; ssPhases = SS-DiT
    /// rope phases for the 16³ grid. Returns a clean textured mesh.
    /// `shapeMean/Std`, `texMean/Std` = the pipeline's per-channel SLat normalization [32].
    /// The sampler emits a NORMALIZED latent (std≈1); the decoder needs it DENORMALIZED
    /// (× std + mean). The tex concat_cond, by contrast, wants the raw normalized shape_slat.
    public func generate(cond: MLXArray, negCond: MLXArray, ssNoise: MLXArray, ssPhases: MLXArray,
                         shapeMean: MLXArray, shapeStd: MLXArray, texMean: MLXArray, texStd: MLXArray,
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
            model: shapeDit, noiseFeats: MLXRandom.normal([coords.dim(0), 32]), coords: coords, cond: cond, negCond: negCond,
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

        // 4) tex SLat sampler (concat_cond = shape_slat, no CFG)
        t = Date(); MLXRandom.seed(seed &+ 1)
        let texSlat = FlowEulerSampler.sampleSLat(
            model: texDit, noiseFeats: MLXRandom.normal([coords.dim(0), 32]), coords: coords, cond: cond, negCond: negCond,
            concatCond: shapeSlat, guidanceStrength: 1.0, guidanceRescale: 0.0, guidanceInterval: (0.6, 0.9), rescaleT: 3.0)
        MLX.eval(texSlat)
        log("  [4] tex SLat sampled (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 5) tex decode (guided by the shape's subdivisions → same coords) → base color.
        //    Denormalize the tex latent before decoding.
        t = Date()
        let texSlatDenorm = texSlat * texStd + texMean
        let texOut = texDec(SparseTensor(feats: texSlatDenorm, coords: coords), guidedMasks: subs)
        let baseColor = MLX.clip(texOut.feats[0..., 0..<3] * 0.5 + 0.5, min: 0, max: 1)
        MLX.eval(baseColor)
        log("  [5] tex decoded: \(texOut.count) voxels (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 6) mesh + texture bake → clean textured mesh
        return try MeshBake.run(shapeFeats: shapeOut.feats, coords: shapeOut.coords, texBaseColor: baseColor,
                                fineRes: 1024, remeshRes: 256, targetFaces: 120_000, atlasSize: 1024, log: log)
    }
}
