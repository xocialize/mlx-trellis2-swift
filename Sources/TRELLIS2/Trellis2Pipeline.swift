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

    /// `ckptDir` = the TRELLIS.2-4B ckpts dir; `ssDecPath` = the TRELLIS-image-large SS decoder.
    public init(ckptDir: String, ssDecPath: String) throws {
        func load(_ f: String) throws -> [String: MLXArray] {
            try loadArrays(url: URL(fileURLWithPath: "\(ckptDir)/\(f).safetensors"))
        }
        ssDit = SparseStructureFlowModel(weights: try load("ss_flow_img_dit_1_3B_64_bf16"))
        ssDec = SparseStructureDecoder(weights: try loadArrays(url: URL(fileURLWithPath: ssDecPath)))
        shapeDit = SLatFlowModel(weights: try load("slat_flow_img2shape_dit_1_3B_512_bf16"))
        shapeDec = ShapeSlatDecoder(weights: try load("shape_dec_next_dc_f16c32_fp16"))
        texDit = SLatFlowModel(weights: try load("slat_flow_imgshape2tex_dit_1_3B_512_bf16"))
        texDec = ShapeSlatDecoder(weights: try load("tex_dec_next_dc_f16c32_fp16"))
    }

    /// cond/negCond = DINOv3 tokens [1,N,D]; ssNoise = [1,8,16,16,16]; ssPhases = SS-DiT
    /// rope phases for the 16³ grid. Returns a clean textured mesh.
    public func generate(cond: MLXArray, negCond: MLXArray, ssNoise: MLXArray, ssPhases: MLXArray,
                         seed: UInt64 = 0, log: (String) -> Void = { print($0) }) throws -> BakedMesh {
        // 1) sparse structure: SS sampler → z_s → decode → occupancy coords
        var t = Date()
        let zs = FlowEulerSampler.sampleSS(model: ssDit, noise: ssNoise, cond: cond, negCond: negCond, phases: ssPhases,
                                           steps: 12, guidanceStrength: 7.5, guidanceRescale: 0.7,
                                           guidanceInterval: (0.6, 1.0), rescaleT: 5.0)
        let coords = ssDec.coords(ssDec(zs)); MLX.eval(coords)
        log("  [1] sparse structure: \(coords.dim(0)) voxels (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 2) shape SLat sampler on those coords
        t = Date(); MLXRandom.seed(seed)
        let shapeSlat = FlowEulerSampler.sampleSLat(
            model: shapeDit, noiseFeats: MLXRandom.normal([coords.dim(0), 32]), coords: coords, cond: cond, negCond: negCond,
            guidanceStrength: 7.5, guidanceRescale: 0.5, guidanceInterval: (0.6, 1.0), rescaleT: 3.0)
        MLX.eval(shapeSlat)
        log("  [2] shape SLat sampled (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 3) shape decode → 7-ch dual-grid + the subdivision masks (for tex guidance)
        t = Date()
        let (shapeOut, subs) = shapeDec.decodeCapturingSubs(SparseTensor(feats: shapeSlat, coords: coords))
        MLX.eval(shapeOut.feats, shapeOut.coords)
        log("  [3] shape decoded: \(shapeOut.count) voxels (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 4) tex SLat sampler (concat_cond = shape_slat, no CFG)
        t = Date(); MLXRandom.seed(seed &+ 1)
        let texSlat = FlowEulerSampler.sampleSLat(
            model: texDit, noiseFeats: MLXRandom.normal([coords.dim(0), 32]), coords: coords, cond: cond, negCond: negCond,
            concatCond: shapeSlat, guidanceStrength: 1.0, guidanceRescale: 0.0, guidanceInterval: (0.6, 0.9), rescaleT: 3.0)
        MLX.eval(texSlat)
        log("  [4] tex SLat sampled (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 5) tex decode (guided by the shape's subdivisions → same coords) → base color
        t = Date()
        let texOut = texDec(SparseTensor(feats: texSlat, coords: coords), guidedMasks: subs)
        let baseColor = MLX.clip(texOut.feats[0..., 0..<3] * 0.5 + 0.5, min: 0, max: 1)
        MLX.eval(baseColor)
        log("  [5] tex decoded: \(texOut.count) voxels (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 6) mesh + texture bake → clean textured mesh
        return try MeshBake.run(shapeFeats: shapeOut.feats, coords: shapeOut.coords, texBaseColor: baseColor,
                                fineRes: 1024, remeshRes: 256, targetFaces: 120_000, atlasSize: 1024, log: log)
    }
}
