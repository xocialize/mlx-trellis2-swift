import Foundation
import MLX

/// Dense 3D DiT for sparse-structure generation (SparseStructureFlowModel).
/// forward: patchify(16³ voxel grid → 4096 tokens) → input Linear → t_embedder
/// + shared adaLN(SiLU→Linear→9216) → 30× ModulatedTransformerCrossBlock →
/// final affine-less LayerNorm → out Linear → unpatchify. Parity: goldens/ssdit_*.
public struct SparseStructureFlowModel {
    public let inputW, inputB, adaLNW, adaLNB, outW, outB: MLXArray
    public let tEmbedder: TimestepEmbedder
    public let blocks: [ModulatedTransformerCrossBlock]
    public let resolution: Int
    public let numHeads: Int

    public init(weights w: [String: MLXArray], numBlocks: Int = 30, resolution: Int = 16, numHeads: Int = 12) {
        func g(_ k: String) -> MLXArray { w[k]!.asType(.float32) }
        let inW = g("input_layer.weight")
        let C = inW.dim(0)
        inputW = inW; inputB = g("input_layer.bias")
        adaLNW = g("adaLN_modulation.1.weight"); adaLNB = g("adaLN_modulation.1.bias")
        outW = g("out_layer.weight"); outB = g("out_layer.bias")
        tEmbedder = TimestepEmbedder(w0: g("t_embedder.mlp.0.weight"), b0: g("t_embedder.mlp.0.bias"),
                                     w2: g("t_embedder.mlp.2.weight"), b2: g("t_embedder.mlp.2.bias"))
        self.resolution = resolution; self.numHeads = numHeads
        blocks = (0..<numBlocks).map { i in
            let p = "blocks.\(i)."
            return ModulatedTransformerCrossBlock(
                modulation: g(p + "modulation"), norm2W: g(p + "norm2.weight"), norm2B: g(p + "norm2.bias"),
                selfAttn: MultiHeadAttention(
                    toQkvW: g(p + "self_attn.to_qkv.weight"), toQkvB: g(p + "self_attn.to_qkv.bias"),
                    qGamma: g(p + "self_attn.q_rms_norm.gamma"), kGamma: g(p + "self_attn.k_rms_norm.gamma"),
                    toOutW: g(p + "self_attn.to_out.weight"), toOutB: g(p + "self_attn.to_out.bias"), numHeads: numHeads),
                crossAttn: MultiHeadCrossAttention(
                    toQW: g(p + "cross_attn.to_q.weight"), toQB: g(p + "cross_attn.to_q.bias"),
                    toKvW: g(p + "cross_attn.to_kv.weight"), toKvB: g(p + "cross_attn.to_kv.bias"),
                    qGamma: g(p + "cross_attn.q_rms_norm.gamma"), kGamma: g(p + "cross_attn.k_rms_norm.gamma"),
                    toOutW: g(p + "cross_attn.to_out.weight"), toOutB: g(p + "cross_attn.to_out.bias"), numHeads: numHeads),
                mlp: FeedForwardNet(w0: g(p + "mlp.mlp.0.weight"), b0: g(p + "mlp.mlp.0.bias"),
                                    w2: g(p + "mlp.mlp.2.weight"), b2: g(p + "mlp.mlp.2.bias")),
                channels: C)
        }
    }

    /// x: [B, Cin, R, R, R]; t: [B]; cond: [B, N, D]; phases: [L, headDim/2, 2].
    public func callAsFunction(_ x: MLXArray, t: MLXArray, cond: MLXArray, phases: MLXArray) -> MLXArray {
        let B = x.dim(0), Cin = x.dim(1)
        var h = x.reshaped([B, Cin, -1]).transposed(0, 2, 1)          // patchify -> [B, L, Cin]
        h = matmul(h, inputW.transposed()) + inputB                    // [B, L, C]
        var tEmb = tEmbedder(t)                                         // [B, C]
        tEmb = matmul(tEmb * MLX.sigmoid(tEmb), adaLNW.transposed()) + adaLNB   // SiLU→Linear -> [B, 6C]
        for blk in blocks { h = blk(h, mod: tEmb, context: cond, phases: phases) }
        h = layerNorm32(h)                                             // final (no affine)
        h = matmul(h, outW.transposed()) + outB                        // [B, L, Cout]
        let R = resolution
        return h.transposed(0, 2, 1).reshaped([B, h.dim(2), R, R, R])  // unpatchify
    }
}
