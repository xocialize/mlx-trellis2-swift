import Foundation
import MLX

/// Coords-driven sparse RoPE phases — matches trellis2 SparseRotaryPositionEmbedder.
/// For voxel coords (x,y,z): angles = [x·freqs, y·freqs, z·freqs, 0(pad)] → headDim/2,
/// freqs[i] = 1/10000^(i/freqDim), freqDim = headDim/2/3. Returns [T, headDim/2, 2] (cos,sin).
public func sparseRopePhases(coords: MLXArray, headDim: Int = 128, dim: Int = 3) -> MLXArray {
    let T = coords.dim(0)
    let freqDim = headDim / 2 / dim
    let idx = MLXArray(Array(0..<freqDim).map { Float($0) })
    let freqs = MLX.exp(idx * Float(-log(10000.0) / Double(freqDim)))     // [freqDim]
    let xyz = coords[0..., 1..<(dim + 1)].asType(.float32)                 // [T, dim]
    let ang = xyz.reshaped([T, dim, 1]) * freqs.reshaped([1, 1, freqDim])  // [T, dim, freqDim]
    var angFlat = ang.reshaped([T, dim * freqDim])                         // [T, dim*freqDim]
    let pad = headDim / 2 - dim * freqDim
    if pad > 0 { angFlat = MLX.concatenated([angFlat, MLXArray.zeros([T, pad])], axis: 1) }
    return MLX.stacked([MLX.cos(angFlat), MLX.sin(angFlat)], axis: -1)     // [T, headDim/2, 2]
}

/// Sparse structured-latent DiT (SLatFlowModel). Same ModulatedTransformerCrossBlock
/// as the SS-DiT, operating on SparseTensor feats [T,C] (num_samples=1 → full attention
/// over the voxel tokens), with coords-driven RoPE. Parity: goldens/slat_*.
public struct SLatFlowModel {
    public let inputW, inputB, adaLNW, adaLNB, outW, outB: MLXArray
    public let tEmbedder: TimestepEmbedder
    public let blocks: [ModulatedTransformerCrossBlock]
    public let numHeads: Int

    public init(weights w: [String: MLXArray], numBlocks: Int = 30, numHeads: Int = 12) {
        func g(_ k: String) -> MLXArray { w[k]!.asType(.float32) }
        let inW = g("input_layer.weight"); let C = inW.dim(0)
        inputW = inW; inputB = g("input_layer.bias")
        adaLNW = g("adaLN_modulation.1.weight"); adaLNB = g("adaLN_modulation.1.bias")
        outW = g("out_layer.weight"); outB = g("out_layer.bias")
        tEmbedder = TimestepEmbedder(w0: g("t_embedder.mlp.0.weight"), b0: g("t_embedder.mlp.0.bias"),
                                     w2: g("t_embedder.mlp.2.weight"), b2: g("t_embedder.mlp.2.bias"))
        self.numHeads = numHeads
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

    /// `concatCond` (feats [T, Cc], same coords) is concatenated to x along the channel
    /// dim before input_layer — the tex SLat DiT's shape conditioning (sp.sparse_cat dim=-1).
    public func callAsFunction(_ x: SparseTensor, t: MLXArray, cond: MLXArray, concatCond: MLXArray? = nil) -> SparseTensor {
        let T = x.count
        let inFeats = concatCond == nil ? x.feats : MLX.concatenated([x.feats, concatCond!], axis: 1)
        var h = (matmul(inFeats, inputW.transposed()) + inputB).reshaped([1, T, -1])   // [1,T,C]
        var tEmb = tEmbedder(t)
        tEmb = matmul(tEmb * MLX.sigmoid(tEmb), adaLNW.transposed()) + adaLNB           // [1,6C]
        let phases = sparseRopePhases(coords: x.coords)
        for blk in blocks { h = blk(h, mod: tEmb, context: cond, phases: phases) }
        h = layerNorm32(h)
        let C = h.dim(2)
        let outFeats = matmul(h.reshaped([T, C]), outW.transposed()) + outB             // [T,Cout]
        return x.replace(outFeats)
    }
}
