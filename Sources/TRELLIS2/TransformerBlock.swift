import Foundation
import MLX

/// LayerNorm (fp32-cast, eps 1e-6) — matches trellis2 LayerNorm32. Optional affine.
func layerNorm32(_ x: MLXArray, weight: MLXArray? = nil, bias: MLXArray? = nil, eps: Float = 1e-6) -> MLXArray {
    let xf = x.asType(.float32)
    let mean = xf.mean(axis: -1, keepDims: true)
    let d = xf - mean
    let v = (d * d).mean(axis: -1, keepDims: true)
    var out = d / MLX.sqrt(v + eps)
    if let weight { out = out * weight }
    if let bias { out = out + bias }
    return out
}

/// GELU with tanh approximation (nn.GELU(approximate='tanh')).
func geluTanh(_ x: MLXArray) -> MLXArray {
    let c = Float(0.7978845608028654)   // sqrt(2/pi)
    let inner = c * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1 + MLX.tanh(inner))
}

/// FeedForwardNet: Linear -> GELU(tanh) -> Linear.
public struct FeedForwardNet {
    public let w0, b0, w2, b2: MLXArray
    public init(w0: MLXArray, b0: MLXArray, w2: MLXArray, b2: MLXArray) {
        self.w0 = w0; self.b0 = b0; self.w2 = w2; self.b2 = b2
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = geluTanh(matmul(x, w0.transposed()) + b0)
        return matmul(h, w2.transposed()) + b2
    }
}

/// ModulatedTransformerCrossBlock (share_mod=True): modulated self-attn -> plain
/// cross-attn -> modulated MLP. norm1/norm3 no-affine, norm2 affine. Matches trellis2.
/// Parity target: goldens/blk_*.
public struct ModulatedTransformerCrossBlock {
    public let modulation: MLXArray      // [6*C]
    public let norm2W, norm2B: MLXArray
    public let selfAttn: MultiHeadAttention
    public let crossAttn: MultiHeadCrossAttention
    public let mlp: FeedForwardNet
    public let channels: Int
    /// Pixal3D `proj_attention` (AB-T-0485): per-block `proj_linear` [C, P] + bias [C]. nil on the
    /// stock TRELLIS.2 flows. When set, the cross-attention output gets `proj_linear(proj)` added
    /// (`ProjectAttention`: `cross_attn_block(x, global) + proj_linear(proj)`).
    public let projW: MLXArray?
    public let projB: MLXArray?

    public init(modulation: MLXArray, norm2W: MLXArray, norm2B: MLXArray,
                selfAttn: MultiHeadAttention, crossAttn: MultiHeadCrossAttention,
                mlp: FeedForwardNet, channels: Int, projW: MLXArray? = nil, projB: MLXArray? = nil) {
        self.modulation = modulation; self.norm2W = norm2W; self.norm2B = norm2B
        self.selfAttn = selfAttn; self.crossAttn = crossAttn; self.mlp = mlp; self.channels = channels
        self.projW = projW; self.projB = projB
    }

    /// The projection-attention term for this block: `proj @ projW^T + projB` ([1,T,C]), or just
    /// the bias when `proj` is nil — Pixal3D's negative conditioning is zeros, so the
    /// unconditional branch adds only `proj_linear.bias`. nil on a stock block.
    func projTerm(_ proj: MLXArray?) -> MLXArray? {
        guard let projW, let projB else { return nil }
        guard let proj else { return projB.reshaped([1, 1, -1]) }
        return matmul(proj, projW.transposed()) + projB
    }

    /// The cross-attention sub-block on an already-normed input (Pixal3D `ProjectAttention`
    /// forward; stock: plain cross-attention). Exposed for the `proj_linear` block-I/O gate.
    public func crossTerm(_ normed: MLXArray, context: MLXArray, proj: MLXArray? = nil) -> MLXArray {
        let h = crossAttn(normed, context: context)
        if let p = projTerm(proj) { return h + p }
        return h
    }

    public func callAsFunction(_ x: MLXArray, mod: MLXArray, context: MLXArray, phases: MLXArray,
                               proj: MLXArray? = nil) -> MLXArray {
        let C = channels
        let parts = (modulation + mod).reshaped([1, 6, C])           // (self.modulation + mod).chunk(6)
        func p(_ i: Int) -> MLXArray { parts[0..., i].reshaped([1, 1, C]) }
        let shiftMsa = p(0), scaleMsa = p(1), gateMsa = p(2)
        let shiftMlp = p(3), scaleMlp = p(4), gateMlp = p(5)

        var h = layerNorm32(x)                                       // norm1 (no affine)
        h = h * (1 + scaleMsa) + shiftMsa
        h = selfAttn(h, phases: phases)
        h = h * gateMsa
        var out = x + h

        h = layerNorm32(out, weight: norm2W, bias: norm2B)          // norm2 (affine)
        h = crossTerm(h, context: context, proj: proj)              // + proj_linear(proj) (Pixal3D)
        out = out + h

        h = layerNorm32(out)                                        // norm3 (no affine)
        h = h * (1 + scaleMlp) + shiftMlp
        h = mlp(h)
        h = h * gateMlp
        return out + h
    }
}
