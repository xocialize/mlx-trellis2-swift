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

    public init(modulation: MLXArray, norm2W: MLXArray, norm2B: MLXArray,
                selfAttn: MultiHeadAttention, crossAttn: MultiHeadCrossAttention,
                mlp: FeedForwardNet, channels: Int) {
        self.modulation = modulation; self.norm2W = norm2W; self.norm2B = norm2B
        self.selfAttn = selfAttn; self.crossAttn = crossAttn; self.mlp = mlp; self.channels = channels
    }

    public func callAsFunction(_ x: MLXArray, mod: MLXArray, context: MLXArray, phases: MLXArray) -> MLXArray {
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
        h = crossAttn(h, context: context)
        out = out + h

        h = layerNorm32(out)                                        // norm3 (no affine)
        h = h * (1 + scaleMlp) + shiftMlp
        h = mlp(h)
        h = h * gateMlp
        return out + h
    }
}
