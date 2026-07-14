import Foundation
import MLX
import MLXFast

/// Multi-head cross-attention: queries from x, keys/values from context (the
/// DINOv3 cond tokens). QK-RMS-norm, NO RoPE. Matches trellis2 cross-attn.
/// Parity target: goldens/xattn_*.
public struct MultiHeadCrossAttention {
    public let toQW, toQB, toKvW, toKvB, qGamma, kGamma, toOutW, toOutB: MLXArray
    public let numHeads: Int, headDim: Int

    public init(toQW: MLXArray, toQB: MLXArray, toKvW: MLXArray, toKvB: MLXArray,
                qGamma: MLXArray, kGamma: MLXArray, toOutW: MLXArray, toOutB: MLXArray, numHeads: Int) {
        self.toQW = toQW; self.toQB = toQB; self.toKvW = toKvW; self.toKvB = toKvB
        self.qGamma = qGamma; self.kGamma = kGamma; self.toOutW = toOutW; self.toOutB = toOutB
        self.numHeads = numHeads; self.headDim = qGamma.dim(1)
    }

    private func qkNorm(_ x: MLXArray, _ gamma: MLXArray) -> MLXArray {
        let denom = MLX.sqrt((x * x).sum(axis: -1, keepDims: true))
        let xn = x / MLX.maximum(denom, MLXArray(Float(1e-12)))
        return xn * gamma.reshaped([1, 1, numHeads, headDim]) * Float(Double(headDim).squareRoot())
    }

    public func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let B = x.dim(0), L = x.dim(1), Lc = context.dim(1)
        var q = (matmul(x, toQW.transposed()) + toQB).reshaped([B, L, numHeads, headDim])
        let kv = (matmul(context, toKvW.transposed()) + toKvB).reshaped([B, Lc, 2, numHeads, headDim])
        var k = kv[0..., 0..., 0]
        let v = kv[0..., 0..., 1]
        q = qkNorm(q, qGamma); k = qkNorm(k, kGamma)
        let qt = q.transposed(0, 2, 1, 3), kt = k.transposed(0, 2, 1, 3), vt = v.transposed(0, 2, 1, 3)
        let scale = Float(1.0 / Double(headDim).squareRoot())
        let fast = TRELLIS2Config.fastAttention
        let (qs, ks, vs) = fast ? (qt.asType(.bfloat16), kt.asType(.bfloat16), vt.asType(.bfloat16)) : (qt, kt, vt)
        let oRaw = MLXFast.scaledDotProductAttention(queries: qs, keys: ks, values: vs, scale: scale, mask: nil)
        let o = fast ? oRaw.asType(.float32) : oRaw
        let oo = o.transposed(0, 2, 1, 3).reshaped([B, L, numHeads * headDim])
        return matmul(oo, toOutW.transposed()) + toOutB
    }
}
