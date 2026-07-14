import Foundation
import MLX
import MLXFast

/// Multi-head self-attention with QK-RMS-norm + RoPE + SDPA — matches trellis2
/// `MultiHeadAttention` (full mode). Order: to_qkv → QK-RMS-norm → RoPE → SDPA → to_out.
/// QK "RMS norm" is actually L2-normalize per head-dim × learned per-head gamma × √headDim
/// (no mean, no eps beyond the normalize floor). Parity target: goldens/attn_*.
public struct MultiHeadAttention {
    public let toQkvW, toQkvB, qGamma, kGamma, toOutW, toOutB: MLXArray
    public let numHeads: Int, headDim: Int

    public init(toQkvW: MLXArray, toQkvB: MLXArray, qGamma: MLXArray, kGamma: MLXArray,
                toOutW: MLXArray, toOutB: MLXArray, numHeads: Int) {
        self.toQkvW = toQkvW; self.toQkvB = toQkvB
        self.qGamma = qGamma; self.kGamma = kGamma
        self.toOutW = toOutW; self.toOutB = toOutB
        self.numHeads = numHeads
        self.headDim = qGamma.dim(1)
    }

    private func qkNorm(_ x: MLXArray, _ gamma: MLXArray) -> MLXArray {
        let denom = MLX.sqrt((x * x).sum(axis: -1, keepDims: true))
        let xn = x / MLX.maximum(denom, MLXArray(Float(1e-12)))
        return xn * gamma.reshaped([1, 1, numHeads, headDim]) * Float(Double(headDim).squareRoot())
    }

    public func callAsFunction(_ h: MLXArray, phases: MLXArray) -> MLXArray {
        let B = h.dim(0), L = h.dim(1)
        let qkv = matmul(h, toQkvW.transposed()) + toQkvB          // [B,L,3*H*Dh]
        let r = qkv.reshaped([B, L, 3, numHeads, headDim])
        var q = r[0..., 0..., 0]                                    // [B,L,H,Dh]
        var k = r[0..., 0..., 1]
        let v = r[0..., 0..., 2]
        q = qkNorm(q, qGamma); k = qkNorm(k, kGamma)
        q = RoPE.apply(q, phasesCosSin: phases); k = RoPE.apply(k, phasesCosSin: phases)
        // [B,L,H,Dh] -> [B,H,L,Dh] for SDPA
        let qt = q.transposed(0, 2, 1, 3), kt = k.transposed(0, 2, 1, 3), vt = v.transposed(0, 2, 1, 3)
        let scale = Float(1.0 / Double(headDim).squareRoot())
        let o = MLXFast.scaledDotProductAttention(queries: qt, keys: kt, values: vt, scale: scale, mask: nil)
        let oo = o.transposed(0, 2, 1, 3).reshaped([B, L, numHeads * headDim])
        return matmul(oo, toOutW.transposed()) + toOutB
    }
}
