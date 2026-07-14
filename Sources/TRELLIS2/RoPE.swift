import Foundation
import MLX

/// Rotary position embedding, interleaved (even,odd) pairing — matches
/// trellis2 `RotaryPositionEmbedder.apply_rotary_embedding`. Complex multiply
/// done as real arithmetic (MLX has no complex type):
///   pair (xe, xo) treated as (real, imag); phase (cos, sin):
///     real' = xe*cos - xo*sin ;  imag' = xe*sin + xo*cos
/// Phases are shared across heads. Parity target: goldens/rope_*.
public enum RoPE {
    /// x: [B, L, H, headDim]; phasesCosSin: [L, headDim/2, 2] (cos, sin).
    public static func apply(_ x: MLXArray, phasesCosSin: MLXArray) -> MLXArray {
        let B = x.dim(0), L = x.dim(1), H = x.dim(2), D = x.dim(3)
        let half = D / 2
        let xr = x.reshaped([B, L, H, half, 2])
        let xe = xr[0..., 0..., 0..., 0..., 0]        // [B,L,H,half]
        let xo = xr[0..., 0..., 0..., 0..., 1]
        let cos = phasesCosSin[0..., 0..., 0].reshaped([1, L, 1, half])
        let sin = phasesCosSin[0..., 0..., 1].reshaped([1, L, 1, half])
        let real = xe * cos - xo * sin
        let imag = xe * sin + xo * cos
        return MLX.stacked([real, imag], axis: -1).reshaped([B, L, H, D])
    }
}
