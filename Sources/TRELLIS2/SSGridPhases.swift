import Foundation
import MLX
import MLXRandom

/// In-package constants for the sparse-structure (SS) DiT stage that were previously loaded
/// from `.npy` goldens (`rope_phases_cossin`, `ss_noise`). Both are deterministic — the RoPE
/// grid is a fixed function of the 16³ voxel resolution, and the SS noise is a seeded randn —
/// so the package computes them rather than depending on a goldens directory (born-clean).
public enum SSGrid {

    /// The SS-DiT RoPE phase grid for a `resolution³` dense voxel grid, matching
    /// `RotaryPositionEmbedder(head_dim, dim=3, rope_freq=(1,10000))` applied to the ij-meshgrid
    /// coords `arange(res)³ → (res³, 3)`. Returns `[res³, headDim/2, 2]` (cos, sin) — identical
    /// to the oracle's `flow.rope_phases` viewed as real, and to the injected `rope_phases_cossin`
    /// golden the SS-DiT gate validated against (cosine 0.9998).
    ///
    /// freqDim = headDim/2/3; freqs[i] = 1/10000^(i/freqDim); per-axis angle = coord·freqs, the
    /// 3 axes concatenated then zero-padded to headDim/2. Token order is ij (x slowest, z fastest),
    /// matching the SS-DiT patchify flatten.
    public static func ropePhases(resolution: Int = 16, headDim: Int = 128) -> MLXArray {
        let dim = 3
        let R = resolution
        let T = R * R * R
        let freqDim = headDim / 2 / dim                      // 21
        // ij-meshgrid coords: (x,y,z) with x slowest, z fastest.
        var xyz = [Float](repeating: 0, count: T * dim)
        var t = 0
        for x in 0..<R { for y in 0..<R { for z in 0..<R {
            xyz[t*dim] = Float(x); xyz[t*dim + 1] = Float(y); xyz[t*dim + 2] = Float(z); t += 1
        } } }
        let coords = MLXArray(xyz, [T, dim])                 // [T,3]
        let idx = MLXArray(Array(0..<freqDim).map { Float($0) })
        let freqs = MLX.exp(idx * Float(-log(10000.0) / Double(freqDim)))     // [freqDim]
        let ang = coords.reshaped([T, dim, 1]) * freqs.reshaped([1, 1, freqDim])  // [T, dim, freqDim]
        var angFlat = ang.reshaped([T, dim * freqDim])                        // [T, dim*freqDim]
        let pad = headDim / 2 - dim * freqDim
        if pad > 0 { angFlat = MLX.concatenated([angFlat, MLXArray.zeros([T, pad])], axis: 1) }
        return MLX.stacked([MLX.cos(angFlat), MLX.sin(angFlat)], axis: -1)     // [T, headDim/2, 2]
    }

    /// The SS-DiT initial noise latent `[1, 8, res, res, res]` — a seeded standard normal. Any
    /// valid randn is a correct diffusion start (the oracle's captured `ss_noise` was one such
    /// draw; geometry can't be bit-identical across RNG backends anyway — see the port ledger), so
    /// the package generates it in-process for seed-reproducible output instead of vendoring it.
    public static func noise(resolution: Int = 16, channels: Int = 8, seed: UInt64 = 0) -> MLXArray {
        MLXRandom.seed(seed)
        return MLXRandom.normal([1, channels, resolution, resolution, resolution])
    }
}
