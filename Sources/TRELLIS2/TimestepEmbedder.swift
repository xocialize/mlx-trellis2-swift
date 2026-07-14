import Foundation
import MLX

/// Sinusoidal timestep embedding + MLP — matches trellis2 TimestepEmbedder.
/// freq_dim=256, half=128, max_period=1e4; emb = cat([cos(args), sin(args)]);
/// then Linear(256→C) → SiLU → Linear(C→C). Input t is already 1000*t_norm.
public struct TimestepEmbedder {
    public let w0, b0, w2, b2: MLXArray
    public let freqDim: Int

    public init(w0: MLXArray, b0: MLXArray, w2: MLXArray, b2: MLXArray, freqDim: Int = 256) {
        self.w0 = w0; self.b0 = b0; self.w2 = w2; self.b2 = b2; self.freqDim = freqDim
    }

    private func sinusoid(_ t: MLXArray) -> MLXArray {
        let half = freqDim / 2
        let idx = MLXArray(Array(0..<half).map { Float($0) })        // [half]
        let freqs = MLX.exp(idx * Float(-log(10000.0) / Double(half)))
        let args = t.reshaped([-1, 1]) * freqs.reshaped([1, -1])     // [B, half]
        return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)  // [B, freqDim]
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        let e = sinusoid(t)
        var h = matmul(e, w0.transposed()) + b0
        h = h * MLX.sigmoid(h)   // SiLU
        return matmul(h, w2.transposed()) + b2
    }
}
