import Foundation

/// Inference-time knobs. Defaults are the parity-safe (fp32) values; flip for speed.
public enum TRELLIS2Config {
    /// Run self/cross-attention SDPA in bf16 (MLXFast flash kernel) instead of fp32.
    /// The oracle runs attention in bf16 anyway, so this matches production; leave OFF
    /// for the fp32 parity gates (which compare against CPU-fp32 goldens).
    public nonisolated(unsafe) static var fastAttention = false
}
