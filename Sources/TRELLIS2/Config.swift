import Foundation
import MLX

/// SDPA cast precision for the attention modules. Both settings use the SAME fused
/// MLXFast.scaledDotProductAttention kernel — this only chooses the Q/K/V cast:
///   .fp32 — no cast (parity-safe; the CFG default: the g·vPos−(g−1)·vNeg near-
///           cancellation amplifies per-forward SDPA rounding noise into speckle)
///   .fp16 — 10-bit mantissa (~8× less rounding noise than bf16; range is safe here:
///           Q/K are L2-normalized per head ×√headDim, V is bounded activations)
///   .bf16 — 7-bit mantissa; the oracle's production dtype; known speckle risk under CFG
public enum SDPAPrecision: String, Sendable {
    case fp32, fp16, bf16
    /// The cast to apply before SDPA (nil = leave fp32 inputs uncast).
    public var castDtype: DType? {
        switch self {
        case .fp32: return nil
        case .fp16: return .float16
        case .bf16: return .bfloat16
        }
    }
}

/// Inference-time knobs. Defaults are the parity-safe (fp32) values; flip for speed.
public enum TRELLIS2Config {
    /// Active SDPA cast for the attention modules (nil = fp32, no cast).
    /// `Trellis2Pipeline.generate` sets this per stage (CFG flows vs CFG-free tex);
    /// the CLIs set it globally via the `fastAttention` shim below.
    public nonisolated(unsafe) static var sdpaCastDtype: DType? = nil

    /// Legacy bool view of `sdpaCastDtype`: true == bf16 SDPA (MLXFast flash kernel),
    /// false == fp32. The oracle runs attention in bf16 anyway, so bf16 matches
    /// production; leave OFF for the fp32 parity gates (CPU-fp32 goldens).
    public static var fastAttention: Bool {
        get { sdpaCastDtype == .bfloat16 }
        set { sdpaCastDtype = newValue ? .bfloat16 : nil }
    }
}
