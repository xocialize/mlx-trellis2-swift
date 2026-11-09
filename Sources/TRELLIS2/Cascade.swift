import Foundation
import MLX

/// Resolution tier — mirrors upstream `pipeline_type` ('512' / '1024_cascade' / '1536_cascade').
/// All tiers sample the sparse structure on a 32³ grid; the cascade tiers refine via the
/// LR(512-flow) → decoder-upsample → re-quantize → HR(1024-flow) ladder.
public enum Trellis2Tier: String, Sendable, CaseIterable {
    case res512, res1024, res1536

    /// The dual-grid resolution the HR (final) SLat targets. For cascade tiers the token
    /// back-off can lower the EFFECTIVE resolution at runtime (see `CascadeQuantize`).
    public var targetResolution: Int {
        switch self {
        case .res512: return 512
        case .res1024: return 1024
        case .res1536: return 1536
        }
    }
    public var isCascade: Bool { self != .res512 }

    /// The HR token ceiling a cascade tier is held to by default — the bound its declared footprint
    /// is measured at (AB-A-0262). Peak memory follows the HR token count, not the resolution
    /// (BENCHMARKS.md), so a token ceiling is a memory ceiling. res1024's keeps the whole t31 corpus
    /// at full 1024 (hair, the largest, is ~25k tokens there); res1536's holds the tier to ~72 GB,
    /// and an asset above it backs off in resolution the way upstream's 49,152 cap does (the dress
    /// lands at 1152 with 29,437 tokens). nil for res512 (no HR pass).
    public var hrTokenCeiling: Int? {
        switch self {
        case .res512: return nil
        case .res1024: return 28_672
        case .res1536: return 32_768
        }
    }
}

/// Host-side re-quantization of the shape decoder's upsampled coords onto the HR SLat grid,
/// with the upstream token back-off (`sample_shape_slat_cascade`, trellis2_image_to_3d.py:344-356):
///   quant = ((c + 0.5) / lrResolution · (hr/16)).int();  coords = unique(quant)
///   while tokens ≥ maxNumTokens and hr > 1024: hr -= 128
/// plus one departure from upstream: with `hardMaxTokens` set, a grid still at or above it at the
/// floor keeps backing off (down to `lrResolution`), so the token count — and with it the peak
/// memory — has a bound a scheduler can charge (AB-A-0262). Upstream stops at 1024 whatever the
/// count, which leaves res1024's working set unbounded by anything but the asset.
public enum CascadeQuantize {
    /// `upCoords` [M,4] (b,x,y,z) at the LR dual-grid resolution (512). Returns the unique
    /// quantized coords in torch.unique's lexicographic ascending order (exact-parity) and the
    /// (possibly backed-off) hr resolution actually chosen.
    public static func requantize(
        upCoords: MLXArray, lrResolution: Int, targetResolution: Int,
        maxNumTokens: Int = 49_152, floorResolution: Int = 1024, hardMaxTokens: Int? = nil
    ) -> (coords: MLXArray, hrResolution: Int) {
        let M = upCoords.dim(0)
        let c = upCoords.asType(.int32).asArray(Int32.self)
        let lr = Float(lrResolution)
        var hr = targetResolution
        while true {
            // fp32 op order matches torch: (c+0.5) / lr * (hr/16), truncate toward zero.
            let n = Float(hr / 16)
            var keys = Set<Int64>(minimumCapacity: M)
            for m in 0..<M {
                let b = Int64(c[m * 4])
                let x = Int64(Int32((Float(c[m * 4 + 1]) + 0.5) / lr * n))
                let y = Int64(Int32((Float(c[m * 4 + 2]) + 0.5) / lr * n))
                let z = Int64(Int32((Float(c[m * 4 + 3]) + 0.5) / lr * n))
                keys.insert(b << 48 | x << 32 | y << 16 | z)   // components < 2^16 ⇒ key sort == lex row sort
            }
            // `<=` (not `==`): a target not congruent to the floor mod 128 must
            // still stop at the floor instead of stepping past it toward hr <= 0.
            let underCap = keys.count < maxNumTokens
            let atFloor = hr <= floorResolution
            let overHardCap = hardMaxTokens.map { keys.count >= $0 } ?? false
            if underCap || (atFloor && (!overHardCap || hr - 128 < lrResolution)) {
                let sorted = keys.sorted()
                var out = [Int32](); out.reserveCapacity(sorted.count * 4)
                for k in sorted {
                    out.append(Int32(truncatingIfNeeded: k >> 48))
                    out.append(Int32(truncatingIfNeeded: (k >> 32) & 0xFFFF))
                    out.append(Int32(truncatingIfNeeded: (k >> 16) & 0xFFFF))
                    out.append(Int32(truncatingIfNeeded: k & 0xFFFF))
                }
                return (MLXArray(out).reshaped([sorted.count, 4]), hr)
            }
            hr -= 128
        }
    }
}
