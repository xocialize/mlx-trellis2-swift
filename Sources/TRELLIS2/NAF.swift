import Foundation
import MLX

/// NAF (Neighborhood Attention Filtering, valeoai/NAF @ 37f2dfc, Apache-2.0) — the image-guided
/// feature upsampler Pixal3D runs in its shape and texture conditioners (AB-T-0485, AB-R-0859 §4).
/// About 0.66 M parameters. Weights: `naf.safetensors` (torch keys, converted from `naf_release.pth`).
///
/// forward(guide [1,3,S,S] in [0,1], features [1,C,L,L], target O):
///   image encoder = concat(encoder(guide), sem_encoder(guide))            → [O·f, O·f, 256]
///     encoder     = conv1x1(3→128, reflect) + 2 × EncBlock(k=1)
///     sem_encoder = conv3x3(3→128, reflect) + 2 × EncBlock(k=3, reflect)
///     EncBlock    = GN(8)→SiLU→conv→GN(8)→SiLU→conv   (no residual, no shortcut: in == out)
///   adaptive average pool to O×O (integer factor here), then axial RoPE (4 heads × 64, base 100)
///   queries = that map; keys = its average pool to L×L; values = the LR features (4 heads × C/4)
///   dilated neighbourhood attention, kernel 9, dilation O/L (natten `na2d`): the HR pixel (i, j)
///   attends the 9×9 LR cells around (i/d, j/d), the window shifted inward at the borders
///   (start = clamp(p − 4, 0, L − 9)), scale = 64^−½.
/// Output [O, O, C] (NHWC, batch dropped). The input-downsample branch of upstream's ImageEncoder
/// (S > 4·O) never fires in Pixal3D (S ∈ {512, 1024}, O ∈ {512, 1024}) and is not ported.
public struct NAF {
    struct EncBlock {
        let n1w, n1b, c1w, c1b, n2w, n2b, c2w, c2b: MLXArray
        let k: Int
    }
    struct Encoder {
        let c0w, c0b: MLXArray
        let k: Int
        let blocks: [EncBlock]
    }
    let encoder: Encoder
    let semEncoder: Encoder
    let periods: MLXArray            // [16]
    public let dim = 256, heads = 4, kernel = 9

    public init(weights w: [String: MLXArray]) {
        func g(_ k: String) -> MLXArray { w[k]!.asType(.float32) }
        // torch conv [Cout, Cin, kh, kw] → MLX conv2d [Cout, kh, kw, Cin]
        func conv(_ k: String) -> MLXArray { g(k).transposed(0, 2, 3, 1) }
        func enc(_ p: String, k: Int) -> Encoder {
            Encoder(c0w: conv(p + ".0.weight"), c0b: g(p + ".0.bias"), k: k, blocks: (1...2).map { i in
                let b = "\(p).\(i)."
                return EncBlock(n1w: g(b + "norm1.weight"), n1b: g(b + "norm1.bias"),
                                c1w: conv(b + "conv1.weight"), c1b: g(b + "conv1.bias"),
                                n2w: g(b + "norm2.weight"), n2b: g(b + "norm2.bias"),
                                c2w: conv(b + "conv2.weight"), c2b: g(b + "conv2.bias"), k: k)
            })
        }
        encoder = enc("image_encoder.encoder", k: 1)
        semEncoder = enc("image_encoder.sem_encoder", k: 3)
        periods = g("image_encoder.rope.periods")
    }

    // MARK: layers (NHWC, batch 1)

    /// torch `padding_mode="reflect"` for a k×k conv (pad k/2 on each side, edge not repeated).
    static func reflectPad(_ x: MLXArray, _ p: Int) -> MLXArray {
        guard p > 0 else { return x }
        let H = x.dim(1), W = x.dim(2)
        let top = (1...p).reversed().map { x[0..., $0 ..< ($0 + 1)] }
        let bottom = (1...p).map { x[0..., (H - 1 - $0) ..< (H - $0)] }
        let xr = MLX.concatenated(top + [x] + bottom, axis: 1)
        let left = (1...p).reversed().map { xr[0..., 0..., $0 ..< ($0 + 1)] }
        let right = (1...p).map { xr[0..., 0..., (W - 1 - $0) ..< (W - $0)] }
        return MLX.concatenated(left + [xr] + right, axis: 2)
    }

    static func conv(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray, k: Int) -> MLXArray {
        MLX.conv2d(reflectPad(x, k / 2), w) + b
    }

    /// nn.GroupNorm(8, C, eps=1e-5) on NHWC.
    static func groupNorm(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray, groups: Int = 8) -> MLXArray {
        let H = x.dim(1), W = x.dim(2), C = x.dim(3)
        let g = x.reshaped([1, H, W, groups, C / groups])
        let mean = g.mean(axes: [1, 2, 4], keepDims: true)
        let d = g - mean
        let v = (d * d).mean(axes: [1, 2, 4], keepDims: true)
        return (d * MLX.rsqrt(v + 1e-5)).reshaped([1, H, W, C]) * w + b
    }

    static func silu(_ x: MLXArray) -> MLXArray { x * MLX.sigmoid(x) }

    func run(_ e: Encoder, _ x: MLXArray) -> MLXArray {
        var h = Self.conv(x, e.c0w, e.c0b, k: e.k)
        for b in e.blocks {
            var y = Self.silu(Self.groupNorm(h, b.n1w, b.n1b))
            y = Self.conv(y, b.c1w, b.c1b, k: b.k)
            y = Self.silu(Self.groupNorm(y, b.n2w, b.n2b))
            h = Self.conv(y, b.c2w, b.c2b, k: b.k)
        }
        return h
    }

    /// F.adaptive_avg_pool2d to `o` for an integer down-factor (NHWC).
    static func avgPool(_ x: MLXArray, to o: Int) -> MLXArray {
        let H = x.dim(1), C = x.dim(3)
        if H == o { return x }
        precondition(H % o == 0, "NAF: non-integer pool factor \(H)→\(o)")
        let f = H / o
        return x.reshaped([1, o, f, o, f, C]).mean(axes: [2, 4])
    }

    /// DINOv3-style axial RoPE (NAF `RoPE`, normalize "separate", eval mode) on [1,O,O,256].
    func rope(_ x: MLXArray) -> MLXArray {
        let O = x.dim(1)
        let dHead = dim / heads                                     // 64
        let coord = (MLXArray(Array(0..<O).map { Float($0) }) + 0.5) / Float(O) * 2 - 1   // [O]
        let twoPi = Float(2 * Double.pi)
        let ah = twoPi * coord.reshaped([O, 1]) / periods.reshaped([1, -1])     // [O,16] (rows)
        let aw = twoPi * coord.reshaped([O, 1]) / periods.reshaped([1, -1])     // [O,16] (cols)
        let q = periods.dim(0)
        // per pixel (i,j): [ah[i] (16), aw[j] (16)] tiled ×2 → 64
        let angH = MLX.broadcast(ah.reshaped([O, 1, q]), to: [O, O, q])
        let angW = MLX.broadcast(aw.reshaped([1, O, q]), to: [O, O, q])
        let ang = MLX.concatenated([angH, angW, angH, angW], axis: 2)          // [O,O,64]
        let cos = MLX.cos(ang).reshaped([1, O, O, 1, dHead])
        let sin = MLX.sin(ang).reshaped([1, O, O, 1, dHead])
        let xh = x.reshaped([1, O, O, heads, dHead])
        let half = dHead / 2
        let rot = MLX.concatenated([-xh[.ellipsis, half...], xh[.ellipsis, ..<half]], axis: -1)
        return (xh * cos + rot * sin).reshaped([1, O, O, dim])
    }

    /// The query map: image encoder → pool to `o` → RoPE. guide is NCHW [1,3,S,S]. Returns [1,o,o,256].
    public func queries(guide: MLXArray, target o: Int) -> MLXArray {
        let S = guide.dim(2)
        precondition(S <= 4 * o, "NAF: guide \(S) > 4·target \(o) needs the unported downsample branch")
        let x = guide.asType(.float32).transposed(0, 2, 3, 1)             // NHWC
        let e = MLX.concatenated([run(encoder, x), run(semEncoder, x)], axis: 3)   // [1,S,S,256]
        return rope(Self.avgPool(e, to: o))
    }

    /// Full NAF forward. `features` NCHW [1,C,L,L]. Returns the HR map [O, O, C] (fp32), computed in
    /// bands of LR rows to bound the transient (the attention logits are [L²,heads,d²,81]).
    public func callAsFunction(guide: MLXArray, features: MLXArray, target o: Int,
                               bandRows: Int = 4) -> MLXArray {
        let q = queries(guide: guide, target: o)                            // [1,o,o,256]
        MLX.eval(q)
        return attend(queries: q, features: features, bandRows: bandRows)
    }

    /// The neighbourhood-attention half of the forward, given the query map.
    public func attend(queries q: MLXArray, features: MLXArray, bandRows: Int = 4) -> MLXArray {
        let o = q.dim(1)
        let C = features.dim(1), L = features.dim(2)
        precondition(features.dim(3) == L && o % L == 0 && L >= kernel, "NAF: shapes o=\(o) L=\(L)")
        let d = o / L
        let dh = dim / heads, dv = C / heads
        let k = Self.avgPool(q, to: L).reshaped([L, L, heads, dh])          // keys at LR
        let v = features.asType(.float32)[0].transposed(1, 2, 0).reshaped([L, L, heads, dv])
        let half = kernel / 2
        // window start per LR index (NATTEN border rule)
        let starts = (0..<L).map { min(max($0 - half, 0), L - kernel) }
        let scale = Float(1.0 / Double(dh).squareRoot())
        var bands = [MLXArray]()
        var p0 = 0
        while p0 < L {
            let nb = min(bandRows, L - p0)
            // rows of the 9×9 windows for LR rows p0..<p0+nb: [nb, 9]
            var rIdx = [Int32](); rIdx.reserveCapacity(nb * kernel)
            for p in p0 ..< (p0 + nb) { for a in 0..<kernel { rIdx.append(Int32(starts[p] + a)) } }
            var cIdx = [Int32](); cIdx.reserveCapacity(L * kernel)
            for qc in 0..<L { for b in 0..<kernel { cIdx.append(Int32(starts[qc] + b)) } }
            let rI = MLXArray(rIdx), cI = MLXArray(cIdx)
            // gather windows: [nb*9, L*9, heads, ch] → [nb, 9, L, 9, heads, ch] → [nb, L, heads, 81, ch]
            func windows(_ t: MLXArray, ch: Int) -> MLXArray {
                let rows = t.take(rI, axis: 0)                               // [nb*9, L, heads, ch]
                let both = rows.take(cI, axis: 1)                            // [nb*9, L*9, heads, ch]
                return both.reshaped([nb, kernel, L, kernel, heads, ch])
                    .transposed(0, 2, 4, 1, 3, 5).reshaped([nb, L, heads, kernel * kernel, ch])
            }
            let kw = windows(k, ch: dh)                                      // [nb, L, h, 81, dh]
            let vw = windows(v, ch: dv)                                      // [nb, L, h, 81, dv]
            // queries of the HR pixels in these LR rows: [nb*d, o, h, dh] → [nb, L, h, d², dh]
            let qb = q[0, (p0 * d) ..< ((p0 + nb) * d)].reshaped([nb, d, L, d, heads, dh])
                .transposed(0, 2, 4, 1, 3, 5).reshaped([nb, L, heads, d * d, dh])
            let logits = matmul(qb, kw.transposed(0, 1, 2, 4, 3)) * scale    // [nb, L, h, d², 81]
            let attn = MLX.softmax(logits, axis: -1, precise: true)
            let out = matmul(attn, vw)                                       // [nb, L, h, d², dv]
            // → [nb, d, L, d, h, dv] → [nb*d, o, C]
            let band = out.reshaped([nb, L, heads, d, d, dv]).transposed(0, 3, 1, 4, 2, 5)
                .reshaped([nb * d, o, C])
            MLX.eval(band)
            bands.append(band)
            p0 += nb
        }
        return MLX.concatenated(bands, axis: 0)                              // [o, o, C]
    }
}
