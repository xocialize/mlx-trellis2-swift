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

    /// Keys and values for the attention (LR grid), and the per-row window starts.
    struct KV { let k: MLXArray, v: MLXArray, starts: [Int], L: Int, d: Int, C: Int }

    func kv(queries q: MLXArray, features: MLXArray) -> KV {
        let o = q.dim(1)
        let C = features.dim(1), L = features.dim(2)
        precondition(features.dim(3) == L && o % L == 0 && L >= kernel, "NAF: shapes o=\(o) L=\(L)")
        let dh = dim / heads, dv = C / heads
        let k = Self.avgPool(q, to: L).reshaped([L, L, heads, dh])          // keys at LR
        let v = features.asType(.float32)[0].transposed(1, 2, 0).reshaped([L, L, heads, dv])
        let half = kernel / 2
        let starts = (0..<L).map { min(max($0 - half, 0), L - kernel) }    // NATTEN border rule
        return KV(k: k, v: v, starts: starts, L: L, d: o / L, C: C)
    }

    /// HR rows [p0·d, (p0+nb)·d) of the output: [nb·d, o, C].
    func band(queries q: MLXArray, _ kv: KV, p0: Int, nb: Int) -> MLXArray {
        let o = q.dim(1), L = kv.L, d = kv.d, C = kv.C
        let dh = dim / heads, dv = C / heads
        var rIdx = [Int32](); rIdx.reserveCapacity(nb * kernel)
        for p in p0 ..< (p0 + nb) { for a in 0..<kernel { rIdx.append(Int32(kv.starts[p] + a)) } }
        var cIdx = [Int32](); cIdx.reserveCapacity(L * kernel)
        for qc in 0..<L { for b in 0..<kernel { cIdx.append(Int32(kv.starts[qc] + b)) } }
        let rI = MLXArray(rIdx), cI = MLXArray(cIdx)
        // gather windows: [nb*9, L*9, heads, ch] → [nb, 9, L, 9, heads, ch] → [nb, L, heads, 81, ch]
        func windows(_ t: MLXArray, ch: Int) -> MLXArray {
            t.take(rI, axis: 0).take(cI, axis: 1)
                .reshaped([nb, kernel, L, kernel, heads, ch])
                .transposed(0, 2, 4, 1, 3, 5).reshaped([nb, L, heads, kernel * kernel, ch])
        }
        let kw = windows(kv.k, ch: dh), vw = windows(kv.v, ch: dv)
        // queries of the HR pixels in these LR rows: [nb*d, o, h, dh] → [nb, L, h, d², dh]
        let qb = q[0, (p0 * d) ..< ((p0 + nb) * d)].reshaped([nb, d, L, d, heads, dh])
            .transposed(0, 2, 4, 1, 3, 5).reshaped([nb, L, heads, d * d, dh])
        let scale = Float(1.0 / Double(dh).squareRoot())
        let logits = matmul(qb, kw.transposed(0, 1, 2, 4, 3)) * scale        // [nb, L, h, d², 81]
        let attn = MLX.softmax(logits, axis: -1, precise: true)
        let out = matmul(attn, vw)                                           // [nb, L, h, d², dv]
        return out.reshaped([nb, L, heads, d, d, dv]).transposed(0, 3, 1, 4, 2, 5).reshaped([nb * d, o, C])
    }

    /// The full HR map [o, o, C] (the gates read crops of it). Peak ≈ 2× the map (bands + concat).
    public func attend(queries q: MLXArray, features: MLXArray, bandRows: Int = 4) -> MLXArray {
        let kv = kv(queries: q, features: features)
        var bands = [MLXArray]()
        var p0 = 0
        while p0 < kv.L {
            let nb = min(bandRows, kv.L - p0)
            let b = band(queries: q, kv, p0: p0, nb: nb); MLX.eval(b)
            bands.append(b)
            p0 += nb
        }
        return MLX.concatenated(bands, axis: 0)
    }

    /// Bilinear samples of the HR map at grid_sample points `norm` [K,2] (align_corners=False,
    /// border) WITHOUT materializing the map: each band of rows is computed, the corner pixels that
    /// fall in it are gathered, and the band is dropped. Same values as
    /// `Pixal3DProjection.sampleBilinear(attend(...), norm)`; peak is one band (~0.27 GB at o=1024)
    /// instead of the 4 GB tex map plus its concatenation copy.
    public func sample(queries q: MLXArray, features: MLXArray, at norm: MLXArray, bandRows: Int = 4) -> MLXArray {
        let o = q.dim(1)
        let kvs = kv(queries: q, features: features)
        let cw = Pixal3DProjection.bilinearCorners(norm, H: o, W: o)
        MLX.eval(cw.y0, cw.y1, cw.x0, cw.x1)
        let ys = [cw.y0, cw.y0, cw.y1, cw.y1].map { $0.asArray(Int32.self) }
        let xs = [cw.x0, cw.x1, cw.x0, cw.x1].map { $0.asArray(Int32.self) }
        let K = ys[0].count
        let rowsPerBand = bandRows * kvs.d
        // unique corner pixels, grouped by band
        var uniq = [Int32: Int32]()                          // pixel → position in the gathered table
        var perBand = [[Int32]](repeating: [], count: (o + rowsPerBand - 1) / rowsPerBand)
        for c in 0..<4 { for i in 0..<K {
            let pix = ys[c][i] * Int32(o) + xs[c][i]
            if uniq[pix] == nil { uniq[pix] = -1; perBand[Int(ys[c][i]) / rowsPerBand].append(pix) }
        } }
        var tables = [MLXArray]()
        var pos: Int32 = 0
        for (b, pixels) in perBand.enumerated() where !pixels.isEmpty {
            let p0 = b * bandRows, nb = min(bandRows, kvs.L - p0)
            let rowBase = Int32(p0 * kvs.d) * Int32(o)
            let bandMap = band(queries: q, kvs, p0: p0, nb: nb).reshaped([-1, kvs.C])
            let local = MLXArray(pixels.map { $0 - rowBase })
            let t = bandMap.take(local, axis: 0); MLX.eval(t)
            tables.append(t)
            for pix in pixels { uniq[pix] = pos; pos += 1 }
        }
        let table = MLX.concatenated(tables, axis: 0)       // [U, C]
        var acc: MLXArray? = nil
        for (c, w) in [cw.w00, cw.w01, cw.w10, cw.w11].enumerated() {
            let idx = MLXArray((0..<K).map { uniq[ys[c][$0] * Int32(o) + xs[c][$0]]! })
            let term = table.take(idx, axis: 0) * w.reshaped([-1, 1])
            acc = acc.map { $0 + term } ?? term
        }
        return acc!
    }
}
