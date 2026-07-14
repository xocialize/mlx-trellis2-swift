import Foundation
import MLX

/// Precomputed 27-neighbor gather table for a fixed sparse coord set, so all
/// submanifold convs sharing that coord set build the map ONCE. table[t,k] is the
/// row of the k-th kernel neighbor of voxel t (or `count` = zero-sink row).
/// Kernel offset order (kd,kh,kw) on coord axes (x,y,z) matches conv_torch / SW2.
public struct NeighborMap27 {
    public let table: MLXArray   // [T,27] Int32
    public let count: Int
    public init(_ coords: MLXArray) {
        let T = coords.dim(0)
        let c = coords.asType(.int32).asArray(Int32.self)
        var b = [Int](repeating: 0, count: T), px = b, py = b, pz = b
        var maxc = 0
        for t in 0..<T {
            b[t] = Int(c[t*4]); px[t] = Int(c[t*4+1]); py[t] = Int(c[t*4+2]); pz[t] = Int(c[t*4+3])
            maxc = max(maxc, max(px[t], max(py[t], pz[t])))
        }
        let R = Int64(maxc + 3)
        @inline(__always) func key(_ bb: Int, _ x: Int, _ y: Int, _ z: Int) -> Int64 {
            ((Int64(bb) &* R &+ Int64(x)) &* R &+ Int64(y)) &* R &+ Int64(z)
        }
        var loc = [Int64: Int32](minimumCapacity: T * 2)
        for t in 0..<T { loc[key(b[t], px[t], py[t], pz[t])] = Int32(t) }
        var tbl = [Int32](repeating: Int32(T), count: T * 27)
        var k = 0
        for kd in 0..<3 { for kh in 0..<3 { for kw in 0..<3 {
            let ox = kd - 1, oy = kh - 1, oz = kw - 1
            for t in 0..<T {
                if let s = loc[key(b[t], px[t] + ox, py[t] + oy, pz[t] + oz)] { tbl[t*27 + k] = s }
            }
            k += 1
        }}}
        self.count = T
        self.table = MLXArray(tbl).reshaped([T, 27])
    }
}

public enum SparseOps {
    public static func silu(_ x: MLXArray) -> MLXArray { x * MLX.sigmoid(x) }

    /// SparseLinear: feats @ W^T + b.  W is [out, in].
    public static func linear(_ feats: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
        matmul(feats, w.transposed()) + b
    }

    /// Submanifold conv via a precomputed neighbor map. weight [Co,3,3,3,Ci].
    public static func conv(_ feats: MLXArray, _ weight: MLXArray, _ bias: MLXArray, _ map: NeighborMap27) -> MLXArray {
        let Co = weight.dim(0), Ci = weight.dim(4)
        let featsPad = MLX.concatenated([feats, MLXArray.zeros([1, Ci], dtype: feats.dtype)], axis: 0)
        let weightR = weight.reshaped([Co, 27, Ci])
        var out = MLXArray.zeros([map.count, Co], dtype: feats.dtype)
        for k in 0..<27 {
            let gathered = featsPad.take(map.table[0..., k], axis: 0)       // [T,Ci]
            out = out + matmul(gathered, weightR[0..., k, 0...].reshaped([Co, Ci]).transposed())
        }
        return out + bias
    }

    /// repeat_interleave along channel dim: [M,K] -> [M,K*rep] with each channel repeated contiguously.
    public static func repeatInterleave(_ x: MLXArray, _ rep: Int) -> MLXArray {
        let M = x.dim(0), K = x.dim(1)
        return MLX.broadcast(x.reshaped([M, K, 1]), to: [M, K, rep]).reshaped([M, K * rep])
    }
}

/// SparseChannel2Spatial (factor 2) upsample driven by a predicted subdivision mask.
/// Topology (which of 8 children of each parent are active, and the gather rows)
/// depends only on the mask, so it is built once and applied to both h and x.
public struct C2STopology {
    public let rows: MLXArray     // [M] Int32 — index into the [N*8, C/8] reshaped feats
    public let newCoords: MLXArray // [M,4] Int32
    public init(coords: MLXArray, subdivLogits: MLXArray) {
        let N = coords.dim(0)
        let c = coords.asType(.int32).asArray(Int32.self)
        let logits = subdivLogits.asType(.float32).asArray(Float.self)  // [N,8] row-major
        var rowsA = [Int32](); rowsA.reserveCapacity(N * 3)
        var coordsA = [Int32](); coordsA.reserveCapacity(N * 12)
        for n in 0..<N {
            let b = c[n*4], x = c[n*4+1] * 2, y = c[n*4+2] * 2, z = c[n*4+3] * 2
            for s in 0..<8 where logits[n*8 + s] > 0 {
                rowsA.append(Int32(n * 8 + s))
                coordsA.append(b)
                coordsA.append(x + Int32(s & 1))
                coordsA.append(y + Int32((s >> 1) & 1))
                coordsA.append(z + Int32((s >> 2) & 1))
            }
        }
        let M = rowsA.count
        self.rows = MLXArray(rowsA)
        self.newCoords = MLXArray(coordsA).reshaped([M, 4])
    }
    /// gather child features from feats [N, C] (C divisible by 8) -> [M, C/8].
    public func apply(_ feats: MLXArray) -> MLXArray {
        let N = feats.dim(0), C = feats.dim(1)
        return feats.reshaped([N * 8, C / 8]).take(rows, axis: 0)
    }
}

/// FlexiDualGridVaeDecoder neural body (SparseUnetVaeDecoder): SLat[T,32] ->
/// raw 7-channel dual-grid features on the subdivided coord set (pre o_voxel mesh).
/// Levels: model_channels [1024,512,256,128,64], num_blocks [4,16,8,4,0],
/// ConvNeXt bodies + C2S up-blocks (with learned subdivision).
public final class ShapeSlatDecoder {
    let w: [String: MLXArray]
    // (channels, numBlocks, upOutChannels?) per level; up nil on last.
    let levels: [(ch: Int, n: Int, up: Int?)] = [
        (1024, 4, 512), (512, 16, 256), (256, 8, 128), (128, 4, 64), (64, 0, nil),
    ]

    public init(weights: [String: MLXArray]) {
        var m = [String: MLXArray]()
        for (k, v) in weights { m[k] = v.asType(.float32) }
        self.w = m
    }

    private func convNeXt(_ x: SparseTensor, _ p: String, _ map: NeighborMap27) -> SparseTensor {
        var h = SparseOps.conv(x.feats, w["\(p).conv.weight"]!, w["\(p).conv.bias"]!, map)
        h = layerNorm32(h, weight: w["\(p).norm.weight"], bias: w["\(p).norm.bias"])
        h = SparseOps.linear(h, w["\(p).mlp.0.weight"]!, w["\(p).mlp.0.bias"]!)
        h = SparseOps.silu(h)
        h = SparseOps.linear(h, w["\(p).mlp.2.weight"]!, w["\(p).mlp.2.bias"]!)
        return x.replace(h + x.feats)
    }

    /// C2S ResBlock: predicts subdivision, upsamples h and x, convs on the finer grid.
    /// `guidedMask` (Int32 [N,8], active != 0) overrides the learned subdivision when provided.
    private func upBlock(_ x: SparseTensor, _ p: String, ch: Int, out: Int, mapCur: NeighborMap27, guidedMask: MLXArray? = nil)
        -> (SparseTensor, NeighborMap27, MLXArray) {
        let subdiv = guidedMask ?? SparseOps.linear(x.feats, w["\(p).to_subdiv.weight"]!, w["\(p).to_subdiv.bias"]!)
        let mask = (subdiv .> 0).asType(.int32)     // subdivision decisions used (for tex-decoder guidance)
        var h = SparseOps.silu(layerNorm32(x.feats, weight: w["\(p).norm1.weight"], bias: w["\(p).norm1.bias"]))
        h = SparseOps.conv(h, w["\(p).conv1.weight"]!, w["\(p).conv1.bias"]!, mapCur)   // [N, out*8]
        let topo = C2STopology(coords: x.coords, subdivLogits: subdiv)
        let hC = topo.apply(h)                 // [M, out]
        let xC = topo.apply(x.feats)           // [M, ch/8]
        let mapNew = NeighborMap27(topo.newCoords)
        var h2 = SparseOps.silu(layerNorm32(hC))   // norm2 non-affine
        h2 = SparseOps.conv(h2, w["\(p).conv2.weight"]!, w["\(p).conv2.bias"]!, mapNew)
        let skip = SparseOps.repeatInterleave(xC, out / (ch / 8))
        return (SparseTensor(feats: h2 + skip, coords: topo.newCoords), mapNew, mask)
    }

    /// Diagnostic: level-0 stages (coords unchanged) for parity localization.
    public func debugLevel0(_ x: SparseTensor) -> (fromLatent: MLXArray, convnext: MLXArray, subdiv: MLXArray) {
        var h = x.replace(SparseOps.linear(x.feats, w["from_latent.weight"]!, w["from_latent.bias"]!))
        let fl = h.feats
        let map = NeighborMap27(h.coords)
        for j in 0..<4 { h = convNeXt(h, "blocks.0.\(j)", map) }
        let cn = h.feats
        let subdiv = SparseOps.linear(h.feats, w["blocks.0.4.to_subdiv.weight"]!, w["blocks.0.4.to_subdiv.bias"]!)
        return (fl, cn, subdiv)
    }

    /// `guidedMasks` (one Int32 [N,8] per up-level) forces the oracle's subdivision
    /// decisions — used to prove exact topology parity modulo fp subdivision ties.
    public func callAsFunction(_ x: SparseTensor, guidedMasks: [MLXArray]? = nil) -> SparseTensor {
        var h = x.replace(SparseOps.linear(x.feats, w["from_latent.weight"]!, w["from_latent.bias"]!))
        var map = NeighborMap27(h.coords)
        for (i, lvl) in levels.enumerated() {
            for j in 0..<lvl.n {
                h = convNeXt(h, "blocks.\(i).\(j)", map)
                MLX.eval(h.feats)
            }
            if let out = lvl.up {
                (h, map, _) = upBlock(h, "blocks.\(i).\(lvl.n)", ch: lvl.ch, out: out, mapCur: map, guidedMask: guidedMasks?[i])
                MLX.eval(h.feats, h.coords)
            }
        }
        var o = layerNorm32(h.feats, eps: 1e-5)                             // final non-affine LN (F.layer_norm default eps)
        o = SparseOps.linear(o, w["output_layer.weight"]!, w["output_layer.bias"]!)
        return h.replace(o)
    }

    /// Free-run decode that also returns the per-level subdivision masks it chose —
    /// feed these as the tex decoder's `guidedMasks` so shape+tex share coords.
    public func decodeCapturingSubs(_ x: SparseTensor) -> (out: SparseTensor, subs: [MLXArray]) {
        var h = x.replace(SparseOps.linear(x.feats, w["from_latent.weight"]!, w["from_latent.bias"]!))
        var map = NeighborMap27(h.coords)
        var subs = [MLXArray]()
        for (i, lvl) in levels.enumerated() {
            for j in 0..<lvl.n {
                h = convNeXt(h, "blocks.\(i).\(j)", map); MLX.eval(h.feats)
            }
            if let out = lvl.up {
                let (hh, mm, mask) = upBlock(h, "blocks.\(i).\(lvl.n)", ch: lvl.ch, out: out, mapCur: map)
                h = hh; map = mm; subs.append(mask)
                MLX.eval(h.feats, h.coords, mask)
            }
        }
        var o = layerNorm32(h.feats, eps: 1e-5)
        o = SparseOps.linear(o, w["output_layer.weight"]!, w["output_layer.bias"]!)
        return (h.replace(o), subs)
    }
}
