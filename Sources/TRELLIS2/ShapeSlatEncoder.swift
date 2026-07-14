import Foundation
import MLX

/// SparseSpatial2Channel (factor 2) downsample: coarsen coords by //2 and pack each
/// parent's ≤8 children into 8 channel-slots (missing children -> zero). Parent order
/// = lexicographic by (b,xp,yp,zp), matching torch.unique's sorted codes. Topology
/// depends only on coords, so `src`/`newCoords` build once and apply to any feats.
public struct S2CTopology {
    public let src: MLXArray       // [nParent*8] Int32 -> voxel row or N (zero sink)
    public let newCoords: MLXArray // [nParent,4] Int32
    public let nParent: Int
    let N: Int
    public init(coords: MLXArray) {
        let n = coords.dim(0)
        let c = coords.asType(.int32).asArray(Int32.self)
        var pb = [Int](repeating: 0, count: n), pxx = pb, pyy = pb, pzz = pb, sub = pb
        var maxp = 0
        for v in 0..<n {
            let b = Int(c[v*4]), x = Int(c[v*4+1]), y = Int(c[v*4+2]), z = Int(c[v*4+3])
            pb[v] = b; pxx[v] = x >> 1; pyy[v] = y >> 1; pzz[v] = z >> 1
            sub[v] = (x & 1) | ((y & 1) << 1) | ((z & 1) << 2)
            maxp = max(maxp, max(pxx[v], max(pyy[v], pzz[v])))
        }
        let R = Int64(maxp + 2)
        @inline(__always) func code(_ v: Int) -> Int64 {
            ((Int64(pb[v]) &* R &+ Int64(pxx[v])) &* R &+ Int64(pyy[v])) &* R &+ Int64(pzz[v])
        }
        var uniq = Set<Int64>(minimumCapacity: n)
        var codeV = [Int64](repeating: 0, count: n)
        for v in 0..<n { let cd = code(v); codeV[v] = cd; uniq.insert(cd) }
        let sorted = uniq.sorted()
        var idxOf = [Int64: Int32](minimumCapacity: sorted.count * 2)
        for (i, cd) in sorted.enumerated() { idxOf[cd] = Int32(i) }
        let np = sorted.count
        var srcA = [Int32](repeating: Int32(n), count: np * 8)
        for v in 0..<n { srcA[Int(idxOf[codeV[v]]!) * 8 + sub[v]] = Int32(v) }
        var coordsA = [Int32](); coordsA.reserveCapacity(np * 4)
        for cd in sorted {
            var e = cd
            let zp = Int32(e % R); e /= R
            let yp = Int32(e % R); e /= R
            let xp = Int32(e % R); e /= R
            coordsA.append(Int32(e)); coordsA.append(xp); coordsA.append(yp); coordsA.append(zp)
        }
        self.N = n; self.nParent = np
        self.src = MLXArray(srcA)
        self.newCoords = MLXArray(coordsA).reshaped([np, 4])
    }
    /// gather each parent's 8 child slots from feats [N,C] -> [nParent, C*8].
    public func apply(_ feats: MLXArray) -> MLXArray {
        let C = feats.dim(1)
        let featsPad = MLX.concatenated([feats, MLXArray.zeros([1, C], dtype: feats.dtype)], axis: 0)
        return featsPad.take(src, axis: 0).reshaped([nParent, C * 8])
    }
}

/// FlexiDualGridVaeEncoder (shape_enc): dual-grid (vertices+intersected) -> shape SLat.
/// input_layer(6->64) + 5 levels model_channels [64,128,256,512,1024] num_blocks
/// [0,4,8,16,4] with ConvNeXt bodies + S2C down-blocks; final non-affine LN + to_latent
/// (1024->64); z = mean = first 32 channels. Reuses SparseOps + NeighborMap27 + layerNorm32.
public final class ShapeSlatEncoder {
    let w: [String: MLXArray]
    let levels: [(ch: Int, n: Int, down: Int?)] = [
        (64, 0, 128), (128, 4, 256), (256, 8, 512), (512, 16, 1024), (1024, 4, nil),
    ]
    public init(weights: [String: MLXArray]) {
        var m = [String: MLXArray](); for (k, v) in weights { m[k] = v.asType(.float32) }
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

    /// S2C ResBlock: norm1->silu->conv1(ch->out/8)->S2C(h)&S2C(x)->norm2->silu->conv2 + group-mean skip.
    private func downBlock(_ x: SparseTensor, _ p: String, ch: Int, out: Int, mapCur: NeighborMap27)
        -> (SparseTensor, NeighborMap27) {
        var h = SparseOps.silu(layerNorm32(x.feats, weight: w["\(p).norm1.weight"], bias: w["\(p).norm1.bias"]))
        h = SparseOps.conv(h, w["\(p).conv1.weight"]!, w["\(p).conv1.bias"]!, mapCur)   // [N, out/8]
        let topo = S2CTopology(coords: x.coords)
        let hC = topo.apply(h)                 // [P, out]
        let xC = topo.apply(x.feats)           // [P, ch*8]
        let mapNew = NeighborMap27(topo.newCoords)
        var h2 = SparseOps.silu(layerNorm32(hC))   // norm2 non-affine
        h2 = SparseOps.conv(h2, w["\(p).conv2.weight"]!, w["\(p).conv2.bias"]!, mapNew)
        let g = (ch * 8) / out
        let skip = xC.reshaped([topo.nParent, out, g]).mean(axis: -1)   // group-mean skip
        return (SparseTensor(feats: h2 + skip, coords: topo.newCoords), mapNew)
    }

    /// vertices.feats [N,3] (∈[0,1]) + intersected.feats [N,3] (∈{0,1}) -> z [P,32].
    public func callAsFunction(_ vertices: SparseTensor, _ intersected: SparseTensor) -> SparseTensor {
        let x6 = MLX.concatenated([vertices.feats - 0.5, intersected.feats - 0.5], axis: 1)   // [N,6]
        var h = vertices.replace(SparseOps.linear(x6, w["input_layer.weight"]!, w["input_layer.bias"]!))
        var map = NeighborMap27(h.coords)
        for (i, lvl) in levels.enumerated() {
            for j in 0..<lvl.n {
                h = convNeXt(h, "blocks.\(i).\(j)", map); MLX.eval(h.feats)
            }
            if let out = lvl.down {
                (h, map) = downBlock(h, "blocks.\(i).\(lvl.n)", ch: lvl.ch, out: out, mapCur: map)
                MLX.eval(h.feats, h.coords)
            }
        }
        var o = layerNorm32(h.feats, eps: 1e-5)                          // final non-affine LN
        o = SparseOps.linear(o, w["to_latent.weight"]!, w["to_latent.bias"]!)   // [P,64]
        let z = o[0..., 0..<32]                                          // mean (chunk-2 first half)
        return h.replace(z)
    }
}
