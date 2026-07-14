import Foundation
import MLX

/// Submanifold sparse 3D convolution (the TRELLIS.2 VAE-decoder workhorse).
/// Output lives on the SAME active voxels as input. Reformulated as pure gather
/// (no scatter): for each of the K³ kernel offsets, each output voxel gathers its
/// active neighbor's features (or a zero row if that neighbor is inactive),
/// matmuls with that offset's weight slice, and the offsets are summed.
///
/// Weight layout matches the checkpoint: [Co, Kd, Kh, Kw, Ci].
/// Parity target: goldens/spconv_* (matches trellis2 conv_torch, cosine ≥ 0.999).
public struct SubmanifoldConv3d {
    public let weight: MLXArray   // [Co, Kd, Kh, Kw, Ci]
    public let bias: MLXArray?
    public let kernel: (Int, Int, Int)

    public init(weight: MLXArray, bias: MLXArray?) {
        self.weight = weight
        self.bias = bias
        self.kernel = (weight.dim(1), weight.dim(2), weight.dim(3))
    }

    public func callAsFunction(_ x: SparseTensor) -> SparseTensor {
        let Co = weight.dim(0), Ci = weight.dim(4)
        let (Kd, Kh, Kw) = kernel
        let cd = (Kd - 1) / 2, ch = (Kh - 1) / 2, cw = (Kw - 1) / 2
        let T = x.count

        // --- host-side neighbor map from coords (b,x,y,z) ---
        let c = x.coords.asType(.int32).asArray(Int32.self)   // row-major [T,4]
        var b = [Int](repeating: 0, count: T)
        var px = [Int](repeating: 0, count: T)
        var py = [Int](repeating: 0, count: T)
        var pz = [Int](repeating: 0, count: T)
        var maxc = 0
        for t in 0..<T {
            b[t] = Int(c[t*4+0]); px[t] = Int(c[t*4+1]); py[t] = Int(c[t*4+2]); pz[t] = Int(c[t*4+3])
            maxc = max(maxc, max(px[t], max(py[t], pz[t])))
        }
        let R = maxc + max(cd, max(ch, cw)) + 2
        func key(_ bb: Int, _ xx: Int, _ yy: Int, _ zz: Int) -> Int64 {
            (((Int64(bb) * Int64(R) + Int64(xx)) * Int64(R) + Int64(yy)) * Int64(R) + Int64(zz))
        }
        var loc = [Int64: Int32](minimumCapacity: T * 2)
        for t in 0..<T { loc[key(b[t], px[t], py[t], pz[t])] = Int32(t) }

        // feats padded with a zero row at index T (the "no neighbor" sink)
        let featsPad = MLX.concatenated([x.feats, MLXArray.zeros([1, Ci], dtype: x.feats.dtype)], axis: 0)
        let weightR = weight.reshaped([Co, Kd * Kh * Kw, Ci])

        var out = MLXArray.zeros([T, Co], dtype: x.feats.dtype)
        var k = 0
        for kd in 0..<Kd { for kh in 0..<Kh { for kw in 0..<Kw {
            let ox = (kd - cd), oy = (kh - ch), oz = (kw - cw)
            var nbr = [Int32](repeating: Int32(T), count: T)   // default -> zero row
            for t in 0..<T {
                if let s = loc[key(b[t], px[t] + ox, py[t] + oy, pz[t] + oz)] { nbr[t] = s }
            }
            let gathered = featsPad.take(MLXArray(nbr), axis: 0)         // [T, Ci]
            let Wk = weightR[0..., k, 0...].reshaped([Co, Ci])           // [Co, Ci]
            out = out + matmul(gathered, Wk.transposed())               // [T, Co]
            k += 1
        }}}
        if let bias { out = out + bias }
        return x.replace(out)
    }
}
