import Foundation
import MLX

/// grid_sample_3d — trilinear/nearest sampling of a SPARSE voxel volume at query
/// points. The texture-bake op: sample the tex decoder's PBR voxels at each surface
/// position. Faithful port of trellis2/portable/grid_sample.py (coord hashmap = dict).
public enum GridSample3d {
    // 8 trilinear corner offsets (±0.5 per axis), matching the reference _CORNERS order.
    static let corners: [(Float, Float, Float)] = [
        (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5),
        (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5),
    ]

    /// feats [N,C], coords [N,4] (b,x,y,z), grid [B,L,3] query points (voxel space).
    /// Returns [B,L,C].
    public static func sample(feats: MLXArray, coords: MLXArray, grid: MLXArray, mode: String = "trilinear") -> MLXArray {
        let N = coords.dim(0), C = feats.dim(1)
        let B = grid.dim(0), L = grid.dim(1)
        let c = coords.asType(.int32).asArray(Int32.self)     // [N*4]
        let g = grid.asType(.float32).asArray(Float.self)     // [B*L*3]
        var maxc: Int32 = 0
        for i in 0..<N { maxc = max(maxc, max(c[i*4+1], max(c[i*4+2], c[i*4+3]))) }
        let R = Int64(maxc) + 2
        @inline(__always) func key(_ b: Int, _ x: Int, _ y: Int, _ z: Int) -> Int64 {
            ((Int64(b) &* R &+ Int64(x)) &* R &+ Int64(y)) &* R &+ Int64(z)
        }
        var idx = [Int64: Int32](minimumCapacity: N * 2)
        for i in 0..<N { idx[key(Int(c[i*4]), Int(c[i*4+1]), Int(c[i*4+2]), Int(c[i*4+3]))] = Int32(i) }

        let K = mode == "nearest" ? 1 : 8
        var rows = [Int32](repeating: Int32(N), count: B * L * K)   // gather rows (N = zero sink)
        var wts = [Float](repeating: 0, count: B * L * K)
        for b in 0..<B {
            for l in 0..<L {
                let base = (b * L + l) * 3
                let qx = g[base], qy = g[base+1], qz = g[base+2]
                let o = (b * L + l) * K
                if mode == "nearest" {
                    let ix = Int(qx.rounded()), iy = Int(qy.rounded()), iz = Int(qz.rounded())
                    if let j = idx[key(b, ix, iy, iz)] { rows[o] = j; wts[o] = 1 }
                } else {
                    for (k, cor) in corners.enumerated() {
                        let ix = Int(qx + cor.0), iy = Int(qy + cor.1), iz = Int(qz + cor.2)   // .int() truncation
                        if let j = idx[key(b, ix, iy, iz)] {
                            let dx = abs(Float(ix) + 0.5 - qx), dy = abs(Float(iy) + 0.5 - qy), dz = abs(Float(iz) + 0.5 - qz)
                            rows[o+k] = j
                            wts[o+k] = (1 - dx) * (1 - dy) * (1 - dz)
                        }
                    }
                }
            }
        }
        let featsPad = MLX.concatenated([feats.asType(.float32), MLXArray.zeros([1, C], dtype: .float32)], axis: 0)
        let gathered = featsPad.take(MLXArray(rows), axis: 0)                       // [B*L*K, C]
        let w = MLXArray(wts).reshaped([B * L * K, 1])
        let weighted = (gathered * w).reshaped([B * L, K, C]).sum(axis: 1)          // [B*L, C]
        if mode == "nearest" {
            return weighted.reshaped([B, L, C])
        }
        let wsum = MLX.maximum(MLXArray(wts).reshaped([B * L, K]).sum(axis: 1, keepDims: true), MLXArray(Float(1e-12)))
        return (weighted / wsum).reshaped([B, L, C])
    }
}
