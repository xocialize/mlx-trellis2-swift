import Foundation
import MLX

/// Barycentric UV-space rasterizer — faithful port of trellis2 `_rasterize_uv`.
/// For each face, walks its integer texel bbox; texels whose center lies inside the
/// triangle are emitted with face id + barycentric weights (for the texture bake).
public enum UVRasterize {
    /// uvsPix [V,2] (UV scaled to [0,T]), faces [F,3] int. Returns per covered texel:
    /// pix [K,2] int32 (x,y), faceId [K] int32, bary [K,3] float (sum≈1).
    /// Output order = face-major, bbox row-major (py outer, px inner), inside-filtered.
    public static func rasterize(uvsPix: MLXArray, faces: MLXArray, textureSize: Int)
        -> (pix: MLXArray, faceId: MLXArray, bary: MLXArray) {
        let uv = uvsPix.asType(.float32).asArray(Float.self)   // [V*2]
        let fc = faces.asType(.int32).asArray(Int32.self)      // [F*3]
        let F = faces.dim(0), T = textureSize
        var pix = [Int32](), fid = [Int32](), bary = [Float]()

        for f in 0..<F {
            let ia = Int(fc[f*3]), ib = Int(fc[f*3+1]), ic = Int(fc[f*3+2])
            let x0 = uv[ia*2], y0 = uv[ia*2+1]
            let x1 = uv[ib*2], y1 = uv[ib*2+1]
            let x2 = uv[ic*2], y2 = uv[ic*2+1]
            let xmin = min(Int(floor(min(x0, min(x1, x2)))).clampedTo(0, T-1), T-1)
            let xmax = Int(floor(max(x0, max(x1, x2)))).clampedTo(0, T-1)
            let ymin = min(Int(floor(min(y0, min(y1, y2)))).clampedTo(0, T-1), T-1)
            let ymax = Int(floor(max(y0, max(y1, y2)))).clampedTo(0, T-1)
            var denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-12 { denom = 1e-12 }
            for py in ymin...max(ymin, ymax) where ymax >= ymin {
                for px in xmin...max(xmin, xmax) where xmax >= xmin {
                    let fx = Float(px) + 0.5, fy = Float(py) + 0.5
                    let l0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) / denom
                    let l1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) / denom
                    let l2 = 1.0 - l0 - l1
                    if l0 >= -1e-5 && l1 >= -1e-5 && l2 >= -1e-5 {
                        pix.append(Int32(px)); pix.append(Int32(py))
                        fid.append(Int32(f))
                        bary.append(l0); bary.append(l1); bary.append(l2)
                    }
                }
            }
        }
        let K = fid.count
        return (MLXArray(pix).reshaped([K, 2]), MLXArray(fid), MLXArray(bary).reshaped([K, 3]))
    }
}

private extension Int {
    func clampedTo(_ lo: Int, _ hi: Int) -> Int { Swift.min(Swift.max(self, lo), hi) }
}
