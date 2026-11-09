import Foundation
import MLX
import MLXFast

/// Per-vertex quadric error metric (QEM) matrices, packed as 10 floats per
/// symmetric 4×4. Layout: `[a00, a01, a02, a03, a11, a12, a13, a22, a23, a33]`.
///
/// Each face contributes `p pᵀ` (where `p = (n, -n·v0)` is its plane equation)
/// to the quadric of each of its 3 vertices. Used as the foundation of
/// edge-collapse simplification.

extension Mesh {
    /// Per-vertex QEM matrices, `[V, 10]` float32.
    ///
    /// Accumulated on the CPU in face order, so the same mesh gives the same bits every run. It was a
    /// Metal kernel scattering with float `atomic_fetch_add`, whose order — and so whose rounding —
    /// changed between runs; through simplify's edge costs that moved the final face count by up to
    /// 0.07 %, and the GLB bytes with it (AB-A-0263: Gaea checks plugin output byte for byte).
    /// Simplify is CPU-bound per iteration already (the link-condition pass), and this is one linear
    /// sweep over the faces.
    public func vertexQuadrics() -> MLXArray { vertexQuadricsCPU() }

    /// The deterministic face-order accumulation `vertexQuadrics` returns.
    func vertexQuadricsCPU() -> MLXArray {
        let v = vertices.asArray(Float.self)
        let f = faces.asArray(Int32.self)
        var Q = [Float](repeating: 0, count: vertexCount * 10)
        for fi in 0..<faceCount {
            let i0 = Int(f[fi*3 + 0])
            let i1 = Int(f[fi*3 + 1])
            let i2 = Int(f[fi*3 + 2])
            let v0x = v[i0*3 + 0], v0y = v[i0*3 + 1], v0z = v[i0*3 + 2]
            let v1x = v[i1*3 + 0], v1y = v[i1*3 + 1], v1z = v[i1*3 + 2]
            let v2x = v[i2*3 + 0], v2y = v[i2*3 + 1], v2z = v[i2*3 + 2]
            let e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z
            let e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z
            let cx = e1y * e2z - e1z * e2y
            let cy = e1z * e2x - e1x * e2z
            let cz = e1x * e2y - e1y * e2x
            let ln = (cx*cx + cy*cy + cz*cz).squareRoot()
            if ln < 1e-12 { continue }
            let nx = cx / ln, ny = cy / ln, nz = cz / ln
            let d = -(nx * v0x + ny * v0y + nz * v0z)
            let k0 = nx*nx, k1 = nx*ny, k2 = nx*nz, k3 = nx*d
            let k4 = ny*ny, k5 = ny*nz, k6 = ny*d
            let k7 = nz*nz, k8 = nz*d
            let k9 = d*d
            for vi in [i0, i1, i2] {
                Q[vi*10 + 0] += k0; Q[vi*10 + 1] += k1
                Q[vi*10 + 2] += k2; Q[vi*10 + 3] += k3
                Q[vi*10 + 4] += k4; Q[vi*10 + 5] += k5
                Q[vi*10 + 6] += k6; Q[vi*10 + 7] += k7
                Q[vi*10 + 8] += k8; Q[vi*10 + 9] += k9
            }
        }
        return MLXArray(Q, [vertexCount, 10])
    }
}
