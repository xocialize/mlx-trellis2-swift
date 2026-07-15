import Foundation
import MLX
import MLXFast

/// Per-vertex quadric error metric (QEM) matrices, packed as 10 floats per
/// symmetric 4×4. Layout: `[a00, a01, a02, a03, a11, a12, a13, a22, a23, a33]`.
///
/// Each face contributes `p pᵀ` (where `p = (n, -n·v0)` is its plane equation)
/// to the quadric of each of its 3 vertices. Used as the foundation of
/// edge-collapse simplification.

private let accumulateQuadricsKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "accumulate_quadrics",
    inputNames: ["vertices", "faces"],
    outputNames: ["Q"],
    source: """
        uint f = thread_position_in_grid.x;
        uint fcount = faces_shape[0];
        if (f >= fcount) return;

        int i0 = faces[f*3 + 0];
        int i1 = faces[f*3 + 1];
        int i2 = faces[f*3 + 2];
        float3 v0 = float3(vertices[i0*3 + 0], vertices[i0*3 + 1], vertices[i0*3 + 2]);
        float3 v1 = float3(vertices[i1*3 + 0], vertices[i1*3 + 1], vertices[i1*3 + 2]);
        float3 v2 = float3(vertices[i2*3 + 0], vertices[i2*3 + 1], vertices[i2*3 + 2]);

        float3 n = cross(v1 - v0, v2 - v0);
        float ln = length(n);
        if (ln < 1e-12f) return;
        n /= ln;
        float d = -dot(n, v0);

        // K = p pᵀ packed into 10 floats.
        float px = n.x, py = n.y, pz = n.z, pw = d;
        float k0 = px*px, k1 = px*py, k2 = px*pz, k3 = px*pw;
        float k4 = py*py, k5 = py*pz, k6 = py*pw;
        float k7 = pz*pz, k8 = pz*pw;
        float k9 = pw*pw;

        int verts[3] = { i0, i1, i2 };
        for (uint vi = 0; vi < 3; ++vi) {
            int v = verts[vi];
            atomic_fetch_add_explicit(&Q[v*10 + 0], k0, memory_order_relaxed);
            atomic_fetch_add_explicit(&Q[v*10 + 1], k1, memory_order_relaxed);
            atomic_fetch_add_explicit(&Q[v*10 + 2], k2, memory_order_relaxed);
            atomic_fetch_add_explicit(&Q[v*10 + 3], k3, memory_order_relaxed);
            atomic_fetch_add_explicit(&Q[v*10 + 4], k4, memory_order_relaxed);
            atomic_fetch_add_explicit(&Q[v*10 + 5], k5, memory_order_relaxed);
            atomic_fetch_add_explicit(&Q[v*10 + 6], k6, memory_order_relaxed);
            atomic_fetch_add_explicit(&Q[v*10 + 7], k7, memory_order_relaxed);
            atomic_fetch_add_explicit(&Q[v*10 + 8], k8, memory_order_relaxed);
            atomic_fetch_add_explicit(&Q[v*10 + 9], k9, memory_order_relaxed);
        }
    """,
    atomicOutputs: true
)

extension Mesh {
    /// Per-vertex QEM matrices, `[V, 10]` float32. Computed via an atomic Metal
    /// kernel scattering each face's plane contribution to its 3 vertices.
    public func vertexQuadrics() -> MLXArray {
        let v = vertexCount
        let f = faceCount
        let tg = min(max(f, 1), 256)
        let outputs = accumulateQuadricsKernel(
            [vertices, faces],
            grid: (f, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[v, 10]],
            outputDTypes: [.float32],
            initValue: 0
        )
        return outputs[0]
    }

    /// CPU reference for `vertexQuadrics`. Used by tests to validate the GPU
    /// kernel; production callers should use the GPU variant.
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
