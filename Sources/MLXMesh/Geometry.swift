import Foundation
import MLX
import MLXLinalg

extension Mesh {
    /// Per-face normal vectors, `[F, 3]`. Not unit-length until `normalize: true`.
    public func faceNormals(normalize: Bool = true) -> MLXArray {
        // Gather triangle corners: [F, 3, 3]
        let tri = vertices[faces]
        let v0 = tri[0..., 0]
        let v1 = tri[0..., 1]
        let v2 = tri[0..., 2]
        let n = MLX.cross(v1 - v0, v2 - v0, axis: -1)
        if !normalize { return n }
        let len = MLXLinalg.norm(n, axis: -1, keepDims: true)
        return n / MLX.maximum(len, MLXArray(1e-20 as Float))
    }

    /// Per-face areas, `[F]`.
    public func faceAreas() -> MLXArray {
        let n = faceNormals(normalize: false)
        return 0.5 * MLXLinalg.norm(n, axis: -1)
    }

    /// Area-weighted per-vertex normals, `[V, 3]`. Each face contributes its
    /// un-normalized cross product (magnitude = 2·area) to its three vertices;
    /// the result is then normalized.
    ///
    /// The per-vertex sum runs on the CPU in face order, so it is the same bits every run. A GPU
    /// scatter-add (`at[].add`) sums duplicate indices in whatever order the threads land, and
    /// float addition is not associative (AB-A-0263: byte-for-byte reproducible GLBs).
    public func vertexNormals() -> MLXArray {
        let fn = faceNormals(normalize: false).asArray(Float.self)   // [F, 3], elementwise on GPU
        let f = faces.asArray(Int32.self)
        var acc = [Float](repeating: 0, count: vertexCount * 3)
        for fi in 0 ..< faceCount {
            let nx = fn[fi*3], ny = fn[fi*3 + 1], nz = fn[fi*3 + 2]
            for c in 0 ..< 3 {
                let v = Int(f[fi*3 + c]) * 3
                acc[v] += nx; acc[v + 1] += ny; acc[v + 2] += nz
            }
        }
        let summed = MLXArray(acc, [vertexCount, 3])
        let len = MLXLinalg.norm(summed, axis: -1, keepDims: true)
        return summed / MLX.maximum(len, MLXArray(1e-20 as Float))
    }

    /// Axis-aligned bounding box as `(min, max)`, each `[3]`.
    public func boundingBox() -> (min: MLXArray, max: MLXArray) {
        (vertices.min(axis: 0), vertices.max(axis: 0))
    }

    /// Centroid of the vertex set (unweighted), `[3]`.
    public func centroid() -> MLXArray {
        vertices.mean(axis: 0)
    }
}
