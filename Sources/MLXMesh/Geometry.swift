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
    public func vertexNormals() -> MLXArray {
        let fn = faceNormals(normalize: false)              // [F, 3]
        // Broadcast each face normal across its 3 vertices: [F, 3, 3].
        let perCorner = MLX.repeated(
            MLX.expandedDimensions(fn, axis: 1), count: 3, axis: 1
        ).reshaped([faceCount * 3, 3])
        let idx = faces.reshaped([faceCount * 3])
        let zeros = MLXArray.zeros([vertexCount, 3], type: Float.self)
        let summed = zeros.at[idx].add(perCorner)
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
