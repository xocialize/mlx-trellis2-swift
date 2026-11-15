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
    ///
    /// Every row comes out unit length. A vertex whose incident faces leave no direction gets a
    /// stand-in: no incident face, only zero-area faces, or faces that cancel — the mesh the bake
    /// unwraps carries a few zero-thickness fins (two coincident triangles, opposite winding),
    /// xatlas gives each fin a 3-vertex chart of its own, and the two cross products sum to exactly
    /// (0,0,0). See `accumulateVertexNormals(faceNormals:faces:vertexCount:)`.
    public func vertexNormals() -> MLXArray {
        let fn = faceNormals(normalize: false).asArray(Float.self)   // [F, 3], elementwise on GPU
        let f = faces.asArray(Int32.self)
        let acc = Mesh.accumulateVertexNormals(faceNormals: fn, faces: f, vertexCount: vertexCount).sums
        let summed = MLXArray(acc, [vertexCount, 3])
        let len = MLXLinalg.norm(summed, axis: -1, keepDims: true)
        return summed / MLX.maximum(len, MLXArray(1e-20 as Float))
    }

    /// The CPU half of `vertexNormals()`: per-vertex sums of the un-normalized face normals
    /// (`faceNormals` is `[F*3]`, `faces` is `[F*3]`), in face order, so the same bits every run.
    ///
    /// A vertex whose sum has lost its direction — shorter than 1e-6 of the summed contribution
    /// lengths, which covers no incident face, zero-area faces only, and cancelling faces — is
    /// `repaired`: its row becomes the unit normal of its largest incident face (lowest face index
    /// on ties), or (0, 0, 1) when it has no face of nonzero area. Every other row is the plain sum,
    /// bit for bit, so meshes without such vertices give the same normals as before the repair.
    public static func accumulateVertexNormals(faceNormals fn: [Float], faces f: [Int32], vertexCount: Int)
        -> (sums: [Float], repaired: [Int]) {
        let faceCount = f.count / 3
        var acc = [Float](repeating: 0, count: vertexCount * 3)
        var contribution = [Float](repeating: 0, count: vertexCount)   // Σ ‖face normal‖ per vertex
        var largest = [Int](repeating: -1, count: vertexCount)
        var largestLen = [Float](repeating: 0, count: vertexCount)
        for fi in 0 ..< faceCount {
            let nx = fn[fi*3], ny = fn[fi*3 + 1], nz = fn[fi*3 + 2]
            let fl = (nx*nx + ny*ny + nz*nz).squareRoot()
            for c in 0 ..< 3 {
                let vi = Int(f[fi*3 + c]), v = vi * 3
                acc[v] += nx; acc[v + 1] += ny; acc[v + 2] += nz
                contribution[vi] += fl
                if fl > largestLen[vi] { largestLen[vi] = fl; largest[vi] = fi }
            }
        }
        var repaired: [Int] = []
        for vi in 0 ..< vertexCount {
            let v = vi * 3
            let len = (acc[v]*acc[v] + acc[v + 1]*acc[v + 1] + acc[v + 2]*acc[v + 2]).squareRoot()
            if len > 1e-6 * contribution[vi] { continue }
            repaired.append(vi)
            // Normalized in Double, so a face whose squared components underflow Float still
            // gives a unit row; the MLX normalization after this leaves a unit row unit.
            var n: (Double, Double, Double) = (0, 0, 1)
            if largest[vi] >= 0 {
                let fi = largest[vi]
                let x = Double(fn[fi*3]), y = Double(fn[fi*3 + 1]), z = Double(fn[fi*3 + 2])
                let l = (x*x + y*y + z*z).squareRoot()
                n = (x / l, y / l, z / l)
            }
            acc[v] = Float(n.0); acc[v + 1] = Float(n.1); acc[v + 2] = Float(n.2)
        }
        return (acc, repaired)
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
