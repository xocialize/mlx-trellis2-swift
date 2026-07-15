import Foundation
import MLX

/// Triangle mesh backed by MLX arrays.
///
/// Mirrors `cumesh.CuMesh` from the upstream CUDA library: vertices are `[V, 3]`
/// floats, faces are `[F, 3]` int32 indices into `vertices`.
public struct Mesh {
    public var vertices: MLXArray  // [V, 3] float32
    public var faces: MLXArray     // [F, 3] int32

    public init(vertices: MLXArray, faces: MLXArray) {
        precondition(vertices.ndim == 2 && vertices.dim(1) == 3,
                     "vertices must be [V, 3], got \(vertices.shape)")
        precondition(faces.ndim == 2 && faces.dim(1) == 3,
                     "faces must be [F, 3], got \(faces.shape)")
        self.vertices = vertices
        self.faces = faces
    }

    public var vertexCount: Int { vertices.dim(0) }
    public var faceCount: Int { faces.dim(0) }

    // MARK: - Convenience counts (cumesh `num_*` parity)

    /// Number of unique undirected edges. Forces a build of `edgeTable`.
    public var numEdges: Int { edgeTable().count }

    /// Number of boundary (incidence == 1) edges.
    public var numBoundaryEdges: Int {
        edgeTable().edgeFaceCount.asArray(Int32.self).filter { $0 == 1 }.count
    }

    /// Number of boundary loops (oriented cycles of boundary edges).
    public var numBoundaryLoops: Int { boundaryLoops().count }

    /// Number of connected components.
    public var numConnectedComponents: Int { componentCount() }
}
