import Foundation
import MLX

/// flexible_dual_grid_to_mesh (inference) — flexicubes-style dual-grid extractor.
/// Turns the shape decoder's per-voxel dual vertex + edge-intersection + quad-lerp
/// into a triangle mesh: each intersected edge is shared by 4 voxels whose dual
/// vertices form a quad, split into 2 triangles along the higher-weight diagonal.
/// Faithful port of trellis2/portable/dual_grid.py (train=False, split_weight given).
public enum DualGridMesh {
    // the 4 voxels sharing an edge along each axis (offset [0,0,0] is the owning voxel)
    static let edgeOffset: [[[Int]]] = [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],   // x-axis
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],   // y-axis
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],   // z-axis
    ]
    static let split1 = [0, 1, 2, 0, 2, 3]   // diagonal 0-2
    static let split2 = [0, 1, 3, 3, 1, 2]   // diagonal 1-3

    /// coords [N,3] int, dualVerts [N,3], intersected [N,3] (nonzero=true), quadLerp [N,1].
    /// Returns (verts [N,3], faces [F,3] int32). voxel_size = (aabbMax-aabbMin)/gridSize.
    public static func extract(
        coords: MLXArray, dualVerts: MLXArray, intersected: MLXArray, quadLerp: MLXArray,
        gridSize: Float = 256, aabbMin: Float = -0.5, aabbMax: Float = 0.5
    ) -> (verts: MLXArray, faces: MLXArray) {
        let N = coords.dim(0)
        let voxelSize = (aabbMax - aabbMin) / gridSize
        let verts = (coords.asType(.float32) + dualVerts) * voxelSize + aabbMin   // [N,3]

        let c = coords.asType(.int32).asArray(Int32.self)          // [N*3]
        let inter = intersected.asType(.int32).asArray(Int32.self) // [N*3]
        let ql = quadLerp.asType(.float32).asArray(Float.self)     // [N]
        var maxc: Int32 = 0
        for v in c { maxc = max(maxc, v) }
        let R = Int64(maxc) + 2
        @inline(__always) func key(_ x: Int, _ y: Int, _ z: Int) -> Int64 {
            (Int64(x) &* R &+ Int64(y)) &* R &+ Int64(z)
        }
        var idx = [Int64: Int32](minimumCapacity: N * 2)
        for n in 0..<N { idx[key(Int(c[n*3]), Int(c[n*3+1]), Int(c[n*3+2]))] = Int32(n) }

        var faces = [Int32](); faces.reserveCapacity(N * 6)
        var q = [Int32](repeating: 0, count: 4)
        for n in 0..<N {
            let x = Int(c[n*3]), y = Int(c[n*3+1]), z = Int(c[n*3+2])
            for a in 0..<3 where inter[n*3 + a] != 0 {
                var ok = true
                for k in 0..<4 {
                    let o = edgeOffset[a][k]
                    if let j = idx[key(x + o[0], y + o[1], z + o[2])] { q[k] = j } else { ok = false; break }
                }
                if !ok { continue }
                let ws02 = ql[Int(q[0])] * ql[Int(q[2])]
                let ws13 = ql[Int(q[1])] * ql[Int(q[3])]
                let split = ws02 > ws13 ? split1 : split2
                for s in split { faces.append(q[s]) }
            }
        }
        let F = faces.count / 3
        return (verts, MLXArray(faces).reshaped([F, 3]))
    }
}
