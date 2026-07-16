import Foundation
import MLX
import Cfdg

/// Mesh -> flexible-dual-grid conversion (the encode-direction O-Voxel step), wrapping the
/// vendored upstream CPU C++ converter. Mirrors the Python wrapper's explicit-aabb path
/// (`o_voxel.convert.mesh_to_flexible_dual_grid(vertices, faces, grid_size=, aabb=)`), which
/// is the only form the TRELLIS.2 pipelines use.
public enum DualGridConvert {

    public struct Result {
        /// [N, 3] int32 voxel coordinates.
        public let coords: MLXArray
        /// [N, 3] float32 dual vertices, in the aabb-shifted mesh space (same convention as
        /// the Python wrapper's return).
        public let dualVertices: MLXArray
        /// [N, 3] bool per-axis intersection flags.
        public let intersected: MLXArray
    }

    public enum Error: Swift.Error {
        case conversionFailed
    }

    /// Convert a triangle mesh to the flexible dual grid.
    ///
    /// - Parameters:
    ///   - vertices: n*3 vertex positions (model space).
    ///   - faces: m*3 triangle vertex indices.
    ///   - gridSize: cubic grid resolution (512 / 1024 / ...).
    ///   - aabb: (min, max) bounds; the pipeline convention is [-0.5, 0.5]^3.
    public static func meshToFlexibleDualGrid(
        vertices: [Float],
        faces: [Int32],
        gridSize: Int,
        aabbMin: SIMD3<Float> = .init(-0.5, -0.5, -0.5),
        aabbMax: SIMD3<Float> = .init(0.5, 0.5, 0.5),
        faceWeight: Float = 1.0,
        boundaryWeight: Float = 0.2,
        regularizationWeight: Float = 1e-2,
        timing: Bool = false
    ) throws -> Result {
        precondition(vertices.count % 3 == 0 && faces.count % 3 == 0)
        let nV = vertices.count / 3
        let nF = faces.count / 3

        // Python wrapper semantics: voxel_size = extent / grid, shift vertices by aabb min,
        // grid_range = [0, grid].
        let extent = aabbMax - aabbMin
        let voxelSize: [Float] = [extent.x / Float(gridSize), extent.y / Float(gridSize), extent.z / Float(gridSize)]
        var shifted = [Float](repeating: 0, count: vertices.count)
        for i in 0..<nV {
            shifted[i * 3 + 0] = vertices[i * 3 + 0] - aabbMin.x
            shifted[i * 3 + 1] = vertices[i * 3 + 1] - aabbMin.y
            shifted[i * 3 + 2] = vertices[i * 3 + 2] - aabbMin.z
        }
        let gridRange: [Int32] = [0, 0, 0, Int32(gridSize), Int32(gridSize), Int32(gridSize)]

        var outCoords: UnsafeMutablePointer<Int32>? = nil
        var outDual: UnsafeMutablePointer<Float>? = nil
        var outIntersected: UnsafeMutablePointer<UInt8>? = nil
        let n = shifted.withUnsafeBufferPointer { vb in
            faces.withUnsafeBufferPointer { fb in
                fdg_mesh_to_flexible_dual_grid(
                    vb.baseAddress, Int64(nV),
                    fb.baseAddress, Int64(nF),
                    voxelSize, gridRange,
                    faceWeight, boundaryWeight, regularizationWeight,
                    timing ? 1 : 0,
                    &outCoords, &outDual, &outIntersected)
            }
        }
        guard n >= 0, let pc = outCoords, let pd = outDual, let pi = outIntersected else {
            throw Error.conversionFailed
        }
        defer { fdg_free(pc); fdg_free(pd); fdg_free(pi) }
        let count = Int(n)
        let coords = MLXArray(UnsafeBufferPointer(start: pc, count: count * 3), [count, 3])
        let dual = MLXArray(UnsafeBufferPointer(start: pd, count: count * 3), [count, 3])
        let inter = MLXArray(UnsafeBufferPointer(start: pi, count: count * 3), [count, 3]).asType(.bool)
        return Result(coords: coords, dualVertices: dual, intersected: inter)
    }

    /// Assemble the shape-encoder inputs from a conversion result, mirroring
    /// `Trellis2TexturingPipeline.encode_shape_slat`'s SparseTensor construction:
    /// vfeats = dual * resolution - coords (local offsets), coords prefixed with batch 0.
    /// Returns (coords4 [N,4] int32, vfeats [N,3] float32, ifeats [N,3] float32-as-bool-source).
    public static func encoderInputs(_ r: Result, resolution: Int) -> (coords4: MLXArray, vfeats: MLXArray, ifeats: MLXArray) {
        let vfeats = r.dualVertices * Float(resolution) - r.coords.asType(.float32)
        let batch = MLXArray.zeros([r.coords.shape[0], 1], dtype: .int32)
        let coords4 = concatenated([batch, r.coords], axis: 1)
        return (coords4, vfeats, r.intersected)
    }
}
