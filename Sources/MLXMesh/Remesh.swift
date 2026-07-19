import Foundation
import MLX

extension Mesh {
    /// Narrow-band UDF + dual contouring remesh. Produces a clean triangle
    /// mesh by extracting the isosurface at distance `eps = band * voxelSize`
    /// from the input.
    ///
    /// Mirrors `cumesh.remeshing.remesh_narrow_band_dc`.
    public func remeshDualContouring(
        resolution: Int,
        band: Float = 1.0,
        bvh: BVH? = nil
    ) -> Mesh {
        let grid = dcVoxelGrid(resolution: resolution, band: band, bvh: bvh)
        // Quad emission produces locally-consistent windings inside each quad
        // and across same-axis neighbours, but cross-axis quads can disagree.
        // Unify orientations propagates consistency across the whole component.
        return grid.extractMesh().unifyFaceOrientations()
    }

    /// `remeshDualContouring` plus the per-face grid-axis provenance tags
    /// (index-aligned; `unifyFaceOrientations` never reorders faces).
    public func remeshDualContouringTagged(
        resolution: Int,
        band: Float = 1.0,
        bvh: BVH? = nil
    ) -> (mesh: Mesh, faceAxis: [UInt8]) {
        let grid = dcVoxelGrid(resolution: resolution, band: band, bvh: bvh)
        let (raw, faceAxis) = grid.extractMeshTagged()
        return (raw.unifyFaceOrientations(), faceAxis)
    }
}

extension DCVoxelGrid {
    /// Run P2 (DC vertices) and P3 (quad emission) on this grid, returning
    /// the final triangle mesh.
    public func extractMesh() -> Mesh { extractMeshTagged().mesh }

    /// `extractMesh` plus per-face provenance: each triangle records which
    /// grid-edge axis its quad crossed (0 = X, 1 = Y, 2 = Z). This is the
    /// build-time information the provenance unwrap consumes — see
    /// UV-UNWRAP-METAL-PLAN.md Phase 2. Tags are index-aligned with faces and
    /// survive `unifyFaceOrientations` (which flips windings in place without
    /// reordering).
    public func extractMeshTagged() -> (mesh: Mesh, faceAxis: [UInt8]) {
        let R = resolution
        let n = R + 1
        let dc = computeDCVertices()
        let active = dc.active.asArray(Bool.self)
        let localDC = dc.localDCVertex.asArray(Float.self)
        let corners = cornerDistances.asArray(Float.self)
        let step = cubeSize / Float(R)

        // 1. Compact active voxels into dense vertex indices, emit world-space
        //    DC vertices.
        var voxelDenseIdx = [Int32](repeating: -1, count: R * R * R)
        var outVerts: [Float] = []
        outVerts.reserveCapacity(active.count * 3 / 8)
        var nextIdx: Int32 = 0
        for v in 0..<(R * R * R) where active[v] {
            voxelDenseIdx[v] = nextIdx
            nextIdx += 1
            let vx = v / (R * R)
            let vy = (v / R) % R
            let vz = v % R
            let lx = localDC[v*3 + 0]
            let ly = localDC[v*3 + 1]
            let lz = localDC[v*3 + 2]
            outVerts.append(gridMin.x + (Float(vx) + lx) * step)
            outVerts.append(gridMin.y + (Float(vy) + ly) * step)
            outVerts.append(gridMin.z + (Float(vz) + lz) * step)
        }

        if nextIdx == 0 {
            return (Mesh(
                vertices: MLXArray([] as [Float], [0, 3]),
                faces: MLXArray([] as [Int32], [0, 3])
            ), [])
        }

        // 2. Emit quads + triangulate, one pass per axis.
        var outFaces: [Int32] = []
        var outAxis: [UInt8] = []

        // For each axis-aligned grid edge, the 4 voxels sharing it are at
        // the four diagonal corners in the plane perpendicular to the edge.
        //
        // X-edge from corner (cx, cy, cz) → (cx+1, cy, cz):
        //   voxels at (cx, cy-1, cz-1), (cx, cy-1, cz),
        //             (cx, cy,   cz-1), (cx, cy,   cz)
        // The same pattern with axes permuted for Y- and Z-edges.

        func voxelIdx(_ vx: Int, _ vy: Int, _ vz: Int) -> Int32 {
            if vx < 0 || vx >= R || vy < 0 || vy >= R || vz < 0 || vz >= R {
                return -1
            }
            return voxelDenseIdx[vx * R * R + vy * R + vz]
        }

        func cornerVal(_ cx: Int, _ cy: Int, _ cz: Int) -> Float {
            corners[cx * n * n + cy * n + cz]
        }

        // Helper: emit a quad as 2 triangles. Order of v0..v3 should be CCW
        // around the edge direction; sign of `forward` flips the winding.
        func emitQuad(_ v0: Int32, _ v1: Int32, _ v2: Int32, _ v3: Int32, forward: Bool, axis: UInt8) {
            if forward {
                outFaces.append(v0); outFaces.append(v1); outFaces.append(v2)
                outFaces.append(v0); outFaces.append(v2); outFaces.append(v3)
            } else {
                outFaces.append(v0); outFaces.append(v2); outFaces.append(v1)
                outFaces.append(v0); outFaces.append(v3); outFaces.append(v2)
            }
            outAxis.append(axis); outAxis.append(axis)
        }

        // X-edges.
        for cx in 0..<R {  // edges go (cx, cy, cz) → (cx+1, cy, cz); cx < R since cx+1 ≤ R
            for cy in 0..<n {
                for cz in 0..<n {
                    let dA = cornerVal(cx, cy, cz)
                    let dB = cornerVal(cx + 1, cy, cz)
                    if (dA < 0) == (dB < 0) { continue }
                    let v0 = voxelIdx(cx, cy - 1, cz - 1)
                    let v1 = voxelIdx(cx, cy - 1, cz)
                    let v2 = voxelIdx(cx, cy,     cz)
                    let v3 = voxelIdx(cx, cy,     cz - 1)
                    if v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0 { continue }
                    emitQuad(v0, v1, v2, v3, forward: dA < dB, axis: 0)
                }
            }
        }

        // Y-edges.
        for cy in 0..<R {
            for cx in 0..<n {
                for cz in 0..<n {
                    let dA = cornerVal(cx, cy, cz)
                    let dB = cornerVal(cx, cy + 1, cz)
                    if (dA < 0) == (dB < 0) { continue }
                    let v0 = voxelIdx(cx - 1, cy, cz - 1)
                    let v1 = voxelIdx(cx,     cy, cz - 1)
                    let v2 = voxelIdx(cx,     cy, cz)
                    let v3 = voxelIdx(cx - 1, cy, cz)
                    if v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0 { continue }
                    emitQuad(v0, v1, v2, v3, forward: dA < dB, axis: 1)
                }
            }
        }

        // Z-edges.
        for cz in 0..<R {
            for cx in 0..<n {
                for cy in 0..<n {
                    let dA = cornerVal(cx, cy, cz)
                    let dB = cornerVal(cx, cy, cz + 1)
                    if (dA < 0) == (dB < 0) { continue }
                    let v0 = voxelIdx(cx - 1, cy - 1, cz)
                    let v1 = voxelIdx(cx,     cy - 1, cz)
                    let v2 = voxelIdx(cx,     cy,     cz)
                    let v3 = voxelIdx(cx - 1, cy,     cz)
                    if v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0 { continue }
                    emitQuad(v0, v1, v2, v3, forward: dA < dB, axis: 2)
                }
            }
        }

        let vCount = outVerts.count / 3
        let fCount = outFaces.count / 3
        return (Mesh(
            vertices: MLXArray(outVerts, [vCount, 3]),
            faces: outFaces.isEmpty
                ? MLXArray([] as [Int32], [0, 3])
                : MLXArray(outFaces, [fCount, 3])
        ), outAxis)
    }
}
