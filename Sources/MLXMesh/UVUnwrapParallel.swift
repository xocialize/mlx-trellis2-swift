import Foundation
import MLX
import simd
import Xatlas

/// Parallel xatlas unwrap (UV-UNWRAP-METAL-PLAN.md "NEXT TRACK" item 1).
///
/// xatlas's chart computation is a serial merge loop with worse-than-linear
/// scaling in workload size (measured 628.6s at 300k faces). Spatial BSP
/// partitioning into N face-balanced buckets and running an independent
/// `Atlas` per bucket concurrently is a near-lossless decomposition: chart
/// quality inside buckets is identical, charts split only along the ~log2(N)
/// cut planes (measured: chart count within 5% of the single run), and the
/// superlinear cost means per-bucket relief far exceeds the concurrency win
/// (measured 64.8×–510.6×). Bucket parameterizations are then merged into ONE
/// atlas via the pack-only seam (`addUvMesh`) — packing was never the cost.
extension Mesh {
    public func uvUnwrapParallel(
        buckets requestedBuckets: Int = 16,
        chartOptions: ChartOptions = ChartOptions(),
        packOptions: PackOptions = PackOptions()
    ) throws -> UVUnwrapResult {
        precondition(vertexCount > 0 && faceCount > 0, "uvUnwrapParallel requires a non-empty mesh")
        // Tiny meshes: partitioning overhead beats the win; use the direct path.
        let N = min(requestedBuckets, max(1, faceCount / 4096))
        if N <= 1 {
            return try uvUnwrap(chartOptions: chartOptions, packOptions: packOptions)
        }

        let vRaw = vertices.asArray(Float.self)
        let fRaw = faces.asArray(Int32.self)
        let F = faceCount

        // --- spatial BSP over face centroids: N face-balanced buckets ---
        var centroids = [SIMD3<Float>](repeating: .zero, count: F)
        for fi in 0..<F {
            let i0 = Int(fRaw[fi*3])*3, i1 = Int(fRaw[fi*3+1])*3, i2 = Int(fRaw[fi*3+2])*3
            centroids[fi] = SIMD3<Float>(
                (vRaw[i0] + vRaw[i1] + vRaw[i2]) / 3,
                (vRaw[i0+1] + vRaw[i1+1] + vRaw[i2+1]) / 3,
                (vRaw[i0+2] + vRaw[i1+2] + vRaw[i2+2]) / 3)
        }
        func bsp(_ faces: [Int32], _ leaves: Int) -> [[Int32]] {
            if leaves <= 1 || faces.count < 64 { return [faces] }
            var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
            var mx = -mn
            for fi in faces { mn = simd_min(mn, centroids[Int(fi)]); mx = simd_max(mx, centroids[Int(fi)]) }
            let ext = mx - mn
            let axis = ext.x >= ext.y && ext.x >= ext.z ? 0 : (ext.y >= ext.z ? 1 : 2)
            let sorted = faces.sorted { centroids[Int($0)][axis] < centroids[Int($1)][axis] }
            let lo = leaves / 2, hi = leaves - lo
            let cut = sorted.count * lo / leaves
            return bsp(Array(sorted[..<cut]), lo) + bsp(Array(sorted[cut...]), hi)
        }
        let bucketFaces = bsp(Array(0..<Int32(F)), N)

        // Shared texel scale so bucket parameterizations merge at uniform
        // density (the final repack re-normalizes the absolute scale anyway).
        var bbMin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var bbMax = -bbMin
        for i in 0..<vertexCount {
            let p = SIMD3<Float>(vRaw[i*3], vRaw[i*3+1], vRaw[i*3+2])
            bbMin = simd_min(bbMin, p); bbMax = simd_max(bbMax, p)
        }
        let extent = max(simd_reduce_max(bbMax - bbMin), 1e-6)
        let sharedTexelsPerUnit = 1024 / extent

        // --- per-bucket xatlas, concurrent ---
        struct BucketOut {
            var origVert: [Int32] = []      // per output vert: original mesh vertex
            var uvTexels: [SIMD2<Float>] = []
            var indices: [Int32] = []       // into this bucket's output verts
            var origFace: [Int32] = []      // per output face: original mesh face
            var atlasWidth: Float = 0
            var error: Error? = nil
        }
        nonisolated(unsafe) var outs = [BucketOut](repeating: BucketOut(), count: bucketFaces.count)
        let bucketFacesF = bucketFaces
        DispatchQueue.concurrentPerform(iterations: bucketFacesF.count) { b in
            var out = BucketOut()
            defer { outs[b] = out }
            // local submesh
            var remap = [Int32: UInt32]()
            var positions = [SIMD3<Float>]()
            var localIdx = [UInt32]()
            var localToOrig = [Int32]()
            for fi in bucketFacesF[b] {
                for k in 0..<3 {
                    let ov = fRaw[Int(fi)*3 + k]
                    if let m = remap[ov] { localIdx.append(m) }
                    else {
                        let m = UInt32(positions.count)
                        remap[ov] = m
                        positions.append(SIMD3<Float>(vRaw[Int(ov)*3], vRaw[Int(ov)*3+1], vRaw[Int(ov)*3+2]))
                        localToOrig.append(ov)
                        localIdx.append(m)
                    }
                }
            }
            let atlas = Atlas()
            do { try atlas.addMesh(MeshInput(positions: positions, indices: localIdx)) }
            catch { out.error = error; return }
            atlas.addMeshJoin()
            atlas.computeCharts(options: chartOptions)
            var bucketPack = PackOptions()
            bucketPack.texelsPerUnit = sharedTexelsPerUnit
            atlas.packCharts(options: bucketPack)
            guard atlas.meshCount > 0 else { return }
            let m = atlas.meshes[0]
            for ov in m.vertices {
                out.origVert.append(localToOrig[Int(ov.xref)])
                out.uvTexels.append(ov.uv)
            }
            out.indices = m.indices.map { Int32($0) }
            for i in 0..<(out.indices.count / 3) {
                out.origFace.append(bucketFacesF[b][min(i, bucketFacesF[b].count - 1)])
            }
            out.atlasWidth = Float(atlas.width)
        }
        for out in outs { if let e = out.error { throw e } }

        // --- concatenate with disjoint UV offsets (UvMeshDecl welds coincident
        // UVs — the milestone-1 streak lesson) and repack into ONE atlas ---
        var allUVs = [SIMD2<Float>]()
        var allIdx = [UInt32]()
        var concatOrigVert = [Int32]()
        var concatOrigFace = [Int32]()
        var offsetU: Float = 0
        for out in outs {
            let base = UInt32(concatOrigVert.count)
            for (i, uv) in out.uvTexels.enumerated() {
                allUVs.append(SIMD2<Float>(uv.x + offsetU, uv.y))
                concatOrigVert.append(out.origVert[i])
            }
            for idx in out.indices { allIdx.append(base + UInt32(idx)) }
            concatOrigFace.append(contentsOf: out.origFace)
            offsetU += out.atlasWidth * 1.05 + 8
        }

        let final = Atlas()
        try final.addUvMesh(UvMeshInput(uvs: allUVs, indices: allIdx))
        final.computeCharts()
        final.packCharts(options: packOptions)
        guard final.meshCount > 0 else { throw UVUnwrapError.noMeshesProduced }
        let out = final.meshes[0]

        var newVerts: [Float] = []
        var newUVs: [Float] = []
        var vmap: [Int32] = []
        newVerts.reserveCapacity(out.vertices.count * 3)
        let aW = Float(final.width == 0 ? 1 : final.width)
        let aH = Float(final.height == 0 ? 1 : final.height)
        for ov in out.vertices {
            let orig = Int(concatOrigVert[Int(ov.xref)])
            newVerts.append(vRaw[orig*3]); newVerts.append(vRaw[orig*3+1]); newVerts.append(vRaw[orig*3+2])
            newUVs.append(ov.uv.x / aW); newUVs.append(ov.uv.y / aH)
            vmap.append(Int32(orig))
        }
        var newFaces: [Int32] = []
        newFaces.reserveCapacity(out.indices.count)
        for i in out.indices { newFaces.append(Int32(i)) }

        let charts: [UVUnwrapResult.Chart] = out.charts.map { c in
            UVUnwrapResult.Chart(
                atlasIndex: c.atlasIndex, material: c.material, type: c.type,
                faceIndices: c.faceIndices.map { fi in
                    UInt32(bitPattern: concatOrigFace[min(Int(fi), concatOrigFace.count - 1)])
                })
        }
        let V = vmap.count
        return UVUnwrapResult(
            mesh: Mesh(vertices: MLXArray(newVerts, [V, 3]),
                       faces: MLXArray(newFaces, [newFaces.count / 3, 3])),
            uvs: MLXArray(newUVs, [V, 2]),
            vertexMap: MLXArray(vmap, [V]),
            atlasWidth: final.width, atlasHeight: final.height,
            charts: charts)
    }
}
