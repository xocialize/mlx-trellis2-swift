import Foundation
import MLX
import simd
import Xatlas

/// Provenance-based UV charting (UV-UNWRAP-METAL-PLAN.md Phase 2).
///
/// Dual-contouring emits every face knowing which grid-edge axis it crossed.
/// Instead of rediscovering that structure from face normals (xatlas's chart
/// computation — 93–99% of measured unwrap cost), charts are derived directly:
///
///   1. sign each face by its geometric normal along its tagged axis → 6-way tag
///   2. connected components over edge-adjacent same-tag faces → chart ids
///   3. per-chart planar projection along the axis (world-scale UVs, so texel
///      density is uniform across charts without per-chart normalization)
///   4. split vertices on chart seams, then hand the result to the pack-only
///      seam (`uvUnwrap(existingUVs:)`) — xatlas packs, nothing more.
public struct ProvenanceUnwrapResult {
    /// Result from the pack-only unwrap of the seam-split mesh.
    public let unwrap: UVUnwrapResult
    /// `[V_split]` — original-mesh vertex index for each seam-split vertex.
    /// Compose with `unwrap.vertexMap` (which maps into the split mesh) to
    /// reach original-mesh vertices from final output vertices.
    public let splitVertexMap: [Int32]
    /// Number of provenance charts before packing.
    public let chartCount: Int
}

extension Mesh {
    /// Transfer per-face tags from `source` to this mesh via nearest source
    /// face per face centroid. Used to carry provenance axis tags across
    /// `simplify` (which has no face mapping). Boundary wobble is tolerated
    /// downstream — the provenance pipeline smooths and CCs tags anyway.
    public func transferFaceTags(
        from source: Mesh, tags: [UInt8], sourceBVH: BVH? = nil
    ) -> [UInt8] {
        precondition(tags.count == source.faceCount, "tags must be per source face")
        let f = faces.asArray(Int32.self)
        let v = vertices.asArray(Float.self)
        var centroids = [Float](repeating: 0, count: faceCount * 3)
        for fi in 0..<faceCount {
            let i0 = Int(f[fi*3])*3, i1 = Int(f[fi*3+1])*3, i2 = Int(f[fi*3+2])*3
            for k in 0..<3 {
                centroids[fi*3 + k] = (v[i0+k] + v[i1+k] + v[i2+k]) / 3
            }
        }
        let bvh = sourceBVH ?? source.bvh()
        let cp = bvh.closestPoints(mesh: source, queries: MLXArray(centroids, [faceCount, 3]))
        return cp.faceIndices.asArray(Int32.self).map { tags[Int($0)] }
    }

    /// Per-face chart ids from provenance axis tags, with tag-field smoothing
    /// (kills single-face noise islands at the source) and small-chart
    /// absorption into their strongest edge-adjacent neighbour.
    ///
    /// - Parameters:
    ///   - faceAxis: `[F]` values in 0...2 from `extractMeshTagged` /
    ///     `remeshDualContouringTagged`.
    ///   - smoothingIterations: majority-relabel sweeps over the 6-way tag field.
    ///   - minChartFaces: charts smaller than this merge into a neighbour
    ///     (distortion from off-axis projection is bounded and localized).
    /// - Returns: per-face chart id, chart count, and each chart's projection axis.
    public func provenanceChartIds(
        faceAxis: [UInt8]?,
        smoothingIterations: Int = 2,
        minChartFaces: Int = 24
    ) -> (chartIds: [Int32], chartCount: Int, chartAxis: [UInt8]) {
        precondition(faceAxis == nil || faceAxis!.count == faceCount, "faceAxis must be per-face")
        let f = faces.asArray(Int32.self)
        let v = vertices.asArray(Float.self)

        // 6-way tag: axis*2 + (normal[axis] < 0 ? 1 : 0). Sign is derived from
        // the *final* winding (post-unify), so it reflects rendered facing.
        // With exact grid tags (raw DC mesh), the axis comes from provenance;
        // without (e.g. after simplify, where faces are large arbitrary
        // triangles), the face normal's dominant axis IS the best tag — it
        // guarantees cos ≥ 1/√3 projection density per face, which transferred
        // grid tags cannot (a mis-tagged big face projects near edge-on).
        var tag = [UInt8](repeating: 0, count: faceCount)
        for fi in 0..<faceCount {
            let i0 = Int(f[fi*3]), i1 = Int(f[fi*3 + 1]), i2 = Int(f[fi*3 + 2])
            let ax = SIMD3<Float>(v[i0*3], v[i0*3+1], v[i0*3+2])
            let bx = SIMD3<Float>(v[i1*3], v[i1*3+1], v[i1*3+2])
            let cx = SIMD3<Float>(v[i2*3], v[i2*3+1], v[i2*3+2])
            let nrm = simd_cross(bx - ax, cx - ax)
            let a: Int
            if let fa = faceAxis {
                a = Int(fa[fi])
            } else {
                let an = simd_abs(nrm)
                a = an.x >= an.y && an.x >= an.z ? 0 : (an.y >= an.z ? 1 : 2)
            }
            tag[fi] = UInt8(a * 2 + (nrm[a] < 0 ? 1 : 0))
        }

        // Face adjacency across shared edges (manifold pairs only; extra
        // non-manifold incidences are ignored for charting).
        let stride = UInt64(vertexCount) + 1
        var edgeFaces = [UInt64: (Int32, Int32)](minimumCapacity: faceCount * 3 / 2)
        for fi in 0..<faceCount {
            for k in 0..<3 {
                let a = UInt32(bitPattern: f[fi*3 + k])
                let b = UInt32(bitPattern: f[fi*3 + (k + 1) % 3])
                let key = UInt64(min(a, b)) * stride + UInt64(max(a, b))
                if var pair = edgeFaces[key] {
                    if pair.1 < 0 { pair.1 = Int32(fi); edgeFaces[key] = pair }
                } else {
                    edgeFaces[key] = (Int32(fi), -1)
                }
            }
        }
        var nbr = [Int32](repeating: -1, count: faceCount * 3)
        var nbrCount = [UInt8](repeating: 0, count: faceCount)
        for (_, pair) in edgeFaces where pair.1 >= 0 {
            let a = Int(pair.0), b = Int(pair.1)
            if nbrCount[a] < 3 { nbr[a*3 + Int(nbrCount[a])] = pair.1; nbrCount[a] += 1 }
            if nbrCount[b] < 3 { nbr[b*3 + Int(nbrCount[b])] = pair.0; nbrCount[b] += 1 }
        }

        // Tag smoothing: relabel a face when ≥2 of its neighbours agree on a
        // different tag. Removes staircase noise and single-face islands.
        for _ in 0..<smoothingIterations {
            var next = tag
            for fi in 0..<faceCount {
                var votes: [UInt8: Int] = [:]
                for k in 0..<Int(nbrCount[fi]) {
                    votes[tag[Int(nbr[fi*3 + k])], default: 0] += 1
                }
                if let (winner, n) = votes.max(by: { $0.value < $1.value }),
                   n >= 2, winner != tag[fi] {
                    next[fi] = winner
                }
            }
            tag = next
        }

        // Union-find over same-tag edge-adjacent faces.
        var parent = [Int32](repeating: 0, count: faceCount)
        for i in 0..<faceCount { parent[i] = Int32(i) }
        func find(_ x: Int32) -> Int32 {
            var r = x
            while parent[Int(r)] != r { r = parent[Int(r)] }
            var c = x
            while parent[Int(c)] != r { let nxt = parent[Int(c)]; parent[Int(c)] = r; c = nxt }
            return r
        }
        func union(_ a: Int32, _ b: Int32) {
            let ra = find(a), rb = find(b)
            if ra != rb { parent[Int(ra)] = rb }
        }
        for (_, pair) in edgeFaces where pair.1 >= 0 {
            if tag[Int(pair.0)] == tag[Int(pair.1)] { union(pair.0, pair.1) }
        }

        // Face normals (2*area-weighted) once, for the merge cone test.
        var faceNrm = [SIMD3<Float>](repeating: .zero, count: faceCount)
        for fi in 0..<faceCount {
            let i0 = Int(f[fi*3]), i1 = Int(f[fi*3 + 1]), i2 = Int(f[fi*3 + 2])
            let ax = SIMD3<Float>(v[i0*3], v[i0*3+1], v[i0*3+2])
            let bx = SIMD3<Float>(v[i1*3], v[i1*3+1], v[i1*3+2])
            let cx = SIMD3<Float>(v[i2*3], v[i2*3+1], v[i2*3+2])
            faceNrm[fi] = simd_cross(bx - ax, cx - ax)
        }

        // Small-chart absorption: iteratively merge charts below the face
        // threshold into the neighbouring chart sharing the most edges — but
        // only when the small chart's normal agrees with the neighbour's signed
        // projection direction (cone test), so merged faces don't back-face the
        // chart axis and fold over in UV space.
        for _ in 0..<4 {
            var rootFaces: [Int32: Int] = [:]
            var rootNrm: [Int32: SIMD3<Float>] = [:]
            for fi in 0..<faceCount {
                let r = find(Int32(fi))
                rootFaces[r, default: 0] += 1
                rootNrm[r, default: .zero] += faceNrm[fi]
            }
            var link: [Int32: [Int32: Int]] = [:]   // small root -> (nbr root -> shared edges)
            var anySmall = false
            for (_, pair) in edgeFaces where pair.1 >= 0 {
                let ra = find(pair.0), rb = find(pair.1)
                guard ra != rb else { continue }
                if rootFaces[ra]! < minChartFaces { link[ra, default: [:]][rb, default: 0] += 1; anySmall = true }
                if rootFaces[rb]! < minChartFaces { link[rb, default: [:]][ra, default: 0] += 1; anySmall = true }
            }
            if !anySmall { break }
            for (small, nbrs) in link {
                let sNrm = rootNrm[small] ?? .zero
                let sLen = simd_length(sNrm)
                guard sLen > 0 else {
                    if let best = nbrs.max(by: { $0.value < $1.value }) { union(small, best.key) }
                    continue
                }
                let sDir = sNrm / sLen
                // strongest-first neighbours; take the first passing the cone test
                for (nbrRoot, _) in nbrs.sorted(by: { $0.value > $1.value }) {
                    let tNrm = rootNrm[find(nbrRoot)] ?? .zero
                    let ta = simd_abs(tNrm)
                    let axis = ta.x >= ta.y && ta.x >= ta.z ? 0 : (ta.y >= ta.z ? 1 : 2)
                    let sign: Float = tNrm[axis] < 0 ? -1 : 1
                    if sDir[axis] * sign > 0.45 {
                        union(small, nbrRoot)
                        break
                    }
                }
                // no passing neighbour: chart stays separate (bounded distortion
                // beats foldover corruption in the bake)
            }
        }

        // Compact ids + per-chart projection axis (area-weighted normal).
        var chartOf = [Int32: Int32]()
        var chartIds = [Int32](repeating: 0, count: faceCount)
        var axisAccum: [SIMD3<Float>] = []
        for fi in 0..<faceCount {
            let root = find(Int32(fi))
            let c: Int32
            if let existing = chartOf[root] {
                c = existing
            } else {
                c = Int32(chartOf.count)
                chartOf[root] = c
                axisAccum.append(.zero)
            }
            chartIds[fi] = c
            axisAccum[Int(c)] += faceNrm[fi]   // 2*area*normal
        }
        let chartAxis = axisAccum.map { n -> UInt8 in
            let a = simd_abs(n)
            return a.x >= a.y && a.x >= a.z ? 0 : (a.y >= a.z ? 1 : 2)
        }
        return (chartIds, chartOf.count, chartAxis)
    }

    /// Full provenance unwrap: charts from tags, axis-plane UVs, seam split,
    /// then pack-only via `uvUnwrap(existingUVs:)`.
    public func provenanceUnwrap(
        faceAxis: [UInt8]?,
        smoothingIterations: Int = 2,
        minChartFaces: Int = 24,
        packOptions: PackOptions = PackOptions()
    ) throws -> ProvenanceUnwrapResult {
        var (chartIds, chartCount, chartAxis) = provenanceChartIds(
            faceAxis: faceAxis,
            smoothingIterations: smoothingIterations,
            minChartFaces: minChartFaces)
        let f = faces.asArray(Int32.self)
        let v = vertices.asArray(Float.self)

        // A face nearly perpendicular to its chart's projection axis has ~zero
        // UV area; xatlas drops such faces from the chart and leaves their
        // vertices UNPACKED at raw input coordinates — atlas-crossing streaks.
        // Point-collapsing them instead renders single-texel "fleck" triangles
        // (user-visible). Correct fix: REASSIGN each edge-on face to an
        // edge-adjacent chart whose axis actually suits it; only faces with no
        // compatible neighbour get a micro-chart along their own dominant axis.
        var faceUnitNrm = [SIMD3<Float>](repeating: .zero, count: faceCount)
        for fi in 0..<faceCount {
            let i0 = Int(f[fi*3]), i1 = Int(f[fi*3 + 1]), i2 = Int(f[fi*3 + 2])
            let ax = SIMD3<Float>(v[i0*3], v[i0*3+1], v[i0*3+2])
            let bx = SIMD3<Float>(v[i1*3], v[i1*3+1], v[i1*3+2])
            let cx = SIMD3<Float>(v[i2*3], v[i2*3+1], v[i2*3+2])
            let nrm = simd_cross(bx - ax, cx - ax)
            let len = simd_length(nrm)
            if len > 0 { faceUnitNrm[fi] = nrm / len }
        }
        // face adjacency (same edge-pair construction as provenanceChartIds)
        var nbr = [Int32](repeating: -1, count: faceCount * 3)
        do {
            let stride = UInt64(vertexCount) + 1
            var edgeFace = [UInt64: Int32](minimumCapacity: faceCount * 3 / 2)
            var nbrCount = [UInt8](repeating: 0, count: faceCount)
            for fi in 0..<faceCount {
                for k in 0..<3 {
                    let a = UInt32(bitPattern: f[fi*3 + k])
                    let b = UInt32(bitPattern: f[fi*3 + (k + 1) % 3])
                    let key = UInt64(min(a, b)) * stride + UInt64(max(a, b))
                    if let other = edgeFace[key], other != Int32(fi) {
                        if nbrCount[fi] < 3 { nbr[fi*3 + Int(nbrCount[fi])] = other; nbrCount[fi] += 1 }
                        let o = Int(other)
                        if nbrCount[o] < 3 { nbr[o*3 + Int(nbrCount[o])] = Int32(fi); nbrCount[o] += 1 }
                    } else {
                        edgeFace[key] = Int32(fi)
                    }
                }
            }
        }
        var extraAxis: [UInt8] = []
        for _ in 0..<3 {   // rounds: reassignment can unlock chains
            var moved = false
            for fi in 0..<faceCount where chartIds[fi] < Int32(chartCount) {
                let n = faceUnitNrm[fi]
                guard simd_length_squared(n) > 0 else { continue }
                let axis = Int(chartAxis[Int(chartIds[fi])])
                guard abs(n[axis]) < 0.45 else { continue }   // foreshortened >2.2x for own chart
                var bestChart: Int32 = -1
                var bestCos: Float = 0.5   // require a genuinely better fit
                for k in 0..<3 {
                    let nb = nbr[fi*3 + k]
                    guard nb >= 0 else { continue }
                    let c = chartIds[Int(nb)]
                    guard c != chartIds[fi], c < Int32(chartCount) else { continue }
                    let cCos = abs(n[Int(chartAxis[Int(c)])])
                    if cCos > bestCos { bestCos = cCos; bestChart = c }
                }
                if bestChart >= 0 { chartIds[fi] = bestChart; moved = true }
            }
            if !moved { break }
        }
        for fi in 0..<faceCount where chartIds[fi] < Int32(chartCount) {
            let n = faceUnitNrm[fi]
            guard simd_length_squared(n) > 0 else { continue }
            if abs(n[Int(chartAxis[Int(chartIds[fi])])] ) < 0.45 {
                let an = simd_abs(n)
                let own: UInt8 = an.x >= an.y && an.x >= an.z ? 0 : (an.y >= an.z ? 1 : 2)
                chartIds[fi] = Int32(chartCount + extraAxis.count)
                extraAxis.append(own)
            }
        }
        if !extraAxis.isEmpty {
            chartAxis.append(contentsOf: extraAxis)
            chartCount += extraAxis.count
        }

        // Seam split: one output vertex per (original vertex, chart) pair.
        // UV = world-space projection onto the chart's axis plane; world scale
        // keeps texel density uniform across charts.
        var remap = [Int64: Int32](minimumCapacity: vertexCount * 5 / 4)
        var splitVerts: [Float] = []
        var splitUVs: [Float] = []
        var splitMap: [Int32] = []
        var splitChart: [Int32] = []
        var splitFaces = [Int32](repeating: 0, count: faceCount * 3)
        splitVerts.reserveCapacity(vertexCount * 3 * 5 / 4)

        _ = faceAxis   // axis decisions are chart-level from here on
        for fi in 0..<faceCount {
            let chart = chartIds[fi]
            let axis = Int(chartAxis[Int(chart)])   // chart-level axis (post-merge)
            let uAxis = (axis + 1) % 3
            let vAxis = (axis + 2) % 3
            for k in 0..<3 {
                let orig = f[fi*3 + k]
                let key = Int64(orig) << 24 | Int64(chart)
                let idx: Int32
                if let existing = remap[key] {
                    idx = existing
                } else {
                    idx = Int32(splitMap.count)
                    remap[key] = idx
                    let o = Int(orig) * 3
                    splitVerts.append(v[o]); splitVerts.append(v[o+1]); splitVerts.append(v[o+2])
                    splitUVs.append(v[o + uAxis]); splitUVs.append(v[o + vAxis])
                    splitMap.append(orig)
                    splitChart.append(chart)
                }
                splitFaces[fi*3 + k] = idx
            }
        }

        // Charts from different world regions can project onto IDENTICAL UV
        // coordinates (stacked surfaces along the axis). UvMeshDecl carries only
        // UVs, so xatlas would weld coincident vertices and fuse unrelated charts
        // into folded islands. Translate every chart into its own disjoint cell
        // of a coarse grid — xatlas repositions charts during packing anyway.
        var minU = [Float](repeating: .greatestFiniteMagnitude, count: chartCount)
        var minV = [Float](repeating: .greatestFiniteMagnitude, count: chartCount)
        var maxU = [Float](repeating: -.greatestFiniteMagnitude, count: chartCount)
        var maxV = [Float](repeating: -.greatestFiniteMagnitude, count: chartCount)
        for i in 0..<splitMap.count {
            let c = Int(splitChart[i])
            minU[c] = min(minU[c], splitUVs[i*2]);     maxU[c] = max(maxU[c], splitUVs[i*2])
            minV[c] = min(minV[c], splitUVs[i*2 + 1]); maxV[c] = max(maxV[c], splitUVs[i*2 + 1])
        }
        // Minimum chart footprint: charts far smaller than the mean pack to
        // sub-texel islands that die in the renderer's mip chain. Scale tiny
        // charts up to a floor tied to the total UV area (≈2 texels at a
        // 1024-class atlas); they are small in 3D, so the extra texels are
        // effectively free.
        var totalUVArea: Float = 0
        for c in 0..<chartCount where maxU[c] >= minU[c] {
            totalUVArea += (maxU[c] - minU[c]) * (maxV[c] - minV[c])
        }
        let minExtent = totalUVArea.squareRoot() / 512
        var chartScale = [Float](repeating: 1, count: chartCount)
        for c in 0..<chartCount where maxU[c] >= minU[c] {
            let ext = max(maxU[c] - minU[c], maxV[c] - minV[c])
            if ext > 0, ext < minExtent { chartScale[c] = minExtent / ext }
        }

        var pitch: Float = 0
        for c in 0..<chartCount where maxU[c] >= minU[c] {
            pitch = max(pitch, max(maxU[c] - minU[c], maxV[c] - minV[c]) * chartScale[c])
        }
        pitch *= 1.05
        if pitch <= 0 { pitch = 1 }
        let cols = Int(Float(chartCount).squareRoot().rounded(.up))
        for i in 0..<splitMap.count {
            let c = Int(splitChart[i])
            let cellU = Float(c % cols) * pitch
            let cellV = Float(c / cols) * pitch
            splitUVs[i*2]     = (splitUVs[i*2]     - minU[c]) * chartScale[c] + cellU
            splitUVs[i*2 + 1] = (splitUVs[i*2 + 1] - minV[c]) * chartScale[c] + cellV
        }

        let splitMesh = Mesh(
            vertices: MLXArray(splitVerts, [splitMap.count, 3]),
            faces: MLXArray(splitFaces, [faceCount, 3])
        )
        var unwrap = try splitMesh.uvUnwrap(
            existingUVs: MLXArray(splitUVs, [splitMap.count, 2]),
            faceMaterials: chartIds.map { UInt32(bitPattern: $0) },
            packOptions: packOptions
        )

        // Safety net: any face still spanning the atlas (xatlas dropped it from
        // its chart and left vertices unpacked — degenerate slivers) gets its
        // UVs collapsed to a point: zero extent, never rasterized, no streak.
        // The face's vertices are DUPLICATED first — collapsing shared vertices
        // in place would re-stretch healthy neighbouring faces.
        var outF = unwrap.mesh.faces.asArray(Int32.self)
        var outUV = unwrap.uvs.asArray(Float.self)
        var outV = unwrap.mesh.vertices.asArray(Float.self)
        var outMap = unwrap.vertexMap.asArray(Int32.self)
        var collapsed = false
        for fi in 0..<(outF.count / 3) {
            let i0 = Int(outF[fi*3]), i1 = Int(outF[fi*3+1]), i2 = Int(outF[fi*3+2])
            let us = [outUV[i0*2], outUV[i1*2], outUV[i2*2]]
            let ws = [outUV[i0*2+1], outUV[i1*2+1], outUV[i2*2+1]]
            if us.max()! - us.min()! > 0.05 || ws.max()! - ws.min()! > 0.05 {
                for (slot, src) in [(1, i1), (2, i2)] {
                    let newIdx = Int32(outMap.count)
                    outV.append(outV[src*3]); outV.append(outV[src*3+1]); outV.append(outV[src*3+2])
                    outUV.append(outUV[i0*2]); outUV.append(outUV[i0*2+1])
                    outMap.append(outMap[src])
                    outF[fi*3 + slot] = newIdx
                }
                collapsed = true
            }
        }
        if collapsed {
            let V = outMap.count
            unwrap = UVUnwrapResult(
                mesh: Mesh(vertices: MLXArray(outV, [V, 3]),
                           faces: MLXArray(outF, [outF.count / 3, 3])),
                uvs: MLXArray(outUV, [V, 2]),
                vertexMap: MLXArray(outMap, [V]), atlasWidth: unwrap.atlasWidth,
                atlasHeight: unwrap.atlasHeight, charts: unwrap.charts)
        }
        return ProvenanceUnwrapResult(
            unwrap: unwrap, splitVertexMap: splitMap, chartCount: chartCount)
    }
}
