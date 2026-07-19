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
    /// Per-face chart ids from provenance axis tags.
    ///
    /// - Parameter faceAxis: `[F]` values in 0...2 from `extractMeshTagged` /
    ///   `remeshDualContouringTagged`.
    /// - Returns: `[F]` chart id per face, plus the chart count.
    public func provenanceChartIds(faceAxis: [UInt8]) -> (chartIds: [Int32], chartCount: Int) {
        precondition(faceAxis.count == faceCount, "faceAxis must be per-face")
        let f = faces.asArray(Int32.self)
        let v = vertices.asArray(Float.self)

        // 6-way tag: axis*2 + (normal[axis] < 0 ? 1 : 0). Sign is derived from
        // the *final* winding (post-unify), so it reflects rendered facing.
        var tag = [UInt8](repeating: 0, count: faceCount)
        for fi in 0..<faceCount {
            let i0 = Int(f[fi*3]), i1 = Int(f[fi*3 + 1]), i2 = Int(f[fi*3 + 2])
            let ax = SIMD3<Float>(v[i0*3], v[i0*3+1], v[i0*3+2])
            let bx = SIMD3<Float>(v[i1*3], v[i1*3+1], v[i1*3+2])
            let cx = SIMD3<Float>(v[i2*3], v[i2*3+1], v[i2*3+2])
            let nrm = simd_cross(bx - ax, cx - ax)
            let a = Int(faceAxis[fi])
            tag[fi] = UInt8(a * 2 + (nrm[a] < 0 ? 1 : 0))
        }

        // Edge → incident faces (same construction as unifyFaceOrientations).
        let stride = UInt64(vertexCount) + 1
        var edgeFaces = [UInt64: (Int32, Int32)](minimumCapacity: faceCount * 3 / 2)
        for fi in 0..<faceCount {
            for k in 0..<3 {
                let a = UInt32(bitPattern: f[fi*3 + k])
                let b = UInt32(bitPattern: f[fi*3 + (k + 1) % 3])
                let key = UInt64(min(a, b)) * stride + UInt64(max(a, b))
                if var pair = edgeFaces[key] {
                    if pair.1 < 0 { pair.1 = Int32(fi); edgeFaces[key] = pair }
                    // non-manifold extra incidences are ignored for charting
                } else {
                    edgeFaces[key] = (Int32(fi), -1)
                }
            }
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
        for (_, pair) in edgeFaces where pair.1 >= 0 {
            if tag[Int(pair.0)] == tag[Int(pair.1)] {
                let ra = find(pair.0), rb = find(pair.1)
                if ra != rb { parent[Int(ra)] = rb }
            }
        }

        var chartOf = [Int32: Int32]()
        var chartIds = [Int32](repeating: 0, count: faceCount)
        for fi in 0..<faceCount {
            let root = find(Int32(fi))
            if let c = chartOf[root] {
                chartIds[fi] = c
            } else {
                let c = Int32(chartOf.count)
                chartOf[root] = c
                chartIds[fi] = c
            }
        }
        return (chartIds, chartOf.count)
    }

    /// Full provenance unwrap: charts from tags, axis-plane UVs, seam split,
    /// then pack-only via `uvUnwrap(existingUVs:)`.
    public func provenanceUnwrap(
        faceAxis: [UInt8],
        packOptions: PackOptions = PackOptions()
    ) throws -> ProvenanceUnwrapResult {
        let (chartIds, chartCount) = provenanceChartIds(faceAxis: faceAxis)
        let f = faces.asArray(Int32.self)
        let v = vertices.asArray(Float.self)

        // Seam split: one output vertex per (original vertex, chart) pair.
        // UV = world-space projection onto the chart's axis plane; world scale
        // keeps texel density uniform across charts.
        var remap = [Int64: Int32](minimumCapacity: vertexCount * 5 / 4)
        var splitVerts: [Float] = []
        var splitUVs: [Float] = []
        var splitMap: [Int32] = []
        var splitFaces = [Int32](repeating: 0, count: faceCount * 3)
        splitVerts.reserveCapacity(vertexCount * 3 * 5 / 4)

        for fi in 0..<faceCount {
            let chart = chartIds[fi]
            let axis = Int(faceAxis[fi])
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
                }
                splitFaces[fi*3 + k] = idx
            }
        }

        let splitMesh = Mesh(
            vertices: MLXArray(splitVerts, [splitMap.count, 3]),
            faces: MLXArray(splitFaces, [faceCount, 3])
        )
        let unwrap = try splitMesh.uvUnwrap(
            existingUVs: MLXArray(splitUVs, [splitMap.count, 2]),
            packOptions: packOptions
        )
        return ProvenanceUnwrapResult(
            unwrap: unwrap, splitVertexMap: splitMap, chartCount: chartCount)
    }
}
