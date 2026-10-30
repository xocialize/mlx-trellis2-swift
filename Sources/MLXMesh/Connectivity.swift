import Foundation
import MLX

/// Undirected edges of a triangle mesh, deduplicated and annotated with
/// face-incidence counts.
public struct EdgeTable {
    /// `[E, 2]` int32 — each row `(lo, hi)` with `lo < hi`, sorted ascending by key.
    public let edges: MLXArray
    /// `[E]` int32 — number of faces incident to each edge.
    /// `1` → boundary, `2` → manifold interior, `≥3` → non-manifold.
    public let edgeFaceCount: MLXArray

    public var count: Int { edges.dim(0) }

    /// Boolean mask `[E]` selecting boundary edges (incidence == 1).
    public var boundaryMask: MLXArray { edgeFaceCount .== MLXArray(Int32(1)) }
    /// Boolean mask `[E]` selecting non-manifold edges (incidence > 2).
    public var nonManifoldMask: MLXArray { edgeFaceCount .> MLXArray(Int32(2)) }
}

extension Mesh {
    /// Build the undirected edge table.
    ///
    /// NOTE: First-pass implementation. Pulls `faces` to CPU and performs the
    /// canonicalize/sort/dedupe pass in Swift. Correct and simple; will be
    /// replaced by a GPU hash-table port of `src/hash/` from upstream CuMesh
    /// once topology workloads need it.
    public func edgeTable() -> EdgeTable {
        let f = faces.asArray(Int32.self)
        let fcount = faceCount
        let stride = UInt64(vertexCount) + 1  // separator for packing (lo, hi) → key

        var keys: [UInt64] = []
        keys.reserveCapacity(fcount * 3)
        for fi in 0..<fcount {
            let i0 = UInt64(UInt32(bitPattern: f[fi * 3 + 0]))
            let i1 = UInt64(UInt32(bitPattern: f[fi * 3 + 1]))
            let i2 = UInt64(UInt32(bitPattern: f[fi * 3 + 2]))
            for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
                let lo = min(a, b), hi = max(a, b)
                keys.append(lo * stride + hi)
            }
        }
        keys.sort()

        var edgesFlat: [Int32] = []
        var counts: [Int32] = []
        var i = 0
        while i < keys.count {
            var j = i + 1
            while j < keys.count && keys[j] == keys[i] { j += 1 }
            let lo = Int32(keys[i] / stride)
            let hi = Int32(keys[i] % stride)
            edgesFlat.append(lo)
            edgesFlat.append(hi)
            counts.append(Int32(j - i))
            i = j
        }
        let e = counts.count
        return EdgeTable(
            edges: MLXArray(edgesFlat, [e, 2]),
            edgeFaceCount: MLXArray(counts, [e])
        )
    }

    /// Dense connected-component IDs per vertex, `[V]` int32, values in `0..<K`.
    /// Isolated vertices (no incident edge) get their own component.
    ///
    /// First-pass CPU union-find; will move to a parallel GPU pass once it
    /// shows up in a profile.
    public func connectedComponents(edgeTable: EdgeTable? = nil) -> MLXArray {
        let table = edgeTable ?? self.edgeTable()
        let edges = table.edges.asArray(Int32.self)
        let V = vertexCount

        var parent = Array(0..<V)
        var rank = Array(repeating: 0, count: V)

        func find(_ x: Int) -> Int {
            var r = x
            while parent[r] != r { r = parent[r] }
            // path compression
            var c = x
            while parent[c] != r {
                let next = parent[c]
                parent[c] = r
                c = next
            }
            return r
        }

        func union(_ a: Int, _ b: Int) {
            let ra = find(a), rb = find(b)
            if ra == rb { return }
            if rank[ra] < rank[rb] {
                parent[ra] = rb
            } else if rank[ra] > rank[rb] {
                parent[rb] = ra
            } else {
                parent[rb] = ra
                rank[ra] += 1
            }
        }

        for i in stride(from: 0, to: edges.count, by: 2) {
            union(Int(edges[i]), Int(edges[i + 1]))
        }

        // Densify: map each root to a dense [0..K) id in first-seen order.
        var rootToDense: [Int: Int32] = [:]
        var labels = [Int32](repeating: 0, count: V)
        var next: Int32 = 0
        for v in 0..<V {
            let r = find(v)
            if let id = rootToDense[r] {
                labels[v] = id
            } else {
                rootToDense[r] = next
                labels[v] = next
                next += 1
            }
        }
        return MLXArray(labels, [V])
    }

    /// Number of connected components, derived from `connectedComponents()`.
    public func componentCount(edgeTable: EdgeTable? = nil) -> Int {
        let labels = connectedComponents(edgeTable: edgeTable).asArray(Int32.self)
        return Int((labels.max() ?? -1) + 1)
    }

    /// Trace boundary edges into vertex cycles.
    ///
    /// A boundary edge is an undirected edge used by exactly one face (the
    /// `EdgeTable` definition), so two neighbours with opposite winding don't
    /// open a false boundary. Each loop runs the way most of its edges run in
    /// their faces, which on a consistently wound mesh is the face winding.
    ///
    /// Where several boundary edges meet at one vertex (two holes sharing a
    /// vertex, or a non-manifold vertex) the walk splits off a simple cycle
    /// whenever it comes back to a vertex already on its path, so every loop
    /// is closed and visits each vertex once. Boundary edges that close no
    /// cycle (odd boundary degree at non-manifold vertices) form no loop.
    public func boundaryLoops() -> BoundaryLoops {
        let c = boundaryCycles()
        let vertsArray = c.vertices.isEmpty
            ? MLXArray([] as [Int32], [0])
            : MLXArray(c.vertices, [c.vertices.count])
        return BoundaryLoops(
            vertices: vertsArray,
            offsets: MLXArray(c.offsets, [c.offsets.count])
        )
    }

    /// `boundaryLoops()` on the CPU, plus, for loop edge `i` (from
    /// `vertices[i]` to the next vertex of its loop), whether its face runs it
    /// the same way. `fillHoles` winds each new triangle by that.
    func boundaryCycles() -> (vertices: [Int32], offsets: [Int32], faceForward: [Bool]) {
        let f = faces.asArray(Int32.self)
        let stride = UInt64(vertexCount) + 1

        // Undirected key -> faces using it, and whether the last one ran it lo→hi.
        var use: [UInt64: (count: Int32, forward: Bool)] = [:]
        use.reserveCapacity(faceCount * 3 / 2)
        for fi in 0..<faceCount {
            for k in 0..<3 {
                let a = UInt64(UInt32(bitPattern: f[fi * 3 + k]))
                let b = UInt64(UInt32(bitPattern: f[fi * 3 + (k + 1) % 3]))
                if a == b { continue }  // degenerate face
                let key = min(a, b) * stride + max(a, b)
                let n = use[key]?.count ?? 0
                use[key] = (n + 1, a < b)
            }
        }

        // Boundary edges in key order (deterministic), and per-vertex incidence.
        var lo: [Int32] = [], hi: [Int32] = [], forward: [Bool] = []
        for (key, u) in use where u.count == 1 {
            lo.append(Int32(key / stride)); hi.append(Int32(key % stride)); forward.append(u.forward)
        }
        let order = lo.indices.sorted { (lo[$0], hi[$0]) < (lo[$1], hi[$1]) }
        lo = order.map { lo[$0] }; hi = order.map { hi[$0] }; forward = order.map { forward[$0] }
        var incident: [Int32: [Int]] = [:]
        for e in lo.indices {
            incident[lo[e], default: []].append(e)
            incident[hi[e], default: []].append(e)
        }

        var used = [Bool](repeating: false, count: lo.count)
        var cursor: [Int32: Int] = [:]
        func nextEdge(_ v: Int32) -> Int? {
            guard let edges = incident[v] else { return nil }
            var i = cursor[v] ?? 0
            while i < edges.count && used[edges[i]] { i += 1 }
            cursor[v] = i
            return i < edges.count ? edges[i] : nil
        }
        // Does a face run a→b along boundary edge e?
        func runs(_ e: Int, from a: Int32) -> Bool { forward[e] == (a == lo[e]) }

        var loopVerts: [Int32] = [], faceForward: [Bool] = [], offsets: [Int32] = [0]
        for e0 in lo.indices where !used[e0] {
            // Walk unused boundary edges from lo[e0]; path holds the open walk and
            // pathEdges the edge leaving each of its vertices.
            var path: [Int32] = [lo[e0]], pathEdges: [Int] = []
            var position: [Int32: Int] = [lo[e0]: 0]
            var u = lo[e0]
            while true {
                guard let e = nextEdge(u) else {
                    // Dead end (an open chain): back up one edge and try again from there.
                    guard path.count > 1 else { break }
                    position[path.removeLast()] = nil
                    pathEdges.removeLast()
                    u = path[path.count - 1]
                    continue
                }
                used[e] = true
                let w = lo[e] == u ? hi[e] : lo[e]
                pathEdges.append(e)
                guard let p = position[w] else {
                    position[w] = path.count
                    path.append(w)
                    u = w
                    continue
                }
                // Back at w: path[p...] is a simple cycle. Emit it, keep walking from w.
                var cycle = Array(path[p...])
                var dirs = (p..<path.count).map { runs(pathEdges[$0], from: path[$0]) }
                if dirs.filter({ $0 }).count * 2 < dirs.count {
                    // Most edges run the other way: reverse so the loop follows the faces.
                    let edgesBack = Array(pathEdges[p...].reversed())
                    cycle = [cycle[0]] + cycle[1...].reversed()
                    dirs = cycle.indices.map { runs(edgesBack[$0], from: cycle[$0]) }
                }
                loopVerts += cycle
                faceForward += dirs
                offsets.append(Int32(loopVerts.count))
                for x in path[(p + 1)...] { position[x] = nil }
                path.removeSubrange((p + 1)...)
                pathEdges.removeSubrange(p...)
                u = w
            }
        }
        return (loopVerts, offsets, faceForward)
    }
}

/// Per-vertex 1-ring (incident neighbors) in CSR layout.
/// Neighbors of vertex `v` are `neighbors[offsets[v]..<offsets[v+1]]`.
public struct VertexAdjacency {
    /// `[2E]` int32 — each undirected edge contributes both endpoints.
    public let neighbors: MLXArray
    /// `[V + 1]` int32 — prefix offsets into `neighbors`.
    public let offsets: MLXArray

    public var vertexCount: Int { offsets.dim(0) - 1 }

    public func neighborsOf(_ v: Int) -> [Int32] {
        let offs = offsets.asArray(Int32.self)
        let all = neighbors.asArray(Int32.self)
        return Array(all[Int(offs[v])..<Int(offs[v + 1])])
    }
}

extension Mesh {
    /// Build per-vertex 1-ring adjacency from the edge table.
    public func vertexAdjacency(edgeTable: EdgeTable? = nil) -> VertexAdjacency {
        let table = edgeTable ?? self.edgeTable()
        let edges = table.edges.asArray(Int32.self)
        let E = table.count
        let V = vertexCount

        // Per-vertex degree.
        var counts = [Int](repeating: 0, count: V)
        for i in 0..<E {
            counts[Int(edges[i * 2])] += 1
            counts[Int(edges[i * 2 + 1])] += 1
        }

        // Prefix-sum to offsets.
        var offsets = [Int32](repeating: 0, count: V + 1)
        for v in 0..<V {
            offsets[v + 1] = offsets[v] + Int32(counts[v])
        }

        // Scatter neighbours using a moving cursor per vertex.
        var cursors = (0..<V).map { Int(offsets[$0]) }
        var neighbors = [Int32](repeating: 0, count: 2 * E)
        for i in 0..<E {
            let a = Int(edges[i * 2])
            let b = Int(edges[i * 2 + 1])
            neighbors[cursors[a]] = Int32(b); cursors[a] += 1
            neighbors[cursors[b]] = Int32(a); cursors[b] += 1
        }

        let neighArr = neighbors.isEmpty
            ? MLXArray([] as [Int32], [0])
            : MLXArray(neighbors, [neighbors.count])
        return VertexAdjacency(
            neighbors: neighArr,
            offsets: MLXArray(offsets, [offsets.count])
        )
    }
}

/// Per-vertex incident-face list in CSR layout.
public struct VertexFaceAdjacency {
    /// `[3F]` int32 — face indices, packed CSR per vertex.
    public let faceIndices: MLXArray
    /// `[V + 1]` int32 — start offsets into `faceIndices`.
    public let offsets: MLXArray

    public var vertexCount: Int { offsets.dim(0) - 1 }

    public func facesOf(_ v: Int) -> [Int32] {
        let offs = offsets.asArray(Int32.self)
        let all = faceIndices.asArray(Int32.self)
        return Array(all[Int(offs[v])..<Int(offs[v + 1])])
    }
}

/// Per-edge incident-face list in CSR layout. References edges of an
/// `EdgeTable`. For a manifold mesh every edge has 1 (boundary) or 2
/// (interior) incident faces; for non-manifold edges this can be larger.
public struct EdgeFaceAdjacency {
    /// `[totalFaces]` int32 — face indices, packed CSR per edge.
    public let faceIndices: MLXArray
    /// `[E + 1]` int32 — start offsets into `faceIndices`.
    public let offsets: MLXArray

    public var edgeCount: Int { offsets.dim(0) - 1 }

    public func facesOf(_ e: Int) -> [Int32] {
        let offs = offsets.asArray(Int32.self)
        let all = faceIndices.asArray(Int32.self)
        return Array(all[Int(offs[e])..<Int(offs[e + 1])])
    }
}

/// Per-vertex incident-boundary-edge list in CSR layout. References edges
/// of an `EdgeTable` filtered to the boundary subset.
public struct VertexBoundaryAdjacency {
    /// Flat int32 — boundary-edge indices (into the parent edge table), CSR per vertex.
    public let boundaryEdges: MLXArray
    /// `[V + 1]` int32 — start offsets.
    public let offsets: MLXArray

    public var vertexCount: Int { offsets.dim(0) - 1 }

    public func boundaryEdgesOf(_ v: Int) -> [Int32] {
        let offs = offsets.asArray(Int32.self)
        let all = boundaryEdges.asArray(Int32.self)
        return Array(all[Int(offs[v])..<Int(offs[v + 1])])
    }
}

extension Mesh {
    /// Per-vertex CSR of incident face indices. Matches
    /// `cumesh.get_vertex_face_adjacency` (cached) + downstream reads.
    public func vertexFaceAdjacency() -> VertexFaceAdjacency {
        let f = faces.asArray(Int32.self)
        let V = vertexCount, F = faceCount
        var counts = [Int](repeating: 0, count: V)
        for fi in 0..<F {
            counts[Int(f[fi*3 + 0])] += 1
            counts[Int(f[fi*3 + 1])] += 1
            counts[Int(f[fi*3 + 2])] += 1
        }
        var offsets = [Int32](repeating: 0, count: V + 1)
        for v in 0..<V { offsets[v + 1] = offsets[v] + Int32(counts[v]) }
        var cursors = (0..<V).map { Int(offsets[$0]) }
        var faceIdx = [Int32](repeating: 0, count: 3 * F)
        for fi in 0..<F {
            for k in 0..<3 {
                let v = Int(f[fi*3 + k])
                faceIdx[cursors[v]] = Int32(fi); cursors[v] += 1
            }
        }
        let arr = faceIdx.isEmpty
            ? MLXArray([] as [Int32], [0])
            : MLXArray(faceIdx, [faceIdx.count])
        return VertexFaceAdjacency(
            faceIndices: arr,
            offsets: MLXArray(offsets, [offsets.count])
        )
    }

    /// Per-edge CSR of incident face indices. Matches
    /// `cumesh.get_edge_face_adjacency`.
    public func edgeFaceAdjacency(edgeTable: EdgeTable? = nil) -> EdgeFaceAdjacency {
        let table = edgeTable ?? self.edgeTable()
        let f = faces.asArray(Int32.self)
        let edgesArr = table.edges.asArray(Int32.self)
        let stride = UInt64(vertexCount) + 1

        // canonical-key → edge-table index
        var edgeIdxMap = [UInt64: Int](minimumCapacity: table.count)
        for i in 0..<table.count {
            let lo = UInt64(UInt32(bitPattern: edgesArr[i*2]))
            let hi = UInt64(UInt32(bitPattern: edgesArr[i*2 + 1]))
            edgeIdxMap[lo * stride + hi] = i
        }

        // First pass: counts per edge.
        var counts = [Int](repeating: 0, count: table.count)
        for fi in 0..<faceCount {
            for k in 0..<3 {
                let a = UInt64(UInt32(bitPattern: f[fi*3 + k]))
                let b = UInt64(UInt32(bitPattern: f[fi*3 + (k + 1) % 3]))
                let lo = min(a, b), hi = max(a, b)
                counts[edgeIdxMap[lo * stride + hi]!] += 1
            }
        }
        var offsets = [Int32](repeating: 0, count: table.count + 1)
        for e in 0..<table.count { offsets[e + 1] = offsets[e] + Int32(counts[e]) }
        let total = Int(offsets.last ?? 0)

        var cursors = (0..<table.count).map { Int(offsets[$0]) }
        var faceIdx = [Int32](repeating: 0, count: total)
        for fi in 0..<faceCount {
            for k in 0..<3 {
                let a = UInt64(UInt32(bitPattern: f[fi*3 + k]))
                let b = UInt64(UInt32(bitPattern: f[fi*3 + (k + 1) % 3]))
                let lo = min(a, b), hi = max(a, b)
                let e = edgeIdxMap[lo * stride + hi]!
                faceIdx[cursors[e]] = Int32(fi); cursors[e] += 1
            }
        }

        let arr = faceIdx.isEmpty
            ? MLXArray([] as [Int32], [0])
            : MLXArray(faceIdx, [faceIdx.count])
        return EdgeFaceAdjacency(
            faceIndices: arr,
            offsets: MLXArray(offsets, [offsets.count])
        )
    }

    /// Per-vertex CSR of incident boundary-edge indices. Matches
    /// `cumesh.get_vertex_boundary_adjacency`.
    public func vertexBoundaryAdjacency(edgeTable: EdgeTable? = nil) -> VertexBoundaryAdjacency {
        let table = edgeTable ?? self.edgeTable()
        let bIdxs = boundaryEdgeIndices(edgeTable: table).asArray(Int32.self)
        let edgesArr = table.edges.asArray(Int32.self)
        let V = vertexCount

        var counts = [Int](repeating: 0, count: V)
        for b in bIdxs {
            let a0 = Int(edgesArr[Int(b)*2])
            let a1 = Int(edgesArr[Int(b)*2 + 1])
            counts[a0] += 1; counts[a1] += 1
        }
        var offsets = [Int32](repeating: 0, count: V + 1)
        for v in 0..<V { offsets[v + 1] = offsets[v] + Int32(counts[v]) }

        var cursors = (0..<V).map { Int(offsets[$0]) }
        var out = [Int32](repeating: 0, count: 2 * bIdxs.count)
        for b in bIdxs {
            let a0 = Int(edgesArr[Int(b)*2])
            let a1 = Int(edgesArr[Int(b)*2 + 1])
            out[cursors[a0]] = b; cursors[a0] += 1
            out[cursors[a1]] = b; cursors[a1] += 1
        }

        let arr = out.isEmpty
            ? MLXArray([] as [Int32], [0])
            : MLXArray(out, [out.count])
        return VertexBoundaryAdjacency(
            boundaryEdges: arr,
            offsets: MLXArray(offsets, [offsets.count])
        )
    }

    /// `[B]` int32 — indices (into `edgeTable.edges`) of boundary edges.
    /// Matches `cumesh.read_boundaries`.
    public func boundaryEdgeIndices(edgeTable: EdgeTable? = nil) -> MLXArray {
        let table = edgeTable ?? self.edgeTable()
        let counts = table.edgeFaceCount.asArray(Int32.self)
        var out: [Int32] = []
        out.reserveCapacity(counts.count / 4)
        for (i, c) in counts.enumerated() where c == 1 {
            out.append(Int32(i))
        }
        return out.isEmpty
            ? MLXArray([] as [Int32], [0])
            : MLXArray(out, [out.count])
    }

    /// `[M, 2]` int32 — `(face_a, face_b)` pairs of faces meeting at each
    /// manifold (count == 2) edge. Matches `cumesh.read_manifold_face_adjacency`.
    public func manifoldFaceAdjacency(edgeTable: EdgeTable? = nil) -> MLXArray {
        let table = edgeTable ?? self.edgeTable()
        let efa = edgeFaceAdjacency(edgeTable: table)
        let counts = table.edgeFaceCount.asArray(Int32.self)
        let offs = efa.offsets.asArray(Int32.self)
        let faceIdx = efa.faceIndices.asArray(Int32.self)
        var pairs: [Int32] = []
        for e in 0..<table.count where counts[e] == 2 {
            pairs.append(faceIdx[Int(offs[e])])
            pairs.append(faceIdx[Int(offs[e]) + 1])
        }
        return pairs.isEmpty
            ? MLXArray([] as [Int32], [0, 2])
            : MLXArray(pairs, [pairs.count / 2, 2])
    }

    /// `[M, 2]` int32 — `(boundary_edge_0, boundary_edge_1)` pairs at each
    /// "manifold boundary vertex" (a boundary vertex with exactly 2 incident
    /// boundary edges — typical for interior of a boundary loop, not for
    /// branch points). Matches `cumesh.read_manifold_boundary_adjacency`.
    public func manifoldBoundaryAdjacency(edgeTable: EdgeTable? = nil) -> MLXArray {
        let table = edgeTable ?? self.edgeTable()
        let vba = vertexBoundaryAdjacency(edgeTable: table)
        let offs = vba.offsets.asArray(Int32.self)
        let bIdx = vba.boundaryEdges.asArray(Int32.self)
        var pairs: [Int32] = []
        for v in 0..<vertexCount {
            let s = Int(offs[v]), e = Int(offs[v + 1])
            if e - s == 2 {
                pairs.append(bIdx[s])
                pairs.append(bIdx[s + 1])
            }
        }
        return pairs.isEmpty
            ? MLXArray([] as [Int32], [0, 2])
            : MLXArray(pairs, [pairs.count / 2, 2])
    }

    /// Connected components considering only the boundary subgraph.
    /// Returns `(count, [B] int32 component IDs per boundary edge)`.
    /// Matches `cumesh.read_boundary_connected_components`.
    public func boundaryConnectedComponents(edgeTable: EdgeTable? = nil) -> (count: Int, labels: MLXArray) {
        let table = edgeTable ?? self.edgeTable()
        let bIdxs = boundaryEdgeIndices(edgeTable: table).asArray(Int32.self)
        let B = bIdxs.count
        if B == 0 { return (0, MLXArray([] as [Int32], [0])) }

        let edgesArr = table.edges.asArray(Int32.self)
        // Build edge-id → position in `bIdxs` for the union-find.
        var bPos = [Int32: Int](minimumCapacity: B)
        for (k, e) in bIdxs.enumerated() { bPos[e] = k }

        // Boundary edges share a component iff they touch the same vertex.
        var parent = Array(0..<B)
        var rank = Array(repeating: 0, count: B)
        func find(_ x: Int) -> Int {
            var r = x; while parent[r] != r { r = parent[r] }
            var c = x; while parent[c] != r {
                let next = parent[c]; parent[c] = r; c = next
            }
            return r
        }
        func union(_ a: Int, _ b: Int) {
            let ra = find(a), rb = find(b); if ra == rb { return }
            if rank[ra] < rank[rb] { parent[ra] = rb }
            else if rank[ra] > rank[rb] { parent[rb] = ra }
            else { parent[rb] = ra; rank[ra] += 1 }
        }

        // For each vertex, union all incident boundary edges.
        let vba = vertexBoundaryAdjacency(edgeTable: table)
        let offs = vba.offsets.asArray(Int32.self)
        let incBE = vba.boundaryEdges.asArray(Int32.self)
        for v in 0..<vertexCount {
            let s = Int(offs[v]), e = Int(offs[v + 1])
            if e - s < 2 { continue }
            let first = bPos[incBE[s]]!
            for i in (s + 1)..<e {
                union(first, bPos[incBE[i]]!)
            }
        }
        _ = edgesArr  // silence unused-warning placeholder

        // Densify labels.
        var rootToDense = [Int: Int32]()
        var labels = [Int32](repeating: 0, count: B)
        var nextDense: Int32 = 0
        for k in 0..<B {
            let r = find(k)
            if let d = rootToDense[r] { labels[k] = d }
            else { rootToDense[r] = nextDense; labels[k] = nextDense; nextDense += 1 }
        }
        return (Int(nextDense), MLXArray(labels, [B]))
    }
}

/// Per-vertex incident-edge list in CSR layout. Built from an existing edge
/// table; the edges referenced are indices into that table's rows.
public struct VertexEdgeAdjacency {
    /// `[2E]` int32 — edge indices, packed CSR per vertex.
    public let edgeIndices: MLXArray
    /// `[V + 1]` int32 — start offsets into `edgeIndices`.
    public let offsets: MLXArray

    public var vertexCount: Int { offsets.dim(0) - 1 }

    public func edgesOf(_ v: Int) -> [Int32] {
        let offs = offsets.asArray(Int32.self)
        let all = edgeIndices.asArray(Int32.self)
        return Array(all[Int(offs[v])..<Int(offs[v + 1])])
    }
}

extension Mesh {
    /// CSR adjacency mapping each vertex to the edges (in `edgeTable`) it
    /// participates in. Built by degree-histogram + cursor-scatter, just like
    /// `vertexAdjacency()`.
    public func vertexEdgeAdjacency(edgeTable: EdgeTable? = nil) -> VertexEdgeAdjacency {
        let table = edgeTable ?? self.edgeTable()
        let edges = table.edges.asArray(Int32.self)
        let E = table.count
        let V = vertexCount

        var counts = [Int](repeating: 0, count: V)
        for i in 0..<E {
            counts[Int(edges[i*2])] += 1
            counts[Int(edges[i*2 + 1])] += 1
        }
        var offsets = [Int32](repeating: 0, count: V + 1)
        for v in 0..<V {
            offsets[v + 1] = offsets[v] + Int32(counts[v])
        }

        var cursors = (0..<V).map { Int(offsets[$0]) }
        var edgeIdx = [Int32](repeating: 0, count: 2 * E)
        for i in 0..<E {
            let a = Int(edges[i*2])
            let b = Int(edges[i*2 + 1])
            edgeIdx[cursors[a]] = Int32(i); cursors[a] += 1
            edgeIdx[cursors[b]] = Int32(i); cursors[b] += 1
        }

        let arr = edgeIdx.isEmpty
            ? MLXArray([] as [Int32], [0])
            : MLXArray(edgeIdx, [edgeIdx.count])
        return VertexEdgeAdjacency(
            edgeIndices: arr,
            offsets: MLXArray(offsets, [offsets.count])
        )
    }
}

/// Boundary loops in CSR layout. Loop `i` is `vertices[offsets[i]..<offsets[i+1]]`.
public struct BoundaryLoops {
    /// `[totalVerts]` int32 — flattened, in walk order.
    public let vertices: MLXArray
    /// `[count + 1]` int32 — start of each loop; `offsets[count]` == `vertices.count`.
    public let offsets: MLXArray

    public var count: Int { offsets.dim(0) - 1 }

    /// Pull loop `i` to a Swift array. Use sparingly — issues an MLX eval.
    public func loop(_ i: Int) -> [Int32] {
        let offs = offsets.asArray(Int32.self)
        let all = vertices.asArray(Int32.self)
        return Array(all[Int(offs[i])..<Int(offs[i + 1])])
    }
}
