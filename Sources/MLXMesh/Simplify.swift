import Foundation
import MLX

extension Mesh {
    /// Reduce the mesh to roughly `targetNumFaces` faces via QEM-driven
    /// parallel edge collapse. Boundary edges are protected (never collapsed).
    ///
    /// Loop body, per iteration:
    /// 1. Recompute QEM, edge table, edge cost, vertex-edge adjacency.
    /// 2. Mask out edges violating the interior link condition (endpoints
    ///    must share exactly two neighbour vertices), then pick an
    ///    independent set of the rest to collapse (Luby, cost-weighted).
    /// 3. Apply collapses CPU-side: move `a` to `v*`, redirect `b → a`,
    ///    invalidate now-degenerate faces, compact.
    ///
    /// Stops when face count ≤ target, no progress is made, or
    /// `maxIterations` is hit. The CPU-side coordination is intentional for
    /// v0.3.0 — the per-edge cost and Luby kernels are GPU; the topology
    /// surgery is small (proportional to IS size, not F). v0.3.1 will move
    /// the surgery to Metal as needed.
    public func simplify(targetNumFaces: Int, maxIterations: Int = 64) -> Mesh {
        var current = self
        for iter in 0..<maxIterations {
            if current.faceCount <= targetNumFaces { return current }
            let next = current.simplifyOneIteration(iter: iter)
            if next.faceCount == current.faceCount { return current }  // no progress
            if next.faceCount == 0 { return next }
            current = next
        }
        return current
    }

    /// One simplification round. Public for testing / step-by-step usage.
    public func simplifyOneIteration(iter: Int) -> Mesh {
        let table = edgeTable()
        if table.count == 0 { return self }

        let Q = vertexQuadrics()
        let (rawCost, targetPos) = edgeCollapseCosts(quadrics: Q, edgeTable: table)
        let vea = vertexEdgeAdjacency(edgeTable: table)
        let edges = table.edges.asArray(Int32.self)

        // Link condition for an interior edge (a, b): the endpoints must
        // share exactly two neighbour vertices — the two face-opposite ones.
        // Collapsing an edge whose endpoints share more folds two distinct
        // edges into one, leaving a >2-face (non-manifold) edge. Violating
        // edges are excluded from this round's candidates (+inf cost) rather
        // than skipped at apply time, so their neighbours can still win the
        // Luby selection and the iteration keeps making progress; the next
        // iteration re-evaluates them on the updated topology.
        var costArr = rawCost.asArray(Float.self)
        let vAdj = vertexAdjacency(edgeTable: table)
        let nbr = vAdj.neighbors.asArray(Int32.self)
        let off = vAdj.offsets.asArray(Int32.self)
        for e in 0..<table.count where costArr[e].isFinite {
            let a = Int(edges[e*2 + 0])
            let b = Int(edges[e*2 + 1])
            var common = 0
            for i in Int(off[a])..<Int(off[a + 1]) {
                let w = nbr[i]
                for j in Int(off[b])..<Int(off[b + 1]) where nbr[j] == w {
                    common += 1
                    break
                }
            }
            if common != 2 { costArr[e] = .infinity }
        }
        let cost = MLXArray(costArr, [table.count])

        let selectedMask = selectIndependentEdges(
            edgeTable: table, cost: cost, vertexEdgeAdj: vea, iter: iter
        )

        let selected = selectedMask.asArray(Bool.self)
        let tgt = targetPos.asArray(Float.self)
        var verts = vertices.asArray(Float.self)
        var facesArr = faces.asArray(Int32.self)

        // Vertex redirect: identity by default. For each selected edge (a, b),
        // move a to v* and direct b → a.
        //
        // Greedy 1-ring lock: accept a selected collapse only when neither
        // endpoint nor any of their neighbours is an endpoint of an already-
        // accepted collapse this iteration. The Luby IS keeps selected edges
        // from sharing vertices, but two collapses at distance one (endpoints
        // joined by a mesh edge) can still fold two distinct edges onto the
        // same vertex pair, leaving a >2-face edge. Accepted collapses are
        // pairwise distance ≥ 2, so they touch disjoint face sets and each is
        // exactly the serial link-condition-checked case, which preserves
        // manifoldness. Rejected edges stay for later iterations.
        var acceptedEndpoint = [Bool](repeating: false, count: vertexCount)
        var remap = (0..<vertexCount).map { Int32($0) }
        for e in 0..<table.count where selected[e] {
            let a = Int(edges[e*2 + 0])
            let b = Int(edges[e*2 + 1])
            var blocked = acceptedEndpoint[a] || acceptedEndpoint[b]
            if !blocked {
                scan: for v in [a, b] {
                    for i in Int(off[v])..<Int(off[v + 1]) where acceptedEndpoint[Int(nbr[i])] {
                        blocked = true
                        break scan
                    }
                }
            }
            if blocked { continue }
            acceptedEndpoint[a] = true
            acceptedEndpoint[b] = true
            verts[a*3 + 0] = tgt[e*3 + 0]
            verts[a*3 + 1] = tgt[e*3 + 1]
            verts[a*3 + 2] = tgt[e*3 + 2]
            remap[b] = Int32(a)
        }

        // Apply remap to face indices; flag any face that became degenerate
        // (two equal vertex indices after redirect).
        var faceValid = [Bool](repeating: true, count: faceCount)
        for fi in 0..<faceCount {
            let n0 = remap[Int(facesArr[fi*3 + 0])]
            let n1 = remap[Int(facesArr[fi*3 + 1])]
            let n2 = remap[Int(facesArr[fi*3 + 2])]
            facesArr[fi*3 + 0] = n0
            facesArr[fi*3 + 1] = n1
            facesArr[fi*3 + 2] = n2
            if n0 == n1 || n1 == n2 || n0 == n2 {
                faceValid[fi] = false
            }
        }

        let rewritten = Mesh(
            vertices: MLXArray(verts, [vertexCount, 3]),
            faces: MLXArray(facesArr, [faceCount, 3])
        )
        return rewritten.selectingFaces(MLXArray(faceValid, [faceCount]))
    }
}
