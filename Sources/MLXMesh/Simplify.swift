import Foundation
import MLX

extension Mesh {
    /// Reduce the mesh to roughly `targetNumFaces` faces via QEM-driven
    /// parallel edge collapse. Boundary edges are protected (never collapsed).
    ///
    /// Loop body, per iteration:
    /// 1. Recompute QEM, edge table, edge cost, vertex-edge adjacency.
    /// 2. Pick an independent set of edges to collapse this round (Luby,
    ///    cost-weighted).
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
        let (cost, targetPos) = edgeCollapseCosts(quadrics: Q, edgeTable: table)
        let vea = vertexEdgeAdjacency(edgeTable: table)
        let selectedMask = selectIndependentEdges(
            edgeTable: table, cost: cost, vertexEdgeAdj: vea, iter: iter
        )

        let selected = selectedMask.asArray(Bool.self)
        let edges = table.edges.asArray(Int32.self)
        let tgt = targetPos.asArray(Float.self)
        var verts = vertices.asArray(Float.self)
        var facesArr = faces.asArray(Int32.self)

        // Vertex redirect: identity by default. For each selected edge (a, b),
        // move a to v* and direct b → a.
        var remap = (0..<vertexCount).map { Int32($0) }
        for e in 0..<table.count where selected[e] {
            let a = Int(edges[e*2 + 0])
            let b = Int(edges[e*2 + 1])
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
