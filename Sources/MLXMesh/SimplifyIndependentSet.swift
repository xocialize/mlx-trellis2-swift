import Foundation
import MLX
import MLXFast

/// Pick a (large) independent set of edges to collapse this iteration via
/// cost-weighted Luby: each candidate edge gets priority `-cost + tiny jitter`;
/// an edge wins a round if its priority is strictly highest among its active
/// neighbours. Won edges are added to the IS; the won edges plus all their
/// neighbours are dropped from the candidate pool. ~3-6 rounds drain the pool.
///
/// Two edges *neighbour* iff they share a vertex.

private let lubyRoundKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "luby_round",
    inputNames: [
        "edges", "cost", "active",
        "vertexEdgeOffsets", "vertexEdges",
        "seed",
    ],
    outputNames: ["won"],
    source: """
        uint e = thread_position_in_grid.x;
        uint ecount = active_shape[0];
        if (e >= ecount) return;
        if (active[e] == 0) { won[e] = 0; return; }

        // Stable priority. Lower cost ⇒ higher priority; jitter via xorshift
        // hash of (edge_id, seed) for tie-breaking. Both endpoints' edges see
        // the same jitter, so comparisons are well-defined.
        uint h = (uint)e ^ ((uint)seed[0] * 2654435761u);
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;
        float jitter = (float)(h & 0xffffff) * (1.0f / 16777216.0f) * 1e-6f;
        float r_e = -cost[e] + jitter;

        int ia = edges[e*2 + 0];
        int ib = edges[e*2 + 1];

        bool win = true;

        // Scan endpoint A's edges.
        int sa = vertexEdgeOffsets[ia];
        int ea = vertexEdgeOffsets[ia + 1];
        for (int i = sa; i < ea && win; ++i) {
            int f = vertexEdges[i];
            if (f == (int)e || active[f] == 0) continue;
            uint hf = (uint)f ^ ((uint)seed[0] * 2654435761u);
            hf ^= hf << 13; hf ^= hf >> 17; hf ^= hf << 5;
            float jf = (float)(hf & 0xffffff) * (1.0f / 16777216.0f) * 1e-6f;
            float r_f = -cost[f] + jf;
            if (r_f > r_e) win = false;
        }
        // Scan endpoint B's edges if still winning.
        if (win) {
            int sb = vertexEdgeOffsets[ib];
            int eb = vertexEdgeOffsets[ib + 1];
            for (int i = sb; i < eb && win; ++i) {
                int f = vertexEdges[i];
                if (f == (int)e || active[f] == 0) continue;
                uint hf = (uint)f ^ ((uint)seed[0] * 2654435761u);
                hf ^= hf << 13; hf ^= hf >> 17; hf ^= hf << 5;
                float jf = (float)(hf & 0xffffff) * (1.0f / 16777216.0f) * 1e-6f;
                float r_f = -cost[f] + jf;
                if (r_f > r_e) win = false;
            }
        }

        won[e] = win ? 1 : 0;
    """
)

extension Mesh {
    /// Return a boolean `[E]` mask selecting an independent set of edges to
    /// collapse. Boundary edges (`+inf` cost) are excluded from the start.
    ///
    /// `seed` lets callers get different IS across iterations. The default is
    /// derived from `iter`.
    public func selectIndependentEdges(
        edgeTable: EdgeTable,
        cost: MLXArray,
        vertexEdgeAdj: VertexEdgeAdjacency,
        iter: Int,
        maxRounds: Int = 6
    ) -> MLXArray {
        let E = edgeTable.count
        if E == 0 { return MLXArray([] as [Bool], [0]) }

        // active = edges with finite cost (i.e. not boundary).
        let costArr = cost.asArray(Float.self)
        var activeArr = costArr.map { $0.isFinite }
        var selectedArr = [Bool](repeating: false, count: E)

        let edgesArr = edgeTable.edges.asArray(Int32.self)

        for round in 0..<maxRounds {
            if !activeArr.contains(true) { break }
            let seed = MLXArray([Int32(truncatingIfNeeded: iter &* 1_000_003 &+ round)], [1])
            let active = MLXArray(activeArr, [E])
            let tg = min(max(E, 1), 256)
            let outputs = lubyRoundKernel(
                [edgeTable.edges, cost, active,
                 vertexEdgeAdj.offsets, vertexEdgeAdj.edgeIndices,
                 seed],
                grid: (E, 1, 1),
                threadGroup: (tg, 1, 1),
                outputShapes: [[E]],
                outputDTypes: [.bool]
            )
            let won = outputs[0].asArray(Bool.self)

            // Merge wins into selected; gather blocked vertices.
            var anyWin = false
            var blocked = Set<Int32>()
            for i in 0..<E where won[i] {
                selectedArr[i] = true
                blocked.insert(edgesArr[i*2])
                blocked.insert(edgesArr[i*2 + 1])
                anyWin = true
            }
            if !anyWin { break }
            // Deactivate won edges and any edge touching a blocked vertex.
            for i in 0..<E where activeArr[i] {
                if won[i] {
                    activeArr[i] = false
                } else if blocked.contains(edgesArr[i*2]) || blocked.contains(edgesArr[i*2 + 1]) {
                    activeArr[i] = false
                }
            }
        }

        return MLXArray(selectedArr, [E])
    }
}
