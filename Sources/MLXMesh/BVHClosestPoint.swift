import Foundation
import MLX
import MLXFast

/// Phase 5 — batched closest-point queries.
///
/// For each query point P, find the closest triangle in the mesh and the
/// closest point on that triangle. Stack-based best-first traversal with
/// pruning: when a sub-tree's AABB is farther than the current best, skip it.
/// Closest-point-on-triangle is Ericson "Real-Time Collision Detection"
/// §5.1.5 (7 case regions: 3 vertices, 3 edges, 1 face).

/// One closest-point result per query.
public struct ClosestPoints {
    public let faceIndices: MLXArray  // [Q] int32
    public let points: MLXArray        // [Q, 3] float32 — surface point
    public let distances: MLXArray     // [Q] float32 — Euclidean distance
}

private let closestPointKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "bvh_closest_point",
    inputNames: [
        "queries",
        "vertices", "faces", "sortedIndices",
        "nodeLeft", "nodeRight", "nodeAABB",
    ],
    outputNames: ["resultFace", "resultPoint", "resultDist"],
    source: """
        uint q = thread_position_in_grid.x;
        uint Q = queries_shape[0];
        if (q >= Q) return;
        float3 P = float3(queries[q*3 + 0], queries[q*3 + 1], queries[q*3 + 2]);

        int F = (int)sortedIndices_shape[0];
        int leafOffset = F - 1;

        int stackNode[64];
        float stackDist[64];
        int sp = 0;
        stackNode[0] = 0;
        stackDist[0] = 0.0f;
        sp = 1;

        float bestDistSq = INFINITY;
        float3 bestPoint = float3(0.0f, 0.0f, 0.0f);
        int bestFace = -1;

        // Helper: squared distance from P to AABB (0 if P is inside).
        auto aabbDistSq = [&](int n) -> float {
            float3 mn = float3(nodeAABB[n*6 + 0], nodeAABB[n*6 + 1], nodeAABB[n*6 + 2]);
            float3 mx = float3(nodeAABB[n*6 + 3], nodeAABB[n*6 + 4], nodeAABB[n*6 + 5]);
            float3 d = max(max(mn - P, P - mx), float3(0.0f));
            return dot(d, d);
        };

        while (sp > 0) {
            sp--;
            int n = stackNode[sp];
            float entryDistSq = stackDist[sp];
            if (entryDistSq >= bestDistSq) continue;

            if (n >= leafOffset) {
                int faceIdx = sortedIndices[n - leafOffset];
                int i0 = faces[faceIdx*3 + 0];
                int i1 = faces[faceIdx*3 + 1];
                int i2 = faces[faceIdx*3 + 2];
                float3 A = float3(vertices[i0*3 + 0], vertices[i0*3 + 1], vertices[i0*3 + 2]);
                float3 B = float3(vertices[i1*3 + 0], vertices[i1*3 + 1], vertices[i1*3 + 2]);
                float3 C = float3(vertices[i2*3 + 0], vertices[i2*3 + 1], vertices[i2*3 + 2]);

                // Ericson §5.1.5 — closest point on triangle ABC to P.
                float3 AB = B - A;
                float3 AC = C - A;
                float3 AP = P - A;
                float d1 = dot(AB, AP);
                float d2 = dot(AC, AP);
                float3 cp;
                if (d1 <= 0.0f && d2 <= 0.0f) { cp = A; }
                else {
                    float3 BP = P - B;
                    float d3 = dot(AB, BP);
                    float d4 = dot(AC, BP);
                    if (d3 >= 0.0f && d4 <= d3) { cp = B; }
                    else {
                        float vc = d1*d4 - d3*d2;
                        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                            float v = d1 / (d1 - d3);
                            cp = A + v * AB;
                        } else {
                            float3 CP = P - C;
                            float d5 = dot(AB, CP);
                            float d6 = dot(AC, CP);
                            if (d6 >= 0.0f && d5 <= d6) { cp = C; }
                            else {
                                float vb = d5*d2 - d1*d6;
                                if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                                    float w = d2 / (d2 - d6);
                                    cp = A + w * AC;
                                } else {
                                    float va = d3*d6 - d5*d4;
                                    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
                                        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                                        cp = B + w * (C - B);
                                    } else {
                                        float denom = 1.0f / (va + vb + vc);
                                        float v = vb * denom;
                                        float w = vc * denom;
                                        cp = A + v * AB + w * AC;
                                    }
                                }
                            }
                        }
                    }
                }
                float3 diff = cp - P;
                float dSq = dot(diff, diff);
                if (dSq < bestDistSq) {
                    bestDistSq = dSq;
                    bestPoint = cp;
                    bestFace = faceIdx;
                }
            } else {
                int lc = nodeLeft[n];
                int rc = nodeRight[n];
                float3 lmn = float3(nodeAABB[lc*6 + 0], nodeAABB[lc*6 + 1], nodeAABB[lc*6 + 2]);
                float3 lmx = float3(nodeAABB[lc*6 + 3], nodeAABB[lc*6 + 4], nodeAABB[lc*6 + 5]);
                float3 rmn = float3(nodeAABB[rc*6 + 0], nodeAABB[rc*6 + 1], nodeAABB[rc*6 + 2]);
                float3 rmx = float3(nodeAABB[rc*6 + 3], nodeAABB[rc*6 + 4], nodeAABB[rc*6 + 5]);
                float3 dl = max(max(lmn - P, P - lmx), float3(0.0f));
                float3 dr = max(max(rmn - P, P - rmx), float3(0.0f));
                float dLSq = dot(dl, dl);
                float dRSq = dot(dr, dr);

                // Push farther child first so the closer one is popped next.
                if (sp + 2 <= 64) {
                    if (dLSq < dRSq) {
                        if (dRSq < bestDistSq) { stackNode[sp] = rc; stackDist[sp] = dRSq; sp++; }
                        if (dLSq < bestDistSq) { stackNode[sp] = lc; stackDist[sp] = dLSq; sp++; }
                    } else {
                        if (dLSq < bestDistSq) { stackNode[sp] = lc; stackDist[sp] = dLSq; sp++; }
                        if (dRSq < bestDistSq) { stackNode[sp] = rc; stackDist[sp] = dRSq; sp++; }
                    }
                }
            }
        }

        resultFace[q] = bestFace;
        resultPoint[q*3 + 0] = bestPoint.x;
        resultPoint[q*3 + 1] = bestPoint.y;
        resultPoint[q*3 + 2] = bestPoint.z;
        resultDist[q] = sqrt(bestDistSq);
    """
)

extension BVH {
    /// For each query point, find the closest triangle and the point on its
    /// surface.
    public func closestPoints(mesh: Mesh, queries: MLXArray) -> ClosestPoints {
        precondition(queries.ndim == 2 && queries.dim(1) == 3,
                     "queries must be [Q, 3]; got \(queries.shape)")
        let Q = queries.dim(0)
        let tg = min(max(Q, 1), 256)
        let outputs = closestPointKernel(
            [
                queries,
                mesh.vertices, mesh.faces, topology.sortedIndices,
                topology.nodeLeft, topology.nodeRight, nodeAABB,
            ],
            grid: (Q, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[Q], [Q, 3], [Q]],
            outputDTypes: [.int32, .float32, .float32]
        )
        return ClosestPoints(
            faceIndices: outputs[0],
            points: outputs[1],
            distances: outputs[2]
        )
    }
}
