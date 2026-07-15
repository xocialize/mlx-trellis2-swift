import Foundation
import MLX
import MLXFast

/// For each edge, the optimal collapse target `v*` and the residual quadric
/// error at that point.
///
/// Math: given `Q_e = Q_a + Q_b` partitioned as `[[A, b], [bᵀ, c]]` (3×3 + 3×1
/// + scalar), the minimum of `xᵀ Q_e x` (in homogeneous coords) is at
/// `v* = -A⁻¹ b` with residual `bᵀ v* + c`. Falls back to the edge midpoint
/// when `A` is near-singular (`|det| < 1e-12`).
///
/// Boundary edges (incidence == 1) get `+infinity` cost so they're never
/// selected when boundary protection is desired. Non-protected boundary
/// handling (virtual boundary-plane quadrics) is deferred to v0.3.2.

private let edgeCollapseCostKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "edge_collapse_cost",
    inputNames: ["vertices", "edges", "edgeFaceCount", "Q"],
    outputNames: ["cost", "target"],
    source: """
        uint e = thread_position_in_grid.x;
        uint ecount = edges_shape[0];
        if (e >= ecount) return;

        int ia = edges[e*2 + 0];
        int ib = edges[e*2 + 1];

        // Boundary protection.
        if (edgeFaceCount[e] == 1) {
            cost[e] = INFINITY;
            target[e*3 + 0] = 0;
            target[e*3 + 1] = 0;
            target[e*3 + 2] = 0;
            return;
        }

        // Q_e = Q_a + Q_b (10 floats, packed symmetric 4x4).
        float q0 = Q[ia*10 + 0] + Q[ib*10 + 0];
        float q1 = Q[ia*10 + 1] + Q[ib*10 + 1];
        float q2 = Q[ia*10 + 2] + Q[ib*10 + 2];
        float q3 = Q[ia*10 + 3] + Q[ib*10 + 3];
        float q4 = Q[ia*10 + 4] + Q[ib*10 + 4];
        float q5 = Q[ia*10 + 5] + Q[ib*10 + 5];
        float q6 = Q[ia*10 + 6] + Q[ib*10 + 6];
        float q7 = Q[ia*10 + 7] + Q[ib*10 + 7];
        float q8 = Q[ia*10 + 8] + Q[ib*10 + 8];
        float q9 = Q[ia*10 + 9] + Q[ib*10 + 9];

        // A (3x3 symmetric, upper-left).
        float a00 = q0, a01 = q1, a02 = q2;
        float a11 = q4, a12 = q5;
        float a22 = q7;
        // b (top-right column of Q_e).
        float bx = q3, by = q6, bz = q8;
        // scalar c.
        float cc = q9;

        // adj(A) for symmetric 3x3.
        float adj00 =  a11*a22 - a12*a12;
        float adj01 =  a02*a12 - a01*a22;
        float adj02 =  a01*a12 - a02*a11;
        float adj11 =  a00*a22 - a02*a02;
        float adj12 =  a01*a02 - a00*a12;
        float adj22 =  a00*a11 - a01*a01;
        float det  =  a00*adj00 + a01*adj01 + a02*adj02;

        // Always-loaded endpoints — used by both the inverse-A path and the
        // midpoint fallback, and required for the sanity check below.
        float3 va = float3(vertices[ia*3 + 0], vertices[ia*3 + 1], vertices[ia*3 + 2]);
        float3 vb = float3(vertices[ib*3 + 0], vertices[ib*3 + 1], vertices[ib*3 + 2]);
        float3 mid = 0.5f * (va + vb);

        // Full quadric eval at an arbitrary point (used for midpoint and for
        // sanity-checking v*).
        auto evalQuadric = [&](float3 p) -> float {
            return p.x*p.x*a00 + 2.0f*p.x*p.y*a01 + 2.0f*p.x*p.z*a02
                 + p.y*p.y*a11 + 2.0f*p.y*p.z*a12
                 + p.z*p.z*a22
                 + 2.0f*(bx*p.x + by*p.y + bz*p.z)
                 + cc;
        };
        float residualMid = evalQuadric(mid);

        // Try the optimal `v* = -A⁻¹ b`. Relative det threshold + post-hoc
        // sanity check protect against the well-known QEM failure mode where
        // a near-singular A flings `v*` away to infinity (Garland 1997 §3.2).
        float maxA = fmax(fmax(fabs(a00), fabs(a11)), fabs(a22));
        maxA = fmax(maxA, fmax(fabs(a01), fmax(fabs(a02), fabs(a12))));
        // Relative threshold: `det / maxA³` is the natural condition measure
        // for a 3×3 (cube of A's scale, since det scales as the cube). When
        // the matrix has tiny scale OR is nearly singular, fall back early.
        float scaleCube = (maxA + 1e-30f) * (maxA + 1e-30f) * (maxA + 1e-30f);
        float vx, vy, vz, residual;
        if (fabs(det) < 1e-9f * scaleCube) {
            vx = mid.x; vy = mid.y; vz = mid.z;
            residual = residualMid;
        } else {
            float invDet = 1.0f / det;
            vx = -invDet * (adj00*bx + adj01*by + adj02*bz);
            vy = -invDet * (adj01*bx + adj11*by + adj12*bz);
            vz = -invDet * (adj02*bx + adj12*by + adj22*bz);
            // Garland's robustness trick: compute the residual at v* by full
            // quadric (not bᵀv*+c — that form catastrophically cancels when
            // v* is numerically bad and can go very negative). Compare against
            // midpoint; keep whichever is non-negative and smaller. Also
            // require v* is finite and within a reasonable region around the
            // edge — at-most edge_length away from the midpoint per axis.
            float3 candidate = float3(vx, vy, vz);
            float3 displacement = candidate - mid;
            float edgeLen = length(vb - va);
            float maxDisp = fmax(fabs(displacement.x),
                                 fmax(fabs(displacement.y), fabs(displacement.z)));
            float residualV = evalQuadric(candidate);
            bool finiteOK = isfinite(residualV) && isfinite(vx) && isfinite(vy) && isfinite(vz);
            bool inRange  = maxDisp <= edgeLen + 1e-6f;
            bool valid    = finiteOK && inRange && residualV >= -1e-6f && residualV <= residualMid;
            if (!valid) {
                vx = mid.x; vy = mid.y; vz = mid.z;
                residual = residualMid;
            } else {
                residual = residualV;
            }
        }

        target[e*3 + 0] = vx;
        target[e*3 + 1] = vy;
        target[e*3 + 2] = vz;
        cost[e] = residual;
    """
)

extension Mesh {
    /// For each edge in `edgeTable`, compute the collapse cost and optimal
    /// target position `v*`. Boundary edges receive `+infinity` cost.
    ///
    /// - Parameters:
    ///   - quadrics: per-vertex QEM `[V, 10]` from `vertexQuadrics()`.
    ///   - edgeTable: precomputed edge table; if `nil` it's built on the fly.
    /// - Returns: `(cost [E], target [E, 3])`.
    public func edgeCollapseCosts(
        quadrics: MLXArray,
        edgeTable: EdgeTable? = nil
    ) -> (cost: MLXArray, target: MLXArray) {
        let table = edgeTable ?? self.edgeTable()
        let e = table.count
        let tg = min(max(e, 1), 256)
        let outputs = edgeCollapseCostKernel(
            [vertices, table.edges, table.edgeFaceCount, quadrics],
            grid: (e, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[e], [e, 3]],
            outputDTypes: [.float32, .float32]
        )
        return (cost: outputs[0], target: outputs[1])
    }
}
