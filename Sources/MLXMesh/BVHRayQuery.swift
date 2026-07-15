import Foundation
import MLX
import MLXFast

/// Phase 4 — batched ray-mesh intersection.
///
/// One thread per ray traverses the BVH with a thread-local stack (depth 64,
/// comfortably larger than `log2(F)` for any practical mesh size). AABB hits
/// use the slab method; leaf hits use Möller-Trumbore.

/// One hit per query ray. `faceIndex == -1` and `t == .infinity` indicate a miss.
public struct RayHits {
    public let faceIndices: MLXArray  // [R] int32
    public let t: MLXArray             // [R] float32 (+inf for misses)
}

private let rayHitKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "bvh_ray_hit",
    inputNames: [
        "origins", "directions", "tMaxArr",
        "vertices", "faces", "sortedIndices",
        "nodeLeft", "nodeRight", "nodeAABB",
    ],
    outputNames: ["hitFace", "hitT"],
    source: """
        uint r = thread_position_in_grid.x;
        uint R = origins_shape[0];
        if (r >= R) return;
        float3 O = float3(origins[r*3 + 0], origins[r*3 + 1], origins[r*3 + 2]);
        float3 D = float3(directions[r*3 + 0], directions[r*3 + 1], directions[r*3 + 2]);
        float3 invD = float3(1.0f / D.x, 1.0f / D.y, 1.0f / D.z);
        float closestT = tMaxArr[0];
        int closestFace = -1;

        int F = (int)sortedIndices_shape[0];
        int leafOffset = F - 1;
        int N = 2 * F - 1;

        int stack[64];
        int sp = 0;
        stack[sp++] = 0;

        while (sp > 0) {
            int n = stack[--sp];
            // AABB intersection (slab).
            float bbMinX = nodeAABB[n*6 + 0];
            float bbMinY = nodeAABB[n*6 + 1];
            float bbMinZ = nodeAABB[n*6 + 2];
            float bbMaxX = nodeAABB[n*6 + 3];
            float bbMaxY = nodeAABB[n*6 + 4];
            float bbMaxZ = nodeAABB[n*6 + 5];
            float t1x = (bbMinX - O.x) * invD.x;
            float t1y = (bbMinY - O.y) * invD.y;
            float t1z = (bbMinZ - O.z) * invD.z;
            float t2x = (bbMaxX - O.x) * invD.x;
            float t2y = (bbMaxY - O.y) * invD.y;
            float t2z = (bbMaxZ - O.z) * invD.z;
            float tEnter = max(max(min(t1x, t2x), min(t1y, t2y)), min(t1z, t2z));
            float tExit  = min(min(max(t1x, t2x), max(t1y, t2y)), max(t1z, t2z));
            if (tEnter > tExit || tExit < 0.0f || tEnter > closestT) continue;

            if (n >= leafOffset) {
                // Leaf — Möller-Trumbore.
                int faceIdx = sortedIndices[n - leafOffset];
                int i0 = faces[faceIdx*3 + 0];
                int i1 = faces[faceIdx*3 + 1];
                int i2 = faces[faceIdx*3 + 2];
                float3 v0 = float3(vertices[i0*3 + 0], vertices[i0*3 + 1], vertices[i0*3 + 2]);
                float3 v1 = float3(vertices[i1*3 + 0], vertices[i1*3 + 1], vertices[i1*3 + 2]);
                float3 v2 = float3(vertices[i2*3 + 0], vertices[i2*3 + 1], vertices[i2*3 + 2]);
                float3 e1 = v1 - v0;
                float3 e2 = v2 - v0;
                float3 P  = cross(D, e2);
                float det = dot(e1, P);
                if (fabs(det) < 1e-12f) continue;
                float inv = 1.0f / det;
                float3 T = O - v0;
                float u = dot(T, P) * inv;
                if (u < 0.0f || u > 1.0f) continue;
                float3 Q = cross(T, e1);
                float v = dot(D, Q) * inv;
                if (v < 0.0f || u + v > 1.0f) continue;
                float t = dot(e2, Q) * inv;
                if (t < 0.0f || t > closestT) continue;
                closestT = t;
                closestFace = faceIdx;
            } else {
                int lc = nodeLeft[n];
                int rc = nodeRight[n];
                if (sp + 2 <= 64) {
                    stack[sp++] = lc;
                    stack[sp++] = rc;
                }
            }
        }

        hitFace[r] = closestFace;
        hitT[r] = (closestFace >= 0) ? closestT : INFINITY;
    """
)

extension BVH {
    /// Cast a batch of rays. `origins` and `directions` are `[R, 3]` float32;
    /// `directions` should be normalized. `tMax` clamps the maximum hit
    /// distance (infinity by default = no clamp).
    ///
    /// The mesh whose BVH this is must be passed in — `BVH` stores only the
    /// tree structure, not vertex / face data.
    public func rayHits(
        mesh: Mesh,
        origins: MLXArray,
        directions: MLXArray,
        tMax: Float = .infinity
    ) -> RayHits {
        precondition(origins.ndim == 2 && origins.dim(1) == 3,
                     "origins must be [R, 3]; got \(origins.shape)")
        precondition(directions.shape == origins.shape,
                     "directions must match origins shape")
        let R = origins.dim(0)
        let tg = min(max(R, 1), 256)
        let tMaxArr = MLXArray([tMax], [1])
        let outputs = rayHitKernel(
            [
                origins, directions, tMaxArr,
                mesh.vertices, mesh.faces, topology.sortedIndices,
                topology.nodeLeft, topology.nodeRight, nodeAABB,
            ],
            grid: (R, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[R], [R]],
            outputDTypes: [.int32, .float32]
        )
        return RayHits(faceIndices: outputs[0], t: outputs[1])
    }
}
