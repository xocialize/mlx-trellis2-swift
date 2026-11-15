import Foundation
import MLX
import MLXFast

/// Batched ambient-occlusion queries: for each surface point, cast `rays` cosine-weighted rays over
/// the hemisphere of its shading normal and count how many hit the mesh within `tMax`.
///
/// One thread per query point. Directions are generated in the kernel, so no `[Q·rays, 3]` ray
/// buffer exists: a Hammersley point set (radical inverse in base 2) rotated per query by a
/// Cranley–Patterson offset hashed from the query index. Nothing is random, so the same inputs give
/// the same bits on every run (the deterministic-GLB rule, AB-A-0263). Traversal is any-hit: the
/// first triangle within `[tMin, tMax]` ends a ray. A direction that falls below the geometric
/// plane (possible when the shading normal tilts away from the face normal) is mirrored back above
/// it, so a ray never starts by entering its own surface.
private let occlusionKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "bvh_occlusion",
    inputNames: [
        "origins", "shadingNormals", "geomNormals", "params", "rayCount",
        "vertices", "faces", "sortedIndices",
        "nodeLeft", "nodeRight", "nodeAABB",
    ],
    outputNames: ["occluded"],
    source: """
        uint q = thread_position_in_grid.x;
        uint Q = origins_shape[0];
        if (q >= Q) return;
        float3 O  = float3(origins[q*3 + 0], origins[q*3 + 1], origins[q*3 + 2]);
        float3 Ns = float3(shadingNormals[q*3 + 0], shadingNormals[q*3 + 1], shadingNormals[q*3 + 2]);
        float3 Ng = float3(geomNormals[q*3 + 0], geomNormals[q*3 + 1], geomNormals[q*3 + 2]);
        float tMin = params[0];
        float tMax = params[1];
        int S = rayCount[0];

        // Orthonormal basis around Ns (Duff et al. 2017, branchless).
        float sgn = Ns.z >= 0.0f ? 1.0f : -1.0f;
        float a = -1.0f / (sgn + Ns.z);
        float b = Ns.x * Ns.y * a;
        float3 Tx = float3(1.0f + sgn * Ns.x * Ns.x * a, sgn * b, -sgn * Ns.x);
        float3 Ty = float3(b, sgn + Ns.y * Ns.y * a, -Ns.y);

        // Per-query rotation of the point set (PCG-style integer hash of q).
        uint h = q * 747796405u + 2891336453u;
        h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
        h = (h >> 22u) ^ h;
        float rot1 = float(h & 0xFFFFu) / 65536.0f;
        float rot2 = float(h >> 16u) / 65536.0f;

        int F = (int)sortedIndices_shape[0];
        int leafOffset = F - 1;
        int hits = 0;

        for (int s = 0; s < S; s++) {
            // Hammersley (i/S, radical inverse base 2 of i), rotated.
            uint bits = (uint)s;
            bits = (bits << 16u) | (bits >> 16u);
            bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
            bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
            bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
            bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
            float u1 = fract((float(s) + 0.5f) / float(S) + rot1);
            float u2 = fract(float(bits) * 2.3283064365386963e-10f + rot2);
            // Cosine-weighted hemisphere (Malley): uniform disk lifted onto the hemisphere.
            float r = sqrt(u1);
            float phi = 6.283185307179586f * u2;
            float3 D = Tx * (r * cos(phi)) + Ty * (r * sin(phi)) + Ns * sqrt(max(0.0f, 1.0f - u1));
            float dg = dot(D, Ng);
            if (dg < 0.0f) D = D - 2.0f * dg * Ng;
            D = normalize(D);
            float3 invD = float3(1.0f / D.x, 1.0f / D.y, 1.0f / D.z);

            bool hit = false;
            int stack[64];
            int sp = 0;
            stack[sp++] = 0;
            while (sp > 0 && !hit) {
                int n = stack[--sp];
                float t1x = (nodeAABB[n*6 + 0] - O.x) * invD.x;
                float t1y = (nodeAABB[n*6 + 1] - O.y) * invD.y;
                float t1z = (nodeAABB[n*6 + 2] - O.z) * invD.z;
                float t2x = (nodeAABB[n*6 + 3] - O.x) * invD.x;
                float t2y = (nodeAABB[n*6 + 4] - O.y) * invD.y;
                float t2z = (nodeAABB[n*6 + 5] - O.z) * invD.z;
                float tEnter = max(max(min(t1x, t2x), min(t1y, t2y)), min(t1z, t2z));
                float tExit  = min(min(max(t1x, t2x), max(t1y, t2y)), max(t1z, t2z));
                if (tEnter > tExit || tExit < tMin || tEnter > tMax) continue;
                if (n >= leafOffset) {
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
                    float3 Qv = cross(T, e1);
                    float v = dot(D, Qv) * inv;
                    if (v < 0.0f || u + v > 1.0f) continue;
                    float t = dot(e2, Qv) * inv;
                    if (t >= tMin && t <= tMax) hit = true;
                } else if (sp + 2 <= 64) {
                    stack[sp++] = nodeLeft[n];
                    stack[sp++] = nodeRight[n];
                }
            }
            if (hit) hits++;
        }
        occluded[q] = float(hits) / float(max(S, 1));
    """
)

extension BVH {
    /// Fraction of `rays` cosine-weighted hemisphere rays that hit the mesh, per query point.
    ///
    /// - origins: `[Q, 3]` ray origins (already lifted off the surface by the caller's bias).
    /// - shadingNormals: `[Q, 3]` unit normals; the hemisphere is centred on these.
    /// - geomNormals: `[Q, 3]` unit face normals on the same side as `shadingNormals`; directions
    ///   below this plane are mirrored above it.
    /// - Returns `[Q]` float32 in [0, 1] (1 = every ray hit). Deterministic.
    public func occlusion(
        mesh: Mesh,
        origins: MLXArray,
        shadingNormals: MLXArray,
        geomNormals: MLXArray,
        rays: Int,
        tMin: Float,
        tMax: Float
    ) -> MLXArray {
        precondition(origins.ndim == 2 && origins.dim(1) == 3, "origins must be [Q, 3]; got \(origins.shape)")
        precondition(shadingNormals.shape == origins.shape && geomNormals.shape == origins.shape,
                     "normals must match origins shape")
        precondition(rays >= 1, "rays must be ≥ 1")
        let Q = origins.dim(0)
        let tg = min(max(Q, 1), 256)
        let outputs = occlusionKernel(
            [
                origins.asType(.float32), shadingNormals.asType(.float32), geomNormals.asType(.float32),
                MLXArray([tMin, tMax], [2]), MLXArray([Int32(rays)], [1]),
                mesh.vertices, mesh.faces, topology.sortedIndices,
                topology.nodeLeft, topology.nodeRight, nodeAABB,
            ],
            grid: (Q, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[Q]],
            outputDTypes: [.float32]
        )
        return outputs[0]
    }
}
