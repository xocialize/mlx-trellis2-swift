import Foundation
import MLX
import MLXFast

/// Phase 3 of LBVH construction: bottom-up AABB propagation.
///
/// Standard "second-thread-wins" pattern (Karras 2012, also Lauterbach 2009):
/// each leaf thread computes its triangle's AABB, then walks up to the root.
/// At every parent it atomically increments a visit counter; the *first*
/// thread to arrive at a parent stops (its sibling will handle the merge);
/// the *second* thread reads both children's AABBs and writes their union
/// into the parent. MSL atomics only support `memory_order_relaxed`, which
/// gives no happens-before between a child's AABB stores and its counter
/// bump — the merging thread could pass the gate yet read stale (zero-init)
/// child boxes. Device-scope `atomic_thread_fence` (MSL 3.2+) supplies the
/// missing release/acquire edges: fence before the bump publishes this
/// thread's subtree stores; fence after winning the gate makes the sibling's
/// stores visible before the child loads.

private let bvhAABBKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "bvh_aabb_bottom_up",
    inputNames: [
        "vertices", "faces", "sortedIndices",
        "nodeLeft", "nodeRight", "nodeParent",
    ],
    outputNames: ["nodeAABB", "visitCount"],
    source: """
        // One thread per leaf.
        uint leaf = thread_position_in_grid.x;
        uint F = sortedIndices_shape[0];
        if (leaf >= F) return;
        int leafOffset = (int)F - 1;
        int cur = (int)leaf + leafOffset;

        // Triangle AABB for this leaf.
        int faceIdx = sortedIndices[leaf];
        int i0 = faces[faceIdx*3 + 0];
        int i1 = faces[faceIdx*3 + 1];
        int i2 = faces[faceIdx*3 + 2];
        float3 v0 = float3(vertices[i0*3 + 0], vertices[i0*3 + 1], vertices[i0*3 + 2]);
        float3 v1 = float3(vertices[i1*3 + 0], vertices[i1*3 + 1], vertices[i1*3 + 2]);
        float3 v2 = float3(vertices[i2*3 + 0], vertices[i2*3 + 1], vertices[i2*3 + 2]);
        float3 mn = min(min(v0, v1), v2);
        float3 mx = max(max(v0, v1), v2);
        atomic_store_explicit(&nodeAABB[cur*6 + 0], mn.x, memory_order_relaxed);
        atomic_store_explicit(&nodeAABB[cur*6 + 1], mn.y, memory_order_relaxed);
        atomic_store_explicit(&nodeAABB[cur*6 + 2], mn.z, memory_order_relaxed);
        atomic_store_explicit(&nodeAABB[cur*6 + 3], mx.x, memory_order_relaxed);
        atomic_store_explicit(&nodeAABB[cur*6 + 4], mx.y, memory_order_relaxed);
        atomic_store_explicit(&nodeAABB[cur*6 + 5], mx.z, memory_order_relaxed);

        // Walk up.
        while (true) {
            int p = nodeParent[cur];
            if (p < 0) break;  // hit root from below
            // Release: publish this subtree's AABB stores before the counter
            // bump can be observed (relaxed atomics alone order nothing).
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                                thread_scope::thread_scope_device);
            int v = atomic_fetch_add_explicit(&visitCount[p], 1, memory_order_relaxed);
            if (v == 0) return;  // first arrival; let sibling do the merge

            // Acquire: we observed the sibling's bump, so after this fence its
            // pre-bump AABB stores are visible to our child loads below.
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                                thread_scope::thread_scope_device);

            // Second arrival: union our children's AABBs.
            int lc = nodeLeft[p];
            int rc = nodeRight[p];
            float lminx = atomic_load_explicit(&nodeAABB[lc*6 + 0], memory_order_relaxed);
            float lminy = atomic_load_explicit(&nodeAABB[lc*6 + 1], memory_order_relaxed);
            float lminz = atomic_load_explicit(&nodeAABB[lc*6 + 2], memory_order_relaxed);
            float lmaxx = atomic_load_explicit(&nodeAABB[lc*6 + 3], memory_order_relaxed);
            float lmaxy = atomic_load_explicit(&nodeAABB[lc*6 + 4], memory_order_relaxed);
            float lmaxz = atomic_load_explicit(&nodeAABB[lc*6 + 5], memory_order_relaxed);
            float rminx = atomic_load_explicit(&nodeAABB[rc*6 + 0], memory_order_relaxed);
            float rminy = atomic_load_explicit(&nodeAABB[rc*6 + 1], memory_order_relaxed);
            float rminz = atomic_load_explicit(&nodeAABB[rc*6 + 2], memory_order_relaxed);
            float rmaxx = atomic_load_explicit(&nodeAABB[rc*6 + 3], memory_order_relaxed);
            float rmaxy = atomic_load_explicit(&nodeAABB[rc*6 + 4], memory_order_relaxed);
            float rmaxz = atomic_load_explicit(&nodeAABB[rc*6 + 5], memory_order_relaxed);
            atomic_store_explicit(&nodeAABB[p*6 + 0], min(lminx, rminx), memory_order_relaxed);
            atomic_store_explicit(&nodeAABB[p*6 + 1], min(lminy, rminy), memory_order_relaxed);
            atomic_store_explicit(&nodeAABB[p*6 + 2], min(lminz, rminz), memory_order_relaxed);
            atomic_store_explicit(&nodeAABB[p*6 + 3], max(lmaxx, rmaxx), memory_order_relaxed);
            atomic_store_explicit(&nodeAABB[p*6 + 4], max(lmaxy, rmaxy), memory_order_relaxed);
            atomic_store_explicit(&nodeAABB[p*6 + 5], max(lmaxz, rmaxz), memory_order_relaxed);
            cur = p;
        }
    """,
    atomicOutputs: true
)

extension Mesh {
    /// Compute the BVH node AABBs given a P2 topology. Returns a `BVH` struct
    /// that's now fully usable for queries (P4 + P5 add the query kernels).
    public func bvhComputeAABBs(topology: BVHTopology) -> BVH {
        let F = topology.faceCount
        let N = topology.nodeCount

        if F == 1 {
            // Single-leaf tree — compute the one AABB directly.
            let v = vertices.asArray(Float.self)
            let f = faces.asArray(Int32.self)
            let i0 = Int(f[0]), i1 = Int(f[1]), i2 = Int(f[2])
            let mn: [Float] = [
                min(v[i0*3 + 0], min(v[i1*3 + 0], v[i2*3 + 0])),
                min(v[i0*3 + 1], min(v[i1*3 + 1], v[i2*3 + 1])),
                min(v[i0*3 + 2], min(v[i1*3 + 2], v[i2*3 + 2])),
            ]
            let mx: [Float] = [
                max(v[i0*3 + 0], max(v[i1*3 + 0], v[i2*3 + 0])),
                max(v[i0*3 + 1], max(v[i1*3 + 1], v[i2*3 + 1])),
                max(v[i0*3 + 2], max(v[i1*3 + 2], v[i2*3 + 2])),
            ]
            return BVH(
                topology: topology,
                nodeAABB: MLXArray(mn + mx, [1, 6])
            )
        }

        let tg = min(max(F, 1), 256)
        let outputs = bvhAABBKernel(
            [
                vertices, faces, topology.sortedIndices,
                topology.nodeLeft, topology.nodeRight, topology.nodeParent,
            ],
            grid: (F, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[N, 6], [N]],  // visitCount sized N for convenience
            outputDTypes: [.float32, .int32],
            initValue: 0
        )
        return BVH(topology: topology, nodeAABB: outputs[0])
    }

    /// Build the complete BVH (P1 + P2 + P3) for this mesh. Single entry point
    /// callers should use unless they need intermediate state.
    public func bvh() -> BVH {
        bvhComputeAABBs(topology: buildBVHTopology())
    }
}

/// Complete BVH: topology + per-node AABBs. Ray and closest-point queries
/// will operate on instances of this struct (P4 and P5).
public struct BVH {
    public let topology: BVHTopology
    /// `[N, 6]` float32 — `(minX, minY, minZ, maxX, maxY, maxZ)` per node.
    public let nodeAABB: MLXArray

    public var faceCount: Int { topology.faceCount }
    public var nodeCount: Int { topology.nodeCount }
    public var leafOffset: Int { topology.leafOffset }
}
