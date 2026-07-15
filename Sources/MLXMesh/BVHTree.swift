import Foundation
import MLX
import MLXFast

/// Phase 2 of LBVH construction: Karras 2012 radix-tree topology.
///
/// Given Morton codes sorted ascending, builds the binary radix tree. Each
/// internal node `i ∈ [0, F-1)` computes its own range and split point from
/// longest-common-prefix comparisons; the algorithm is embarrassingly
/// parallel (one thread per internal node, no dependencies between
/// threads).
///
/// Storage convention:
/// - Indices `[0, F-1)` are internal nodes; root is at index 0.
/// - Indices `[F-1, 2F-1)` are leaves; leaf `(F-1)+k` references face
///   `sortedIndices[k]`.

/// One node's topology — `left`, `right`, `parent`.
public struct BVHTopology {
    /// Number of input faces (= leaf count).
    public let faceCount: Int
    /// `[F]` int32 — Morton codes in sorted order.
    public let sortedCodes: MLXArray
    /// `[F]` int32 — permutation taking sorted position → original face index.
    public let sortedIndices: MLXArray
    /// `[2F-1]` int32 — child indices (leaves encoded as `>= F-1`).
    public let nodeLeft: MLXArray
    /// `[2F-1]` int32 — child indices (leaves encoded as `>= F-1`).
    public let nodeRight: MLXArray
    /// `[2F-1]` int32 — parent index; root has parent `-1`.
    public let nodeParent: MLXArray

    public var nodeCount: Int { 2 * faceCount - 1 }
    public var leafOffset: Int { faceCount - 1 }
}

private let karrasTreeKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "karras_radix_tree",
    inputNames: ["sortedCodes"],
    outputNames: ["nodeLeft", "nodeRight", "nodeParent"],
    source: """
        // One thread per internal node i in [0, F-1).
        int i = (int)thread_position_in_grid.x;
        int F = (int)sortedCodes_shape[0];
        int internalCount = F - 1;
        if (i >= internalCount) return;

        // δ(a, b): length of the longest common prefix of the sorted keys
        // (code, index) at positions a and b. Returns -1 for out-of-range b.
        // Keys are 30-bit Morton codes concatenated with the 32-bit index;
        // the LCP is computed in two stages.
        auto delta = [&](int a, int b) -> int {
            if (b < 0 || b >= F) return -1;
            int ca = sortedCodes[a];
            int cb = sortedCodes[b];
            if (ca != cb) {
                // Morton codes occupy 30 LSBs; clz on 32-bit minus 2 unused MSBs.
                return (int)metal::clz((uint)(ca ^ cb)) - 2;
            }
            // Codes identical → extend the key with the index for tie-break.
            uint xa = (uint)a, xb = (uint)b;
            return 30 + (int)metal::clz(xa ^ xb);
        };

        // Direction of the range: +1 right, -1 left.
        int d_pos = delta(i, i + 1);
        int d_neg = delta(i, i - 1);
        int d = (d_pos > d_neg) ? 1 : -1;
        int dmin = (d == 1) ? d_neg : d_pos;

        // Upper bound for range length via exponential search.
        int lmax = 2;
        while (delta(i, i + lmax * d) > dmin) lmax <<= 1;

        // Binary-search the precise far end.
        int l = 0;
        for (int t = lmax >> 1; t >= 1; t >>= 1) {
            if (delta(i, i + (l + t) * d) > dmin) l += t;
        }
        int j = i + l * d;

        // Binary-search the split γ.
        int dnode = delta(i, j);
        int s = 0;
        int divisor = 2;
        int t;
        do {
            // Ceiling of l / divisor — Karras' formulation.
            t = (l + divisor - 1) / divisor;
            if (delta(i, i + (s + t) * d) > dnode) s += t;
            divisor <<= 1;
        } while (t > 1);
        int gamma = i + s * d + (d < 0 ? -1 : 0);

        // Decode children: leaf vs internal.
        int leafBase = F - 1;
        int leftChild;
        int rightChild;
        int lo = min(i, j);
        int hi = max(i, j);
        if (lo == gamma) {
            leftChild = leafBase + gamma;        // leaf
        } else {
            leftChild = gamma;                    // internal
        }
        if (hi == gamma + 1) {
            rightChild = leafBase + gamma + 1;    // leaf
        } else {
            rightChild = gamma + 1;               // internal
        }

        nodeLeft[i] = leftChild;
        nodeRight[i] = rightChild;
        nodeParent[leftChild] = i;
        nodeParent[rightChild] = i;
    """
)

extension Mesh {
    /// Build the Karras radix tree from per-face Morton codes. Returns the
    /// full P2 topology; AABBs are filled in by P3 (`bvhComputeAABBs`,
    /// upcoming). `faceCount` must be ≥ 1; single-face meshes degenerate to
    /// a single-leaf tree with no internal nodes.
    public func buildBVHTopology() -> BVHTopology {
        precondition(faceCount >= 1, "BVH requires at least one face")
        let sortedIndices = bvhSortedFaceIndices()
        let codes = faceMortonCodes()
        let sortedCodes = codes[sortedIndices]

        let F = faceCount
        let N = 2 * F - 1

        if F == 1 {
            // No internal nodes; the single leaf has no parent.
            let parent = MLXArray([Int32(-1)], [1])
            return BVHTopology(
                faceCount: F,
                sortedCodes: sortedCodes,
                sortedIndices: sortedIndices,
                nodeLeft: MLXArray([] as [Int32], [0]),
                nodeRight: MLXArray([] as [Int32], [0]),
                nodeParent: parent
            )
        }

        let internalCount = F - 1
        let tg = min(max(internalCount, 1), 256)
        // The kernel writes left/right for every internal node and writes parent
        // for every child slot. Only the root's parent (index 0) is never
        // touched. We rebuild parent CPU-side after the dispatch so the root
        // keeps `-1` regardless of MLXFast's output-init behavior.
        let outputs = karrasTreeKernel(
            [sortedCodes],
            grid: (internalCount, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[N], [N], [N]],
            outputDTypes: [.int32, .int32, .int32]
        )
        let nodeLeft = outputs[0]
        let nodeRight = outputs[1]

        let lArr = nodeLeft.asArray(Int32.self)
        let rArr = nodeRight.asArray(Int32.self)
        var pArr = [Int32](repeating: -1, count: N)
        for i in 0..<internalCount {
            pArr[Int(lArr[i])] = Int32(i)
            pArr[Int(rArr[i])] = Int32(i)
        }
        return BVHTopology(
            faceCount: F,
            sortedCodes: sortedCodes,
            sortedIndices: sortedIndices,
            nodeLeft: nodeLeft,
            nodeRight: nodeRight,
            nodeParent: MLXArray(pArr, [N])
        )
    }
}
