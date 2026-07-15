import Foundation
import MLX
import MLXFast

/// Phase 2 — per-voxel dual contouring vertex.
///
/// One thread per voxel reads 8 corner distances; an edge "crosses" when its
/// endpoints differ in sign. The DC vertex is the mean of edge crossings in
/// voxel-local fractional coords (or `(0.5, 0.5, 0.5)` if no crossings).

/// One DC vertex slot per voxel. `active` and `localDCVertex` are dense
/// `[R³]` and `[R³, 3]`; consumers compact to drop inactive voxels.
public struct DCVoxelVertices {
    public let resolution: Int
    public let active: MLXArray         // [R³] bool
    public let localDCVertex: MLXArray  // [R³, 3] float32 — fractional coords in voxel's [0, 1]³
}

private let dcVertexKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "dc_voxel_vertex",
    inputNames: ["cornerDistances", "RArr"],
    outputNames: ["active", "localDC"],
    source: """
        uint v = thread_position_in_grid.x;
        int R = RArr[0];
        int R3 = R * R * R;
        if (v >= (uint)R3) return;

        int n = R + 1;
        int vx = (int)v / (R * R);
        int vy = ((int)v / R) % R;
        int vz = (int)v % R;

        // 8 corner distances; bit 2 = cx, bit 1 = cy, bit 0 = cz.
        float d[8];
        for (uint c = 0; c < 8; ++c) {
            int cx = vx + (int)((c >> 2) & 1u);
            int cy = vy + (int)((c >> 1) & 1u);
            int cz = vz + (int)(c & 1u);
            d[c] = cornerDistances[cx * n * n + cy * n + cz];
        }

        // 12 edges as corner-pair indices.
        const int edges[12][2] = {
            {0, 4}, {1, 5}, {2, 6}, {3, 7},   // X-aligned
            {0, 2}, {1, 3}, {4, 6}, {5, 7},   // Y-aligned
            {0, 1}, {2, 3}, {4, 5}, {6, 7},   // Z-aligned
        };

        float3 sumLocal = float3(0.0f, 0.0f, 0.0f);
        int cnt = 0;
        for (uint e = 0; e < 12; ++e) {
            int a = edges[e][0];
            int b = edges[e][1];
            if ((d[a] < 0.0f) == (d[b] < 0.0f)) continue;
            float t = -d[a] / (d[b] - d[a]);
            float ax = (float)((a >> 2) & 1), ay = (float)((a >> 1) & 1), az = (float)(a & 1);
            float bx = (float)((b >> 2) & 1), by = (float)((b >> 1) & 1), bz = (float)(b & 1);
            sumLocal.x += ax + t * (bx - ax);
            sumLocal.y += ay + t * (by - ay);
            sumLocal.z += az + t * (bz - az);
            cnt++;
        }
        float3 dc = (cnt > 0) ? sumLocal / float(cnt) : float3(0.5f, 0.5f, 0.5f);
        localDC[v*3 + 0] = dc.x;
        localDC[v*3 + 1] = dc.y;
        localDC[v*3 + 2] = dc.z;
        active[v] = (cnt > 0);
    """
)

extension DCVoxelGrid {
    /// Compute per-voxel DC vertex positions (fractional within each voxel).
    public func computeDCVertices() -> DCVoxelVertices {
        let R = resolution
        let R3 = R * R * R
        let tg = min(max(R3, 1), 256)
        let outputs = dcVertexKernel(
            [cornerDistances, MLXArray([Int32(R)], [1])],
            grid: (R3, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[R3], [R3, 3]],
            outputDTypes: [.bool, .float32]
        )
        return DCVoxelVertices(
            resolution: R,
            active: outputs[0],
            localDCVertex: outputs[1]
        )
    }
}
