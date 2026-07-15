import Foundation
import MLX
import MLXFast

/// Cached metal kernel for face normals.
private let faceNormalsMetalKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "face_normals",
    inputNames: ["vertices", "faces"],
    outputNames: ["normals"],
    source: """
        uint f = thread_position_in_grid.x;
        uint fcount = faces_shape[0];
        if (f >= fcount) return;
        int i0 = faces[f * 3 + 0];
        int i1 = faces[f * 3 + 1];
        int i2 = faces[f * 3 + 2];
        float3 v0 = float3(vertices[i0*3 + 0], vertices[i0*3 + 1], vertices[i0*3 + 2]);
        float3 v1 = float3(vertices[i1*3 + 0], vertices[i1*3 + 1], vertices[i1*3 + 2]);
        float3 v2 = float3(vertices[i2*3 + 0], vertices[i2*3 + 1], vertices[i2*3 + 2]);
        float3 n = cross(v1 - v0, v2 - v0);
        normals[f*3 + 0] = n.x;
        normals[f*3 + 1] = n.y;
        normals[f*3 + 2] = n.z;
    """
)

extension Mesh {
    /// Per-face un-normalized normals, computed by a custom Metal kernel.
    /// Behaviourally equivalent to `faceNormals(normalize: false)`; exists to
    /// validate the MLXFast custom-kernel pipeline.
    public func faceNormalsKernel() -> MLXArray {
        let f = faceCount
        let tg = min(max(f, 1), 256)
        let outputs = faceNormalsMetalKernel(
            [vertices, faces],
            grid: (f, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[f, 3]],
            outputDTypes: [.float32]
        )
        return outputs[0]
    }
}
