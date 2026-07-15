import Foundation
import MLX
import MLXFast

/// Phase 1 of LBVH construction: per-face 30-bit Morton codes from normalized
/// centroids, then an argsort to obtain a spatially-coherent face ordering.
///
/// Codes use the standard 10-bits-per-axis interleave: top 2 bits of the
/// resulting int32 are unused (always 0).

private let mortonEncodeKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "morton3d_encode",
    inputNames: ["normalized"],
    outputNames: ["codes"],
    source: """
        uint f = thread_position_in_grid.x;
        uint fcount = normalized_shape[0];
        if (f >= fcount) return;

        // Clamp to [0, 1023] then interleave bits.
        auto quantise = [](float v) -> uint {
            float scaled = v * 1024.0f;
            int i = (int)scaled;
            if (i < 0) return 0u;
            if (i > 1023) return 1023u;
            return (uint)i;
        };
        uint x = quantise(normalized[f*3 + 0]);
        uint y = quantise(normalized[f*3 + 1]);
        uint z = quantise(normalized[f*3 + 2]);

        // Expand 10 bits into 30 with two zero bits between each pair.
        auto expand = [](uint v) -> uint {
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        };
        uint code = (expand(x) << 2) | (expand(y) << 1) | expand(z);
        codes[f] = (int)code;
    """
)

extension Mesh {
    /// Per-face 30-bit Morton codes from normalized face centroids.
    ///
    /// Process: compute centroids in MLX, normalize to `[0, 1)³` using the
    /// mesh's bounding box, then bit-interleave to a 30-bit code.
    ///
    /// Returns `[F]` int32 (the top 2 bits are always 0).
    public func faceMortonCodes() -> MLXArray {
        let bbMin = vertices.min(axis: 0)
        let bbMax = vertices.max(axis: 0)
        let extent = MLX.maximum(bbMax - bbMin, MLXArray(Float(1e-12)))
        // Triangle centroids: gather [F, 3, 3] then mean along the 3-vertex axis.
        let centroids = vertices[faces].mean(axis: 1)
        let normalized = (centroids - bbMin) / extent

        let f = faceCount
        let tg = min(max(f, 1), 256)
        let outputs = mortonEncodeKernel(
            [normalized],
            grid: (f, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[f]],
            outputDTypes: [.int32]
        )
        return outputs[0]
    }

    /// Indices into `faces` sorted by Morton code (the BVH leaf order).
    /// Equivalent to `argsort(faceMortonCodes())` — just packaged.
    public func bvhSortedFaceIndices() -> MLXArray {
        MLX.argSort(faceMortonCodes())
    }
}
