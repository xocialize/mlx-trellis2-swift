import Foundation
import MLX

/// Core sparse voxel tensor for the TRELLIS.2 SLat models.
/// Mirrors the Python `SparseTensor`: features live on active voxels, with
/// coords = [T, 4] int32 (batch, x, y, z). Batch items are contiguous.
public struct SparseTensor {
    public var feats: MLXArray    // [T, C] float
    public var coords: MLXArray   // [T, 4] int32 (b, x, y, z)

    public init(feats: MLXArray, coords: MLXArray) {
        self.feats = feats
        self.coords = coords
    }

    public var count: Int { feats.dim(0) }
    public var channels: Int { feats.dim(1) }

    /// Replace features, keep coords (the Python `.replace(feats)`).
    public func replace(_ newFeats: MLXArray) -> SparseTensor {
        SparseTensor(feats: newFeats, coords: coords)
    }
}
