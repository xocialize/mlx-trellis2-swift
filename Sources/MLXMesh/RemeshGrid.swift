import Foundation
import MLX

/// One layer below `remeshDualContouring`: the corner grid + pseudo-signed
/// distance values that the DC kernels consume. Exposed publicly so callers
/// can inspect or visualize the underlying field (e.g., for SDF debugging).
public struct DCVoxelGrid {
    /// Voxels per axis. There are `(resolution + 1)` corners per axis.
    public let resolution: Int
    /// World-space minimum corner of the cubic grid.
    public let gridMin: SIMD3<Float>
    /// Cube edge length (`gridMin + cubeSize` = gridMax).
    public let cubeSize: Float
    /// `cubeSize / resolution`.
    public let voxelSize: Float
    /// Shift applied to raw UDF so the isosurface sits one band out.
    public let eps: Float
    /// `[(R+1)³]` float32 — `udf(corner) - eps`. Negative inside the band,
    /// positive outside. Corner index = `cx*(R+1)² + cy*(R+1) + cz`.
    public let cornerDistances: MLXArray
}

extension Mesh {
    /// Build the corner-grid pseudo-SDF for narrow-band dual contouring.
    /// `band` is the narrow-band width in voxel units.
    public func dcVoxelGrid(
        resolution R: Int,
        band: Float = 1.0,
        bvh: BVH? = nil
    ) -> DCVoxelGrid {
        precondition(R >= 1, "resolution must be ≥ 1")

        // Cubic grid wrapping the mesh bbox with a 2-voxel margin.
        let (bbMinArr, bbMaxArr) = boundingBox()
        let mn = bbMinArr.asArray(Float.self)
        let mx = bbMaxArr.asArray(Float.self)
        let extent = SIMD3<Float>(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
        let center = SIMD3<Float>(
            (mn[0] + mx[0]) * 0.5,
            (mn[1] + mx[1]) * 0.5,
            (mn[2] + mx[2]) * 0.5
        )
        let maxExt = max(extent.x, max(extent.y, extent.z))
        let voxelSize = maxExt / Float(R)
        // 2-voxel margin on each side → cube side = maxExt + 4 voxels.
        let cubeSize = maxExt + 4 * voxelSize
        let half = cubeSize * 0.5
        let gridMin = SIMD3<Float>(center.x - half, center.y - half, center.z - half)
        let eps = band * voxelSize

        // Build corner positions CPU-side: simpler than meshgrid plumbing.
        let n = R + 1
        var positions = [Float]()
        positions.reserveCapacity(n * n * n * 3)
        let step = cubeSize / Float(R)
        for ix in 0..<n {
            let x = gridMin.x + Float(ix) * step
            for iy in 0..<n {
                let y = gridMin.y + Float(iy) * step
                for iz in 0..<n {
                    let z = gridMin.z + Float(iz) * step
                    positions.append(x); positions.append(y); positions.append(z)
                }
            }
        }
        let queries = MLXArray(positions, [n * n * n, 3])

        let actualBVH = bvh ?? self.bvh()
        let r = actualBVH.closestPoints(mesh: self, queries: queries)
        let signed = r.distances - MLXArray(eps)

        return DCVoxelGrid(
            resolution: R,
            gridMin: gridMin,
            cubeSize: cubeSize,
            voxelSize: voxelSize,
            eps: eps,
            cornerDistances: signed
        )
    }
}
