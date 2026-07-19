import Foundation
import MLX
import Xatlas

/// Result of a UV unwrap pass.
///
/// `mesh` may have more vertices than the input because xatlas duplicates
/// vertices along chart seams (a single 3D vertex shared by two charts
/// becomes two output vertices with distinct UVs). `vertexMap[i]` is the
/// index in the *original* mesh's vertex array that output vertex `i`
/// corresponds to; use it to map per-vertex attributes through.
public struct UVUnwrapResult {
    /// Mesh with reindexed faces and (possibly) split vertices.
    public let mesh: Mesh
    /// `[V_out, 2]` float32 — UVs in `[0, 1)`.
    public let uvs: MLXArray
    /// `[V_out]` int32 — `vertexMap[i]` is the original vertex index that
    /// output vertex `i` came from.
    public let vertexMap: MLXArray
    /// Atlas width / height in texels (from `packCharts`).
    public let atlasWidth: UInt32
    public let atlasHeight: UInt32
    /// Per-chart metadata (one entry per chart of mesh 0). Mirrors what
    /// `cumesh.read_atlas_charts` exposes — chart index, atlas index, type,
    /// material, and the list of original face indices in each chart.
    public let charts: [Chart]

    public struct Chart: Sendable {
        public let atlasIndex: UInt32
        public let material: UInt32
        public let type: ChartType
        public let faceIndices: [UInt32]
    }
}

extension Mesh {
    /// UV unwrap via xatlas. Mirrors `cumesh.Atlas` from the upstream
    /// library; the inputs come from this `Mesh` and the outputs are
    /// returned as `MLXArray`s for chaining with the rest of MLXMesh.
    ///
    /// - Parameters:
    ///   - chartOptions: forwarded to `Atlas.computeCharts`.
    ///   - packOptions: forwarded to `Atlas.packCharts`. Defaults give a
    ///     square atlas with sensible texel density.
    public func uvUnwrap(
        chartOptions: ChartOptions = ChartOptions(),
        packOptions: PackOptions = PackOptions()
    ) throws -> UVUnwrapResult {
        precondition(vertexCount > 0 && faceCount > 0, "uvUnwrap requires a non-empty mesh")

        // Marshal MLXArrays → Swift native arrays for xatlas.
        let vRaw = vertices.asArray(Float.self)
        var positions = [SIMD3<Float>]()
        positions.reserveCapacity(vertexCount)
        for i in 0..<vertexCount {
            positions.append(SIMD3<Float>(vRaw[i*3 + 0], vRaw[i*3 + 1], vRaw[i*3 + 2]))
        }
        let fRaw = faces.asArray(Int32.self)
        var indices = [UInt32]()
        indices.reserveCapacity(faceCount * 3)
        for v in fRaw {
            precondition(v >= 0, "face indices must be non-negative")
            indices.append(UInt32(bitPattern: v))
        }

        let atlas = Atlas()
        try atlas.addMesh(MeshInput(positions: positions, indices: indices))
        atlas.generate(chartOptions: chartOptions, packOptions: packOptions)

        return try Self.buildResult(atlas: atlas, vRaw: vRaw)
    }

    /// Pack-only unwrap: the mesh already carries per-vertex UVs (seams pre-split,
    /// so faces index UV and position arrays identically). xatlas derives charts
    /// from UV islands and packs them into an atlas — no chart computation, no
    /// re-parameterisation. This is the integration seam for precomputed
    /// (provenance/GPU) parameterisations; see UV-UNWRAP-METAL-PLAN.md Phase 1.
    ///
    /// - Parameter existingUVs: `[V, 2]` float32, aligned with `vertices`.
    public func uvUnwrap(
        existingUVs: MLXArray,
        packOptions: PackOptions = PackOptions()
    ) throws -> UVUnwrapResult {
        precondition(vertexCount > 0 && faceCount > 0, "uvUnwrap requires a non-empty mesh")
        precondition(existingUVs.dim(0) == vertexCount && existingUVs.dim(1) == 2,
                     "existingUVs must be [V, 2] aligned with mesh vertices")

        let vRaw = vertices.asArray(Float.self)
        let uvRaw = existingUVs.asArray(Float.self)
        var uvs = [SIMD2<Float>]()
        uvs.reserveCapacity(vertexCount)
        for i in 0..<vertexCount {
            uvs.append(SIMD2<Float>(uvRaw[i*2], uvRaw[i*2 + 1]))
        }
        let fRaw = faces.asArray(Int32.self)
        var indices = [UInt32]()
        indices.reserveCapacity(faceCount * 3)
        for v in fRaw {
            precondition(v >= 0, "face indices must be non-negative")
            indices.append(UInt32(bitPattern: v))
        }

        let atlas = Atlas()
        try atlas.addUvMesh(UvMeshInput(uvs: uvs, indices: indices))
        atlas.computeCharts()
        atlas.packCharts(options: packOptions)

        return try Self.buildResult(atlas: atlas, vRaw: vRaw)
    }

    private static func buildResult(atlas: Atlas, vRaw: [Float]) throws -> UVUnwrapResult {
        guard atlas.meshCount > 0 else {
            throw UVUnwrapError.noMeshesProduced
        }
        let out = atlas.meshes[0]

        // Build MLXArrays from the output.
        let outVerts = out.vertices
        let outIdx = out.indices

        var newVerts: [Float] = []
        var newUVs: [Float] = []
        var vmap: [Int32] = []
        newVerts.reserveCapacity(outVerts.count * 3)
        newUVs.reserveCapacity(outVerts.count * 2)
        vmap.reserveCapacity(outVerts.count)
        let atlasW = Float(atlas.width == 0 ? 1 : atlas.width)
        let atlasH = Float(atlas.height == 0 ? 1 : atlas.height)
        for ov in outVerts {
            // xref is the original-mesh vertex index this output vertex
            // descended from — we use it to fetch the 3D position.
            let src = Int(ov.xref)
            newVerts.append(vRaw[src*3 + 0])
            newVerts.append(vRaw[src*3 + 1])
            newVerts.append(vRaw[src*3 + 2])
            // xatlas reports UVs in *texel* coordinates; normalize to [0, 1).
            newUVs.append(ov.uv.x / atlasW)
            newUVs.append(ov.uv.y / atlasH)
            vmap.append(Int32(src))
        }

        var newFaces: [Int32] = []
        newFaces.reserveCapacity(outIdx.count)
        for i in outIdx { newFaces.append(Int32(i)) }

        let V = outVerts.count
        let F = outIdx.count / 3
        let charts: [UVUnwrapResult.Chart] = out.charts.map { c in
            UVUnwrapResult.Chart(
                atlasIndex: c.atlasIndex,
                material: c.material,
                type: c.type,
                faceIndices: c.faceIndices
            )
        }

        return UVUnwrapResult(
            mesh: Mesh(
                vertices: MLXArray(newVerts, [V, 3]),
                faces: MLXArray(newFaces, [F, 3])
            ),
            uvs: MLXArray(newUVs, [V, 2]),
            vertexMap: MLXArray(vmap, [V]),
            atlasWidth: atlas.width,
            atlasHeight: atlas.height,
            charts: charts
        )
    }
}

public enum UVUnwrapError: Error {
    case noMeshesProduced
}
