import Cxatlas

public enum IndexFormat: Sendable {
    case uint16, uint32

    var cxx: xatlas.IndexFormat {
        switch self {
        case .uint16: return xatlas.IndexFormat.UInt16
        case .uint32: return xatlas.IndexFormat.UInt32
        }
    }

    var stride: Int {
        switch self {
        case .uint16: return MemoryLayout<UInt16>.stride
        case .uint32: return MemoryLayout<UInt32>.stride
        }
    }
}

/// High-level input mesh, backed by Swift arrays. xatlas copies all data
/// internally during `addMesh`, so the buffers don't need to outlive the call.
public struct MeshInput {
    public var positions: [SIMD3<Float>]
    public var normals: [SIMD3<Float>]
    public var uvs: [SIMD2<Float>]
    public var indices32: [UInt32]
    public var indices16: [UInt16]
    public var faceMaterial: [UInt32]
    public var faceIgnore: [Bool]
    public var indexFormat: IndexFormat
    public var indexOffset: Int32
    public var epsilon: Float

    public init(
        positions: [SIMD3<Float>],
        indices: [UInt32],
        normals: [SIMD3<Float>] = [],
        uvs: [SIMD2<Float>] = [],
        faceMaterial: [UInt32] = [],
        faceIgnore: [Bool] = [],
        indexOffset: Int32 = 0,
        epsilon: Float = 1.192092896e-07
    ) {
        self.positions = positions
        self.normals = normals
        self.uvs = uvs
        self.indices32 = indices
        self.indices16 = []
        self.faceMaterial = faceMaterial
        self.faceIgnore = faceIgnore
        self.indexFormat = .uint32
        self.indexOffset = indexOffset
        self.epsilon = epsilon
    }

    public init(
        positions: [SIMD3<Float>],
        indices: [UInt16],
        normals: [SIMD3<Float>] = [],
        uvs: [SIMD2<Float>] = [],
        faceMaterial: [UInt32] = [],
        faceIgnore: [Bool] = [],
        indexOffset: Int32 = 0,
        epsilon: Float = 1.192092896e-07
    ) {
        self.positions = positions
        self.normals = normals
        self.uvs = uvs
        self.indices32 = []
        self.indices16 = indices
        self.faceMaterial = faceMaterial
        self.faceIgnore = faceIgnore
        self.indexFormat = .uint16
        self.indexOffset = indexOffset
        self.epsilon = epsilon
    }

    /// Build a transient `xatlas::MeshDecl` referencing Swift-owned buffers,
    /// then call `body`. Pointers must not escape.
    func withCxxDecl<R>(_ body: (xatlas.MeshDecl) throws -> R) rethrows -> R {
        try positions.withUnsafeBufferPointer { pos in
            try normals.withUnsafeBufferPointer { nrm in
                try uvs.withUnsafeBufferPointer { uv in
                    try indices32.withUnsafeBufferPointer { idx32 in
                        try indices16.withUnsafeBufferPointer { idx16 in
                            try faceMaterial.withUnsafeBufferPointer { mat in
                                try faceIgnore.withUnsafeBufferPointer { ignore in
                                    var decl = xatlas.MeshDecl()
                                    decl.vertexCount = UInt32(positions.count)
                                    decl.vertexPositionData = UnsafeRawPointer(pos.baseAddress)
                                    decl.vertexPositionStride = UInt32(MemoryLayout<SIMD3<Float>>.stride)
                                    if let n = nrm.baseAddress, !normals.isEmpty {
                                        decl.vertexNormalData = UnsafeRawPointer(n)
                                        decl.vertexNormalStride = UInt32(MemoryLayout<SIMD3<Float>>.stride)
                                    }
                                    if let u = uv.baseAddress, !uvs.isEmpty {
                                        decl.vertexUvData = UnsafeRawPointer(u)
                                        decl.vertexUvStride = UInt32(MemoryLayout<SIMD2<Float>>.stride)
                                    }
                                    switch indexFormat {
                                    case .uint32:
                                        decl.indexData = UnsafeRawPointer(idx32.baseAddress)
                                        decl.indexCount = UInt32(indices32.count)
                                        decl.indexFormat = xatlas.IndexFormat.UInt32
                                    case .uint16:
                                        decl.indexData = UnsafeRawPointer(idx16.baseAddress)
                                        decl.indexCount = UInt32(indices16.count)
                                        decl.indexFormat = xatlas.IndexFormat.UInt16
                                    }
                                    decl.indexOffset = indexOffset
                                    if !faceMaterial.isEmpty {
                                        decl.faceMaterialData = mat.baseAddress
                                    }
                                    if !faceIgnore.isEmpty {
                                        decl.faceIgnoreData = ignore.baseAddress
                                    }
                                    decl.epsilon = epsilon
                                    return try body(decl)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Raw / interleaved input

/// A single vertex attribute view into a (possibly interleaved) buffer.
public struct VertexAttribute {
    public let data: UnsafeRawPointer
    public let stride: Int

    public init(data: UnsafeRawPointer, stride: Int) {
        self.data = data
        self.stride = stride
    }
}

/// Low-level input descriptor — point straight at MTLBuffer / Data / raw memory
/// with custom strides for interleaved vertex layouts. Caller is responsible
/// for keeping the referenced memory valid until `addMesh` returns. xatlas
/// then copies the data internally.
public struct RawMeshInput {
    public var positions: VertexAttribute
    public var normals: VertexAttribute?
    public var uvs: VertexAttribute?
    public var vertexCount: Int
    public var indices: UnsafeRawBufferPointer
    public var indexFormat: IndexFormat
    public var indexOffset: Int32
    public var faceMaterial: UnsafeBufferPointer<UInt32>?
    public var faceIgnore: UnsafeBufferPointer<Bool>?
    public var epsilon: Float

    public init(
        positions: VertexAttribute,
        vertexCount: Int,
        indices: UnsafeRawBufferPointer,
        indexFormat: IndexFormat,
        normals: VertexAttribute? = nil,
        uvs: VertexAttribute? = nil,
        indexOffset: Int32 = 0,
        faceMaterial: UnsafeBufferPointer<UInt32>? = nil,
        faceIgnore: UnsafeBufferPointer<Bool>? = nil,
        epsilon: Float = 1.192092896e-07
    ) {
        self.positions = positions
        self.normals = normals
        self.uvs = uvs
        self.vertexCount = vertexCount
        self.indices = indices
        self.indexFormat = indexFormat
        self.indexOffset = indexOffset
        self.faceMaterial = faceMaterial
        self.faceIgnore = faceIgnore
        self.epsilon = epsilon
    }

    var cxx: xatlas.MeshDecl {
        var decl = xatlas.MeshDecl()
        decl.vertexCount = UInt32(vertexCount)
        decl.vertexPositionData = positions.data
        decl.vertexPositionStride = UInt32(positions.stride)
        if let n = normals {
            decl.vertexNormalData = n.data
            decl.vertexNormalStride = UInt32(n.stride)
        }
        if let u = uvs {
            decl.vertexUvData = u.data
            decl.vertexUvStride = UInt32(u.stride)
        }
        decl.indexData = UnsafeRawPointer(indices.baseAddress)
        decl.indexCount = UInt32(indices.count / indexFormat.stride)
        decl.indexFormat = indexFormat.cxx
        decl.indexOffset = indexOffset
        if let m = faceMaterial { decl.faceMaterialData = m.baseAddress }
        if let f = faceIgnore { decl.faceIgnoreData = f.baseAddress }
        decl.epsilon = epsilon
        return decl
    }
}

/// Input descriptor for `addUvMesh` — used when you already have UV coords
/// and want xatlas to repack them into an atlas without re-parameterising.
public struct UvMeshInput {
    public var uvs: [SIMD2<Float>]
    public var indices32: [UInt32]
    public var indices16: [UInt16]
    public var faceMaterial: [UInt32]
    public var indexFormat: IndexFormat
    public var indexOffset: Int32

    public init(
        uvs: [SIMD2<Float>],
        indices: [UInt32],
        faceMaterial: [UInt32] = [],
        indexOffset: Int32 = 0
    ) {
        self.uvs = uvs
        self.indices32 = indices
        self.indices16 = []
        self.faceMaterial = faceMaterial
        self.indexFormat = .uint32
        self.indexOffset = indexOffset
    }

    public init(
        uvs: [SIMD2<Float>],
        indices: [UInt16],
        faceMaterial: [UInt32] = [],
        indexOffset: Int32 = 0
    ) {
        self.uvs = uvs
        self.indices32 = []
        self.indices16 = indices
        self.faceMaterial = faceMaterial
        self.indexFormat = .uint16
        self.indexOffset = indexOffset
    }

    func withCxxDecl<R>(_ body: (xatlas.UvMeshDecl) throws -> R) rethrows -> R {
        try uvs.withUnsafeBufferPointer { uv in
            try indices32.withUnsafeBufferPointer { idx32 in
                try indices16.withUnsafeBufferPointer { idx16 in
                    try faceMaterial.withUnsafeBufferPointer { mat in
                        var decl = xatlas.UvMeshDecl()
                        decl.vertexCount = UInt32(uvs.count)
                        decl.vertexUvData = UnsafeRawPointer(uv.baseAddress)
                        decl.vertexStride = UInt32(MemoryLayout<SIMD2<Float>>.stride)
                        switch indexFormat {
                        case .uint32:
                            decl.indexData = UnsafeRawPointer(idx32.baseAddress)
                            decl.indexCount = UInt32(indices32.count)
                            decl.indexFormat = xatlas.IndexFormat.UInt32
                        case .uint16:
                            decl.indexData = UnsafeRawPointer(idx16.baseAddress)
                            decl.indexCount = UInt32(indices16.count)
                            decl.indexFormat = xatlas.IndexFormat.UInt16
                        }
                        decl.indexOffset = indexOffset
                        if !faceMaterial.isEmpty {
                            decl.faceMaterialData = mat.baseAddress
                        }
                        return try body(decl)
                    }
                }
            }
        }
    }
}
