import Cxatlas

/// Output vertex from a packed atlas. Mirrors `xatlas::Vertex`.
public struct OutputVertex: Sendable {
    public let atlasIndex: Int32
    public let chartIndex: Int32
    public let uv: SIMD2<Float>
    public let xref: UInt32

    init(_ v: xatlas.Vertex) {
        self.atlasIndex = v.atlasIndex
        self.chartIndex = v.chartIndex
        self.uv = SIMD2<Float>(v.uv.0, v.uv.1)
        self.xref = v.xref
    }
}

public struct OutputChart: Sendable {
    public let faceIndices: [UInt32]
    public let atlasIndex: UInt32
    public let type: ChartType
    public let material: UInt32

    init(_ c: xatlas.Chart) {
        self.faceIndices = Array(UnsafeBufferPointer(start: c.faceArray, count: Int(c.faceCount)))
        self.atlasIndex = c.atlasIndex
        self.type = ChartType(c.type)
        self.material = c.material
    }
}

/// A read-only view onto one of the atlas's output meshes. Borrows the
/// underlying buffers from the `Atlas`; do not retain past the atlas's
/// lifetime.
public struct OutputMesh {
    private let mesh: UnsafePointer<xatlas.Mesh>

    init(_ mesh: UnsafePointer<xatlas.Mesh>) {
        self.mesh = mesh
    }

    public var vertexCount: Int { Int(mesh.pointee.vertexCount) }
    public var indexCount: Int { Int(mesh.pointee.indexCount) }
    public var chartCount: Int { Int(mesh.pointee.chartCount) }

    // MARK: Eager (copies into Array)

    public var indices: [UInt32] {
        Array(UnsafeBufferPointer(start: mesh.pointee.indexArray, count: indexCount))
    }

    public var vertices: [OutputVertex] {
        UnsafeBufferPointer(start: mesh.pointee.vertexArray, count: vertexCount).map(OutputVertex.init)
    }

    public var charts: [OutputChart] {
        UnsafeBufferPointer(start: mesh.pointee.chartArray, count: chartCount).map(OutputChart.init)
    }

    // MARK: Zero-copy borrows
    //
    // The buffer pointers passed to `body` are owned by the parent `Atlas` and
    // must not escape the closure. Safe as long as the `Atlas` is alive and not
    // re-generated while the closure runs.

    public func withIndices<R>(_ body: (UnsafeBufferPointer<UInt32>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(start: mesh.pointee.indexArray, count: indexCount))
    }

    public func withVertices<R>(_ body: (UnsafeBufferPointer<xatlas.Vertex>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(start: mesh.pointee.vertexArray, count: vertexCount))
    }

    public func withCharts<R>(_ body: (UnsafeBufferPointer<xatlas.Chart>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(start: mesh.pointee.chartArray, count: chartCount))
    }
}

public struct MeshOutputCollection: RandomAccessCollection {
    private let atlas: Atlas

    init(atlas: Atlas) { self.atlas = atlas }

    public var startIndex: Int { 0 }
    public var endIndex: Int { Int(atlas.handle.pointee.meshCount) }

    public subscript(position: Int) -> OutputMesh {
        precondition(position >= 0 && position < endIndex, "mesh index out of range")
        let base = atlas.handle.pointee.meshes!
        return OutputMesh(base.advanced(by: position))
    }
}
