// Minimal PLY / GLB position+index readers for the unwrap benchmark ONLY.
// The production package is generation-only (no import path) by design; these
// readers stay in the bench target and support just enough of each format to
// load corpus meshes: binary/ascii PLY, and non-sparse non-Draco GLB triangles.
import Foundation

enum MeshReadError: Error, CustomStringConvertible {
    case unsupported(String)
    case malformed(String)
    var description: String {
        switch self {
        case .unsupported(let s): return "unsupported: \(s)"
        case .malformed(let s): return "malformed: \(s)"
        }
    }
}

struct LoadedMesh {
    var positions: [Float]   // V*3
    var indices: [Int32]     // F*3
    var vertexCount: Int { positions.count / 3 }
    var faceCount: Int { indices.count / 3 }
}

func readMesh(path: String) throws -> LoadedMesh {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    switch url.pathExtension.lowercased() {
    case "ply": return try readPLY(data)
    case "glb": return try readGLB(data)
    default: throw MeshReadError.unsupported("extension \(url.pathExtension)")
    }
}

// MARK: - PLY

private enum PLYType: String {
    case char, uchar, short, ushort, int, uint, float, double
    init(_ s: String) throws {
        switch s {
        case "char", "int8": self = .char
        case "uchar", "uint8": self = .uchar
        case "short", "int16": self = .short
        case "ushort", "uint16": self = .ushort
        case "int", "int32": self = .int
        case "uint", "uint32": self = .uint
        case "float", "float32": self = .float
        case "double", "float64": self = .double
        default: throw MeshReadError.unsupported("ply type \(s)")
        }
    }
    var size: Int {
        switch self {
        case .char, .uchar: return 1
        case .short, .ushort: return 2
        case .int, .uint, .float: return 4
        case .double: return 8
        }
    }
}

private func readPLY(_ data: Data) throws -> LoadedMesh {
    guard let headerEnd = data.range(of: Data("end_header\n".utf8)) else {
        throw MeshReadError.malformed("no end_header")
    }
    guard let header = String(data: data[..<headerEnd.upperBound], encoding: .ascii) else {
        throw MeshReadError.malformed("non-ascii header")
    }
    var isBinary = false, isBigEndian = false
    struct Prop { var name: String; var type: PLYType; var listCount: PLYType? }
    struct Element { var name: String; var count: Int; var props: [Prop] = [] }
    var elements: [Element] = []
    for line in header.split(separator: "\n") {
        let t = line.split(separator: " ").map(String.init)
        guard !t.isEmpty else { continue }
        switch t[0] {
        case "format":
            isBinary = t[1].hasPrefix("binary")
            isBigEndian = t[1] == "binary_big_endian"
        case "element":
            elements.append(Element(name: t[1], count: Int(t[2]) ?? 0))
        case "property":
            guard !elements.isEmpty else { continue }
            if t[1] == "list" {
                elements[elements.count - 1].props.append(
                    Prop(name: t[4], type: try PLYType(t[3]), listCount: try PLYType(t[2])))
            } else {
                elements[elements.count - 1].props.append(
                    Prop(name: t[2], type: try PLYType(t[1]), listCount: nil))
            }
        default: break
        }
    }
    if isBigEndian { throw MeshReadError.unsupported("big-endian ply") }

    var positions: [Float] = []
    var indices: [Int32] = []

    if isBinary {
        var off = headerEnd.upperBound
        let bytes = [UInt8](data)
        func scalar(_ type: PLYType, _ o: Int) -> Double {
            switch type {
            case .char: return Double(Int8(bitPattern: bytes[o]))
            case .uchar: return Double(bytes[o])
            case .short: return Double(Int16(littleEndian: load(bytes, o)))
            case .ushort: return Double(UInt16(littleEndian: load(bytes, o)))
            case .int: return Double(Int32(littleEndian: load(bytes, o)))
            case .uint: return Double(UInt32(littleEndian: load(bytes, o)))
            case .float: return Double(Float(bitPattern: UInt32(littleEndian: load(bytes, o))))
            case .double: return Double(bitPattern: UInt64(littleEndian: load(bytes, o)))
            }
        }
        for el in elements {
            if el.name == "vertex" {
                positions.reserveCapacity(el.count * 3)
                let stride = el.props.reduce(0) { $0 + $1.type.size }
                var xo = -1, yo = -1, zo = -1, running = 0
                var xt = PLYType.float, yt = PLYType.float, zt = PLYType.float
                for p in el.props {
                    if p.listCount != nil { throw MeshReadError.unsupported("list prop in vertex") }
                    if p.name == "x" { xo = running; xt = p.type }
                    if p.name == "y" { yo = running; yt = p.type }
                    if p.name == "z" { zo = running; zt = p.type }
                    running += p.type.size
                }
                guard xo >= 0, yo >= 0, zo >= 0 else { throw MeshReadError.malformed("no xyz") }
                for _ in 0..<el.count {
                    positions.append(Float(scalar(xt, off + xo)))
                    positions.append(Float(scalar(yt, off + yo)))
                    positions.append(Float(scalar(zt, off + zo)))
                    off += stride
                }
            } else if el.name == "face" {
                indices.reserveCapacity(el.count * 3)
                for _ in 0..<el.count {
                    var faceVerts: [Int32] = []
                    for p in el.props {
                        if let ct = p.listCount {
                            let n = Int(scalar(ct, off)); off += ct.size
                            faceVerts = (0..<n).map { i in Int32(scalar(p.type, off + i * p.type.size)) }
                            off += n * p.type.size
                        } else {
                            off += p.type.size
                        }
                    }
                    // triangulate fans for quads+
                    for i in 1..<(max(faceVerts.count, 2) - 1) {
                        indices.append(faceVerts[0]); indices.append(faceVerts[i]); indices.append(faceVerts[i + 1])
                    }
                }
            } else {
                // skip unknown fixed-stride element; lists unsupported here
                let stride = el.props.reduce(0) { $0 + ($1.listCount == nil ? $1.type.size : -1) }
                if stride < 0 { throw MeshReadError.unsupported("list in element \(el.name)") }
                off += stride * el.count
            }
        }
    } else {
        // ascii
        guard let body = String(data: data[headerEnd.upperBound...], encoding: .ascii) else {
            throw MeshReadError.malformed("non-ascii body")
        }
        var lines = body.split(separator: "\n", omittingEmptySubsequences: true).makeIterator()
        for el in elements {
            for _ in 0..<el.count {
                guard let line = lines.next() else { throw MeshReadError.malformed("truncated ascii") }
                let vals = line.split(separator: " ").compactMap { Double($0) }
                if el.name == "vertex" {
                    var running = 0
                    for p in el.props {
                        if p.name == "x" || p.name == "y" || p.name == "z" { positions.append(Float(vals[running])) }
                        running += 1
                    }
                } else if el.name == "face" {
                    let n = Int(vals[0])
                    let fv = (0..<n).map { Int32(vals[1 + $0]) }
                    for i in 1..<(n - 1) {
                        indices.append(fv[0]); indices.append(fv[i]); indices.append(fv[i + 1])
                    }
                }
            }
        }
    }
    return LoadedMesh(positions: positions, indices: indices)
}

private func load<T>(_ bytes: [UInt8], _ offset: Int) -> T {
    bytes.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: T.self) }
}

// MARK: - GLB

private func readGLB(_ data: Data) throws -> LoadedMesh {
    guard data.count > 20 else { throw MeshReadError.malformed("too small") }
    let magic: UInt32 = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self) }
    guard magic == 0x46546C67 else { throw MeshReadError.malformed("not glb") }
    var off = 12
    var json: [String: Any]? = nil
    var bin: Data? = nil
    while off + 8 <= data.count {
        let len = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: UInt32.self) })
        let type = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off + 4, as: UInt32.self) }
        let chunk = data.subdata(in: (off + 8)..<(off + 8 + len))
        if type == 0x4E4F534A { json = try JSONSerialization.jsonObject(with: chunk) as? [String: Any] }
        if type == 0x004E4942 { bin = chunk }
        off += 8 + len
    }
    guard let gltf = json, let binData = bin else { throw MeshReadError.malformed("missing chunk") }
    guard let accessors = gltf["accessors"] as? [[String: Any]],
          let views = gltf["bufferViews"] as? [[String: Any]],
          let meshes = gltf["meshes"] as? [[String: Any]] else {
        throw MeshReadError.malformed("missing gltf arrays")
    }

    func accessorBytes(_ idx: Int) throws -> (Data, Int, Int, Int) {
        // returns (bytes at start of accessor, count, componentType, byteStride or 0)
        let acc = accessors[idx]
        guard let viewIdx = acc["bufferView"] as? Int else { throw MeshReadError.unsupported("sparse accessor") }
        let view = views[viewIdx]
        let viewOff = view["byteOffset"] as? Int ?? 0
        let accOff = acc["byteOffset"] as? Int ?? 0
        let count = acc["count"] as? Int ?? 0
        let comp = acc["componentType"] as? Int ?? 0
        let stride = view["byteStride"] as? Int ?? 0
        let start = viewOff + accOff
        return (binData.subdata(in: start..<binData.count), count, comp, stride)
    }

    var positions: [Float] = []
    var indices: [Int32] = []
    for mesh in meshes {
        for prim in (mesh["primitives"] as? [[String: Any]] ?? []) {
            if let mode = prim["mode"] as? Int, mode != 4 { continue }
            guard let attrs = prim["attributes"] as? [String: Any],
                  let posIdx = attrs["POSITION"] as? Int,
                  let idxIdx = prim["indices"] as? Int else { continue }
            let base = Int32(positions.count / 3)

            let (pBytes, pCount, pComp, pStride) = try accessorBytes(posIdx)
            guard pComp == 5126 else { throw MeshReadError.unsupported("POSITION comp \(pComp)") }
            let stride = pStride == 0 ? 12 : pStride
            pBytes.withUnsafeBytes { raw in
                for i in 0..<pCount {
                    positions.append(raw.loadUnaligned(fromByteOffset: i * stride + 0, as: Float.self))
                    positions.append(raw.loadUnaligned(fromByteOffset: i * stride + 4, as: Float.self))
                    positions.append(raw.loadUnaligned(fromByteOffset: i * stride + 8, as: Float.self))
                }
            }

            let (iBytes, iCount, iComp, _) = try accessorBytes(idxIdx)
            guard iComp == 5121 || iComp == 5123 || iComp == 5125 else {
                throw MeshReadError.unsupported("index comp \(iComp)")
            }
            iBytes.withUnsafeBytes { raw in
                for i in 0..<iCount {
                    let v: Int32
                    switch iComp {
                    case 5121: v = Int32(raw.loadUnaligned(fromByteOffset: i, as: UInt8.self))
                    case 5123: v = Int32(raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))
                    default: v = Int32(bitPattern: raw.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self))
                    }
                    indices.append(v + base)
                }
            }
        }
    }
    guard !positions.isEmpty, !indices.isEmpty else { throw MeshReadError.malformed("no triangle data") }
    return LoadedMesh(positions: positions, indices: indices)
}
