import Cxatlas

public enum AtlasError: Error, CustomStringConvertible {
    case unspecified
    case indexOutOfRange
    case invalidFaceVertexCount
    case invalidIndexCount

    init(_ raw: xatlas.AddMeshError) {
        switch raw {
        case xatlas.AddMeshError.IndexOutOfRange: self = .indexOutOfRange
        case xatlas.AddMeshError.InvalidFaceVertexCount: self = .invalidFaceVertexCount
        case xatlas.AddMeshError.InvalidIndexCount: self = .invalidIndexCount
        default: self = .unspecified
        }
    }

    public var description: String {
        switch self {
        case .unspecified: return "xatlas: unspecified error"
        case .indexOutOfRange: return "xatlas: index out of range"
        case .invalidFaceVertexCount: return "xatlas: invalid face vertex count (must be >= 3)"
        case .invalidIndexCount: return "xatlas: invalid index count (not divisible by 3)"
        }
    }
}
