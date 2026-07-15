import Cxatlas

public struct ChartOptions: @unchecked Sendable {
    // @unchecked because of `customParameterize`: the closure is invoked from
    // xatlas's worker threads, and we route it through a process-global slot
    // (per the doc on that property). Treat instances as single-owner.
    public var maxChartArea: Float = 0
    public var maxBoundaryLength: Float = 0
    public var normalDeviationWeight: Float = 2.0
    public var roundnessWeight: Float = 0.01
    public var straightnessWeight: Float = 6.0
    public var normalSeamWeight: Float = 4.0
    public var textureSeamWeight: Float = 0.5
    public var maxCost: Float = 2.0
    public var maxIterations: UInt32 = 1
    public var useInputMeshUvs: Bool = false
    public var fixWinding: Bool = false

    /// Optional custom UV parameterisation. Receives input positions and
    /// triangle indices; must fill `texcoords` with a UV per vertex.
    ///
    /// xatlas's `paramFunc` slot is process-global (no userData). Setting this
    /// installs a single global Swift closure for the duration of the next
    /// `computeCharts`/`generate` call. Concurrent calls across different
    /// `Atlas` instances must serialise. Cleared automatically after use.
    public var customParameterize: ((
        _ positions: UnsafeBufferPointer<SIMD3<Float>>,
        _ texcoords: UnsafeMutableBufferPointer<SIMD2<Float>>,
        _ indices: UnsafeBufferPointer<UInt32>
    ) -> Void)?

    public init() {}

    var cxx: xatlas.ChartOptions {
        var o = xatlas.ChartOptions()
        o.maxChartArea = maxChartArea
        o.maxBoundaryLength = maxBoundaryLength
        o.normalDeviationWeight = normalDeviationWeight
        o.roundnessWeight = roundnessWeight
        o.straightnessWeight = straightnessWeight
        o.normalSeamWeight = normalSeamWeight
        o.textureSeamWeight = textureSeamWeight
        o.maxCost = maxCost
        o.maxIterations = maxIterations
        o.useInputMeshUvs = useInputMeshUvs
        o.fixWinding = fixWinding
        // paramFunc is installed by ParameterizeBridge.use(...) at call sites.
        return o
    }
}

public struct PackOptions: Sendable {
    public var maxChartSize: UInt32 = 0
    public var padding: UInt32 = 0
    public var texelsPerUnit: Float = 0
    public var resolution: UInt32 = 0
    public var bilinear: Bool = true
    public var blockAlign: Bool = false
    public var bruteForce: Bool = false
    public var createImage: Bool = false
    public var rotateChartsToAxis: Bool = true
    public var rotateCharts: Bool = true

    public init() {}

    var cxx: xatlas.PackOptions {
        var o = xatlas.PackOptions()
        o.maxChartSize = maxChartSize
        o.padding = padding
        o.texelsPerUnit = texelsPerUnit
        o.resolution = resolution
        o.bilinear = bilinear
        o.blockAlign = blockAlign
        o.bruteForce = bruteForce
        o.createImage = createImage
        o.rotateChartsToAxis = rotateChartsToAxis
        o.rotateCharts = rotateCharts
        return o
    }
}

public enum ProgressCategory: Sendable {
    case addMesh, computeCharts, packCharts, buildOutputMeshes

    init(_ raw: xatlas.ProgressCategory) {
        switch raw {
        case xatlas.ProgressCategory.AddMesh: self = .addMesh
        case xatlas.ProgressCategory.ComputeCharts: self = .computeCharts
        case xatlas.ProgressCategory.PackCharts: self = .packCharts
        default: self = .buildOutputMeshes
        }
    }
}

public enum ChartType: Sendable {
    case planar, ortho, lscm, piecewise, invalid

    init(_ raw: xatlas.ChartType) {
        switch raw {
        case xatlas.ChartType.Planar: self = .planar
        case xatlas.ChartType.Ortho: self = .ortho
        case xatlas.ChartType.LSCM: self = .lscm
        case xatlas.ChartType.Piecewise: self = .piecewise
        default: self = .invalid
        }
    }
}
