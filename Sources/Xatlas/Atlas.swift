import Cxatlas
import Dispatch
import Foundation

/// Reference handle to an xatlas atlas. Holds a `xatlas::Atlas*` and tears it
/// down on deinit.
///
/// Thread-safety: a single `Atlas` instance is **not** safe for concurrent use.
/// Serialise all calls on a given instance. Marked `@unchecked Sendable` so it
/// can be moved between actors / Tasks, but callers must enforce one-active-op
/// at a time.
public final class Atlas: @unchecked Sendable {
    @usableFromInline let handle: UnsafeMutablePointer<xatlas.Atlas>
    private var progressBox: ProgressBox?

    public init() {
        guard let h = xatlas.Create() else {
            fatalError("xatlas::Create returned null")
        }
        self.handle = h
    }

    deinit {
        xatlas.Destroy(handle)
    }

    // MARK: - Inputs

    public func addMesh(_ mesh: MeshInput, meshCountHint: UInt32 = 0) throws {
        try mesh.withCxxDecl { decl in
            let err = xatlas.AddMesh(handle, decl, meshCountHint)
            if err != xatlas.AddMeshError.Success {
                throw AtlasError(err)
            }
        }
    }

    /// Low-level overload — caller-owned raw memory + custom strides.
    public func addMesh(_ raw: RawMeshInput, meshCountHint: UInt32 = 0) throws {
        let err = xatlas.AddMesh(handle, raw.cxx, meshCountHint)
        if err != xatlas.AddMeshError.Success {
            throw AtlasError(err)
        }
    }

    public func addUvMesh(_ mesh: UvMeshInput) throws {
        try mesh.withCxxDecl { decl in
            let err = xatlas.AddUvMesh(handle, decl)
            if err != xatlas.AddMeshError.Success {
                throw AtlasError(err)
            }
        }
    }

    public func addMeshJoin() {
        xatlas.AddMeshJoin(handle)
    }

    // MARK: - Generation

    public func computeCharts(options: ChartOptions = ChartOptions()) {
        withParameterize(options) { cxx in
            xatlas.ComputeCharts(handle, cxx)
        }
    }

    public func packCharts(options: PackOptions = PackOptions()) {
        xatlas.PackCharts(handle, options.cxx)
    }

    public func generate(
        chartOptions: ChartOptions = ChartOptions(),
        packOptions: PackOptions = PackOptions()
    ) {
        withParameterize(chartOptions) { cxx in
            xatlas.Generate(handle, cxx, packOptions.cxx)
        }
    }

    /// Async wrapper — runs `generate` off the calling actor on a global
    /// dispatch queue. For cooperative cancellation, install a progress
    /// callback that returns `false`, or use `generateProgress(...)` which
    /// translates `Task.cancel` into a stop signal automatically.
    public func generate(
        chartOptions: ChartOptions = ChartOptions(),
        packOptions: PackOptions = PackOptions()
    ) async throws {
        try Task.checkCancellation()
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                self.generate(chartOptions: chartOptions, packOptions: packOptions)
                cont.resume()
            }
        }
        try Task.checkCancellation()
    }

    /// Stream progress events while xatlas runs `generate` on a background
    /// queue. The stream finishes when generation completes. If the consuming
    /// task is cancelled (or `for-await` is broken out of), the stream's
    /// `onTermination` translates that into the xatlas-side cancel by
    /// returning `false` from the progress callback.
    public func generateProgress(
        chartOptions: ChartOptions = ChartOptions(),
        packOptions: PackOptions = PackOptions()
    ) -> AsyncStream<(ProgressCategory, Int32)> {
        AsyncStream { continuation in
            let cancel = CancelFlag()
            continuation.onTermination = { _ in cancel.cancel() }

            setProgressCallback { category, progress in
                continuation.yield((category, progress))
                return !cancel.isCancelled
            }

            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self else { continuation.finish(); return }
                self.generate(chartOptions: chartOptions, packOptions: packOptions)
                self.setProgressCallback(nil)
                continuation.finish()
            }
        }
    }

    // MARK: - Progress

    public func setProgressCallback(_ callback: ((ProgressCategory, Int32) -> Bool)?) {
        guard let callback else {
            progressBox = nil
            xatlas.SetProgressCallback(handle, nil, nil)
            return
        }
        let box = ProgressBox(callback: callback)
        progressBox = box
        let raw = Unmanaged.passUnretained(box).toOpaque()
        xatlas.SetProgressCallback(handle, progressThunk, raw)
    }

    // MARK: - Output

    public var width: UInt32 { handle.pointee.width }
    public var height: UInt32 { handle.pointee.height }
    public var atlasCount: UInt32 { handle.pointee.atlasCount }
    public var chartCount: UInt32 { handle.pointee.chartCount }
    public var meshCount: UInt32 { handle.pointee.meshCount }
    public var texelsPerUnit: Float { handle.pointee.texelsPerUnit }

    public var meshes: MeshOutputCollection { MeshOutputCollection(atlas: self) }

    public var utilization: [Float] {
        let count = Int(handle.pointee.atlasCount)
        guard count > 0, let ptr = handle.pointee.utilization else { return [] }
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    // MARK: - Helpers

    private func withParameterize(_ options: ChartOptions, body: (xatlas.ChartOptions) -> Void) {
        var cxx = options.cxx
        if let closure = options.customParameterize {
            cxx.paramFunc = ParameterizeBridge.install(closure)
            body(cxx)
            ParameterizeBridge.uninstall()
        } else {
            body(cxx)
        }
    }
}

private final class ProgressBox {
    let callback: (ProgressCategory, Int32) -> Bool
    init(callback: @escaping (ProgressCategory, Int32) -> Bool) {
        self.callback = callback
    }
}

private let progressThunk: @convention(c) (xatlas.ProgressCategory, Int32, UnsafeMutableRawPointer?) -> Bool = { category, progress, userData in
    guard let userData else { return true }
    let box = Unmanaged<ProgressBox>.fromOpaque(userData).takeUnretainedValue()
    return box.callback(ProgressCategory(category), progress)
}

/// Thread-safe cancellation flag for async wrappers.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var _cancelled = false
    var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return _cancelled
    }
    func cancel() {
        lock.lock(); defer { lock.unlock() }
        _cancelled = true
    }
}
