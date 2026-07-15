import Cxatlas

/// Helper that installs the global Swift parameterise closure for the duration
/// of `body`, then clears it. Used internally by `Atlas.computeCharts` /
/// `Atlas.generate` when `ChartOptions.customParameterize` is non-nil.
enum ParameterizeBridge {
    static var current: ((UnsafeBufferPointer<SIMD3<Float>>, UnsafeMutableBufferPointer<SIMD2<Float>>, UnsafeBufferPointer<UInt32>) -> Void)?

    static let thunk: @convention(c) (
        UnsafePointer<Float>?,
        UnsafeMutablePointer<Float>?,
        UInt32,
        UnsafePointer<UInt32>?,
        UInt32
    ) -> Void = { pos, tex, vcount, idx, icount in
        guard let cb = current, let pos, let tex, let idx else { return }
        let positions = UnsafeRawPointer(pos).bindMemory(to: SIMD3<Float>.self, capacity: Int(vcount))
        let texcoords = UnsafeMutableRawPointer(tex).bindMemory(to: SIMD2<Float>.self, capacity: Int(vcount))
        let posBuf = UnsafeBufferPointer(start: positions, count: Int(vcount))
        let texBuf = UnsafeMutableBufferPointer(start: texcoords, count: Int(vcount))
        let idxBuf = UnsafeBufferPointer(start: idx, count: Int(icount))
        cb(posBuf, texBuf, idxBuf)
    }

    /// Install `closure` as the global paramFunc and return a C function
    /// pointer to assign to `ChartOptions::paramFunc`. Caller must clear via
    /// `uninstall()` after the xatlas call returns.
    static func install(_ closure: @escaping (UnsafeBufferPointer<SIMD3<Float>>, UnsafeMutableBufferPointer<SIMD2<Float>>, UnsafeBufferPointer<UInt32>) -> Void) -> xatlas.ParameterizeFunc {
        current = closure
        xatlas_install_parameterize { pos, tex, vcount, idx, icount in
            ParameterizeBridge.thunk(pos, tex, vcount, idx, icount)
        }
        // We registered above; xatlas_parameterize_thunk now returns our
        // C ABI thunk pointer. Cast to the xatlas typedef.
        return unsafeBitCast(xatlas_parameterize_thunk(), to: xatlas.ParameterizeFunc.self)
    }

    static func uninstall() {
        current = nil
        xatlas_install_parameterize(nil)
    }
}
