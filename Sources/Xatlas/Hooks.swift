import Cxatlas
import Foundation

/// Process-global hooks. xatlas's `SetPrint` / `SetAlloc` are not per-Atlas;
/// these helpers install bridges that route to Swift closures or C function
/// pointers.
public enum XatlasHooks {

    // MARK: - Print

    private static let printLock = NSLock()
    private static var printCallback: ((String) -> Void)?

    /// Install a process-wide print callback. xatlas calls this for log /
    /// diagnostic output. Pass `nil` to remove. Useful for capturing log
    /// output in tests or routing to OSLog. Setting a callback enables
    /// xatlas's print path; `verbose` toggles xatlas's verbose mode.
    public static func setPrintCallback(_ callback: ((String) -> Void)?, verbose: Bool = false) {
        printLock.lock()
        printCallback = callback
        printLock.unlock()
        if callback != nil {
            xatlas_install_print(printThunk, verbose ? 1 : 0)
        } else {
            xatlas_install_print(nil, 0)
        }
    }

    fileprivate static let printThunk: @convention(c) (UnsafePointer<CChar>?) -> Void = { cstr in
        guard let cstr else { return }
        let s = String(cString: cstr)
        printLock.lock()
        let cb = printCallback
        printLock.unlock()
        cb?(s)
    }

    // MARK: - Alloc

    /// Install a custom realloc/free pair. Both must be plain C functions
    /// (`@convention(c)`, no captures) — xatlas calls them from many threads
    /// with no userData slot. Mostly useful for tracking memory in tests or
    /// redirecting to a sandbox allocator.
    public static func setAllocator(
        realloc: (@convention(c) (UnsafeMutableRawPointer?, Int) -> UnsafeMutableRawPointer?)?,
        free: (@convention(c) (UnsafeMutableRawPointer?) -> Void)? = nil
    ) {
        xatlas.SetAlloc(realloc, free)
    }
}
