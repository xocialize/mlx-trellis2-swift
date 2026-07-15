import Cxatlas

#if canImport(CoreGraphics)
import CoreGraphics
#endif

/// Bit-mask flags stored in `Atlas.image` pixels. Mirror the `kImage*`
/// constants in `xatlas.h`.
public enum ChartImageBits {
    public static let chartIndexMask: UInt32 = 0x1FFF_FFFF
    public static let hasChartIndexBit: UInt32 = 0x8000_0000
    public static let isBilinearBit: UInt32 = 0x4000_0000
    public static let isPaddingBit: UInt32 = 0x2000_0000
}

extension Atlas {
    /// True if the atlas was packed with `PackOptions.createImage = true` and
    /// the chart image is available.
    public var hasChartImage: Bool { handle.pointee.image != nil }

    /// Zero-copy borrow of one sub-atlas's chart image (width × height pixels).
    /// `body` receives a packed-UInt32 buffer in row-major order. Each pixel is
    /// bit-encoded — see `ChartImageBits`.
    public func withChartImage<R>(atlasIndex: Int = 0, _ body: (UnsafeBufferPointer<UInt32>, _ width: Int, _ height: Int) throws -> R) rethrows -> R? {
        guard let base = handle.pointee.image else { return nil }
        let w = Int(handle.pointee.width)
        let h = Int(handle.pointee.height)
        precondition(atlasIndex >= 0 && atlasIndex < Int(handle.pointee.atlasCount), "atlasIndex out of range")
        let start = base.advanced(by: atlasIndex * w * h)
        let buf = UnsafeBufferPointer(start: start, count: w * h)
        return try body(buf, w, h)
    }

    #if canImport(CoreGraphics)
    /// Render the chart image as a CGImage for quick inspection / debugging.
    /// Empty pixels are black, chart fills white, bilinear borders mid-grey,
    /// padding dark-grey. Returns nil if `createImage` was not requested.
    public func chartImageCGImage(atlasIndex: Int = 0) -> CGImage? {
        return withChartImage(atlasIndex: atlasIndex) { (buf, w, h) -> CGImage? in
            var bytes = [UInt8](repeating: 0, count: w * h)
            for i in 0..<(w * h) {
                let px = buf[i]
                if (px & ChartImageBits.hasChartIndexBit) != 0 {
                    bytes[i] = 255
                } else if (px & ChartImageBits.isBilinearBit) != 0 {
                    bytes[i] = 160
                } else if (px & ChartImageBits.isPaddingBit) != 0 {
                    bytes[i] = 80
                }
            }
            guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
            return CGImage(
                width: w,
                height: h,
                bitsPerComponent: 8,
                bitsPerPixel: 8,
                bytesPerRow: w,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
            )
        } ?? nil
    }
    #endif
}

#if canImport(CoreGraphics)
import Foundation
#endif
