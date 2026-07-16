import Foundation
import MLX
import MLXToolKit
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Host-side image preprocessing for the DINOv3 conditioner. Reproduces TRELLIS.2's
/// `preprocess_image`: honor a foreground alpha (a pre-masked RGBA input), crop to a square bbox
/// around the subject, composite `RGB×alpha` on black, resize to 512×512, ImageNet-normalize, and
/// emit **NCHW `[1,3,512,512]`** — the input layout of `Trellis2Pipeline.encodeImage`.
///
/// Background removal for a plain (opaque) RGB input rides the app-injected
/// `Trellis2Configuration.matting` hook (T0.4): `mattedIfNeeded` reproduces upstream's rembg branch
/// (matte the ≤1024-capped frame, graft the matte in as alpha) using only the canonical MLXToolKit
/// artifacts, so the package still carries no cross-package ModelPackage dependency (C13). With no
/// hook and no usable alpha we fall back to a full-frame resize (the background then leaks into the
/// mesh).
enum ImagePreprocess {
    static let size = 512
    static let maxSide = 1024
    static let alphaThreshold: Float = 0.8
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    /// Decode the canonical `Image`, apply the rembg-aware crop/composite, and return the normalized
    /// NCHW `[1,3,512,512]` pixel array DINOv3 expects.
    static func dinoPixels(from image: Image, size: Int = Self.size) throws -> MLXArray {
        let (cg, hasAlpha) = try decodeCGImage(image)
        let side0 = max(cg.width, cg.height)
        let scale = side0 > maxSide ? Double(maxSide) / Double(side0) : 1.0
        let W = max(1, Int((Double(cg.width) * scale).rounded()))
        let H = max(1, Int((Double(cg.height) * scale).rounded()))
        let pm = drawRGBA8(cg, width: W, height: H)     // premultipliedLast ⇒ RGB already RGB×alpha on black

        let rgb: [Float]   // channel-last [size*size*3], values [0,1]
        if hasAlpha, let bbox = foregroundBBox(pm, W: W, H: H) {
            rgb = cropCompositeResize(pm, W: W, H: H, bbox: bbox, target: size)
        } else {
            rgb = drawRGB(cg, side: size)
        }

        // ImageNet-normalize into an NCHW buffer directly: plane c, row y, col x → c*size*size + y*size + x.
        let n = size * size
        var chw = [Float](repeating: 0, count: 3 * n)
        for i in 0..<n {
            chw[0*n + i] = (rgb[3*i]     - mean[0]) / std[0]
            chw[1*n + i] = (rgb[3*i + 1] - mean[1]) / std[1]
            chw[2*n + i] = (rgb[3*i + 2] - mean[2]) / std[2]
        }
        return MLXArray(chw, [1, 3, size, size])
    }

    // MARK: matting (T0.4)

    /// Upstream `preprocess_image`'s rembg branch. Returns `image` untouched when it already carries
    /// a USABLE alpha (an alpha channel with any pixel < 255 — upstream's `not np.all(alpha == 255)`
    /// test, so a fully-opaque RGBA counts as raw). Otherwise: cap the frame at `maxSide` (upstream
    /// resizes before rembg), hand the opaque frame to the injected matter, and graft the returned
    /// matte in as the alpha channel. The RGBA result then flows through `dinoPixels`' alpha path
    /// (threshold-bbox crop → RGB×alpha composite on black), matching upstream end-to-end.
    static func mattedIfNeeded(_ image: Image, using matter: Trellis2Matting) async throws -> Image {
        let (cg, hasAlpha) = try decodeCGImage(image)
        let side0 = max(cg.width, cg.height)
        let scale = side0 > maxSide ? Double(maxSide) / Double(side0) : 1.0
        let W = max(1, Int((Double(cg.width) * scale).rounded()))
        let H = max(1, Int((Double(cg.height) * scale).rounded()))
        var pm = drawRGBA8(cg, width: W, height: H)
        if hasAlpha, stride(from: 3, to: pm.count, by: 4).contains(where: { pm[$0] < 255 }) {
            return image   // pre-masked input — honor its alpha, skip matting (upstream has_alpha)
        }

        // Opaque frame (alpha uniformly 255, so premultiplied == straight): matte it at the capped
        // resolution. The matte comes back at the input's dimensions (MattingContract), soft alpha.
        let framePNG = try encodePNG(pm, width: W, height: H, alphaInfo: .premultipliedLast)
        let matte = try await matter(Image(format: .png, data: framePNG, width: W, height: H))
        let alpha = try decodeMatte(matte, width: W, height: H)
        for i in 0..<(W * H) { pm[4*i + 3] = alpha[i] }
        // Straight (non-premultiplied) alpha: the RGB planes still hold the full-color frame.
        let rgbaPNG = try encodePNG(pm, width: W, height: H, alphaInfo: .last)
        return Image(format: .png, data: rgbaPNG, width: W, height: H)
    }

    /// Rasterize a `Matte` (grayscale PNG) to a `width×height` alpha plane; a matter that returns
    /// off-size (against the contract) is rescaled rather than rejected.
    private static func decodeMatte(_ matte: Matte, width: Int, height: Int) throws -> [UInt8] {
        guard let src = CGImageSourceCreateWithData(matte.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw Trellis2Error.matteDecodeFailed
        }
        var px = [UInt8](repeating: 0, count: width * height)
        try px.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width,
                                      space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
                throw Trellis2Error.matteDecodeFailed
            }
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return px
    }

    private static func encodePNG(_ rgba: [UInt8], width: Int, height: Int,
                                  alphaInfo: CGImageAlphaInfo) throws -> Data {
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                               bytesPerRow: width * 4, space: cs,
                               bitmapInfo: CGBitmapInfo(rawValue: alphaInfo.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: true,
                               intent: .defaultIntent) else {
            throw Trellis2Error.imageDecodeFailed
        }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else {
            throw Trellis2Error.imageDecodeFailed
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw Trellis2Error.imageDecodeFailed }
        return out as Data
    }

    // MARK: decode

    private static func decodeCGImage(_ image: Image) throws -> (cg: CGImage, hasAlpha: Bool) {
        switch image.format {
        case .png, .jpeg:
            guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                throw Trellis2Error.imageDecodeFailed
            }
            return (cg, cgHasAlpha(cg))
        case .rawBGRA8:
            guard let w = image.width, let h = image.height else { throw Trellis2Error.imageDecodeFailed }
            let bpr = image.bytesPerRow ?? (w * 4)
            let cs = CGColorSpaceCreateDeviceRGB()
            let info: CGBitmapInfo = [.byteOrder32Little,
                                      CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)]
            guard let provider = CGDataProvider(data: image.data as CFData),
                  let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                                   bytesPerRow: bpr, space: cs, bitmapInfo: info, provider: provider,
                                   decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
                throw Trellis2Error.imageDecodeFailed
            }
            return (cg, true)   // rawBGRA8 (premultipliedFirst) carries alpha by construction
        }
    }

    private static func cgHasAlpha(_ cg: CGImage) -> Bool {
        switch cg.alphaInfo {
        case .none, .noneSkipFirst, .noneSkipLast: return false
        default: return true
        }
    }

    // MARK: raster helpers

    private static func drawRGBA8(_ cg: CGImage, width: Int, height: Int) -> [UInt8] {
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        var px = [UInt8](repeating: 0, count: width * height * 4)
        px.withUnsafeMutableBytes { buf in
            let ctx = CGContext(data: buf.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: cs,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return px
    }

    private struct BBox { let x0: Int; let y0: Int; let side: Int }
    private static func foregroundBBox(_ pm: [UInt8], W: Int, H: Int) -> BBox? {
        let thr = UInt8(min(255, Int(alphaThreshold * 255)))
        var minX = W, minY = H, maxX = -1, maxY = -1
        for y in 0..<H {
            let row = y * W * 4
            for x in 0..<W where pm[row + x*4 + 3] > thr {
                if x < minX { minX = x }; if x > maxX { maxX = x }
                if y < minY { minY = y }; if y > maxY { maxY = y }
            }
        }
        guard maxX >= minX, maxY >= minY else { return nil }
        let cx = (minX + maxX) / 2, cy = (minY + maxY) / 2
        let side = max(maxX - minX, maxY - minY)
        return BBox(x0: cx - side/2, y0: cy - side/2, side: max(1, side))
    }

    private static func cropCompositeResize(_ pm: [UInt8], W: Int, H: Int, bbox: BBox, target: Int) -> [Float] {
        let s = bbox.side
        var crop = [UInt8](repeating: 0, count: s * s * 4)
        for j in 0..<s {
            let sy = bbox.y0 + j
            for i in 0..<s {
                let sx = bbox.x0 + i
                let d = (j * s + i) * 4
                if sx >= 0, sx < W, sy >= 0, sy < H {
                    let srcAlpha = pm[(sy * W + sx) * 4 + 3]
                    if srcAlpha == 0 { crop[d+3] = 255; continue }
                    crop[d]   = pm[(sy * W + sx) * 4]
                    crop[d+1] = pm[(sy * W + sx) * 4 + 1]
                    crop[d+2] = pm[(sy * W + sx) * 4 + 2]
                }
                crop[d+3] = 255
            }
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(crop) as CFData),
              let cropCG = CGImage(width: s, height: s, bitsPerComponent: 8, bitsPerPixel: 32,
                                   bytesPerRow: s * 4, space: cs,
                                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                   provider: provider, decode: nil, shouldInterpolate: true,
                                   intent: .defaultIntent) else {
            return drawRGBFromBytes(crop, side: s)
        }
        return drawRGB(cropCG, side: target)
    }

    private static func drawRGB(_ cg: CGImage, side: Int) -> [Float] {
        let cs = CGColorSpaceCreateDeviceRGB()
        var px = [UInt8](repeating: 0, count: side * side * 4)
        px.withUnsafeMutableBytes { buf in
            let ctx = CGContext(data: buf.baseAddress, width: side, height: side, bitsPerComponent: 8,
                                bytesPerRow: side * 4, space: cs,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: side, height: side))
        }
        return drawRGBFromBytes(px, side: side)
    }

    private static func drawRGBFromBytes(_ px: [UInt8], side: Int) -> [Float] {
        var rgb = [Float](repeating: 0, count: side * side * 3)
        for i in 0..<(side * side) {
            rgb[3*i]   = Float(px[4*i])   / 255.0
            rgb[3*i+1] = Float(px[4*i+1]) / 255.0
            rgb[3*i+2] = Float(px[4*i+2]) / 255.0
        }
        return rgb
    }
}
