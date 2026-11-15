import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXToolKit
import TRELLIS2
import Trellis2Kit

// PIXAL3D EVAL (AB-T-0485 P3) — how well a Pixal3D GLB lands on its photo, CPU only.
// Renders the textured GLB through the run's camera (the metrics record's `pixal3d_camera0_to_mesh`
// and `pixal3d_fov_deg`) at 512² in the single-view crop frame (×1.1, `Pixal3DPreprocess`), and
// compares it with the photo:
//   iou  — rendered silhouette vs the photo's alpha > 0.5 (P1's alignment measure, AB-R-0865)
//   dE   — mean CIE76 ΔE between the rendered base colour and the photo, over pixels inside both
//          masks with alpha > 0.9 (unlit base colour vs a lit photo: compare variants, not absolutes)
//   atlasDE — P1's crude global measure: mean Lab of the covered atlas vs the photo's masked mean
// Usage: pixal3deval <glb> <metrics.json> <input image> [out.json] [render.png]

setvbuf(stdout, nil, _IONBF, 0)
let args = CommandLine.arguments
guard args.count >= 4 else { fatalError("usage: pixal3deval <glb> <metrics.json> <image> [out.json] [render.png]") }
let res = 512

// ---- GLB ----
let glb = try Data(contentsOf: URL(fileURLWithPath: args[1]))
func u32(_ o: Int) -> Int { Int(glb.subdata(in: o ..< o + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian) }
let jsonLen = u32(12)
let json = try JSONSerialization.jsonObject(with: glb.subdata(in: 20 ..< 20 + jsonLen)) as! [String: Any]
let binStart = 20 + jsonLen + 8
let bufferViews = json["bufferViews"] as! [[String: Any]]
let accessors = json["accessors"] as! [[String: Any]]
func viewBytes(_ bv: Int) -> Data {
    let v = bufferViews[bv]
    let off = (v["byteOffset"] as? Int) ?? 0, len = v["byteLength"] as! Int
    return glb.subdata(in: binStart + off ..< binStart + off + len)
}
func floats(_ acc: Int) -> [Float] {
    viewBytes(accessors[acc]["bufferView"] as! Int).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}
let prim = ((json["meshes"] as! [[String: Any]])[0]["primitives"] as! [[String: Any]])[0]
let attrs = prim["attributes"] as! [String: Int]
let P = floats(attrs["POSITION"]!), UV = floats(attrs["TEXCOORD_0"]!)
let I: [UInt32] = viewBytes(accessors[prim["indices"] as! Int]["bufferView"] as! Int)
    .withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
let images = json["images"] as! [[String: Any]]
let pngData = viewBytes(images[0]["bufferView"] as! Int)
guard let src = CGImageSourceCreateWithData(pngData as CFData, nil),
      let atlasCG = CGImageSourceCreateImageAtIndex(src, 0, nil) else { fatalError("atlas decode") }
let AW = atlasCG.width, AH = atlasCG.height
var atlas = [UInt8](repeating: 0, count: AW * AH * 4)
atlas.withUnsafeMutableBytes { buf in
    let ctx = CGContext(data: buf.baseAddress, width: AW, height: AH, bitsPerComponent: 8, bytesPerRow: AW * 4,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(atlasCG, in: CGRect(x: 0, y: 0, width: AW, height: AH))
}

// ---- camera ----
let rec = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[2]))) as! [String: Any]
let c2w = Mat4(flat: (rec["pixal3d_camera0_to_mesh"] as! [Any]).map { ($0 as! NSNumber).doubleValue })
let fov = (rec["pixal3d_fov_deg"] as! NSNumber).doubleValue * .pi / 180
let w2c = c2w.inverse
let f = Double(res) / (2 * tan(fov / 2))

// ---- photo ----
let img = Image(format: .png, data: try Data(contentsOf: URL(fileURLWithPath: args[3])))
let guide = try Pixal3DPreprocess.singleView(img, size: res).guide.asArray(Float.self)   // [3,res,res]
guard let alpha = try Pixal3DPreprocess.singleViewAlpha(img, size: res) else { fatalError("input has no alpha") }

// ---- rasterize (z-buffer, perspective-correct UV) ----
let nV = P.count / 3
var sx = [Double](repeating: 0, count: nV), sy = sx, sz = sx
for i in 0..<nV {
    let x = Double(P[3*i]), y = Double(P[3*i+1]), z = Double(P[3*i+2])
    let cx = w2c[0, 0]*x + w2c[0, 1]*y + w2c[0, 2]*z + w2c[0, 3]
    let cy = w2c[1, 0]*x + w2c[1, 1]*y + w2c[1, 2]*z + w2c[1, 3]
    let cz = w2c[2, 0]*x + w2c[2, 1]*y + w2c[2, 2]*z + w2c[2, 3]
    let depth = -cz
    sx[i] = f * cx / depth + Double(res) / 2       // pixel i's centre sits at i (upstream convention)
    sy[i] = -f * cy / depth + Double(res) / 2
    sz[i] = depth
}
var zbuf = [Double](repeating: .infinity, count: res * res)
var cu = [Float](repeating: 0, count: res * res), cv = cu
var cov = [Bool](repeating: false, count: res * res)
let nF = I.count / 3
for t in 0..<nF {
    let a = Int(I[3*t]), b = Int(I[3*t+1]), c = Int(I[3*t+2])
    let x0 = sx[a], y0 = sy[a], x1 = sx[b], y1 = sy[b], x2 = sx[c], y2 = sy[c]
    let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(area) < 1e-12 { continue }
    let minX = max(0, Int(floor(min(x0, x1, x2)))), maxX = min(res - 1, Int(ceil(max(x0, x1, x2))))
    let minY = max(0, Int(floor(min(y0, y1, y2)))), maxY = min(res - 1, Int(ceil(max(y0, y1, y2))))
    if minX > maxX || minY > maxY { continue }
    for py in minY...maxY {
        for px in minX...maxX {
            let X = Double(px), Y = Double(py)
            let w0 = ((x1 - X) * (y2 - Y) - (x2 - X) * (y1 - Y)) / area
            let w1 = ((x2 - X) * (y0 - Y) - (x0 - X) * (y2 - Y)) / area
            let w2 = 1 - w0 - w1
            if w0 < 0 || w1 < 0 || w2 < 0 { continue }
            let iz = w0 / sz[a] + w1 / sz[b] + w2 / sz[c]
            let z = 1 / iz
            let k = py * res + px
            if z >= zbuf[k] { continue }
            zbuf[k] = z
            let p0 = w0 / sz[a] * z, p1 = w1 / sz[b] * z, p2 = w2 / sz[c] * z
            cu[k] = Float(p0 * Double(UV[2*a]) + p1 * Double(UV[2*b]) + p2 * Double(UV[2*c]))
            cv[k] = Float(p0 * Double(UV[2*a+1]) + p1 * Double(UV[2*b+1]) + p2 * Double(UV[2*c+1]))
            cov[k] = true
        }
    }
}

// ---- metrics ----
func lab(_ r: Float, _ g: Float, _ b: Float) -> (Float, Float, Float) {
    func lin(_ c: Float) -> Float { c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
    let R = lin(r), G = lin(g), B = lin(b)
    let X = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047
    let Yv = 0.2126 * R + 0.7152 * G + 0.0722 * B
    let Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883
    func ff(_ t: Float) -> Float { t > 0.008856 ? cbrt(t) : 7.787 * t + 16 / 116 }
    let fx = ff(X), fy = ff(Yv), fz = ff(Z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))
}
func atlasRGB(_ u: Float, _ v: Float) -> (Float, Float, Float) {
    let x = min(max(u * Float(AW) - 0.5, 0), Float(AW - 1)), y = min(max(v * Float(AH) - 0.5, 0), Float(AH - 1))
    let x0 = Int(x), y0 = Int(y), x1 = min(x0 + 1, AW - 1), y1 = min(y0 + 1, AH - 1)
    let fx = x - Float(x0), fy = y - Float(y0)
    func px(_ xx: Int, _ yy: Int, _ ch: Int) -> Float { Float(atlas[(yy * AW + xx) * 4 + ch]) / 255 }
    func ch(_ c: Int) -> Float {
        (px(x0, y0, c) * (1 - fx) + px(x1, y0, c) * fx) * (1 - fy) + (px(x0, y1, c) * (1 - fx) + px(x1, y1, c) * fx) * fy
    }
    return (ch(0), ch(1), ch(2))
}
var inter = 0, uni = 0, n = 0
var dE = 0.0
var render = [UInt8](repeating: 0, count: res * res * 4)
var photoLab = (0.0, 0.0, 0.0), photoN = 0
let plane = res * res
for k in 0..<plane {
    let a = alpha[k] > 0.5
    if a && cov[k] { inter += 1 }
    if a || cov[k] { uni += 1 }
    if alpha[k] > 0.9 {
        let p = lab(guide[k], guide[plane + k], guide[2 * plane + k])
        photoLab.0 += Double(p.0); photoLab.1 += Double(p.1); photoLab.2 += Double(p.2); photoN += 1
    }
    if cov[k] {
        let (r, g, b) = atlasRGB(cu[k], cv[k])
        render[4*k] = UInt8(r * 255); render[4*k+1] = UInt8(g * 255); render[4*k+2] = UInt8(b * 255); render[4*k+3] = 255
        if alpha[k] > 0.9 {
            let q = lab(r, g, b), p = lab(guide[k], guide[plane + k], guide[2 * plane + k])
            dE += Double(((q.0 - p.0) * (q.0 - p.0) + (q.1 - p.1) * (q.1 - p.1) + (q.2 - p.2) * (q.2 - p.2)).squareRoot())
            n += 1
        }
    }
}
// P1's global atlas measure: mean Lab of covered (non-black, opaque) atlas texels vs the photo's mean
var aLab = (0.0, 0.0, 0.0), aN = 0
for i in Swift.stride(from: 0, to: AW * AH, by: 7) where atlas[4*i+3] > 0 && (Int(atlas[4*i]) + Int(atlas[4*i+1]) + Int(atlas[4*i+2])) > 0 {
    let q = lab(Float(atlas[4*i]) / 255, Float(atlas[4*i+1]) / 255, Float(atlas[4*i+2]) / 255)
    aLab.0 += Double(q.0); aLab.1 += Double(q.1); aLab.2 += Double(q.2); aN += 1
}
let pm = (photoLab.0 / Double(photoN), photoLab.1 / Double(photoN), photoLab.2 / Double(photoN))
let am = (aLab.0 / Double(max(aN, 1)), aLab.1 / Double(max(aN, 1)), aLab.2 / Double(max(aN, 1)))
let atlasDE = ((am.0 - pm.0) * (am.0 - pm.0) + (am.1 - pm.1) * (am.1 - pm.1) + (am.2 - pm.2) * (am.2 - pm.2)).squareRoot()
let iou = Double(inter) / Double(max(uni, 1))
let meanDE = dE / Double(max(n, 1))
let out: [String: Any] = ["glb": args[1], "iou": iou, "front_dE": meanDE, "front_px": n, "atlas_dE": atlasDE,
                          "faces": nF, "hr_tokens": rec["hr_tokens"] ?? NSNull(), "tier": rec["tier"] ?? NSNull()]
print(String(format: "EVAL iou=%.4f front_dE=%.2f (n=%d) atlas_dE=%.2f faces=%d  %@", iou, meanDE, n, atlasDE, nF,
             (args[1] as NSString).lastPathComponent))
if args.count >= 5, let d = try? JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys]) {
    try d.write(to: URL(fileURLWithPath: args[4]))
}
if args.count >= 6 {
    // side by side: photo | render
    var both = [UInt8](repeating: 0, count: res * 2 * res * 4)
    for y in 0..<res { for x in 0..<res {
        let k = y * res + x, o = (y * 2 * res + x) * 4, o2 = (y * 2 * res + res + x) * 4
        both[o] = UInt8(guide[k] * 255); both[o+1] = UInt8(guide[plane + k] * 255); both[o+2] = UInt8(guide[2 * plane + k] * 255); both[o+3] = 255
        both[o2] = render[4*k]; both[o2+1] = render[4*k+1]; both[o2+2] = render[4*k+2]; both[o2+3] = 255
    } }
    let provider = CGDataProvider(data: Data(both) as CFData)!
    let cg = CGImage(width: 2 * res, height: res, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 2 * res * 4,
                     space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                     provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let dst = CGImageDestinationCreateWithURL(URL(fileURLWithPath: args[5]) as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(dst, cg, nil); CGImageDestinationFinalize(dst)
}
