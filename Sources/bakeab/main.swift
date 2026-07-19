// bakeab — bake-quality A/B for UnwrapBackend (UV-UNWRAP-METAL-PLAN.md Phase 2 gate).
//
// Runs MeshBake.run with .xatlas and .provenance on the same golden fixtures,
// then scores each bake against GROUND TRUTH: uniform surface samples on the
// result mesh, baked color fetched via that mesh's UVs, truth color sampled
// from the source voxel attribute field (BVH remap to raw shell + trilinear —
// the same machinery the bake itself uses). Captures texel starvation, folds,
// and seam bleed as color error. Writes both GLBs for visual inspection.
//
//   swift run -c release bakeab [octant] [samples N]
import Foundation
import simd
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXMesh
import TRELLIS2

func writePNG(_ pixels: [UInt8], _ width: Int, _ height: Int, to path: String) {
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    var px = pixels
    let ctx = px.withUnsafeMutableBytes { buf in
        CGContext(data: buf.baseAddress, width: width, height: height,
                  bitsPerComponent: 8, bytesPerRow: width * 4, space: cs,
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    }
    guard let img = ctx?.makeImage(),
          let dest = CGImageDestinationCreateWithURL(
            URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { return }
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
}

let goldens = "/Volumes/Satechi/TrellisRedux/trellis2-port/goldens"
func golden(_ n: String) throws -> MLXArray { try loadArray(url: URL(fileURLWithPath: "\(goldens)/\(n).npy")) }
let outDir = "/private/tmp/claude-501/-Users-dustinnielson-Development-vroid-xwear-interop/af2c103b-8a4f-4a6b-a495-b75e5e787ea5/scratchpad"

let full = !CommandLine.arguments.contains("octant")
var sampleCount = 150_000
var targetFaces = 120_000
var atlasSize = 1024
func argValue(_ name: String) -> Int? {
    guard let i = CommandLine.arguments.firstIndex(of: name),
          i + 1 < CommandLine.arguments.count else { return nil }
    return Int(CommandLine.arguments[i + 1])
}
sampleCount = argValue("samples") ?? sampleCount
targetFaces = argValue("target") ?? targetFaces
atlasSize = argValue("atlas") ?? atlasSize
var remeshRes = argValue("remesh") ?? 256
var onlyBackend: String? = nil
if let i = CommandLine.arguments.firstIndex(of: "only"), i + 1 < CommandLine.arguments.count {
    onlyBackend = CommandLine.arguments[i + 1]
}
let (shN, texN) = full ? ("shapedec_out_feats", "texdec_full_feats")
                       : ("shapedec_sm_out_feats", "texdec_out_feats")
let coordsN = full ? "shapedec_out_coords" : "shapedec_sm_out_coords"

let shapeFeats = try golden(shN)
let coords = (try golden(coordsN)).asType(.int32)
let texFeats = try golden(texN)
let baseColor = MLX.clip(texFeats[0..., 0..<3] * 0.5 + 0.5, min: 0, max: 1)
let fineRes: Float = 1024
print("[bakeab] fixtures: \(coords.dim(0)) voxels (\(full ? "FULL" : "octant")), \(sampleCount) eval samples")

// Rebuild the raw dual-grid shell once — the truth-sampling reference surface
// (mirrors MeshBake.run steps 0-1).
func softplus(_ x: MLXArray) -> MLXArray { MLX.maximum(x, 0) + MLX.log(1 + MLX.exp(-MLX.abs(x))) }
let margin: Float = 0.5
let dv = (1 + 2 * margin) * MLX.sigmoid(shapeFeats[0..., 0..<3]) - margin
let inter = shapeFeats[0..., 3..<6] .> 0
let ql = softplus(shapeFeats[0..., 6..<7])
let (rawV, rawF) = DualGridMesh.extract(
    coords: coords[0..., 1..<4], dualVerts: dv, intersected: inter, quadLerp: ql, gridSize: fineRes)
MLX.eval(rawV, rawF)
let shell = Mesh(vertices: rawV, faces: rawF)
let shellBVH = shell.bvh()

struct EvalResult {
    var bakeSeconds: Double
    var meanErr: Double
    var p95Err: Double
    var badFrac: Double     // fraction of samples with L2 error > 0.1
    var coverage: Float
    var faces: Int
}

func evaluate(_ backend: UnwrapBackend) throws -> EvalResult {
    print("=== backend \(backend.rawValue)")
    let t0 = CFAbsoluteTimeGetCurrent()
    let baked = try MeshBake.run(
        shapeFeats: shapeFeats, coords: coords, texBaseColor: baseColor,
        fineRes: fineRes, remeshRes: remeshRes, targetFaces: targetFaces, atlasSize: atlasSize,
        backend: backend)
    let bakeSeconds = CFAbsoluteTimeGetCurrent() - t0

    // --- uniform surface samples (area-weighted faces, fixed seed) ---
    let v = baked.vertices.asArray(Float.self)
    let f = baked.faces.asArray(Int32.self)
    let uvArr = baked.uvs.asArray(Float.self)
    let F = f.count / 3
    var cumArea = [Double](repeating: 0, count: F)
    var acc = 0.0
    for fi in 0..<F {
        let i0 = Int(f[fi*3])*3, i1 = Int(f[fi*3+1])*3, i2 = Int(f[fi*3+2])*3
        let e1 = SIMD3<Float>(v[i1]-v[i0], v[i1+1]-v[i0+1], v[i1+2]-v[i0+2])
        let e2 = SIMD3<Float>(v[i2]-v[i0], v[i2+1]-v[i0+1], v[i2+2]-v[i0+2])
        let cr = SIMD3<Float>(e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x)
        acc += Double((cr.x*cr.x + cr.y*cr.y + cr.z*cr.z).squareRoot())
        cumArea[fi] = acc
    }
    var rng = SystemRandomNumberGenerator()  // eval variance is fine; n is large
    var pts = [Float](); pts.reserveCapacity(sampleCount * 3)
    var bakedCol = [Float](); bakedCol.reserveCapacity(sampleCount * 3)
    var sampleFace = [Int](); sampleFace.reserveCapacity(sampleCount)
    let S = baked.atlasSize
    for _ in 0..<sampleCount {
        let target = Double.random(in: 0..<acc, using: &rng)
        var lo = 0, hi = F - 1
        while lo < hi { let mid = (lo + hi) / 2; if cumArea[mid] < target { lo = mid + 1 } else { hi = mid } }
        let fi = lo
        sampleFace.append(fi)
        var a = Float.random(in: 0...1, using: &rng), b = Float.random(in: 0...1, using: &rng)
        if a + b > 1 { a = 1 - a; b = 1 - b }
        let c = 1 - a - b
        let i0 = Int(f[fi*3]), i1 = Int(f[fi*3+1]), i2 = Int(f[fi*3+2])
        for k in 0..<3 {
            pts.append(a * v[i0*3+k] + b * v[i1*3+k] + c * v[i2*3+k])
        }
        let u = a * uvArr[i0*2] + b * uvArr[i1*2] + c * uvArr[i2*2]
        let w = a * uvArr[i0*2+1] + b * uvArr[i1*2+1] + c * uvArr[i2*2+1]
        // nearest texel, same convention MeshBake.run writes (row = v*S)
        let x = max(0, min(S - 1, Int(u * Float(S))))
        let y = max(0, min(S - 1, Int(w * Float(S))))
        let p = (y * S + x) * 4
        bakedCol.append(Float(baked.texRGBA[p]) / 255)
        bakedCol.append(Float(baked.texRGBA[p+1]) / 255)
        bakedCol.append(Float(baked.texRGBA[p+2]) / 255)
    }

    // --- ground truth at the same points ---
    let ptsArr = MLXArray(pts, [sampleCount, 3])
    let onShell = shellBVH.closestPoints(mesh: shell, queries: ptsArr).points
    let query = ((onShell + 0.5) * fineRes).reshaped([1, sampleCount, 3])
    let truth = GridSample3d.sample(feats: baseColor, coords: coords, grid: query, mode: "trilinear")[0]
    MLX.eval(truth)
    let truthArr = MLX.clip(truth, min: 0, max: 1).asArray(Float.self)

    var errs = [Double](repeating: 0, count: sampleCount)
    for i in 0..<sampleCount {
        let dr = Double(bakedCol[i*3] - truthArr[i*3])
        let dg = Double(bakedCol[i*3+1] - truthArr[i*3+1])
        let db = Double(bakedCol[i*3+2] - truthArr[i*3+2])
        errs[i] = (dr*dr + dg*dg + db*db).squareRoot()
    }

    // Diagnosis: bucket bad-rate by the sampled face's UV footprint in texels
    // (separates texel starvation from seam/fold/other causes).
    var bucketBad = [0, 0, 0], bucketAll = [0, 0, 0]
    for i in 0..<sampleCount {
        let fi = sampleFace[i]
        let i0 = Int(f[fi*3]), i1 = Int(f[fi*3+1]), i2 = Int(f[fi*3+2])
        let e1u = uvArr[i1*2]-uvArr[i0*2], e1v = uvArr[i1*2+1]-uvArr[i0*2+1]
        let e2u = uvArr[i2*2]-uvArr[i0*2], e2v = uvArr[i2*2+1]-uvArr[i0*2+1]
        let texArea = 0.5 * abs(Double(e1u*e2v - e1v*e2u)) * Double(S) * Double(S)
        let b = texArea < 2 ? 0 : (texArea < 8 ? 1 : 2)
        bucketAll[b] += 1
        if errs[i] > 0.1 { bucketBad[b] += 1 }
    }
    // atlas-spanning faces: UV bbox extent > 10% of atlas — should be ~zero
    var spanning = 0
    for fi in 0..<F {
        let i0 = Int(f[fi*3]), i1 = Int(f[fi*3+1]), i2 = Int(f[fi*3+2])
        let us = [uvArr[i0*2], uvArr[i1*2], uvArr[i2*2]]
        let vs = [uvArr[i0*2+1], uvArr[i1*2+1], uvArr[i2*2+1]]
        if (us.max()! - us.min()!) > 0.1 || (vs.max()! - vs.min()!) > 0.1 { spanning += 1 }
    }
    print("    atlas-spanning faces (UV extent > 0.1): \(spanning)")
    // fleck proxy: faces with real 3D area but sub-texel UV footprint render
    // as solid single-texel triangles (user-visible dark shards)
    var fleckCount = 0
    var fleckArea = 0.0, total3DArea = 0.0
    for fi in 0..<F {
        let i0 = Int(f[fi*3]), i1 = Int(f[fi*3+1]), i2 = Int(f[fi*3+2])
        let e1 = SIMD3<Float>(v[i1*3]-v[i0*3], v[i1*3+1]-v[i0*3+1], v[i1*3+2]-v[i0*3+2])
        let e2 = SIMD3<Float>(v[i2*3]-v[i0*3], v[i2*3+1]-v[i0*3+1], v[i2*3+2]-v[i0*3+2])
        let cr = SIMD3<Float>(e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x)
        let a3d = Double(simd_length(cr)) / 2
        total3DArea += a3d
        let us = [uvArr[i0*2], uvArr[i1*2], uvArr[i2*2]]
        let vs = [uvArr[i0*2+1], uvArr[i1*2+1], uvArr[i2*2+1]]
        let ext = max(us.max()! - us.min()!, vs.max()! - vs.min()!) * Float(S)
        if ext < 1.0 && a3d > 0 { fleckCount += 1; fleckArea += a3d }
    }
    print(String(format: "    sub-texel faces w/ real 3D area (flecks): %d, %.4f%% of surface",
                 fleckCount, fleckArea / max(total3DArea, 1e-12) * 100))
    for (b, name) in [(0, "<2tx"), (1, "2-8tx"), (2, ">8tx")] {
        let share = Double(bucketAll[b]) / Double(sampleCount) * 100
        let bad = bucketAll[b] > 0 ? Double(bucketBad[b]) / Double(bucketAll[b]) * 100 : 0
        print(String(format: "    faceArea %@: %5.1f%% of samples, bad %5.1f%%", name as NSString, share, bad))
    }
    errs.sort()
    let mean = errs.reduce(0, +) / Double(sampleCount)
    let p95 = errs[Int(Double(sampleCount) * 0.95)]
    let bad = Double(errs.lazy.filter { $0 > 0.1 }.count) / Double(sampleCount)

    writePNG(baked.texRGBA, S, S, to: "\(outDir)/bakeab_\(backend.rawValue)_atlas.png")
    let glbURL = URL(fileURLWithPath: "\(outDir)/bakeab_\(backend.rawValue).glb")
    try GLTFExport.writeGLB(to: glbURL, positions: baked.vertices, indices: baked.faces,
                            normals: baked.normals, uvs: baked.uvs,
                            baseColorRGBA: (baked.texRGBA, S, S))
    print("  wrote \(glbURL.lastPathComponent)")
    return EvalResult(bakeSeconds: bakeSeconds, meanErr: mean, p95Err: p95,
                      badFrac: bad, coverage: baked.coverage, faces: F)
}

let rx = onlyBackend == "provenance" ? nil : try evaluate(.xatlas)
let rp = onlyBackend == "xatlas" ? nil : try evaluate(.provenance)
func row(_ n: String, _ r: EvalResult) {
    print(String(format: "%12@  bake %7.1fs  faces %8d  coverage %.3f  meanErr %.4f  p95 %.4f  bad>0.1 %.3f%%",
                 n as NSString, r.bakeSeconds, r.faces, r.coverage, r.meanErr, r.p95Err, r.badFrac * 100))
}
print("--- RESULTS (color L2 vs voxel ground truth, \(sampleCount) surface samples)")
if let rx { row("xatlas", rx) }
if let rp { row("provenance", rp) }
print("BAKEAB DONE")
