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
let texMR = MLX.clip(texFeats[0..., 3..<5] * 0.5 + 0.5, min: 0, max: 1)   // metallic, roughness
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
    var baked: BakedMesh
    var badPoints: [Float] = []     // up to 2000 bad-sample xyz
    var badColors: [Float] = []     // this backend's baked color at those samples
}

func evaluate(_ backend: UnwrapBackend) throws -> EvalResult {
    print("=== backend \(backend.rawValue)")
    let t0 = CFAbsoluteTimeGetCurrent()
    let baked = try MeshBake.run(
        shapeFeats: shapeFeats, coords: coords, texBaseColor: baseColor,
        fineRes: fineRes, remeshRes: remeshRes, targetFaces: targetFaces, atlasSize: atlasSize,
        backend: backend, texMetallicRoughness: texMR)
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
    var badPoints: [Float] = []
    var badColors: [Float] = []
    var badSampleIdx: [Int] = []
    for i in 0..<sampleCount where errs[i] > 0.1 && badPoints.count < 6000 {
        badPoints.append(pts[i*3]); badPoints.append(pts[i*3+1]); badPoints.append(pts[i*3+2])
        badColors.append(bakedCol[i*3]); badColors.append(bakedCol[i*3+1]); badColors.append(bakedCol[i*3+2])
        badSampleIdx.append(i)
    }

    // Texel-center diagnosis for bad samples: is the texel color correct for
    // the texel center's OWN surface position (footprint/aliasing problem —
    // supersampling territory), or wrong even there (remap defect)?
    if !badSampleIdx.isEmpty {
        var centerPts: [Float] = []
        for i in badSampleIdx {
            let fi = sampleFace[i]
            let i0 = Int(f[fi*3]), i1 = Int(f[fi*3+1]), i2 = Int(f[fi*3+2])
            // the texel this sample read
            let su = pts[0]; _ = su
            // recompute sample's uv (same as read path)
            // solve bary of the texel CENTER in the face's UV triangle
            let u0 = uvArr[i0*2], w0 = uvArr[i0*2+1]
            let u1 = uvArr[i1*2], w1 = uvArr[i1*2+1]
            let u2 = uvArr[i2*2], w2 = uvArr[i2*2+1]
            // sample's own uv:
            // (recompute via stored face + the point's world pos is complex; use
            // texel center = floor(sample uv * S) + 0.5)
            // sample uv from its bary was not stored; reconstruct from the point
            // by solving world-space bary:
            let a3 = SIMD3<Float>(v[i0*3], v[i0*3+1], v[i0*3+2])
            let b3 = SIMD3<Float>(v[i1*3], v[i1*3+1], v[i1*3+2])
            let c3 = SIMD3<Float>(v[i2*3], v[i2*3+1], v[i2*3+2])
            let pw = SIMD3<Float>(pts[i*3], pts[i*3+1], pts[i*3+2])
            let e0 = b3 - a3, e1v = c3 - a3, e2v = pw - a3
            let d00 = simd_dot(e0, e0), d01 = simd_dot(e0, e1v), d11 = simd_dot(e1v, e1v)
            let d20 = simd_dot(e2v, e0), d21 = simd_dot(e2v, e1v)
            let den = d00 * d11 - d01 * d01
            var bw: Float = 0, cw: Float = 0
            if abs(den) > 1e-20 { bw = (d11 * d20 - d01 * d21) / den; cw = (d00 * d21 - d01 * d20) / den }
            let aw = 1 - bw - cw
            let su2 = aw * u0 + bw * u1 + cw * u2
            let sw2 = aw * w0 + bw * w1 + cw * w2
            let tcu = (Float(Int(su2 * Float(S))) + 0.5) / Float(S)
            let tcw = (Float(Int(sw2 * Float(S))) + 0.5) / Float(S)
            // bary of texel center in UV triangle
            let q0 = SIMD2<Float>(u1 - u0, w1 - w0)
            let q1 = SIMD2<Float>(u2 - u0, w2 - w0)
            let q2 = SIMD2<Float>(tcu - u0, tcw - w0)
            let e00 = simd_dot(q0, q0), e01 = simd_dot(q0, q1), e11 = simd_dot(q1, q1)
            let e20 = simd_dot(q2, q0), e21 = simd_dot(q2, q1)
            let den2 = e00 * e11 - e01 * e01
            var b2: Float = 0, c2: Float = 0
            if abs(den2) > 1e-20 { b2 = (e11 * e20 - e01 * e21) / den2; c2 = (e00 * e21 - e01 * e20) / den2 }
            let a2 = 1 - b2 - c2
            // texel center's surface position via that bary (clamped to face)
            let cb = max(0, min(1, b2)), cc = max(0, min(1, c2)), ca = max(0, min(1, a2))
            let norm = max(ca + cb + cc, 1e-9)
            let cpos = (a3 * ca + b3 * cb + c3 * cc) / norm
            centerPts.append(cpos.x); centerPts.append(cpos.y); centerPts.append(cpos.z)
        }
        let n = badSampleIdx.count
        let cArr = MLXArray(centerPts, [n, 3])
        let cOnShell = shellBVH.closestPoints(mesh: shell, queries: cArr).points
        let cQuery = ((cOnShell + 0.5) * fineRes).reshaped([1, n, 3])
        let cTruth = GridSample3d.sample(feats: baseColor, coords: coords, grid: cQuery, mode: "trilinear")[0]
        MLX.eval(cTruth)
        let cT = MLX.clip(cTruth, min: 0, max: 1).asArray(Float.self)
        var texelCorrect = 0
        for j in 0..<n {
            let dr = Double(badColors[j*3] - cT[j*3]), dg = Double(badColors[j*3+1] - cT[j*3+1]), db = Double(badColors[j*3+2] - cT[j*3+2])
            if (dr*dr + dg*dg + db*db).squareRoot() < 0.08 { texelCorrect += 1 }
        }
        print(String(format: "    TEXEL-CENTER DIAG: %d bad samples | texel correct for its own center: %.1f%% (footprint/aliasing) | wrong even at center: %.1f%% (remap/sampling defect)",
                     n, Double(texelCorrect) / Double(n) * 100, 100 - Double(texelCorrect) / Double(n) * 100))
    }
    errs.sort()
    let mean = errs.reduce(0, +) / Double(sampleCount)
    let p95 = errs[Int(Double(sampleCount) * 0.95)]
    let bad = Double(errs.lazy.filter { $0 > 0.1 }.count) / Double(sampleCount)

    writePNG(baked.texRGBA, S, S, to: "\(outDir)/bakeab_\(backend.rawValue)_atlas.png")
    // Diagnostic GLB: gray = rasterized ok, RED = inpainted, BLUE = wall-flipped remap
    var diag = [UInt8](repeating: 0, count: S * S * 4)
    for p in 0..<(S * S) {
        let (r, g, b): (UInt8, UInt8, UInt8) =
            baked.wallFlipped[p] ? (40, 80, 255) :
            (baked.filledBeforeInpaint[p] ? (170, 170, 170) : (255, 40, 40))
        diag[p*4] = r; diag[p*4+1] = g; diag[p*4+2] = b; diag[p*4+3] = 255
    }
    let inpFrac = Double(baked.filledBeforeInpaint.lazy.filter { !$0 }.count) / Double(S * S)
    let flipFrac = Double(baked.wallFlipped.lazy.filter { $0 }.count) / Double(S * S)
    print(String(format: "    texel classes: inpainted %.1f%% of atlas, wall-flipped %.2f%%", inpFrac * 100, flipFrac * 100))
    try GLTFExport.writeGLB(
        to: URL(fileURLWithPath: "\(outDir)/bakeab_\(backend.rawValue)_diag.glb"),
        positions: baked.vertices, indices: baked.faces,
        normals: baked.normals, uvs: baked.uvs, baseColorRGBA: (diag, S, S))
    let glbURL = URL(fileURLWithPath: "\(outDir)/bakeab_\(backend.rawValue).glb")
    try GLTFExport.writeGLB(to: glbURL, positions: baked.vertices, indices: baked.faces,
                            normals: baked.normals, uvs: baked.uvs,
                            baseColorRGBA: (baked.texRGBA, S, S),
                            metallicRoughnessRGBA: baked.mrRGBA.map { ($0, S, S) })
    print("  wrote \(glbURL.lastPathComponent)")
    return EvalResult(bakeSeconds: bakeSeconds, meanErr: mean, p95Err: p95,
                      badFrac: bad, coverage: baked.coverage, faces: F,
                      baked: baked, badPoints: badPoints, badColors: badColors)
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

// Cross-backend probe: read the xatlas bake at provenance's bad-sample 3D
// points. If xatlas is also dark there, the surface color is legitimately
// dark; if bright, provenance's sampling diverges per-point.
if let rx, let rp, !rp.badPoints.isEmpty {
    let n = rp.badPoints.count / 3
    let xm = Mesh(vertices: rx.baked.vertices, faces: rx.baked.faces)
    let cp2 = xm.bvh().closestPoints(mesh: xm, queries: MLXArray(rp.badPoints, [n, 3]))
    let cpP = cp2.points.asArray(Float.self)
    let cpF = cp2.faceIndices.asArray(Int32.self)
    let xv = rx.baked.vertices.asArray(Float.self)
    let xf = rx.baked.faces.asArray(Int32.self)
    let xuv = rx.baked.uvs.asArray(Float.self)
    let XS = rx.baked.atlasSize
    var agreeDark = 0, xBright = 0
    var meanDelta = 0.0
    for i in 0..<n {
        let fi = Int(cpF[i])
        let i0 = Int(xf[fi*3]), i1 = Int(xf[fi*3+1]), i2 = Int(xf[fi*3+2])
        // barycentric of cp point in the triangle
        let a = SIMD3<Float>(xv[i0*3], xv[i0*3+1], xv[i0*3+2])
        let b = SIMD3<Float>(xv[i1*3], xv[i1*3+1], xv[i1*3+2])
        let c = SIMD3<Float>(xv[i2*3], xv[i2*3+1], xv[i2*3+2])
        let pnt = SIMD3<Float>(cpP[i*3], cpP[i*3+1], cpP[i*3+2])
        let v0 = b - a, v1 = c - a, v2 = pnt - a
        let d00 = simd_dot(v0, v0), d01 = simd_dot(v0, v1), d11 = simd_dot(v1, v1)
        let d20 = simd_dot(v2, v0), d21 = simd_dot(v2, v1)
        let den = d00 * d11 - d01 * d01
        var bw: Float = 0, cw: Float = 0
        if abs(den) > 1e-20 { bw = (d11 * d20 - d01 * d21) / den; cw = (d00 * d21 - d01 * d20) / den }
        let aw = 1 - bw - cw
        let u = aw * xuv[i0*2] + bw * xuv[i1*2] + cw * xuv[i2*2]
        let w = aw * xuv[i0*2+1] + bw * xuv[i1*2+1] + cw * xuv[i2*2+1]
        let x = max(0, min(XS - 1, Int(u * Float(XS))))
        let y = max(0, min(XS - 1, Int(w * Float(XS))))
        let p = (y * XS + x) * 4
        let xr = Float(rx.baked.texRGBA[p]) / 255
        let xg = Float(rx.baked.texRGBA[p+1]) / 255
        let xb = Float(rx.baked.texRGBA[p+2]) / 255
        let pr = rp.badColors[i*3], pg = rp.badColors[i*3+1], pb = rp.badColors[i*3+2]
        let xMean = (xr + xg + xb) / 3, pMean = (pr + pg + pb) / 3
        if pMean < 0.25 && xMean < 0.25 { agreeDark += 1 }
        if pMean < 0.25 && xMean > 0.4 { xBright += 1 }
        meanDelta += Double(abs(xr - pr) + abs(xg - pg) + abs(xb - pb)) / 3
    }
    print(String(format: "CROSS-PROBE: %d prov-bad points | xatlas-also-dark %.1f%% | xatlas-bright-while-prov-dark %.1f%% | mean |dCol| %.3f",
                 n, Double(agreeDark) / Double(n) * 100, Double(xBright) / Double(n) * 100, meanDelta / Double(n)))
}
print("BAKEAB DONE")
