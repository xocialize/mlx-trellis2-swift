import Foundation
import MLX
import TRELLIS2

// SW8 TEXTURING GATE — the full shape-conditioned texturing chain on-device, staged
// against Python-oracle fixtures (same garment piece, same DINOv3 cond, SAME sampling
// noise → every stage is deterministic and cosine-comparable):
//   mesh -> DualGridConvert -> ShapeSlatEncoder -> normalize -> tex DiT (concat_cond)
//   -> denorm -> tex decode -> bakeWithUVs (original UVs)

let goldens = "/Volumes/Satechi/TrellisRedux/trellis2-port/goldens"
let ckpts = "/Volumes/Satechi/TrellisRedux/models/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts"
func golden(_ name: String) throws -> MLXArray {
    try loadArray(url: URL(fileURLWithPath: "\(goldens)/texfix_\(name).npy"))
}
func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let af = a.reshaped([-1]).asType(.float32), bf = b.reshaped([-1]).asType(.float32)
    return ((af * bf).sum() / (MLX.sqrt((af * af).sum()) * MLX.sqrt((bf * bf).sum()))).item(Float.self)
}
/// Row permutation taking OUR (coords, rows) into REF coord order. Coords must be equal as sets.
func permutation(ours: [Int32], ref: [Int32], n: Int) -> [Int]? {
    var index = [Int64: Int](minimumCapacity: n)
    for i in 0..<n {
        index[Int64(ours[i*3]) << 40 | Int64(ours[i*3+1]) << 20 | Int64(ours[i*3+2])] = i
    }
    var perm = [Int](repeating: 0, count: n)
    for i in 0..<n {
        guard let j = index[Int64(ref[i*3]) << 40 | Int64(ref[i*3+1]) << 20 | Int64(ref[i*3+2])] else { return nil }
        perm[i] = j
    }
    return perm
}

let RES = 512
var allPass = true
func gate(_ name: String, _ ok: Bool, _ detail: String) {
    allPass = allPass && ok
    print("SW8 \(name): \(detail)  \(ok ? "PASS" : "FAIL")")
}

// ---- 1) converter + encoder: mesh -> shape SLat (deterministic) ----
setvbuf(stdout, nil, _IONBF, 0)   // crash-safe prints (fatalError loses buffered stdout)
let verts: [Float] = (try golden("in_vertices")).asType(.float32).reshaped([-1]).asArray(Float.self)
let faces: [Int32] = (try golden("in_faces")).asType(.int32).reshaped([-1]).asArray(Int32.self)
var t0 = Date()
let conv = try DualGridConvert.meshToFlexibleDualGrid(vertices: verts, faces: faces, gridSize: RES)
let n = conv.coords.shape[0]
// align converter rows to the oracle's coord order so the noise fixture stays row-aligned
let refShapeCoords4 = (try golden("shape_coords")).asType(.int32)
let ourC: [Int32] = conv.coords.reshaped([-1]).asArray(Int32.self)
// oracle input-coord order fixture is not saved; alignment happens at the LATENT level below.
let enc = DualGridConvert.encoderInputs(conv, resolution: RES)
MLX.eval(enc.coords4, enc.vfeats, enc.ifeats)
let encoder = ShapeSlatEncoder(weights: try loadArrays(url: URL(fileURLWithPath: "\(ckpts)/shape_enc_next_dc_f16c32_fp16.safetensors")))
let (slat, encMasks) = encoder.encodeCapturingMasks(
    SparseTensor(feats: enc.vfeats, coords: enc.coords4),
    SparseTensor(feats: enc.ifeats.asType(.float32), coords: enc.coords4))
MLX.eval(slat.feats, slat.coords)
let tEnc = -t0.timeIntervalSinceNow

// align OUR latent rows to the oracle latent coord order
let refLatC3: [Int32] = refShapeCoords4[0..., 1..<4].reshaped([-1]).asArray(Int32.self)
let ourLatC3: [Int32] = slat.coords[0..., 1..<4].reshaped([-1]).asArray(Int32.self)
let nLat = slat.coords.dim(0)
guard nLat == refShapeCoords4.dim(0), let permLat = permutation(ours: ourLatC3, ref: refLatC3, n: nLat) else {
    gate("ENC", false, "latent coord set mismatch (ours \(nLat) vs oracle \(refShapeCoords4.dim(0)))")
    print("SW8 TEXTURING GATE FAIL"); exit(1)
}
let slatAligned = slat.feats.take(MLXArray(permLat.map { Int32($0) }), axis: 0)
let encCos = cosine(slatAligned, try golden("shape_feats"))
gate("ENC  mesh->shape SLat", encCos >= 0.999,
     String(format: "N=%d latents=%d cosine=%.7f (%.1fs)", n, nLat, encCos, tEnc))

// ---- 2) tex SLat sampling (oracle noise, oracle cond; concat_cond = normalized shape) ----
let cond = try golden("cond"), neg = try golden("neg")
let sMean = try golden("shape_mean"), sStd = try golden("shape_std")
let tMean = try golden("tex_mean"), tStd = try golden("tex_std")
let shapeNorm = (slatAligned - sMean) / sStd
let latCoords4 = refShapeCoords4   // oracle order from here on
let texDit = SLatFlowModel(weights: try loadArrays(url: URL(fileURLWithPath: "\(ckpts)/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors")))
t0 = Date()
let texSlatNorm = FlowEulerSampler.sampleSLat(
    model: texDit, noiseFeats: try golden("noise"), coords: latCoords4,
    cond: cond, negCond: neg, concatCond: shapeNorm,
    guidanceStrength: 1.0, guidanceRescale: 0.0, guidanceInterval: (0.6, 0.9), rescaleT: 3.0)
let texSlat = texSlatNorm * tStd + tMean
MLX.eval(texSlat)
let texCos = cosine(texSlat, try golden("tex_slat"))
gate("TEX  SLat sample", texCos >= 0.99,
     String(format: "cosine=%.6f (%.1fs)", texCos, -t0.timeIntervalSinceNow))

// ---- 3) tex decode, guided by the ENCODER's subdivision record ----
// (tex_dec has pred_subdiv=false — upstream reproduces the encoder's voxel structure via
// the SparseTensor spatial cache; here that record is encMasks, coarsest LAST. The
// decoder's C2S child order differs from the encoder's lex parent order, so re-align
// each level's mask rows to the decoder's coords via a coord-keyed lookup, expanding
// levels with the decoder's own C2STopology.)
func key4(_ c: [Int32], _ i: Int) -> Int64 {
    Int64(c[i*4]) << 60 | Int64(c[i*4+1]) << 40 | Int64(c[i*4+2]) << 20 | Int64(c[i*4+3])
}
var guided: [MLXArray] = []
var curCoords = latCoords4
for k in 0..<encMasks.count {
    let (pc, mask) = encMasks[encMasks.count - 1 - k]   // decoder up-level k <- encoder level (last-k)
    let pcA: [Int32] = pc.reshaped([-1]).asArray(Int32.self)
    let maskA: [Int32] = mask.reshaped([-1]).asArray(Int32.self)
    let nP = pc.dim(0)
    var lookup = [Int64: Int](minimumCapacity: nP)
    for i in 0..<nP { lookup[key4(pcA, i)] = i }
    let curA: [Int32] = curCoords.asType(.int32).reshaped([-1]).asArray(Int32.self)
    let nCur = curCoords.dim(0)
    var ordered = [Int32](repeating: 0, count: nCur * 8)
    for i in 0..<nCur {
        guard let j = lookup[key4(curA, i)] else {
            print("SW8 DEC: coord missing in encoder mask at level \(k)"); exit(1)
        }
        for s in 0..<8 { ordered[i*8+s] = maskA[j*8+s] }
    }
    let orderedMask = MLXArray(ordered).reshaped([nCur, 8])
    guided.append(orderedMask)
    curCoords = C2STopology(coords: curCoords, subdivLogits: orderedMask.asType(.float32)).newCoords
}
let texDec = ShapeSlatDecoder(weights: try loadArrays(url: URL(fileURLWithPath: "\(ckpts)/tex_dec_next_dc_f16c32_fp16.safetensors")))
t0 = Date()
let pbrOut = texDec(SparseTensor(feats: texSlat, coords: latCoords4), guidedMasks: guided)
let pbr = pbrOut.feats * 0.5 + 0.5
MLX.eval(pbr, pbrOut.coords)
let refPbrC = (try golden("pbr_coords")).asType(.int32)
let ourPbrC3: [Int32] = pbrOut.coords[0..., 1..<4].reshaped([-1]).asArray(Int32.self)
let refPbrC3: [Int32] = refPbrC[0..., 1..<4].reshaped([-1]).asArray(Int32.self)
let nPbr = pbrOut.coords.dim(0)
var decCos: Float = 0
var pbrAligned = pbr
if nPbr == refPbrC.dim(0), let permPbr = permutation(ours: ourPbrC3, ref: refPbrC3, n: nPbr) {
    pbrAligned = pbr.take(MLXArray(permPbr.map { Int32($0) }), axis: 0)
    decCos = cosine(pbrAligned, try golden("pbr_feats"))
}
gate("DEC  tex decode", decCos >= 0.99,
     String(format: "pbr voxels=%d (oracle %d) cosine=%.6f (%.1fs)", nPbr, refPbrC.dim(0), decCos, -t0.timeIntervalSinceNow))

// ---- 4) bake onto ORIGINAL UVs ----
let uvs = (try golden("in_uvs")).asType(.float32)
let vertsA = (try golden("in_vertices")).asType(.float32)
let facesA = (try golden("in_faces")).asType(.int32)
t0 = Date()
let baked = MeshBake.bakeWithUVs(
    vertices: vertsA, faces: facesA, uvs: uvs,
    texFeats: pbr, texCoords: pbrOut.coords, fineRes: Float(RES), atlasSize: 1024)
// compare covered texels against the oracle atlas (both row orientations; report best)
let refAtlas: [Float] = (try golden("atlas")).asType(.float32).reshaped([-1]).asArray(Float.self)
let S = 1024
func coveredCosine(flip: Bool) -> Float {
    var dot: Double = 0, na: Double = 0, nb: Double = 0
    for y in 0..<S {
        let ry = flip ? (S - 1 - y) : y
        for x in 0..<S where baked.filled[y * S + x] {
            for ch in 0..<3 {
                let a = Double(baked.rgba[(y * S + x) * 4 + ch]) / 255.0
                let b = Double(refAtlas[(ry * S + x) * 3 + ch])
                dot += a * b; na += a * a; nb += b * b
            }
        }
    }
    return na > 0 && nb > 0 ? Float(dot / (na.squareRoot() * nb.squareRoot())) : 0
}
let cosSame = coveredCosine(flip: false), cosFlip = coveredCosine(flip: true)
let bakeCos = max(cosSame, cosFlip)
gate("BAKE original-UV atlas", bakeCos >= 0.98,
     String(format: "coverage=%.3f cosine=%.5f (orient %@, %.1fs)", baked.coverage, bakeCos,
            cosSame >= cosFlip ? "same" : "flipped", -t0.timeIntervalSinceNow))

print("SW8 TEXTURING GATE \(allPass ? "PASS" : "FAIL")")
exit(allPass ? 0 : 1)
