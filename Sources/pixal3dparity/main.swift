import Foundation
import MLX
import MLXRandom
import TRELLIS2

// PIXAL3D PARITY GATE (AB-T-0485 P2) — the Swift Pixal3D path staged against the CUDA goldens of
// AB-R-0865 (HF Job 6af8e2ae23985fb9cd82a3b0; upstream f7cf384, weights b0cb2e1b, NAF 37f2dfc,
// natten 0.21.6). Goldens are the .npz dump mirrored to .safetensors by Tools/pixal3d/convert_goldens.py.
// Thresholds follow this repo's parity targets (parity / cascadeparity / dinoparity):
//   single op (projection, NAF, proj_linear block I/O)           cosine ≥ 0.999
//   DINOv3 tokens, full DiT forward (velocities), samplers        cosine ≥ 0.99
//   coords (SS decode, HR requantize)                             exact, or ≤ 0.2 % symmetric diff
//   mesh: decoded from the golden latents                         F-score@0.5 voxel ≥ 0.99
//   mesh: from the full Swift chain (golden noise)                F-score@1 voxel ≥ 0.90, Chamfer ≤ 1 voxel
// The oracle ran its flows in bf16 (config dtype); the Swift flows run fp32 (parity-safe default).
//
// Env: PIXAL3D_RUN (golden run name), PIXAL3D_GOLDENS, PIXAL3D_WEIGHTS (converted pixal3d_*.safetensors +
// naf.safetensors), TRELLIS2_WEIGHTS (consolidated xocialize/trellis2-mlx snapshot),
// PIXAL3D_SECTIONS (comma list; default all for the run's mode), PIXAL3D_MV=1 for the multi-view run.
// Run: swift run -c release pixal3dparity

setvbuf(stdout, nil, _IONBF, 0)
let env = ProcessInfo.processInfo.environment
let run = env["PIXAL3D_RUN"] ?? "golden_C2_fov035_dress_1024"
let goldRoot = env["PIXAL3D_GOLDENS"] ?? "/Volumes/DEV_VOL1/Evidence/pixal3d-oracle-2026-11-14-st"
let pixWeights = env["PIXAL3D_WEIGHTS"] ?? "/Volumes/DEV_VOL1/Models/pixal3d-mlx"
let trellisWeights = env["TRELLIS2_WEIGHTS"] ?? "/Volumes/DEV_VOL1/Models/models--xocialize--trellis2-mlx"
let isMV = env["PIXAL3D_MV"] == "1" || run.contains("_mv")
let allSections = ["camera", "dino", "proj", "naf", "cond", "blocks", "vel", "sample", "hr", "mesh", "e2e"]
let sections = Set((env["PIXAL3D_SECTIONS"] ?? allSections.joined(separator: ",")).split(separator: ",").map(String.init))
let gdir = "\(goldRoot)/\(run)/golden"

func G(_ f: String) -> [String: MLXArray] {
    do { return try loadArrays(url: URL(fileURLWithPath: "\(gdir)/\(f).safetensors")) }
    catch { fatalError("golden \(f): \(error)") }
}
func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let af = a.reshaped([-1]).asType(.float32), bf = b.reshaped([-1]).asType(.float32)
    return ((af * bf).sum() / (MLX.sqrt((af * af).sum()) * MLX.sqrt((bf * bf).sum()))).item(Float.self)
}
func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a.asType(.float32).reshaped([-1]) - b.asType(.float32).reshaped([-1])).max().item(Float.self)
}
func scalar(_ a: MLXArray) -> Double { Double(a.asType(.float32).reshaped([-1])[0].item(Float.self)) }
func packedKeys(_ coords: MLXArray) -> [Int64] {
    let n = coords.dim(0), c = coords.asType(.int32).asArray(Int32.self)
    return (0..<n).map { i in Int64(c[i*4]) << 48 | Int64(c[i*4+1]) << 32 | Int64(c[i*4+2]) << 16 | Int64(c[i*4+3]) }
}

var allPass = true
var results: [[String: Any]] = []
func gate(_ name: String, _ ok: Bool, _ detail: String, _ value: Double? = nil) {
    allPass = allPass && ok
    print("PX3D \(name): \(detail)  \(ok ? "PASS" : "FAIL")")
    var r: [String: Any] = ["gate": name, "pass": ok, "detail": detail]
    if let value { r["value"] = value }
    results.append(r)
}
func cosGate(_ name: String, _ ours: MLXArray, _ gold: MLXArray, _ thr: Float, extra: String = "") {
    let c = cosine(ours, gold)
    gate(name, c >= thr, String(format: "cosine=%.6f maxAbs=%.4g (≥ %.3f)%@", c, maxAbs(ours, gold), thr, extra), Double(c))
}

print("=== Pixal3D parity: \(run) (\(isMV ? "multi-view" : "single-view")) ===")
let t0All = Date()

// ---- weights ----
let pipe = try Trellis2Pipeline(consolidatedDir: trellisWeights, loadStockStructFlow: false)
pipe.pixal3dDir = pixWeights
let normObj = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: "\(trellisWeights)/normalization.json"))) as! [String: Any]
func normVec(_ k1: String, _ k2: String) -> MLXArray {
    MLXArray(((normObj[k1] as! [String: Any])[k2] as! [Double]).map { Float($0) })
}
let sMean = normVec("shape_slat_normalization", "mean"), sStd = normVec("shape_slat_normalization", "std")
let tMean = normVec("tex_slat_normalization", "mean"), tStd = normVec("tex_slat_normalization", "std")
let cond = try pipe.pixal3dConditioner()
let ssPhases = SSGrid.ropePhases(resolution: 16)

// ---- golden inputs → views ----
let inp = G("input")
let stageNames: [Pixal3DStage: String] = [.ss: "ss", .shape512: "shape512", .shape1024: "shape1024", .tex1024: "tex1024"]
let dino512 = G("shape512.dino"), dino1024 = G("shape1024.dino")
let naf512 = G("shape512.naf"), naf1024 = G("shape1024.naf")
// Multi-view goldens hold all views in `<stage>.dino` (batch V); single view has batch 1.
let V = dino512["image_normalized"]!.dim(0)
var cams = [Pixal3DCamera]()
if isMV {
    let c2w = inp["transform_matrix"] ?? inp["c2w"]!
    let fov = inp["camera_angle_x"]!.asType(.float32).reshaped([-1]).asArray(Float.self)
    let m = c2w.asType(.float32).reshaped([-1, 16])
    for v in 0..<m.dim(0) {
        cams.append(Pixal3DCamera(fovX: Double(fov[min(v, fov.count - 1)]), meshScale: 1.0,
                                  c2w: Mat4(flat: m[v].asArray(Float.self).map { Double($0) })))
    }
} else {
    cams.append(Pixal3DCamera(fovX: scalar(inp["camera_angle_x"]!), meshScale: scalar(inp["mesh_scale"]!)))
}
precondition(cams.count == V, "views \(V) vs cameras \(cams.count)")
// NAF guides: single-view goldens carry the [0,1] guide; the DINOv3 input is its ImageNet normalization.
let imMean = MLXArray([0.485, 0.456, 0.406] as [Float]).reshaped([1, 3, 1, 1])
let imStd = MLXArray([0.229, 0.224, 0.225] as [Float]).reshaped([1, 3, 1, 1])
func guideFrom(_ normalized: MLXArray) -> MLXArray { normalized * imStd + imMean }
let views: [Pixal3DView] = (0..<V).map { v in
    let n512 = dino512["image_normalized"]![v ..< (v + 1)], n1024 = dino1024["image_normalized"]![v ..< (v + 1)]
    return Pixal3DView(normalized512: n512, guide512: guideFrom(n512), normalized1024: n1024, guide1024: guideFrom(n1024),
                       camera: cams[v])
}
let calc = Pixal3DCamera.calcMats(cams)

// ===== camera =====
if sections.contains("camera") {
    let gc = G("ss.cond")
    let d = scalar(gc["distance"]!)
    gate("CAMERA  distance = 0.5/(ms·tan(fov/2))", abs(cams[0].distance - d) < 1e-4,
         String(format: "ours=%.6f oracle=%.6f fov=%.4f rad", cams[0].distance, d, cams[0].fovX))
    if isMV, let gm = gc["calc_mat"] {
        let g = gm.asType(.float32).reshaped([-1, 16])
        var worst = 0.0
        for v in 0..<V { for k in 0..<16 { worst = max(worst, abs(Double(g[v, k].item(Float.self)) - calc[v].m[k])) } }
        gate("CAMERA  calc_mat = F·inv(C0)·Ci", worst < 1e-4, String(format: "max|Δ|=%.2e over %d views", worst, V))
    }
}

// ===== DINOv3 (our port on the golden normalized inputs) =====
if sections.contains("dino") {
    for (name, g) in [("512", dino512), ("1024", dino1024)] {
        var ours = [MLXArray]()
        for v in views { ours.append(cond.tokens(v, size: name == "512" ? 512 : 1024)) }
        let o = MLX.concatenated(ours, axis: 0)
        cosGate("DINO\(name) tokens", o, g["tokens"]!, 0.99, extra: " [\(o.dim(0))×\(o.dim(1))]")
    }
    for st in Pixal3DStage.allCases {
        let gz = G("\(stageNames[st]!).cond")["z_global"]!
        let og = views.map { cond.tokens($0, size: st.imageSize)[0..., 0..<5] }
        let o = V == 1 ? og[0] : MLX.concatenated(og, axis: 0).mean(axis: 0, keepDims: true)
        cosGate("ZGLOBAL \(stageNames[st]!)", o, gz, 0.99)
    }
}

// ===== projection (camera back-projection + sampling) on the GOLDEN DINOv3 patch map =====
func goldPatch(_ st: Pixal3DStage, view v: Int) -> MLXArray {
    Pixal3DConditioner.patchMap(G("\(stageNames[st]!).dino")["tokens"]![v ..< (v + 1)])
}
if sections.contains("proj") && !isMV {
    // SS: dense 16³, 1024 channels
    let dense16 = Pixal3DProjection.denseCoords(grid: 16)
    let n16 = Pixal3DProjection.normalizedCoords(coords: dense16, grid: 16, calcMat: calc[0], fovX: cams[0].fovX,
                                                 meshScale: cams[0].meshScale, imageSize: 512)
    cosGate("PROJ    ss dense 16³ (golden tokens)", Pixal3DProjection.sampleBilinear(goldPatch(.ss, view: 0), n16),
            G("ss.cond")["z_proj_dense"]![0], 0.999)
    // shape512: dense 32³ LR half (golden is fp16 above 20M elements)
    let dense32 = Pixal3DProjection.denseCoords(grid: 32)
    let n32 = Pixal3DProjection.normalizedCoords(coords: dense32, grid: 32, calcMat: calc[0], fovX: cams[0].fovX,
                                                 meshScale: cams[0].meshScale, imageSize: 512)
    cosGate("PROJ    shape512 dense 32³ LR half", Pixal3DProjection.sampleBilinear(goldPatch(.shape512, view: 0), n32),
            G("shape512.cond")["z_proj_dense"]![0, 0..., 0..<1024], 0.999)
    for st in [Pixal3DStage.shape512, .shape1024, .tex1024] {
        let pt = G("\(stageNames[st]!).proj_at_tokens")
        let R = st == .shape512 ? 32 : Int(scalar(G("\(stageNames[st]!).cond")["grid_resolution"]!))
        let n = Pixal3DProjection.normalizedCoords(coords: pt["coords"]!, grid: R, calcMat: calc[0], fovX: cams[0].fovX,
                                                   meshScale: cams[0].meshScale, imageSize: st.imageSize)
        cosGate("PROJ    \(stageNames[st]!) at tokens LR half (R=\(R))",
                Pixal3DProjection.sampleBilinear(goldPatch(st, view: 0), n), pt["feats"]![0..., 0..<1024], 0.999)
    }
}

// ===== NAF (golden guide + golden LR features) =====
if sections.contains("naf") {
    for st in [Pixal3DStage.shape512, .shape1024, .tex1024] {
        let g = G("\(stageNames[st]!).naf")
        let t = Date()
        let map = cond.naf(guide: g["guide"]!, features: g["lr_features"]!, target: st.nafTarget!)   // [O,O,C]
        MLX.eval(map)
        let O = map.dim(0), c0 = (O - 128) / 2
        let centre = map[c0 ..< (c0 + 128), c0 ..< (c0 + 128)].transposed(2, 0, 1)
        let border = map[0 ..< 128, 0 ..< 128].transposed(2, 0, 1)
        let dt = -t.timeIntervalSinceNow
        cosGate("NAF     \(stageNames[st]!) centre 128² crop", centre, g["hr_centre_128"]![0], 0.999,
                extra: String(format: " (O=%d, %.1fs)", O, dt))
        cosGate("NAF     \(stageNames[st]!) top-left 128² crop (edge rule)", border, g["hr_border_128"]![0], 0.999)
        let flat = map.reshaped([-1, map.dim(2)])
        let mean = flat.mean(axis: 0), sd = MLX.sqrt(((flat - mean) * (flat - mean)).mean(axis: 0))
        cosGate("NAF     \(stageNames[st]!) per-channel mean", mean, g["hr_mean"]![0], 0.999)
        cosGate("NAF     \(stageNames[st]!) per-channel std", sd, g["hr_std"]![0], 0.999)
        // full projected features (LR golden tokens + HR our NAF) vs proj_at_tokens
        if !isMV {
            let pt = G("\(stageNames[st]!).proj_at_tokens")
            let R = st == .shape512 ? 32 : Int(scalar(G("\(stageNames[st]!).cond")["grid_resolution"]!))
            let n = Pixal3DProjection.normalizedCoords(coords: pt["coords"]!, grid: R, calcMat: calc[0], fovX: cams[0].fovX,
                                                       meshScale: cams[0].meshScale, imageSize: st.imageSize)
            cosGate("PROJ    \(stageNames[st]!) at tokens HR half (our NAF)",
                    Pixal3DProjection.sampleBilinear(map, n), pt["feats"]![0..., 1024...], 0.999)
        }
        MLX.Memory.clearCache()
    }
}

// ===== end-to-end conditioning from the golden images (our DINOv3 + NAF + projection) =====
var ourCond: [Pixal3DStage: (global: MLXArray, proj: MLXArray)] = [:]
func stageCoords(_ st: Pixal3DStage) -> (coords: MLXArray?, R: Int) {
    switch st {
    case .ss: return (nil, 16)
    case .shape512: return (G("shape512.proj_at_tokens")["coords"]!, 32)
    default:
        let pt = G("\(stageNames[st]!).proj_at_tokens")
        return (pt["coords"]!, Int(scalar(G("\(stageNames[st]!).cond")["grid_resolution"]!)))
    }
}
if sections.contains("cond") {
    for st in Pixal3DStage.allCases {
        let (coords, R) = stageCoords(st)
        let t = Date()
        let c = cond.condition(views, stage: st, coords: coords, grid: R)
        ourCond[st] = c
        let gold = st == .ss ? G("ss.cond")["z_proj_dense"]![0] : G("\(stageNames[st]!).proj_at_tokens")["feats"]!
        cosGate("COND    \(stageNames[st]!) proj (our DINOv3+NAF+projection, \(V) view\(V > 1 ? "s" : ""))", c.proj, gold, 0.99,
                extra: String(format: " [%d×%d, %.1fs]", c.proj.dim(0), c.proj.dim(1), -t.timeIntervalSinceNow))
        if st != .ss {
            cosGate("COND    \(stageNames[st]!) proj LR half", c.proj[0..., 0..<1024], gold[0..., 0..<1024], 0.99)
            cosGate("COND    \(stageNames[st]!) proj HR half", c.proj[0..., 1024...], gold[0..., 1024...], 0.99)
        }
        MLX.Memory.clearCache()
    }
}

// golden conditioning per stage (isolates the flows from the conditioner)
func goldCond(_ st: Pixal3DStage) -> (global: MLXArray, proj: MLXArray) {
    let gl = G("\(stageNames[st]!).cond")["z_global"]!
    let pr = st == .ss ? G("ss.cond")["z_proj_dense"]! : G("\(stageNames[st]!).proj_at_tokens")["feats"]!
    return (gl, pr)
}
let zeros5 = MLXArray.zeros([1, 5, 1024])

// ===== flows: block I/O, velocities, samplers (one flow resident at a time) =====
func withFlow<T>(_ st: Pixal3DStage, _ body: (SparseStructureFlowModel?, SLatFlowModel?) throws -> T) throws -> T {
    switch st {
    case .ss:
        let m = try pipe.pixal3dStructFlow(multiView: isMV)
        defer { pipe.releasePixal3D(); MLX.Memory.clearCache() }
        return try body(m, nil)
    case .shape512, .shape1024, .tex1024:
        let kind: Pixal3DFlowKind = st == .shape512 ? .shape512(multiView: isMV)
            : st == .shape1024 ? .shape1024(multiView: isMV) : .tex1024(multiView: isMV)
        let m = try pipe.pixal3dFlow(kind)
        defer { pipe.releasePixal3D(); MLX.Memory.clearCache() }
        return try body(nil, m)
    }
}
let samplerParams: [Pixal3DStage: (g: Float, r: Float, iv: (Float, Float), rt: Float)] = [
    .ss: (7.5, 0.7, (0.6, 1.0), 5.0), .shape512: (7.5, 0.5, (0.6, 1.0), 3.0),
    .shape1024: (7.5, 0.5, (0.6, 1.0), 3.0), .tex1024: (1.0, 0.0, (0.6, 0.9), 3.0)]
var ourSamples: [Pixal3DStage: MLXArray] = [:]

let flowSections = sections.intersection(["blocks", "vel", "sample"])
if !flowSections.isEmpty {
    for st in Pixal3DStage.allCases {
        let sn = stageNames[st]!
        let gc = goldCond(st)
        let concat: MLXArray? = st == .tex1024 ? G("tex1024.concat_cond")["feats"]! : nil
        let coords: MLXArray? = st == .ss ? nil : G("\(sn).noise")["coords"]!
        try withFlow(st) { ss, slat in
            let blocks = ss?.blocks ?? slat!.blocks
            if flowSections.contains("blocks") {
                for i in [0, 15, 29] {
                    let g = G("\(sn).s0.projattn\(i)")
                    let w = g["proj_linear.weight"]!, b = g["proj_linear.bias"]!
                    let blk = blocks[i]
                    let wc = cosine(blk.projW!, w), bc = cosine(blk.projB!, b)
                    let xin = g["x_in"]!.reshaped([1, -1, 1536])
                    let projIn = (st == .ss ? gc.proj : gc.proj.reshaped([1, -1, gc.proj.dim(1)]))
                    let out = blk.crossTerm(xin, context: gc.global, proj: projIn)
                    cosGate("BLOCK   \(sn) proj-attn \(i) I/O", out, g["out"]!.reshaped([1, -1, 1536]), 0.999,
                            extra: String(format: " (proj_linear W cos %.6f, b cos %.6f vs oracle)", wc, bc))
                }
            }
            if flowSections.contains("vel") {
                for step in [0, 5, 11] {
                    let s = G("\(sn).s\(step)")
                    let xt = st == .ss ? s["x_t"]! : s["x_t.feats"]!
                    let t = Float(scalar(s["t"]!))
                    let t1000 = MLXArray([1000 * t])
                    func call(_ useCond: Bool) -> MLXArray {
                        if let ss {
                            return ss(xt, t: t1000, cond: useCond ? gc.global : zeros5, phases: ssPhases,
                                      proj: useCond ? gc.proj : nil)
                        }
                        return slat!(SparseTensor(feats: xt, coords: coords!), t: t1000, cond: useCond ? gc.global : zeros5,
                                     concatCond: concat, proj: useCond ? gc.proj : nil).feats
                    }
                    let v1 = call(true); MLX.eval(v1)
                    let g1 = G("\(sn).s\(step).v1"); let gv1 = st == .ss ? g1["value"]! : g1["feats"]!
                    cosGate("VEL     \(sn) step \(step) conditional", v1, gv1, 0.99, extra: String(format: " t=%.4f", t))
                    let p = samplerParams[st]!
                    let inIv = Double(t) >= Double(p.iv.0) && Double(t) <= Double(p.iv.1)
                    var vg = v1
                    if p.g != 1.0 && inIv {
                        let v2 = call(false); MLX.eval(v2)
                        let g2 = G("\(sn).s\(step).v2"); let gv2 = st == .ss ? g2["value"]! : g2["feats"]!
                        cosGate("VEL     \(sn) step \(step) unconditional", v2, gv2, 0.99)
                        vg = FlowEulerSampler.guided(x: xt, t: t, vPos: v1, vNeg: v2, strength: p.g, rescale: p.r)
                    }
                    cosGate("VEL     \(sn) step \(step) guided", vg, s["v_guided"]!, 0.99)
                }
            }
            if flowSections.contains("sample") {
                let n = G("\(sn).noise")
                let noise = st == .ss ? n["value"]! : n["feats"]!
                let p = samplerParams[st]!
                let t = Date()
                let out: MLXArray
                if let ss {
                    out = FlowEulerSampler.sampleSS(model: ss, noise: noise, cond: gc.global, negCond: zeros5, phases: ssPhases,
                                                    proj: gc.proj, guidanceStrength: p.g, guidanceRescale: p.r,
                                                    guidanceInterval: p.iv, rescaleT: p.rt)
                } else {
                    out = FlowEulerSampler.sampleSLat(model: slat!, noiseFeats: noise, coords: coords!, cond: gc.global,
                                                      negCond: zeros5, concatCond: concat, proj: gc.proj,
                                                      guidanceStrength: p.g, guidanceRescale: p.r,
                                                      guidanceInterval: p.iv, rescaleT: p.rt)
                }
                MLX.eval(out)
                ourSamples[st] = out
                let gs = G("\(sn).samples"); let gsv = st == .ss ? gs["value"]! : gs["feats"]!
                cosGate("SAMPLE  \(sn) (golden noise + golden cond)", out, gsv, 0.99,
                        extra: String(format: " (%.1fs)", -t.timeIntervalSinceNow))
                if st == .ss {
                    let ours = pipe.pixal3dSSDecode(out)
                    let oursG = pipe.pixal3dSSDecode(gsv)
                    let gold = Set(packedKeys(G("ss.coords")["value"]!))
                    let a = Set(packedKeys(ours)), b = Set(packedKeys(oursG))
                    let iou = Float(a.intersection(gold).count) / Float(a.union(gold).count)
                    gate("COORDS  ss decode of golden samples", b == gold, "tokens=\(b.count) oracle=\(gold.count) \(b == gold ? "exact" : "MISMATCH")")
                    gate("COORDS  ss decode of our samples", iou >= 0.99, String(format: "tokens=%d IoU=%.5f", a.count, iou), Double(iou))
                }
            }
        }
    }
}

// ===== HR requantize: our decoder upsample of the golden LR slat → Pixal3D requantize =====
if sections.contains("hr") {
    let lr = G("shape512.samples")
    let up = pipe.pixal3dUpsample(lrSlat: lr["feats"]!, coords: lr["coords"]!, shapeMean: sMean, shapeStd: sStd)
    let target = Int(scalar(G("latents")["resolution"]!))
    let (q, hr) = Pixal3DProjection.requantize(upCoords: up, lrResolution: 512, targetResolution: target)
    let gold = Set(packedKeys(G("shape1024.noise")["coords"]!))
    let ours = Set(packedKeys(q))
    let d = ours.symmetricDifference(gold).count
    gate("HR      requantize ((c+0.5)/512·(g−1)).round()", hr == target && Float(d) / Float(gold.count) <= 0.002,
         String(format: "tokens=%d oracle=%d symdiff=%d (%.3f%%) hr=%d", ours.count, gold.count, d,
                Float(d) * 100 / Float(gold.count), hr), Double(d))
    // the stock rule on the same coords, for the record (why the variant exists)
    let (qs, _) = CascadeQuantize.requantize(upCoords: up, lrResolution: 512, targetResolution: target)
    print("PX3D note: the stock TRELLIS.2 requantize gives \(qs.dim(0)) tokens on the same upsample (Pixal3D \(q.dim(0)))")
}

// ===== mesh geometry =====
/// Symmetric nearest-neighbour distances between point sets (voxel units at `res`) via a hash grid.
func chamfer(_ a: [Float], _ b: [Float], res: Float, sample: Int = 200_000) -> (meanAB: Float, meanBA: Float, f05: Float, f1: Float, p99: Float) {
    func grid(_ p: [Float]) -> [Int64: [Int32]] {
        var g = [Int64: [Int32]](minimumCapacity: p.count / 6)
        for i in 0..<(p.count / 3) {
            let k = key(p[i*3], p[i*3+1], p[i*3+2])
            g[k, default: []].append(Int32(i))
        }
        return g
    }
    let cell: Float = 2.0 / res
    func cellIdx(_ v: Float) -> Int64 { Int64(floor((v + 1) / cell)) }
    func key(_ x: Float, _ y: Float, _ z: Float) -> Int64 { cellIdx(x) << 42 | cellIdx(y) << 21 | cellIdx(z) }
    func nn(_ src: [Float], _ dst: [Float], _ g: [Int64: [Int32]]) -> [Float] {
        let n = src.count / 3
        let stride = max(1, n / sample)
        var out = [Float](); out.reserveCapacity(n / stride + 1)
        var i = 0
        while i < n {
            let x = src[i*3], y = src[i*3+1], z = src[i*3+2]
            let cx = cellIdx(x), cy = cellIdx(y), cz = cellIdx(z)
            var best = Float.infinity
            for r in 1...3 {
                for dx in -r...r { for dy in -r...r { for dz in -r...r {
                    if let list = g[(cx + Int64(dx)) << 42 | (cy + Int64(dy)) << 21 | (cz + Int64(dz))] {
                        for j in list {
                            let ex = dst[Int(j)*3] - x, ey = dst[Int(j)*3+1] - y, ez = dst[Int(j)*3+2] - z
                            best = min(best, ex*ex + ey*ey + ez*ez)
                        }
                    }
                } } }
                if best <= Float(r) * cell * Float(r) * cell { break }
            }
            out.append(best.squareRoot() * res)
            i += stride
        }
        return out
    }
    let ga = grid(a), gb = grid(b)
    let ab = nn(a, b, gb).map { $0.isFinite ? $0 : 3 * 2 }, ba = nn(b, a, ga).map { $0.isFinite ? $0 : 3 * 2 }
    func frac(_ d: [Float], _ tau: Float) -> Float { Float(d.filter { $0 <= tau }.count) / Float(d.count) }
    func fscore(_ tau: Float) -> Float { let p = frac(ab, tau), r = frac(ba, tau); return p + r > 0 ? 2 * p * r / (p + r) : 0 }
    let all = (ab + ba).sorted()
    return (ab.reduce(0, +) / Float(ab.count), ba.reduce(0, +) / Float(ba.count), fscore(0.5), fscore(1.0),
            all[min(all.count - 1, Int(Float(all.count) * 0.99))])
}
let goldMesh = sections.contains("mesh") || sections.contains("e2e") ? G("mesh_raw")["vertices"]!.asType(.float32).asArray(Float.self) : []

if sections.contains("mesh") {
    // decode the GOLDEN latents with our decoders → raw dual-grid mesh vs the oracle's raw mesh
    let lat = G("latents")
    let res = Int(scalar(lat["resolution"]!))
    let t = Date()
    let rawV = pipe.pixal3dRawMesh(shapeSlatDenorm: lat["shape_slat.feats"]!, coords: lat["shape_slat.coords"]!, resolution: res)
    let ch = chamfer(rawV.asType(.float32).asArray(Float.self), goldMesh, res: Float(res))
    gate("MESH    decode of golden latents vs oracle raw mesh", ch.f05 >= 0.99,
         String(format: "verts ours=%d oracle=%d chamfer ours→oracle=%.3f oracle→ours=%.3f vox, p99=%.3f, F@0.5=%.4f F@1=%.4f (%.1fs)",
                rawV.dim(0), goldMesh.count / 3, ch.meanAB, ch.meanBA, ch.p99, ch.f05, ch.f1, -t.timeIntervalSinceNow), Double(ch.f05))
}

// ===== full Swift chain with golden noise (conditioning from the golden images) =====
if sections.contains("e2e") {
    let ssN = G("ss.noise")["value"]!
    // per-token noise follows the coords: golden rows where our coords match the oracle's, seeded otherwise
    func noiseFor(_ st: String) -> (MLXArray) -> MLXArray {
        let g = G("\(st).noise")
        let keys = packedKeys(g["coords"]!)
        var idx = [Int64: Int](); for (i, k) in keys.enumerated() { idx[k] = i }
        let feats = g["feats"]!
        return { coords in
            let ks = packedKeys(coords)
            let fill = MLXRandom.normal([ks.count, 32])
            let rows = ks.enumerated().map { (i, k) -> MLXArray in idx[k].map { feats[$0 ..< ($0 + 1)] } ?? fill[i ..< (i + 1)] }
            return MLX.concatenated(rows, axis: 0)
        }
    }
    var cap: [String: [String: MLXArray]] = [:]
    let t = Date()
    let out = try pipe.generatePixal3D(
        tier: .res1024, views: views, multiViewWeights: isMV, ssPhases: ssPhases,
        shapeMean: sMean, shapeStd: sStd, texMean: tMean, texStd: tStd,
        noise: Pixal3DNoise(ss: ssN), noiseFn: ["shape512": noiseFor("shape512"), "shape1024": noiseFor("shape1024"),
                                                 "tex": noiseFor("tex1024")],
        bake: false, capture: { cap[$0] = $1 }, log: { print($0) })
    let genS = -t.timeIntervalSinceNow
    print(String(format: "PX3D e2e generate (no bake): %.1fs, hr=%d", genS, out.hrResolution))
    // coords
    let ssGold = Set(packedKeys(G("ss.coords")["value"]!)), ssOurs = Set(packedKeys(cap["ss"]!["coords"]!))
    let ssIoU = Float(ssGold.intersection(ssOurs).count) / Float(ssGold.union(ssOurs).count)
    gate("E2E     ss coords", ssIoU >= 0.99, String(format: "ours=%d oracle=%d IoU=%.5f", ssOurs.count, ssGold.count, ssIoU), Double(ssIoU))
    let hrGold = Set(packedKeys(G("shape1024.noise")["coords"]!)), hrOurs = Set(packedKeys(cap["hr"]!["coords"]!))
    let hrIoU = Float(hrGold.intersection(hrOurs).count) / Float(hrGold.union(hrOurs).count)
    gate("E2E     HR tokens", hrIoU >= 0.98, String(format: "ours=%d oracle=%d IoU=%.5f", hrOurs.count, hrGold.count, hrIoU), Double(hrIoU))
    // latents on the shared coords
    func onShared(_ ours: MLXArray, _ oursCoords: MLXArray, _ gold: MLXArray, _ goldCoords: MLXArray) -> (MLXArray, MLXArray) {
        var gi = [Int64: Int](); for (i, k) in packedKeys(goldCoords).enumerated() { gi[k] = i }
        var a = [Int32](), b = [Int32]()
        for (i, k) in packedKeys(oursCoords).enumerated() { if let j = gi[k] { a.append(Int32(i)); b.append(Int32(j)) } }
        return (ours.take(MLXArray(a), axis: 0), gold.take(MLXArray(b), axis: 0))
    }
    for (st, key) in [("shape512", "shape512"), ("shape1024", "shape1024"), ("tex1024", "tex1024")] {
        let g = G("\(st).samples")
        let (o, gg) = onShared(cap[key]!["samples"]!, cap[key]!["coords"]!, g["feats"]!, g["coords"]!)
        cosGate("E2E     \(st) latent (shared coords)", o, gg, 0.99, extra: " [\(o.dim(0)) shared]")
    }
    // mesh
    let dec = cap["shape_decoded"]!
    let rawV = pipe.pixal3dRawMeshFromDecoded(feats: dec["feats"]!, coords: dec["coords"]!, resolution: out.hrResolution)
    let ch = chamfer(rawV.asType(.float32).asArray(Float.self), goldMesh, res: Float(out.hrResolution))
    gate("E2E     mesh (full chain) vs oracle raw mesh", ch.f1 >= 0.90 && ch.meanAB <= 1 && ch.meanBA <= 1,
         String(format: "verts ours=%d oracle=%d chamfer ours→oracle=%.3f oracle→ours=%.3f vox, p99=%.3f, F@0.5=%.4f F@1=%.4f",
                rawV.dim(0), goldMesh.count / 3, ch.meanAB, ch.meanBA, ch.p99, ch.f05, ch.f1), Double(ch.f1))
}

print(String(format: "PX3D total %.1fs", -t0All.timeIntervalSinceNow))
print("PX3D GATE \(allPass ? "PASS" : "FAIL") (\(run))")
if let outPath = env["PIXAL3D_JSON"] {
    let rec: [String: Any] = ["run": run, "pass": allPass, "gates": results]
    if let d = try? JSONSerialization.data(withJSONObject: rec, options: [.prettyPrinted, .sortedKeys]) {
        try? d.write(to: URL(fileURLWithPath: outPath))
    }
}
exit(allPass ? 0 : 1)
