import Foundation
import MLX
import TRELLIS2

// T0.3 CASCADE GATE — the 1024/1536 cascade chain staged against Python-oracle fixtures
// (_goldens_cascade.py; same T.png conditioning, SAME sampling noise per stage):
//   LRSAMP   512 flow on the 32³ LR coords, cond_512                cosine ≥ 0.99
//   UPCOORD  shape decoder forward coords on the LR slat            IoU ≥ 0.998 raw
//            (fp subdivision-logit ties flip a few 512-grid subtrees — the known
//            "parity modulo fp subdivision ties"; the cascade only consumes these coords
//            through re-quantization, so the GATE is on the quantized token sets:
//            EXACT at 1024 (64³ absorbs the flips), ≤0.2% symmetric diff at 1536 (96³))
//   QUANT    host re-quantization + token back-off (1024/1536/bk)   exact coords + hr_res
//   HRSAMP   1024 flow on the re-quantized coords, cond_1024        cosine ≥ 0.99
//   TEXSAMP  tex 1024 flow, concat_cond = normalized HR shape slat  cosine ≥ 0.99
//   DINO1024 ported DINOv3 on 1024² pixels vs oracle cond_1024      cosine ≥ 0.99

let goldens = "/Volumes/Satechi/TrellisRedux/trellis2-port/goldens"
let ckpts = "/Volumes/Satechi/TrellisRedux/models/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts"
let dinoPath = "/Volumes/Satechi/TrellisRedux/models/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51/model.safetensors"
func golden(_ n: String) throws -> MLXArray { try loadArray(url: URL(fileURLWithPath: "\(goldens)/\(n).npy")) }
func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let af = a.reshaped([-1]).asType(.float32), bf = b.reshaped([-1]).asType(.float32)
    return ((af * bf).sum() / (MLX.sqrt((af * af).sum()) * MLX.sqrt((bf * bf).sum()))).item(Float.self)
}
/// Coord rows packed for order-insensitive set compare (components < 2^16).
func packedKeys(_ coords: MLXArray) -> [Int64] {
    let n = coords.dim(0)
    let c = coords.asType(.int32).asArray(Int32.self)
    var keys = [Int64](); keys.reserveCapacity(n)
    for i in 0..<n {
        keys.append(Int64(c[i*4]) << 48 | Int64(c[i*4+1]) << 32 | Int64(c[i*4+2]) << 16 | Int64(c[i*4+3]))
    }
    return keys
}

setvbuf(stdout, nil, _IONBF, 0)   // crash-safe prints (fatalError loses buffered stdout)
var allPass = true
func gate(_ name: String, _ ok: Bool, _ detail: String) {
    allPass = allPass && ok
    print("T03 \(name): \(detail)  \(ok ? "PASS" : "FAIL")")
}

let coordsLR = (try golden("casc_coords_lr")).asType(.int32)
let cond512 = try golden("cond_512"), neg512 = try golden("neg_cond_512")
let cond1024 = try golden("cond_1024")
let neg1024 = MLXArray.zeros(cond1024.shape, dtype: cond1024.dtype)
let sMean = try golden("shape_slat_mean"), sStd = try golden("shape_slat_std")
let tMean = try golden("tex_slat_mean"), tStd = try golden("tex_slat_std")

// ---- 1) LR pass: 512 shape flow on the 32³ subset (oracle noise, cond_512) ----
var t0 = Date()
let flow512 = SLatFlowModel(weights: try loadArrays(url: URL(fileURLWithPath: "\(ckpts)/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors")))
let lrNorm = FlowEulerSampler.sampleSLat(
    model: flow512, noiseFeats: try golden("casc_noise_lr"), coords: coordsLR,
    cond: cond512, negCond: neg512,
    guidanceStrength: 7.5, guidanceRescale: 0.5, guidanceInterval: (0.6, 1.0), rescaleT: 3.0)
let slatLR = lrNorm * sStd + sMean
MLX.eval(slatLR)
let refSlatLR = try golden("casc_slat_lr")
let lrCos = cosine(slatLR, refSlatLR)
gate("LRSAMP  512-flow LR slat", lrCos >= 0.99,
     String(format: "N=%d cosine=%.6f (%.1fs)", coordsLR.dim(0), lrCos, -t0.timeIntervalSinceNow))

// ---- 2) "upsample" = shape decoder forward-output coords on the ORACLE LR slat ----
t0 = Date()
let shapeDec = ShapeSlatDecoder(weights: try loadArrays(url: URL(fileURLWithPath: "\(ckpts)/shape_dec_next_dc_f16c32_fp16.safetensors")))
let upOut = shapeDec(SparseTensor(feats: refSlatLR.asType(.float32), coords: coordsLR))
MLX.eval(upOut.coords)
let refUp = (try golden("casc_up_coords")).asType(.int32)
let ourKeySet = Set(packedKeys(upOut.coords))
let refKeySet = Set(packedKeys(refUp))
let iou = Float(ourKeySet.intersection(refKeySet).count) / Float(ourKeySet.union(refKeySet).count)
// The cascade consumes upsample coords only through re-quantization — gate on that
// contract exactly (fp subdivision ties flip a handful of raw 512-grid children).
let (q1024ours, hr1024ours) = CascadeQuantize.requantize(upCoords: upOut.coords, lrResolution: 512, targetResolution: 1024)
let (q1536ours, hr1536ours) = CascadeQuantize.requantize(upCoords: upOut.coords, lrResolution: 512, targetResolution: 1536)
let refQ1024 = Set(packedKeys((try golden("casc_coords_hr1024")).asType(.int32)))
let refQ1536 = Set(packedKeys((try golden("casc_coords_hr1536")).asType(.int32)))
let q1024set = Set(packedKeys(q1024ours)), q1536set = Set(packedKeys(q1536ours))
let q1024ok = hr1024ours == 1024 && q1024set == refQ1024
// At the finer 96³ grid a tie-flipped subtree can still move a token cell; allow ≤0.2%
// symmetric difference there (each cell = one SLat token; the HR sampler's token set is
// functionally identical at that level).
let d1536 = q1536set.symmetricDifference(refQ1536).count
let q1536ratio = Float(d1536) / Float(refQ1536.count)
let q1536ok = hr1536ours == 1536 && q1536ratio <= 0.002
gate("UPCOORD decoder upsample coords", iou >= 0.998 && q1024ok && q1536ok,
     String(format: "ours=%d oracle=%d IoU=%.6f quantized@1024 %@ @1536 diff=%d (%.4f%%) (%.1fs)",
            ourKeySet.count, refKeySet.count, iou, q1024ok ? "exact" : "MISMATCH",
            d1536, q1536ratio * 100, -t0.timeIntervalSinceNow))

// ---- 3) re-quantization + token back-off (exact, incl. chosen hr_res) ----
t0 = Date()
func quantGate(_ name: String, upFixture: String, target: Int, coordsFixture: String, hrresFixture: String) throws {
    let up = (try golden(upFixture)).asType(.int32)
    let (q, hr) = CascadeQuantize.requantize(upCoords: up, lrResolution: 512, targetResolution: target)
    let refQ = (try golden(coordsFixture)).asType(.int32)
    let refHr = Int((try golden(hrresFixture)).asType(.int32).asArray(Int32.self)[0])
    let ok = hr == refHr && q.dim(0) == refQ.dim(0) && packedKeys(q) == packedKeys(refQ)
    gate(name, ok, "tokens=\(q.dim(0)) (oracle \(refQ.dim(0))) hr=\(hr) (oracle \(refHr))")
}
try quantGate("QUANT   target 1024", upFixture: "casc_up_coords", target: 1024,
              coordsFixture: "casc_coords_hr1024", hrresFixture: "casc_hrres_1024")
try quantGate("QUANT   target 1536", upFixture: "casc_up_coords", target: 1536,
              coordsFixture: "casc_coords_hr1536", hrresFixture: "casc_hrres_1536")
try quantGate("QUANT   back-off", upFixture: "casc_bk_up_coords", target: 1536,
              coordsFixture: "casc_bk_coords", hrresFixture: "casc_bk_hrres")
print("T03 QUANT   (\(String(format: "%.1f", -t0.timeIntervalSinceNow))s)")

// ---- 4) HR pass: 1024 shape flow on the re-quantized coords, cond_1024 ----
t0 = Date()
let coordsHR = (try golden("casc_coords_hr1024")).asType(.int32)
let flow1024 = SLatFlowModel(weights: try loadArrays(url: URL(fileURLWithPath: "\(ckpts)/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors")))
let hrNorm = FlowEulerSampler.sampleSLat(
    model: flow1024, noiseFeats: try golden("casc_noise_hr"), coords: coordsHR,
    cond: cond1024, negCond: neg1024,
    guidanceStrength: 7.5, guidanceRescale: 0.5, guidanceInterval: (0.6, 1.0), rescaleT: 3.0)
let slatHR = hrNorm * sStd + sMean
MLX.eval(slatHR)
let refSlatHR = try golden("casc_slat_hr")
let hrCos = cosine(slatHR, refSlatHR)
gate("HRSAMP  1024-flow HR slat", hrCos >= 0.99,
     String(format: "K=%d cosine=%.6f (%.1fs)", coordsHR.dim(0), hrCos, -t0.timeIntervalSinceNow))

// ---- 5) tex cascade mirror: tex 1024 flow, concat_cond = NORMALIZED oracle HR slat ----
t0 = Date()
let tex1024 = SLatFlowModel(weights: try loadArrays(url: URL(fileURLWithPath: "\(ckpts)/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors")))
let texNorm = FlowEulerSampler.sampleSLat(
    model: tex1024, noiseFeats: try golden("casc_noise_tex"), coords: coordsHR,
    cond: cond1024, negCond: neg1024, concatCond: (refSlatHR - sMean) / sStd,
    guidanceStrength: 1.0, guidanceRescale: 0.0, guidanceInterval: (0.6, 0.9), rescaleT: 3.0)
let texSlat = texNorm * tStd + tMean
MLX.eval(texSlat)
let texCos = cosine(texSlat, try golden("casc_tex_slat"))
gate("TEXSAMP tex-1024 slat", texCos >= 0.99,
     String(format: "cosine=%.6f (%.1fs)", texCos, -t0.timeIntervalSinceNow))

// ---- 6) 1024² DINOv3 conditioning (patch-count-agnostic port at 64×64 patches) ----
t0 = Date()
let dino = DINOv3(weights: try loadArrays(url: URL(fileURLWithPath: dinoPath)))
let condOurs = dino(try golden("dino_in_pixels_1024"))
MLX.eval(condOurs)
let dinoCos = cosine(condOurs, cond1024)
gate("DINO1024 cond", dinoCos >= 0.99,
     String(format: "tokens=%d cosine=%.7f (%.1fs)", condOurs.dim(1), dinoCos, -t0.timeIntervalSinceNow))

print("T03 CASCADE GATE \(allPass ? "PASS" : "FAIL")")
exit(allPass ? 0 : 1)
