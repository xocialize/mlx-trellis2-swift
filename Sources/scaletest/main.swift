import Foundation
import MLX
import MLXRandom
import TRELLIS2

// Full-scale stress + parity: run the shape SLat decoder on the FULL 19548-voxel
// shape_slat (the whole T.png object, not the octant) → ~7.85M output voxels, the
// biggest scaling risk (host-dict neighbor map at 7.85M). Verify vs the full golden.
let goldens = "/Volumes/Satechi/TrellisRedux/trellis2-port/goldens"
func golden(_ n: String) throws -> MLXArray { try loadArray(url: URL(fileURLWithPath: "\(goldens)/\(n).npy")) }
func err(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let decPath = "/Volumes/Satechi/TrellisRedux/models/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
let w = try loadArrays(url: URL(fileURLWithPath: decPath))
let dec = ShapeSlatDecoder(weights: w)
let inFeats = try golden("shapedec_in_feats")
let inCoords = (try golden("shapedec_in_coords")).asType(.int32)
err("[scaletest] input: \(inCoords.dim(0)) voxels — running full shape decoder…")

let t0 = Date()
let out = dec(SparseTensor(feats: inFeats, coords: inCoords))
MLX.eval(out.feats, out.coords)
let dt = -t0.timeIntervalSinceNow
err("[scaletest] decoded \(out.count) voxels in \(String(format: "%.1f", dt))s")

// verify vs full golden (free-run: intersection metric, since fp subdivision ties compound)
let gFeats = try golden("shapedec_out_feats")
let gCoords = (try golden("shapedec_out_coords")).asType(.int32)
let Ns = out.count, Ng = gFeats.dim(0)
let cs = out.coords.asType(.int32).asArray(Int32.self)
let cg = gCoords.asArray(Int32.self)
func key(_ a: [Int32], _ i: Int) -> Int64 { ((Int64(a[i*4]) << 33) | (Int64(a[i*4+1]) << 22) | (Int64(a[i*4+2]) << 11) | Int64(a[i*4+3])) }
var goldRow = [Int64: Int32](minimumCapacity: Ng * 2)
for i in 0..<Ng { goldRow[key(cg, i)] = Int32(i) }
var sIdx = [Int32](), gIdx = [Int32](); sIdx.reserveCapacity(Ns); gIdx.reserveCapacity(Ns)
for i in 0..<Ns { if let g = goldRow[key(cs, i)] { sIdx.append(Int32(i)); gIdx.append(g) } }
let inter = sIdx.count
let sf = gFeats.take(MLXArray(gIdx), axis: 0)   // reuse: gather golden at matched rows
let swf = out.feats.take(MLXArray(sIdx), axis: 0)
let af = swf.reshaped([-1]).asType(.float32), bf = sf.reshaped([-1]).asType(.float32)
let cos = ((af * bf).sum() / (MLX.sqrt((af*af).sum()) * MLX.sqrt((bf*bf).sum()))).item(Float.self)
let symFrac = Double((Ns - inter) + (Ng - inter)) / Double(Ng)
err("[scaletest] swift=\(Ns) gold=\(Ng) shared=\(inter) symDiff=\(String(format: "%.3f", symFrac*100))%  feats cosine(shared)=\(cos)")
err((symFrac < 0.02 && cos >= 0.99) ? "SCALETEST PASS (full 7.85M decode verified)" : "SCALETEST CHECK — review numbers")

// --- DiT/sampler scaling: shape SLat sampler at full production scale (19548 tokens) ---
// Only unproven-at-scale piece left (DiTs were gated on small fixtures). Times a full
// 12-step CFG sparse sampling loop over ~19548 voxels — the sparse-attention envelope.
let slatPath = "/Volumes/Satechi/TrellisRedux/models/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"
let slat = SLatFlowModel(weights: try loadArrays(url: URL(fileURLWithPath: slatPath)))
let ssCoords = (try golden("ssdec_coords")).asType(.int32)   // ~19548 SS-decoded voxels
let cond = try golden("cond_512"), neg = try golden("neg_cond_512")

// bf16-SDPA accuracy check: single forward vs the fp32 slat golden (expect ≈0.99+).
TRELLIS2Config.fastAttention = true
let bfIn = SparseTensor(feats: try golden("slat_in_feats"), coords: (try golden("slat_in_coords")).asType(.int32))
let bfOut = slat(bfIn, t: try golden("ssdit_in_t"), cond: cond).feats
let bg = try golden("slat_out_feats")
let ba = bfOut.reshaped([-1]).asType(.float32), bb = bg.reshaped([-1]).asType(.float32)
let bfCos = ((ba*bb).sum() / (MLX.sqrt((ba*ba).sum()) * MLX.sqrt((bb*bb).sum()))).item(Float.self)
err("[scaletest] bf16-SDPA forward cosine vs fp32 golden = \(bfCos)  (expect ≈0.99+)")

// bf16-SDPA shape SLat sampler timing (production inference path).
MLXRandom.seed(0)
let noise = MLXRandom.normal([ssCoords.dim(0), 32])
err("[scaletest] shape SLat sampler (bf16 SDPA): \(ssCoords.dim(0)) tokens, 12-step CFG (24 forwards)…")
let t1 = Date()
let shapeSlat = FlowEulerSampler.sampleSLat(
    model: slat, noiseFeats: noise, coords: ssCoords, cond: cond, negCond: neg,
    guidanceStrength: 7.5, guidanceRescale: 0.5, guidanceInterval: (0.6, 1.0), rescaleT: 3.0)
MLX.eval(shapeSlat)
let dt1 = -t1.timeIntervalSinceNow
let fin = shapeSlat.reshaped([-1])
let allFinite = MLX.all(fin .== fin).item(Bool.self)   // NaN check
let fmean = fin.mean(); let fstd = MLX.sqrt(((fin - fmean) * (fin - fmean)).mean()).item(Float.self)
err("[scaletest] shape SLat sampled \(shapeSlat.dim(0))×\(shapeSlat.dim(1)) in \(String(format: "%.1f", dt1))s (\(String(format: "%.1f", dt1/12)) s/step)  finite=\(allFinite) std=\(fstd)")
err("SLAT-SCALE DONE — bf16-SDPA DiT at production scale")
print("SCALETEST DONE decode \(String(format: "%.1f", dt))s  slat-sampler(bf16) \(String(format: "%.1f", dt1))s  fwdCos \(bfCos)")
