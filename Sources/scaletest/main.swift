import Foundation
import MLX
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
print("SCALETEST DONE \(out.count) voxels \(String(format: "%.1f", dt))s")
