import Foundation
import MLX
import TRELLIS2

// Full native TRELLIS.2 generation in Swift: (injected DINOv3 cond) → textured GLB.
// bf16-SDPA fast path on. Latents produced natively by the ported DiTs/samplers.
let goldens = "/Volumes/Satechi/TrellisRedux/trellis2-port/goldens"
let ckpts = "/Volumes/Satechi/TrellisRedux/models/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts"
let ssDecPath = "/Volumes/Satechi/TrellisRedux/models/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
func golden(_ n: String) throws -> MLXArray { try loadArray(url: URL(fileURLWithPath: "\(goldens)/\(n).npy")) }
func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

TRELLIS2Config.fastAttention = true   // bf16 SDPA for inference speed

let t0 = Date()
log("[generate] loading models…")
let pipe = try Trellis2Pipeline(ckptDir: ckpts, ssDecPath: ssDecPath)
log("[generate] loaded (\(String(format: "%.1f", -t0.timeIntervalSinceNow))s). Generating…")

let baked = try pipe.generate(
    cond: try golden("cond_512"), negCond: try golden("neg_cond_512"),
    ssNoise: try golden("ss_noise"), ssPhases: try golden("rope_phases_cossin"),
    seed: 0, log: log)

log("[generate] baked: \(baked.vertices.dim(0)) verts, \(baked.faces.dim(0)) faces, atlas \(baked.atlasSize), coverage \(String(format: "%.1f", baked.coverage*100))%")
let outURL = URL(fileURLWithPath: "/private/tmp/claude-501/-Volumes-Satechi-TrellisRedux/7650dae1-8f9c-4462-a6f9-f2974ee27db5/scratchpad/trellis_native.glb")
try GLTFExport.writeGLB(to: outURL, positions: baked.vertices, indices: baked.faces,
                        normals: baked.normals, uvs: baked.uvs,
                        baseColorRGBA: (baked.texRGBA, baked.atlasSize, baked.atlasSize))
let bytes = (try Data(contentsOf: outURL)).count
log("[generate] wrote \(outURL.lastPathComponent) (\(bytes) bytes) — total \(String(format: "%.1f", -t0.timeIntervalSinceNow))s")
print("GENERATE DONE \(baked.faces.dim(0)) faces \(String(format: "%.0f", -t0.timeIntervalSinceNow))s")
