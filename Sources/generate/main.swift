import Foundation
import MLX
import TRELLIS2

// Full native TRELLIS.2 generation in Swift: (injected DINOv3 cond) → textured GLB.
// bf16-SDPA fast path on. Latents produced natively by the ported DiTs/samplers.
let goldens = "/Volumes/Satechi/TrellisRedux/trellis2-port/goldens"
let ckpts = "/Volumes/Satechi/TrellisRedux/models/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts"
let ssDecPath = "/Volumes/Satechi/TrellisRedux/models/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
let dinoPath = "/Volumes/Satechi/TrellisRedux/models/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51/model.safetensors"
func golden(_ n: String) throws -> MLXArray { try loadArray(url: URL(fileURLWithPath: "\(goldens)/\(n).npy")) }
func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

TRELLIS2Config.fastAttention = true   // bf16 SDPA for inference speed

let t0 = Date()
log("[generate] loading models…")
let pipe = try Trellis2Pipeline(ckptDir: ckpts, ssDecPath: ssDecPath, dinoPath: dinoPath)
log("[generate] loaded (\(String(format: "%.1f", -t0.timeIntervalSinceNow))s). Encoding image + generating…")

// cond computed NATIVELY via the ported DINOv3 from the preprocessed image pixels
// (dino_in_pixels = the bg-removed/cropped/normalized T.png). Last injected fixture removed.
let (cond, negCond) = pipe.encodeImage(try golden("dino_in_pixels"))
MLX.eval(cond, negCond)
log("[generate] DINOv3 cond \(cond.shape) computed natively")

// Smoke harness: only the 512 golden pixels exist, so reuse cond for both tiers (the app builds a
// real cond_1024 from 1024² preprocess). Exercises the pipeline end-to-end, not a strict 1024 golden.
let baked = try pipe.generate(
    cond: cond, negCond: negCond, cond1024: cond, negCond1024: negCond,
    ssNoise: try golden("ss_noise"), ssPhases: try golden("rope_phases_cossin"),
    shapeMean: try golden("shape_slat_mean"), shapeStd: try golden("shape_slat_std"),
    texMean: try golden("tex_slat_mean"), texStd: try golden("tex_slat_std"),
    yUp: true,   // glTF convention — T0.2: HF/GLB parity is Y-up
    seed: 0, log: log)

log("[generate] baked: \(baked.vertices.dim(0)) verts, \(baked.faces.dim(0)) faces, atlas \(baked.atlasSize), coverage \(String(format: "%.1f", baked.coverage*100))%")
let outURL = URL(fileURLWithPath: "/Users/dustinnielson/Development/mlxengine-3d/DEV/TrellisDev/texturing/t02_native_yup.glb")
try GLTFExport.writeGLB(to: outURL, positions: baked.vertices, indices: baked.faces,
                        normals: baked.normals, uvs: baked.uvs,
                        baseColorRGBA: (baked.texRGBA, baked.atlasSize, baked.atlasSize))
let bytes = (try Data(contentsOf: outURL)).count
log("[generate] wrote \(outURL.lastPathComponent) (\(bytes) bytes) — total \(String(format: "%.1f", -t0.timeIntervalSinceNow))s")
print("GENERATE DONE \(baked.faces.dim(0)) faces \(String(format: "%.0f", -t0.timeIntervalSinceNow))s")
