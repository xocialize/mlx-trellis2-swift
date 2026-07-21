import Foundation
import MLX
import TRELLIS2

// Full native TRELLIS.2 generation in Swift: (native DINOv3 cond) → textured GLB.
// bf16-SDPA fast path on. Latents produced natively by the ported DiTs/samplers.
// Usage: swift run -c release generate [res512|res1024|res1536]   (default res1024 = HF-default cascade)
let goldens = "/Volumes/Satechi/TrellisRedux/trellis2-port/goldens"
let ckpts = "/Volumes/Satechi/TrellisRedux/models/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts"
let ssDecPath = "/Volumes/Satechi/TrellisRedux/models/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
let dinoPath = "/Volumes/Satechi/TrellisRedux/models/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51/model.safetensors"
func golden(_ n: String) throws -> MLXArray { try loadArray(url: URL(fileURLWithPath: "\(goldens)/\(n).npy")) }
func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let tier = Trellis2Tier(rawValue: CommandLine.arguments.dropFirst().first ?? "res1024") ?? .res1024

// phys_footprint peak sampler — the admission-basis measurement for the manifest re-baseline.
func physFootprintGB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1e9 : 0
}
nonisolated(unsafe) var peakPhys = 0.0
let physTimer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "phys"))
physTimer.schedule(deadline: .now(), repeating: .milliseconds(200))
physTimer.setEventHandler { peakPhys = max(peakPhys, physFootprintGB()) }
physTimer.resume()

TRELLIS2Config.fastAttention = true   // bf16 SDPA for inference speed

let t0 = Date()
log("[generate] tier \(tier.rawValue) — loading models…")
let pipe = try Trellis2Pipeline(ckptDir: ckpts, ssDecPath: ssDecPath, dinoPath: dinoPath)
log("[generate] loaded (\(String(format: "%.1f", -t0.timeIntervalSinceNow))s). Encoding image + generating…")

// cond computed NATIVELY via the ported DINOv3 from the preprocessed image pixels
// (dino_in_pixels[_1024] = the bg-removed/cropped/normalized T.png at each size).
let (cond, negCond) = pipe.encodeImage(try golden("dino_in_pixels"))
MLX.eval(cond, negCond)
let (cond1024, negCond1024) = tier.isCascade ? pipe.encodeImage(try golden("dino_in_pixels_1024"))
                                             : (cond, negCond)
MLX.eval(cond1024)
log("[generate] DINOv3 cond \(cond.shape) / \(cond1024.shape) computed natively")

let (baked, hrRes, _) = try pipe.generate(
    tier: tier,
    cond: cond, negCond: negCond, cond1024: cond1024, negCond1024: negCond1024,
    ssNoise: try golden("ss_noise"), ssPhases: try golden("rope_phases_cossin"),
    shapeMean: try golden("shape_slat_mean"), shapeStd: try golden("shape_slat_std"),
    texMean: try golden("tex_slat_mean"), texStd: try golden("tex_slat_std"),
    yUp: true,   // glTF convention — T0.2: HF/GLB parity is Y-up
    seed: 0, log: { log($0 + String(format: "   [phys %.1f GB]", physFootprintGB())) })

log("[generate] baked @\(hrRes): \(baked.vertices.dim(0)) verts, \(baked.faces.dim(0)) faces, atlas \(baked.atlasSize), coverage \(String(format: "%.1f", baked.coverage*100))%")
let outURL = URL(fileURLWithPath: "/Users/dustinnielson/Development/mlxengine-3d/DEV/TrellisDev/texturing/t03_\(tier.rawValue)_native.glb")
try GLTFExport.writeGLB(to: outURL, positions: baked.vertices, indices: baked.faces,
                        normals: baked.normals, uvs: baked.uvs,
                        baseColorRGBA: (baked.texRGBA, baked.atlasSize, baked.atlasSize))
let bytes = (try Data(contentsOf: outURL)).count
log("[generate] wrote \(outURL.lastPathComponent) (\(bytes) bytes) — total \(String(format: "%.1f", -t0.timeIntervalSinceNow))s, peak phys \(String(format: "%.2f", peakPhys)) GB")
print("GENERATE DONE \(tier.rawValue)@\(hrRes) \(baked.faces.dim(0)) faces \(String(format: "%.0f", -t0.timeIntervalSinceNow))s peakphys \(String(format: "%.2f", peakPhys))GB")
