import Foundation
import MLX
import TRELLIS2

// SW-DINO parity gate: verify the Swift DINOv3 ViT-L/16 port stage-by-stage
// against the PyTorch oracle goldens, then end-to-end vs cond_512.
// Run: swift run -c release dinoparity
let goldens = "/Volumes/Satechi/TrellisRedux/trellis2-port/goldens"
let dinoWeights = "/Volumes/Satechi/TrellisRedux/models/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51/model.safetensors"

func golden(_ name: String) throws -> MLXArray {
    try loadArray(url: URL(fileURLWithPath: "\(goldens)/\(name).npy"))
}
func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let af = a.reshaped([-1]).asType(.float32)
    let bf = b.reshaped([-1]).asType(.float32)
    let den = MLX.sqrt((af * af).sum()) * MLX.sqrt((bf * bf).sum())
    return ((af * bf).sum() / den).item(Float.self)
}
func report(_ tag: String, _ got: MLXArray, _ gold: MLXArray, gate: Float = 0.999) -> Bool {
    let c = cosine(got, gold)
    let m = MLX.abs(got.asType(.float32) - gold.asType(.float32)).max().item(Float.self)
    let pass = c >= gate
    print("[\(tag)] cosine = \(c)  maxAbs = \(m)  \(pass ? "PASS" : "FAIL")")
    return pass
}

print("=== SW-DINO parity ===")
let w = try loadArrays(url: URL(fileURLWithPath: dinoWeights))
print("[weights] loaded \(w.count) DINOv3 tensors")
let dino = DINOv3(weights: w)

var allPass = true

// 1. RoPE cos/sin construction (32×32 patch grid, headDim 64)
let (cos, sin) = dino.ropeCosSin(32, 32)
allPass = report("RoPE cos", cos, try golden("dino_rope_cos")) && allPass
allPass = report("RoPE sin", sin, try golden("dino_rope_sin")) && allPass

// 2. per-stage localization + end-to-end: dino_in_pixels -> affine-less-LN == cond_512
let pixels = try golden("dino_in_pixels")               // [1,3,512,512]
let (patchEmbed, afterBlk0, afterBlk23, out) = dino.forwardStages(pixels)
print("[shape] dino output \(out.shape)  (expect [1, 1029, 1024])")
allPass = report("patch_embed", patchEmbed, try golden("dino_patch_embed")) && allPass
allPass = report("after_blk0", afterBlk0, try golden("dino_after_blk0")) && allPass
allPass = report("after_blk23", afterBlk23, try golden("dino_after_blk23")) && allPass
allPass = report("DINOv3 cond", out, try golden("cond_512"), gate: 0.99) && allPass

// 3. neg cond = zeros (get_cond: torch.zeros_like) — direct equality (cosine is 0/0 here)
let negGold = try golden("neg_cond_512")
let negMax = MLX.abs(negGold.asType(.float32)).max().item(Float.self)
let negOk = negGold.shape == out.shape && negMax == 0
print("[DINOv3 neg_cond] all-zero = \(negMax == 0)  shape \(negGold.shape)  \(negOk ? "PASS" : "FAIL")")
allPass = negOk && allPass

print(allPass ? "\nSW-DINO GATE PASS ✅" : "\nSW-DINO GATE FAIL ❌")
exit(allPass ? 0 : 1)
