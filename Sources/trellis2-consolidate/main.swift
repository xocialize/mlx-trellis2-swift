import Foundation

// trellis2-consolidate — assemble the VERIFIED upstream checkpoints into the consolidated
// module-key layout the package loads (and that will be published to xocialize/trellis2-mlx).
//
// USER DECISION (SW6): the consolidated repo keeps the ORIGINAL microsoft/facebook key layout — NOT
// the old port's renamed-key conversion. So this is a pure per-file BYTE COPY under canonical
// filenames (keys + bf16/fp16 dtype preserved exactly; each component already self-verifies via its
// parity gate), plus a normalization.json distilled from pipeline.json. No MLX load/save, no remap.
//
//   MODELS=<hf-cache-root> OUT=<dir> swift run -c release trellis2-consolidate
//
// MODELS defaults to the SW-port weights volume; OUT defaults to ./trellis2-mlx-weights.

setvbuf(stdout, nil, _IONBF, 0)
let fm = FileManager.default

let modelsRoot = ProcessInfo.processInfo.environment["MODELS"]
    ?? "/Volumes/Satechi/TrellisRedux/models"
let outDir = ProcessInfo.processInfo.environment["OUT"]
    ?? "\(fm.currentDirectoryPath)/trellis2-mlx-weights"
try? fm.createDirectory(atPath: outDir, withIntermediateDirectories: true)

/// Resolve `<repo>/<rel>` under the HF-cache snapshot layout inside `modelsRoot`.
func resolve(_ repo: String, _ rel: String) -> String? {
    let base = "\(modelsRoot)/models--\(repo.replacingOccurrences(of: "/", with: "--"))/snapshots"
    guard let snaps = try? fm.contentsOfDirectory(atPath: base) else { return nil }
    for s in snaps { let p = "\(base)/\(s)/\(rel)"; if fm.fileExists(atPath: p) { return p } }
    // also accept a flat <modelsRoot>/<repo>/<rel> layout
    let flat = "\(modelsRoot)/\(repo)/\(rel)"
    return fm.fileExists(atPath: flat) ? flat : nil
}

let T2 = "microsoft/TRELLIS.2-4B"
// (outName, repo, relPath) — no remap; the published key ARE the Swift module keys.
let jobs: [(String, String, String)] = [
    ("dino",           "facebook/dinov3-vitl16-pretrain-lvd1689m", "model.safetensors"),
    ("struct_flow",    T2,                                          "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
    ("struct_dec",     "microsoft/TRELLIS-image-large",             "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
    ("shape_flow_512", T2,                                          "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"),
    ("shape_dec",      T2,                                          "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"),
    ("tex_flow_512",   T2,                                          "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors"),
    ("tex_dec",        T2,                                          "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"),
]

var copied = 0
for (name, repo, rel) in jobs {
    guard let src = resolve(repo, rel) else {
        FileHandle.standardError.write(Data("[consolidate] MISSING upstream \(repo)/\(rel)\n".utf8))
        continue
    }
    let dst = "\(outDir)/\(name).safetensors"
    try? fm.removeItem(atPath: dst)
    // HF-cache files are symlinks into blobs/ — resolve to the real blob so we copy the bytes,
    // not the link. resolvingSymlinksInPath handles the relative `../../blobs/<hash>` links.
    let realSrc = URL(fileURLWithPath: src).resolvingSymlinksInPath().path
    try fm.copyItem(atPath: realSrc, toPath: dst)
    let sz = ((try? fm.attributesOfItem(atPath: dst)[.size]) as? Int) ?? 0
    print("[consolidate] \(name).safetensors  (\(String(format: "%.1f", Double(sz)/1e6)) MB)  ← \(repo)/\(rel)")
    copied += 1
}

// normalization.json (shape + tex slat mean/std, from the upstream pipeline.json args).
if let pj = resolve(T2, "pipeline.json"),
   let data = fm.contents(atPath: pj),
   let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
   let args = obj["args"] as? [String: Any] {
    let norm: [String: Any] = [
        "shape_slat_normalization": args["shape_slat_normalization"] ?? [:],
        "tex_slat_normalization": args["tex_slat_normalization"] ?? [:],
    ]
    let nd = try JSONSerialization.data(withJSONObject: norm, options: [.prettyPrinted, .sortedKeys])
    try nd.write(to: URL(fileURLWithPath: "\(outDir)/normalization.json"))
    print("[consolidate] normalization.json written")
} else {
    FileHandle.standardError.write(Data("[consolidate] WARNING: could not build normalization.json (no pipeline.json)\n".utf8))
}

print("[consolidate] DONE → \(outDir) (\(copied)/\(jobs.count) safetensors)")
print("[consolidate] NOTE: also add LICENSE (MIT), NOTICE, and DINOv3_LICENSE.md + \"Built with DINOv3\" attribution before publishing to xocialize/trellis2-mlx.")
