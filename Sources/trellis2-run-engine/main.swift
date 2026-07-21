import Foundation
import MLX
import MLXToolKit
import MLXServeCore
import MLXBiRefNet
import Trellis2Kit

// Engine-driven GPU end-to-end driver (mlx-swift-integration Stage-2 step 7).
//
// Drives the TRELLIS.2 image→3D package THROUGH `MLXServeEngine` — the real coordinator path:
//   register (license gate C7/C8 + device eligibility C10) → prepare (memory admission, timed)
//   → run (timed) → decode the `Mesh`/GLB artifact → write it + report engine-charged footprint.
// Proves the engine integration without the Xcode app. Real weights on the GPU stream (~10 min @512).
//
// Run DETACHED (nohup … & disown) — heavy GPU. Default device = GPU (the package's own weight loads
// pin the CPU stream internally per the Metal-watchdog lesson).
//
// Env:
//   IMG          input PNG/JPEG (required)
//   WEIGHTS_DIR  consolidated weights root (the trellis2-consolidate OUT dir) for pre-publish validation
//   OUT_GLB      output GLB path (default ./out_mesh_engine.glb)
//   HR_RES       512 (default) / 1024 / 1536

setvbuf(stdout, nil, _IONBF, 0)
func gb(_ bytes: UInt64) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }

// phys_footprint peak sampler — the admission-basis measurement (manifest re-baseline).
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

let env = ProcessInfo.processInfo.environment
let here = FileManager.default.currentDirectoryPath
guard let imgPath = env["IMG"], let pngData = FileManager.default.contents(atPath: imgPath) else {
    fatalError("set IMG=<png/jpeg> to a readable input image")
}
let outGLB = env["OUT_GLB"] ?? "\(here)/out_mesh_engine.glb"
let fmt: Image.Format = imgPath.lowercased().hasSuffix(".jpg") || imgPath.lowercased().hasSuffix(".jpeg") ? .jpeg : .png
let image = Image(format: fmt, data: pngData)

print("================ TRELLIS.2 image→3D · ENGINE-DRIVEN e2e ================")
print("Built with DINOv3.")   // C7 attribution obligation (DINOv3 conditioner)
print("[engine] input image: \(imgPath)")

let engine = MLXServeEngine()   // .permissiveOnly runs the two-layer license gate at register
print("[engine] device budget: \(gb(await engine.memory.budgetBytes)) | available \(gb(await engine.memory.availableBytes))")

do {
    let weightsOverride = env["WEIGHTS_DIR"].map { URL(fileURLWithPath: $0) }
    if let w = weightsOverride { print("[engine] weights override: \(w.path)") }
    // Full-workflow stage-timing profile (BENCHMARKS.md). The package writes one flat JSON
    // record here + logs a `JSON {…}` line; we read it back below and dump a stage table.
    let metricsPath = env["METRICS_JSON"] ?? "\(here)/trellis2_metrics.json"
    print("[engine] metrics → \(metricsPath)")
    // SDPA precision A/B knobs (fp32|fp16|bf16): HR_SDPA = the CFG shape SLat flows,
    // TEX_SDPA = the CFG-free tex flow. Unset = fp32 (the pre-knob behavior).
    if let v = env["HR_SDPA"] { print("[engine] slat CFG SDPA: \(v)") }
    if let v = env["TEX_SDPA"] { print("[engine] tex SDPA: \(v)") }
    let steps = env["STEPS"].flatMap { Int($0) } ?? 12
    if steps != 12 { print("[engine] sampler steps: \(steps)") }
    // CFG-cost experiment knobs (shape SLat flows): interval "lo,hi" + vNeg cache stride.
    if let v = env["SLAT_CFG_INTERVAL"] { print("[engine] slat CFG interval: \(v)") }
    if let v = env["SLAT_NEG_EVERY"] { print("[engine] slat CFG neg-every: \(v)") }

    // Foreground matting hook (T0.4), composed the way the app does it: BiRefNet registered as a
    // second package, prepared, and injected as the Trellis2Matting closure. Default ON — raw
    // (no-usable-alpha) inputs otherwise leak background slabs/color shifts into the mesh
    // (BENCHMARKS.md corpus validation). MATTING=off restores the old raw-input behavior.
    // Pre-matted RGBA inputs skip the hook per upstream has_alpha, so leaving it on is free there.
    var matting: Trellis2Matting? = nil
    var mattePkgID: PackageID? = nil
    if env["MATTING"]?.lowercased() != "off" {
        let modelsRoot = env["MODELS_ROOT"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: "Development/MLXModelStore")
        let matteID = try await engine.register(BiRefNetPackage.registration,
                                                configuration: BiRefNetConfiguration(modelsRootDirectory: modelsRoot))
        mattePkgID = matteID
        let tMat = Date()
        _ = try await engine.prepare(.matting, package: matteID)
        print(String(format: "[engine] matting: BiRefNet prepared in %.1fs (MATTING=off to disable)",
                     -tMat.timeIntervalSinceNow))
        matting = { image in
            let resp = try await engine.run(MattingRequest(image: image), package: matteID)
            guard let m = resp as? MattingResponse else { fatalError("unexpected matting response \(type(of: resp))") }
            return m.matte
        }
    } else {
        print("[engine] matting: OFF (raw inputs — background may leak into the mesh)")
    }

    let pkgID = try await engine.register(Trellis2Package.registration,
                                          configuration: Trellis2Configuration(
                                              weightsRootOverride: weightsOverride,
                                              steps: steps,
                                              unwrapBackend: env["UNWRAP_BACKEND"],
                                              metricsPath: metricsPath,
                                              slatCfgAttention: env["HR_SDPA"],
                                              texAttention: env["TEX_SDPA"],
                                              slatCfgInterval: env["SLAT_CFG_INTERVAL"],
                                              slatCfgNegEvery: env["SLAT_NEG_EVERY"].flatMap { Int($0) },
                                              matting: matting))
    print("[engine] registered packageID=\(pkgID) | backers(imageTo3D)=\(await engine.packages(for: .imageTo3D))")

    let tLoad = Date()
    _ = try await engine.prepare(.imageTo3D, package: pkgID)
    let afterLoad = await engine.memory
    print(String(format: "[engine] prepare OK in %.1fs | charged %@ | resident %@ / budget %@",
                 -tLoad.timeIntervalSinceNow, gb(afterLoad.residents[.imageTo3D] ?? 0),
                 gb(afterLoad.residentBytes), gb(afterLoad.budgetBytes)))

    let hrRes = env["HR_RES"].flatMap { Int($0) } ?? 512
    let mode: Mode = hrRes == 1024 ? ImageTo3DContract.res1024
                   : hrRes == 1536 ? ImageTo3DContract.res1536 : ImageTo3DContract.res512
    print("[engine] tier: \(mode.rawValue)")

    let tRun = Date()
    let response = try await engine.run(ImageTo3DRequest(image: image, mode: mode), package: pkgID)
    guard let resp = response as? ImageTo3DResponse else { fatalError("unexpected response \(type(of: response))") }
    try resp.mesh.data.write(to: URL(fileURLWithPath: outGLB))

    print(String(format: "[engine] run OK in %.0fs | GPU peak %.2f GB | peak phys %.2f GB",
                 -tRun.timeIntervalSinceNow, Double(GPU.peakMemory)/1e9, peakPhys))
    print("[engine] mesh: verts=\(resp.mesh.vertexCount ?? -1) faces=\(resp.mesh.faceCount ?? -1) "
        + "vertexColors=\(resp.mesh.hasVertexColors) bytes=\(resp.mesh.data.count)")
    print("[engine] engine-charged footprint(imageTo3D)=\(gb((await engine.memory).residents[.imageTo3D] ?? 0))")
    print("[engine] wrote GLB → \(outGLB)")

    // Dump the stage-timing profile (the point of this harness): stages sorted by cost,
    // each with its share of the end-to-end wall clock. Measure before optimizing.
    if let data = FileManager.default.contents(atPath: metricsPath),
       let rec = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
        let e2e = (rec["e2e_total_s"] as? Double) ?? 0
        print("================ STAGE PROFILE (e2e \(String(format: "%.1f", e2e))s) ================")
        let stages = rec.filter { $0.key.hasSuffix("_s") && $0.key != "e2e_total_s" && $0.key != "generate_total_s" }
            .compactMap { k, v -> (String, Double)? in (v as? Double).map { (k, $0) } }
            .sorted { $0.1 > $1.1 }
        for (k, s) in stages where s > 0.05 {
            let pct = e2e > 0 ? s / e2e * 100 : 0
            print(String(format: "  %-26@ %7.1fs  %5.1f%%", k as NSString, s, pct))
        }
        print("========================================================")
    } else {
        print("[engine] no metrics record at \(metricsPath)")
    }

    await engine.evict(.imageTo3D, package: pkgID)
    if let m = mattePkgID { await engine.evict(.matting, package: m) }
    print("[engine] evicted. resident now \(gb(await engine.memory.residentBytes))")
    print("================ DONE — engine path validated ================")
} catch {
    print("[engine] FAILED: \(error)")
    exit(1)
}
