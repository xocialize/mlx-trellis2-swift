import Foundation
import MLX
import MLXToolKit
import MLXServeCore
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
    let pkgID = try await engine.register(Trellis2Package.registration,
                                          configuration: Trellis2Configuration(
                                              weightsRootOverride: weightsOverride,
                                              unwrapBackend: env["UNWRAP_BACKEND"]))
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

    await engine.evict(.imageTo3D, package: pkgID)
    print("[engine] evicted. resident now \(gb(await engine.memory.residentBytes))")
    print("================ DONE — engine path validated ================")
} catch {
    print("[engine] FAILED: \(error)")
    exit(1)
}
