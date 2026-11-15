import Foundation
import MLX
import TRELLIS2

// Detail-map bake bench: re-run MeshBake.run (+ the opt-in normal / AO maps) on saved decoder
// outputs and write the GLB, so the bake can be iterated without re-running diffusion.
//
//   swift run -c release bakemaps <bake-input.safetensors> <out.glb> [options]
//     --faces N    decimate target (default 300000, the package default)
//     --atlas N    base-colour atlas (default 1024)
//     --normal N   normal map edge (off unless given)
//     --ao N       AO map edge (off unless given)
//     --rays N     AO rays per texel (64)    --dist X   AO distance × bbox diagonal (0.25)
//     --bias X     AO lift × bbox diagonal (0.002)    --cage X   normal-map cage × bbox diagonal (0.01)
//     --pbr        also bake the metallic-roughness map (needs metallic_roughness in the input)
//     --repeat     bake twice and compare the GLB bytes (determinism)
//
// The input holds `shape_feats` [N,7], `coords` [N,4], `base_color` [N,3], optional
// `metallic_roughness` [N,2], `fine_res` [1] and `remesh` [1]: the arrays MeshBake.run takes,
// saved by a local dump right before the pipeline's bake call.

setvbuf(stdout, nil, _IONBF, 0)
var args = Array(CommandLine.arguments.dropFirst())
guard args.count >= 2 else { fatalError("usage: bakemaps <bake-input.safetensors> <out.glb> [options]") }
let input = URL(fileURLWithPath: args.removeFirst()), output = URL(fileURLWithPath: args.removeFirst())
func opt(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}
let faces = opt("--faces").flatMap(Int.init) ?? 300_000
let atlas = opt("--atlas").flatMap(Int.init) ?? 1024
let pbr = args.contains("--pbr")
var detail = DetailMapOptions(normalMapSize: opt("--normal").flatMap(Int.init), aoMapSize: opt("--ao").flatMap(Int.init))
if let r = opt("--rays").flatMap(Int.init) { detail.aoRays = r }
if let d = opt("--dist").flatMap(Float.init) { detail.aoMaxDistance = d }
if let b = opt("--bias").flatMap(Float.init) { detail.aoBias = b }
if let c = opt("--cage").flatMap(Float.init) { detail.normalCage = c }

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
physTimer.schedule(deadline: .now(), repeating: .milliseconds(100))
physTimer.setEventHandler { peakPhys = max(peakPhys, physFootprintGB()) }
physTimer.resume()

let a = try MLX.loadArrays(url: input)
guard let shape = a["shape_feats"], let coords = a["coords"], let base = a["base_color"] else {
    fatalError("input needs shape_feats, coords, base_color")
}
let fineRes = a["fine_res"]?.item(Float.self) ?? 1024
let remesh = a["remesh"].map { Int($0.item(Int32.self)) } ?? 256
let mr = pbr ? a["metallic_roughness"] : nil
if pbr && mr == nil { fatalError("--pbr: the input has no metallic_roughness") }
print("[bakemaps] \(coords.dim(0)) voxels @\(Int(fineRes)), remesh \(remesh), faces \(faces), atlas \(atlas), pbr \(pbr), normal \(detail.normalMapSize.map(String.init) ?? "off"), ao \(detail.aoMapSize.map(String.init) ?? "off") (rays \(detail.aoRays), dist \(detail.aoMaxDistance))")

func bake() throws -> (Data, BakeStageMetrics, Double) {
    Memory.peakMemory = 0
    let t = Date()
    let (baked, m) = try MeshBake.run(shapeFeats: shape, coords: coords.asType(.int32), texBaseColor: base,
                                      fineRes: fineRes, remeshRes: remesh, targetFaces: faces, atlasSize: atlas,
                                      texMetallicRoughness: mr, detailMaps: detail, log: { print($0) })
    let s = -t.timeIntervalSinceNow
    let d = baked.detail
    let occ: GLTFExport.Occlusion? = d.occlusionInMetallicRoughness ? .metallicRoughnessRed
        : d.occlusionRGBA.map { .texture(pixels: $0, width: d.occlusionSize, height: d.occlusionSize) }
    let glb = try GLTFExport.glbData(positions: baked.vertices, indices: baked.faces, normals: baked.normals, uvs: baked.uvs,
                                     baseColorRGBA: (baked.texRGBA, baked.atlasSize, baked.atlasSize),
                                     metallicRoughnessRGBA: baked.mrRGBA.map { ($0, baked.atlasSize, baked.atlasSize) },
                                     tangents: d.normalRGBA != nil ? d.tangents : nil,
                                     normalRGBA: d.normalRGBA.map { ($0, d.normalSize, d.normalSize) }, occlusion: occ)
    return (glb, m, s)
}

let (glb, m, s) = try bake()
try glb.write(to: output)
let dm = m.detail ?? DetailMapMetrics()
print(String(format: "[bakemaps] MeshBake.run %.2fs (tangents %.2fs, normal map %.2fs, AO %.2fs) | GPU peak %.2f GB | peak phys %.2f GB | %d bytes → %@",
             s, dm.tangentsS, dm.normalMapS, dm.aoMapS, Double(Memory.peakMemory) / 1e9, peakPhys, glb.count, output.path))
if let data = try? JSONEncoder().encode(m), let str = String(data: data, encoding: .utf8) { print("METRICS \(str)") }
if args.contains("--repeat") {
    let (again, _, s2) = try bake()
    print(String(format: "[bakemaps] repeat %.2fs: GLB bytes %@", s2, again == glb ? "IDENTICAL" : "DIFFER"))
}
