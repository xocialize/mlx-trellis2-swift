// unwrapbench — Phase 0 of UV-UNWRAP-METAL-PLAN.md.
// Stage-split timing of the production unwrap path on corpus meshes:
//   load -> [DC remesh] -> [simplify] -> xatlas computeCharts / packCharts (timed separately)
// plus an optional ModelIO baseline on the same prepared mesh.
//
// Usage:
//   swift run -c release unwrapbench [--remesh 256] [--target 120000] [--no-remesh] [--modelio] mesh1.ply mesh2.glb ...
// Emits one human line per stage and a final JSONL line per mesh (prefix "JSON ").
import Foundation
import MLX
import MLXMesh
import Xatlas
import ModelIO

struct Config {
    var remeshRes = 256
    var targetFaces = 120_000
    var doRemesh = true
    var doModelIO = false
    var doGolden = false
    var doProvenance = false
    var smoothIters = 2
    var minChartFaces = 24
    var paths: [String] = []
}

var cfg = Config()
var it = CommandLine.arguments.dropFirst().makeIterator()
while let a = it.next() {
    switch a {
    case "--remesh": cfg.remeshRes = Int(it.next() ?? "256") ?? 256
    case "--target": cfg.targetFaces = Int(it.next() ?? "120000") ?? 120_000
    case "--no-remesh": cfg.doRemesh = false
    case "--modelio": cfg.doModelIO = true
    case "--golden": cfg.doGolden = true
    case "--provenance": cfg.doProvenance = true
    case "--smooth": cfg.smoothIters = Int(it.next() ?? "2") ?? 2
    case "--min-chart": cfg.minChartFaces = Int(it.next() ?? "24") ?? 24
    default: cfg.paths.append(a)
    }
}
guard !cfg.paths.isEmpty else {
    FileHandle.standardError.write(Data("usage: unwrapbench [--remesh N] [--target N] [--no-remesh] [--modelio] <mesh...>\n".utf8))
    exit(2)
}

func now() -> Double { CFAbsoluteTimeGetCurrent() }
func log(_ s: String) { print(s); fflush(stdout) }

for path in cfg.paths {
    let name = (path as NSString).lastPathComponent
    log("=== \(name)")
    var record: [String: Any] = ["mesh": name, "remesh": cfg.doRemesh ? cfg.remeshRes : 0, "target": cfg.targetFaces]
    do {
        var t = now()
        let loaded = try readMesh(path: path)
        log("  load: \(loaded.vertexCount) verts, \(loaded.faceCount) faces in \(String(format: "%.2f", now() - t))s")
        record["raw_faces"] = loaded.faceCount

        var mesh = Mesh(
            vertices: MLXArray(loaded.positions, [loaded.vertexCount, 3]),
            faces: MLXArray(loaded.indices, [loaded.faceCount, 3])
        )
        if cfg.doRemesh {
            t = now()
            mesh = mesh.remeshDualContouring(resolution: cfg.remeshRes)
            MLX.eval(mesh.vertices, mesh.faces)
            record["remesh_s"] = now() - t
            log("  remesh(\(cfg.remeshRes)): \(mesh.faceCount) faces in \(String(format: "%.2f", now() - t))s")
        }
        if mesh.faceCount > cfg.targetFaces {
            t = now()
            mesh = mesh.simplify(targetNumFaces: cfg.targetFaces)
            MLX.eval(mesh.vertices, mesh.faces)
            record["simplify_s"] = now() - t
            log("  simplify: \(mesh.faceCount) faces in \(String(format: "%.2f", now() - t))s")
        }
        record["unwrap_faces"] = mesh.faceCount

        // Marshal once (mirrors Mesh.uvUnwrap's marshalling).
        let vRaw = mesh.vertices.asArray(Float.self)
        var positions = [SIMD3<Float>]()
        positions.reserveCapacity(mesh.vertexCount)
        for i in 0..<mesh.vertexCount {
            positions.append(SIMD3<Float>(vRaw[i*3], vRaw[i*3 + 1], vRaw[i*3 + 2]))
        }
        let indices = mesh.faces.asArray(Int32.self).map { UInt32(bitPattern: $0) }

        // Staged xatlas: addMesh -> computeCharts -> packCharts, timed separately.
        let atlas = Atlas()
        t = now()
        try atlas.addMesh(MeshInput(positions: positions, indices: indices))
        atlas.addMeshJoin()
        let tAdd = now() - t

        t = now()
        atlas.computeCharts()
        let tCharts = now() - t

        t = now()
        atlas.packCharts()
        let tPack = now() - t

        record["xatlas_add_s"] = tAdd
        record["xatlas_charts_s"] = tCharts
        record["xatlas_pack_s"] = tPack
        record["chart_count"] = Int(atlas.chartCount)
        record["atlas_wh"] = [Int(atlas.width), Int(atlas.height)]
        record["utilization"] = atlas.utilization.first.map { Double($0) } ?? -1
        log("  xatlas: add \(String(format: "%.2f", tAdd))s | charts \(String(format: "%.2f", tCharts))s | pack \(String(format: "%.2f", tPack))s | \(atlas.chartCount) charts, \(atlas.width)x\(atlas.height), util \(atlas.utilization.first ?? -1)")

        if cfg.doProvenance {
            func windingBad(_ m: Mesh) -> Int {
                let ff = m.faces.asArray(Int32.self)
                var dir = [UInt64: (Int8, Int8)]()   // edge -> (first dir, second dir)
                let stride = UInt64(m.vertexCount) + 1
                var bad = 0
                for fi in 0..<(ff.count / 3) {
                    for k in 0..<3 {
                        let a = UInt32(bitPattern: ff[fi*3 + k])
                        let b = UInt32(bitPattern: ff[fi*3 + (k + 1) % 3])
                        let key = UInt64(min(a, b)) * stride + UInt64(max(a, b))
                        let d: Int8 = a < b ? 1 : -1
                        if let (d0, d1) = dir[key] {
                            if d1 == 0 { dir[key] = (d0, d)
                                if d0 == d { bad += 1 } }
                        } else { dir[key] = (d, 0) }
                    }
                }
                return bad
            }
            // Phase-2 E2E: tagged remesh -> provenance charts -> axis UVs ->
            // seam split -> pack-only. Runs on the UNSIMPLIFIED remesh output
            // (label-preserving simplify is the next sub-phase).
            var t2 = now()
            let raw = Mesh(
                vertices: MLXArray(loaded.positions, [loaded.vertexCount, 3]),
                faces: MLXArray(loaded.indices, [loaded.faceCount, 3])
            )
            let (tagged, faceAxis) = raw.remeshDualContouringTagged(resolution: cfg.remeshRes)
            MLX.eval(tagged.vertices, tagged.faces)
            let tTagRemesh = now() - t2
            t2 = now()
            var provIn = tagged
            var provInAxis: [UInt8]? = faceAxis
            if tagged.faceCount > cfg.targetFaces {
                provIn = tagged.simplify(targetNumFaces: cfg.targetFaces)
                    .unifyFaceOrientations()
                MLX.eval(provIn.vertices, provIn.faces)
                provInAxis = nil
            }
            log("  WINDING-AUDIT input(simplified): \(windingBad(provIn)) bad edges, \(provIn.faceCount) faces")
            let prov = try provIn.provenanceUnwrap(
                faceAxis: provInAxis,
                smoothingIterations: cfg.smoothIters,
                minChartFaces: cfg.minChartFaces)
            log("  WINDING-AUDIT output(final): \(windingBad(prov.unwrap.mesh)) bad edges (note: split verts hide seam edges; merged-space audit needs vertexMap)")
            let tProv = now() - t2
            let u = prov.unwrap
            record["prov_remesh_s"] = tTagRemesh
            record["prov_unwrap_s"] = tProv
            record["prov_faces"] = tagged.faceCount
            record["prov_charts"] = prov.chartCount
            record["prov_packed_charts"] = u.charts.count
            record["prov_atlas_wh"] = [Int(u.atlasWidth), Int(u.atlasHeight)]
            record["prov_out_verts"] = u.mesh.vertexCount
            // UV validity on the packed result
            let uvArr = u.uvs.asArray(Float.self)
            var uvOK = true
            for x in uvArr where !x.isFinite || x < -0.001 || x > 1.001 { uvOK = false; break }
            record["prov_uvs_valid"] = uvOK
            // Foldover metric: per packed chart, area of the minority UV winding.
            // Mirrored charts are fine (uniform sign); folds are mixed signs.
            let outF = u.mesh.faces.asArray(Int32.self)
            var foldedCharts = 0
            var foldedArea = 0.0, totalArea = 0.0
            for chart in u.charts {
                var pos = 0.0, neg = 0.0
                for fIdx in chart.faceIndices {
                    let b = Int(fIdx) * 3
                    let i0 = Int(outF[b]), i1 = Int(outF[b+1]), i2 = Int(outF[b+2])
                    let e1u = uvArr[i1*2] - uvArr[i0*2], e1v = uvArr[i1*2+1] - uvArr[i0*2+1]
                    let e2u = uvArr[i2*2] - uvArr[i0*2], e2v = uvArr[i2*2+1] - uvArr[i0*2+1]
                    let a2 = Double(e1u * e2v - e1v * e2u)
                    if a2 >= 0 { pos += a2 } else { neg -= a2 }
                }
                let minority = min(pos, neg)
                if pos + neg > 0, minority / (pos + neg) > 0.01 { foldedCharts += 1 }
                foldedArea += minority
                totalArea += pos + neg
            }
            let foldFrac = totalArea > 0 ? foldedArea / totalArea : 0
            // uvArr is normalized [0,1): summed |triangle area|*2 / 2 = fraction
            // of the atlas actually covered = utilization (atlas dims themselves
            // are cosmetic — production rasterizes at its own atlasSize).
            let utilization = totalArea / 2
            record["prov_folded_charts"] = foldedCharts
            record["prov_fold_area_frac"] = foldFrac
            record["prov_utilization"] = utilization
            log("  provenance: remesh+tags \(String(format: "%.2f", tTagRemesh))s | charts+split+pack \(String(format: "%.2f", tProv))s | \(tagged.faceCount) faces -> \(prov.chartCount) charts (\(u.charts.count) packed), util \(String(format: "%.3f", utilization)), uvs valid: \(uvOK), folded: \(foldedCharts) charts / \(String(format: "%.3f", foldFrac * 100))% area")
        }
        if cfg.doGolden {
            // Phase-1 golden round-trip: full unwrap, then feed its own output
            // through the pack-only seam; atlas metrics must reproduce.
            t = now()
            let full = try mesh.uvUnwrap()
            let tFull = now() - t
            t = now()
            let repack = try full.mesh.uvUnwrap(existingUVs: full.uvs)
            let tRepack = now() - t
            let chartsMatch = abs(repack.charts.count - full.charts.count) <= max(5, full.charts.count / 200)
            let utilFull = full.atlasWidth > 0 ? Double(full.atlasWidth) * Double(full.atlasHeight) : 0
            let utilRepack = repack.atlasWidth > 0 ? Double(repack.atlasWidth) * Double(repack.atlasHeight) : 0
            let areaRatio = utilFull > 0 ? utilRepack / utilFull : -1
            record["golden_full_s"] = tFull
            record["golden_repack_s"] = tRepack
            record["golden_charts_full"] = full.charts.count
            record["golden_charts_repack"] = repack.charts.count
            record["golden_area_ratio"] = areaRatio
            record["golden_verts_full"] = full.mesh.vertexCount
            record["golden_verts_repack"] = repack.mesh.vertexCount
            // Equal-or-better packing passes: smaller repacked atlas (ratio < 1) is a win,
            // only penalize a repack that needs >10% MORE area or loses vertices/charts.
            let pass = chartsMatch && areaRatio < 1.1 && areaRatio > 0.5
                && repack.mesh.vertexCount == full.mesh.vertexCount
            record["golden_pass"] = pass
            log("  golden: full \(String(format: "%.2f", tFull))s (\(full.charts.count) charts) -> repack \(String(format: "%.2f", tRepack))s (\(repack.charts.count) charts), area ratio \(String(format: "%.3f", areaRatio)) => \(pass ? "PASS" : "FAIL")")
        }
        if cfg.doModelIO {
            t = now()
            let alloc = MDLMeshBufferDataAllocator()
            let vData = positions.withUnsafeBytes { Data($0) }
            let vBuf = alloc.newBuffer(with: vData, type: .vertex)
            var idx32 = indices
            let iData = idx32.withUnsafeMutableBytes { Data($0) }
            let iBuf = alloc.newBuffer(with: iData, type: .index)
            let sub = MDLSubmesh(indexBuffer: iBuf, indexCount: indices.count,
                                 indexType: .uInt32, geometryType: .triangles, material: nil)
            let desc = MDLVertexDescriptor()
            desc.attributes[0] = MDLVertexAttribute(name: MDLVertexAttributePosition,
                                                    format: .float3, offset: 0, bufferIndex: 0)
            desc.layouts[0] = MDLVertexBufferLayout(stride: MemoryLayout<SIMD3<Float>>.stride)
            let mdl = MDLMesh(vertexBuffer: vBuf, vertexCount: positions.count,
                              descriptor: desc, submeshes: [sub])
            mdl.addUnwrappedTextureCoordinates(forAttributeNamed: MDLVertexAttributeTextureCoordinate)
            let tMdl = now() - t
            let got = mdl.vertexDescriptor.attributeNamed(MDLVertexAttributeTextureCoordinate) != nil
            record["modelio_s"] = tMdl
            record["modelio_produced_uvs"] = got
            log("  modelio: \(String(format: "%.2f", tMdl))s, produced UVs: \(got)")
        }
    } catch {
        record["error"] = "\(error)"
        log("  ERROR: \(error)")
    }
    let json = try! JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
    log("JSON \(String(data: json, encoding: .utf8)!)")
}
