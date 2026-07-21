import Foundation
import MLX
import MLXMesh
import Xatlas

/// End-to-end mesh+texture stage: shape/tex decoder voxel outputs -> clean textured
/// mesh. Chains the verified ops: FlexiDualGrid head -> DualGridMesh -> mlx-swift-mesh
/// DC-remesh + simplify + UV unwrap -> UVRasterize -> GridSample3d bake -> inpaint.
public struct BakedMesh {
    public let vertices: MLXArray   // [V,3]
    public let faces: MLXArray      // [F,3] int32
    public let normals: MLXArray    // [V,3]
    public let uvs: MLXArray        // [V,2] in [0,1)
    public let texRGBA: [UInt8]     // atlasSize*atlasSize*4
    public let atlasSize: Int
    public let coverage: Float      // fraction of texels covered before inpaint
    /// glTF metallicRoughness atlas (G=roughness, B=metallic), when the tex
    /// decoder's metallic/roughness channels were provided to the bake.
    public let mrRGBA: [UInt8]?
    /// Diagnostics (atlasSize² each): texel rasterized before inpaint; texel's
    /// closest-point remap landed on an opposing-normal (wrong-wall) raw face.
    public let filledBeforeInpaint: [Bool]
    public let wallFlipped: [Bool]
}

/// Which unwrap produces the bake's UV atlas (UV-UNWRAP-METAL-PLAN.md).
/// `.xatlas`: remesh → simplify → xatlas chart computation (CPU, minutes on
/// production meshes). `.provenance`: tagged remesh → grid-provenance charts →
/// axis projection → xatlas pack only (seconds; simplify integration pending,
/// so the mesh stays at remesh resolution).
public enum UnwrapBackend: String, Sendable {
    case xatlas, provenance
}

public enum MeshBake {
    static func softplus(_ x: MLXArray) -> MLXArray {   // stable: max(x,0)+log(1+exp(-|x|))
        MLX.maximum(x, 0) + MLX.log(1 + MLX.exp(-MLX.abs(x)))
    }

    /// shapeFeats [N,7] (raw shape-decoder out), coords [N,4] (b,x,y,z),
    /// texBaseColor [N,3] in [0,1] (tex-decoder basecolor at the same coords),
    /// fineRes = the dual-grid voxel resolution the coords live in.
    public static func run(
        shapeFeats: MLXArray, coords: MLXArray, texBaseColor: MLXArray,
        fineRes: Float, remeshRes: Int = 256, targetFaces: Int = 120_000, atlasSize: Int = 1024,
        backend: UnwrapBackend = .xatlas,
        texMetallicRoughness: MLXArray? = nil,   // [N,2] metallic, roughness in [0,1]
        log: (String) -> Void = { print($0) }
    ) throws -> (BakedMesh, BakeStageMetrics) {
        // Sub-stage timers: each `-t.timeIntervalSinceNow` read is preceded by the
        // MLX.eval that forces the stage's GPU work to complete, so the delta is real
        // wall-clock (MLX is lazy). Do not move a read ahead of its barrier.
        var m = BakeStageMetrics()
        var t = Date()
        let margin: Float = 0.5
        let dv = (1 + 2 * margin) * MLX.sigmoid(shapeFeats[0..., 0..<3]) - margin
        let inter = shapeFeats[0..., 3..<6] .> 0
        let ql = softplus(shapeFeats[0..., 6..<7])
        let coords3 = coords[0..., 1..<4]

        // 1) raw dual-grid mesh
        let (rawV, rawF) = DualGridMesh.extract(coords: coords3, dualVerts: dv, intersected: inter, quadLerp: ql, gridSize: fineRes)
        MLX.eval(rawV, rawF)
        // keep the raw dual-grid mesh (verts sit ON the PBR voxel shell) as the
        // attribute-sampling reference surface — texel positions get BVH-remapped onto
        // it before sampling (matches postprocess.py's closest-point remap).
        let origMesh = Mesh(vertices: rawV, faces: rawF)
        let origBVH = origMesh.bvh()
        var mesh = origMesh
        m.rawDualgridS = -t.timeIntervalSinceNow   // includes the raw-shell BVH build (CPU, synchronous)
        m.rawFaces = mesh.faceCount
        log("  raw dual-grid: \(mesh.vertexCount) verts, \(mesh.faceCount) faces, \(mesh.numConnectedComponents) components, \(mesh.numBoundaryEdges) boundary edges")

        // 2+3) clean (DC remesh, watertight) then unwrap, per backend.
        let finalMesh: Mesh
        let finalUVs: MLXArray
        let finalNormals: MLXArray
        let chartCount: Int
        switch backend {
        case .xatlas:
            // simplify to budget, then xatlas. Unwrap MUST come after remesh+
            // simplify: xatlas on the raw dual-grid mesh is pathologically slow
            // (~2.2h); on the cleaned mesh it's ~minutes (measured, Phase 0).
            t = Date()
            mesh = mesh.remeshDualContouring(resolution: remeshRes)
            MLX.eval(mesh.vertices, mesh.faces)
            m.remeshS = -t.timeIntervalSinceNow; m.remeshFaces = mesh.faceCount; m.simplifyFaces = mesh.faceCount
            log("  remeshed: \(mesh.vertexCount) verts, \(mesh.faceCount) faces, \(mesh.numConnectedComponents) components, \(mesh.numBoundaryEdges) boundary edges")
            if mesh.faceCount > targetFaces {
                t = Date()
                mesh = mesh.simplify(targetNumFaces: targetFaces)
                MLX.eval(mesh.vertices, mesh.faces)
                m.simplifyS = -t.timeIntervalSinceNow; m.simplifyFaces = mesh.faceCount
                log("  simplified: \(mesh.vertexCount) verts, \(mesh.faceCount) faces")
            }
            // Parallel partitioned xatlas (measured 64.8–510.6× over the single
            // instance at identical chart quality within ~5%); falls back to the
            // direct path automatically for small meshes.
            t = Date()
            let uv = try mesh.uvUnwrapParallel()
            finalMesh = uv.mesh
            finalUVs = uv.uvs
            finalNormals = finalMesh.vertexNormals()
            chartCount = uv.charts.count
            m.unwrapS = -t.timeIntervalSinceNow   // CPU xatlas (swift-vendored-parallel lane), synchronous
        case .provenance:
            // remeshRes <= 0 skips the DC remesh entirely (official-pipeline
            // parity): the 256-grid remesh aliases the fine dual-grid's thin
            // double shell into interpenetrating walls — the visible shard
            // defect. The remesh exists to make xatlas tolerable (raw-mesh
            // xatlas ≈ 2.2 h); the provenance unwrap doesn't need it.
            var provMesh: Mesh
            var provAxis: [UInt8]?
            t = Date()
            if remeshRes > 0 {
                let (tagged, faceAxis) = mesh.remeshDualContouringTagged(resolution: remeshRes)
                MLX.eval(tagged.vertices, tagged.faces)
                log("  remeshed(tagged): \(tagged.vertexCount) verts, \(tagged.faceCount) faces")
                provMesh = tagged
                provAxis = faceAxis
            } else {
                provMesh = mesh
                provAxis = nil   // axis from face normals
                log("  remesh skipped (official-parity): \(provMesh.faceCount) raw faces")
            }
            m.remeshS = -t.timeIntervalSinceNow; m.remeshFaces = provMesh.faceCount; m.simplifyFaces = provMesh.faceCount
            if provMesh.faceCount > targetFaces {
                t = Date()
                provMesh = provMesh.simplify(targetNumFaces: targetFaces)
                MLX.eval(provMesh.vertices, provMesh.faces)
                // nil tags → axis from each simplified face's own normal;
                // measured better than BVH tag transfer (mis-tagged large
                // faces project near edge-on and starve of texels)
                provAxis = nil
                log("  simplified: \(provMesh.vertexCount) verts, \(provMesh.faceCount) faces")
                // Simplify's collapses can STITCH the o-voxel shell's two walls
                // together, seaming outward-wound outer faces to inward-wound
                // inner faces (~10k winding-inconsistent edges measured — the
                // seam-crack root cause). The full-xatlas backend silently
                // repairs this by re-normalizing winding per chart; the
                // pack-only path must repair it explicitly.
                provMesh = provMesh.unifyFaceOrientations()
                MLX.eval(provMesh.vertices, provMesh.faces)
                m.simplifyS = -t.timeIntervalSinceNow; m.simplifyFaces = provMesh.faceCount
            }
            // (padding gutters and stronger absorption were both measured NULL
            // on the seam-crack artifact and cost texel density — keep defaults)
            t = Date()
            let prov = try provMesh.provenanceUnwrap(faceAxis: provAxis)
            finalMesh = prov.unwrap.mesh
            finalUVs = prov.unwrap.uvs
            chartCount = prov.unwrap.charts.count
            m.unwrapS = -t.timeIntervalSinceNow
            // Normals from the PRE-split mesh, carried through the split maps.
            // Computing them on the seam-split mesh gives every chart-border
            // vertex one-sided normals; sliver/poke-through triangles then
            // light up dark instead of shading like the surrounding surface
            // (user-visible shards; xatlas's smoother normals camouflage the
            // same geometry). Normals are geometric, not chart-dependent.
            let preNormals = provMesh.vertexNormals()
            let splitMapArr = MLXArray(prov.splitVertexMap, [prov.splitVertexMap.count])
            let origIdx = splitMapArr.take(prov.unwrap.vertexMap, axis: 0)
            finalNormals = preNormals.take(origIdx, axis: 0)
        }
        MLX.eval(finalMesh.vertices, finalMesh.faces, finalUVs)
        m.finalFaces = finalMesh.faceCount; m.charts = chartCount
        log("  unwrapped[\(backend.rawValue)]: \(finalMesh.vertexCount) verts, \(chartCount) charts")

        // 4) rasterize + bake. Center-inside rasterization leaves sub-texel
        // gaps along jagged chart seams (thin slivers never cover a texel
        // center); those gaps then inpaint-fill with unrepresentative color —
        // visible dark seam "cracks" (worst for provenance charts, whose
        // boundaries are staircase-jagged). Approximate conservative
        // rasterization with 4 half-texel-jittered passes appended after the
        // main one; later duplicate writes lose (first-write-wins below), so
        // jitter only fills otherwise-empty texels.
        // Supersample 2×: bake into a double-resolution grid and box-average
        // down. Each final texel becomes the mean of up to 4 surface samples —
        // fixes the foreshortened-seam-texel footprint problem (texel covers
        // ~2× surface, single center sample unrepresentative) and dilutes
        // single-sample crevice pinning (measured: 64%/36% of provenance's
        // remaining bad samples respectively).
        t = Date()
        let ss = 2
        let ssSize = atlasSize * ss
        let uvPix = finalUVs * Float(ssSize)
        var pixParts: [MLXArray] = [], fidParts: [MLXArray] = [], baryParts: [MLXArray] = []
        for (dx, dy): (Float, Float) in [(0, 0), (0.49, 0.49), (-0.49, 0.49), (0.49, -0.49), (-0.49, -0.49)] {
            let jittered = dx == 0 && dy == 0 ? uvPix
                : MLX.stacked([uvPix[0..., 0] + dx, uvPix[0..., 1] + dy], axis: 1)
            let (p, f, b) = UVRasterize.rasterize(uvsPix: jittered, faces: finalMesh.faces, textureSize: ssSize)
            pixParts.append(p); fidParts.append(f); baryParts.append(b)
        }
        let pix = MLX.concatenated(pixParts, axis: 0)
        let fid = MLX.concatenated(fidParts, axis: 0)
        let bary = MLX.concatenated(baryParts, axis: 0)
        MLX.eval(pix, fid, bary)
        m.rasterizeS = -t.timeIntervalSinceNow
        let K = fid.dim(0)
        m.coveredTexelWrites = K
        log("  rasterized: \(K) covered texel-writes (5-pass conservative)")

        // surface position per texel: Σ bary · verts[faces[fid]]
        t = Date()
        let faceVerts = finalMesh.faces.take(fid, axis: 0)                 // [K,3] vertex ids
        let v0 = finalMesh.vertices.take(faceVerts[0..., 0], axis: 0)      // [K,3]
        let v1 = finalMesh.vertices.take(faceVerts[0..., 1], axis: 0)
        let v2 = finalMesh.vertices.take(faceVerts[0..., 2], axis: 0)
        let posRaw = bary[0..., 0..<1] * v0 + bary[0..., 1..<2] * v1 + bary[0..., 2..<3] * v2   // [K,3]
        // remap each texel position onto the original on-shell surface, so trilinear hits the thin PBR shell
        let cp = origBVH.closestPoints(mesh: origMesh, queries: posRaw)
        let pos = cp.points                                                                     // [K,3]
        // diagnostic: does the remap land on a raw face whose normal opposes
        // the texel's final-mesh face normal? (wrong-wall snap)
        let fe1 = v1 - v0, fe2 = v2 - v0
        let fnl = MLX.stacked([
            fe1[0..., 1] * fe2[0..., 2] - fe1[0..., 2] * fe2[0..., 1],
            fe1[0..., 2] * fe2[0..., 0] - fe1[0..., 0] * fe2[0..., 2],
            fe1[0..., 0] * fe2[0..., 1] - fe1[0..., 1] * fe2[0..., 0],
        ], axis: 1)
        let rawFN = origMesh.faceNormals().take(cp.faceIndices, axis: 0)
        let flipDot = (fnl * rawFN).sum(axis: 1)
        MLX.eval(flipDot)
        let flipArr = flipDot.asArray(Float.self)
        MLX.eval(pos)
        let query = ((pos + 0.5) * fineRes).reshaped([1, K, 3])
        let sampled = GridSample3d.sample(feats: texBaseColor, coords: coords, grid: query, mode: "trilinear")[0]  // [K,3]
        MLX.eval(sampled)
        var mrSampled: [Float]? = nil
        if let mr = texMetallicRoughness {
            let mrS = GridSample3d.sample(feats: mr, coords: coords, grid: query, mode: "trilinear")[0]  // [K,2]
            MLX.eval(mrS)
            mrSampled = MLX.clip(mrS, min: 0, max: 1).asArray(Float.self)
        }
        m.bakeSampleS = -t.timeIntervalSinceNow   // BVH closest-point remap + GridSample3d trilinear
        let qmin = query.min(), qmax = query.max()
        let nonzero = (sampled.sum(axis: 1) .> 0).asType(.float32).mean().item(Float.self)
        log("  bake diag: baseColor mean=\(texBaseColor.mean().item(Float.self))  query[\(qmin.item(Float.self))..\(qmax.item(Float.self))]  sampled mean=\(sampled.mean().item(Float.self))  hitFrac=\(nonzero)")

        // 5) write texels + host-side dilation inpaint
        t = Date()
        let px = pix.asType(.int32).asArray(Int32.self)                   // [K*2]
        let col = MLX.clip(sampled, min: 0, max: 1).asArray(Float.self)   // [K*3]
        // --- write at supersample resolution, first-write-wins per sub-texel ---
        var ssRgb = [Float](repeating: 0, count: ssSize * ssSize * 3)
        var ssMr = mrSampled != nil ? [Float](repeating: 0, count: ssSize * ssSize * 2) : []
        var ssFilled = [Bool](repeating: false, count: ssSize * ssSize)
        var ssFlip = [Bool](repeating: false, count: ssSize * ssSize)
        for k in 0..<K {
            let x = Int(px[k*2]), y = Int(px[k*2+1])
            if x < 0 || x >= ssSize || y < 0 || y >= ssSize { continue }  // jitter can push 1 texel out
            let p = y * ssSize + x
            if ssFilled[p] { continue }   // first write wins: center pass beats jitter passes
            ssRgb[p*3] = col[k*3]; ssRgb[p*3+1] = col[k*3+1]; ssRgb[p*3+2] = col[k*3+2]
            if let mrS = mrSampled { ssMr[p*2] = mrS[k*2]; ssMr[p*2+1] = mrS[k*2+1] }
            ssFilled[p] = true
            if flipArr[k] < 0 { ssFlip[p] = true }
        }

        // --- box-average down to the final atlas ---
        var rgb = [Float](repeating: 0, count: atlasSize * atlasSize * 3)
        var mrBuf = mrSampled != nil ? [Float](repeating: 0, count: atlasSize * atlasSize * 3) : []
        var filled = [Bool](repeating: false, count: atlasSize * atlasSize)
        var wallFlipped = [Bool](repeating: false, count: atlasSize * atlasSize)
        for y in 0..<atlasSize {
            for x in 0..<atlasSize {
                var r: Float = 0, g: Float = 0, b: Float = 0, m: Float = 0, ro: Float = 0
                var cnt = 0
                var anyFlip = false
                for sy in 0..<ss {
                    for sx in 0..<ss {
                        let sp = (y * ss + sy) * ssSize + (x * ss + sx)
                        if ssFilled[sp] {
                            r += ssRgb[sp*3]; g += ssRgb[sp*3+1]; b += ssRgb[sp*3+2]
                            if mrSampled != nil { m += ssMr[sp*2]; ro += ssMr[sp*2+1] }
                            if ssFlip[sp] { anyFlip = true }
                            cnt += 1
                        }
                    }
                }
                guard cnt > 0 else { continue }
                let p = y * atlasSize + x
                let inv = 1 / Float(cnt)
                rgb[p*3] = r * inv; rgb[p*3+1] = g * inv; rgb[p*3+2] = b * inv
                if mrSampled != nil { mrBuf[p*3] = m * inv; mrBuf[p*3+1] = ro * inv }
                filled[p] = true
                wallFlipped[p] = anyFlip
            }
        }
        m.writeDownsampleS = -t.timeIntervalSinceNow
        let filledBeforeInpaint = filled
        let coverage = Float(filled.lazy.filter { $0 }.count) / Float(atlasSize * atlasSize)
        m.coverage = coverage
        // Inpaint to COMPLETION, not a fixed ring count: any unfilled texel is
        // black, and the renderer's mip chain averages small charts against it,
        // producing dark shard artifacts at distance (user-visible; worst for
        // the provenance backend's lower packing utilization). The dilation
        // loop exits as soon as nothing new fills, so completion is cheap.
        t = Date()
        var mrFilled = filled   // copy BEFORE base-color inpaint mutates it
        dilateInpaint(&rgb, &filled, atlasSize, iterations: atlasSize)
        var mrRGBA: [UInt8]? = nil
        if mrSampled != nil {
            dilateInpaint(&mrBuf, &mrFilled, atlasSize, iterations: atlasSize)
            var out = [UInt8](repeating: 255, count: atlasSize * atlasSize * 4)
            for p in 0..<(atlasSize * atlasSize) {
                out[p*4] = 0                                                        // R unused
                out[p*4+1] = UInt8(max(0, min(255, Int(mrBuf[p*3+1] * 255 + 0.5)))) // G = roughness
                out[p*4+2] = UInt8(max(0, min(255, Int(mrBuf[p*3] * 255 + 0.5))))   // B = metallic
            }
            mrRGBA = out
        }

        var rgba = [UInt8](repeating: 0, count: atlasSize * atlasSize * 4)
        for p in 0..<(atlasSize * atlasSize) {
            rgba[p*4] = UInt8(max(0, min(255, Int(rgb[p*3] * 255 + 0.5))))
            rgba[p*4+1] = UInt8(max(0, min(255, Int(rgb[p*3+1] * 255 + 0.5))))
            rgba[p*4+2] = UInt8(max(0, min(255, Int(rgb[p*3+2] * 255 + 0.5))))
            rgba[p*4+3] = 255
        }
        m.inpaintS = -t.timeIntervalSinceNow
        let baked = BakedMesh(vertices: finalMesh.vertices, faces: finalMesh.faces,
                              normals: finalNormals, uvs: finalUVs, texRGBA: rgba,
                              atlasSize: atlasSize, coverage: coverage, mrRGBA: mrRGBA,
                              filledBeforeInpaint: filledBeforeInpaint, wallFlipped: wallFlipped)
        return (baked, m)
    }

    /// Fill unfilled texels from filled 4-neighbours (seam/gap inpaint, push-out).
    /// UV-preserving bake for TEXTURING an existing mesh: rasterize the mesh's OWN UVs and
    /// sample the PBR voxel field at each texel's surface position. No remesh/unwrap/BVH —
    /// the voxels were generated from this exact geometry, so texel positions lie on the
    /// shell already (mirrors trellis2_texturing.postprocess_mesh's keep-existing-UVs branch).
    ///
    /// `uvs` are glTF-convention [V,2] (v down); the atlas is written so that a glTF sampler
    /// reading these UVs sees the right texel (row = (1-v)·S). `texFeats` [N,C>=3] in [0,1]
    /// (decoder output already mapped via ·0.5+0.5); channel 0..3 baked as RGB.
    public static func bakeWithUVs(
        vertices: MLXArray, faces: MLXArray, uvs: MLXArray,
        texFeats: MLXArray, texCoords: MLXArray, fineRes: Float, atlasSize: Int,
        log: (String) -> Void = { print($0) }
    ) -> (rgba: [UInt8], filled: [Bool], coverage: Float) {
        let uvPix = MLX.concatenated([uvs[0..., 0..<1], 1 - uvs[0..., 1..<2]], axis: 1) * Float(atlasSize)
        let (pix, fid, bary) = UVRasterize.rasterize(uvsPix: uvPix, faces: faces, textureSize: atlasSize)
        MLX.eval(pix, fid, bary)
        let K = fid.dim(0)

        let faceVerts = faces.take(fid, axis: 0)
        let v0 = vertices.take(faceVerts[0..., 0], axis: 0)
        let v1 = vertices.take(faceVerts[0..., 1], axis: 0)
        let v2 = vertices.take(faceVerts[0..., 2], axis: 0)
        let pos = bary[0..., 0..<1] * v0 + bary[0..., 1..<2] * v1 + bary[0..., 2..<3] * v2
        let query = ((pos + 0.5) * fineRes).reshaped([1, K, 3])
        let sampled = GridSample3d.sample(feats: texFeats[0..., 0..<3], coords: texCoords, grid: query, mode: "trilinear")[0]
        MLX.eval(sampled)

        let px = pix.asType(.int32).asArray(Int32.self)
        let col = MLX.clip(sampled, min: 0, max: 1).asArray(Float.self)
        var rgb = [Float](repeating: 0, count: atlasSize * atlasSize * 3)
        var filled = [Bool](repeating: false, count: atlasSize * atlasSize)
        for k in 0..<K {
            let x = Int(px[k*2]), y = Int(px[k*2+1])
            let p = y * atlasSize + x
            rgb[p*3] = col[k*3]; rgb[p*3+1] = col[k*3+1]; rgb[p*3+2] = col[k*3+2]
            filled[p] = true
        }
        let coverage = Float(filled.lazy.filter { $0 }.count) / Float(atlasSize * atlasSize)
        let filledBefore = filled
        dilateInpaint(&rgb, &filled, atlasSize, iterations: 12)
        var rgba = [UInt8](repeating: 0, count: atlasSize * atlasSize * 4)
        for p in 0..<(atlasSize * atlasSize) {
            rgba[p*4] = UInt8(max(0, min(255, Int(rgb[p*3] * 255 + 0.5))))
            rgba[p*4+1] = UInt8(max(0, min(255, Int(rgb[p*3+1] * 255 + 0.5))))
            rgba[p*4+2] = UInt8(max(0, min(255, Int(rgb[p*3+2] * 255 + 0.5))))
            rgba[p*4+3] = 255
        }
        log("  bakeWithUVs: \(K) texels, coverage \(String(format: "%.3f", coverage))")
        return (rgba, filledBefore, coverage)
    }

    static func dilateInpaint(_ rgb: inout [Float], _ filled: inout [Bool], _ S: Int, iterations: Int) {
        for _ in 0..<iterations {
            var newlyFilled = [(Int, Float, Float, Float)]()
            for y in 0..<S {
                for x in 0..<S where !filled[y*S + x] {
                    var r: Float = 0, g: Float = 0, b: Float = 0, c = 0
                    for (dx, dy) in [(-1,0),(1,0),(0,-1),(0,1)] {
                        let nx = x+dx, ny = y+dy
                        if nx >= 0 && nx < S && ny >= 0 && ny < S && filled[ny*S+nx] {
                            let q = (ny*S+nx)*3; r += rgb[q]; g += rgb[q+1]; b += rgb[q+2]; c += 1
                        }
                    }
                    if c > 0 { newlyFilled.append((y*S+x, r/Float(c), g/Float(c), b/Float(c))) }
                }
            }
            if newlyFilled.isEmpty { break }
            for (p, r, g, b) in newlyFilled { rgb[p*3]=r; rgb[p*3+1]=g; rgb[p*3+2]=b; filled[p]=true }
        }
    }
}
