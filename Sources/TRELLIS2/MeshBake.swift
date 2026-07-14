import Foundation
import MLX
import MLXMesh

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
        log: (String) -> Void = { print($0) }
    ) throws -> BakedMesh {
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
        log("  raw dual-grid: \(mesh.vertexCount) verts, \(mesh.faceCount) faces, \(mesh.numConnectedComponents) components, \(mesh.numBoundaryEdges) boundary edges")

        // 2) clean: DC remesh (watertight) then simplify to budget
        mesh = mesh.remeshDualContouring(resolution: remeshRes)
        MLX.eval(mesh.vertices, mesh.faces)
        log("  remeshed: \(mesh.vertexCount) verts, \(mesh.faceCount) faces, \(mesh.numConnectedComponents) components, \(mesh.numBoundaryEdges) boundary edges")
        if mesh.faceCount > targetFaces {
            mesh = mesh.simplify(targetNumFaces: targetFaces)
            MLX.eval(mesh.vertices, mesh.faces)
            log("  simplified: \(mesh.vertexCount) verts, \(mesh.faceCount) faces")
        }

        // 3) UV unwrap
        let uv = try mesh.uvUnwrap()
        let finalMesh = uv.mesh
        MLX.eval(finalMesh.vertices, finalMesh.faces, uv.uvs)
        log("  unwrapped: \(finalMesh.vertexCount) verts, atlas \(uv.atlasWidth)x\(uv.atlasHeight), \(uv.charts.count) charts")

        // 4) rasterize + bake
        let uvPix = uv.uvs * Float(atlasSize)
        let (pix, fid, bary) = UVRasterize.rasterize(uvsPix: uvPix, faces: finalMesh.faces, textureSize: atlasSize)
        MLX.eval(pix, fid, bary)
        let K = fid.dim(0)
        log("  rasterized: \(K) covered texels")

        // surface position per texel: Σ bary · verts[faces[fid]]
        let faceVerts = finalMesh.faces.take(fid, axis: 0)                 // [K,3] vertex ids
        let v0 = finalMesh.vertices.take(faceVerts[0..., 0], axis: 0)      // [K,3]
        let v1 = finalMesh.vertices.take(faceVerts[0..., 1], axis: 0)
        let v2 = finalMesh.vertices.take(faceVerts[0..., 2], axis: 0)
        let posRaw = bary[0..., 0..<1] * v0 + bary[0..., 1..<2] * v1 + bary[0..., 2..<3] * v2   // [K,3]
        // remap each texel position onto the original on-shell surface, so trilinear hits the thin PBR shell
        let pos = origBVH.closestPoints(mesh: origMesh, queries: posRaw).points                 // [K,3]
        MLX.eval(pos)
        let query = ((pos + 0.5) * fineRes).reshaped([1, K, 3])
        let sampled = GridSample3d.sample(feats: texBaseColor, coords: coords, grid: query, mode: "trilinear")[0]  // [K,3]
        MLX.eval(sampled)
        let qmin = query.min(), qmax = query.max()
        let nonzero = (sampled.sum(axis: 1) .> 0).asType(.float32).mean().item(Float.self)
        log("  bake diag: baseColor mean=\(texBaseColor.mean().item(Float.self))  query[\(qmin.item(Float.self))..\(qmax.item(Float.self))]  sampled mean=\(sampled.mean().item(Float.self))  hitFrac=\(nonzero)")

        // 5) write texels + host-side dilation inpaint
        let px = pix.asType(.int32).asArray(Int32.self)                   // [K*2]
        let col = MLX.clip(sampled, min: 0, max: 1).asArray(Float.self)   // [K*3]
        var rgb = [Float](repeating: 0, count: atlasSize * atlasSize * 3)
        var filled = [Bool](repeating: false, count: atlasSize * atlasSize)
        for k in 0..<K {
            let x = Int(px[k*2]), y = Int(px[k*2+1])
            let p = y * atlasSize + x
            rgb[p*3] = col[k*3]; rgb[p*3+1] = col[k*3+1]; rgb[p*3+2] = col[k*3+2]
            filled[p] = true
        }
        let coverage = Float(filled.lazy.filter { $0 }.count) / Float(atlasSize * atlasSize)
        dilateInpaint(&rgb, &filled, atlasSize, iterations: 12)

        var rgba = [UInt8](repeating: 0, count: atlasSize * atlasSize * 4)
        for p in 0..<(atlasSize * atlasSize) {
            rgba[p*4] = UInt8(max(0, min(255, Int(rgb[p*3] * 255 + 0.5))))
            rgba[p*4+1] = UInt8(max(0, min(255, Int(rgb[p*3+1] * 255 + 0.5))))
            rgba[p*4+2] = UInt8(max(0, min(255, Int(rgb[p*3+2] * 255 + 0.5))))
            rgba[p*4+3] = 255
        }
        return BakedMesh(vertices: finalMesh.vertices, faces: finalMesh.faces,
                         normals: finalMesh.vertexNormals(), uvs: uv.uvs, texRGBA: rgba,
                         atlasSize: atlasSize, coverage: coverage)
    }

    /// Fill unfilled texels from filled 4-neighbours (seam/gap inpaint, push-out).
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
