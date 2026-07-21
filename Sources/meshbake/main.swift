import Foundation
import MLX
import TRELLIS2

// End-to-end mesh+texture bake on real decoder-output fixtures (octant of the T.png
// decode): shape decoder 7-ch + tex decoder 6-ch -> clean textured GLB.
let goldens = "/Volumes/Satechi/TrellisRedux/trellis2-port/goldens"
func golden(_ n: String) throws -> MLXArray { try loadArray(url: URL(fileURLWithPath: "\(goldens)/\(n).npy")) }

// arg "octant" → the 990k-voxel corner fixture; default → the FULL 7.85M-voxel object.
let full = !CommandLine.arguments.contains("octant")
let (shN, texN, outName) = full
    ? ("shapedec_out_feats", "texdec_full_feats", "trellis_full.glb")
    : ("shapedec_sm_out_feats", "texdec_out_feats", "trellis_octant.glb")
let coordsN = full ? "shapedec_out_coords" : "shapedec_sm_out_coords"

let t0 = Date()
let shapeFeats = try golden(shN)                                // [N,7]
let coords = (try golden(coordsN)).asType(.int32)              // [N,4]
let texFeats = try golden(texN)                                // [N,6]
let baseColor = MLX.clip(texFeats[0..., 0..<3] * 0.5 + 0.5, min: 0, max: 1)   // [N,3] in [0,1]
print("[meshbake] loaded fixtures: \(coords.dim(0)) voxels (\(full ? "FULL object" : "octant"))")

let (baked, _) = try MeshBake.run(
    shapeFeats: shapeFeats, coords: coords, texBaseColor: baseColor,
    fineRes: 1024, remeshRes: 256, targetFaces: 120_000, atlasSize: 1024)

print("[meshbake] baked: \(baked.vertices.dim(0)) verts, \(baked.faces.dim(0)) faces, atlas \(baked.atlasSize), coverage \(String(format: "%.1f", baked.coverage*100))%")

let outURL = URL(fileURLWithPath: "/private/tmp/claude-501/-Volumes-Satechi-TrellisRedux/7650dae1-8f9c-4462-a6f9-f2974ee27db5/scratchpad/\(outName)")
try GLTFExport.writeGLB(to: outURL, positions: baked.vertices, indices: baked.faces,
                        normals: baked.normals, uvs: baked.uvs,
                        baseColorRGBA: (baked.texRGBA, baked.atlasSize, baked.atlasSize))
let bytes = (try Data(contentsOf: outURL)).count
print("[meshbake] wrote \(outURL.path)  (\(bytes) bytes)  in \(String(format: "%.1f", -t0.timeIntervalSinceNow))s")
print("MESHBAKE DONE")
