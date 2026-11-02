import Foundation
import MLX
import TRELLIS2

// Two faces share ONE UV triangle but sit in differently coloured voxels: face 0 in the red
// half (z < 0), face 1 in the blue half (z > 0). Keep-first must bake face 0's red everywhere.
let verts: [Float] = [-0.4, -0.4, -0.3,  0.4, -0.4, -0.3,  -0.4, 0.4, -0.3,
                      -0.4, -0.4,  0.3,  0.4, -0.4,  0.3,  -0.4, 0.4,  0.3]
let faces: [Int32] = [0, 1, 2, 3, 4, 5]
let uv: [Float] = [0.1, 0.1, 0.9, 0.1, 0.1, 0.9]
let R = 8
var coords = [Int32](), feats = [Float]()
for x in 0..<R { for y in 0..<R { for z in 0..<R {
    coords += [0, Int32(x), Int32(y), Int32(z)]
    feats += z < R / 2 ? [1, 0, 0] : [0, 0, 1]
} } }
let baked = MeshBake.bakeWithUVs(
    vertices: MLXArray(verts, [6, 3]), faces: MLXArray(faces, [2, 3]), uvs: MLXArray(uv + uv, [6, 2]),
    texFeats: MLXArray(feats, [R * R * R, 3]), texCoords: MLXArray(coords, [R * R * R, 4]),
    fineRes: Float(R), atlasSize: 64, log: { _ in })
var red = 0, blue = 0, other = 0
for p in 0..<(64 * 64) where baked.filled[p] {
    let r = baked.rgba[p * 4], b = baked.rgba[p * 4 + 2]
    if r > 200 && b < 55 { red += 1 } else if b > 200 && r < 55 { blue += 1 } else { other += 1 }
}
let ok = red > 0 && blue == 0 && other == 0
print("bakeWithUVs overlap: red(face 0)=\(red) blue(face 1)=\(blue) other=\(other)  \(ok ? "PASS keep-first" : "FAIL")")
exit(ok ? 0 : 1)
