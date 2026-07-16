import Foundation
import MLX
import TRELLIS2

// SW7 FDG GATE — vendored mesh->flexible-dual-grid converter vs the Python oracle.
// Oracle fixtures are lex-sorted by voxel coords (x,y,z); we sort our output the same way
// (voxel SET is order-independent — sparse convs downstream are permutation-equivariant).

let goldens = "/Volumes/Satechi/TrellisRedux/trellis2-port/goldens"
func golden(_ name: String) throws -> MLXArray {
    try loadArray(url: URL(fileURLWithPath: "\(goldens)/\(name).npy"))
}

let verts = try golden("fdg_in_vertices").asType(.float32)
let faces = try golden("fdg_in_faces").asType(.int32)
let vertsArr: [Float] = verts.reshaped([-1]).asArray(Float.self)
let facesArr: [Int32] = faces.reshaped([-1]).asArray(Int32.self)

var allPass = true
for res in [64, 256] {
    let t0 = Date()
    let r = try DualGridConvert.meshToFlexibleDualGrid(
        vertices: vertsArr, faces: facesArr, gridSize: res)
    let dt = Date().timeIntervalSince(t0)

    // lex-sort rows by (x, y, z)
    let n = r.coords.shape[0]
    let c: [Int32] = r.coords.reshaped([-1]).asArray(Int32.self)
    let d: [Float] = r.dualVertices.reshaped([-1]).asArray(Float.self)
    let iRaw: [Bool] = r.intersected.reshaped([-1]).asArray(Bool.self)
    let order = (0..<n).sorted { a, b in
        (c[a*3], c[a*3+1], c[a*3+2]) < (c[b*3], c[b*3+1], c[b*3+2])
    }

    let gc: [Int32] = try golden("fdg_out_coords_\(res)").reshaped([-1]).asArray(Int32.self)
    let gd: [Float] = try golden("fdg_out_dual_\(res)").reshaped([-1]).asArray(Float.self)
    let gi: [UInt8] = try golden("fdg_out_inter_\(res)").reshaped([-1]).asArray(UInt8.self)

    var coordsExact = gc.count == n * 3
    var interExact = coordsExact
    var maxDual: Float = 0
    if coordsExact {
        for (row, src) in order.enumerated() {
            for a in 0..<3 {
                if c[src*3+a] != gc[row*3+a] { coordsExact = false }
                if (iRaw[src*3+a] ? 1 : 0) != gi[row*3+a] { interExact = false }
                maxDual = max(maxDual, abs(d[src*3+a] - gd[row*3+a]))
            }
        }
    }
    let pass = coordsExact && interExact && maxDual < 1e-5
    allPass = allPass && pass
    print(String(format: "SW7 FDG res %d: N=%d (oracle %d)  coords %@  intersected %@  max|dual Δ|=%.2e  %.2fs  %@",
                 res, n, gc.count / 3,
                 coordsExact ? "EXACT" : "MISMATCH",
                 interExact ? "EXACT" : "MISMATCH",
                 maxDual, dt, pass ? "PASS" : "FAIL"))
}
// Pipeline-scale speed check (no fixture — oracle count for this torus at 1024 is 2,504,632).
let t0 = Date()
let r1024 = try DualGridConvert.meshToFlexibleDualGrid(vertices: vertsArr, faces: facesArr, gridSize: 1024)
let n1024 = r1024.coords.shape[0]
let countOK = n1024 == 2_504_632
allPass = allPass && countOK
print(String(format: "SW7 FDG res 1024: N=%d (oracle 2504632)  %.2fs  %@",
             n1024, Date().timeIntervalSince(t0), countOK ? "PASS" : "FAIL"))

print("SW7 FDG GATE \(allPass ? "PASS" : "FAIL")")
exit(allPass ? 0 : 1)
