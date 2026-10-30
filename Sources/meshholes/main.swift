import Foundation
import MLX
import MLXMesh

// Boundary-loop and hole-fill gate (AB-T-0386). `boundaryLoops()` once kept a single
// outgoing boundary edge per vertex and stopped a walk at any visited vertex, so two
// holes sharing a vertex came out as one closed loop plus an open chain, and
// `fillHoles` fanned the chain into new boundaries (Hi3D dress: 190 loops -> 1,782).
//
// Part 1 is synthetic: a 5x5 grid with two triangular holes that share one vertex
// must give 3 loops (outer + 2 holes), and filling the holes must leave the outer only.
// Two faces that wind the same way round a shared edge must not count as boundary.
// Part 2 (optional): `meshholes <verts.npy> <faces.npy>` runs the CuMesh-style chain
// and reports loops before and after; it fails with more than 5 loops left.

setvbuf(stdout, nil, _IONBF, 0)
var allPass = true
func gate(_ name: String, _ ok: Bool, _ detail: String) {
    allPass = allPass && ok
    print("HOLES \(name): \(detail)  \(ok ? "PASS" : "FAIL")")
}

func grid(n: Int, dropping drop: Set<Int>) -> Mesh {
    var v = [Float](), f = [Int32]()
    for y in 0..<n { for x in 0..<n { v += [Float(x), Float(y), 0] } }
    var tri = 0
    for y in 0..<(n - 1) {
        for x in 0..<(n - 1) {
            let i0 = Int32(y * n + x), i1 = i0 + 1, i2 = i0 + Int32(n), i3 = i2 + 1
            if !drop.contains(tri) { f += [i0, i1, i3] }
            if !drop.contains(tri + 1) { f += [i0, i3, i2] }
            tri += 2
        }
    }
    return Mesh(vertices: MLXArray(v, [n * n, 3]), faces: MLXArray(f, [f.count / 3, 3]))
}

func loopLengths(_ m: Mesh) -> [Int] {
    let offs = m.boundaryLoops().offsets.asArray(Int32.self)
    return (0..<(offs.count - 1)).map { Int(offs[$0 + 1] - offs[$0]) }.sorted()
}

// ---- 1) two holes sharing vertex (2,2) of a 5x5 grid ----
// Quad (x,y) holds triangles 2*(4y+x) = (x,y),(x+1,y),(x+1,y+1) and 2*(4y+x)+1.
// Dropping the first triangle of quads (1,1) and (2,2) leaves holes that touch only at (2,2).
let shared = grid(n: 5, dropping: [2 * (4 * 1 + 1), 2 * (4 * 2 + 2)])
let lens = loopLengths(shared)
gate("shared vertex  loops", lens == [3, 3, 16], "lengths \(lens) (want [3, 3, 16])")
let filled = shared.fillHoles(maxHolePerimeter: 4)
let filledLens = loopLengths(filled)
gate("shared vertex  fill", filledLens == [16] && filled.numBoundaryEdges == 16,
     "after fill \(filledLens), \(filled.numBoundaryEdges) boundary edges (want [16], 16)")

// ---- 2) a single hole, and an untouched grid ----
let single = grid(n: 5, dropping: [2 * (4 * 1 + 1)])
gate("single hole", loopLengths(single) == [3, 16] && loopLengths(single.fillHoles(maxHolePerimeter: 4)) == [16],
     "loops \(loopLengths(single))")
gate("no holes", loopLengths(grid(n: 5, dropping: [])) == [16], "loops \(loopLengths(grid(n: 5, dropping: [])))")

// ---- 3) a flipped neighbour: both faces walk the diagonal 2->0, so it is interior ----
// The square's outline is one loop although its edges don't all run the same way, and
// filling it must close it.
let flipped = Mesh(vertices: MLXArray([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0] as [Float], [4, 3]),
                   faces: MLXArray([0, 1, 2, 0, 3, 2] as [Int32], [2, 3]))
let flippedFilled = flipped.fillHoles(maxHolePerimeter: .infinity)
gate("flipped neighbour", loopLengths(flipped) == [4] && flippedFilled.numBoundaryEdges == 0,
     "loops \(loopLengths(flipped)), \(flippedFilled.numBoundaryEdges) boundary edges after fill (want [4], 0)")

// ---- 4) optional real mesh: the AB-T-0386 repro chain ----
let args = CommandLine.arguments
if args.count >= 3 {
    var m = Mesh(vertices: try loadArray(url: URL(fileURLWithPath: args[1])).asType(.float32),
                 faces: try loadArray(url: URL(fileURLWithPath: args[2])).asType(.int32))
    let t = Date()
    let before = (m.faceCount, m.numBoundaryLoops, m.numBoundaryEdges)
    m = m.removeDuplicateFaces().removeSmallConnectedComponents(minArea: 2e-4).fillHoles(maxHolePerimeter: 0.05)
    let after = (m.faceCount, m.numBoundaryLoops, m.numBoundaryEdges)
    gate("mesh chain", after.1 <= 5, String(format: "%d faces, %d loops, %d boundary edges -> %d faces, %d loops, %d boundary edges (%.1fs)",
                                            before.0, before.1, before.2, after.0, after.1, after.2, -t.timeIntervalSinceNow))
    if args.count >= 4 {
        try save(array: m.vertices, url: URL(fileURLWithPath: "\(args[3])/verts_filled.npy"))
        try save(array: m.faces, url: URL(fileURLWithPath: "\(args[3])/faces_filled.npy"))
    }
}

print("HOLES GATE \(allPass ? "PASS" : "FAIL")")
exit(allPass ? 0 : 1)
