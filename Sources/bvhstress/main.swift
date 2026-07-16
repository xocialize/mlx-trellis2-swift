import Foundation
import MLX
import MLXMesh

// BVH AABB correctness stress: validates the bottom-up AABB kernel's fence-gated
// visit-counter handoff (BVHAABB.swift). The invariant is exact and total: every
// leaf box equals its triangle's box, every internal box equals the min/max union
// of its children, and the root equals the whole-mesh box. A stale read at the
// gate (the pre-fence hazard: second arrival passing the counter but seeing the
// sibling's zero-init AABB) shows up as an internal box that is not the union of
// its children — most visibly a spurious 0 face slicing through the mesh extent.
// Random triangle soup offset away from the origin so a stale zero is never a
// legitimate coordinate. Repeated at several sizes to give the race scheduling
// room to manifest.

func err(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

// Deterministic LCG so runs are reproducible without seeding MLXRandom.
struct LCG {
    var state: UInt64
    mutating func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(state >> 40) / Float(1 << 24)
    }
}

func randomSoup(faces F: Int, seed: UInt64) -> Mesh {
    var rng = LCG(state: seed)
    var v = [Float](); v.reserveCapacity(F * 9)
    for _ in 0..<(F * 3) {
        // Cluster verts in [10, 20)^3 — strictly positive, away from zero.
        v.append(10 + rng.next() * 10)
        v.append(10 + rng.next() * 10)
        v.append(10 + rng.next() * 10)
    }
    let f = [Int32](0..<Int32(F * 3))
    return Mesh(
        vertices: MLXArray(v, [F * 3, 3]),
        faces: MLXArray(f, [F, 3])
    )
}

var failures = 0

func check(faces F: Int, seed: UInt64, tag: String) {
    let mesh = randomSoup(faces: F, seed: seed)
    let bvh = mesh.bvh()
    MLX.eval(bvh.nodeAABB)

    let aabb = bvh.nodeAABB.asArray(Float.self)          // [N, 6]
    let left = bvh.topology.nodeLeft.asArray(Int32.self)
    let right = bvh.topology.nodeRight.asArray(Int32.self)
    let sorted = bvh.topology.sortedIndices.asArray(Int32.self)
    let verts = mesh.vertices.asArray(Float.self)
    let faceIdx = mesh.faces.asArray(Int32.self)
    let leafOffset = bvh.leafOffset

    // Leaf boxes: exact triangle AABBs.
    var leafBad = 0
    for leaf in 0..<F {
        let n = leaf + leafOffset
        let fi = Int(sorted[leaf])
        var mn = [Float](repeating: .infinity, count: 3)
        var mx = [Float](repeating: -.infinity, count: 3)
        for c in 0..<3 {
            let vi = Int(faceIdx[fi * 3 + c])
            for a in 0..<3 {
                mn[a] = min(mn[a], verts[vi * 3 + a])
                mx[a] = max(mx[a], verts[vi * 3 + a])
            }
        }
        for a in 0..<3 where aabb[n * 6 + a] != mn[a] || aabb[n * 6 + 3 + a] != mx[a] {
            leafBad += 1
        }
    }

    // Internal boxes: exact union of children (same min/max ops as the kernel).
    var innerBad = 0
    var firstBad = -1
    for n in 0..<leafOffset {
        let l = Int(left[n]), r = Int(right[n])
        for a in 0..<6 {
            let op: (Float, Float) -> Float = a < 3 ? min : max
            if aabb[n * 6 + a] != op(aabb[l * 6 + a], aabb[r * 6 + a]) {
                innerBad += 1
                if firstBad < 0 { firstBad = n }
            }
        }
    }

    if leafBad == 0 && innerBad == 0 {
        err("[bvhstress] \(tag) F=\(F) seed=\(seed): OK")
    } else {
        failures += 1
        err("[bvhstress] \(tag) F=\(F) seed=\(seed): FAIL — \(leafBad) leaf, "
            + "\(innerBad) internal mismatches (first bad node \(firstBad))")
    }
}

// Small trees exercise edge topology; large trees give the visit-counter race
// contention. Repetitions matter more than size for catching a stale-read gate.
for seed in 1...4 { check(faces: 1_000, seed: UInt64(seed), tag: "small") }
for seed in 1...4 { check(faces: 100_000, seed: UInt64(seed), tag: "medium") }
for seed in 1...8 { check(faces: 1_000_000, seed: UInt64(seed), tag: "large") }

if failures == 0 {
    err("BVHSTRESS PASS (all AABB trees exact)")
} else {
    err("BVHSTRESS FAIL (\(failures) runs with mismatches)")
    exit(1)
}
