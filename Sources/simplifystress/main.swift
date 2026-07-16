import Foundation
import MLX
import MLXMesh

// Luby independent-set correctness stress (tie-break total order in
// SimplifyIndependentSet.swift). The kernel's win test compares priorities
// `-cost + jitter`; if two ADJACENT edges have exactly equal priority,
// neither sees the other as strictly greater and — without the
// (priority, edge_id) lexicographic tie-break — both win the same round,
// breaking the independent-set property and corrupting the single-step
// collapse remap downstream.
//
// The jitter hash cannot be tie-engineered through seeds: xorshift of
// `e ^ seed*K` is linear over GF(2), so `jitter_e ^ jitter_f` depends only
// on `e ^ f` — whether two edge ids collide in the low 24 bits is a FIXED
// property of the id pair (256 "bad deltas" in 2^32), identical for every
// seed and round. That makes the hazard persistent when it exists at all,
// and it means this harness must engineer ties through the `cost` input
// instead: pick adjacent edge pairs (e, f), replicate the kernel's jitter
// arithmetic CPU-side, and set cost[e] = 0, cost[f] = jitter_f - jitter_e
// so the two priorities are bit-exactly equal (verified in fp32 before
// planting). All other edges get cost 1, making each planted pair the local
// maximum — pre-fix, both edges of every pair win round 0; post-fix, the
// lower edge id must win alone.
//
// Part 2 simplifies a coplanar grid aggressively and requires the result to
// stay edge-manifold (this gate also covers the link-condition guard in
// Simplify.swift, which it originally smoked out).

func err(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

func coplanarGrid(n: Int) -> Mesh {
    var v = [Float](); v.reserveCapacity(n * n * 3)
    for y in 0..<n {
        for x in 0..<n {
            v.append(Float(x)); v.append(Float(y)); v.append(0)
        }
    }
    var f = [Int32](); f.reserveCapacity((n - 1) * (n - 1) * 6)
    for y in 0..<(n - 1) {
        for x in 0..<(n - 1) {
            let i0 = Int32(y * n + x), i1 = i0 + 1
            let i2 = i0 + Int32(n), i3 = i2 + 1
            f.append(contentsOf: [i0, i1, i3])
            f.append(contentsOf: [i0, i3, i2])
        }
    }
    return Mesh(
        vertices: MLXArray(v, [n * n, 3]),
        faces: MLXArray(f, [(n - 1) * (n - 1) * 2, 3])
    )
}

// Exact CPU replica of the kernel's jitter arithmetic.
@inline(__always)
func jitterFloat(edge e: Int, seed: Int32) -> Float {
    var h = UInt32(truncatingIfNeeded: e) ^ (UInt32(bitPattern: seed) &* 2654435761)
    h ^= h << 13; h ^= h >> 17; h ^= h << 5
    return Float(h & 0xffffff) * (1.0 / 16777216.0) * 1e-6
}

var failures = 0

// ---- Part 1: engineered exact-tie independent-set invariant ---------------

let n = 256
let mesh = coplanarGrid(n: n)
let table = mesh.edgeTable()
let Q = mesh.vertexQuadrics()
let (pipelineCost, _) = mesh.edgeCollapseCosts(quadrics: Q, edgeTable: table)
let vea = mesh.vertexEdgeAdjacency(edgeTable: table)

let E = table.count
let pipelineCostArr = pipelineCost.asArray(Float.self)
let edgesArr = table.edges.asArray(Int32.self)
let veaOff = vea.offsets.asArray(Int32.self)
let veaEdges = vea.edgeIndices.asArray(Int32.self)
print("grid \(n)x\(n): \(mesh.faceCount) faces, \(E) edges")

func otherEndpoint(_ e: Int, _ v: Int) -> Int {
    let a = Int(edgesArr[e * 2]), b = Int(edgesArr[e * 2 + 1])
    return a == v ? b : a
}

var tieEvents = 0
for tieIter in 0..<16 {
    let seed = Int32(truncatingIfNeeded: tieIter &* 1_000_003)  // round-0 seed

    // Default: everyone expensive; boundary edges stay protected like the
    // real pipeline (+inf).
    var costArr = [Float](repeating: 1, count: E)
    for e in 0..<E where !pipelineCostArr[e].isFinite { costArr[e] = .infinity }

    // Plant vertex-disjoint tie pairs at spaced interior vertices. Accept a
    // pair only when the kernel's fp32 arithmetic provably ties:
    // r_e = -0 + je == je and r_f = -(jf - je) + jf must equal je bit-exactly.
    var used = [Bool](repeating: false, count: mesh.vertexCount)
    var pairs: [(lo: Int, hi: Int)] = []
    outer: for y in stride(from: 2, to: n - 2, by: 5) {
        for x in stride(from: 2, to: n - 2, by: 5) {
            if pairs.count >= 128 { break outer }
            let v = y * n + x
            if used[v] { continue }
            let incident = (Int(veaOff[v])..<Int(veaOff[v + 1])).map { Int(veaEdges[$0]) }
            var planted = false
            for (i, e) in incident.enumerated() where !planted && costArr[e].isFinite {
                for f in incident[(i + 1)...] where !planted && costArr[f].isFinite {
                    let ve = otherEndpoint(e, v), vf = otherEndpoint(f, v)
                    if used[ve] || used[vf] { continue }
                    let je = jitterFloat(edge: e, seed: seed)
                    let jf = jitterFloat(edge: f, seed: seed)
                    let c = jf - je
                    guard je > 0, (-c) + jf == je else { continue }
                    costArr[e] = 0
                    costArr[f] = c
                    used[v] = true; used[ve] = true; used[vf] = true
                    pairs.append((lo: min(e, f), hi: max(e, f)))
                    planted = true
                }
            }
        }
    }
    tieEvents += pairs.count

    let mask = mesh.selectIndependentEdges(
        edgeTable: table, cost: MLXArray(costArr, [E]), vertexEdgeAdj: vea, iter: tieIter
    )
    let sel = mask.asArray(Bool.self)

    // Exact IS invariant: no two selected edges share a vertex.
    var seen = Set<Int32>()
    var shared = 0
    var count = 0
    for e in 0..<E where sel[e] {
        count += 1
        if !seen.insert(edgesArr[e * 2]).inserted { shared += 1 }
        if !seen.insert(edgesArr[e * 2 + 1]).inserted { shared += 1 }
    }
    // Per pair: the tie must resolve to exactly the lower edge id.
    var doubleWins = 0
    var wrongWinner = 0
    for p in pairs {
        switch (sel[p.lo], sel[p.hi]) {
        case (true, true): doubleWins += 1
        case (false, true): wrongWinner += 1
        case (false, false): wrongWinner += 1
        case (true, false): break  // correct: lower id wins the exact tie
        }
    }
    if shared > 0 || doubleWins > 0 || wrongWinner > 0 {
        failures += 1
        err("FAIL [tie] iter=\(tieIter): \(pairs.count) planted pairs -> \(doubleWins) double wins, \(wrongWinner) wrong winners, \(shared) shared vertices among \(count) selected")
    } else {
        print("ok   [tie] iter=\(tieIter): \(pairs.count) exact ties all resolved to lower id, \(count) selected, independent")
    }
}
print("engineered \(tieEvents) exact-tie events total")
if tieEvents == 0 {
    failures += 1
    err("FAIL [tie]: no ties planted — harness lost its teeth")
}

// Baseline: the real pipeline cost (all zeros on a coplanar grid) across
// several seeds must also yield independent sets.
for it in 0..<8 {
    let mask = mesh.selectIndependentEdges(
        edgeTable: table, cost: pipelineCost, vertexEdgeAdj: vea, iter: it
    )
    let sel = mask.asArray(Bool.self)
    var seen = Set<Int32>()
    var shared = 0
    var count = 0
    for e in 0..<E where sel[e] {
        count += 1
        if !seen.insert(edgesArr[e * 2]).inserted { shared += 1 }
        if !seen.insert(edgesArr[e * 2 + 1]).inserted { shared += 1 }
    }
    if shared > 0 {
        failures += 1
        err("FAIL [baseline] iter=\(it): \(shared) shared vertices among \(count) selected edges")
    } else {
        print("ok   [baseline] iter=\(it): \(count) selected, independent")
    }
}

// ---- Part 2: aggressive coplanar simplify stays manifold ------------------

let g = coplanarGrid(n: 256)
let startFaces = g.faceCount
let simplified = g.simplify(targetNumFaces: startFaces / 20)
let t2 = simplified.edgeTable()
let nonManifold = t2.edgeFaceCount.asArray(Int32.self).filter { $0 > 2 }.count
print("simplify 256x256: \(startFaces) -> \(simplified.faceCount) faces, \(nonManifold) non-manifold edges")
if simplified.faceCount == 0 || simplified.faceCount > startFaces / 10 {
    failures += 1
    err("FAIL [simplify]: insufficient reduction (\(startFaces) -> \(simplified.faceCount), wanted <= \(startFaces / 10))")
}
if nonManifold > 0 {
    failures += 1
    err("FAIL [simplify]: \(nonManifold) non-manifold edges after aggressive coplanar collapse")
}

if failures > 0 {
    err("simplifystress: \(failures) failure(s)")
    exit(1)
}
print("simplifystress: all checks passed")
