import Foundation
import MLX

extension Mesh {
    /// Return a new mesh keeping only faces where `mask[i]` is true. Vertices
    /// no longer referenced by any kept face are dropped and face indices are
    /// remapped to the dense `[0..<V')` range. CPU pass for now.
    public func selectingFaces(_ mask: MLXArray) -> Mesh {
        precondition(mask.dim(0) == faceCount, "mask must have length F=\(faceCount), got \(mask.dim(0))")
        let keep = mask.asArray(Bool.self)
        let f = faces.asArray(Int32.self)
        let v = vertices.asArray(Float.self)

        // Walk faces, collect kept indices and the set of referenced vertices.
        var referenced = Array(repeating: false, count: vertexCount)
        var keptFaces: [Int] = []
        keptFaces.reserveCapacity(faceCount)
        for i in 0..<faceCount where keep[i] {
            keptFaces.append(i)
            referenced[Int(f[i*3 + 0])] = true
            referenced[Int(f[i*3 + 1])] = true
            referenced[Int(f[i*3 + 2])] = true
        }

        // Dense remap old vertex idx → new.
        var remap = Array(repeating: Int32(-1), count: vertexCount)
        var newVerts: [Float] = []
        newVerts.reserveCapacity(vertexCount * 3)
        var nextNewIdx: Int32 = 0
        for oldIdx in 0..<vertexCount where referenced[oldIdx] {
            remap[oldIdx] = nextNewIdx
            newVerts.append(v[oldIdx*3 + 0])
            newVerts.append(v[oldIdx*3 + 1])
            newVerts.append(v[oldIdx*3 + 2])
            nextNewIdx += 1
        }

        var newFaces: [Int32] = []
        newFaces.reserveCapacity(keptFaces.count * 3)
        for i in keptFaces {
            newFaces.append(remap[Int(f[i*3 + 0])])
            newFaces.append(remap[Int(f[i*3 + 1])])
            newFaces.append(remap[Int(f[i*3 + 2])])
        }

        let newV = newVerts.isEmpty
            ? MLXArray([] as [Float], [0, 3])
            : MLXArray(newVerts, [newVerts.count / 3, 3])
        let newF = newFaces.isEmpty
            ? MLXArray([] as [Int32], [0, 3])
            : MLXArray(newFaces, [newFaces.count / 3, 3])
        return Mesh(vertices: newV, faces: newF)
    }

    /// Drop faces by mask. `dropMask[i] == true` removes face `i`. Sibling of
    /// `selectingFaces` with the opposite convention. Matches
    /// `cumesh.remove_faces(face_mask)`.
    public func removeFaces(_ dropMask: MLXArray) -> Mesh {
        precondition(dropMask.dim(0) == faceCount,
                     "dropMask must have length F=\(faceCount), got \(dropMask.dim(0))")
        let drop = dropMask.asArray(Bool.self)
        let keep = drop.map { !$0 }
        return selectingFaces(MLXArray(keep, [faceCount]))
    }

    /// Drop faces with near-zero area or with a repeated vertex index.
    ///
    /// `areaEpsilon` is the absolute area threshold. `relativeEpsilon`, when
    /// set (defaults to `nil` for backward compatibility), additionally drops
    /// faces where `area / longest_edge²` is below that ratio. Both gates
    /// must trip for a face to be dropped — this matches CuMesh's
    /// `abs_thresh` + `rel_thresh` AND semantics
    /// (`remove_degenerate_faces` docstring: "considered degenerate if both
    /// conditions are met").
    public func removeDegenerateFaces(
        areaEpsilon: Float = 1e-12,
        relativeEpsilon: Float? = nil
    ) -> Mesh {
        let areas = faceAreas().asArray(Float.self)
        let f = faces.asArray(Int32.self)
        let v = vertices.asArray(Float.self)
        var keep = [Bool](repeating: false, count: faceCount)
        for i in 0..<faceCount {
            let i0 = f[i*3 + 0], i1 = f[i*3 + 1], i2 = f[i*3 + 2]
            if i0 == i1 || i1 == i2 || i0 == i2 { continue }
            let smallAbs = areas[i] <= areaEpsilon
            var smallRel = false
            if let rel = relativeEpsilon {
                let i0i = Int(i0), i1i = Int(i1), i2i = Int(i2)
                func sqLen(_ a: Int, _ b: Int) -> Float {
                    let dx = v[a*3 + 0] - v[b*3 + 0]
                    let dy = v[a*3 + 1] - v[b*3 + 1]
                    let dz = v[a*3 + 2] - v[b*3 + 2]
                    return dx*dx + dy*dy + dz*dz
                }
                let longestSq = max(sqLen(i0i, i1i), max(sqLen(i1i, i2i), sqLen(i2i, i0i)))
                smallRel = longestSq > 0 && (areas[i] / longestSq) <= rel
            } else {
                smallRel = true  // single-gate behaviour when caller didn't pass rel
            }
            keep[i] = !(smallAbs && smallRel)
        }
        return selectingFaces(MLXArray(keep, [faceCount]))
    }

    /// Drop entire connected components whose total face area is below `minArea`.
    /// Match `cumesh.remove_small_connected_components(min_area)`.
    public func removeSmallConnectedComponents(minArea: Float) -> Mesh {
        let table = edgeTable()
        let labels = connectedComponents(edgeTable: table).asArray(Int32.self)
        guard let maxLabel = labels.max() else { return self }
        let nComps = Int(maxLabel) + 1
        let f = faces.asArray(Int32.self)
        let areas = faceAreas().asArray(Float.self)

        var areaPerComp = Array(repeating: Float(0), count: nComps)
        var faceComp = Array(repeating: Int(0), count: faceCount)
        for i in 0..<faceCount {
            // A face belongs to whatever component its vertices do (all 3
            // share the same component since they're edge-connected).
            let c = Int(labels[Int(f[i*3])])
            faceComp[i] = c
            areaPerComp[c] += areas[i]
        }

        var keep = [Bool](repeating: false, count: faceCount)
        for i in 0..<faceCount {
            keep[i] = areaPerComp[faceComp[i]] >= minArea
        }
        return selectingFaces(MLXArray(keep, [faceCount]))
    }

    /// Propagate face winding consistency across each connected component.
    ///
    /// Two adjacent triangles agree on winding iff their shared edge appears
    /// in opposite directions in their respective windings. We DFS from one
    /// face per component, flipping neighbours that disagree. The seed face's
    /// orientation is taken as-is (we don't choose outward vs inward — match
    /// CuMesh, which does the same).
    public func unifyFaceOrientations() -> Mesh {
        let table = edgeTable()
        let f = faces.asArray(Int32.self)
        let edges = table.edges.asArray(Int32.self)
        let stride = UInt64(vertexCount) + 1

        // canonical-key → edge-table index
        var edgeIndex = [UInt64: Int](minimumCapacity: table.count)
        for i in 0..<table.count {
            let lo = UInt64(UInt32(bitPattern: edges[i*2]))
            let hi = UInt64(UInt32(bitPattern: edges[i*2 + 1]))
            edgeIndex[lo * stride + hi] = i
        }

        // For each face, its 3 contributed (edge_idx, sign) pairs.
        // sign = +1 if face traverses lo→hi on this edge, -1 if hi→lo.
        var faceEdges = Array(
            repeating: Array(repeating: (idx: 0, sign: Int8(0)), count: 3),
            count: faceCount
        )
        // For each edge in the table, list of incident (face, sign).
        var edgeFaces = Array(repeating: [(face: Int, sign: Int8)](), count: table.count)

        for fi in 0..<faceCount {
            for k in 0..<3 {
                let a = UInt32(bitPattern: f[fi*3 + k])
                let b = UInt32(bitPattern: f[fi*3 + (k + 1) % 3])
                let sign: Int8 = a < b ? 1 : -1
                let lo = min(a, b), hi = max(a, b)
                let key = UInt64(lo) * stride + UInt64(hi)
                let eIdx = edgeIndex[key]!
                faceEdges[fi][k] = (eIdx, sign)
                edgeFaces[eIdx].append((fi, sign))
            }
        }

        // DFS from each unvisited face. flip[fi] = true means reverse winding.
        var flip = [Bool](repeating: false, count: faceCount)
        var visited = [Bool](repeating: false, count: faceCount)
        var stack: [Int] = []
        for seed in 0..<faceCount where !visited[seed] {
            visited[seed] = true
            stack.append(seed)
            while let cur = stack.popLast() {
                let curFlipFactor: Int8 = flip[cur] ? -1 : 1
                for (eIdx, curOrigSign) in faceEdges[cur] {
                    let curEffective = curOrigSign * curFlipFactor
                    for (neighbour, nOrigSign) in edgeFaces[eIdx] where neighbour != cur && !visited[neighbour] {
                        visited[neighbour] = true
                        // Want neighbour's effective sign to be opposite of cur's.
                        // neighbourEffective = nOrigSign * (flip[neighbour] ? -1 : 1)
                        // Solve for flip[neighbour]:
                        flip[neighbour] = (nOrigSign == curEffective)
                        stack.append(neighbour)
                    }
                }
            }
        }

        // Apply flips by swapping the last two indices of each flipped face.
        var newFaces = f
        for fi in 0..<faceCount where flip[fi] {
            newFaces.swapAt(fi*3 + 1, fi*3 + 2)
        }
        return Mesh(
            vertices: vertices,
            faces: MLXArray(newFaces, [faceCount, 3])
        )
    }

    /// Fill boundary loops whose perimeter is below `maxHolePerimeter` by
    /// triangulating them as a fan from a newly-inserted centroid vertex.
    ///
    /// The new triangles' winding matches the existing face winding around
    /// the hole (boundary loops are already oriented consistently with faces).
    /// Fan from centroid is the simplest correct triangulation; CuMesh's
    /// version may produce different geometry but the topology — i.e. "the
    /// loop is now closed" — matches.
    ///
    /// Pass `.infinity` to fill all holes regardless of size.
    public func fillHoles(maxHolePerimeter: Float) -> Mesh {
        let loops = boundaryLoops()
        if loops.count == 0 { return self }

        let v = vertices.asArray(Float.self)
        let f = faces.asArray(Int32.self)
        let offs = loops.offsets.asArray(Int32.self)
        let loopVerts = loops.vertices.asArray(Int32.self)

        var newVerts = v
        var newFaces = f
        var nextVertIdx = Int32(vertexCount)

        for li in 0..<loops.count {
            let start = Int(offs[li]), end = Int(offs[li + 1])
            let n = end - start
            if n < 3 { continue }

            // Perimeter: sum of |V_{i+1} - V_i| for i in 0..<n (wrapping).
            var perim: Float = 0
            for i in 0..<n {
                let a = Int(loopVerts[start + i])
                let b = Int(loopVerts[start + (i + 1) % n])
                let dx = v[a*3 + 0] - v[b*3 + 0]
                let dy = v[a*3 + 1] - v[b*3 + 1]
                let dz = v[a*3 + 2] - v[b*3 + 2]
                perim += (dx*dx + dy*dy + dz*dz).squareRoot()
            }
            if perim > maxHolePerimeter { continue }

            // Centroid of the loop's vertices.
            var cx: Float = 0, cy: Float = 0, cz: Float = 0
            for i in 0..<n {
                let a = Int(loopVerts[start + i])
                cx += v[a*3 + 0]; cy += v[a*3 + 1]; cz += v[a*3 + 2]
            }
            cx /= Float(n); cy /= Float(n); cz /= Float(n)
            newVerts.append(cx); newVerts.append(cy); newVerts.append(cz)
            let center = nextVertIdx
            nextVertIdx += 1

            // Boundary loop walks a→b in the same direction faces traverse
            // their boundary edge; the hole sits to the right. The new
            // triangle filling the hole reverses that: (b, a, center).
            for i in 0..<n {
                let a = loopVerts[start + i]
                let b = loopVerts[start + (i + 1) % n]
                newFaces.append(b)
                newFaces.append(a)
                newFaces.append(center)
            }
        }

        let newVCount = newVerts.count / 3
        let newFCount = newFaces.count / 3
        return Mesh(
            vertices: MLXArray(newVerts, [newVCount, 3]),
            faces: MLXArray(newFaces, [newFCount, 3])
        )
    }

    /// Drop faces whose canonical (sorted) vertex triple has already been seen.
    /// Treats `(0,1,2)`, `(1,2,0)`, `(2,1,0)` as the same triangle regardless of
    /// winding — match `cumesh.remove_duplicate_faces` semantics.
    public func removeDuplicateFaces() -> Mesh {
        struct FaceKey: Hashable { let a, b, c: Int32 }
        let f = faces.asArray(Int32.self)
        var seen = Set<FaceKey>()
        seen.reserveCapacity(faceCount)
        var keep = [Bool](repeating: false, count: faceCount)
        for i in 0..<faceCount {
            let s = [f[i*3 + 0], f[i*3 + 1], f[i*3 + 2]].sorted()
            let key = FaceKey(a: s[0], b: s[1], c: s[2])
            if seen.insert(key).inserted {
                keep[i] = true
            }
        }
        return selectingFaces(MLXArray(keep, [faceCount]))
    }

    /// Drop vertices that no face references. Matches
    /// `cumesh.remove_unreferenced_vertices`. Equivalent to
    /// `selectingFaces(all-true)` since that already drops orphans.
    public func removeUnreferencedVertices() -> Mesh {
        selectingFaces(MLXArray([Bool](repeating: true, count: faceCount), [faceCount]))
    }

    /// For each non-manifold edge (>2 incident faces), keep only the first 2
    /// incident faces. Drops the surplus. Matches
    /// `cumesh.remove_non_manifold_faces`. "First" is by face index.
    public func removeNonManifoldFaces() -> Mesh {
        let f = faces.asArray(Int32.self)
        let stride = UInt64(vertexCount) + 1

        // Canonical edge key → ordered list of incident face indices.
        var edgeFaces = [UInt64: [Int]](minimumCapacity: faceCount * 3)
        for fi in 0..<faceCount {
            let i0 = UInt64(UInt32(bitPattern: f[fi*3 + 0]))
            let i1 = UInt64(UInt32(bitPattern: f[fi*3 + 1]))
            let i2 = UInt64(UInt32(bitPattern: f[fi*3 + 2]))
            for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
                let lo = min(a, b), hi = max(a, b)
                edgeFaces[lo * stride + hi, default: []].append(fi)
            }
        }

        var keep = [Bool](repeating: true, count: faceCount)
        for (_, list) in edgeFaces where list.count > 2 {
            for k in 2..<list.count { keep[list[k]] = false }
        }
        return selectingFaces(MLXArray(keep, [faceCount]))
    }

    /// Resolve non-manifold edges (>2 incident faces) by duplicating the edge's
    /// two endpoint vertices for each face beyond the first two. The duplicated
    /// vertices share the original's position; topology becomes manifold at the
    /// cost of new boundary edges where the splits happened.
    ///
    /// Matches `cumesh.repair_non_manifold_edges` semantics. Note that a
    /// vertex on multiple non-manifold edges may be duplicated several times.
    public func repairNonManifoldEdges() -> Mesh {
        let f = faces.asArray(Int32.self)
        var newVerts = vertices.asArray(Float.self)
        var newFaces = f
        let stride = UInt64(vertexCount) + 1

        // Canonical edge key → ordered list of (face, slot) where slot is the
        // 0..<3 corner of the face whose outgoing edge is this one.
        var edgeFaces = [UInt64: [(face: Int, slot: Int)]](minimumCapacity: faceCount * 3)
        for fi in 0..<faceCount {
            for k in 0..<3 {
                let a = UInt64(UInt32(bitPattern: f[fi*3 + k]))
                let b = UInt64(UInt32(bitPattern: f[fi*3 + (k + 1) % 3]))
                let lo = min(a, b), hi = max(a, b)
                edgeFaces[lo * stride + hi, default: []].append((fi, k))
            }
        }

        for (_, list) in edgeFaces where list.count > 2 {
            // Leave the first 2 incident faces using the original endpoints;
            // duplicate for every face from index 2 onward.
            for k in 2..<list.count {
                let (fi, slot) = list[k]
                let aIdx = Int(newFaces[fi*3 + slot])
                let bIdx = Int(newFaces[fi*3 + (slot + 1) % 3])
                let newA = Int32(newVerts.count / 3)
                newVerts.append(newVerts[aIdx*3 + 0])
                newVerts.append(newVerts[aIdx*3 + 1])
                newVerts.append(newVerts[aIdx*3 + 2])
                let newB = Int32(newVerts.count / 3)
                newVerts.append(newVerts[bIdx*3 + 0])
                newVerts.append(newVerts[bIdx*3 + 1])
                newVerts.append(newVerts[bIdx*3 + 2])
                newFaces[fi*3 + slot] = newA
                newFaces[fi*3 + (slot + 1) % 3] = newB
            }
        }

        return Mesh(
            vertices: MLXArray(newVerts, [newVerts.count / 3, 3]),
            faces: MLXArray(newFaces, [faceCount, 3])
        )
    }
}
