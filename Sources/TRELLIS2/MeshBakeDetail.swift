import Foundation
import MLX
import MLXMesh

/// Opt-in detail maps for `MeshBake.run`: a tangent-space normal map baked from the raw dual-grid
/// surface onto the decimated, unwrapped mesh, and an ambient-occlusion map ray-cast on the output
/// mesh. Both are off by default (`.none`), and a bake with them off is byte-identical to one
/// without this type.
///
/// Distances are fractions of the output mesh's bounding-box diagonal, so they mean the same at any
/// resolution tier.
///
/// Neither map trusts face winding. The raw dual-grid shell's winding is not consistent, so its
/// normals are treated as lines and turned to the texel's side. The output mesh is wound
/// consistently only per chart (the double-sided material hides it in a viewer), so AO casts both
/// hemispheres of each texel and keeps the less occluded one: the inner side of a thin shell or a
/// closed body is ~fully occluded at short range.
public struct DetailMapOptions: Sendable, Equatable {
    /// Normal-map edge in texels; nil = no normal map.
    public var normalMapSize: Int?
    /// Ambient-occlusion map edge in texels; nil = no AO map.
    public var aoMapSize: Int?
    /// Cosine-weighted hemisphere rays per AO texel and side (each texel casts twice this).
    public var aoRays: Int
    /// AO ray length (× bbox diagonal). Geometry farther than this does not occlude.
    public var aoMaxDistance: Float
    /// AO ray origin lift and minimum hit distance (× bbox diagonal), against self-hits.
    public var aoBias: Float
    /// AO = 1 − strength · (hit fraction), clamped to [0, 1].
    public var aoStrength: Float
    /// Normal-map cage (× bbox diagonal): each texel casts from this far out along its normal, back
    /// across the surface, and takes the outermost raw-surface hit within twice the distance.
    public var normalCage: Float

    public init(normalMapSize: Int? = nil, aoMapSize: Int? = nil, aoRays: Int = 64,
                aoMaxDistance: Float = 0.25, aoBias: Float = 0.002, aoStrength: Float = 1,
                normalCage: Float = 0.01) {
        self.normalMapSize = normalMapSize
        self.aoMapSize = aoMapSize
        self.aoRays = aoRays
        self.aoMaxDistance = aoMaxDistance
        self.aoBias = aoBias
        self.aoStrength = aoStrength
        self.normalCage = normalCage
    }

    public static let none = DetailMapOptions()
    public var isEmpty: Bool { normalMapSize == nil && aoMapSize == nil }
}

/// What the detail-map bake produced (all nil when nothing was asked for).
public struct BakedDetailMaps {
    /// [V,4] glTF TANGENT: unit xyz along +u, w = ±1 handedness (bitangent = w · N × T).
    public var tangents: MLXArray?
    /// RGBA8 tangent-space normal map (glTF convention: +X right, +Y up in the image, +Z out).
    public var normalRGBA: [UInt8]?
    public var normalSize = 0
    /// RGBA8 occlusion map (AO in R; G and B copy it so it reads as gray), when it is not packed
    /// into the metallic-roughness texture.
    public var occlusionRGBA: [UInt8]?
    public var occlusionSize = 0
    /// AO was written into the metallic-roughness texture's R channel (glTF ORM packing).
    public var occlusionInMetallicRoughness = false

    public init() {}
}

/// Detail-map sub-stage timings and diagnostics (seconds; each window ends on an eval or CPU work).
public struct DetailMapMetrics: Codable, Sendable {
    public var tangentsS = 0.0
    public var normalMapS = 0.0
    public var normalMapTexels = 0
    public var normalMapRayHitFrac: Float = 0   // texels whose cage ray hit the raw surface
    public var normalMapFlatFrac: Float = 0     // texels that fell back to the flat normal
    public var aoMapS = 0.0
    public var aoMapTexels = 0
    public var aoMean: Float = 0                // mean AO over covered texels (1 = unoccluded)
    public init() {}
}

extension BakedMesh {
    /// The same bake with positions, normals and tangents re-expressed in another frame.
    /// `positions` maps vertex positions, `directions` maps unit vectors (normals, tangent xyz);
    /// both must be the same proper rotation (a reflection would flip the tangent handedness and the
    /// winding). Texture data is carried unchanged.
    public func mapped(positions: (MLXArray) -> MLXArray, directions: (MLXArray) -> MLXArray) -> BakedMesh {
        var out = BakedMesh(vertices: positions(vertices), faces: faces, normals: directions(normals), uvs: uvs,
                            texRGBA: texRGBA, atlasSize: atlasSize, coverage: coverage, mrRGBA: mrRGBA,
                            filledBeforeInpaint: filledBeforeInpaint, wallFlipped: wallFlipped)
        var d = detail
        if let t = d.tangents {
            d.tangents = MLX.concatenated([directions(t[0..., 0..<3]), t[0..., 3..<4]], axis: 1)
        }
        out.detail = d
        return out
    }
}

extension MeshBake {
    /// Per-vertex tangents (Lengyel's method) for glTF: per-face dP/du accumulated onto the face's
    /// vertices, Gram–Schmidt against the vertex normal, handedness from the image-up bitangent.
    ///
    /// glTF UVs have v pointing down the image, and a glTF normal map's +Y is image-up, so the
    /// bitangent the map's green channel measures is −dP/dv; `w` is the sign of
    /// dot(N × T, −dP/dv). Sums run on the CPU in face order (deterministic, as `vertexNormals`).
    /// Vertices whose UV derivatives degenerate get any unit vector perpendicular to the normal.
    public static func vertexTangents(vertices: MLXArray, faces: MLXArray, uvs: MLXArray,
                                      normals: MLXArray) -> MLXArray {
        let p = vertices.asType(.float32).asArray(Float.self)
        let f = faces.asType(.int32).asArray(Int32.self)
        let t = uvs.asType(.float32).asArray(Float.self)
        let n = normals.asType(.float32).asArray(Float.self)
        let V = vertices.dim(0), F = faces.dim(0)
        var tacc = [Float](repeating: 0, count: V * 3)
        var bacc = [Float](repeating: 0, count: V * 3)
        for fi in 0 ..< F {
            let i0 = Int(f[fi*3]), i1 = Int(f[fi*3+1]), i2 = Int(f[fi*3+2])
            let e1 = (p[i1*3] - p[i0*3], p[i1*3+1] - p[i0*3+1], p[i1*3+2] - p[i0*3+2])
            let e2 = (p[i2*3] - p[i0*3], p[i2*3+1] - p[i0*3+1], p[i2*3+2] - p[i0*3+2])
            let du1 = t[i1*2] - t[i0*2], dv1 = t[i1*2+1] - t[i0*2+1]
            let du2 = t[i2*2] - t[i0*2], dv2 = t[i2*2+1] - t[i0*2+1]
            let det = du1 * dv2 - du2 * dv1
            guard abs(det) > 1e-20 else { continue }
            // Unnormalized (scaled by 1/det): the sum weights each face by its UV-space footprint ratio,
            // and the sign of det orients the face's tangent correctly for mirrored charts.
            let r = 1 / det
            let tx = (dv2 * e1.0 - dv1 * e2.0) * r, ty = (dv2 * e1.1 - dv1 * e2.1) * r, tz = (dv2 * e1.2 - dv1 * e2.2) * r
            let bx = (du1 * e2.0 - du2 * e1.0) * r, by = (du1 * e2.1 - du2 * e1.1) * r, bz = (du1 * e2.2 - du2 * e1.2) * r
            for v in [i0, i1, i2] {
                tacc[v*3] += tx; tacc[v*3+1] += ty; tacc[v*3+2] += tz
                bacc[v*3] += bx; bacc[v*3+1] += by; bacc[v*3+2] += bz
            }
        }
        var out = [Float](repeating: 0, count: V * 4)
        for v in 0 ..< V {
            var nx = n[v*3], ny = n[v*3+1], nz = n[v*3+2]
            let nl = (nx*nx + ny*ny + nz*nz).squareRoot()
            if nl > 1e-20 { nx /= nl; ny /= nl; nz /= nl } else { nx = 0; ny = 0; nz = 1 }
            var tx = tacc[v*3], ty = tacc[v*3+1], tz = tacc[v*3+2]
            let d = tx*nx + ty*ny + tz*nz
            tx -= nx * d; ty -= ny * d; tz -= nz * d
            var tl = (tx*tx + ty*ty + tz*tz).squareRoot()
            if tl < 1e-12 {   // degenerate: any perpendicular
                if abs(nx) < 0.9 { tx = 0; ty = -nz; tz = ny } else { tx = nz; ty = 0; tz = -nx }
                tl = (tx*tx + ty*ty + tz*tz).squareRoot()
            }
            tx /= tl; ty /= tl; tz /= tl
            // N × T against the image-up bitangent −dP/dv
            let cx = ny*tz - nz*ty, cy = nz*tx - nx*tz, cz = nx*ty - ny*tx
            let s = -(cx * bacc[v*3] + cy * bacc[v*3+1] + cz * bacc[v*3+2])
            out[v*4] = tx; out[v*4+1] = ty; out[v*4+2] = tz; out[v*4+3] = s < 0 ? -1 : 1
        }
        return MLXArray(out, [V, 4])
    }

    /// Texel samples of a UV layout at `size`²: the centre pass plus four half-texel-jittered passes
    /// (the main bake's conservative rasterization), reduced to one sample per texel, first write
    /// wins (centre beats jitter, lower face beats higher). Returns texel indices (y·size + x), and
    /// each winner's face id and barycentrics.
    static func uniqueTexelSamples(uvs: MLXArray, faces: MLXArray, size: Int)
        -> (texel: [Int32], faceId: MLXArray, bary: MLXArray) {
        let uvPix = uvs * Float(size)
        var texel = [Int32]()
        var fids: [MLXArray] = [], barys: [MLXArray] = []
        var taken = [Bool](repeating: false, count: size * size)
        // One pass alive at a time: each pass keeps only its winners, so the peak is one pass plus the
        // winners so far, not all five passes and their concatenation (at 8192² that was +13 GB).
        for (dx, dy): (Float, Float) in [(0, 0), (0.49, 0.49), (-0.49, 0.49), (0.49, -0.49), (-0.49, -0.49)] {
            var keep = [Int32]()
            do {
                let jittered = dx == 0 && dy == 0 ? uvPix
                    : MLX.stacked([uvPix[0..., 0] + dx, uvPix[0..., 1] + dy], axis: 1)
                let (p, f, b) = UVRasterize.rasterize(uvsPix: jittered, faces: faces, textureSize: size)
                let px = p.asArray(Int32.self)
                for k in 0 ..< f.dim(0) {
                    let x = Int(px[k*2]), y = Int(px[k*2+1])
                    if x < 0 || x >= size || y < 0 || y >= size { continue }
                    let q = y * size + x
                    if taken[q] { continue }
                    taken[q] = true
                    texel.append(Int32(q)); keep.append(Int32(k))
                }
                if !keep.isEmpty {
                    let kp = MLXArray(keep)
                    let fk = f.take(kp, axis: 0), bk = b.take(kp, axis: 0)
                    MLX.eval(fk, bk)
                    fids.append(fk); barys.append(bk)
                }
            }
        }
        guard !fids.isEmpty else { return ([], MLXArray.zeros([0], dtype: .int32), MLXArray.zeros([0, 3])) }
        let fid = fids.count == 1 ? fids[0] : MLX.concatenated(fids, axis: 0)
        let bary = barys.count == 1 ? barys[0] : MLX.concatenated(barys, axis: 0)
        MLX.eval(fid, bary)
        return (texel, fid, bary)
    }

    /// Bake the opt-in detail maps for a finished unwrap. `rawMesh`/`rawBVH` are the raw dual-grid
    /// surface (the high-detail source for the normal map); `mrRGBA`, when present and the same
    /// size as the AO map, receives AO in its R channel instead of a separate texture.
    static func bakeDetailMaps(
        mesh: Mesh, normals: MLXArray, uvs: MLXArray,
        rawMesh: Mesh, rawBVH: BVH,
        mrRGBA: inout [UInt8]?, atlasSize: Int,
        options o: DetailMapOptions,
        log: (String) -> Void
    ) -> (BakedDetailMaps, DetailMapMetrics) {
        var out = BakedDetailMaps(), m = DetailMapMetrics()
        guard !o.isEmpty else { return (out, m) }
        let (bbMin, bbMax) = mesh.boundingBox()
        let diag = MLX.sqrt(((bbMax - bbMin) * (bbMax - bbMin)).sum()).item(Float.self)

        // Unit vertex normals as written to the GLB; the tangent frame and the AO hemisphere use them.
        let unitN = normals / MLX.maximum(MLX.sqrt((normals * normals).sum(axis: 1, keepDims: true)), MLXArray(1e-20 as Float))

        if let size = o.normalMapSize {
            var t = Date()
            let tangents = vertexTangents(vertices: mesh.vertices, faces: mesh.faces, uvs: uvs, normals: unitN)
            MLX.eval(tangents)
            m.tangentsS = -t.timeIntervalSinceNow
            out.tangents = tangents
            t = Date()
            let rawN = unorientedVertexNormals(rawMesh)
            let ss = 2, ssSize = size * ss
            let (texel, fid, bary) = uniqueTexelSamples(uvs: uvs, faces: mesh.faces, size: ssSize)
            let K = texel.count
            var ssN = [Float](repeating: 0, count: ssSize * ssSize * 3)
            var ssFilled = [Bool](repeating: false, count: ssSize * ssSize)
            let cage = o.normalCage * diag
            var hitCount = 0, flatCount = 0
            let chunk = 1 << 20   // per-dispatch texel count: keeps each GPU command buffer short
            for k0 in stride(from: 0, to: K, by: chunk) {
                let k1 = min(K, k0 + chunk)
                let fc = fid[k0 ..< k1], bc = bary[k0 ..< k1]
                let fv = mesh.faces.take(fc, axis: 0)
                func interp(_ a: MLXArray) -> MLXArray {
                    bc[0..., 0..<1] * a.take(fv[0..., 0], axis: 0) + bc[0..., 1..<2] * a.take(fv[0..., 1], axis: 0)
                        + bc[0..., 2..<3] * a.take(fv[0..., 2], axis: 0)
                }
                let P = interp(mesh.vertices)
                let Nl = normalized(interp(unitN))
                let Tl = interp(tangents[0..., 0..<3])
                let Wl = MLX.where(interp(tangents[0..., 3..<4]) .< 0, MLXArray(-1 as Float), MLXArray(1 as Float))
                // cage ray: from outside the surface, back along −N; the closest hit is the outermost
                let origin = P + Nl * cage
                let hits = rawBVH.rayHits(mesh: rawMesh, origins: origin, directions: -Nl, tMax: 2 * cage)
                let hit = hits.faceIndices .>= 0
                let cp = rawBVH.closestPoints(mesh: rawMesh, queries: P)   // fallback for misses
                let tHit = MLX.where(hit, hits.t, MLXArray(0 as Float)).expandedDimensions(axis: 1)
                let hp = MLX.where(hit.expandedDimensions(axis: 1), origin - Nl * tHit, cp.points)
                let hf = MLX.where(hit, hits.faceIndices, cp.faceIndices)
                // high normal: raw vertex normals interpolated at the hit's barycentrics
                let rf = rawMesh.faces.take(hf, axis: 0)
                let a = rawMesh.vertices.take(rf[0..., 0], axis: 0)
                let b = rawMesh.vertices.take(rf[0..., 1], axis: 0)
                let c = rawMesh.vertices.take(rf[0..., 2], axis: 0)
                let (w0, w1, w2) = barycentrics(hp, a, b, c)
                let Nh = normalized(w0 * rawN.take(rf[0..., 0], axis: 0) + w1 * rawN.take(rf[0..., 1], axis: 0)
                                    + w2 * rawN.take(rf[0..., 2], axis: 0))
                // texel TBN: Gram–Schmidt T against the interpolated N, B = w · N × T
                let T = normalized(Tl - Nl * (Nl * Tl).sum(axis: 1, keepDims: true))
                let B = Wl * MLX.cross(Nl, T, axis: -1)
                // The raw shell's winding is not consistent, so its normals are lines, not directions:
                // take the one on the texel's side.
                let side = MLX.where((Nh * Nl).sum(axis: 1, keepDims: true) .< 0, MLXArray(-1 as Float), MLXArray(1 as Float))
                let Nho = Nh * side
                let nz = (Nho * Nl).sum(axis: 1)
                var ts = normalized(MLX.stacked([(Nho * T).sum(axis: 1), (Nho * B).sum(axis: 1), nz], axis: 1))
                // a raw normal near-perpendicular to the texel's (a side wall or crevice the ray or the
                // fallback grabbed) would bake a wrong normal; the flat one is the safe answer
                let flat = nz .< Self.normalMapMinCos
                ts = MLX.where(flat.expandedDimensions(axis: 1), MLXArray([0, 0, 1] as [Float]), ts)
                MLX.eval(ts, hit, flat)
                hitCount += hit.asType(.int32).sum().item(Int.self)
                flatCount += flat.asType(.int32).sum().item(Int.self)
                let v = ts.asArray(Float.self)
                for k in k0 ..< k1 {
                    let q = Int(texel[k]), j = (k - k0) * 3
                    ssN[q*3] = v[j]; ssN[q*3+1] = v[j+1]; ssN[q*3+2] = v[j+2]
                    ssFilled[q] = true
                }
            }
            // box-average 2× down, renormalize, inpaint the gutters, encode
            var nb = [Float](repeating: 0, count: size * size * 3)
            var filled = [Bool](repeating: false, count: size * size)
            for y in 0 ..< size {
                for x in 0 ..< size {
                    var sx: Float = 0, sy: Float = 0, sz: Float = 0, cnt = 0
                    for dy in 0 ..< ss {
                        for dx in 0 ..< ss {
                            let q = (y * ss + dy) * ssSize + (x * ss + dx)
                            if ssFilled[q] { sx += ssN[q*3]; sy += ssN[q*3+1]; sz += ssN[q*3+2]; cnt += 1 }
                        }
                    }
                    guard cnt > 0 else { continue }
                    let l = (sx*sx + sy*sy + sz*sz).squareRoot()
                    let p = y * size + x
                    if l > 1e-6 { nb[p*3] = sx / l; nb[p*3+1] = sy / l; nb[p*3+2] = sz / l } else { nb[p*3+2] = 1 }
                    filled[p] = true
                }
            }
            if !filled.contains(true) { for p in 0 ..< size * size { nb[p*3+2] = 1 } }
            dilateInpaint(&nb, &filled, size, iterations: size)
            var rgba = [UInt8](repeating: 255, count: size * size * 4)
            for p in 0 ..< size * size {
                var x = nb[p*3], y = nb[p*3+1], z = nb[p*3+2]
                let l = (x*x + y*y + z*z).squareRoot()   // inpaint averages: renormalize
                if l > 1e-6 { x /= l; y /= l; z /= l } else { x = 0; y = 0; z = 1 }
                rgba[p*4] = unorm8(x * 0.5 + 0.5); rgba[p*4+1] = unorm8(y * 0.5 + 0.5); rgba[p*4+2] = unorm8(z * 0.5 + 0.5)
            }
            out.normalRGBA = rgba
            out.normalSize = size
            m.normalMapS = -t.timeIntervalSinceNow
            m.normalMapTexels = K
            m.normalMapRayHitFrac = Float(hitCount) / Float(max(K, 1))
            m.normalMapFlatFrac = Float(flatCount) / Float(max(K, 1))
            log("  normal map \(size)²: \(K) sub-texels, cage-ray hits \(String(format: "%.3f", m.normalMapRayHitFrac)), flat fallback \(String(format: "%.4f", m.normalMapFlatFrac)) (\(String(format: "%.1f", m.tangentsS + m.normalMapS))s)")
        }

        if let size = o.aoMapSize {
            let t = Date()
            let bvh = mesh.bvh()
            let (texel, fid, bary) = uniqueTexelSamples(uvs: uvs, faces: mesh.faces, size: size)
            let K = texel.count
            let faceN = mesh.faceNormals()
            let bias = o.aoBias * diag, tMax = o.aoMaxDistance * diag
            var ao = [Float](repeating: 0, count: size * size * 3)
            var filled = [Bool](repeating: false, count: size * size)
            var sum: Double = 0
            // rays per dispatch ≈ 2^24: one command buffer stays well under the GPU watchdog
            let chunk = max(1, (1 << 24) / max(1, o.aoRays))
            for k0 in stride(from: 0, to: K, by: chunk) {
                let k1 = min(K, k0 + chunk)
                let fc = fid[k0 ..< k1], bc = bary[k0 ..< k1]
                let fv = mesh.faces.take(fc, axis: 0)
                func interp(_ a: MLXArray) -> MLXArray {
                    bc[0..., 0..<1] * a.take(fv[0..., 0], axis: 0) + bc[0..., 1..<2] * a.take(fv[0..., 1], axis: 0)
                        + bc[0..., 2..<3] * a.take(fv[0..., 2], axis: 0)
                }
                let P = interp(mesh.vertices)
                let Ns = normalized(interp(unitN))
                let Fg = faceN.take(fc, axis: 0)
                let Ng = MLX.where(((Fg * Ns).sum(axis: 1, keepDims: true)) .< 0, -Fg, Fg)   // same side as Ns
                // Both hemispheres; the less occluded one is the outside. The winding can't say which side
                // is out: xatlas re-normalizes it per chart, so whole charts face inward, and the
                // double-sided material shades them right anyway. The inner side is the one that sees
                // the other wall of a thin shell, or the inside of a closed body, at short range: ~fully
                // occluded either way.
                let occFront = bvh.occlusion(mesh: mesh, origins: P + Ng * bias, shadingNormals: Ns, geomNormals: Ng,
                                             rays: o.aoRays, tMin: bias, tMax: tMax)
                let occBack = bvh.occlusion(mesh: mesh, origins: P - Ng * bias, shadingNormals: -Ns, geomNormals: -Ng,
                                            rays: o.aoRays, tMin: bias, tMax: tMax)
                let occ = MLX.minimum(occFront, occBack)
                let v = MLX.clip(1 - o.aoStrength * occ, min: 0, max: 1)
                MLX.eval(v)
                let a = v.asArray(Float.self)
                for k in k0 ..< k1 {
                    let q = Int(texel[k]), x = a[k - k0]
                    ao[q*3] = x; ao[q*3+1] = x; ao[q*3+2] = x
                    filled[q] = true
                    sum += Double(x)
                }
            }
            if !filled.contains(true) { ao = [Float](repeating: 1, count: size * size * 3) }
            dilateInpaint(&ao, &filled, size, iterations: size)
            if var mr = mrRGBA, size == atlasSize, mr.count == size * size * 4 {
                for p in 0 ..< size * size { mr[p*4] = unorm8(ao[p*3]) }   // ORM: R = occlusion
                mrRGBA = mr
                out.occlusionInMetallicRoughness = true
            } else {
                var rgba = [UInt8](repeating: 255, count: size * size * 4)
                for p in 0 ..< size * size {
                    let g = unorm8(ao[p*3])
                    rgba[p*4] = g; rgba[p*4+1] = g; rgba[p*4+2] = g
                }
                out.occlusionRGBA = rgba
            }
            out.occlusionSize = size
            m.aoMapS = -t.timeIntervalSinceNow
            m.aoMapTexels = K
            m.aoMean = Float(sum / Double(max(K, 1)))
            log("  AO map \(size)²: \(K) texels × 2×\(o.aoRays) rays, mean AO \(String(format: "%.3f", m.aoMean))\(out.occlusionInMetallicRoughness ? " (packed into metallicRoughness R)" : "") (\(String(format: "%.1f", m.aoMapS))s)")
        }
        return (out, m)
    }

    /// Below this cosine between the raw and the low-poly normal (≈ 75°) the normal map falls back to
    /// flat: past it the matched raw surface is rarely the same surface.
    static let normalMapMinCos: Float = 0.25

    /// Smooth vertex normals for a mesh whose winding is not consistent (the raw dual-grid shell):
    /// each face's area-weighted normal is added with the sign that agrees with the first face seen at
    /// that vertex, so neighbours wound oppositely reinforce instead of cancelling. The result is a
    /// line per vertex (its sign arbitrary); callers orient it. CPU, face order: deterministic.
    static func unorientedVertexNormals(_ mesh: Mesh) -> MLXArray {
        let fn = mesh.faceNormals(normalize: false).asArray(Float.self)
        let f = mesh.faces.asType(.int32).asArray(Int32.self)
        let V = mesh.vertexCount
        var acc = [Float](repeating: 0, count: V * 3)
        var ref = [Float](repeating: 0, count: V * 3)
        var seen = [Bool](repeating: false, count: V)
        for fi in 0 ..< mesh.faceCount {
            let nx = fn[fi*3], ny = fn[fi*3+1], nz = fn[fi*3+2]
            for c in 0 ..< 3 {
                let v = Int(f[fi*3 + c])
                if !seen[v] { seen[v] = true; ref[v*3] = nx; ref[v*3+1] = ny; ref[v*3+2] = nz }
                let s: Float = nx * ref[v*3] + ny * ref[v*3+1] + nz * ref[v*3+2] < 0 ? -1 : 1
                acc[v*3] += s * nx; acc[v*3+1] += s * ny; acc[v*3+2] += s * nz
            }
        }
        return normalized(MLXArray(acc, [V, 3]))
    }

    private static func normalized(_ a: MLXArray) -> MLXArray {
        a / MLX.maximum(MLX.sqrt((a * a).sum(axis: 1, keepDims: true)), MLXArray(1e-12 as Float))
    }

    /// Barycentric weights of points `p` (projected onto each triangle's plane), [K,1] each.
    private static func barycentrics(_ p: MLXArray, _ a: MLXArray, _ b: MLXArray, _ c: MLXArray)
        -> (MLXArray, MLXArray, MLXArray) {
        let v0 = b - a, v1 = c - a, v2 = p - a
        let d00 = (v0 * v0).sum(axis: 1, keepDims: true), d01 = (v0 * v1).sum(axis: 1, keepDims: true)
        let d11 = (v1 * v1).sum(axis: 1, keepDims: true)
        let d20 = (v2 * v0).sum(axis: 1, keepDims: true), d21 = (v2 * v1).sum(axis: 1, keepDims: true)
        let den = d00 * d11 - d01 * d01
        let safe = MLX.where(MLX.abs(den) .< 1e-30, MLXArray(1e-30 as Float), den)
        var v = (d11 * d20 - d01 * d21) / safe
        var w = (d00 * d21 - d01 * d20) / safe
        // a closest point sits inside its triangle; clamp the float noise
        v = MLX.clip(v, min: 0, max: 1); w = MLX.clip(w, min: 0, max: 1)
        let s = MLX.maximum(v + w, MLXArray(1 as Float))
        v = v / s; w = w / s
        return (1 - v - w, v, w)
    }

    @inline(__always) static func unorm8(_ x: Float) -> UInt8 {
        UInt8(max(0, min(255, Int(x * 255 + 0.5))))
    }
}
