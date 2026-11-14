import Foundation
import MLX

// Pixal3D (TencentARC, SIGGRAPH 2026; MIT, code @ f7cf384, weights @ b0cb2e1b) — projection
// conditioning for the TRELLIS.2 flows (AB-T-0485). Each Pixal3D flow block adds
// `proj_linear(proj)` to its cross-attention output, where `proj` is the image feature map
// back-projected through a camera onto every voxel token, and cross-attends to the 5 global DINOv3
// tokens (CLS + 4 registers) only. Ported from `pixal3d/trainers/flow_matching/mixins/
// image_conditioned_proj.py` (ProjGrid, ProjGridMV, compute_relative_calc_mat,
// DinoV3ProjFeatureExtractor{,MultiView}) and `pixal3d/pipelines/pixal3d_image_to_3d.py`.

/// A 4×4 matrix, row-major, in Double (camera maths stays on the host).
public struct Mat4: Sendable, Equatable {
    public var m: [Double]                         // 16, row-major
    public init(_ rows: [[Double]]) { m = rows.flatMap { $0 } }
    public init(flat: [Double]) { precondition(flat.count == 16); m = flat }
    public subscript(r: Int, c: Int) -> Double { m[r * 4 + c] }
    public static let identity = Mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    public static func * (a: Mat4, b: Mat4) -> Mat4 {
        var o = [Double](repeating: 0, count: 16)
        for r in 0..<4 { for c in 0..<4 { var s = 0.0; for k in 0..<4 { s += a[r, k] * b[k, c] }; o[r * 4 + c] = s } }
        return Mat4(flat: o)
    }

    /// General 4×4 inverse (Gauss–Jordan with partial pivoting).
    public var inverse: Mat4 {
        var a = m, inv = Mat4.identity.m
        for col in 0..<4 {
            var piv = col
            for r in col..<4 where abs(a[r * 4 + col]) > abs(a[piv * 4 + col]) { piv = r }
            if piv != col {
                for c in 0..<4 { a.swapAt(col * 4 + c, piv * 4 + c); inv.swapAt(col * 4 + c, piv * 4 + c) }
            }
            let d = a[col * 4 + col]
            for c in 0..<4 { a[col * 4 + c] /= d; inv[col * 4 + c] /= d }
            for r in 0..<4 where r != col {
                let f = a[r * 4 + col]
                if f == 0 { continue }
                for c in 0..<4 { a[r * 4 + c] -= f * a[col * 4 + c]; inv[r * 4 + c] -= f * inv[col * 4 + c] }
            }
        }
        return Mat4(flat: inv)
    }

    /// Pixal3D's canonical front camera (camera-to-world): camera at (0, −d, 0) looking +Y, Z up.
    /// `ProjGrid.front_view_transform_matrix` with `[1,3] = −distance`.
    public static func front(distance d: Double) -> Mat4 {
        Mat4([[1, 0, 0, 0], [0, 0, -1, -d], [0, 1, 0, 0], [0, 0, 0, 1]])
    }

    /// Translation norm (`norm(c2w[:3, 3])`, the multi-view distance).
    public var translationNorm: Double { (self[0, 3] * self[0, 3] + self[1, 3] * self[1, 3] + self[2, 3] * self[2, 3]).squareRoot() }
}

/// One view's camera. Single view: `fovX` + `meshScale`, distance from the closed form, frame = the
/// canonical front. Multi-view: each view's camera-to-world `c2w` (Z-up world, camera looks down its
/// own −Z with +Y up — Blender/NeRF); the views are placed relative to view 0 snapped to the front.
public struct Pixal3DCamera: Sendable, Equatable {
    /// Horizontal field of view, radians (upstream `camera_angle_x`). Pixal3D's default here: 20°.
    public var fovX: Double
    /// Object scale against the unit cube (upstream constant 1.0).
    public var meshScale: Double
    /// Camera-to-world (multi-view only; nil = the canonical front at `distance`).
    public var c2w: Mat4?

    public static let defaultFovDegrees = 20.0

    public init(fovX: Double = Pixal3DCamera.defaultFovDegrees * .pi / 180, meshScale: Double = 1.0, c2w: Mat4? = nil) {
        self.fovX = fovX; self.meshScale = meshScale; self.c2w = c2w
    }

    /// `distance = 0.5 / (mesh_scale · tan(fov/2))` (upstream `distance_from_fov`, extend_pixel 0): the
    /// distance at which the unit cube's mid-plane fills the image width. Multi-view: `|c2w.t|`.
    public var distance: Double {
        if let c2w { return c2w.translationNorm }
        return 0.5 / (meshScale * tan(fovX / 2))
    }

    /// The per-view projection matrices (`calc_mat_i = F(d0) · inv(C_0) · C_i`), view 0 = F exactly.
    /// For a single camera without `c2w`, `[F(distance)]`.
    public static func calcMats(_ cams: [Pixal3DCamera]) -> [Mat4] {
        guard let c0 = cams.first else { return [] }
        guard let C0 = c0.c2w else {
            precondition(cams.count == 1, "Pixal3D multi-view needs c2w on every view")
            return [Mat4.front(distance: c0.distance)]
        }
        let F = Mat4.front(distance: C0.translationNorm)
        let invC0 = C0.inverse
        return cams.map { cam in
            guard let Ci = cam.c2w else { preconditionFailure("Pixal3D multi-view needs c2w on every view") }
            return F * (invC0 * Ci)
        }
    }

    /// Maps the canonical front frame (where the mesh comes out) back into the cameras' own world:
    /// `C_0 · inv(F(d0))` — rigid. Identity for single view.
    public static func frontToWorld(_ cams: [Pixal3DCamera]) -> Mat4 {
        guard let C0 = cams.first?.c2w else { return .identity }
        return C0 * Mat4.front(distance: C0.translationNorm).inverse
    }
}

/// The four Pixal3D conditioning stages (`inference.py` IMAGE_COND_CONFIGS).
public enum Pixal3DStage: String, Sendable, CaseIterable {
    case ss, shape512, shape1024, tex1024

    /// DINOv3 input size (and the projection's image resolution).
    public var imageSize: Int { self == .ss || self == .shape512 ? 512 : 1024 }
    /// NAF output size (nil: no NAF — the SS stage projects DINOv3 alone, 1024 channels).
    public var nafTarget: Int? {
        switch self {
        case .ss: return nil
        case .shape512, .shape1024: return 512
        case .tex1024: return 1024
        }
    }
    /// proj_in_channels.
    public var projChannels: Int { nafTarget == nil ? 1024 : 2048 }
}

/// Camera back-projection of voxel tokens (ProjGrid / ProjGridMV) and bilinear feature sampling.
public enum Pixal3DProjection {
    /// Normalized grid_sample coordinates [K, 2] (x, y) for voxel `coords` [K, 4] (b, x, y, z) on an
    /// R³ grid, seen through camera-to-world `calcMat` with `fovX` at `imageSize`:
    ///   grid point  g = linspace(−1, 1, R)[x, y, z]       (corner-aligned)
    ///   world       p = (g.x, −g.z, g.y) / (2·mesh_scale)
    ///   camera      c = inv(calcMat) · [p, 1]
    ///   pixel       u = f·c.x / (−c.z + 1e−8) + res/2,  v = −f·c.y / (−c.z + 1e−8) + res/2,
    ///               f = (16 / tan(fov/2)) · res / 32
    ///   normalized  (u + 0.5) / res · 2 − 1               (grid_sample, align_corners = False)
    public static func normalizedCoords(coords: MLXArray, grid R: Int, calcMat: Mat4, fovX: Double,
                                        meshScale: Double, imageSize res: Int) -> MLXArray {
        let ijk = coords[0..., 1...].asType(.float32)                        // [K,3]
        let step = Float(2.0 / Double(R - 1))
        let g = ijk * step - 1                                               // linspace(-1,1,R)
        let s = Float(1.0 / (2.0 * meshScale))
        let px = g[0..., 0] * s, py = -g[0..., 2] * s, pz = g[0..., 1] * s
        let w2c = calcMat.inverse
        func row(_ r: Int) -> MLXArray {
            Float(w2c[r, 0]) * px + Float(w2c[r, 1]) * py + Float(w2c[r, 2]) * pz + Float(w2c[r, 3])
        }
        let xc = row(0), yc = row(1), zc = row(2)
        let f = Float(16.0 / tan(fovX / 2.0) * Double(res) / 32.0)
        let denom = -zc + Float(1e-8)
        let u = f * xc / denom + Float(res) / 2
        let v = -(f * yc / denom) + Float(res) / 2
        let nu = (u + 0.5) / Float(res) * 2 - 1
        let nv = (v + 0.5) / Float(res) * 2 - 1
        return MLX.stacked([nu, nv], axis: 1)                                // [K,2]
    }

    /// All R³ grid coords in ProjGrid order (meshgrid 'ij' over x, y, z → x slowest), [R³, 4].
    public static func denseCoords(grid R: Int) -> MLXArray {
        var out = [Int32](); out.reserveCapacity(R * R * R * 4)
        for x in 0..<R { for y in 0..<R { for z in 0..<R { out += [0, Int32(x), Int32(y), Int32(z)] } } }
        return MLXArray(out).reshaped([R * R * R, 4])
    }

    /// Corner pixels and weights of F.grid_sample(bilinear, align_corners=False, padding_mode="border")
    /// at points `norm` [K,2] (x, y) on an H×W map.
    public struct Corners { public let x0, x1, y0, y1, w00, w01, w10, w11: MLXArray }
    public static func bilinearCorners(_ norm: MLXArray, H: Int, W: Int) -> Corners {
        // unnormalize (align_corners=False) then clip to the border
        let ix = MLX.clip(((norm[0..., 0] + 1) * Float(W) - 1) / 2, min: 0, max: Float(W - 1))
        let iy = MLX.clip(((norm[0..., 1] + 1) * Float(H) - 1) / 2, min: 0, max: Float(H - 1))
        let x0f = MLX.floor(ix), y0f = MLX.floor(iy)
        let wx1 = ix - x0f, wy1 = iy - y0f
        let wx0 = 1 - wx1, wy0 = 1 - wy1
        let x0 = x0f.asType(.int32), y0 = y0f.asType(.int32)
        let x1 = MLX.minimum(x0 + 1, MLXArray(Int32(W - 1))), y1 = MLX.minimum(y0 + 1, MLXArray(Int32(H - 1)))
        return Corners(x0: x0, x1: x1, y0: y0, y1: y1, w00: wx0 * wy0, w01: wx1 * wy0, w10: wx0 * wy1, w11: wx1 * wy1)
    }

    /// F.grid_sample(map, grid, bilinear, align_corners=False, padding_mode="border") at points
    /// `norm` [K,2] (x, y) on a channel-last map [H, W, C]. Returns [K, C].
    public static func sampleBilinear(_ map: MLXArray, _ norm: MLXArray) -> MLXArray {
        let H = map.dim(0), W = map.dim(1), C = map.dim(2)
        let c = bilinearCorners(norm, H: H, W: W)
        let flat = map.reshaped([H * W, C])
        func at(_ y: MLXArray, _ x: MLXArray, _ w: MLXArray) -> MLXArray {
            flat.take(y * Int32(W) + x, axis: 0) * w.reshaped([-1, 1])
        }
        return at(c.y0, c.x0, c.w00) + at(c.y0, c.x1, c.w01) + at(c.y1, c.x0, c.w10) + at(c.y1, c.x1, c.w11)
    }

    /// Pixal3D's HR re-quantization of the decoder's upsampled coords (`run()`,
    /// pixal3d_image_to_3d.py:713-723): `((c + 0.5) / lr · (grid − 1)).round()` (torch.round =
    /// half-to-even), then unique (lexicographic). Upstream's token back-off: while the count is ≥
    /// `maxNumTokens` and hr > 1024, hr −= 128. `hardMaxTokens` keeps backing off past 1024 like
    /// stock's `CascadeQuantize` (AB-A-0262 bound) — nil = upstream exactly.
    public static func requantize(upCoords: MLXArray, lrResolution: Int = 512, targetResolution: Int,
                                  maxNumTokens: Int = 49_152, floorResolution: Int = 1024,
                                  hardMaxTokens: Int? = nil) -> (coords: MLXArray, hrResolution: Int) {
        let M = upCoords.dim(0)
        let c = upCoords.asType(.int32).asArray(Int32.self)
        let lr = Float(lrResolution)
        var hr = targetResolution
        while true {
            let n = Float(hr / 16 - 1)
            var keys = Set<Int64>(minimumCapacity: M)
            for m in 0..<M {
                let b = Int64(c[m * 4])
                let x = Int64(((Float(c[m * 4 + 1]) + 0.5) / lr * n).rounded(.toNearestOrEven))
                let y = Int64(((Float(c[m * 4 + 2]) + 0.5) / lr * n).rounded(.toNearestOrEven))
                let z = Int64(((Float(c[m * 4 + 3]) + 0.5) / lr * n).rounded(.toNearestOrEven))
                keys.insert(b << 48 | x << 32 | y << 16 | z)
            }
            let underCap = keys.count < maxNumTokens
            let atFloor = hr <= floorResolution
            let overHardCap = hardMaxTokens.map { keys.count >= $0 } ?? false
            if underCap || (atFloor && (!overHardCap || hr - 128 < lrResolution)) {
                let sorted = keys.sorted()
                var out = [Int32](); out.reserveCapacity(sorted.count * 4)
                for k in sorted {
                    out.append(Int32(truncatingIfNeeded: k >> 48))
                    out.append(Int32(truncatingIfNeeded: (k >> 32) & 0xFFFF))
                    out.append(Int32(truncatingIfNeeded: (k >> 16) & 0xFFFF))
                    out.append(Int32(truncatingIfNeeded: k & 0xFFFF))
                }
                return (MLXArray(out).reshaped([sorted.count, 4]), hr)
            }
            hr -= 128
        }
    }
}

/// One input view for the Pixal3D conditioner: the preprocessed image at 512² and 1024², each
/// ImageNet-normalized (DINOv3 input) and raw [0,1] (NAF guide), NCHW [1,3,S,S], plus its camera.
/// DINOv3 tokens are cached per size on first use.
public final class Pixal3DView {
    public let normalized512: MLXArray, guide512: MLXArray
    public let normalized1024: MLXArray?, guide1024: MLXArray?
    public let camera: Pixal3DCamera
    var tokens: [Int: MLXArray] = [:]

    public init(normalized512: MLXArray, guide512: MLXArray, normalized1024: MLXArray?, guide1024: MLXArray?,
                camera: Pixal3DCamera) {
        self.normalized512 = normalized512; self.guide512 = guide512
        self.normalized1024 = normalized1024; self.guide1024 = guide1024; self.camera = camera
    }

    func normalized(_ size: Int) -> MLXArray {
        size == 512 ? normalized512 : (normalized1024 ?? { preconditionFailure("Pixal3D view lacks 1024² input") }())
    }
    func guide(_ size: Int) -> MLXArray {
        size == 512 ? guide512 : (guide1024 ?? { preconditionFailure("Pixal3D view lacks 1024² guide") }())
    }
}

/// Builds Pixal3D's per-stage conditioning (`z_global`, `z_proj`) from one or more posed views.
/// Multi-view fusion is upstream's "average": projected features and global tokens are averaged over
/// the views, with no visibility test.
public struct Pixal3DConditioner {
    public let dino: DINOv3
    public let naf: NAF

    public init(dino: DINOv3, naf: NAF) { self.dino = dino; self.naf = naf }

    /// The DINOv3 tokens [1, 5 + P, 1024] of a view at a size (cached on the view).
    public func tokens(_ view: Pixal3DView, size: Int) -> MLXArray {
        if let t = view.tokens[size] { return t }
        let t = dino(view.normalized(size)); MLX.eval(t)
        view.tokens[size] = t
        return t
    }

    /// The DINOv3 patch map [L, L, 1024] of a token set.
    public static func patchMap(_ tokens: MLXArray) -> MLXArray {
        let P = tokens.dim(1) - 5
        let L = Int(Double(P).squareRoot())
        return tokens[0, 5...].reshaped([L, L, tokens.dim(2)])
    }

    /// One view's projected features for `stage` at `coords` [K,4] on an R³ grid → [K, proj channels].
    /// `nafMap` may be passed in (the gates feed the golden one); nil computes it.
    public func project(_ view: Pixal3DView, stage: Pixal3DStage, calcMat: Mat4, coords: MLXArray, grid R: Int,
                        nafMap: MLXArray? = nil) -> MLXArray {
        let res = stage.imageSize
        let tok = tokens(view, size: res)
        let norm = Pixal3DProjection.normalizedCoords(coords: coords, grid: R, calcMat: calcMat,
                                                      fovX: view.camera.fovX, meshScale: view.camera.meshScale,
                                                      imageSize: res)
        let lr = Pixal3DProjection.sampleBilinear(Self.patchMap(tok), norm)
        guard let target = stage.nafTarget else { return lr }
        let hr: MLXArray
        if let nafMap {
            hr = Pixal3DProjection.sampleBilinear(nafMap, norm)
        } else {
            // banded: the HR map is never materialized (NAF.sample)
            let q = naf.queries(guide: view.guide(res), target: target); MLX.eval(q)
            hr = naf.sample(queries: q, features: Self.patchMap(tok).transposed(2, 0, 1).expandedDimensions(axis: 0), at: norm)
        }
        return MLX.concatenated([lr, hr], axis: 1)
    }

    /// Stage conditioning over all views: global [1,5,1024] (mean over views) and proj [K, P] (mean).
    /// `coords` nil = the dense R³ grid (the SS stage).
    public func condition(_ views: [Pixal3DView], stage: Pixal3DStage, coords: MLXArray?, grid R: Int)
        -> (global: MLXArray, proj: MLXArray) {
        precondition(!views.isEmpty)
        let calc = Pixal3DCamera.calcMats(views.map(\.camera))
        let pts = coords ?? Pixal3DProjection.denseCoords(grid: R)
        var projAcc: MLXArray? = nil
        var globAcc: MLXArray? = nil
        for (i, v) in views.enumerated() {
            let p = project(v, stage: stage, calcMat: calc[i], coords: pts, grid: R)
            MLX.eval(p)
            projAcc = projAcc.map { $0 + p } ?? p
            let g = tokens(v, size: stage.imageSize)[0..., 0..<5]
            globAcc = globAcc.map { $0 + g } ?? g
            MLX.eval(projAcc!, globAcc!)
        }
        let n = Float(views.count)
        let proj = views.count == 1 ? projAcc! : projAcc! / n
        let glob = views.count == 1 ? globAcc! : globAcc! / n
        MLX.eval(proj, glob)
        return (glob, proj)
    }
}
