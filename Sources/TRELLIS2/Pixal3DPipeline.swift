import Foundation
import Metal
import MLX
import MLXRandom

/// Pixal3D flow checkpoints (converted names, `Tools/pixal3d/convert_weights.py`).
public enum Pixal3DFlowKind: Hashable, Sendable {
    case shape512(multiView: Bool), shape1024(multiView: Bool), tex1024(multiView: Bool)

    public var fileStem: String {
        switch self {
        case .shape512(let mv): return "pixal3d_shape_flow_512" + (mv ? "_mv" : "")
        case .shape1024(let mv): return "pixal3d_shape_flow_1024" + (mv ? "_mv" : "")
        case .tex1024(let mv): return "pixal3d_tex_flow_1024" + (mv ? "_mv" : "")
        }
    }
    public static func structStem(multiView: Bool) -> String { "pixal3d_struct_flow" + (multiView ? "_mv" : "") }
}

/// Which frame the Pixal3D mesh is returned in (AB-T-0485 P2 orientation decision, README
/// "Pixal3D mode"). Pixal3D's decoded mesh lives in its voxel frame: +Y is the photo's up, the
/// camera sits on +Z. The package always rotates that into Pixal3D's own world frame — Z-up, the
/// canonical front camera at (0, −d, 0) looking +Y — which is the axis convention of the stock
/// TRELLIS.2 output, so `yUp`, the bake and every Z-up consumer read it the same way.
public enum Pixal3DFrame: String, Sendable {
    /// The canonical front frame: view 0 is the front camera. Up is the photo's up, not gravity — a
    /// product shot taken from above comes out tilted by the camera's elevation.
    case front
    /// Multi-view only: map the front frame back into the frame the request's c2w matrices are
    /// written in (`C_0 · inv(F(d_0))`, rigid). With gravity-aligned cameras this is upright.
    /// Single view: identical to `.front`.
    case world
}

/// Golden-noise injection for the parity gates (nil fields → the seeded MLXRandom noise).
public struct Pixal3DNoise {
    public var ss: MLXArray?, shape512: MLXArray?, shape1024: MLXArray?, tex: MLXArray?
    public init(ss: MLXArray? = nil, shape512: MLXArray? = nil, shape1024: MLXArray? = nil, tex: MLXArray? = nil) {
        self.ss = ss; self.shape512 = shape512; self.shape1024 = shape1024; self.tex = tex
    }
}

/// What a Pixal3D run reports about its camera, so a host can re-orient the mesh.
public struct Pixal3DCameraReport: Sendable {
    public let fovXRadians: Double
    public let distance: Double
    public let meshScale: Double
    public let views: Int
    public let frame: Pixal3DFrame
    /// Row-major 4×4 taking the decoder's voxel frame to the returned mesh (rotation (x,−z,y), then
    /// front→world for `.world`, then the Y-up export rotation when `yUp`).
    public let appliedTransform: [Double]
    /// Camera-to-world of view 0 in the returned mesh's frame.
    public let camera0ToMesh: [Double]
}

extension Trellis2Pipeline {
    // MARK: gate helpers (stage pieces with golden inputs)

    /// SS decode → occupancy coords on the 32³ grid.
    public func pixal3dSSDecode(_ zs: MLXArray) -> MLXArray {
        let c = ssDec.coords(ssDec(zs), resolution: 32); MLX.eval(c); return c
    }
    /// The cascade "upsample": shape-decoder output coords for a NORMALIZED LR slat.
    public func pixal3dUpsample(lrSlat: MLXArray, coords: MLXArray, shapeMean: MLXArray, shapeStd: MLXArray) -> MLXArray {
        let up = shapeDec(SparseTensor(feats: lrSlat * shapeStd + shapeMean, coords: coords)).coords
        MLX.eval(up); return up
    }
    /// Raw dual-grid mesh vertices (MeshBake step 1) from shape-decoder output.
    public func pixal3dRawMeshFromDecoded(feats: MLXArray, coords: MLXArray, resolution: Int) -> MLXArray {
        let margin: Float = 0.5
        let dv = (1 + 2 * margin) * MLX.sigmoid(feats[0..., 0..<3]) - margin
        let inter = feats[0..., 3..<6] .> 0
        let ql = MeshBake.softplus(feats[0..., 6..<7])
        let (v, _) = DualGridMesh.extract(coords: coords[0..., 1..<4], dualVerts: dv, intersected: inter, quadLerp: ql,
                                          gridSize: Float(resolution))
        MLX.eval(v); return v
    }
    /// Raw dual-grid mesh vertices from a DENORMALIZED shape slat.
    public func pixal3dRawMesh(shapeSlatDenorm: MLXArray, coords: MLXArray, resolution: Int) -> MLXArray {
        let out = shapeDec(SparseTensor(feats: shapeSlatDenorm, coords: coords))
        MLX.eval(out.feats, out.coords)
        return pixal3dRawMeshFromDecoded(feats: out.feats, coords: out.coords, resolution: resolution)
    }

    func pixal3dURL(_ stem: String) -> URL {
        let dir = pixal3dDir ?? weightsDir ?? "."
        return URL(fileURLWithPath: "\(dir)/\(stem).safetensors")
    }

    /// Lazily load + cache a Pixal3D SLat flow (fp32-cast, ~5.5 GB resident).
    public func pixal3dFlow(_ kind: Pixal3DFlowKind) throws -> SLatFlowModel {
        if let m = pixal3dFlows[kind] { return m }
        let m = SLatFlowModel(weights: try Self.loadMaterialized(pixal3dURL(kind.fileStem)))
        precondition(m.isProjection, "\(kind.fileStem) is not a projection flow")
        pixal3dFlows[kind] = m
        return m
    }

    /// The Pixal3D SS flow (single-view or `_mv` twin; one resident at a time).
    public func pixal3dStructFlow(multiView: Bool) throws -> SparseStructureFlowModel {
        if let c = pixal3dSS, c.multiView == multiView { return c.model }
        pixal3dSS = nil
        let m = SparseStructureFlowModel(weights: try Self.loadMaterialized(pixal3dURL(Pixal3DFlowKind.structStem(multiView: multiView))))
        precondition(m.isProjection, "Pixal3D SS flow has no proj_linear")
        pixal3dSS = (multiView, m)
        return m
    }

    public func nafModel() throws -> NAF {
        if let n = nafCache { return n }
        let n = NAF(weights: try Self.loadMaterialized(pixal3dURL("naf")))
        nafCache = n
        return n
    }

    public func pixal3dConditioner() throws -> Pixal3DConditioner {
        Pixal3DConditioner(dino: dino, naf: try nafModel())
    }

    public static func pixal3dFlowKinds(multiView: Bool) -> [Pixal3DFlowKind] {
        [.shape512(multiView: multiView), .shape1024(multiView: multiView), .tex1024(multiView: multiView)]
    }

    /// Load the Pixal3D working set now (SS + the three cascade flows + NAF) and drop the stock SLat
    /// flows, so a Pixal3D job reads nothing from disk.
    public func preloadPixal3D(multiView: Bool) throws {
        releaseFlows()
        pixal3dFlows = pixal3dFlows.filter { Self.pixal3dFlowKinds(multiView: multiView).contains($0.key) }
        _ = try pixal3dStructFlow(multiView: multiView)
        for k in Self.pixal3dFlowKinds(multiView: multiView) { _ = try pixal3dFlow(k) }
        _ = try nafModel()
    }

    /// Drop the Pixal3D flows (and SS / NAF) so their buffers can be reclaimed.
    public func releasePixal3D() {
        pixal3dFlows = [:]; pixal3dSS = nil; nafCache = nil
    }

    /// Pixal3D generation (AB-T-0485): projection-conditioned SS → shape 512 → decoder upsample →
    /// Pixal3D requantize → shape 1024 → tex 1024 → decode → bake. Cascade tiers only (upstream has no
    /// 512 mode). One view = single-view; several views with c2w = posed multi-view (the `_mv`
    /// weights, features averaged over views). `capture` (gates) receives named intermediates.
    public func generatePixal3D(
        tier: Trellis2Tier = .res1024, views: [Pixal3DView], multiViewWeights: Bool? = nil,
        ssPhases: MLXArray,
        shapeMean: MLXArray, shapeStd: MLXArray, texMean: MLXArray, texStd: MLXArray,
        texture: Bool = true, targetFaces: Int = 120_000, yUp: Bool = false,
        atlasSize: Int = 1024, pbrMaps: Bool = false, unwrapBackend: UnwrapBackend = .xatlas,
        slatCfgSDPA: SDPAPrecision = .fp32, texSDPA: SDPAPrecision? = nil, steps: Int = 12,
        maxHRTokens: Int? = nil, remeshRes: Int? = nil, seed: UInt64 = 0,
        frame: Pixal3DFrame = .front, noise: Pixal3DNoise? = nil,
        noiseFn: [String: (MLXArray) -> MLXArray]? = nil, bake: Bool = true,
        capture: ((String, [String: MLXArray]) -> Void)? = nil,
        log: (String) -> Void = { print($0) }
    ) throws -> (mesh: BakedMesh?, hrResolution: Int, metrics: Trellis2StageMetrics, camera: Pixal3DCameraReport) {
        precondition(tier.isCascade, "Pixal3D has no 512 mode (AB-R-0859 §3); use res1024 or res1536")
        precondition(!views.isEmpty)
        let mv = multiViewWeights ?? (views.count > 1)
        releaseFlows()                          // the stock SLat flows are not part of this job
        let cams = views.map(\.camera)
        let cond = try pixal3dConditioner()

        var metrics = Trellis2StageMetrics()
        metrics.tier = "pixal3d-" + tier.rawValue + (mv ? "-mv" : "")
        metrics.seed = seed
        metrics.backend = unwrapBackend.rawValue
        metrics.stepsSS = steps; metrics.stepsSLat = steps
        let savedCast = TRELLIS2Config.sdpaCastDtype
        defer { TRELLIS2Config.sdpaCastDtype = savedCast }
        metrics.slatCfgSdpa = slatCfgSDPA.rawValue
        metrics.texSdpa = texSDPA?.rawValue ?? (savedCast == .bfloat16 ? "bf16" : "fp32")
        let zeros5 = MLXArray.zeros([1, 5, 1024])

        // 1) sparse structure (proj on the dense 16³ grid, DINOv3 at 512, no NAF)
        var t = Date()
        TRELLIS2Config.sdpaCastDtype = nil
        let cSS = cond.condition(views, stage: .ss, coords: nil, grid: 16)
        capture?("ss.cond", ["global": cSS.global, "proj": cSS.proj])
        let ssNoise = noise?.ss ?? SSGrid.noise(resolution: 16, seed: seed)
        let zs = FlowEulerSampler.sampleSS(model: try pixal3dStructFlow(multiView: mv), noise: ssNoise,
                                           cond: cSS.global, negCond: zeros5, phases: ssPhases,
                                           proj: cSS.proj.reshaped([1, -1, cSS.proj.dim(1)]),
                                           steps: steps, guidanceStrength: 7.5, guidanceRescale: 0.7,
                                           guidanceInterval: (0.6, 1.0), rescaleT: 5.0)
        MLX.eval(zs); metrics.ssSampleS = -t.timeIntervalSinceNow
        t = Date()
        let coords = ssDec.coords(ssDec(zs), resolution: 32); MLX.eval(coords)
        metrics.ssDecodeS = -t.timeIntervalSinceNow
        metrics.voxels32 = coords.dim(0)
        capture?("ss", ["samples": zs, "coords": coords])
        log("  [1] Pixal3D sparse structure: \(coords.dim(0)) voxels @32³ (\(String(format: "%.1f", metrics.ssSampleS + metrics.ssDecodeS))s)")

        // 2) shape 512 (LR) on the 32³ coords — proj at 32³, DINOv3 512 + NAF 512→512
        TRELLIS2Config.sdpaCastDtype = slatCfgSDPA.castDtype
        t = Date()
        let c512 = cond.condition(views, stage: .shape512, coords: coords, grid: 32)
        capture?("shape512.cond", ["global": c512.global, "proj": c512.proj])
        MLXRandom.seed(seed)
        let lrNoise = noise?.shape512 ?? noiseFn?["shape512"]?(coords) ?? MLXRandom.normal([coords.dim(0), 32])
        let lrSlat = FlowEulerSampler.sampleSLat(
            model: try pixal3dFlow(.shape512(multiView: mv)), noiseFeats: lrNoise, coords: coords,
            cond: c512.global, negCond: zeros5, proj: c512.proj,
            steps: steps, guidanceStrength: 7.5, guidanceRescale: 0.5, guidanceInterval: (0.6, 1.0), rescaleT: 3.0)
        MLX.eval(lrSlat)
        metrics.shapeSlatS = -t.timeIntervalSinceNow
        metrics.shapeTokens = coords.dim(0)
        capture?("shape512", ["samples": lrSlat, "coords": coords])
        log("  [2] Pixal3D LR shape SLat: \(coords.dim(0)) tokens (\(String(format: "%.1f", metrics.shapeSlatS))s)")

        // 2b) decoder upsample → Pixal3D requantize ((c+0.5)/512·(grid−1)).round(), token back-off
        t = Date()
        let upCoords = shapeDec(SparseTensor(feats: lrSlat * shapeStd + shapeMean, coords: coords)).coords
        MLX.eval(upCoords)
        let physBudget = Double(ProcessInfo.processInfo.physicalMemory) * 0.70
        let metalBudget = MTLCreateSystemDefaultDevice().map { Double($0.recommendedMaxWorkingSetSize) }
        let budgetBytes = min(metalBudget ?? physBudget, physBudget)
        let derivedCap = Int((budgetBytes - 5e9 - 28e9) / 1.55e6)
        let ceiling = maxHRTokens.map { min(49_152, $0) } ?? tier.hrTokenCeiling ?? 49_152
        let tokenCap = maxHRTokens != nil ? ceiling : min(ceiling, max(12_000, derivedCap))
        metrics.hrTokenCap = tokenCap
        let (hrCoords, hrResolution) = Pixal3DProjection.requantize(
            upCoords: upCoords, lrResolution: 512, targetResolution: tier.targetResolution,
            maxNumTokens: tokenCap, hardMaxTokens: ceiling)
        MLX.eval(hrCoords)
        metrics.upsampleRequantizeS = -t.timeIntervalSinceNow
        metrics.hrTokens = hrCoords.dim(0)
        let grid = hrResolution / 16
        capture?("hr", ["up_coords": upCoords, "coords": hrCoords, "hr_resolution": MLXArray(Int32(hrResolution))])
        log("  [2b] upsample \(upCoords.dim(0)) → requantized \(hrCoords.dim(0)) tokens @\(grid)³ (hr \(hrResolution))")

        // 2c) shape 1024 (HR) — proj at hr/16, DINOv3 1024 + NAF 1024→512
        t = Date()
        let cHR = cond.condition(views, stage: .shape1024, coords: hrCoords, grid: grid)
        capture?("shape1024.cond", ["global": cHR.global, "proj": cHR.proj])
        MLXRandom.seed(seed &+ 2)
        let hrNoise = noise?.shape1024 ?? noiseFn?["shape1024"]?(hrCoords) ?? MLXRandom.normal([hrCoords.dim(0), 32])
        let shapeSlat = FlowEulerSampler.sampleSLat(
            model: try pixal3dFlow(.shape1024(multiView: mv)), noiseFeats: hrNoise, coords: hrCoords,
            cond: cHR.global, negCond: zeros5, proj: cHR.proj,
            steps: steps, guidanceStrength: 7.5, guidanceRescale: 0.5, guidanceInterval: (0.6, 1.0), rescaleT: 3.0)
        MLX.eval(shapeSlat)
        metrics.shapeSlatHrS = -t.timeIntervalSinceNow
        capture?("shape1024", ["samples": shapeSlat, "coords": hrCoords])
        log("  [2c] Pixal3D HR shape SLat: \(hrCoords.dim(0)) tokens (\(String(format: "%.1f", metrics.shapeSlatHrS ?? 0))s)")
        TRELLIS2Config.sdpaCastDtype = texSDPA?.castDtype ?? savedCast

        // 3) shape decode
        t = Date()
        let shapeSlatDenorm = shapeSlat * shapeStd + shapeMean
        let (shapeOut, subs) = shapeDec.decodeCapturingSubs(SparseTensor(feats: shapeSlatDenorm, coords: hrCoords))
        MLX.eval(shapeOut.feats, shapeOut.coords)
        metrics.shapeDecodeS = -t.timeIntervalSinceNow
        metrics.decodedVoxels = shapeOut.count
        capture?("shape_decoded", ["feats": shapeOut.feats, "coords": shapeOut.coords])

        // 4-5) texture — proj at hr/16, DINOv3 1024 + NAF 1024→1024; concat_cond = normalized shape slat
        let baseColor: MLXArray
        var metallicRoughness: MLXArray? = nil
        if texture {
            t = Date()
            let cTex = cond.condition(views, stage: .tex1024, coords: hrCoords, grid: grid)
            capture?("tex1024.cond", ["global": cTex.global, "proj": cTex.proj])
            MLXRandom.seed(seed &+ 1)
            let texNoise = noise?.tex ?? noiseFn?["tex"]?(hrCoords) ?? MLXRandom.normal([hrCoords.dim(0), 32])
            let texSlat = FlowEulerSampler.sampleSLat(
                model: try pixal3dFlow(.tex1024(multiView: mv)), noiseFeats: texNoise, coords: hrCoords,
                cond: cTex.global, negCond: zeros5, concatCond: shapeSlat, proj: cTex.proj,
                steps: steps, guidanceStrength: 1.0, guidanceRescale: 0.0, guidanceInterval: (0.6, 0.9), rescaleT: 3.0)
            MLX.eval(texSlat)
            metrics.texSlatS = -t.timeIntervalSinceNow
            capture?("tex1024", ["samples": texSlat, "coords": hrCoords])
            t = Date()
            let texOut = texDec(SparseTensor(feats: texSlat * texStd + texMean, coords: hrCoords), guidedMasks: subs)
            baseColor = MLX.clip(texOut.feats[0..., 0..<3] * 0.5 + 0.5, min: 0, max: 1)
            MLX.eval(baseColor)
            if pbrMaps {
                let mr = MLX.clip(texOut.feats[0..., 3..<5] * 0.5 + 0.5, min: 0, max: 1)
                MLX.eval(mr); metallicRoughness = mr
            }
            metrics.texDecodeS = -t.timeIntervalSinceNow
            log("  [4-5] Pixal3D tex SLat + decode (\(String(format: "%.1f", (metrics.texSlatS ?? 0) + (metrics.texDecodeS ?? 0)))s)")
        } else {
            baseColor = MLXArray.zeros([shapeOut.count, 3]) + 0.5
        }

        // Output frame. Voxel frame (Y = photo up, camera on +Z) → Pixal3D world (Z-up, front camera at
        // −Y): (x, y, z) → (x, −z, y) [ProjGrid's rotation]; then the optional front→world rigid map;
        // then the optional Y-up export rotation (x, y, z) → (x, z, −y). All proper rotations.
        let toWorld = frame == .world ? Pixal3DCamera.frontToWorld(cams) : .identity
        let yUpM = yUp ? Mat4([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) : .identity
        let voxToWorld = Mat4([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        let post = yUpM * toWorld
        let full = post * voxToWorld                                  // voxel frame → returned mesh
        let report = Pixal3DCameraReport(
            fovXRadians: cams[0].fovX, distance: cams[0].distance, meshScale: cams[0].meshScale,
            views: views.count, frame: frame, appliedTransform: full.m,
            camera0ToMesh: (post * Mat4.front(distance: cams[0].distance)).m)
        metrics.hrResolution = hrResolution
        guard bake else { return (nil, hrResolution, metrics, report) }

        // 6) bake at the (possibly backed-off) resolution, in the voxel frame
        t = Date()
        let remesh = remeshRes ?? (hrResolution >= 1536 ? 384 : 256)
        let (bakedVoxel, bakeMetrics) = try MeshBake.run(shapeFeats: shapeOut.feats, coords: shapeOut.coords, texBaseColor: baseColor,
                                                         fineRes: Float(hrResolution), remeshRes: remesh,
                                                         targetFaces: targetFaces, atlasSize: atlasSize,
                                                         backend: unwrapBackend, texMetallicRoughness: metallicRoughness, log: log)
        metrics.meshbakeS = -t.timeIntervalSinceNow
        metrics.bake = bakeMetrics

        // 7) re-express the mesh in the output frame (`full`, see above)
        t = Date()
        let R: [[Float]] = (0..<3).map { r in (0..<3).map { c in Float(full[r, c]) } }
        let tr: [Float] = (0..<3).map { Float(full[$0, 3]) }
        func rotate(_ a: MLXArray, _ R: [[Float]], _ tr: [Float]? = nil) -> MLXArray {
            let x = a[0..., 0], y = a[0..., 1], z = a[0..., 2]
            var cols = (0..<3).map { r in R[r][0] * x + R[r][1] * y + R[r][2] * z }
            if let tr { cols = cols.enumerated().map { $0.element + tr[$0.offset] } }
            return MLX.stacked(cols, axis: 1)
        }
        let baked = BakedMesh(vertices: rotate(bakedVoxel.vertices, R, tr), faces: bakedVoxel.faces,
                              normals: rotate(bakedVoxel.normals, R), uvs: bakedVoxel.uvs,
                              texRGBA: bakedVoxel.texRGBA, atlasSize: bakedVoxel.atlasSize, coverage: bakedVoxel.coverage,
                              mrRGBA: bakedVoxel.mrRGBA, filledBeforeInpaint: bakedVoxel.filledBeforeInpaint,
                              wallFlipped: bakedVoxel.wallFlipped)
        metrics.yupS = -t.timeIntervalSinceNow
        return (baked, hrResolution, metrics, report)
    }
}
