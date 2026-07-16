import Foundation
import MLX

/// Shape-conditioned texturing (the Trellis2TexturingPipeline counterpart): re-texture an
/// EXISTING mesh from a reference image, preserving its UVs. Parity: `swift run texturize`
/// (SW8 gate — every stage cosine-gated against the Python oracle).
///
/// Orchestration is generate-once / bake-per-material (proven on the Python side): ONE
/// texture-SLat generation over the full mesh, then a cheap original-UV bake per piece
/// against the shared PBR voxel field — so multi-material outfits stay style-consistent
/// and per-piece cost is ~milliseconds.
///
/// Inputs are in the pipeline's normalized model space: caller centers/scales the mesh
/// into [-0.5, 0.5]^3 and applies the glTF→model Y/Z swap (y = -z, z = y), mirroring
/// `preprocess_mesh`. `cond`/`negCond` are DINOv3 features of the reference image
/// (from `Trellis2Pipeline.encodeImage`).
public struct TexturingPipeline {
    public let encoder: ShapeSlatEncoder
    public let texDit: SLatFlowModel
    public let texDec: ShapeSlatDecoder
    public let shapeMean: MLXArray, shapeStd: MLXArray
    public let texMean: MLXArray, texStd: MLXArray

    public struct Piece {
        public let name: String
        public let vertices: MLXArray   // [V,3] normalized model space
        public let faces: MLXArray      // [F,3] int32
        public let uvs: MLXArray        // [V,2] glTF convention
        public init(name: String, vertices: MLXArray, faces: MLXArray, uvs: MLXArray) {
            self.name = name; self.vertices = vertices; self.faces = faces; self.uvs = uvs
        }
    }
    public struct PieceTexture {
        public let name: String
        public let rgba: [UInt8]        // atlasSize² × 4, glTF row convention
        public let atlasSize: Int
        public let coverage: Float
    }

    public init(encoderWeights: [String: MLXArray], texDitWeights: [String: MLXArray],
                texDecWeights: [String: MLXArray],
                shapeMean: MLXArray, shapeStd: MLXArray, texMean: MLXArray, texStd: MLXArray) {
        self.encoder = ShapeSlatEncoder(weights: encoderWeights)
        self.texDit = SLatFlowModel(weights: texDitWeights)
        self.texDec = ShapeSlatDecoder(weights: texDecWeights)
        self.shapeMean = shapeMean; self.shapeStd = shapeStd
        self.texMean = texMean; self.texStd = texStd
    }

    /// Texture `pieces` (submeshes of one garment/outfit, already normalized TOGETHER so
    /// they share one model space) from a single reference image conditioning.
    public func run(pieces: [Piece], cond: MLXArray, negCond: MLXArray,
                    resolution: Int = 512, atlasSize: Int = 1024, seed: UInt64 = 42,
                    log: (String) -> Void = { print($0) }) throws -> [PieceTexture] {
        // 1) full-mesh encode: concat pieces -> dual grid -> shape SLat (+ subdivision record)
        var allV: [Float] = [], allF: [Int32] = []
        for p in pieces {
            let base = Int32(allV.count / 3)
            allV.append(contentsOf: p.vertices.asType(.float32).reshaped([-1]).asArray(Float.self))
            allF.append(contentsOf: p.faces.asType(.int32).reshaped([-1]).asArray(Int32.self).map { $0 + base })
        }
        var t = Date()
        let conv = try DualGridConvert.meshToFlexibleDualGrid(vertices: allV, faces: allF, gridSize: resolution)
        let enc = DualGridConvert.encoderInputs(conv, resolution: resolution)
        let (slat, encMasks) = encoder.encodeCapturingMasks(
            SparseTensor(feats: enc.vfeats, coords: enc.coords4),
            SparseTensor(feats: enc.ifeats.asType(.float32), coords: enc.coords4))
        MLX.eval(slat.feats, slat.coords)
        log("  [1] encoded: \(conv.coords.dim(0)) voxels -> \(slat.coords.dim(0)) latents (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 2) tex SLat sampling, concat_cond = normalized shape SLat (g1.0, interval .6-.9)
        t = Date(); MLXRandom.seed(seed)
        let shapeNorm = (slat.feats - shapeMean) / shapeStd
        let texSlatNorm = FlowEulerSampler.sampleSLat(
            model: texDit, noiseFeats: MLXRandom.normal([slat.coords.dim(0), 32]), coords: slat.coords,
            cond: cond, negCond: negCond, concatCond: shapeNorm,
            guidanceStrength: 1.0, guidanceRescale: 0.0, guidanceInterval: (0.6, 0.9), rescaleT: 3.0)
        let texSlat = texSlatNorm * texStd + texMean
        MLX.eval(texSlat)
        log("  [2] tex SLat sampled (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 3) tex decode guided by the encoder's subdivision record (re-aligned per level to
        //    the decoder's C2S coord order — upstream's spatial-cache mechanism, explicit)
        t = Date()
        func key4(_ c: [Int32], _ i: Int) -> Int64 {
            Int64(c[i*4]) << 60 | Int64(c[i*4+1]) << 40 | Int64(c[i*4+2]) << 20 | Int64(c[i*4+3])
        }
        var guided: [MLXArray] = []
        var cur = slat.coords
        for k in 0..<encMasks.count {
            let (pc, mask) = encMasks[encMasks.count - 1 - k]
            let pcA: [Int32] = pc.reshaped([-1]).asArray(Int32.self)
            let maskA: [Int32] = mask.reshaped([-1]).asArray(Int32.self)
            var lookup = [Int64: Int](minimumCapacity: pc.dim(0))
            for i in 0..<pc.dim(0) { lookup[key4(pcA, i)] = i }
            let curA: [Int32] = cur.asType(.int32).reshaped([-1]).asArray(Int32.self)
            var ordered = [Int32](repeating: 0, count: cur.dim(0) * 8)
            for i in 0..<cur.dim(0) {
                guard let j = lookup[key4(curA, i)] else { throw DualGridConvert.Error.conversionFailed }
                for s in 0..<8 { ordered[i*8+s] = maskA[j*8+s] }
            }
            let m = MLXArray(ordered).reshaped([cur.dim(0), 8])
            guided.append(m)
            cur = C2STopology(coords: cur, subdivLogits: m.asType(.float32)).newCoords
        }
        let pbrOut = texDec(SparseTensor(feats: texSlat, coords: slat.coords), guidedMasks: guided)
        let pbr = pbrOut.feats * 0.5 + 0.5
        MLX.eval(pbr, pbrOut.coords)
        log("  [3] tex decoded: \(pbrOut.coords.dim(0)) PBR voxels (\(String(format: "%.1f", -t.timeIntervalSinceNow))s)")

        // 4) per-piece original-UV bake against the shared field
        return pieces.map { p in
            let baked = MeshBake.bakeWithUVs(
                vertices: p.vertices.asType(.float32), faces: p.faces.asType(.int32), uvs: p.uvs.asType(.float32),
                texFeats: pbr, texCoords: pbrOut.coords, fineRes: Float(resolution), atlasSize: atlasSize, log: log)
            return PieceTexture(name: p.name, rgba: baked.rgba, atlasSize: atlasSize, coverage: baked.coverage)
        }
    }
}
