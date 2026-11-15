import Foundation
import MLX
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Binary glTF 2.0 (.glb) writer for a single textured triangle mesh — the final
/// export step of the garment pipeline. One buffer (the BIN chunk), interleaved
/// bufferViews/accessors for POSITION (+ optional NORMAL, TEXCOORD_0) and
/// UNSIGNED_INT indices, and an optional embedded PNG base-color texture.
/// Little-endian throughout (GLB requires it). Dependency-free beyond Foundation
/// + macOS system frameworks.
public enum GLTFExport {
    /// Where the material's occlusion comes from.
    public enum Occlusion {
        /// The metallic-roughness texture's R channel (glTF ORM packing: R = occlusion).
        case metallicRoughnessRed
        /// Its own RGBA8 texture (occlusion read from R).
        case texture(pixels: [UInt8], width: Int, height: Int)
    }

    /// Write a single-primitive textured mesh as binary glTF (.glb).
    /// - positions: [V,3] Float, indices: [F,3] Int32, optional [V,3] normals,
    ///   optional [V,2] uvs, optional RGBA8 base-color texture (row-major).
    public static func writeGLB(
        to url: URL,
        positions: MLXArray,
        indices: MLXArray,
        normals: MLXArray? = nil,
        uvs: MLXArray? = nil,
        baseColorRGBA: (pixels: [UInt8], width: Int, height: Int)? = nil,
        metallicRoughnessRGBA: (pixels: [UInt8], width: Int, height: Int)? = nil,
        tangents: MLXArray? = nil,
        normalRGBA: (pixels: [UInt8], width: Int, height: Int)? = nil,
        occlusion: Occlusion? = nil
    ) throws {
        let glb = try glbData(positions: positions, indices: indices, normals: normals,
                              uvs: uvs, baseColorRGBA: baseColorRGBA,
                              metallicRoughnessRGBA: metallicRoughnessRGBA,
                              tangents: tangents, normalRGBA: normalRGBA, occlusion: occlusion)
        try glb.write(to: url)
    }

    /// Serialize a single-primitive textured mesh as in-memory binary glTF (.glb) `Data` — the
    /// same bytes `writeGLB` writes to disk. Used by the engine `ModelPackage` path, whose
    /// canonical `Mesh` output is GLB bytes (no file needed).
    ///
    /// The optional detail maps: `tangents` [V,4] (glTF TANGENT, xyz unit + w handedness; needed for
    /// `normalRGBA`, a tangent-space normal map), and `occlusion` (AO, either packed into the
    /// metallic-roughness texture's R channel or its own texture). Each one left nil writes nothing,
    /// so a call without them gives the same bytes as before they existed.
    public static func glbData(
        positions: MLXArray,
        indices: MLXArray,
        normals: MLXArray? = nil,
        uvs: MLXArray? = nil,
        baseColorRGBA: (pixels: [UInt8], width: Int, height: Int)? = nil,
        metallicRoughnessRGBA: (pixels: [UInt8], width: Int, height: Int)? = nil,
        tangents: MLXArray? = nil,
        normalRGBA: (pixels: [UInt8], width: Int, height: Int)? = nil,
        occlusion: Occlusion? = nil
    ) throws -> Data {
        let V = positions.dim(0)
        let pos = positions.asType(.float32).asArray(Float.self)   // [V*3]
        let idx = indices.asType(.int32).asArray(Int32.self)       // [F*3]
        let nrm = normals.map { $0.asType(.float32).asArray(Float.self) }
        let uv = uvs.map { $0.asType(.float32).asArray(Float.self) }
        let png = baseColorRGBA.flatMap { encodePNG(pixels: $0.pixels, width: $0.width, height: $0.height) }
        let mrPng = metallicRoughnessRGBA.flatMap { encodePNG(pixels: $0.pixels, width: $0.width, height: $0.height) }
        let tan = tangents.map { $0.asType(.float32).asArray(Float.self) }
        precondition(tan == nil || tan!.count == V * 4, "tangents must be [V,4]")
        precondition(normalRGBA == nil || (tan != nil && uv != nil), "a normal map needs tangents and uvs")
        let nPng = normalRGBA.flatMap { encodePNG(pixels: $0.pixels, width: $0.width, height: $0.height) }
        var occPng: Data? = nil
        if case .texture(let px, let w, let h)? = occlusion { occPng = encodePNG(pixels: px, width: w, height: h) }
        if case .metallicRoughnessRed? = occlusion {
            precondition(mrPng != nil, "occlusion packed into metallicRoughness needs that texture")
        }

        var bin = Data()
        var bufferViews: [[String: Any]] = []
        var accessors: [[String: Any]] = []
        @inline(__always) func pad4() { while bin.count % 4 != 0 { bin.append(0) } }

        // POSITION (with required min/max)
        pad4()
        let posOff = bin.count
        for f in pos { putF32LE(f, &bin) }
        bufferViews.append(["buffer": 0, "byteOffset": posOff, "byteLength": pos.count * 4, "target": 34962])
        var minP = [Double](repeating: .greatestFiniteMagnitude, count: 3)
        var maxP = [Double](repeating: -.greatestFiniteMagnitude, count: 3)
        for v in 0..<V { for a in 0..<3 {
            let x = Double(pos[v*3 + a]); minP[a] = min(minP[a], x); maxP[a] = max(maxP[a], x)
        } }
        accessors.append(["bufferView": bufferViews.count - 1, "componentType": 5126,
                          "count": V, "type": "VEC3", "min": minP, "max": maxP])
        let posAcc = accessors.count - 1

        // NORMAL
        var nrmAcc: Int? = nil
        if let nrm {
            pad4()
            let off = bin.count
            for f in nrm { putF32LE(f, &bin) }
            bufferViews.append(["buffer": 0, "byteOffset": off, "byteLength": nrm.count * 4, "target": 34962])
            accessors.append(["bufferView": bufferViews.count - 1, "componentType": 5126,
                              "count": V, "type": "VEC3"])
            nrmAcc = accessors.count - 1
        }

        // TEXCOORD_0
        var uvAcc: Int? = nil
        if let uv {
            pad4()
            let off = bin.count
            for f in uv { putF32LE(f, &bin) }
            bufferViews.append(["buffer": 0, "byteOffset": off, "byteLength": uv.count * 4, "target": 34962])
            accessors.append(["bufferView": bufferViews.count - 1, "componentType": 5126,
                              "count": V, "type": "VEC2"])
            uvAcc = accessors.count - 1
        }

        // TANGENT (VEC4: xyz + handedness)
        var tanAcc: Int? = nil
        if let tan {
            pad4()
            let off = bin.count
            for f in tan { putF32LE(f, &bin) }
            bufferViews.append(["buffer": 0, "byteOffset": off, "byteLength": tan.count * 4, "target": 34962])
            accessors.append(["bufferView": bufferViews.count - 1, "componentType": 5126,
                              "count": V, "type": "VEC4"])
            tanAcc = accessors.count - 1
        }

        // indices (UNSIGNED_INT)
        pad4()
        let idxOff = bin.count
        for i in idx { putU32LE(UInt32(bitPattern: i), &bin) }
        bufferViews.append(["buffer": 0, "byteOffset": idxOff, "byteLength": idx.count * 4, "target": 34963])
        accessors.append(["bufferView": bufferViews.count - 1, "componentType": 5125,
                          "count": idx.count, "type": "SCALAR"])
        let idxAcc = accessors.count - 1

        // embedded PNG texture
        var imgBV: Int? = nil
        if let png {
            pad4()
            let off = bin.count
            bin.append(png)
            bufferViews.append(["buffer": 0, "byteOffset": off, "byteLength": png.count])
            imgBV = bufferViews.count - 1
        }
        var mrBV: Int? = nil
        if let mrPng {
            pad4()
            let off = bin.count
            bin.append(mrPng)
            bufferViews.append(["buffer": 0, "byteOffset": off, "byteLength": mrPng.count])
            mrBV = bufferViews.count - 1
        }
        var nBV: Int? = nil
        if let nPng {
            pad4()
            let off = bin.count
            bin.append(nPng)
            bufferViews.append(["buffer": 0, "byteOffset": off, "byteLength": nPng.count])
            nBV = bufferViews.count - 1
        }
        var occBV: Int? = nil
        if let occPng {
            pad4()
            let off = bin.count
            bin.append(occPng)
            bufferViews.append(["buffer": 0, "byteOffset": off, "byteLength": occPng.count])
            occBV = bufferViews.count - 1
        }

        // --- assemble attributes / material ---
        var attributes: [String: Any] = ["POSITION": posAcc]
        if let nrmAcc { attributes["NORMAL"] = nrmAcc }
        if let uvAcc { attributes["TEXCOORD_0"] = uvAcc }
        if let tanAcc { attributes["TANGENT"] = tanAcc }

        var primitive: [String: Any] = ["attributes": attributes, "indices": idxAcc, "mode": 4]

        var root: [String: Any] = [
            "asset": ["version": "2.0", "generator": "TRELLIS2 GLTFExport"],
            "scene": 0,
            "scenes": [["nodes": [0]]],
            "nodes": [["mesh": 0]],
            "buffers": [["byteLength": bin.count]],
            "bufferViews": bufferViews,
            "accessors": accessors,
        ]

        // With a metallicRoughness texture the factors act as multipliers and
        // must be 1.0 (glTF spec); without one, stay fully dielectric.
        var pbr: [String: Any] = mrBV != nil
            ? ["metallicFactor": 1.0, "roughnessFactor": 1.0]
            : ["metallicFactor": 0.0, "roughnessFactor": 1.0]
        var images: [[String: Any]] = []
        var textures: [[String: Any]] = []
        if let imgBV {
            images.append(["bufferView": imgBV, "mimeType": "image/png"])
            textures.append(["source": images.count - 1, "sampler": 0])
            pbr["baseColorTexture"] = ["index": textures.count - 1]
        }
        if let mrBV {
            images.append(["bufferView": mrBV, "mimeType": "image/png"])
            textures.append(["source": images.count - 1, "sampler": 0])
            pbr["metallicRoughnessTexture"] = ["index": textures.count - 1]
        }
        var normalTex: Int? = nil, occlusionTex: Int? = nil
        if let nBV {
            images.append(["bufferView": nBV, "mimeType": "image/png"])
            textures.append(["source": images.count - 1, "sampler": 0])
            normalTex = textures.count - 1
        }
        if let occBV {
            images.append(["bufferView": occBV, "mimeType": "image/png"])
            textures.append(["source": images.count - 1, "sampler": 0])
            occlusionTex = textures.count - 1
        } else if case .metallicRoughnessRed? = occlusion, mrBV != nil {
            occlusionTex = (pbr["metallicRoughnessTexture"] as? [String: Any])?["index"] as? Int
        }
        if !images.isEmpty {
            root["images"] = images
            root["samplers"] = [[String: Any]()]
            root["textures"] = textures
        }
        var material: [String: Any] = [
            "pbrMetallicRoughness": pbr,
            "alphaMode": "OPAQUE",
            "doubleSided": true,
        ]
        if let normalTex { material["normalTexture"] = ["index": normalTex] }
        if let occlusionTex { material["occlusionTexture"] = ["index": occlusionTex] }
        root["materials"] = [material]
        primitive["material"] = 0
        root["meshes"] = [["primitives": [primitive]]]

        // --- GLB container ---
        let jsonData = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        var jsonChunk = jsonData
        while jsonChunk.count % 4 != 0 { jsonChunk.append(0x20) }   // pad with spaces
        var binChunk = bin
        while binChunk.count % 4 != 0 { binChunk.append(0x00) }     // pad with zeros

        let total = 12 + 8 + jsonChunk.count + 8 + binChunk.count
        var glb = Data()
        putU32LE(0x4654_6C67, &glb)          // magic "glTF"
        putU32LE(2, &glb)                    // version
        putU32LE(UInt32(total), &glb)        // total length
        putU32LE(UInt32(jsonChunk.count), &glb)
        putU32LE(0x4E4F_534A, &glb)          // "JSON"
        glb.append(jsonChunk)
        putU32LE(UInt32(binChunk.count), &glb)
        putU32LE(0x004E_4942, &glb)          // "BIN\0"
        glb.append(binChunk)

        return glb
    }

    // MARK: - helpers

    private static func putF32LE(_ v: Float, _ data: inout Data) {
        var le = v.bitPattern.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }
    private static func putU32LE(_ v: UInt32, _ data: inout Data) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }

    /// RGBA8 (straight alpha, row-major) -> PNG bytes via ImageIO. Data maps (normal, occlusion,
    /// metallic-roughness) carry the same sRGB tag as colour: glTF readers treat them as linear
    /// regardless, and to a colour-managed browser decode on an sRGB pipeline the tag is a no-op,
    /// where a linear tag would get the values re-curved.
    private static func encodePNG(pixels: [UInt8], width: Int, height: Int) -> Data? {
        guard width > 0, height > 0, pixels.count >= width * height * 4 else { return nil }
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)   // non-premultiplied, alpha last
        guard let cg = CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: cs, bitmapInfo: bitmapInfo, provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return out as Data
    }
}
