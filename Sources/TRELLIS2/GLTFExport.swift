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
    /// Write a single-primitive textured mesh as binary glTF (.glb).
    /// - positions: [V,3] Float, indices: [F,3] Int32, optional [V,3] normals,
    ///   optional [V,2] uvs, optional RGBA8 base-color texture (row-major).
    public static func writeGLB(
        to url: URL,
        positions: MLXArray,
        indices: MLXArray,
        normals: MLXArray? = nil,
        uvs: MLXArray? = nil,
        baseColorRGBA: (pixels: [UInt8], width: Int, height: Int)? = nil
    ) throws {
        let V = positions.dim(0)
        let pos = positions.asType(.float32).asArray(Float.self)   // [V*3]
        let idx = indices.asType(.int32).asArray(Int32.self)       // [F*3]
        let nrm = normals.map { $0.asType(.float32).asArray(Float.self) }
        let uv = uvs.map { $0.asType(.float32).asArray(Float.self) }
        let png = baseColorRGBA.flatMap { encodePNG(pixels: $0.pixels, width: $0.width, height: $0.height) }

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

        // --- assemble attributes / material ---
        var attributes: [String: Any] = ["POSITION": posAcc]
        if let nrmAcc { attributes["NORMAL"] = nrmAcc }
        if let uvAcc { attributes["TEXCOORD_0"] = uvAcc }

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

        var pbr: [String: Any] = ["metallicFactor": 0.0, "roughnessFactor": 1.0]
        if let imgBV {
            root["images"] = [["bufferView": imgBV, "mimeType": "image/png"]]
            root["samplers"] = [[String: Any]()]
            root["textures"] = [["source": 0, "sampler": 0]]
            pbr["baseColorTexture"] = ["index": 0]
        }
        let material: [String: Any] = [
            "pbrMetallicRoughness": pbr,
            "alphaMode": "OPAQUE",
            "doubleSided": true,
        ]
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

        try glb.write(to: url)
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

    /// RGBA8 (straight alpha, row-major) -> PNG bytes via ImageIO.
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
