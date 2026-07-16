import Foundation
import MLX

/// SparseStructureDecoder (TRELLIS SS VAE decoder). Dense conv3d net:
/// z_s[1,8,16³] -> logits[1,1,64³]; occupancy = logits>0 -> coords.
/// Ported channels-LAST (MLX conv3d is NDHWC), so ChannelLayerNorm == LayerNorm
/// over the last axis. channels=[512,128,32], res=2, middle=2, norm=layer.
public final class SparseStructureDecoder {
    // conv weights already transposed to MLX layout [Cout,kD,kH,kW,Cin]; biases [Cout].
    let w: [String: MLXArray]

    public init(weights: [String: MLXArray]) {
        var m = [String: MLXArray]()
        for (k, v) in weights {
            if k.hasSuffix(".weight") && v.ndim == 5 {
                m[k] = v.asType(.float32).transposed(0, 2, 3, 4, 1)   // [Cout,Cin,kD,kH,kW]->[Cout,kD,kH,kW,Cin]
            } else {
                m[k] = v.asType(.float32)
            }
        }
        self.w = m
    }

    private func conv(_ x: MLXArray, _ p: String, pad: Int = 1) -> MLXArray {
        conv3d(x, w["\(p).weight"]!, stride: 1, padding: IntOrTriple(pad)) + w["\(p).bias"]!
    }
    private static func silu(_ x: MLXArray) -> MLXArray { x * MLX.sigmoid(x) }

    /// ChannelLayerNorm32 in channels-last == LayerNorm over last axis, fp32, eps 1e-5.
    private func chLayerNorm(_ x: MLXArray, _ p: String) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let d = x - mean
        let varr = (d * d).mean(axis: -1, keepDims: true)
        let normed = d * MLX.rsqrt(varr + 1e-5)
        return normed * w["\(p).weight"]! + w["\(p).bias"]!
    }

    /// ResBlock3d: norm1->silu->conv1->norm2->silu->conv2 + skip(Identity here).
    private func resBlock(_ x: MLXArray, _ p: String) -> MLXArray {
        var h = chLayerNorm(x, "\(p).norm1")
        h = SparseStructureDecoder.silu(h)
        h = conv(h, "\(p).conv1")
        h = chLayerNorm(h, "\(p).norm2")
        h = SparseStructureDecoder.silu(h)
        h = conv(h, "\(p).conv2")
        return h + x
    }

    /// UpsampleBlock3d: conv(->out*8) then pixel_shuffle_3d(scale 2), channels-last.
    private func upsample(_ x: MLXArray, _ p: String, outC: Int) -> MLXArray {
        let h = conv(x, "\(p).conv")                      // [B,s0,s1,s2,outC*8]
        let s = h.shape
        let (B, a, b, c) = (s[0], s[1], s[2], s[3])
        return h.reshaped([B, a, b, c, outC, 2, 2, 2])
                .transposed(0, 1, 5, 2, 6, 3, 7, 4)       // [B,a,sf0,b,sf1,c,sf2,outC]
                .reshaped([B, a * 2, b * 2, c * 2, outC])
    }

    /// z_s channels-first [1,8,16,16,16] -> logits channels-first [1,1,64,64,64].
    public func callAsFunction(_ zs: MLXArray) -> MLXArray {
        var h = zs.transposed(0, 2, 3, 4, 1)              // -> channels-last [1,16,16,16,8]
        h = conv(h, "input_layer")                        // [1,16,16,16,512]
        h = resBlock(h, "middle_block.0")
        h = resBlock(h, "middle_block.1")
        // blocks: 0,1 res(512); 2 up(512->128); 3,4 res(128); 5 up(128->32); 6,7 res(32)
        h = resBlock(h, "blocks.0"); h = resBlock(h, "blocks.1")
        h = upsample(h, "blocks.2", outC: 128)            // -> 32³,128
        h = resBlock(h, "blocks.3"); h = resBlock(h, "blocks.4")
        h = upsample(h, "blocks.5", outC: 32)             // -> 64³,32
        h = resBlock(h, "blocks.6"); h = resBlock(h, "blocks.7")
        h = chLayerNorm(h, "out_layer.0")
        h = SparseStructureDecoder.silu(h)
        h = conv(h, "out_layer.2")                        // [1,64,64,64,1]
        return h.transposed(0, 4, 1, 2, 3)                // -> channels-first [1,1,64,64,64]
    }

    /// occupancy = logits>0 -> coords[[b,d0,d1,d2]] (int32), matching torch.argwhere order.
    /// Host-compacted (mlx-swift has no boolean-mask gather); ascending linear index
    /// preserves argwhere's row-major order.
    ///
    /// `resolution` < the decoded grid (64) max-pools the occupancy down first
    /// (upstream sample_sparse_structure's `max_pool3d(ratio) > 0.5`, i.e. logical OR
    /// over each ratio³ block) — the 512/cascade tiers run shape SLat on a 32³ grid.
    public func coords(_ logits: MLXArray, resolution: Int? = nil) -> MLXArray {
        var occ = (logits.squeezed(axis: 1) .> 0)         // [1,D0,D1,D2]
        let D = occ.dim(1)
        if let res = resolution, res != D {
            let r = D / res
            occ = occ.asType(.float32)
                .reshaped([1, res, r, res, r, res, r])
                .max(axes: [2, 4, 6]) .> 0.5              // OR over each r³ block
        }
        let D0 = occ.dim(1), D1 = occ.dim(2), D2 = occ.dim(3)
        let flat = occ.reshaped([-1]).asType(.int32).asArray(Int32.self)
        let plane = D1 * D2, vol = D0 * plane
        var out = [Int32](); out.reserveCapacity(flat.count / 8)
        for i in 0..<flat.count where flat[i] != 0 {
            let b = i / vol, r = i % vol
            out.append(Int32(b)); out.append(Int32(r / plane)); out.append(Int32((r % plane) / D2)); out.append(Int32(r % D2))
        }
        return MLXArray(out).reshaped([out.count / 4, 4])
    }
}
