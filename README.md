# mlx-trellis2-swift

A **verified Swift/MLX re-port of Microsoft TRELLIS.2** (image → textured 3D) for Apple
Silicon — no CUDA. Every numerical component is ported directly from the PyTorch model and
gated with a cosine/exactness parity test against the original running on CPU-fp32.

The build target is on-device garment/accessory generation (the mesh half of a VRoid
`.xwear` closet); it emits standard glTF (positions / normals / uv1 / indices + base-color
texture, `alphaMode=OPAQUE`).

Successor to the archived `mlx-trellis2-swift-old` (which hit a geometry-holes wall);
this is a from-scratch verified re-port built against a PyTorch-on-Apple-Silicon oracle.

## Runs at production scale

The decode → mesh → texture → GLB half runs end-to-end in Swift on the full object:
the shape decoder handles **7.85M voxels in 19 s** (`swift run -c release scaletest`,
cosine 0.999 vs golden), and the full mesh+texture bake produces a **watertight textured
GLB in 30 s** (`swift run -c release meshbake`) — DC-remesh (0 boundary edges) → simplify
→ UV unwrap → BVH-remap bake (hitFrac 1.0).

## Status — the whole pipeline is ported and parity-gated

| Stage | File | Parity vs PyTorch oracle |
|---|---|---|
| DINOv3 image conditioning (ViT-L/16) | `DINOv3.swift` | cosine 0.9999988 |
| Sparse-structure DiT (dense) | `SparseStructureFlowModel.swift` | 0.9998 |
| SLat DiT (sparse, shape + tex) | `SLatFlowModel.swift` | 0.99999565 |
| Flow-Euler CFG sampler (SS + sparse SLat) | `FlowEulerSampler.swift` | SS 0.9997 · SLat g1 0.99999243 · tex 0.9999982 |
| Sparse-structure decoder | `SparseStructureDecoder.swift` | exact (1.0) |
| Shape / tex VAE decoders (FlexiDualGrid) | `ShapeSlatDecoder.swift` | guided coords-exact · 0.9999999 / 0.9999998 |
| Shape encoder | `ShapeSlatEncoder.swift` | 1.0 |
| Dual-grid → mesh extraction | `DualGridMesh.swift` | **bit-exact** |
| Sparse-voxel grid_sample_3d | `GridSample3d.swift` | trilinear 1.0 · nearest exact |
| UV rasterizer | `UVRasterize.swift` | exact |
| glTF/GLB export | `GLTFExport.swift` | valid GLB |
| Mesh+texture orchestration | `MeshBake.swift` | end-to-end textured GLB (rendered) |

Supporting ops: `SubmanifoldConv3d` (+ cached `NeighborMap27`), `RoPE`, `MultiHeadAttention`,
`MultiHeadCrossAttention`, `TimestepEmbedder`, `TransformerBlock`, `SparseTensor`, and the
`C2STopology` (upsample) / `S2CTopology` (downsample) sparse resampling ops.

## Pipeline

```
image → DINOv3 cond
      → SS sampler → SS decoder → coords
      → shape SLat sampler → shape decoder → 7-ch dual-grid → DualGridMesh
      → mlx-swift-mesh: DC remesh (watertight) → simplify → UV unwrap
      → tex SLat sampler → tex decoder → 6-ch PBR voxels
      → UVRasterize → BVH closest-point remap → GridSample3d bake → inpaint
      → GLTFExport → textured .glb
```

The mesh cleanup / remesh / UV unwrap / BVH stages come from
[`mlx-swift-mesh`](https://github.com/mnmly/mlx-swift-mesh) (a Swift/Metal port of CuMesh by
the TRELLIS author).

## Layout

- `Sources/TRELLIS2/` — the library (all ported components).
- `Sources/parity/` — the parity harness. `swift run -c release parity` runs every gate.
- `Sources/dinoparity/` — the DINOv3 conditioning gate (`swift run -c release dinoparity`).
- `Sources/meshbake/` — end-to-end mesh+texture demo (`swift run -c release meshbake`).

## Build & run

Requires Xcode-beta / Swift 6.4 / macOS 26+ SDK and three sibling packages checked out
next to this one:

```
../mlx-swift        (0.31.3; run `git submodule update --init --recursive`)
../mlx-swift-mesh   (Package.swift pointed at ../mlx-swift + ../SwiftXatlas)
../SwiftXatlas      (fix its xatlas submodule SSH→HTTPS in .gitmodules)
```

Use `swift run` / `xcodebuild` (not `swift test` — the Metal `.metallib` isn't produced
under the test bundle). Builds are incremental and fast (MLX is pre-compiled).

The parity gates load `.npy` fixtures produced by the PyTorch oracle's golden harnesses
(`trellis2-port/_goldens_*.py`). Model weights load directly from the original
`microsoft/TRELLIS.2-4B` and `facebook/dinov3-vitl16-pretrain-lvd1689m` safetensors via MLX
`loadArrays`.

## Parity methodology

Direct PyTorch → Swift-MLX translation, verified per-op: the oracle dumps granular `.npy`
fixtures (deterministic, CPU-fp32; exact noise tensors are injected so no RNG parity is
needed), each Swift op/model reproduces the fixture, and a cosine (or exact-match) gate
guards it. Topology-adaptive stages (the subdivision decoders) are gated by injecting the
oracle's subdivision masks for a bit-exact coords check, plus a free-run agreement metric —
free-run divergence there is fp tie-nondeterminism, not a port defect.
