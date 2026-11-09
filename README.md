# mlx-trellis2-swift

A **verified Swift/MLX re-port of Microsoft TRELLIS.2** (image → textured 3D) for Apple
Silicon — no CUDA. Every numerical component is ported directly from the PyTorch model and
gated with a cosine/exactness parity test against the original running on CPU-fp32.

The build target is on-device garment/accessory generation (the mesh half of a VRoid
`.xwear` closet); it emits standard glTF (positions / normals / uv1 / indices + base-color
texture, `alphaMode=OPAQUE`).

Successor to the archived `mlx-trellis2-swift-old` (which hit a geometry-holes wall);
this is a from-scratch verified re-port built against a PyTorch-on-Apple-Silicon oracle.

## Mesh + bake stage (v0.8.0)

`MeshBake.run` turns decoder voxel output into a textured GLB:
dual-grid extract → DC remesh → simplify (QEM + normal-flip collapse guard) →
**UV unwrap** → conservative 5-pass rasterize at 2× supersample → BVH remap +
trilinear attribute bake (baseColor + metallicRoughness) → inpaint-to-completion
→ glTF export with both PBR textures.

**Unwrap backends** (`Trellis2Configuration.unwrapBackend`):

| backend | how | production bake (300k faces / 2048 atlas) |
|---|---|---|
| `.xatlas` **(default)** — `Mesh.uvUnwrapParallel()` | BSP-partition → concurrent per-bucket xatlas → union repack via the pack-only seam | **24.8s**, meanErr 0.0128, bad 1.62% |
| `.provenance` | grid/normal tags → CC charts → axis projection → pack-only | 25.4s, meanErr 0.0146, bad 2.19% |

Single-instance xatlas on the same input: 188.9s (and 628.6s for the chart
stage alone at 300k). See `BENCHMARKS.md` for the harnesses and
`mlxengine-3d/UV-UNWRAP-METAL-PLAN.md` for the full measurement record.

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

## Multi-view conditioning

`ImageTo3DRequest.additionalViews` takes more views of the same subject (a side or back, say)
next to the primary `image`. Each view is matted, preprocessed and DINOv3-encoded at every
resolution its tier needs. The tokens are concatenated to `[1, K·N, D]`, as in the oracle's
multi-image `get_cond`. The views carry no camera: order and pose are free.

The engine driver passes them through `EXTRA_VIEWS` (comma-separated PNG/JPEG paths):

```
IMG=front.png EXTRA_VIEWS=side.png,back.png HR_RES=1024 WEIGHTS_DIR=<consolidated weights> \
  swift run -c release trellis2-run-engine
```

The metrics record's `views` field reports how many views conditioned the run.

Because the views carry no pose, a feature seen in one view can land on any side of the object.
`Trellis2Configuration.additionalViewsScope = "texture"` (driver: `EXTRA_VIEWS_SCOPE=texture`)
conditions the sparse-structure and shape flows on the primary image alone and gives every view
only to the texture flow, so extra views can add colour evidence without steering geometry.
`views_scope` in the metrics record says which scope ran.

## Footprint per tier (v0.11.0)

Peak memory follows the number of HR tokens the cascade samples, not the tier's nominal
resolution. Since v0.11.0 each cascade tier has a **hard HR token ceiling**
(`Trellis2Tier.hrTokenCeiling`: res1024 28,672, res1536 32,768). An asset over the ceiling backs
off in resolution, 128 at a time, below 1024 if it has to. That keeps each tier's declared
footprint a bound rather than a typical case.

| Lane | Resident | Activation | Charged | Worst measured peak (phys) |
|---|---|---|---|---|
| res512 | 25 GB | 9 GB | **34 GB** | 30.54 GB (dress, 4,856 tokens) |
| res1024 | 31 GB | 38 GB | **69 GB** | 60.74 GB (hair, 25,977 tokens @1024) |
| res1536 | 31 GB | 41 GB | **72 GB** | 65.23 GB (dress, 29,437 tokens, backed off to 1152) |

Measurement setup:
- Release build on an M5 Max with 128 GB, mlx-engine-swift 0.74.1 and xocialize/mlx-swift 0.32.4.
- BiRefNet matting registered in the same process; inputs were the t31 corpus objects on white,
  matted by it.
- `phys_footprint` sampled every 200 ms by `trellis2-run-engine`, one fresh process per run.
- Resident is phys after a run with MLX's cache cleared, BiRefNet included. The table is also in
  `Trellis2Footprint`.

**A host that schedules against one memory budget registers one lane per tier**,
`Trellis2Configuration.lane(ImageTo3DContract.res1024)` and so on:
- A lane sets `defaultMode` and `tierCeiling` to its tier.
- `FootprintConfigured` reports that tier's resident and activation.
- `run()` refuses a request above the lane's tier before touching weights
  (`PackageError.unsupportedRequestFeature`); lower tiers fit inside its charge and run.
- Setting `maxHRTokens` moves the bound, and the activation hint follows it. Past 32,768 tokens
  it is charged at the older 1.55 MB/token slope.

Without `tierCeiling` the configuration behaves as before: it is charged for `defaultMode`, and
any request mode is accepted.

`trellis2-gate`'s FIT section checks every lane, the manifest (which carries the res1536 lane, the
largest) and the refusal.

## Per-request options (v0.12.0)

A host that keeps one package resident per lane sets each job's knobs through
`ImageTo3DRequest.metaData` (`Trellis2RequestOptions`). Any key that is absent or `null` falls back
to the configuration, so a request with no keys runs exactly the configured job.

| Key | Values | Falls back to |
|---|---|---|
| `seed` | non-negative integer, or its decimal string for values past 2^53 | `seed` |
| `steps` | 1…50 | `steps` |
| `texture` | bool | `texture` |
| `decimateFaces` | 1,000…2,000,000; a hard ceiling | `decimateFaces` (else 120,000) |
| `atlasSize` | 512, 1024, 2048, 4096 | 1024 |
| `yUp` | bool | `yUpOutput` |
| `pbrMaps` | bool: also write the metallic-roughness map (G = roughness, B = metallic) | false |

A malformed value is refused (`PackageError.unsupportedRequestFeature`) before weights are touched.
Nothing is clamped silently. `pbrMaps` needs `texture`, because there is no tex decode to read the
map from otherwise.

**`decimateFaces` is a ceiling (v0.12.1).** The result never has more faces than asked for, for any
target in 1,000…2,000,000 (`Mesh.simplify(hardCeiling:)`, AB-A-0265). The first pass is the plain
QEM simplify, so a mesh that lands under its target there is byte-identical to v0.12.0. Only if the
simplify stalls above the target does it continue, relaxing one guard per stage, in this order:
1. more rounds;
2. boundary edges become collapsible;
3. no normal-flip guard;
4. no link condition;
5. then it drops the smallest connected components;
6. as a last resort, it drops the smallest-area faces.

The metrics record names the stage that met the target (`bake_simplify_stage`).

**Weights load at prepare (v0.12.1).** Each safetensors file is materialized when it is loaded, and a
lane (`tierCeiling`) loads its tier's flows in `load()`. That keeps disk reads out of the GPU command
buffers. A lazy load read inside the first job's `eval`, behind another reader of the same disk, ran
past Metal's command-buffer watchdog in the Gaea worker ("Caused GPU Timeout Error"). The weight
prewarm now pages only the charged tier's files, not the whole 35 GB snapshot.

**Orientation.** With `yUp`, the mesh is +Y up, and the object's front faces +Z. The shoe's toe
points +Z, and +Y is up for every asset checked. TRELLIS.2 builds the object in its own canonical
frame, not in the input camera's frame. A frontal photo is therefore what a camera on +Z sees, but
a 3/4 photo still comes out facing +Z. A silhouette sweep over the shoe put the photo's camera at
60° yaw and 30° elevation.

**Determinism.** The same image, options and lane give the same GLB bytes, within one process and
across processes. That was checked at res512 and res1024. Two GPU scatter-adds used to sum in
thread order, and float addition is not associative:
- the simplifier's quadrics, in a Metal kernel with float atomics;
- `vertexNormals`, via `at[].add`.

They now sum on the CPU in face order. Before this change the final face count drifted by up to
0.07 % between identical runs.

**Atlas size and memory.** The bake supersamples 2×, so a 4096 atlas rasterizes an 8192² grid.
Since v0.12.0 the per-texel stage runs in 16M-texel chunks. Before chunking, a 4096 atlas peaked at
about 100 GB even at res512. At atlas 1024 the chunked stage is a single chunk, the same computation
as before, with byte-identical output.

`Trellis2Configuration.maxAtlasSize` (default 1024) caps what a request may ask for, and the res512
lane's charge follows it. Measured peak phys for the dress with `pbrMaps` on:

| res512 lane `maxAtlasSize` | Measured peak | Charged |
|---|---|---|
| 1024 | 30.54 GB | 34 GB |
| 2048 | 34.73 GB | 38 GB |
| 4096 | 43.59 GB | 48 GB |

On the cascade lanes the HR stage stays the peak. res1536 with a 4096 atlas peaked at 65.23 GB,
the same as with 1024, so their charges don't move.

## Engine and mlx-swift pins

From v0.11.0 the package needs mlx-engine-swift `from: "0.74.1"` and the fleet's mlx-swift fork,
`xocialize/mlx-swift` `from: "0.32.4"`. The fork is upstream 0.32.3 plus ml-explore/mlx-swift#501,
the quantized-load leak fix (AB-D-0208).

v0.10.0 still named `ml-explore/mlx-swift`. That clashed with the engine's fork URL, and SwiftPM
picked upstream, so apps lost the leak fix.

## Layout

- `Sources/TRELLIS2/` — the library (all ported components).
- `Sources/parity/` — the parity harness. `swift run -c release parity` runs every gate.
- `Sources/dinoparity/` — the DINOv3 conditioning gate (`swift run -c release dinoparity`).
- `Sources/meshbake/` — end-to-end mesh+texture demo (`swift run -c release meshbake`).

## Build & run

Requires Xcode 27 (release since 2026-09-17; Swift 6.4, macOS 26+ SDK) and three sibling packages checked out
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
