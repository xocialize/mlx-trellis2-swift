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

## Pixal3D mode (v0.13.0; lanes and weights v0.14.0; own package and typed cameras v0.15.0; turnaround preset and bf16 flows v0.16.0; AB-T-0485)

Pixal3D single view is released on the cascade tiers (`res1024`, `res1536`). Since v0.14.0:
- it has its own **lanes**, `Trellis2Configuration.pixal3dLane(_:)`, with measured footprints (table below);
- its weights auto-download from the public HF repo
  [`xocialize/pixal3d-mlx`](https://huggingface.co/xocialize/pixal3d-mlx) (MIT/Apache, ungated) next to
  `xocialize/trellis2-mlx`.

Posed multi-view is implemented and gated, but its samplers are 0.988 against the 0.99 gate, so its
`_mv` weights are not published (AB-R-0868); a multi-view lane needs them locally in
`pixal3dWeightsDir`. `res512` has no Pixal3D path (the P3 study below kept it stock).

**`Pixal3DPackage` (v0.15.0, engine contract 1.64.0 / mlx-engine-swift ≥ 0.79.0).** Pixal3D is its own
`ModelPackage` type with its own manifest. As mlx-engine decided in AB-A-0284 / AB-D-0230, it is a
separate lane, not a mode:
- provenance `xocialize/pixal3d-mlx`;
- the Pixal3D res1536 footprint;
- a surface `pixal3d-image-to-3d` advertising only `res1024`/`res1536`;
- `ImageTo3DControls(features: [.fov, .pose, .worldFrame, .cameraReport], fovXDegreesRange: 1…119)`.

Register it beside the stock lanes with an explicit id:

```swift
let pixal3d = try await engine.register(Pixal3DPackage.registration,
                                        configuration: .pixal3dLane(ImageTo3DContract.res1024), id: "pixal3d")
let response = try await engine.run(
    ImageTo3DRequest(image: photo, mode: ImageTo3DContract.res1024,
                     cameras: [ImageTo3DCamera(fovXDegrees: 20)]), package: pixal3d) as! ImageTo3DResponse
// response.camera = view 0's camera in the returned mesh's frame; response.frame = .front / .world
```

The typed fields win; the v0.13.0 `metaData` keys stay as the fallback for older hosts:
- `ImageTo3DRequest.cameras`: one per view, `fovXDegrees` alone for a single photo, `cameraToWorld` on
  every view for posed multi-view. It replaces `pixal3dFovDegrees` and `pixal3dCameras`.
- `ImageTo3DRequest.frame`: replaces `pixal3dFrame`.

The stock `Trellis2Package` surface declares no camera controls, so the engine, and the package itself,
refuse cameras there. Before v0.15.0 a Pixal3D lane was `Trellis2Package` + `pixal3dLane`; that still
runs, but `Pixal3DPackage` is the supported registration.

**Lanes (v0.14.0).** A Pixal3D lane backs the same `imageTo3D` capability as the stock lanes. Every request
on it runs Pixal3D; it never loads the stock SS or SLat flows. It declares two weight sources: the
stock snapshot (decoders, DINOv3) and `xocialize/pixal3d-mlx`.

| Lane | Resident | Activation | Charged | Worst measured peak (phys) |
|---|---|---|---|---|
| Pixal3D res1024 | 32 GB | 38.4 GB | **70.4 GB** | 51.58 GB (shoes, 15,242 tokens) |
| Pixal3D res1536 | 32 GB | 43.3 GB | **75.3 GB** | 66.90 GB (hair, 28,243 tokens) |

**bf16-resident flows (v0.16.0, opt-in).** Set `Trellis2Configuration.flowWeightsBF16 = true` (driver:
`FLOW_WEIGHTS=bf16`). The flow weights then stay in their stored bf16 instead of being cast to fp32 at
load, and MLX promotes them inside each fp32 matmul. bf16→fp32 is exact, so the output is
**byte-identical**: the same GLB on three runs.

| Run | Resident (bf16 / fp32) | Peak phys (bf16 / fp32) | Time (bf16 / fp32) |
|---|---|---|---|
| res1024 dress | 20.3 / 31.2 GB | 30.7 / 41.6 GB | 128 / 114 s |
| res1024 hair | 20.0 / 31.0 GB | 32.4 / 43.8 GB | 155 / 143 s |
| res1536 hair | 20.4 / 31.4 GB | 55.9 / 66.9 GB | 564 / 490 s |

It saves 11 GB everywhere and costs 8–15 % time (the per-forward weight conversion). A Pixal3D lane with
it is charged 21 GB resident: **59.4 GB** at res1024 and **64.3 GB** at res1536. It is off by default,
because the memory-for-speed trade is the host's call. Stock lanes with it keep their fp32 charge
(unmeasured). An int8 lane (ComfyUI ships int8 flows at half of bf16) is not in reach without new
quantized kernels and a new parity study, so it is not attempted.

Activation is 4 GB + 1.2 MB per HR token, charged at the tier's hard token ceiling (28,672 / 32,768),
as for stock. That line clears every one of the 10 measured runs by at least 5.6 %. The Pixal3D
res1536 lane is the package's largest, so the manifest footprint is now 75.3 GB. Measured with a
release build on the M5 Max with 128 GB: `pixal3dOnly` lanes through `trellis2-run-engine`, five corpus
items per tier, one fresh process per run, matted inputs. The per-run table is in `Trellis2Footprint`.

**res512 (P3 study).** Upstream ships no Pixal3D tex-512 flow. Two candidates ran Pixal3D SS +
shape-512 on the 32³ grid, decoded at 512, and differed only in texture:
- **hybrid:** the stock tex-512 flow;
- **tex-1024 on 32³:** the Pixal3D tex-1024 flow on the 32³ grid.

`pixal3deval` renders each GLB through the run's camera and scores it against the photo. The rule
was fixed before the runs: every corpus item at silhouette IoU ≥ 0.95, and front-view colour ΔE
within +2 of Pixal3D res1024.

| item | res1024 IoU / front ΔE | res512 IoU | hybrid front ΔE | tex-1024@32³ front ΔE |
|---|---|---|---|---|
| dress | 0.983 / 8.6 | 0.968 | 8.2 | **11.2** |
| hair | 0.977 / 8.6 | 0.964 | 8.4 | 8.4 |
| skirt | 0.987 / 13.7 | 0.982 | 15.3 | 16.7 |
| shoes | 0.973 / 10.2 | 0.958 | 11.9 | 10.7 |
| top | 0.988 / 15.1 | 0.979 | **30.5** | **18.1** |

The geometry holds at 512 (IoU 0.958–0.982). Neither texture passes on every item: the hybrid
paints the sailor collar red (its texture is not pixel-aligned), and the tex-1024 flow off-resolution
misses on the dress and the top. So `res512` stays stock TRELLIS.2, and a Pixal3D request at res512
is refused. The code path (`generatePixal3D(tier: .res512, res512Texture:)`) stays in the library,
unexposed, for future work. A res512 run peaks at 28–31 GB. res1536 also got the alignment check:
IoU 0.974–0.989 on all five items.

**What is different from stock TRELLIS.2**

- Each flow block adds `proj_linear(proj)` to its cross-attention output
  (`ModulatedTransformerCrossBlock.projW/projB`). `proj` is the image feature map sampled where each
  voxel lands in the photo: DINOv3 patch tokens, and in the shape and texture stages also the NAF
  upsampled map (2048 channels). Cross-attention sees only the 5 global DINOv3 tokens (CLS + 4
  registers). Negative conditioning is zeros, so the unconditional branch adds only the bias.
- Four conditioners: SS (DINOv3 512, dense 16³, no NAF), shape 512 (DINOv3 512 + NAF 512, 32³),
  shape 1024 (DINOv3 1024 + NAF 512, hr/16), texture 1024 (DINOv3 1024 + NAF 1024, hr/16).
- NAF (`valeoai/NAF` @ `37f2dfc`, Apache-2.0, 0.66 M params) is ported in `NAF.swift`, not NATTEN:
  a small conv encoder with axial RoPE, then 9×9 dilated neighbourhood attention from the HR
  pixels to the DINOv3 patches, with NATTEN's border rule. The HR map is never materialized: the
  corner pixels the projection needs are gathered band by band (`NAF.sample`).
- The cascade requantizes with `((c + 0.5) / 512 · (grid − 1)).round()` (stock: `· grid` and
  truncate), `Pixal3DProjection.requantize`.
- No 512 mode upstream: Pixal3D runs on the cascade tiers (`res1024`, `res1536`) only. `res1024` is
  the parity-gated one; `res1536` runs the same code at grid 96, has no golden, and passes the
  alignment check (IoU ≥ 0.97 on the corpus).
- Single view crops with a 1.1 margin (stock: none). The camera model maps the unit cube to that
  framing, so the margin is part of the camera.

**Camera.** Single view takes a horizontal FOV, default 20°; `distance = 0.5 / (mesh_scale · tan(fov/2))`,
`mesh_scale` 1.

**The 20° default is proven only on matted product shots.** P1 measured a fixed 20° aligning as well as
or better than MoGe-2 on the five corpus items (AB-R-0865), but those are studio product photos; MoGe-2
read them as 30–49°. For in-the-wild photos (a phone shot, a wide lens), the host should supply the
FOV, for example from EXIF focal length. Upstream defaults to a MoGe-2 estimate, and ComfyUI's template
wires MoGe-2 with a 49.13° fallback. MoGe-2 is not ported, and nothing here estimates the FOV. Posed multi-view takes one camera-to-world matrix per view (Z-up world, the
camera looks down its −Z with +Y up; view 0 is the canonical front
`[[1,0,0,0],[0,0,−1,−d],[0,1,0,0],[0,0,0,1]]`). Every view is placed relative to view 0
(`calc_mat_i = F(d₀)·inv(C₀)·C_i`), the projected features and global tokens are averaged over
views with no visibility test, and the `_mv` weights run. Views are not cropped (the cameras fix
the framing).

**Turnaround preset (v0.16.0).** For turnaround sheets with named views, use
`Pixal3DPackage.turnaroundCameras(["front", "left", "back", "right"], fovXDegrees: 20, pad: 1.1)` (or
`Pixal3DCamera.turnaround`). It builds the contract cameras: a Z-up orbit at azimuth 0/90/180/270,
elevation 0, distance `pad · 0.5 / tan(fov/2)`, and view 0 should be the front. This is the shape of
ComfyUI's `Pixal3DMultiViewConditioning`.

`pad` is the framing: the unit cube spans 1/pad of each view.
- **1.1:** views cropped to the subject with a 1.1 margin (ComfyUI; upstream's example rig, d = 3.1192).
- **1.0:** full-width framing, as in the P1 oracle rig (d = 2.8356), which the preset reproduces exactly
  in `pixal3dparity`.

The views must be framed to match: the package resizes posed views without cropping.

**Output frame (decision).** Pixal3D's mesh is in the photo's camera frame, not upright: the
decoder's voxel frame has the photo's up on +Y and the camera on +Z. The package always rotates it
into Pixal3D's own world frame, `(x, y, z) → (x, −z, y)`: Z-up with the front camera at −Y, which
is the axis convention of the stock output, so `yUp`, the bake and every Z-up consumer read it the
same way. It does not try to make the object upright: "up" is the photo's up, and a product shot
taken from above comes out tilted by the camera's elevation; no gravity estimate is made. For posed
multi-view, `pixal3dFrame: "world"` maps the mesh into the frame the request's cameras are written
in (`C₀·inv(F(d₀))`, rigid), which is upright when the host's cameras are gravity-aligned. The metrics
record carries the camera (`pixal3d_fov_deg`, `pixal3d_distance`, `pixal3d_camera0_to_mesh`: view 0's
camera-to-world in the returned mesh's frame), so a host that knows the photo's elevation can
re-orient. `Pixal3DCameraReport` carries the same from `generatePixal3D`.

**Request keys** (`ImageTo3DRequest.metaData`, with the other per-request options):

| Key | Values | Default |
|---|---|---|
| `pixal3d` | bool | false (`true` on a `pixal3dOnly` registration) |
| `pixal3dFovDegrees` | number in (0, 120) | 20 |
| `pixal3dFrame` | `"front"` or `"world"` | `"front"` |
| `pixal3dCameras` | array of 16-number row-major camera-to-world matrices, image first then `additionalViews` | none |

`additionalViews` without `pixal3dCameras` is refused in Pixal3D mode (pose-free views are the stock
path's `additionalViews`). `res512` is refused. Configuration:
- `pixal3dLane(_:)`: a Pixal3D lane (`pixal3dOnly`, pinned tier, Pixal3D footprint, both weight sources);
- `pixal3dWeightsDir`: where the converted weights are. Default: the ModelStore copy of
  `xocialize/pixal3d-mlx`, then the stock snapshot directory;
- `pixal3dMultiView`: preload the `_mv` set (local only).

A stock registration can still take `pixal3d: true` per request when the Pixal3D weights are on disk;
it loads them lazily and is charged the stock footprint, so use a lane for scheduling. Since v0.15.0 the
camera travels as typed contract fields (`ImageTo3DRequest.cameras` / `.frame`, contract 1.64.0); the
`metaData` keys are the fallback.

**Weights.** `Tools/pixal3d/convert_weights.py` converts the eight flows (fp32 on disk, 5.4–5.5 GB
each) to bf16 like the stock flows (2.68 GB SS, 2.77 GB each SLat) with the stock module keys
(`cross_attn.cross_attn_block.*` → `cross_attn.*`, `rope_phases` dropped, `proj_linear` kept),
and `naf_release.pth` to `naf.safetensors`. Output names: `pixal3d_struct_flow`,
`pixal3d_shape_flow_512`, `pixal3d_shape_flow_1024`, `pixal3d_tex_flow_1024`, each with an `_mv` twin,
plus `naf`. Tencent's `LICENSE` and `NOTICE` and NAF's Apache-2.0 text travel with them
(AB-D-0229). The single-view set (4 flows + `naf`) is in the public HF repo `xocialize/pixal3d-mlx`; the `_mv` set is local only.

**Gate.** `swift run -c release pixal3dparity` checks every stage against the CUDA goldens of
AB-R-0865 (mirrored to safetensors by `Tools/pixal3d/convert_goldens.py`): camera, DINOv3 tokens,
projection, NAF crops (centre and edge), banded NAF sampling, end-to-end conditioning,
`proj_linear` block I/O, velocities at steps 0/5/11, each sampler, the HR requantize, the mesh
decoded from the golden latents, and the whole chain. Env: `PIXAL3D_RUN`, `PIXAL3D_GOLDENS`,
`PIXAL3D_WEIGHTS`, `TRELLIS2_WEIGHTS`, `PIXAL3D_SECTIONS`, `PIXAL3D_STAGES`, `PIXAL3D_JSON`.
`trellis2-gate` checks the request keys and the camera maths.

**Gate results (2026-11-14, M5 Max 128 GB, release; goldens: HF Job `6af8e2ae23985fb9cd82a3b0`).**
Thresholds are this repo's: 0.999 per op, 0.99 for DINOv3 / full forwards / samplers, coords exact or
≤ 0.2 %. The oracle ran its flows in bf16; ours run fp32.

| Stage | dress, single view | shoes, single view | dress, posed 3 views |
|---|---|---|---|
| camera (distance; multi-view `calc_mat`) | exact | exact | exact |
| preprocessing (our crop/resize → DINOv3 tokens 512 / 1024) | 0.99990 / 0.99971 | 0.99988 / 0.99975 | 0.99970 / 0.99952 |
| DINOv3 tokens 512 / 1024, `z_global` | 0.999999 | 0.999999 | 0.999998 |
| projection (golden patch maps) | 1.000000 | 1.000000 | 1.000000 (mean of views) |
| NAF crops, centre and edge, all 3 stages | 1.000000 | 1.000000 | 1.000000 (last view) |
| conditioning end to end, every stage | 0.999999 | 0.999999 | 0.999999 |
| `proj_linear` block I/O, blocks 0/15/29 | ≥ 0.99964 | ≥ 0.99909 | ≥ 0.99941 |
| velocities, steps 0/5/11 (cond, uncond, guided) | ≥ 0.99769 | ≥ 0.99733 | ≥ 0.99749 |
| samplers SS / shape 512 / shape 1024 / tex | 0.9987 / 0.9961 / 0.9949 / 0.99994 | pass, ≥ 0.99597 | 0.9981 / **0.9876** / **0.9881** / 0.99985 |
| SS decode of the golden latent | exact | exact | exact |
| HR requantize | 1 token in 8,546 | exact | 2 tokens in 16,050 |
| mesh decoded from the golden latents, F@0.5 voxel | 0.9991 | 0.9991 | 0.9986 |
| anchored chain latents shape 512 / 1024 / tex | 0.9961 / 0.9949 / **0.9885** | 0.9972 / 0.9945 / 0.9961 | **0.9883 / 0.9884 / 0.9890** |
| anchored chain mesh, F@1 voxel (Chamfer, voxels) | 0.983 (0.18) | 0.987 (0.18) | 0.975 (0.26) |
| free chain mesh, F@1 voxel (measured) | 0.686 | 0.645 | 0.439 |

Bold is under its gate. What the misses are:
- **Free chain.** With golden noise the SS sample matches at 0.998–0.999, but 16–47 voxels flip
  through the occupancy threshold. They are low-margin voxels: median |logit| 7–17 against 109–122
  for occupied ones. On a different token set the CFG-7.5 shape sampler lands on a different sample.
  Inside our own port, moving only the token set (oracle coords → our coords) gives the same drop
  (0.968 vs 0.966 on the dress). The gate injects the oracle's topology at the two coordinate
  decisions, as this repo's parity methodology does for topology-adaptive stages; the free chain is
  measured, not gated.
- **Dress texture in the anchored chain (0.9885).** It is inherited from the shape latent (0.9949,
  inside its gate) through `concat_cond`. Our conditioning with the oracle's shape latent gives
  0.99992.
- **Multi-view shape samplers (0.988).** Per-step velocities are as close as in single view
  (0.99995 cond/uncond), but the multi-view CFG trajectory is more sensitive. Changing only the SDPA
  cast (fp32 → bf16) moves our own multi-view shape-512 sample by 1 − cos = 1.1e-3, against 9e-5 in
  single view (about 12×). The anchored multi-view mesh passes (F@1 0.975).

**Generate time and memory on this Mac** (engine driver, `pixal3dOnly` registration, release, first
job after prepare, 1024 lane, default bake at ≤ 300k faces):

| Run | HR tokens | Generate | Peak `phys_footprint` | Resident between jobs |
|---|---|---|---|---|
| Pixal3D dress, single view | 8,500 | 113 s | 42.2 GB | 31.3 GB |
| Pixal3D shoes, single view | 15,242 | 221 s | 51.9 GB | 31.1 GB |
| Pixal3D dress, posed 3 views | 15,846 | 234 s | 52.5 GB | 31.3 GB |
| stock TRELLIS.2 dress (reference) | 22,283 | 375 s | 58.2 GB | 30.3 GB |
| stock TRELLIS.2 shoes (reference) | 11,054 | 156 s | 43.1 GB | 30.0 GB |

Time and memory follow the HR token count, as for stock. Pixal3D makes fewer tokens than stock on the
dress and more on the shoes, so neither is uniformly cheaper. The declared lane footprints are still
stock's; per-lane Pixal3D footprints are P3.

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
| `normalMap` | bool: also write a tangent-space normal map (`normalTexture` + TANGENT) | false |
| `normalMapSize` | 512, 1024, 2048, 4096 | the atlas size |
| `aoMap` | bool: also write an ambient-occlusion map (`occlusionTexture`) | false |
| `aoMapSize` | 512, 1024, 2048, 4096 | the atlas size |
| `aoRays` | 1…256 rays per texel and side | 64 |
| `aoDistance` | (0, 2], × the mesh's bounding-box diagonal | 0.25 |

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

## Normal and AO maps

Two opt-in outputs on the GLB, off by default. A request without them gives the same bytes as
before they existed (checked against v0.13.0 and v0.16.0, below). Both packages take them (the
`pixal3d` lane runs the same bake). `normalMapSize` and `aoMapSize` are capped by
`maxAtlasSize`, like `atlasSize`. Neither needs `texture`.

**Normal map (`normalMap`).** The decimated mesh loses the decoded surface's fine relief: lace,
seams, fold edges. The bake keeps the raw dual-grid shell (10–18M faces at res1024, the same surface
the colour is sampled from) and, per texel, casts a short cage ray from just outside the low-poly
surface back along its normal; the outermost raw hit is the matching surface (closest point where
the ray misses). The raw normal there goes into the texel's tangent frame and is written as a glTF
tangent-space map (+X right, +Y image-up, +Z out) at 2× supersampling, box-averaged down. The GLB
gets a TANGENT attribute: Lengyel per-vertex tangents, Gram–Schmidt against the written normals,
handedness from the image-up bitangent (−dP/dv, glTF v runs down the image), the convention
MikkTSpace-based tools use for glTF. The bake uses exactly the frame it writes, so a viewer that
reads TANGENT reconstructs the baked normal.

- The raw shell's winding is not consistent. Its vertex normals are summed with each face's sign
  aligned at the vertex, so oppositely wound neighbours reinforce, and the result is turned to the
  texel's side. With plain area-weighted normals, 51 % of the dress's texels fell back to flat.
  With this sum, 2.9 % do.
- A raw normal more than ~75° off the low-poly normal falls back to flat (a side wall or crevice
  that the ray grabbed).

**AO (`aoMap`).** Ray-cast on the output mesh with the BVH from `MLXMesh`: a new any-hit kernel
(`BVH.occlusion`) generates cosine-weighted directions in the kernel (Hammersley set, per-texel
rotation hashed from the index, so no ray buffer and no randomness) and stops a ray at its first hit.
The output mesh is wound consistently only per chart (xatlas re-normalizes winding per chart, and
the double-sided material hides inward charts in a viewer), so each texel casts both hemispheres and
keeps the less occluded one. The inner side of a thin shell or a closed body sees the other wall at
short range and is ~fully occluded. Following the normal alone blacked out whole charts.

With `pbrMaps` on and the AO map at the atlas size, AO goes into the metallic-roughness texture's
R channel, and `occlusionTexture` points at that texture (glTF ORM). Otherwise it is its own
texture.

Cost on the M5 Max, measured with `bakemaps` on the Pixal3D res1024 dress (292,520 faces, raw shell
10.7M faces). Other GPU work was running on the machine during some runs; the best run is shown.

| map edge | normal map (2× supersampled) | AO map (2 × 64 rays per texel) |
|---|---|---|
| 1024 | 0.45 s (2.9M sub-texels) | 0.30 s (0.78M texels) |
| 2048 | 1.6 s | 1.2 s |
| 4096 | 5.7 s | 4.7 s |

The stock res1024 shoes (270,288 faces) at 1024: 0.46 s and 0.32 s. Tangents take under 10 ms. A
1024 dress GLB with `pbrMaps` grows from 13.1 MB to 19.6 MB with both maps.

**Memory.** The maps run after the colour bake. The colour bake's raster arrays are scoped to free
before then, and the maps rasterize one jitter pass at a time. Measured peak phys:
- `bakemaps`, atlas 4096: 28.91 GB without maps, 33.68 GB with both maps at 4096.
- End to end on a res512 lane with `maxAtlasSize` 4096, dress, atlas 4096 + `pbrMaps` + both maps
  at 4096: **43.27 GB** (43.54 GB on v0.13.0). That is level with the 43.59 GB above without maps,
  so the 48 GB charge holds.
- Pixal3D res1024 dress (`pixal3d` lane), maps at 1024: 42.10 GB (41.99 GB without them).
- Stock res1024 shoes, maps at 1024: 43.04 GB. The HR stage stays the peak on the cascade lanes.

**Checked.** With the maps off, the GLB bytes are identical to the release the branch was built on:
v0.13.0, and again after rebasing onto v0.16.0 (Pixal3D res1024 dress sha256 `db75e0d5…`, stock
res512 shoes `b6c9bbff…`). With the maps on, two bakes give identical bytes. The Khronos
glTF validator raises nothing on the TANGENT accessor, the textures or the ORM sharing. The
zero-length NORMAL entries it reports (3 on the Pixal3D dress, 33 on the res1024 shoes, the same
counts on the v0.13.0 GLBs) predate this change.

`swift run -c release bakemaps <bake-input.safetensors> <out.glb> --normal N --ao N [--pbr]` re-bakes
saved decoder outputs (the arrays `MeshBake.run` takes), so the maps can be iterated without running
diffusion. `--repeat` bakes twice and compares the bytes: identical with both maps on.

## Engine and mlx-swift pins

From v0.15.0 the package needs mlx-engine-swift `from: "0.79.0"` (contract 1.64.0: typed imageTo3D cameras,
for `Pixal3DPackage`). From v0.11.0 to v0.14.0 it needed `from: "0.74.1"`, and the fleet's mlx-swift fork,
`xocialize/mlx-swift` `from: "0.32.4"`. The fork is upstream 0.32.3 plus ml-explore/mlx-swift#501,
the quantized-load leak fix (AB-D-0208).

v0.10.0 still named `ml-explore/mlx-swift`. That clashed with the engine's fork URL, and SwiftPM
picked upstream, so apps lost the leak fix.

## Layout

- `Sources/TRELLIS2/` — the library (all ported components).
- `Sources/parity/` — the parity harness. `swift run -c release parity` runs every gate.
- `Sources/dinoparity/` — the DINOv3 conditioning gate (`swift run -c release dinoparity`).
- `Sources/pixal3dparity/` — the Pixal3D gate against the CUDA goldens (`swift run -c release pixal3dparity`; README "Pixal3D mode").
- `Sources/pixal3deval/` — renders a Pixal3D GLB through its camera and scores it against the photo (IoU, colour).
- `Tools/pixal3d/` — Pixal3D weight and golden converters.
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
