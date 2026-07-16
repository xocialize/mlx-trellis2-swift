# SW6 — MLXEngine `ModelPackage` packaging

Wraps the parity-verified TRELLIS.2 Swift-MLX neural core (`TRELLIS2` target) as an MLXEngine
`imageTo3D` `ModelPackage` (`Trellis2Kit` target). Stage 2 of the `mlx-swift-integration` workflow.
The neural port is done and verified; this layer is the engine packaging (manifest, license gate,
weight sourcing/auto-materialization, `load()`/`run()`, born-clean gates).

## What this adds

| Piece | File |
|---|---|
| `Trellis2Configuration` (`PackageConfiguration`, `ModelStorable`, `QuantConfigured`, `WeightSourcing`, `WeightPrewarming`) | `Sources/Trellis2Kit/Trellis2Package.swift` |
| `Trellis2Package: ModelPackage` (manifest + `load()`/`run()`) | `Sources/Trellis2Kit/Trellis2Package.swift` |
| First-run weight download (HubClient → ModelStore layout) | `Sources/Trellis2Kit/WeightMaterializer.swift` |
| DINOv3 image preprocess → NCHW `[1,3,512,512]` | `Sources/Trellis2Kit/ImagePreprocess.swift` |
| In-package SS RoPE grid + seeded noise (replaces `.npy` goldens) | `Sources/TRELLIS2/SSGridPhases.swift` |
| GLB **bytes** path for `CanonicalOutput.mesh` | `GLTFExport.glbData(...)` |
| Consolidated-snapshot pipeline loader | `Trellis2Pipeline(consolidatedDir:)` |
| Offline conformance gate (C0–C13 + MAT-1..5 + CAN-1..3) | `Sources/trellis2-gate/main.swift` |
| Weight consolidation tool | `Sources/trellis2-consolidate/main.swift` |
| Engine-driven GPU e2e driver | `Sources/trellis2-run-engine/main.swift` |

## Manifest (Trellis2Package)

- **License (two-layer):** weight `C7 = .dinov3` (LicenseRef-DINOv3, allowlisted → `.permissiveOnly`
  admits; **display "Built with DINOv3"** wherever shipped); port code `C8 = .mit`. DINOv3 is the
  most-restrictive component and governs (TRELLIS.2 itself is MIT).
- **Provenance:** `microsoft/TRELLIS.2-4B`, tier 3 (multi-component pipeline).
- **Requirements:** `.metalGPU`; macOS floor 26.0; chip floor `.pro`.
- **Footprint (split, contract 1.14):** `residentBytes = 21 GB`, `peakActivationBytes = 11 GB`.
  Grounded in a **measured engine e2e run** (below); still needs an in-app phys_footprint
  re-baseline before the registry Eff flips to ✅.
- **Specialty:** `3d-generation` (strength 1.0).
- **Surfaces:** `ImageTo3DContract` `res512` / `res1024` / `res1536` (res512 is the validated tier).

## Gate results (offline, `swift run trellis2-gate`)

All pass:

- **C0–C13 static:** C0 contract 1.20.0 · C1 single `imageTo3D` surface · C2/C11 descriptor
  well-formed · C7 weight license admitted · C8 port-code license · C6 specialty · C10
  footprint-split + backend + chip + OS · C13 registration IoC.
- **MAT-1..5** (`MaterializationConformance`): ModelStorable · non-empty `WeightSourcing` ·
  role/repo hygiene · fresh-machine posture (nil store ⇒ source missing) · explicit-paths satisfy.
- **CAN-1..3** (`CancellationConformance`): pre-cancelled `run()` surfaces `CancellationError`
  unchanged (entry checkpoint is the first act of `run()`) · long-run implied · cadence
  `denoise/step` + `decode/layer`.

The gate is a **CLI lane** (`swift run`), not XCTest, per the skill — nothing runs under
`swift test` (metallib unreliable there). It evaluates no Metal kernels and touches no weights.

## Engine e2e validation (`trellis2-run-engine`, MLXServeEngine, GPU)

The full coordinator path was run end-to-end against the locally-consolidated snapshot:

```
register (license C7/C8 admitted) → prewarm paged 8 files / 11 GB in 1.1s
prepare (load) OK in 1.1s | governor charged 21.00 GB resident
run OK in 123s | MLX-pool peak 31.17 GB
mesh: verts=94502 faces=118248 vertexColors=false bytes=5,952,036
evict → resident 0.00 GB   (unload() + clearCache; phys falls)
```

The output GLB validates as glTF 2.0 with POSITION/NORMAL/TEXCOORD_0 + an embedded base-color
texture (OPAQUE) — the UV-textured output (the improvement over the old port's vertex colors).
This exercises load() (consolidated-dir resolution + normalization.json), run() (preprocess →
DINOv3 → in-package SS grid/noise → samplers → decoders → MeshBake → GLB), the license gate,
governor charging, and clean eviction.

## Footprint — measured basis and the re-baseline

Grounded in the engine run above (resident 21.00 GB charged; MLX-pool peak 31.17 GB ⇒ activation
~10.2 GB → declared 11 GB). Still flagged for re-measure. Basis:

- `residentBytes ≈ 21 GB`: the pipeline holds **all 7 components resident at once (no per-stage
  eviction)** and every model constructor casts weights to **fp32** (the parity dtype). On-disk
  bf16/fp16 (~11 GB: 3×1.3B DiTs 7.8 GB, 2 sparse decoders 1.9 GB, ss_dec 0.15 GB, DINOv3 1.2 GB
  already-fp32) → ~21 GB fp32-resident. Declaring `.bf16` quant with an fp32-resident floor is
  deliberate: the manifest reports the memory this build actually occupies.
- `peakActivationBytes ≈ 8 GB`: transient sparse full-attention over ~19.5k SLat tokens (O(T²)) +
  the ~7.85M-voxel shape/tex decodes, anchored to the old port's **measured** res512 total ~14 GB
  (which held *bf16* weights).

**Re-baseline procedure:** run the package through the in-app `MLXEngineTestKit` footprint probe
(true `task_vm_info.phys_footprint`, not the CLI `GPU.peakMemory`, which under-reads phys ~2.7×),
read the floor **post-load** and the peak during one res512 run, then flip the registry **Eff** cell.
A P0 efficiency follow-up (bf16-resident weights + per-stage load→use→evict) would roughly halve
`residentBytes`.

## Consolidated weights — `xocialize/trellis2-mlx` (to be published by the user)

USER DECISION: republish as a VERIFIED consolidated snapshot in the **ORIGINAL microsoft/facebook
key layout** (NOT the old port's renamed-key conversion). So consolidation is a pure per-file
**byte copy** under canonical filenames (keys + dtype preserved; each component self-verifies via
its parity gate) + a `normalization.json` distilled from `pipeline.json`. No key remap.

Produce locally with:

```
MODELS=/Volumes/Satechi/TrellisRedux/models OUT=<dir> swift run -c release trellis2-consolidate
```

**Exact file list for the repo (validated — 10 GB total):**

| Repo file | Bytes | Assembled from |
|---|---|---|
| `dino.safetensors` | 1212.6 MB | `facebook/dinov3-vitl16-pretrain-lvd1689m/model.safetensors` |
| `struct_flow.safetensors` | 2584.4 MB | `microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors` |
| `struct_dec.safetensors` | 147.6 MB | `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors` |
| `shape_flow_512.safetensors` | 2584.6 MB | `microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors` |
| `shape_dec.safetensors` | 948.5 MB | `microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors` |
| `tex_flow_512.safetensors` | 2584.7 MB | `microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors` |
| `tex_dec.safetensors` | 948.5 MB | `microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors` |
| `normalization.json` | ~3 KB | `microsoft/TRELLIS.2-4B/pipeline.json` → `{shape,tex}_slat_normalization{mean,std}` |

Plus, before publishing: `LICENSE` (MIT, for the TRELLIS.2 port code + weights), `NOTICE`, and
`DINOv3_LICENSE.md` (the DINOv3 license text — redistribution under §1.b requires shipping it), and
the "Built with DINOv3" attribution obligation carried through. The repo should be **gated
(auto-approval)** matching the old port; first-run materialization env-detects `HF_TOKEN`.

Cascade HR (res1024/res1536) DiTs (`shape_flow_1024` / `tex_flow_1024`) are intentionally **not**
included yet — res512 is the validated tier; add those two files (+ probe entries) when the cascade
tiers are wired.

## Dependency conflict + resolution

`mlx-engine-swift` (0.30.0) declares `mlx-swift from: 0.31.5`; the spike's sibling `../mlx-swift`
is **0.31.3**. Resolution: mlx-swift is consumed via `.package(path:)`, and a **path dependency
overrides the version constraint** (SPM uses the on-disk package for that identity and does not
enforce `from:` against it), so the graph resolves on 0.31.3. SPM emits a *warning* ("Conflicting
identity for mlx-swift … will be escalated to an error in future versions") — benign today. The
offline contract build compiles only `MLXToolKit` (Foundation-only), so the mlx-swift version is
irrelevant to it; the full engine graph builds and links on 0.31.3. If a future engine needs a
0.31.5 symbol, bump the sibling `../mlx-swift` to a ≥0.31.5 tag (shared with `mlx-swift-mesh`;
re-init its `mlx/mlx-c` submodule after).

## Build / run

```
swift build --target Trellis2Kit          # offline contract (compiles vs MLXToolKit)
swift run   trellis2-gate                  # C0–C13 + MAT + CAN (offline, no GPU/weights)
swift run -c release trellis2-consolidate  # assemble the consolidated snapshot locally
IMG=<png> WEIGHTS_DIR=<consolidated-dir> swift run -c release trellis2-run-engine   # GPU e2e
```

## TODOs handed back (user + coordinator)

1. **Publish weights** — upload the consolidated snapshot to `xocialize/trellis2-mlx` (needs the
   user's HF auth; gated repo + DINOv3 license files + "Built with DINOv3"). Not done here.
2. **App cutover** — update the MLXEngine3D app's residual code refs to the new clean names
   (`Trellis2Kit` / `Trellis2Package` / `Trellis2Configuration`). Coordinated with the user; the
   package link was already removed on the app side.
3. **Model-registry row** — update `mlx-engine-swift/docs/model-registry.md`'s `mlx-trellis2-swift`
   row to point at this successor (it still describes the OLD vertex-color port). Set Avail per
   publish, Val per the engine e2e run, Eff ⬜ pending the footprint re-baseline.
4. **BiRefNet preprocess** — ✅ DONE 2026-07-15 (T0.4). `Trellis2Configuration.matting` is an
   optional app-injected hook (`Trellis2Matting = @Sendable (Image) async throws -> Matte` —
   canonical MLXToolKit artifacts only, so no cross-package ModelPackage dependency (C13) and the
   offline build is untouched; excluded from the config's Codable surface). `run()` routes every
   view WITHOUT a usable alpha (opaque RGB/JPEG, or RGBA whose alpha is uniformly 255 — upstream
   `preprocess_image`'s `has_alpha` test) through `ImagePreprocess.mattedIfNeeded`: cap ≤1024²
   (upstream resizes before rembg), matte, graft the matte in as the alpha channel → the existing
   threshold-bbox crop / RGB×alpha-on-black path. The app composes the shipped BiRefNet `matting`
   package here (e.g. `{ try await engine.run(MattingRequest(image: $0), package: birefnetID) }` —
   nested `engine.run` is safe: the admission gate is released before a run executes). Verified
   e2e through MLXServeEngine: RAW 3024² phone-photo-style JPEG (cluttered background) →
   BiRefNet fast@1024 matte (4.6 s, clean silhouette) → res512 generation 84 s → 152k-vert
   textured GLB (`TrellisDev/out_mesh_engine_t04_matted.glb`); skip-path verified (pre-masked
   RGBA input: hook not invoked, alpha honored as before). With no hook injected, a raw input
   still falls back to the full-frame resize (unchanged prior behavior).
5. **Footprint re-baseline** — measure `residentBytes`/`peakActivationBytes` with the in-app
   phys_footprint probe (replaces the documented estimate above).
6. **Cascade tiers** — wire res1024/res1536 (add `shape_flow_1024`/`tex_flow_1024` to the snapshot +
   probe + the cascade code path); today all three modes map to the res512 pipeline.
