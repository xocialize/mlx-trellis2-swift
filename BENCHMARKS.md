# Bake benchmarks & quality harnesses

Measurement infrastructure from the UV-unwrap campaign (see
`mlxengine-3d/UV-UNWRAP-METAL-PLAN.md` for the full record and reference
numbers). These are the foundation for the full-workflow metrics
harness (below) — extend them rather than starting over.

## Full-workflow metrics harness (trellis2-run-engine)

Every `Trellis2Package.run` with `metricsPath` set (env `METRICS_JSON` in
`trellis2-run-engine`) emits ONE flat JSON record — per-stage wall seconds
(each timer window ends on that stage's `MLX.eval` barrier; MLX is lazy, so a
timer without a barrier mis-attributes to the next eval), token/voxel/face
counts, SDPA lanes, GPU peak. run-engine dumps a cost-sorted stage table.

Reference profiles (top_01.png, seed 0, first-run-of-batch, canary ≈20 s):
res512 = 54 s · res1024 ≈ 300 s (fp32 tex) / 216 s (bf16 tex) · res1536 ≈ 1153 s.
The HR SLat flows are 72–83 % of cascade e2e; bake is 3.5–5 %.

**Measurement protocol (each rule was violated once today and caught):**
- `ss_sample_s` is the built-in machine-speed canary — always fp32, identical
  work every run. Reject any cross-run comparison whose canaries differ.
- Sequential heavy runs thermally soak the SoC: first-run-of-batch canaries
  were 19.4–20.6 s all day; every subsequent run 36–55 s. Compare
  first-runs-of-batch only (or interleave + repeat the baseline).
- No WebGL/viewer tab open during timing (GPU contention), no builds.
- Stage timers include lazy flow-weight load on each process's first use.
- SDPA lanes are recorded per record (`slat_cfg_sdpa`, `tex_sdpa`) — state
  them with any timing, same rule as the xatlas lane.

**TF32 / NAX (M5-class GPUs, macOS ≥26.2, discovered 2026-07-20):** MLX dispatches
SDPA (and fp32 GEMMs) to the NAX kernel with TF32 operand rounding whenever
`MLX_ENABLE_TF32` is unset (it DEFAULTS ON) — so "fp32" on gen≥17 hardware is
TF32 storage-fp32/accumulate-fp32, proven by the voxel gate (`MLX_ENABLE_TF32=0`
changes coords/voxels). True steel-fp32 is 1.7–1.9× slower on every fp32 matmul
and produced no visual improvement (different sample; pale + a streak defect on
the test asset) → keep TF32 on. The quality cliff is HALF-PRECISION STORAGE
(fp16/bf16 casts), not operand mantissa. `mlx_tf32_env` is recorded per record.

**Sampler steps (wired 2026-07-20 — `Trellis2Configuration.steps` was a dead
knob before; env `STEPS`):** 12 (default) = reference quality. 10 = same color/
drape, mild trim-continuity erosion (broken dashes), ~15% e2e off cascade —
acceptable as an opt-in fast mode. 8 = REJECTED (white blotch texture artifacts).
Changing steps changes SS too → different sample; judge end-to-end, not per-stage.

**Corpus scaling + matting (t31 canonical, res1024, 2026-07-20):** e2e scales with
TOKEN COUNT, not tier; the O(N²) shapeHR fit from one asset predicts the others
within ~8%. tex-bf16 validated across all 5 classes (no washout). The run-engine
path has NO matting hook → background leak on raw white-bg inputs: a photographic
slab fused to the mesh (shoes, skirt — visually obvious), a color shift on the
asset (skirt teal-tinted vs true navy), and a SILENT token inflation even on
visually-clean outputs (dress carried ~1.8k background tokens with no visible
slab). MATTED baseline (border-connected flood-fill matte — `matte.py` in
mlxengine-3d/DEV/Trellis2Metrics; global threshold would hole out white-interior
assets like top/shoes):
  top 10,928 tok / 222 s · shoes 11,053 / 309 · skirt 20,990 / 763 ·
  dress 21,932 / 789 · hair 23,247 / 864   (sequential batch, thermally soaked —
  first-run-of-batch clean numbers are ~15-25 % lower).
Matting crops-to-bbox (raises tokens on clean assets, lowers them on contaminated
ones) — never compare matted-vs-unmatted timings.

**SDPA precision (controlled seeded A/B, mode-split visual review, 2026-07-20):**
tex flow (CFG-free) bf16 = output-identical shape path (bit-equal decoded
voxels) + visually indistinguishable texture → engine DEFAULT since this date
(`texAttention` nil = bf16; "fp32" opts out). CFG shape flows: fp32 REQUIRED —
fp16 AND bf16 both visibly degrade the final asset (texture washout, lost
grain, missing trim, seam artifacts) via concatCond + geometry propagation.
Per-dtype latents are cross-process deterministic → voxel-count equality is a
cheap no-render regression gate for future attention changes.

## unwrapbench — stage-split unwrap timing

```
swift run -c release unwrapbench [flags] <mesh.ply|mesh.glb> ...
  --remesh N      DC remesh resolution (default 256); --no-remesh skips prep remesh
  --target N      simplify face budget (default 120000)
  --parallel N    ALSO run the BSP-partitioned parallel xatlas and report speedup
  --provenance    ALSO run the provenance unwrap path (+ winding audits)
  --golden        pack-only round-trip test (xatlas output fed back via addUvMesh)
  --modelio       ModelIO addUnwrappedTextureCoordinates baseline
```
Prep mirrors MeshBake (remesh → simplify), then times xatlas
addMesh/computeCharts/packCharts separately. Emits human lines + one `JSON `
record per mesh. Bench-only PLY/GLB readers live in `Sources/unwrapbench/`
(the production package is deliberately import-free).

## bakeab — end-to-end bake quality A/B vs voxel ground truth

```
swift run -c release bakeab [octant] [samples N] [target N] [atlas N] [remesh N] [only xatlas|provenance]
```
Runs `MeshBake.run` per backend on the golden fixtures
(`/Volumes/Satechi/TrellisRedux/trellis2-port/goldens`), then scores each bake
against ground truth: area-weighted surface samples → baked color via the
mesh's own UVs vs trilinear samples of the source attr volume (BVH remap to
the raw shell — NOTE: shares the bake's remap, so it is blind to remap-class
errors; visual review is the complement, not an optional extra).

Reports: mean/p95 L2 color error, bad-sample fraction, per-face-texel-size
bad-rate buckets, atlas-spanning faces, sub-texel "fleck" faces, texel classes
(inpainted %, wall-flipped %), cross-backend probe (reads the xatlas bake at
provenance's bad points). Writes per-backend GLBs + atlas PNGs + color-coded
diagnostic GLBs (gray=rasterized, red=inpainted, blue=wall-flip) to the
session scratchpad.

## The A/B viewer (three.js) — the decisive instrument

A minimal page (recreate from the plan-doc history; scratchpad copies are
volatile) that loads the bakeab GLBs with mode toggles. **Mode-splitting is
the debugging method**:

| Mode | Isolates |
|---|---|
| unlit | texture content only |
| flat-lit gray | geometry + winding + normals (no texture) |
| normals view | orientation defects (backfacing = inverted colors) |
| mips ON/OFF | filtering/mip-chain interactions |
| FrontSide vs DoubleSide | holes vs flipped-winding patches |

Material overrides MUST set `side: THREE.DoubleSide` or the viewer fabricates
hole artifacts on thin-shell o-voxel geometry.

## Oracle comparison

The official TRELLIS.2 HuggingFace space GLB is the reference. Structure-diff
with trimesh: verts/faces, merged components, double-wall fraction (sampled
opposing-normal neighbor pairs), UV island count, covered UV area, material
stack (baseColor/metallicRoughness sizes, factors, doubleSided). Known
official profile: ~290k faces, 2048² baseColor + metallicRoughness,
metallicFactor 1.0, ~93% double-wall, ~22k islands.

## Rules learned the hard way

- Always state the xatlas lane with a timing (pypi / cumesh / Swift-vendored);
  wall time varies >15× on nominally identical inputs.
- Scalar metrics miss thin-line, renderer-visible defects — end with eyes on a
  lit render.
- Any ground-truth eval that shares machinery with the thing under test is
  circular exactly where that machinery is wrong.
