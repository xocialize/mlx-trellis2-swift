# Bake benchmarks & quality harnesses

Measurement infrastructure from the UV-unwrap campaign (see
`mlxengine-3d/UV-UNWRAP-METAL-PLAN.md` for the full record and reference
numbers). These are the foundation for the planned full-workflow metrics
harness — extend them rather than starting over.

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
