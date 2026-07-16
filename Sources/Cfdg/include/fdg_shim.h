// C ABI over the vendored TRELLIS.2 o-voxel mesh -> flexible-dual-grid converter
// (upstream microsoft/TRELLIS.2 o-voxel/src/convert/flexible_dual_grid.cpp, MIT;
//  the torch entry point is replaced by this plain-buffer one — the algorithm body
//  is verbatim upstream).
#ifndef FDG_SHIM_H
#define FDG_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Convert a triangle mesh to the flexible dual grid (encode-direction O-Voxel step).
///
/// Inputs mirror the upstream torch entry:
///  - vertices: n_vertices*3 floats, ALREADY shifted by aabb-min (caller does `v - aabb[0]`)
///  - faces: n_faces*3 int32 vertex indices
///  - voxel_size: 3 floats
///  - grid_range: 6 int32 (min xyz, max xyz) — normally {0,0,0, gx,gy,gz}
///
/// Outputs are malloc'd by the callee; free each with fdg_free():
///  - out_coords: N*3 int32 voxel coordinates
///  - out_dual: N*3 float dual vertices (in the shifted mesh space)
///  - out_intersected: N*3 uint8 per-axis intersection flags
///
/// Returns N (number of active voxels), or -1 on error.
int64_t fdg_mesh_to_flexible_dual_grid(
    const float* vertices, int64_t n_vertices,
    const int32_t* faces, int64_t n_faces,
    const float voxel_size[3],
    const int32_t grid_range[6],
    float face_weight,
    float boundary_weight,
    float regularization_weight,
    int timing,
    int32_t** out_coords,
    float** out_dual,
    uint8_t** out_intersected);

void fdg_free(void* p);

#ifdef __cplusplus
}
#endif

#endif
