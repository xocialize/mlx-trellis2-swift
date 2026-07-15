#pragma once
// C-ABI shims over xatlas APIs that take varargs (PrintFunc) or have no
// userData slot (ParameterizeFunc). These let Swift register fixed C function
// pointers and route them to closures stored in a global slot inside the shim.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// SetPrint bridge --------------------------------------------------------

typedef void (*xatlas_swift_print_cb)(const char *msg);

// Install a Swift print callback. Pass NULL to remove. `verbose` is
// forwarded to xatlas::SetPrint.
void xatlas_install_print(xatlas_swift_print_cb cb, int verbose);

// ParameterizeFunc bridge ------------------------------------------------

typedef void (*xatlas_swift_param_cb)(
    const float *positions,
    float *texcoords,
    uint32_t vertexCount,
    const uint32_t *indices,
    uint32_t indexCount);

// Install a global parameterize callback. xatlas's ChartOptions::paramFunc has
// no userData slot, so this slot is process-global. Caller must serialise
// across atlases. Returns a C function pointer suitable for ChartOptions.paramFunc
// (or NULL to clear).
void xatlas_install_parameterize(xatlas_swift_param_cb cb);

// The C function pointer to assign to ChartOptions::paramFunc. Returns NULL
// when no callback is installed.
void *xatlas_parameterize_thunk(void);

#ifdef __cplusplus
}
#endif
