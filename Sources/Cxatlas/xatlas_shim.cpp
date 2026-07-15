#include "xatlas.h"
#include "xatlas_shim.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>

namespace {

std::atomic<xatlas_swift_print_cb> g_print_cb{nullptr};
std::atomic<xatlas_swift_param_cb> g_param_cb{nullptr};

int xatlas_print_thunk(const char *fmt, ...) {
    auto cb = g_print_cb.load(std::memory_order_acquire);
    if (!cb) return 0;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    cb(buf);
    return n;
}

void xatlas_param_thunk_impl(
    const float *positions,
    float *texcoords,
    uint32_t vertexCount,
    const uint32_t *indices,
    uint32_t indexCount)
{
    auto cb = g_param_cb.load(std::memory_order_acquire);
    if (!cb) return;
    cb(positions, texcoords, vertexCount, indices, indexCount);
}

} // namespace

extern "C" void xatlas_install_print(xatlas_swift_print_cb cb, int verbose) {
    g_print_cb.store(cb, std::memory_order_release);
    xatlas::SetPrint(cb ? xatlas_print_thunk : nullptr, verbose != 0);
}

extern "C" void xatlas_install_parameterize(xatlas_swift_param_cb cb) {
    g_param_cb.store(cb, std::memory_order_release);
}

extern "C" void *xatlas_parameterize_thunk(void) {
    if (!g_param_cb.load(std::memory_order_acquire)) return nullptr;
    return reinterpret_cast<void *>(&xatlas_param_thunk_impl);
}
