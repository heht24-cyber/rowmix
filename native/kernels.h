#ifndef ROWMIX_KERNELS_H
#define ROWMIX_KERNELS_H
#include <cstdint>
namespace rowmix {
using Dot = int32_t (*)(const int8_t *, const uint8_t *, uint32_t start, uint32_t n, uint8_t bits);
struct Kernel { Dot dot; const char *name; };
Kernel choose_kernel(int requested);
}
#endif
