#include "kernels.h"
#include <stdexcept>
#if (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__)) && !defined(ROWMIX_PORTABLE_ONLY)
#define ROWMIX_X86 1
#include <immintrin.h>
#endif

namespace rowmix {
static int signed_nibble(uint8_t v) { return int(v ^ 8) - 8; }
static int32_t scalar(const int8_t *x,const uint8_t *w,uint32_t at,uint32_t n,uint8_t bits) {
    int32_t sum=0;
    if (bits==8) {
        for (uint32_t i=0;i<n;++i) {
            const int v=w[at+i]; sum+=int(x[i])*(v>=128?v-256:v);
        }
        return sum;
    }
    if ((at&1) && n) {
        sum=int(*x++)*signed_nibble(w[at/2]>>4); ++at; --n;
    }
    w+=at/2;
    for (uint32_t i=0;i<n/2;++i)
        sum+=int(x[2*i])*signed_nibble(w[i]&15)+int(x[2*i+1])*signed_nibble(w[i]>>4);
    if (n&1) sum+=int(x[n-1])*signed_nibble(w[n/2]&15);
    return sum;
}
#ifdef ROWMIX_X86
__attribute__((target("avx2")))
static int32_t avx2(const int8_t *x,const uint8_t *w,uint32_t at,uint32_t n,uint8_t bits) {
    if (n<16) return scalar(x,w,at,n,bits);
    int32_t sum=0;
    if (bits==4 && (at&1)) {
        sum=int(*x++)*signed_nibble(w[at/2]>>4); ++at; --n;
    }
    __m256i lanes=_mm256_setzero_si256();
    uint32_t i=0;
    const __m128i mask=_mm_set1_epi8(15), sign=_mm_set1_epi8(8);
    for (;i+16<=n;i+=16) {
        __m128i weights;
        if (bits==8) weights=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+at+i));
        else {
            auto packed=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w+(at+i)/2));
            auto low=_mm_and_si128(packed,mask);
            auto high=_mm_and_si128(_mm_srli_epi16(packed,4),mask);
            auto nibbles=_mm_unpacklo_epi8(low,high);
            weights=_mm_sub_epi8(_mm_xor_si128(nibbles,sign),sign);
        }
        auto inputs=_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+i));
        auto pairs=_mm256_madd_epi16(_mm256_cvtepi8_epi16(inputs),_mm256_cvtepi8_epi16(weights));
        lanes=_mm256_add_epi32(lanes,pairs);
    }
    alignas(32) int32_t partial[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(partial),lanes);
    for (auto v:partial) sum+=v;
    return sum+scalar(x+i,w,at+i,n-i,bits);
}
#endif
Kernel choose_kernel(int requested) {
    if (requested<0 || requested>2) throw std::runtime_error("unknown kernel selector");
#ifdef ROWMIX_X86
    if (requested!=1 && __builtin_cpu_supports("avx2")) return {avx2,"avx2"};
#endif
    if (requested==2) throw std::runtime_error("AVX2 unavailable in this build or CPU");
    return {scalar,"scalar"};
}
}
