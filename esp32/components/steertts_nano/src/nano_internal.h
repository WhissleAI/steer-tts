/* SteerTTS Nano runtime internals. */
#ifndef NANO_INTERNAL_H
#define NANO_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "steertts_nano.h"

#if defined(ESP_PLATFORM)
#include "sdkconfig.h"
#endif
/* ESP32-S3 PIE SIMD kernels (nano_simd_s3.c); NANO_NO_SIMD forces the portable C kernels */
#if defined(__XTENSA__) && defined(CONFIG_IDF_TARGET_ESP32S3) && !defined(NANO_NO_SIMD)
#define NANO_HAVE_SIMD 1
#else
#define NANO_HAVE_SIMD 0
#endif
#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32S3) && defined(__has_include) && !defined(NANO_NO_ESP_DSP)
#if __has_include("dsps_fft2r.h")
#define NANO_HAVE_ESP_DSP 1
#endif
#endif

#define NANO_MAX_DIM 512
#define NANO_MAX_K 9

#if defined(__GNUC__)
#define NANO_INLINE static inline __attribute__((always_inline))
#else
#define NANO_INLINE static inline
#endif

void nano_seterr(const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

/* float32 -> int8 with round-half-to-even (the 1.5 * 2^23 trick, exact for |v| < 2^22) and saturation.
 * The Python reference does the same float32 operations: np.float32(v + 12582912.0) - 12582912.0. */
NANO_INLINE int nano_q8(float v) {
  if (v > 127.f) return 127;
  if (v < -128.f) return -128;
  float t = v + 12582912.0f;   /* needs IEEE float32 arithmetic: no -ffast-math, no x87 */
  return (int)(t - 12582912.0f);
}

/* ------------------------------------------------------------------ arena */
typedef struct {
  uint8_t* base;
  size_t cap, top, peak;
} nano_arena;

NANO_INLINE void* nano_alloc(nano_arena* a, size_t n) {
  size_t off = (a->top + 15) & ~(size_t)15;
  if (off + n > a->cap) return NULL;
  a->top = off + n;
  if (a->top > a->peak) a->peak = a->top;
  return a->base + off;
}

/* ------------------------------------------------------------------ kernels (nano_kernels.c, port) */
/* LayerNorm (plain or adaptive) of int8 rows -> int8 rows, see docs/NANO_BIN.md "Norms".
 * g / b: per-channel affine (plain LN); for AdaLN pass g = scale (1 + s), b = shift. */
void nano_norm_rows(const int8_t* in, int in_stride, int rows, int d, float s_in, const float* g, const float* b,
                    float eps, float s_out, int8_t* out, int out_stride);
/* out = q8(x * (s_x / s_y) + h * (s_h / s_y)) */
void nano_residual(const int8_t* x, int x_stride, const int8_t* h, int rows, int d, float s_x, float s_h, float s_y,
                   int8_t* out, int out_stride);
void nano_lut_rows(int8_t* x, int n, const int8_t* lut);
void nano_gelu_lut(float s_in, float s_out, int8_t lut[256]);
float nano_expf(float x);
void nano_sincosf(float x, float* s, float* c);

/* Complex FFT, forward, in place, natural order output: data = re0 im0 re1 im1 ..., n a power of two <= 1024. */
void nano_fft_c(float* data, int n);
int nano_fft_init(void);   /* builds the platform tables (no-op on hosts) */

/* ------------------------------------------------------------------ profiling */
#define NANO_PROF_OP(ctx, op, ...)                          \
  do {                                                      \
    uint64_t t0_ = nano_cycles();                           \
    __VA_ARGS__;                                            \
    (ctx)->prof.op_cycles[op] += nano_cycles() - t0_;       \
  } while (0)

#endif
