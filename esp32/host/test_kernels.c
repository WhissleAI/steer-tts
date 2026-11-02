/* Kernel tests: requantization against TFLite reference values, optimized vs reference int8 kernels, FFT vs DFT. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "nano_internal.h"

static int fails = 0;
#define EXPECT(cond, ...)          \
  do {                             \
    if (!(cond)) {                 \
      printf("FAIL: " __VA_ARGS__); \
      printf("\n");                \
      fails++;                     \
    }                              \
  } while (0)

static uint32_t s = 12345;
static int r8(void) {
  s = s * 1664525u + 1013904223u;
  return (int)((s >> 24) & 255) - 128;
}

int main(void) {
  /* requant: mult 2^30 (0.5) with shift 0 -> x / 2 rounded half away from zero */
  EXPECT(nano_requant(3, 1 << 30, 0) == 2, "requant 3*0.5");
  EXPECT(nano_requant(-3, 1 << 30, 0) == -1, "requant -3*0.5 (gemmlowp rounds the high-mul half up)");
  EXPECT(nano_requant(1000, 1 << 30, -3) == 63, "requant 1000*0.5/8 = %d", nano_requant(1000, 1 << 30, -3));
  EXPECT(nano_requant(100, 1 << 30, 2) == 200, "requant left shift");

  /* gemm / dw: optimized == reference */
  enum { R = 13, K = 160, O = 48, C = 32, KW = 5 };
  static int8_t in[(R + 8) * 384], w[O * 384], out1[R * O], out2[R * O];
  static int32_t b[O], m[O], sh[O];
  for (int i = 0; i < (int)sizeof in; i++) in[i] = (int8_t)r8();
  for (int i = 0; i < (int)sizeof w; i++) w[i] = (int8_t)r8();
  for (int o = 0; o < O; o++) {
    b[o] = r8() * 50;
    m[o] = (1 << 30) + r8() * 1000000;
    sh[o] = -(8 + (o % 5));
  }
  nano_gemm_s8_ref(in, K, R, w, K, b, m, sh, out1, O, O, -128, 127);
  nano_gemm_s8(in, K, R, w, K, b, m, sh, out2, O, O, -128, 127);
  int d = 0;
  for (int i = 0; i < R * O; i++) d += out1[i] != out2[i];
  EXPECT(d == 0, "gemm: %d outputs differ", d);
  nano_gemm_s8_ref(in, C, R, w, KW * C, b, m, sh, out1, O, O, 0, 127);   /* conv: overlapping rows */
  nano_gemm_s8(in, C, R, w, KW * C, b, m, sh, out2, O, O, 0, 127);
  d = 0;
  for (int i = 0; i < R * O; i++) d += out1[i] != out2[i];
  EXPECT(d == 0, "conv gemm: %d outputs differ", d);
  nano_dw_s8_ref(in, R, C, w, KW, b, m, sh, out1, -128, 127);
  nano_dw_s8(in, R, C, w, KW, b, m, sh, out2, -128, 127);
  d = 0;
  for (int i = 0; i < R * C; i++) d += out1[i] != out2[i];
  EXPECT(d == 0, "dw: %d outputs differ", d);

  /* FFT vs a double-precision DFT */
  static float x[2 * 512];
  static double ref[2 * 512];
  for (int n = 512; n >= 16; n /= 2) {
    for (int i = 0; i < 2 * n; i++) x[i] = (float)r8() / 128.0f;
    for (int k = 0; k < n; k++) {
      double re = 0, im = 0;
      for (int t = 0; t < n; t++) {
        double a = -2 * M_PI * (double)k * t / n;
        re += x[2 * t] * cos(a) - x[2 * t + 1] * sin(a);
        im += x[2 * t] * sin(a) + x[2 * t + 1] * cos(a);
      }
      ref[2 * k] = re;
      ref[2 * k + 1] = im;
    }
    nano_fft_c(x, n);
    double err = 0;
    for (int i = 0; i < 2 * n; i++) err = fmax(err, fabs(x[i] - ref[i]));
    EXPECT(err < 1e-3, "fft %d: max error %g", n, err);
  }
  printf("%s (%s kernels)\n", fails ? "FAILED" : "all kernel tests passed", nano_kernel_impl());
  return fails ? 1 : 0;
}
