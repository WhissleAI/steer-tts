/* torch-compatible RNG: mt19937 (at::mt19937), the float uniform conversion (24 random bits / 2^24) and
 * torch.randn's CPU paths:
 *
 *   n >= 16 (float32, contiguous): normal_fill_AVX2 on an AVX2 build (the x86-64 wheels): fill all n uniforms, then
 *     Box-Muller on blocks of 16 (u1 = 1 - data[j], u2 = data[j+8]; radius = sqrt(-2 log u1); theta = 2 pi u2;
 *     data[j] = radius cos theta, data[j+8] = radius sin theta) with avx_mathfun's Cephes log256_ps / sincos256_ps.
 *     If n % 16 != 0 the last 16 slots are refilled with 16 fresh uniforms and transformed again.
 *   n < 16: at::normal_distribution<double> (53-bit uniforms from two draws, r = sqrt(-2 log1p(-u2)),
 *     theta = 2 pi u1, r cos theta then the cached r sin theta), cast to float.
 *
 * The Cephes polynomials are evaluated here in scalar float arithmetic in the same operation order. PyTorch's
 * AVX2 kernels are compiled by GCC with -mfma and the default -ffp-contract=fast, so a multiply whose only use is
 * an add / subtract is fused; FMA() below marks exactly those places (fmaf: correctly rounded, as the hardware
 * instruction). The result matches torch bit for bit (tests/test_flow.c); the scalar normal_fill (non-AVX2
 * builds: ARM, ppc) uses libm sinf/cosf/logf instead and is not reproduced. */
#include "st_rng.h"

#include <math.h>
#include <string.h>

#ifdef __clang__
#pragma STDC FP_CONTRACT OFF
#endif

/* ------------------------------------------------------------------ mt19937 */
void st_mt_seed(st_mt19937* mt, uint64_t seed) {
  mt->s[0] = (uint32_t)(seed & 0xffffffffu);
  for (int j = 1; j < 624; ++j) mt->s[j] = 1812433253u * (mt->s[j - 1] ^ (mt->s[j - 1] >> 30)) + (uint32_t)j;
  mt->next = 624;
}

static void mt_twist(st_mt19937* mt) {
  uint32_t* s = mt->s;
  for (int i = 0; i < 624; ++i) {
    uint32_t y = (s[i] & 0x80000000u) | (s[(i + 1) % 624] & 0x7fffffffu);
    s[i] = s[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
  }
  mt->next = 0;
}

uint32_t st_mt_next(st_mt19937* mt) {
  if (mt->next >= 624) mt_twist(mt);
  uint32_t y = mt->s[mt->next++];
  y ^= y >> 11;
  y ^= (y << 7) & 0x9d2c5680u;
  y ^= (y << 15) & 0xefc60000u;
  y ^= y >> 18;
  return y;
}

static float uniform_f(st_mt19937* mt) {
  return (float)((double)(st_mt_next(mt) & 0xffffffu) * (1.0 / 16777216.0));
}

static double uniform_d(st_mt19937* mt) {
  uint64_t hi = st_mt_next(mt), lo = st_mt_next(mt);
  uint64_t v = (hi << 32) | lo;
  return (double)(v & ((1ull << 53) - 1)) * (1.0 / 9007199254740992.0);
}

/* ------------------------------------------------------------------ avx_mathfun (Cephes) in scalar form */
#define FMA(a, b, c) fmaf((a), (b), (c))

static float f_from_u(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint32_t u_from_f(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}

/* log256_ps for x in (0, 1] (u1 = 1 - uniform is never 0 or negative) */
static float cephes_logf(float x) {
  const float min_norm = f_from_u(0x00800000u);
  x = x > min_norm ? x : min_norm;   /* _mm256_max_ps(x, min_norm) */
  int32_t imm0 = (int32_t)(u_from_f(x) >> 23);
  x = f_from_u((u_from_f(x) & ~0x7f800000u) | u_from_f(0.5f));
  imm0 -= 0x7f;
  float e = (float)imm0;
  e = e + 1.0f;
  int lt = x < 0.707106781186547524f;
  float tmp = lt ? x : 0.0f;
  x = x - 1.0f;
  e = e - (lt ? 1.0f : 0.0f);
  x = x + tmp;
  float z = x * x;
  float y = FMA(7.0376836292E-2f, x, -1.1514610310E-1f);
  y = FMA(y, x, 1.1676998740E-1f);
  y = FMA(y, x, -1.2420140846E-1f);
  y = FMA(y, x, +1.4249322787E-1f);
  y = FMA(y, x, -1.6668057665E-1f);
  y = FMA(y, x, +2.0000714765E-1f);
  y = FMA(y, x, -2.4999993993E-1f);
  y = FMA(y, x, +3.3333331174E-1f);
  y = y * x;
  tmp = e * -2.12194440e-4f;
  y = FMA(y, z, tmp);
  y = FMA(-z, 0.5f, y);   /* y - z * 0.5 */
  x = x + y;
  x = FMA(e, 0.693359375f, x);
  return x;
}

/* sincos256_ps for x >= 0 (theta = 2 pi u2) */
static void cephes_sincosf(float x, float* s, float* c) {
  uint32_t sign_sin = u_from_f(x) & 0x80000000u;
  x = f_from_u(u_from_f(x) & 0x7fffffffu);
  float y = x * 1.27323954473516f;
  int32_t imm2 = (int32_t)y;   /* cvttps: truncation (y is small and positive) */
  imm2 = (imm2 + 1) & ~1;
  y = (float)imm2;
  int32_t imm4 = imm2;
  uint32_t swap_sin = (uint32_t)(imm2 & 4) << 29;
  int poly2 = (imm2 & 2) == 0;
  x = FMA(y, -0.78515625f, x);
  x = FMA(y, -2.4187564849853515625e-4f, x);
  x = FMA(y, -3.77489497744594108e-8f, x);
  imm4 -= 2;
  uint32_t sign_cos = (uint32_t)(~imm4 & 4) << 29;
  sign_sin ^= swap_sin;
  float z = x * x;
  float yc = FMA(2.443315711809948E-005f, z, -1.388731625493765E-003f);
  yc = FMA(yc, z, 4.166664568298827E-002f);
  yc = yc * z;
  float tmp = z * 0.5f;
  yc = FMA(yc, z, -tmp);   /* y*z - tmp */
  yc = yc + 1.0f;
  float ys = FMA(-1.9515295891E-4f, z, 8.3321608736E-3f);
  ys = FMA(ys, z, -1.6666654611E-1f);
  ys = ys * z;
  ys = FMA(ys, x, x);
  float vs = poly2 ? ys : yc, vc = poly2 ? yc : ys;
  *s = f_from_u(u_from_f(vs) ^ sign_sin);
  *c = f_from_u(u_from_f(vc) ^ sign_cos);
}

static void normal_fill_16(float* d) {
  const float two_pi = (float)(2.0f * 3.14159265358979323846);
  for (int j = 0; j < 8; ++j) {
    float u1 = 1.0f - d[j], u2 = d[j + 8];
    float radius = sqrtf(-2.0f * cephes_logf(u1));
    float theta = two_pi * u2, sn, cs;
    cephes_sincosf(theta, &sn, &cs);
    d[j] = radius * cs;   /* fmadd(n1, std = 1, mean = 0) is exact */
    d[j + 8] = radius * sn;
  }
}

void st_randn_torch(uint64_t seed, float* out, size_t n) {
  st_mt19937 mt;
  st_mt_seed(&mt, seed);
  if (n < 16) {
    int have = 0;
    double cached = 0;
    for (size_t i = 0; i < n; ++i) {
      if (have) {
        out[i] = (float)cached;
        have = 0;
        continue;
      }
      double u1 = uniform_d(&mt), u2 = uniform_d(&mt);
      double r = sqrt(-2.0 * log1p(-u2)), th = 2.0 * 3.14159265358979323846 * u1;
      cached = r * sin(th);
      have = 1;
      out[i] = (float)(r * cos(th));
    }
    return;
  }
  for (size_t i = 0; i < n; ++i) out[i] = uniform_f(&mt);
  for (size_t i = 0; i + 16 <= n; i += 16) normal_fill_16(out + i);
  if (n % 16) {
    float* d = out + n - 16;
    for (int i = 0; i < 16; ++i) d[i] = uniform_f(&mt);
    normal_fill_16(d);
  }
}
