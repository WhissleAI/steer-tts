/* Portable int8 kernels (reference semantics) and the element-wise float helpers.
 *
 * Requantization is the TFLite / gemmlowp scheme (the same as esp-nn's bit-exact path):
 *   out = clamp(RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(acc << max(shift, 0), mult), max(-shift, 0)))
 * The ESP32-S3 SIMD kernels (nano_simd_s3.c) produce bit-identical results. */
#include <math.h>

#include "nano_internal.h"

static int32_t srdhm(int32_t a, int32_t b) {
  if (a == b && a == INT32_MIN) return INT32_MAX;
  int64_t ab = (int64_t)a * (int64_t)b;
  int64_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
  int64_t r = ab + nudge;
  /* division by 2^31 rounding toward zero (C semantics), as in gemmlowp */
  return (int32_t)(r / ((int64_t)1 << 31));
}

static int32_t rdpot(int32_t x, int e) {
  if (e <= 0) return x;
  int32_t mask = (int32_t)((1u << e) - 1);
  int32_t rem = x & mask;
  int32_t thr = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> e) + (rem > thr ? 1 : 0);
}

int32_t nano_requant(int32_t acc, int32_t mult, int32_t shift) {
  int ls = shift > 0 ? shift : 0, rs = shift > 0 ? 0 : -shift;
  return rdpot(srdhm((int32_t)((uint32_t)acc << ls), mult), rs);
}

NANO_INLINE int8_t clamp8(int32_t v, int lo, int hi) { return (int8_t)(v < lo ? lo : v > hi ? hi : v); }

int32_t nano_dot_s8_ref(const int8_t* a, const int8_t* b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

void nano_gemm_s8_ref(const int8_t* in, int in_stride, int rows, const int8_t* w, int K, const int32_t* b,
                      const int32_t* m, const int32_t* s, int8_t* out, int out_stride, int out_ch, int lo, int hi) {
  for (int o = 0; o < out_ch; o++) {
    const int8_t* wr = w + (size_t)o * K;
    for (int r = 0; r < rows; r++) {
      int32_t acc = nano_dot_s8_ref(in + (size_t)r * in_stride, wr, K) + (b ? b[o] : 0);
      out[(size_t)r * out_stride + o] = clamp8(nano_requant(acc, m[o], s[o]), lo, hi);
    }
  }
}

void nano_dw_s8_ref(const int8_t* in, int rows, int C, const int8_t* w, int k, const int32_t* b, const int32_t* m,
                    const int32_t* s, int8_t* out, int lo, int hi) {
  for (int t = 0; t < rows; t++)
    for (int c = 0; c < C; c++) {
      int32_t acc = 0;
      for (int j = 0; j < k; j++) acc += (int32_t)in[(size_t)(t + j) * C + c] * (int32_t)w[j * C + c];
      acc += b[c];
      out[(size_t)t * C + c] = clamp8(nano_requant(acc, m[c], s[c]), lo, hi);
    }
}

#if !NANO_HAVE_SIMD
const char* nano_kernel_impl(void) { return "c"; }
int32_t nano_dot_s8(const int8_t* a, const int8_t* b, int n) { return nano_dot_s8_ref(a, b, n); }

void nano_gemm_s8(const int8_t* in, int in_stride, int rows, const int8_t* w, int K, const int32_t* b,
                  const int32_t* m, const int32_t* s, int8_t* out, int out_stride, int out_ch, int lo, int hi) {
  nano_gemm_s8_ref(in, in_stride, rows, w, K, b, m, s, out, out_stride, out_ch, lo, hi);
}

/* channel-blocked: 16 accumulators per time step (the shape of the SIMD kernel) */
void nano_dw_s8(const int8_t* in, int rows, int C, const int8_t* w, int k, const int32_t* b, const int32_t* m,
                const int32_t* s, int8_t* out, int lo, int hi) {
  for (int t = 0; t < rows; t++)
    for (int c0 = 0; c0 < C; c0 += 16) {
      int32_t acc[16];
      for (int l = 0; l < 16; l++) acc[l] = 0;
      for (int j = 0; j < k; j++) {
        const int8_t* x = in + (size_t)(t + j) * C + c0;
        const int8_t* wj = w + j * C + c0;
        for (int l = 0; l < 16; l++) acc[l] += (int32_t)x[l] * (int32_t)wj[l];
      }
      for (int l = 0; l < 16; l++) {
        int c = c0 + l;
        out[(size_t)t * C + c] = clamp8(nano_requant(acc[l] + b[c], m[c], s[c]), lo, hi);
      }
    }
}
#endif

/* ------------------------------------------------------------------ element-wise (float32, no libm per element) */
void nano_norm_rows(const int8_t* in, int in_stride, int rows, int d, float s_in, const float* g, const float* b,
                    float eps, float s_out, int8_t* out, int out_stride) {
  float eps_q = eps / (s_in * s_in);
  float inv_out = 1.0f / s_out;
  float inv_d = 1.0f / (float)d;
  for (int r = 0; r < rows; r++) {
    const int8_t* x = in + (size_t)r * in_stride;
    int8_t* y = out + (size_t)r * out_stride;
    int32_t S = 0, SS = 0;
    for (int i = 0; i < d; i++) {
      S += x[i];
      SS += (int32_t)x[i] * x[i];
    }
    float mean = (float)S * inv_d;
    float var = (float)SS * inv_d - mean * mean;
    if (var < 0.f) var = 0.f;
    float rs = 1.0f / sqrtf(var + eps_q);
    for (int i = 0; i < d; i++) {
      float z = ((float)x[i] - mean) * rs;
      y[i] = (int8_t)nano_q8((z * g[i] + b[i]) * inv_out);
    }
  }
}

void nano_residual(const int8_t* x, int x_stride, const int8_t* h, int rows, int d, float s_x, float s_h, float s_y,
                   int8_t* out, int out_stride) {
  float a = s_x / s_y, c = s_h / s_y;
  for (int r = 0; r < rows; r++) {
    const int8_t* xr = x + (size_t)r * x_stride;
    const int8_t* hr = h + (size_t)r * d;
    int8_t* o = out + (size_t)r * out_stride;
    for (int i = 0; i < d; i++) o[i] = (int8_t)nano_q8((float)xr[i] * a + (float)hr[i] * c);
  }
}

void nano_lut_rows(int8_t* x, int n, const int8_t* lut) {
  for (int i = 0; i < n; i++) x[i] = lut[(uint8_t)x[i]];
}

/* ------------------------------------------------------------------ deterministic float32 math
 * Used for the tables built at run time (LUTs, conditioning), so that every platform computes the same bits
 * (libm implementations differ in the last ulp). Not in the per-frame hot path. */
NANO_INLINE float rintf_det(float v) { return (v + 12582912.0f) - 12582912.0f; }

float nano_expf(float x) {
  if (x < -87.0f) return 0.0f;
  if (x > 88.0f) x = 88.0f;
  float n = rintf_det(x * 1.44269504f);
  float r = x - n * 0.693359375f;
  r = r - n * -2.12194440e-4f;
  float p = 1.0f / 5040.0f;
  p = p * r + 1.0f / 720.0f;
  p = p * r + 1.0f / 120.0f;
  p = p * r + 1.0f / 24.0f;
  p = p * r + 1.0f / 6.0f;
  p = p * r + 0.5f;
  p = p * r + 1.0f;
  p = p * r + 1.0f;
  int e = (int)n;
  union { uint32_t u; float f; } sc;
  if (e < -126) {   /* two steps to stay out of denormal scale factors */
    sc.u = (uint32_t)(e + 126 + 127) << 23;
    return p * sc.f * 1.17549435e-38f;
  }
  sc.u = (uint32_t)(e + 127) << 23;
  return p * sc.f;
}

/* sin and cos of x, |x| < 1e5 */
void nano_sincosf(float x, float* s_out, float* c_out) {
  float k = rintf_det(x * 0.159154943f);
  float r = x - k * 6.28125f;
  r = r - k * 1.93530717e-3f;   /* r in about [-pi, pi] */
  float sgn_c = 1.0f;
  if (r > 1.57079637f) {
    r = 3.14159274f - r;
    sgn_c = -1.0f;
  } else if (r < -1.57079637f) {
    r = -3.14159274f - r;
    sgn_c = -1.0f;
  }
  float r2 = r * r;
  float sp = -2.50521084e-8f;
  sp = sp * r2 + 2.75573192e-6f;
  sp = sp * r2 - 1.98412698e-4f;
  sp = sp * r2 + 8.33333333e-3f;
  sp = sp * r2 - 1.66666667e-1f;
  sp = sp * r2 + 1.0f;
  float cp = 2.08767570e-9f;
  cp = cp * r2 - 2.75573192e-7f;
  cp = cp * r2 + 2.48015873e-5f;
  cp = cp * r2 - 1.38888889e-3f;
  cp = cp * r2 + 4.16666667e-2f;
  cp = cp * r2 - 0.5f;
  cp = cp * r2 + 1.0f;
  *s_out = sp * r;
  *c_out = cp * sgn_c;
}

/* GELU (tanh form) table, indexed by (uint8_t)q */
void nano_gelu_lut(float s_in, float s_out, int8_t lut[256]) {
  float inv = 1.0f / s_out;
  for (int q = -128; q < 128; q++) {
    float v = (float)q * s_in;
    float y = 0.7978845608f * (v + 0.044715f * v * v * v);
    float t = 1.0f - 2.0f / (nano_expf(2.0f * y) + 1.0f);
    float g = 0.5f * v * (1.0f + t);
    lut[(uint8_t)q] = (int8_t)nano_q8(g * inv);
  }
}
