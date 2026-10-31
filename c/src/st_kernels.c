/* Dense kernels: packed GEMM (fp32 / int8 / int4 weights, AVX2+FMA or NEON micro-kernels with a portable C
 * fallback), conv1d via im2col, depthwise conv, norms, activations, softmax, attention, FFT.
 *
 * GEMM: C[M,N] = A[M,K] * B(N,K). A is read in place (row pointers); B is packed per (K-block, 16 columns) panel,
 * which is also where quantized weights are dequantized, so int8 / int4 weights are never expanded in full. Work is
 * split into (96-row, 64-column) tiles over the thread pool. */
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "st_base.h"

/* no a*b+c contraction into FMA in the C code: results must not depend on the compiler (GCC gets this from
 * -std=c11; clang honours the pragma; the SIMD kernels use explicit FMA intrinsics either way) */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

/* GCC / clang need a target attribute to emit AVX2 code in a generic build; MSVC emits intrinsics anywhere */
#if defined(__GNUC__)
#define ST_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
#define ST_TARGET_AVX2
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define ST_X86 1
#include <immintrin.h>
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#define ST_NEON 1
#include <arm_neon.h>
#endif

enum { MR = 6, NR = 16, KC = 256, MC = 96, NC = 64 };

/* ------------------------------------------------------------------ B sources */
typedef enum { SRC_NT, SRC_NN, SRC_MAT } src_kind;
typedef struct {
  src_kind kind;
  const float* b;   /* NT: B[n*ldb + k]; NN: B[k*ldb + n] */
  int ldb;
  const st_mat* m;  /* SRC_MAT */
} bsrc;

/* dst[kk*NR + j] = B(n0 + j, k0 + kk), zero for j >= nn */
static void pack_b(const bsrc* s, int n0, int nn, int k0, int kc, float* dst) {
  if (nn < NR) memset(dst, 0, sizeof(float) * (size_t)kc * NR);
  switch (s->kind) {
    case SRC_NT:
      for (int j = 0; j < nn; ++j) {
        const float* r = s->b + (size_t)(n0 + j) * s->ldb + k0;
        for (int kk = 0; kk < kc; ++kk) dst[kk * NR + j] = r[kk];
      }
      break;
    case SRC_NN:
      for (int kk = 0; kk < kc; ++kk) {
        const float* r = s->b + (size_t)(k0 + kk) * s->ldb + n0;
        float* d = dst + kk * NR;
        for (int j = 0; j < nn; ++j) d[j] = r[j];
      }
      break;
    case SRC_MAT: {
      const st_mat* m = s->m;
      if (m->type == ST_W_F32) {
        for (int j = 0; j < nn; ++j) {
          const float* r = m->w + (size_t)(n0 + j) * m->K + k0;
          for (int kk = 0; kk < kc; ++kk) dst[kk * NR + j] = r[kk];
        }
      } else if (m->type == ST_W_I8) {
        for (int j = 0; j < nn; ++j) {
          const int8_t* r = m->q8 + (size_t)(n0 + j) * m->K + k0;
          float sc = m->s[n0 + j];
          for (int kk = 0; kk < kc; ++kk) dst[kk * NR + j] = (float)r[kk] * sc;
        }
      } else {
        int ng = m->Kp / m->g;
        for (int j = 0; j < nn; ++j) {
          size_t base = (size_t)(n0 + j) * m->Kp;
          const float* sc = m->s + (size_t)(n0 + j) * ng;
          int kk = 0;
          while (kk < kc) {
            int k = k0 + kk, gi = k / m->g, gend = (gi + 1) * m->g - k0;
            if (gend > kc) gend = kc;
            float g = sc[gi];
            for (; kk < gend; ++kk) {
              size_t idx = base + (size_t)(k0 + kk);
              int nib = (m->q4[idx >> 1] >> ((idx & 1) << 2)) & 15;
              dst[kk * NR + j] = (float)(nib - 8) * g;
            }
          }
        }
      }
      break;
    }
  }
}

/* ------------------------------------------------------------------ micro-kernels
 * C[mr, nr] (+)= A[mr, kc] * Bp[kc, NR]; mode 0 store, 1 accumulate. */
typedef void (*kern_fn)(int kc, const float* A, int lda, int mr, const float* Bp, float* C, int ldc, int nr, int mode);

static void kern_c(int kc, const float* A, int lda, int mr, const float* Bp, float* C, int ldc, int nr, int mode) {
  float acc[MR][NR];
  memset(acc, 0, sizeof acc);
  const float* a[MR];
  for (int i = 0; i < MR; ++i) a[i] = A + (size_t)(i < mr ? i : 0) * lda;
  for (int k = 0; k < kc; ++k) {
    const float* b = Bp + k * NR;
    for (int i = 0; i < MR; ++i) {
      float av = a[i][k];
      for (int j = 0; j < NR; ++j) acc[i][j] += av * b[j];
    }
  }
  for (int i = 0; i < mr; ++i) {
    float* c = C + (size_t)i * ldc;
    if (mode)
      for (int j = 0; j < nr; ++j) c[j] += acc[i][j];
    else
      for (int j = 0; j < nr; ++j) c[j] = acc[i][j];
  }
}

static void store_tile(float acc[MR][NR], float* C, int ldc, int mr, int nr, int mode) {
  for (int i = 0; i < mr; ++i) {
    float* c = C + (size_t)i * ldc;
    if (mode)
      for (int j = 0; j < nr; ++j) c[j] += acc[i][j];
    else
      for (int j = 0; j < nr; ++j) c[j] = acc[i][j];
  }
}

#ifdef ST_X86
ST_TARGET_AVX2 static void kern_avx2(int kc, const float* A, int lda, int mr, const float* Bp,
                                                           float* C, int ldc, int nr, int mode) {
  const float *a0 = A, *a1 = A + (mr > 1 ? lda : 0), *a2 = A + (size_t)(mr > 2 ? 2 * lda : 0),
              *a3 = A + (size_t)(mr > 3 ? 3 * lda : 0), *a4 = A + (size_t)(mr > 4 ? 4 * lda : 0),
              *a5 = A + (size_t)(mr > 5 ? 5 * lda : 0);
  __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00,
         c40 = c00, c41 = c00, c50 = c00, c51 = c00;
  for (int k = 0; k < kc; ++k) {
    __m256 b0 = _mm256_loadu_ps(Bp), b1 = _mm256_loadu_ps(Bp + 8);
    __m256 x;
    x = _mm256_broadcast_ss(a0 + k); c00 = _mm256_fmadd_ps(x, b0, c00); c01 = _mm256_fmadd_ps(x, b1, c01);
    x = _mm256_broadcast_ss(a1 + k); c10 = _mm256_fmadd_ps(x, b0, c10); c11 = _mm256_fmadd_ps(x, b1, c11);
    x = _mm256_broadcast_ss(a2 + k); c20 = _mm256_fmadd_ps(x, b0, c20); c21 = _mm256_fmadd_ps(x, b1, c21);
    x = _mm256_broadcast_ss(a3 + k); c30 = _mm256_fmadd_ps(x, b0, c30); c31 = _mm256_fmadd_ps(x, b1, c31);
    x = _mm256_broadcast_ss(a4 + k); c40 = _mm256_fmadd_ps(x, b0, c40); c41 = _mm256_fmadd_ps(x, b1, c41);
    x = _mm256_broadcast_ss(a5 + k); c50 = _mm256_fmadd_ps(x, b0, c50); c51 = _mm256_fmadd_ps(x, b1, c51);
    Bp += NR;
  }
  if (mr == MR && nr == NR) {
    __m256 r[12] = {c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51};
    for (int i = 0; i < MR; ++i) {
      float* c = C + (size_t)i * ldc;
      if (mode) {
        _mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), r[2 * i]));
        _mm256_storeu_ps(c + 8, _mm256_add_ps(_mm256_loadu_ps(c + 8), r[2 * i + 1]));
      } else {
        _mm256_storeu_ps(c, r[2 * i]);
        _mm256_storeu_ps(c + 8, r[2 * i + 1]);
      }
    }
    return;
  }
  float acc[MR][NR];
  _mm256_storeu_ps(acc[0], c00); _mm256_storeu_ps(acc[0] + 8, c01);
  _mm256_storeu_ps(acc[1], c10); _mm256_storeu_ps(acc[1] + 8, c11);
  _mm256_storeu_ps(acc[2], c20); _mm256_storeu_ps(acc[2] + 8, c21);
  _mm256_storeu_ps(acc[3], c30); _mm256_storeu_ps(acc[3] + 8, c31);
  _mm256_storeu_ps(acc[4], c40); _mm256_storeu_ps(acc[4] + 8, c41);
  _mm256_storeu_ps(acc[5], c50); _mm256_storeu_ps(acc[5] + 8, c51);
  store_tile(acc, C, ldc, mr, nr, mode);
}
#endif

#ifdef ST_NEON
static void kern_neon(int kc, const float* A, int lda, int mr, const float* Bp, float* C, int ldc, int nr, int mode) {
  const float* a[MR];
  for (int i = 0; i < MR; ++i) a[i] = A + (size_t)(i < mr ? i : 0) * lda;
  float32x4_t c[MR][4];
  for (int i = 0; i < MR; ++i)
    for (int j = 0; j < 4; ++j) c[i][j] = vdupq_n_f32(0.f);
  for (int k = 0; k < kc; ++k) {
    float32x4_t b0 = vld1q_f32(Bp), b1 = vld1q_f32(Bp + 4), b2 = vld1q_f32(Bp + 8), b3 = vld1q_f32(Bp + 12);
    for (int i = 0; i < MR; ++i) {
      float32x4_t x = vdupq_n_f32(a[i][k]);
      c[i][0] = vfmaq_f32(c[i][0], x, b0);
      c[i][1] = vfmaq_f32(c[i][1], x, b1);
      c[i][2] = vfmaq_f32(c[i][2], x, b2);
      c[i][3] = vfmaq_f32(c[i][3], x, b3);
    }
    Bp += NR;
  }
  float acc[MR][NR];
  for (int i = 0; i < MR; ++i)
    for (int j = 0; j < 4; ++j) vst1q_f32(acc[i] + 4 * j, c[i][j]);
  store_tile(acc, C, ldc, mr, nr, mode);
}
#endif

#ifdef ST_X86
#if defined(_MSC_VER)
#include <intrin.h>
static void cpuid(int leaf, int sub, unsigned r[4]) {
  int v[4];
  __cpuidex(v, leaf, sub);
  for (int i = 0; i < 4; ++i) r[i] = (unsigned)v[i];
}
static unsigned long long xgetbv0(void) { return _xgetbv(0); }
#else
#include <cpuid.h>
static void cpuid(int leaf, int sub, unsigned r[4]) { __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]); }
static unsigned long long xgetbv0(void) {
  unsigned lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return ((unsigned long long)hi << 32) | lo;
}
#endif
#endif

int st_cpu_avx2(void) {
  /* computed once; concurrent first calls compute the same value, so a relaxed atomic is enough */
  static atomic_int cached = -1;
  int v = atomic_load_explicit(&cached, memory_order_relaxed);
  if (v >= 0) return v;
  int ok = 0;
#ifdef ST_X86
  unsigned r[4];
  cpuid(0, 0, r);
  if (r[0] >= 7) {
    cpuid(1, 0, r);
    int fma = (r[2] >> 12) & 1, osx = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (fma && osx && avx && (xgetbv0() & 6) == 6) {   /* OS saves XMM + YMM state */
      cpuid(7, 0, r);
      ok = (r[1] >> 5) & 1;   /* AVX2 */
    }
  }
#endif
  v = ok && !getenv("STEERTTS_NO_SIMD");
  atomic_store_explicit(&cached, v, memory_order_relaxed);
  return v;
}

enum { KERN_UNSET, KERN_C, KERN_AVX2, KERN_NEON };
static atomic_int g_kern = KERN_UNSET;   /* which micro-kernel; races only ever store the same value */

static kern_fn pick_kernel(void) {
  int id = atomic_load_explicit(&g_kern, memory_order_relaxed);
  if (id == KERN_UNSET) {
    id = KERN_C;
#ifdef ST_X86
    if (st_cpu_avx2()) id = KERN_AVX2;
#endif
#ifdef ST_NEON
    if (!getenv("STEERTTS_NO_SIMD")) id = KERN_NEON;
#endif
    atomic_store_explicit(&g_kern, id, memory_order_relaxed);
  }
  switch (id) {
#ifdef ST_X86
    case KERN_AVX2: return kern_avx2;
#endif
#ifdef ST_NEON
    case KERN_NEON: return kern_neon;
#endif
    default: return kern_c;
  }
}

/* ------------------------------------------------------------------ GEMM driver */
typedef struct {
  int M, N, K;
  const float* A;
  int lda;
  const bsrc* B;
  float* C;
  int ldc;
  const float* bias;
  int acc;
  int nmb, nnb, mc;
  kern_fn kern;
} gemm_job;

static void gemm_tile(void* ctx, int task, int n_tasks) {
  (void)n_tasks;
  const gemm_job* g = (const gemm_job*)ctx;
  int mb = task / g->nnb, nb = task % g->nnb;
  int m0 = mb * g->mc, m1 = m0 + g->mc < g->M ? m0 + g->mc : g->M;
  int n0 = nb * NC, n1 = n0 + NC < g->N ? n0 + NC : g->N;
  _Alignas(64) float bp[(NC / NR) * KC * NR];
  for (int k0 = 0; k0 < g->K; k0 += KC) {
    int kc = g->K - k0 < KC ? g->K - k0 : KC;
    int np = 0;
    for (int n = n0; n < n1; n += NR, ++np) pack_b(g->B, n, n1 - n < NR ? n1 - n : NR, k0, kc, bp + np * KC * NR);
    int mode = (k0 > 0 || g->acc) ? 1 : 0;
    for (int m = m0; m < m1; m += MR) {
      int mr = m1 - m < MR ? m1 - m : MR;
      np = 0;
      for (int n = n0; n < n1; n += NR, ++np) {
        int nr = n1 - n < NR ? n1 - n : NR;
        g->kern(kc, g->A + (size_t)m * g->lda + k0, g->lda, mr, bp + np * KC * NR, g->C + (size_t)m * g->ldc + n,
                g->ldc, nr, mode);
      }
    }
  }
  if (g->K == 0 && !g->acc)
    for (int m = m0; m < m1; ++m) memset(g->C + (size_t)m * g->ldc + n0, 0, sizeof(float) * (size_t)(n1 - n0));
  if (g->bias)
    for (int m = m0; m < m1; ++m) {
      float* c = g->C + (size_t)m * g->ldc;
      for (int n = n0; n < n1; ++n) c[n] += g->bias[n];
    }
}

static void gemm(st_pool* p, int M, int N, int K, const float* A, int lda, const bsrc* B, float* C, int ldc,
                 const float* bias, int acc) {
  if (M <= 0 || N <= 0) return;
  double flops = 2.0 * M * N * (double)K;
  int nnb = (N + NC - 1) / NC, th = flops < 2e5 ? 1 : st_pool_size(p);
  /* B panels are packed (and quantized weights dequantized) once per row block: use one row block when the column
   * blocks alone give every thread enough work, else split rows too */
  int mc = (th == 1 || nnb >= 3 * th) ? M : MC;
  gemm_job g = {M, N, K, A, lda, B, C, ldc, bias, acc, (M + mc - 1) / mc, nnb, mc, pick_kernel()};
  int tasks = g.nmb * g.nnb;
  st_parallel_for(flops < 2e5 ? NULL : p, tasks, gemm_tile, &g);
}

void st_gemm_nt(st_pool* p, int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                const float* bias, int acc) {
  bsrc s = {SRC_NT, B, ldb, NULL};
  gemm(p, M, N, K, A, lda, &s, C, ldc, bias, acc);
}

void st_gemm_nn(st_pool* p, int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                const float* bias, int acc) {
  bsrc s = {SRC_NN, B, ldb, NULL};
  gemm(p, M, N, K, A, lda, &s, C, ldc, bias, acc);
}

/* small-M path: direct dot products against the (possibly quantized) rows */
typedef struct {
  const float* x;
  int T, ldx;
  const st_mat* W;
  const float* bias;
  float* y;
  int ldy, acc;
} gemv_job;

static void gemv_rows(void* ctx, int task, int n_tasks) {
  const gemv_job* g = (const gemv_job*)ctx;
  const st_mat* m = g->W;
  int o0 = (int)((long long)m->O * task / n_tasks), o1 = (int)((long long)m->O * (task + 1) / n_tasks);
  for (int o = o0; o < o1; ++o) {
    for (int t = 0; t < g->T; ++t) {
      const float* x = g->x + (size_t)t * g->ldx;
      double s = 0;
      if (m->type == ST_W_F32) {
        const float* w = m->w + (size_t)o * m->K;
        float acc = 0;
        for (int k = 0; k < m->K; ++k) acc += x[k] * w[k];
        s = acc;
      } else if (m->type == ST_W_I8) {
        const int8_t* w = m->q8 + (size_t)o * m->K;
        float acc = 0;
        for (int k = 0; k < m->K; ++k) acc += x[k] * (float)w[k];
        s = acc * m->s[o];
      } else {
        int ng = m->Kp / m->g;
        size_t base = (size_t)o * m->Kp;
        for (int gi = 0; gi < ng; ++gi) {
          float acc = 0;
          int k1 = (gi + 1) * m->g < m->K ? (gi + 1) * m->g : m->K;
          for (int k = gi * m->g; k < k1; ++k) {
            size_t idx = base + (size_t)k;
            int nib = (m->q4[idx >> 1] >> ((idx & 1) << 2)) & 15;
            acc += x[k] * (float)(nib - 8);
          }
          s += acc * m->s[(size_t)o * ng + gi];
        }
      }
      float v = (float)s + (g->bias ? g->bias[o] : 0.f);
      float* y = g->y + (size_t)t * g->ldy + o;
      *y = g->acc ? *y + v : v;
    }
  }
}

void st_linear(st_pool* p, const float* x, int T, int ldx, const st_mat* W, const float* bias, float* y, int ldy,
               int acc) {
  if (T <= 0) return;
  if (T < 4) {
    gemv_job g = {x, T, ldx, W, bias, y, ldy, acc};
    int tasks = st_pool_size(p);
    if ((double)W->O * W->K < 2e5) tasks = 1;
    if (tasks > W->O) tasks = W->O;
    st_parallel_for(tasks > 1 ? p : NULL, tasks, gemv_rows, &g);
    return;
  }
  bsrc s = {SRC_MAT, NULL, 0, W};
  gemm(p, T, W->O, W->K, x, ldx, &s, y, ldy, bias, acc);
}

void st_mat_rows(const st_mat* m, int r0, int n, st_mat* out) {
  *out = *m;
  out->O = n;
  if (m->type == ST_W_F32) out->w = m->w + (size_t)r0 * m->K;
  else if (m->type == ST_W_I8) {
    out->q8 = m->q8 + (size_t)r0 * m->K;
    out->s = m->s + r0;
  } else {
    out->q4 = m->q4 + (size_t)r0 * (m->Kp / 2);
    out->s = m->s + (size_t)r0 * (m->Kp / m->g);
  }
}

void st_mat_row(const st_mat* m, int r, float* out) {
  if (m->type == ST_W_F32) memcpy(out, m->w + (size_t)r * m->K, sizeof(float) * (size_t)m->K);
  else if (m->type == ST_W_I8)
    for (int k = 0; k < m->K; ++k) out[k] = (float)m->q8[(size_t)r * m->K + k] * m->s[r];
  else
    for (int k = 0; k < m->K; ++k) {
      size_t idx = (size_t)r * m->Kp + (size_t)k;
      out[k] = (float)(((m->q4[idx >> 1] >> ((idx & 1) << 2)) & 15) - 8) * m->s[(size_t)r * (m->Kp / m->g) + k / m->g];
    }
}

/* ------------------------------------------------------------------ convolution */
int st_conv1d(st_pool* p, const float* x, int T, int C, int ldx, const st_mat* W, int k, int stride, int dil,
              int pad_l, int pad_r, const float* bias, float* y, int ldy) {
  if (k < 1 || stride < 1 || dil < 1 || pad_l < 0 || pad_r < 0 || T < 0 || C < 0) {
    st_seterr("conv1d: bad geometry (k %d, stride %d, dilation %d, padding %d/%d)", k, stride, dil, pad_l, pad_r);
    return -1;
  }
  int64_t num = (int64_t)T + pad_l + pad_r - (int64_t)dil * (k - 1) - 1;
  if (num < 0) return 0;   /* the input (with padding) is shorter than the kernel span: no output */
  if (num / stride + 1 > INT_MAX) {
    st_seterr("conv1d: output too long");
    return -1;
  }
  int To = (int)(num / stride + 1);
  if (k == 1 && stride == 1 && pad_l == 0 && pad_r == 0) {   /* a pointwise conv reads x in place */
    st_linear(p, x, To, ldx, W, bias, y, ldy, 0);
    return To;
  }
  size_t K = (size_t)C * k;
  /* im2col in row blocks so a long input does not need a [T, C*k] buffer at once */
  int rows = K ? (int)(((size_t)8 << 20) / (K * sizeof(float))) : To;
  if (rows < 64) rows = 64;
  if (rows > To) rows = To;
  float* col = (float*)st_malloc(sizeof(float) * K * (size_t)rows);
  if (!col) return -1;
  for (int t0 = 0; t0 < To; t0 += rows) {
    int nr = To - t0 < rows ? To - t0 : rows;
    for (int r = 0; r < nr; ++r) {
      float* c = col + (size_t)r * K;
      int base = (t0 + r) * stride - pad_l;
      for (int j = 0; j < k; ++j) {
        int ti = base + j * dil;
        if (ti < 0 || ti >= T) {
          for (int ch = 0; ch < C; ++ch) c[(size_t)ch * k + j] = 0.f;
        } else {
          const float* xr = x + (size_t)ti * ldx;
          for (int ch = 0; ch < C; ++ch) c[(size_t)ch * k + j] = xr[ch];
        }
      }
    }
    st_linear(p, col, nr, (int)K, W, bias, y + (size_t)t0 * ldy, ldy, 0);
  }
  st_free(col);
  return To;
}

void st_dwconv1d(const float* x, int T, int C, const float* w, int k, int dil, int pad_l, const float* bias, float* y) {
  for (int t = 0; t < T; ++t) {
    float* yr = y + (size_t)t * C;
    for (int c = 0; c < C; ++c) yr[c] = bias ? bias[c] : 0.f;
    for (int j = 0; j < k; ++j) {
      int ti = t + j * dil - pad_l;
      if (ti < 0 || ti >= T) continue;
      const float* xr = x + (size_t)ti * C;
      for (int c = 0; c < C; ++c) yr[c] += xr[c] * w[(size_t)c * k + j];
    }
  }
}

/* ------------------------------------------------------------------ elementwise / norms */
void st_layernorm(float* x, int T, int C, int ld, const float* gamma, const float* beta, float eps) {
  for (int t = 0; t < T; ++t) {
    float* r = x + (size_t)t * ld;
    double m = 0, v = 0;
    for (int c = 0; c < C; ++c) m += r[c];
    m /= C;
    for (int c = 0; c < C; ++c) {
      double d = r[c] - m;
      v += d * d;
    }
    v /= C;
    float inv = (float)(1.0 / sqrt(v + eps)), mf = (float)m;
    for (int c = 0; c < C; ++c) {
      float o = (r[c] - mf) * inv;
      if (gamma) o *= gamma[c];
      if (beta) o += beta[c];
      r[c] = o;
    }
  }
}

void st_rmsnorm(float* x, int T, int C, int ld, const float* gamma, float eps) {
  for (int t = 0; t < T; ++t) {
    float* r = x + (size_t)t * ld;
    double s = 0;
    for (int c = 0; c < C; ++c) s += (double)r[c] * r[c];
    float inv = (float)(1.0 / sqrt(s / C + eps));
    for (int c = 0; c < C; ++c) r[c] = r[c] * inv * (gamma ? gamma[c] : 1.f);
  }
}

/* GELU with erf as Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7): the form of PyTorch's own AVX2 kernel; 8 lanes at
 * a time with AVX2, libm erff otherwise */
#ifdef ST_X86
ST_TARGET_AVX2 static __m256 exp_neg256(__m256 x) {   /* Cephes expf for x <= 0 */
  x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
  __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
  __m256 y = _mm256_set1_ps(1.9875691500E-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
  y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.f)));
  __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

ST_TARGET_AVX2 static size_t gelu_avx2(float* x, size_t n) {
  const __m256 sgn = _mm256_set1_ps(-0.f), one = _mm256_set1_ps(1.f), half = _mm256_set1_ps(0.5f);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(x + i);
    __m256 z = _mm256_mul_ps(v, _mm256_set1_ps(0.70710678118654752f));
    __m256 s = _mm256_and_ps(sgn, z), a = _mm256_xor_ps(s, z);
    __m256 t = _mm256_div_ps(one, _mm256_fmadd_ps(_mm256_set1_ps(0.3275911f), a, one));
    __m256 r = _mm256_fmadd_ps(_mm256_set1_ps(1.061405429f), t, _mm256_set1_ps(-1.453152027f));
    r = _mm256_fmadd_ps(r, t, _mm256_set1_ps(1.421413741f));
    r = _mm256_fmadd_ps(r, t, _mm256_set1_ps(-0.284496736f));
    r = _mm256_fmadd_ps(r, t, _mm256_set1_ps(0.254829592f));
    __m256 ex = exp_neg256(_mm256_xor_ps(sgn, _mm256_mul_ps(z, z)));
    __m256 e = _mm256_xor_ps(s, _mm256_fmadd_ps(_mm256_mul_ps(_mm256_xor_ps(sgn, ex), t), r, one));
    _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(v, half), _mm256_add_ps(one, e)));
  }
  return i;
}
#endif

void st_gelu(float* x, size_t n) {
  size_t i = 0;
#ifdef ST_X86
  if (pick_kernel() == kern_avx2) i = gelu_avx2(x, n);
#endif
  for (; i < n; ++i) x[i] = 0.5f * x[i] * (1.f + erff(x[i] * 0.70710678118654752f));
}

void st_gelu_tanh(float* x, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float v = x[i];
    x[i] = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
  }
}

void st_silu(float* x, size_t n) {
  for (size_t i = 0; i < n; ++i) x[i] = x[i] / (1.f + expf(-x[i]));
}

void st_relu(float* x, size_t n) {
  for (size_t i = 0; i < n; ++i) x[i] = x[i] > 0.f ? x[i] : 0.f;
}

void st_softmax_rows(float* x, int rows, int cols, int ld) {
  for (int r = 0; r < rows; ++r) {
    float* v = x + (size_t)r * ld;
    float mx = -INFINITY;
    for (int c = 0; c < cols; ++c) mx = v[c] > mx ? v[c] : mx;
    if (mx == -INFINITY) {
      for (int c = 0; c < cols; ++c) v[c] = NAN;   /* torch: a fully masked row is NaN */
      continue;
    }
    float s = 0;
    for (int c = 0; c < cols; ++c) {
      v[c] = expf(v[c] - mx);
      s += v[c];
    }
    float inv = 1.f / s;
    for (int c = 0; c < cols; ++c) v[c] *= inv;
  }
}

/* ------------------------------------------------------------------ attention
 * Work items are (head, 64-query block) pairs. One task per pool thread claims items from a shared counter and
 * uses its own score buffer, so the scratch is allocated once per call, not per item. */
enum { QB = 64 };
typedef struct {
  const float *q, *k, *v;
  int ldq, ldk, ldv, ldo;
  float* out;
  int Tq, Tk, H, dh, nqb;
  const uint8_t* valid;
  const float* bias;
  float scale;
  float* scratch;   /* per task slot: QB * Tk */
  int n_items;
  atomic_int next;
} attn_job;

static void attn_item(const attn_job* a, int item, float* s) {
  int h = item / a->nqb, qb = item % a->nqb;
  int q0 = qb * QB, nq = a->Tq - q0 < QB ? a->Tq - q0 : QB;
  st_gemm_nt(NULL, nq, a->Tk, a->dh, a->q + (size_t)q0 * a->ldq + (size_t)h * a->dh, a->ldq, a->k + (size_t)h * a->dh,
             a->ldk, s, a->Tk, NULL, 0);
  for (int i = 0; i < nq; ++i) {
    float* r = s + (size_t)i * a->Tk;
    const float* b = a->bias ? a->bias + ((size_t)h * a->Tq + (size_t)(q0 + i)) * a->Tk : NULL;
    for (int j = 0; j < a->Tk; ++j) {
      r[j] *= a->scale;
      if (b) r[j] += b[j];
      if (a->valid && !a->valid[j]) r[j] = -INFINITY;
    }
  }
  st_softmax_rows(s, nq, a->Tk, a->Tk);
  st_gemm_nn(NULL, nq, a->dh, a->Tk, s, a->Tk, a->v + (size_t)h * a->dh, a->ldv,
             a->out + (size_t)q0 * a->ldo + (size_t)h * a->dh, a->ldo, NULL, 0);
}

static void attn_task(void* ctx, int task, int n_tasks) {
  (void)n_tasks;
  attn_job* a = (attn_job*)ctx;
  float* s = a->scratch + (size_t)task * QB * (size_t)(a->Tk > 0 ? a->Tk : 1);
  for (;;) {
    int item = atomic_fetch_add_explicit(&a->next, 1, memory_order_relaxed);
    if (item >= a->n_items) break;
    attn_item(a, item, s);
  }
}

int st_attention(st_pool* p, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv, float* out,
                 int ldo, int Tq, int Tk, int H, int dh, const uint8_t* key_valid, const float* bias, float scale) {
  if (Tq <= 0 || H <= 0) return 0;
  attn_job a = {q, k, v, ldq, ldk, ldv, ldo, out, Tq, Tk, H, dh, (Tq + QB - 1) / QB, key_valid, bias,
                scale > 0 ? scale : 1.f / sqrtf((float)dh), NULL, 0, 0};
  a.n_items = H * a.nqb;
  atomic_init(&a.next, 0);
  int slots = st_pool_size(p);
  if (slots > a.n_items) slots = a.n_items;
  a.scratch = (float*)st_malloc(sizeof(float) * (size_t)slots * QB * (size_t)(Tk > 0 ? Tk : 1));
  if (!a.scratch) {   /* st_malloc set the error; make the failure visible to callers that ignore the result */
    for (int t = 0; t < Tq; ++t)
      for (int c = 0; c < H * dh; ++c) out[(size_t)t * ldo + c] = NAN;
    return -1;
  }
  st_parallel_for(slots > 1 ? p : NULL, slots, attn_task, &a);
  st_free(a.scratch);
  return 0;
}

/* ------------------------------------------------------------------ FFT */
struct st_fft {
  int n;
  double* tw;   /* n/2 complex twiddles exp(-2 pi i k / n) */
  int* rev;
};

st_fft* st_fft_create(int n) {
  if (n < 2 || (n & (n - 1))) {
    st_seterr("FFT size %d is not a power of two", n);
    return NULL;
  }
  st_fft* f = (st_fft*)st_calloc(sizeof *f);
  if (!f) return NULL;
  f->n = n;
  f->tw = (double*)st_malloc(sizeof(double) * (size_t)n);
  f->rev = (int*)st_malloc(sizeof(int) * (size_t)n);
  if (!f->tw || !f->rev) {
    st_fft_destroy(f);
    return NULL;
  }
  const double pi = 3.14159265358979323846;
  for (int k = 0; k < n / 2; ++k) {
    f->tw[2 * k] = cos(2 * pi * k / n);
    f->tw[2 * k + 1] = -sin(2 * pi * k / n);
  }
  int bits = 0;
  while ((1 << bits) < n) ++bits;
  for (int i = 0; i < n; ++i) {
    int r = 0;
    for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
    f->rev[i] = r;
  }
  return f;
}

void st_fft_destroy(st_fft* f) {
  if (!f) return;
  st_free(f->tw);
  st_free(f->rev);
  st_free(f);
}

void st_fft_run(const st_fft* f, double* a, int inverse) {
  int n = f->n;
  for (int i = 0; i < n; ++i) {
    int r = f->rev[i];
    if (i < r) {
      double t0 = a[2 * i], t1 = a[2 * i + 1];
      a[2 * i] = a[2 * r], a[2 * i + 1] = a[2 * r + 1];
      a[2 * r] = t0, a[2 * r + 1] = t1;
    }
  }
  double sg = inverse ? -1.0 : 1.0;
  for (int len = 2; len <= n; len <<= 1) {
    int step = n / len, half = len / 2;
    for (int i = 0; i < n; i += len)
      for (int j = 0; j < half; ++j) {
        double wr = f->tw[2 * j * step], wi = sg * f->tw[2 * j * step + 1];
        double* u = a + 2 * (i + j);
        double* v = a + 2 * (i + j + half);
        double vr = v[0] * wr - v[1] * wi, vi = v[0] * wi + v[1] * wr;
        v[0] = u[0] - vr, v[1] = u[1] - vi;
        u[0] += vr, u[1] += vi;
      }
  }
}

void st_rfft(const st_fft* f, const float* x, double* spec, double* work) {
  int n = f->n;
  for (int i = 0; i < n; ++i) work[2 * i] = x[i], work[2 * i + 1] = 0;
  st_fft_run(f, work, 0);
  memcpy(spec, work, sizeof(double) * (size_t)(n + 2));
}

void st_irfft(const st_fft* f, const double* spec, float* y, double* work) {
  int n = f->n;
  work[0] = spec[0], work[1] = 0;
  work[n] = spec[n], work[n + 1] = 0;
  for (int k = 1; k < n / 2; ++k) {
    work[2 * k] = spec[2 * k], work[2 * k + 1] = spec[2 * k + 1];
    work[2 * (n - k)] = spec[2 * k], work[2 * (n - k) + 1] = -spec[2 * k + 1];
  }
  st_fft_run(f, work, 1);
  for (int i = 0; i < n; ++i) y[i] = (float)(work[2 * i] / n);
}

void st_hann_periodic(float* w, int n) {
  const double pi = 3.14159265358979323846;
  for (int i = 0; i < n; ++i) w[i] = (float)(0.5 - 0.5 * cos(2 * pi * i / n));
}
