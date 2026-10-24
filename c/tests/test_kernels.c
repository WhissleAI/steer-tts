/* Kernel self-test: GEMM (all weight storages, edge sizes, threads) against a double-precision reference, plus
 * conv1d, attention and FFT round trips. Prints GFLOP/s for a model-sized matmul. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/st_base.h"

static float frand(unsigned* s) { *s = *s * 1664525u + 1013904223u; return ((*s >> 8) / 16777216.0f) * 2.f - 1.f; }

static double maxrel(const float* a, const double* b, size_t n) {
  double m = 0, s = 0;
  for (size_t i = 0; i < n; ++i) { double d = fabs(a[i] - b[i]); m = d > m ? d : m; s = fabs(b[i]) > s ? fabs(b[i]) : s; }
  return m / (s > 0 ? s : 1);
}

int main(void) {
  unsigned seed = 1;
  int fails = 0;
  st_pool* pool = st_pool_create(4);
  int sizes[][3] = {{1, 7, 5}, {3, 33, 300}, {7, 16, 16}, {100, 192, 576}, {257, 100, 192}, {64, 1000, 64}, {13, 1, 9}};
  for (int si = 0; si < 7; ++si) {
    int M = sizes[si][0], N = sizes[si][1], K = sizes[si][2];
    float* A = malloc(sizeof(float) * M * K);
    float* B = malloc(sizeof(float) * N * K);
    float* Bt = malloc(sizeof(float) * N * K);
    float* bias = malloc(sizeof(float) * N);
    float* C = malloc(sizeof(float) * M * N);
    double* R = malloc(sizeof(double) * M * N);
    for (int i = 0; i < M * K; ++i) A[i] = frand(&seed);
    for (int i = 0; i < N * K; ++i) B[i] = frand(&seed);
    for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) Bt[k * N + n] = B[n * K + k];
    for (int i = 0; i < N; ++i) bias[i] = frand(&seed);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
      double s = bias[n];
      for (int k = 0; k < K; ++k) s += (double)A[m * K + k] * B[n * K + k];
      R[m * N + n] = s;
    }
    for (int th = 0; th < 2; ++th) {
      st_pool* p = th ? pool : NULL;
      st_gemm_nt(p, M, N, K, A, K, B, K, C, N, bias, 0);
      double e1 = maxrel(C, R, (size_t)M * N);
      st_gemm_nn(p, M, N, K, A, K, Bt, N, C, N, bias, 0);
      double e2 = maxrel(C, R, (size_t)M * N);
      st_mat W = {ST_W_F32, N, K, B, NULL, NULL, NULL, 0, 0};
      st_linear(p, A, M, K, &W, bias, C, N, 0);
      double e3 = maxrel(C, R, (size_t)M * N);
      /* int8: quantize B per row, reference with the dequantized weights */
      int8_t* q8 = malloc(N * K); float* s8 = malloc(sizeof(float) * N);
      int g = 32, Kp = (K + g - 1) / g * g; uint8_t* q4 = calloc(N * Kp / 2 + 1, 1); float* s4 = malloc(sizeof(float) * N * (Kp / g));
      double* R8 = malloc(sizeof(double) * M * N); double* R4 = malloc(sizeof(double) * M * N);
      float* D8 = malloc(sizeof(float) * N * K); float* D4 = malloc(sizeof(float) * N * K);
      for (int n = 0; n < N; ++n) {
        float mx = 0; for (int k = 0; k < K; ++k) mx = fmaxf(mx, fabsf(B[n * K + k]));
        s8[n] = mx / 127; for (int k = 0; k < K; ++k) { q8[n * K + k] = (int8_t)lrintf(B[n * K + k] / s8[n]); D8[n * K + k] = q8[n * K + k] * s8[n]; }
        for (int gi = 0; gi < Kp / g; ++gi) {
          float m4 = 0; for (int k = gi * g; k < (gi + 1) * g && k < K; ++k) m4 = fmaxf(m4, fabsf(B[n * K + k]));
          float sc = m4 / 7; s4[n * (Kp / g) + gi] = sc;
          for (int k = gi * g; k < (gi + 1) * g; ++k) {
            int v = k < K && sc > 0 ? (int)lrintf(B[n * K + k] / sc) : 0; v = v < -8 ? -8 : v > 7 ? 7 : v;
            size_t idx = (size_t)n * Kp + k; q4[idx >> 1] |= (uint8_t)((v + 8) << ((idx & 1) * 4));
            if (k < K) D4[n * K + k] = v * sc;
          }
        }
      }
      for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        double a = bias[n], b = bias[n];
        for (int k = 0; k < K; ++k) { a += (double)A[m * K + k] * D8[n * K + k]; b += (double)A[m * K + k] * D4[n * K + k]; }
        R8[m * N + n] = a; R4[m * N + n] = b;
      }
      st_mat W8 = {ST_W_I8, N, K, NULL, q8, NULL, s8, 0, 0};
      st_linear(p, A, M, K, &W8, bias, C, N, 0);
      double e4 = maxrel(C, R8, (size_t)M * N);
      st_mat W4 = {ST_W_I4, N, K, NULL, NULL, q4, s4, g, Kp};
      st_linear(p, A, M, K, &W4, bias, C, N, 0);
      double e5 = maxrel(C, R4, (size_t)M * N);
      int bad = e1 > 1e-5 || e2 > 1e-5 || e3 > 1e-5 || e4 > 1e-5 || e5 > 1e-5;
      fails += bad;
      printf("%s gemm %4dx%4dx%4d thr=%d  nt %.1e nn %.1e lin %.1e i8 %.1e i4 %.1e\n", bad ? "FAIL" : "ok  ", M, N, K, th ? 4 : 1, e1, e2, e3, e4, e5);
      free(q8); free(s8); free(q4); free(s4); free(R8); free(R4); free(D8); free(D4);
    }
    free(A); free(B); free(Bt); free(bias); free(C); free(R);
  }
  /* conv1d (k=5, stride 2, dil 2, pads 3/1) vs direct */
  {
    int T = 50, Cc = 7, O = 9, k = 5, st = 2, dil = 2, pl = 3, pr = 1;
    float x[50 * 7], w[9 * 7 * 5], b[9], y[64 * 9];
    for (int i = 0; i < T * Cc; ++i) x[i] = frand(&seed);
    for (int i = 0; i < O * Cc * k; ++i) w[i] = frand(&seed);
    for (int i = 0; i < O; ++i) b[i] = frand(&seed);
    st_mat W = {ST_W_F32, O, Cc * k, w, NULL, NULL, NULL, 0, 0};
    int To = st_conv1d(pool, x, T, Cc, Cc, &W, k, st, dil, pl, pr, b, y, O);
    double err = 0;
    for (int t = 0; t < To; ++t) for (int o = 0; o < O; ++o) {
      double s = b[o];
      for (int c = 0; c < Cc; ++c) for (int j = 0; j < k; ++j) { int ti = t * st + j * dil - pl; if (ti >= 0 && ti < T) s += x[ti * Cc + c] * w[(o * Cc + c) * k + j]; }
      err = fmax(err, fabs(s - y[t * O + o]));
    }
    int expT = (T + pl + pr - dil * (k - 1) - 1) / st + 1;
    printf("%s conv1d To=%d err %.1e\n", (err < 1e-4 && To == expT) ? "ok  " : "FAIL", To, err);
    fails += !(err < 1e-4 && To == expT);
  }
  /* attention with a key mask vs direct */
  {
    int Tq = 70, Tk = 90, H = 3, dh = 8, D = H * dh;
    float *q = malloc(4 * Tq * D), *kk = malloc(4 * Tk * D), *v = malloc(4 * Tk * D), *o = malloc(4 * Tq * D);
    uint8_t valid[90];
    for (int i = 0; i < Tq * D; ++i) q[i] = frand(&seed);
    for (int i = 0; i < Tk * D; ++i) { kk[i] = frand(&seed); v[i] = frand(&seed); }
    for (int j = 0; j < Tk; ++j) valid[j] = j < 80;
    st_attention(pool, q, D, kk, D, v, D, o, D, Tq, Tk, H, dh, valid, NULL, 0);
    double err = 0;
    for (int h = 0; h < H; ++h) for (int i = 0; i < Tq; ++i) {
      double s[90], mx = -1e30, z = 0;
      for (int j = 0; j < Tk; ++j) { double d = 0; for (int c = 0; c < dh; ++c) d += q[i * D + h * dh + c] * kk[j * D + h * dh + c]; s[j] = valid[j] ? d / sqrt(dh) : -1e30; mx = fmax(mx, s[j]); }
      for (int j = 0; j < Tk; ++j) { s[j] = valid[j] ? exp(s[j] - mx) : 0; z += s[j]; }
      for (int c = 0; c < dh; ++c) { double a = 0; for (int j = 0; j < Tk; ++j) a += s[j] / z * v[j * D + h * dh + c]; err = fmax(err, fabs(a - o[i * D + h * dh + c])); }
    }
    printf("%s attention err %.1e\n", err < 1e-5 ? "ok  " : "FAIL", err);
    fails += !(err < 1e-5);
    free(q); free(kk); free(v); free(o);
  }
  /* FFT round trip */
  {
    st_fft* f = st_fft_create(1024);
    float x[1024], y[1024]; double spec[1026], work[2048];
    for (int i = 0; i < 1024; ++i) x[i] = frand(&seed);
    st_rfft(f, x, spec, work); st_irfft(f, spec, y, work);
    double err = 0; for (int i = 0; i < 1024; ++i) err = fmax(err, fabs(x[i] - y[i]));
    printf("%s fft round trip err %.1e\n", err < 1e-6 ? "ok  " : "FAIL", err);
    fails += !(err < 1e-6);
    st_fft_destroy(f);
  }
  /* speed: [512 x 384] x [1152 x 384]^T, the flow refiner's pointwise layer at ~5 s of audio */
  {
    int M = 512, N = 1152, K = 384;
    float *A = malloc(4 * M * K), *B = malloc(4 * N * K), *C = malloc(4 * M * N);
    for (int i = 0; i < M * K; ++i) A[i] = frand(&seed);
    for (int i = 0; i < N * K; ++i) B[i] = frand(&seed);
    st_mat W = {ST_W_F32, N, K, B, NULL, NULL, NULL, 0, 0};
    for (int th = 0; th < 2; ++th) {
      st_pool* p = th ? pool : NULL;
      st_linear(p, A, M, K, &W, NULL, C, N, 0);
      double t0 = st_time_ms(); int reps = 20;
      for (int r = 0; r < reps; ++r) st_linear(p, A, M, K, &W, NULL, C, N, 0);
      double ms = (st_time_ms() - t0) / reps;
      printf("speed f32 %dx%dx%d threads=%d: %.2f ms, %.1f GFLOP/s\n", M, N, K, th ? 4 : 1, ms, 2.0 * M * N * K / ms / 1e6);
    }
    free(A); free(B); free(C);
  }
  st_pool_destroy(pool);
  printf(fails ? "%d FAILED\n" : "all ok\n", fails);
  return fails != 0;
}
