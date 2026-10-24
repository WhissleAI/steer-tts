/* iSTFT, fades and WAV output (see st_dsp.h). The iFFT runs in double (st_fft), like native/src/dsp.cpp. */
#include "st_dsp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t st_istft_len(int T, int hop) { return T > 1 ? (size_t)(T - 1) * (size_t)hop : 0; }

typedef struct {
  const st_fft* fft;
  const float *re, *im, *win;
  int ld, T, n_fft;
  float* frames;   /* [T, n_fft] windowed frames */
  double* scratch; /* [n_tasks, 4 n_fft + 4], allocated by st_istft so a failure is reported, not skipped */
} istft_job;

static void istft_task(void* ctx, int task, int n_tasks) {
  const istft_job* j = (const istft_job*)ctx;
  int n = j->n_fft, nb = n / 2 + 1;
  int t0 = (int)((long long)j->T * task / n_tasks), t1 = (int)((long long)j->T * (task + 1) / n_tasks);
  double* buf = j->scratch + (size_t)task * (size_t)(4 * n + 4);
  double* spec = buf;
  double* work = buf + 2 * n + 2;
  for (int t = t0; t < t1; ++t) {
    const float *re = j->re + (size_t)t * j->ld, *im = j->im + (size_t)t * j->ld;
    for (int k = 0; k < nb; ++k) spec[2 * k] = re[k], spec[2 * k + 1] = im[k];
    float* f = j->frames + (size_t)t * n;
    st_irfft(j->fft, spec, f, work);
    for (int i = 0; i < n; ++i) f[i] *= j->win[i];
  }
}

int st_istft(st_pool* p, const float* re, const float* im, int ld, int T, int n_fft, int hop, float* out) {
  size_t n_out = st_istft_len(T, hop);
  if (n_out == 0) return 0;
  size_t size = (size_t)n_fft + (size_t)hop * (size_t)(T - 1);
  st_fft* fft = st_fft_create(n_fft);
  float* win = (float*)st_malloc(sizeof(float) * (size_t)n_fft);
  float* frames = (float*)st_malloc(sizeof(float) * (size_t)T * n_fft);
  float* y = (float*)st_calloc(sizeof(float) * size);
  float* env = (float*)st_calloc(sizeof(float) * size);
  int tasks = st_pool_size(p) * 4;
  if (tasks > T) tasks = T;
  double* scratch = (double*)st_malloc(sizeof(double) * (size_t)tasks * (size_t)(4 * n_fft + 4));
  int rc = -1;
  if (!fft || !win || !frames || !y || !env || !scratch) goto done;
  st_hann_periodic(win, n_fft);
  istft_job j = {fft, re, im, win, ld, T, n_fft, frames, scratch};
  st_parallel_for(p, tasks, istft_task, &j);
  /* overlap-add in frame order (float, as torch's fold), envelope = sum of window^2 */
  for (int t = 0; t < T; ++t) {
    float* yo = y + (size_t)t * hop;
    float* eo = env + (size_t)t * hop;
    const float* f = frames + (size_t)t * n_fft;
    for (int i = 0; i < n_fft; ++i) {
      yo[i] += f[i];
      eo[i] += win[i] * win[i];
    }
  }
  size_t a0 = (size_t)n_fft / 2;
  for (size_t i = 0; i < n_out; ++i) out[i] = y[a0 + i] / env[a0 + i];
  rc = 0;
done:
  st_free(scratch);
  st_fft_destroy(fft);
  st_free(win);
  st_free(frames);
  st_free(y);
  st_free(env);
  return rc;
}

void st_fade_ends(float* y, size_t n, int fade) {
  if (fade < 2 || n <= 2 * (size_t)fade) return;
  double step = 1.0 / (fade - 1);
  for (int i = 0; i < fade; ++i) {
    float r = i == fade - 1 ? 1.0f : (float)(i * step);
    y[i] *= r;
    y[n - 1 - (size_t)i] *= r;
  }
}

static void put32(unsigned char* b, unsigned v) {
  b[0] = (unsigned char)v, b[1] = (unsigned char)(v >> 8), b[2] = (unsigned char)(v >> 16), b[3] = (unsigned char)(v >> 24);
}
static void put16(unsigned char* b, unsigned v) { b[0] = (unsigned char)v, b[1] = (unsigned char)(v >> 8); }

int st_wav_encode(const float* y, size_t n, int sr, unsigned char** buf, size_t* len) {
  if (n > 0x7FFFFFF0u / 2 - 44) {
    st_seterr("wav: too many samples");
    return -1;
  }
  size_t bytes = n * 2;
  unsigned char* b = (unsigned char*)st_malloc(44 + bytes);
  if (!b) return -1;
  memcpy(b, "RIFF", 4), put32(b + 4, (unsigned)(36 + bytes)), memcpy(b + 8, "WAVE", 4);
  memcpy(b + 12, "fmt ", 4), put32(b + 16, 16), put16(b + 20, 1), put16(b + 22, 1), put32(b + 24, (unsigned)sr);
  put32(b + 28, (unsigned)sr * 2), put16(b + 32, 2), put16(b + 34, 16);
  memcpy(b + 36, "data", 4), put32(b + 40, (unsigned)bytes);
  for (size_t i = 0; i < n; ++i) {
    float v = y[i];
    v = v > 1.f ? 1.f : (v < -1.f ? -1.f : v);   /* NaN stays NaN -> 0 below */
    float r = nearbyintf(v * 32767.0f);
    int s = r == r ? (int)r : 0;
    put16(b + 44 + 2 * i, (unsigned)(s & 0xFFFF));
  }
  *buf = b;
  *len = 44 + bytes;
  return 0;
}

int st_wav_write(const char* path, const float* y, size_t n, int sr) {
  unsigned char* b;
  size_t len;
  ST_CHECK(st_wav_encode(y, n, sr, &b, &len));
  FILE* f = fopen(path, "wb");
  if (!f) {
    st_free(b);
    st_seterr("cannot write %s", path);
    return -1;
  }
  size_t w = fwrite(b, 1, len, f);
  int bad = fclose(f) != 0 || w != len;
  st_free(b);
  if (bad) {
    st_seterr("write error on %s", path);
    return -1;
  }
  return 0;
}
