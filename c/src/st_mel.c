/* Vocos / torchaudio log-mel spectrogram and SteerTTS.prompt_mel.
 *
 * STFT in double (st_rfft) on the reflect-padded signal with a periodic Hann window; |X|; mel filterbank built in
 * float32 the way torchaudio.functional.melscale_fbanks does (HTK scale, f_min 0, f_max sr/2, no norm);
 * log(max(mel, 1e-7)). */
#include "st_mel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "st_base.h"

enum { SR = 24000, NFFT = 1024, HOP = 256, NB = 100, NF = NFFT / 2 + 1 };

/* torch.linspace(a, b, n) in float32 (see linspace01 in st_flow.c) */
static void linspace_f(float a, float b, int n, float* y) {
  float step = (b - a) / (float)(n - 1);
  int half = n / 2;
  for (int i = 0; i < n; ++i)
    y[i] = i < half ? (float)((double)a + (double)step * i) : (float)((double)b - (double)step * (n - 1 - i));
}

/* fb [NF, NB] (torchaudio layout) as per-band ranges: band m covers bins [lo[m], hi[m]) with weights w[m][bin] */
typedef struct {
  float w[NB][NF];
  int lo[NB], hi[NB];
} melfb;

static void make_fb(melfb* fb) {
  float freqs[NF], mpts[NB + 2], fpts[NB + 2], fdiff[NB + 1];
  linspace_f(0.f, (float)(SR / 2), NF, freqs);
  double mmax = 2595.0 * log10(1.0 + (SR / 2) / 700.0);
  linspace_f(0.f, (float)mmax, NB + 2, mpts);
  for (int i = 0; i < NB + 2; ++i) fpts[i] = 700.0f * ((float)pow(10.0, (double)(mpts[i] / 2595.0f)) - 1.0f);
  for (int i = 0; i < NB + 1; ++i) fdiff[i] = fpts[i + 1] - fpts[i];
  for (int m = 0; m < NB; ++m) {
    fb->lo[m] = NF, fb->hi[m] = 0;
    for (int k = 0; k < NF; ++k) {
      float down = -(fpts[m] - freqs[k]) / fdiff[m];
      float up = (fpts[m + 2] - freqs[k]) / fdiff[m + 1];
      float v = down < up ? down : up;
      v = v > 0.f ? v : 0.f;
      fb->w[m][k] = v;
      if (v > 0.f) {
        if (k < fb->lo[m]) fb->lo[m] = k;
        fb->hi[m] = k + 1;
      }
    }
  }
}

int st_mel_spectrogram(const float* wav, size_t n, float** mel, int* T) {
  *mel = NULL, *T = 0;
  if (n <= NFFT / 2) {
    st_seterr("mel: %zu samples is too short (need > %d)", n, NFFT / 2);
    return -1;
  }
  int nt = 1 + (int)(n / HOP), pad = NFFT / 2;
  melfb* fb = (melfb*)st_malloc(sizeof *fb);
  st_fft* fft = st_fft_create(NFFT);
  float* out = (float*)st_malloc(sizeof(float) * (size_t)nt * NB);
  float *win = (float*)st_malloc(sizeof(float) * NFFT), *frame = (float*)st_malloc(sizeof(float) * NFFT);
  double* spec = (double*)st_malloc(sizeof(double) * (NFFT + 2));
  double* work = (double*)st_malloc(sizeof(double) * 2 * NFFT);
  float* mag = (float*)st_malloc(sizeof(float) * NF);
  if (!fb || !fft || !out || !win || !frame || !spec || !work || !mag) {
    st_free(fb), st_fft_destroy(fft), st_free(out), st_free(win), st_free(frame), st_free(spec), st_free(work);
    st_free(mag);
    return -1;
  }
  make_fb(fb);
  st_hann_periodic(win, NFFT);
  for (int t = 0; t < nt; ++t) {
    long s0 = (long)t * HOP - pad;
    for (int i = 0; i < NFFT; ++i) {
      long j = s0 + i;
      if (j < 0) j = -j;                                   /* reflect (edge sample not repeated) */
      else if (j >= (long)n) j = 2 * ((long)n - 1) - j;
      frame[i] = wav[j] * win[i];
    }
    st_rfft(fft, frame, spec, work);
    for (int k = 0; k < NF; ++k) mag[k] = (float)sqrt(spec[2 * k] * spec[2 * k] + spec[2 * k + 1] * spec[2 * k + 1]);
    float* o = out + (size_t)t * NB;
    for (int m = 0; m < NB; ++m) {
      double s = 0;
      for (int k = fb->lo[m]; k < fb->hi[m]; ++k) s += (double)fb->w[m][k] * mag[k];
      float v = (float)s;
      o[m] = logf(v > 1e-7f ? v : 1e-7f);
    }
  }
  st_free(fb), st_fft_destroy(fft), st_free(win), st_free(frame), st_free(spec), st_free(work), st_free(mag);
  *mel = out, *T = nt;
  return 0;
}

static int cmp_float(const void* a, const void* b) {
  float x = *(const float*)a, y = *(const float*)b;
  return (x > y) - (x < y);
}

/* mean over the upper half (sorted) of the per-frame mean of bands [0, nb) */
static float loudness(const float* m, int T, int nb, float* scratch) {
  for (int t = 0; t < T; ++t) {
    double s = 0;
    for (int b = 0; b < nb; ++b) s += m[(size_t)t * NB + b];
    scratch[t] = (float)(s / nb);
  }
  qsort(scratch, (size_t)T, sizeof(float), cmp_float);
  double s = 0;
  for (int t = T / 2; t < T; ++t) s += scratch[t];
  return (float)(s / (T - T / 2));
}

int st_prompt_mel(const float* wav24, size_t n, int orig_sr, const float* render_mel, int T, float** pm, int* P) {
  *pm = NULL, *P = 0;
  if (n > (size_t)8 * SR) n = (size_t)8 * SR;
  if (n < SR / 2) return 0;
  if (!render_mel || T <= 0) {
    st_seterr("prompt_mel: no render mel");
    return -1;
  }
  float* m;
  int np;
  ST_CHECK(st_mel_spectrogram(wav24, n, &m, &np));
  int nb = NB;
  if (orig_sr > 0 && orig_sr < SR) {
    /* numpy: hz = 700 (10^(linspace(0, mel(sr/2), nb + 2) / 2595) - 1); kb = searchsorted(hz[2:], 0.95 sr / 2) */
    double mmax = 2595.0 * log10(1.0 + (SR / 2) / 700.0), step = mmax / (NB + 1), v = 0.95 * orig_sr / 2;
    int kb = 0;
    for (int i = 2; i < NB + 2; ++i) {
      double mv = i == NB + 1 ? mmax : (double)i * step;
      double hz = 700.0 * (pow(10.0, mv / 2595.0) - 1.0);
      if (hz < v) ++kb;
    }
    nb = kb < NB ? kb : NB;
    if (nb < 8) nb = 8;
  }
  float* scratch = (float*)st_malloc(sizeof(float) * (size_t)(np > T ? np : T));
  if (!scratch) {
    st_free(m);
    return -1;
  }
  float shift = loudness(render_mel, T, nb, scratch) - loudness(m, np, nb, scratch);
  for (size_t i = 0; i < (size_t)np * NB; ++i) m[i] += shift;
  if (nb < NB) {
    float shape[NB];
    for (int b = nb - 1; b < NB; ++b) {
      double s = 0;
      for (int t = 0; t < T; ++t) s += render_mel[(size_t)t * NB + b];
      shape[b] = (float)(s / T);
    }
    for (int t = 0; t < np; ++t) {
      float* r = m + (size_t)t * NB;
      for (int b = nb; b < NB; ++b) r[b] = r[nb - 1] + (shape[b] - shape[nb - 1]);
    }
  }
  st_free(scratch);
  *pm = m, *P = np;
  return 0;
}
