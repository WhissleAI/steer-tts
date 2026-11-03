/* Output level (see st_output.h); mirrors steertts/output.py. */
#include "st_output.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int st_output_load(st_output* o, const st_json* cfg, char* const* voices, int n_voices) {
  memset(o, 0, sizeof *o);
  const st_json* j = st_json_get(cfg, "output");
  o->target_lufs = (float)st_json_num(st_json_get(j, "target_lufs"), -17.0);
  o->true_peak_db = (float)st_json_num(st_json_get(j, "true_peak_db"), -1.0);
  o->knee_db = (float)st_json_num(st_json_get(j, "knee_db"), -3.0);
  o->gain_db = (float)st_json_num(st_json_get(j, "gain_db"), 0.0);
  o->n_voices = n_voices;
  if (n_voices > 0) {
    o->voice_gain = (float*)malloc(sizeof(float) * (size_t)n_voices);
    if (!o->voice_gain) {
      st_seterr("out of memory");
      return -1;
    }
    const st_json* vg = st_json_get(j, "voice_gain_db");
    for (int i = 0; i < n_voices; ++i) {
      const st_json* v = st_json_get(vg, voices[i]);
      o->voice_gain[i] = v && v->type == ST_JNUM ? (float)v->num : NAN;
    }
  }
  return 0;
}

void st_output_free(st_output* o) {
  free(o->voice_gain);
  memset(o, 0, sizeof *o);
}

int st_output_level(const char* name) {
  if (!name) return -1;
  if (!strcmp(name, "auto")) return ST_LEVEL_AUTO;
  if (!strcmp(name, "clip")) return ST_LEVEL_CLIP;
  if (!strcmp(name, "off")) return ST_LEVEL_OFF;
  return -1;
}

const char* st_output_level_name(int level) {
  return level == ST_LEVEL_OFF ? "off" : level == ST_LEVEL_CLIP ? "clip" : "auto";
}

float st_output_gain(const st_output* o, int vi) {
  if (vi >= 0 && vi < o->n_voices && o->voice_gain && !isnan(o->voice_gain[vi])) return o->voice_gain[vi];
  return o->gain_db;
}

static float db2lin(double db) { return (float)pow(10.0, db / 20.0); }

void st_soft_limit(float* y, size_t n, float knee_db, float ceiling_db) {
  const float t = db2lin(knee_db), c = db2lin(ceiling_db), w = c - t;
  for (size_t i = 0; i < n; ++i) {
    float a = fabsf(y[i]);
    if (a > t) y[i] = copysignf(t + w * tanhf((a - t) / w), y[i]);
  }
}

/* 48-tap windowed-sinc (Hann) 4x interpolator as 4 phases of 12 taps, the table steertts/output.py builds */
#define OS 4
#define TAPS 48
static void interp_fir(double h[OS][TAPS / OS]) {
  for (int k = 0; k < TAPS; ++k) {
    double m = k - (TAPS - 1) / 2.0, x = m / OS;
    double s = x == 0 ? 1.0 : sin(M_PI * x) / (M_PI * x);
    h[k % OS][k / OS] = s * (0.5 - 0.5 * cos(2 * M_PI * (k + 0.5) / TAPS));
  }
}

double st_true_peak(const float* y, size_t n) {
  double p = 0, h[OS][TAPS / OS];
  for (size_t i = 0; i < n; ++i) p = fmax(p, fabs((double)y[i]));
  interp_fir(h);
  const int L = TAPS / OS;
  for (int ph = 0; ph < OS; ++ph)
    for (size_t k = 0; k < n + L - 1; ++k) {   /* np.convolve(x, h[ph], "full") */
      double acc = 0;
      for (int j = 0; j < L; ++j)
        if (k >= (size_t)j && k - j < n) acc += h[ph][j] * y[k - j];
      p = fmax(p, fabs(acc));
    }
  return p;
}

static void biquad(const double b[3], const double a[3], double* x, size_t n) {
  double z1 = 0, z2 = 0;
  for (size_t i = 0; i < n; ++i) {
    double v = x[i], o = b[0] * v + z1;
    z1 = b[1] * v - a[1] * o + z2;
    z2 = b[2] * v - a[2] * o;
    x[i] = o;
  }
}

double st_lufs(const float* y, size_t n, int sr) {
  if (!n) return -70.0;
  double* x = (double*)malloc(sizeof(double) * n);
  if (!x) return -70.0;
  for (size_t i = 0; i < n; ++i) x[i] = y[i];
  {  /* K-weighting: high shelf +4 dB at 1500 Hz, then a 38 Hz high pass (RBJ biquads) */
    double A = pow(10, 4.0 / 40), w = 2 * M_PI * 1500.0 / sr, q = 1 / sqrt(2.0), al = sin(w) / (2 * q), c = cos(w);
    double b[3] = {A * ((A + 1) + (A - 1) * c + 2 * sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c),
                   A * ((A + 1) + (A - 1) * c - 2 * sqrt(A) * al)};
    double a[3] = {(A + 1) - (A - 1) * c + 2 * sqrt(A) * al, 2 * ((A - 1) - (A + 1) * c),
                   (A + 1) - (A - 1) * c - 2 * sqrt(A) * al};
    for (int k = 2; k >= 0; --k) b[k] /= a[0], a[k] /= a[0];
    biquad(b, a, x, n);
    w = 2 * M_PI * 38.0 / sr, q = 0.5, al = sin(w) / (2 * q), c = cos(w);
    double b2[3] = {(1 + c) / 2, -(1 + c), (1 + c) / 2}, a2[3] = {1 + al, -2 * c, 1 - al};
    for (int k = 2; k >= 0; --k) b2[k] /= a2[0], a2[k] /= a2[0];
    biquad(b2, a2, x, n);
  }
  size_t blk = (size_t)(0.4 * sr), hop = (size_t)(0.1 * sr), nb = n < blk ? 1 : (n - blk) / hop + 1;
  double* z = (double*)malloc(sizeof(double) * nb);
  double* cs = (double*)malloc(sizeof(double) * (n + 1));
  if (!z || !cs) {
    free(x), free(z), free(cs);
    return -70.0;
  }
  cs[0] = 0;
  for (size_t i = 0; i < n; ++i) cs[i + 1] = cs[i] + x[i] * x[i];
  if (n < blk) z[0] = cs[n] / n;
  else
    for (size_t b = 0; b < nb; ++b) z[b] = (cs[b * hop + blk] - cs[b * hop]) / blk;
  double s = 0;
  size_t m = 0;
  for (size_t b = 0; b < nb; ++b)
    if (-0.691 + 10 * log10(z[b] + 1e-12) > -70) s += z[b], ++m;
  double out = -70.0;
  if (m) {
    double rel = -0.691 + 10 * log10(s / m) - 10, s2 = 0;
    size_t m2 = 0;
    for (size_t b = 0; b < nb; ++b)
      if (-0.691 + 10 * log10(z[b] + 1e-12) > -70 && -0.691 + 10 * log10(z[b]) > rel) s2 += z[b], ++m2;
    out = -0.691 + 10 * log10(s2 / m2);
  }
  free(x), free(z), free(cs);
  return out;
}

static void apply(const st_output* o, float* y, size_t n, double g_db) {
  const float g = db2lin(g_db);
  for (size_t i = 0; i < n; ++i) y[i] *= g;
  st_soft_limit(y, n, o->knee_db, o->true_peak_db);
}

void st_output_chunk(const st_output* o, float* y, size_t n, int level, int vi, double gain_db) {
  if (level == ST_LEVEL_OFF || !n) return;
  apply(o, y, n, isnan(gain_db) ? st_output_gain(o, vi) : gain_db);
}

void st_output_clip(const st_output* o, float* y, size_t n, int sr, int level, int vi, double gain_db) {
  if (level == ST_LEVEL_OFF || !n) return;
  double g = gain_db;
  if (isnan(g)) {
    if (level == ST_LEVEL_CLIP) {
      int any = 0;
      for (size_t i = 0; i < n && !any; ++i) any = y[i] != 0;
      g = any ? fmin(fmax(o->target_lufs - st_lufs(y, n, sr), ST_GAIN_MIN), 30.0) : 0.0;
    } else {
      g = st_output_gain(o, vi);
    }
  }
  apply(o, y, n, g);
  double tp = st_true_peak(y, n), cap = db2lin(o->true_peak_db);
  if (tp > cap) {
    float k = (float)(cap / tp);
    for (size_t i = 0; i < n; ++i) y[i] *= k;
  }
}
