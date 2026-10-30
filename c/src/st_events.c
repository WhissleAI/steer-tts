/* Recorded sound events (steertts/events.py EventBank): see st_events.h. */
#include "st_events.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "st_base.h"

#define N_MELS 100
#define SR 24000
#define HOP 256

static const char* const k_nonverbal[] = {"laugh", "cough", "sigh", "sneeze", "throat"};
static const double k_gain_db[2] = {-2.0, -3.0};   /* nonverbal, environmental: dB relative to the speech */
static const double k_max_shift = 4.0;             /* semitones */
static const double k_gap_s = 0.08;

typedef struct {
  char* type;
  char gender;
  double f0;
  int start, n;
} clip_t;

struct st_events {
  st_file* f;
  const uint8_t* q;
  float *lo, *step;
  int n_clips;
  clip_t* clips;
};

static int is_nonverbal(const char* t) {
  for (size_t i = 0; i < sizeof k_nonverbal / sizeof *k_nonverbal; ++i)
    if (!strcmp(t, k_nonverbal[i])) return 1;
  return 0;
}

void st_events_free(st_events* b) {
  if (!b) return;
  for (int i = 0; i < b->n_clips; ++i) st_free(b->clips[i].type);
  st_free(b->clips);
  st_free(b->lo);
  st_free(b->step);
  st_file_close(b->f);
  st_free(b);
}

static float* f32_copy(const st_tensor* t, int n) {
  if (!t || st_numel(t) != n) return NULL;
  float* o = (float*)st_malloc(sizeof(float) * (size_t)(n ? n : 1));
  if (!o) return NULL;
  if (t->dtype == ST_F32) {
    memcpy(o, t->data, sizeof(float) * (size_t)n);
  } else if (t->dtype == ST_F16) {
    for (int i = 0; i < n; ++i) {
      uint16_t h;
      memcpy(&h, (const uint8_t*)t->data + 2 * i, 2);
      o[i] = st_half_to_float(h);
    }
  } else {
    st_free(o);
    return NULL;
  }
  return o;
}

st_events* st_events_open(const char* path) {
  st_events* b = (st_events*)st_calloc(sizeof *b);
  if (!b) return NULL;
  st_json* idx = NULL;
  if (!(b->f = st_file_open(path))) goto fail;
  const char* fmt = st_file_metadata(b->f, "format");
  const char* is = st_file_metadata(b->f, "index");
  if (!fmt || strcmp(fmt, "steertts-events-1") || !is) {
    st_seterr("%s: not a SteerTTS event bank", path);
    goto fail;
  }
  if (!(idx = st_json_parse(is, strlen(is)))) goto fail;
  if (idx->type != ST_JARR) {
    st_seterr("%s: bad event index", path);
    goto fail;
  }
  b->n_clips = idx->n;
  const st_tensor* q = st_file_find(b->f, "mel_q");
  if (!q || q->dtype != ST_U8 || q->ndim != 2 || q->shape[1] != N_MELS) {
    st_seterr("%s: bad mel_q", path);
    goto fail;
  }
  b->q = (const uint8_t*)q->data;
  b->lo = f32_copy(st_file_find(b->f, "lo"), b->n_clips);
  b->step = f32_copy(st_file_find(b->f, "step"), b->n_clips);
  b->clips = (clip_t*)st_calloc(sizeof(clip_t) * (size_t)(b->n_clips ? b->n_clips : 1));
  if (!b->lo || !b->step || !b->clips) {
    st_seterr("%s: bad lo / step", path);
    goto fail;
  }
  for (int i = 0; i < b->n_clips; ++i) {
    const st_json* c = st_json_at(idx, i);
    clip_t* k = &b->clips[i];
    const char* g = st_json_str(st_json_get(c, "gender"), "");
    k->type = st_strdup(st_json_str(st_json_get(c, "type"), ""));
    k->gender = !strcmp(g, "female") ? 'f' : !strcmp(g, "male") ? 'm' : 0;
    k->f0 = st_json_num(st_json_get(c, "f0"), 0.0);
    k->start = (int)st_json_num(st_json_get(c, "start"), 0);
    k->n = (int)st_json_num(st_json_get(c, "n"), 0);
    if (!k->type || k->n < 1 || k->start < 0 || k->start + k->n > q->shape[0]) {
      st_seterr("%s: bad event index entry %d", path, i);
      goto fail;
    }
  }
  st_json_free(idx);
  return b;
fail:
  st_json_free(idx);
  st_events_free(b);
  return NULL;
}

int st_events_has(const st_events* b, const char* type) {
  for (int i = 0; b && i < b->n_clips; ++i)
    if (!strcmp(b->clips[i].type, type)) return 1;
  return 0;
}

static int cmp_float(const void* a, const void* b) {
  float x = *(const float*)a, y = *(const float*)b;
  return (x > y) - (x < y);
}

float st_events_loudness(const float* mel, int T) {
  if (T < 1) return -6.0f;
  float* e = (float*)st_malloc(sizeof(float) * (size_t)T);
  if (!e) return -6.0f;
  for (int t = 0; t < T; ++t) {
    double s = 0;
    for (int k = 0; k < N_MELS; ++k) s += mel[(size_t)t * N_MELS + k];
    e[t] = (float)(s / N_MELS);
  }
  qsort(e, (size_t)T, sizeof(float), cmp_float);
  double s = 0;
  for (int t = T / 2; t < T; ++t) s += e[t];
  float r = (float)(s / (T - T / 2));
  st_free(e);
  return r;
}

static double hz2mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
static double mel2hz(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

/* events.pitch_shift on a time-major [T, 100] mel, in place via a copy */
static void pitch_shift(float* mel, int T, double ratio) {
  if (fabs(log(ratio)) < 1e-3) return;
  double top = hz2mel(SR / 2.0), step = top / (N_MELS + 1);
  int i0[N_MELS], i1[N_MELS];
  double w[N_MELS];
  for (int k = 0; k < N_MELS; ++k) {
    double c = step * (k + 1);
    double src = (hz2mel(mel2hz(c) / ratio) - step) / step;
    if (src < 0) src = 0;
    if (src > N_MELS - 1) src = N_MELS - 1;
    i0[k] = (int)floor(src);
    i1[k] = i0[k] + 1 < N_MELS ? i0[k] + 1 : N_MELS - 1;
    w[k] = src - i0[k];
  }
  float row[N_MELS];
  for (int t = 0; t < T; ++t) {
    float* m = mel + (size_t)t * N_MELS;
    memcpy(row, m, sizeof row);
    for (int k = 0; k < N_MELS; ++k) m[k] = (float)((1 - w[k]) * row[i0[k]] + w[k] * row[i1[k]]);
  }
}

typedef struct {
  int i;
  double d;
} cand_t;

static int cmp_cand(const void* a, const void* b) {
  const cand_t *x = (const cand_t*)a, *y = (const cand_t*)b;
  if (x->d != y->d) return (x->d > y->d) - (x->d < y->d);
  return x->i - y->i;
}

/* EventBank.pick */
static int pick(const st_events* b, const char* type, char gender, double f0, int64_t seed) {
  cand_t* c = (cand_t*)st_malloc(sizeof(cand_t) * (size_t)(b->n_clips ? b->n_clips : 1));
  if (!c) return -1;
  int n = 0, same = 0;
  for (int i = 0; i < b->n_clips; ++i)
    if (!strcmp(b->clips[i].type, type)) {
      c[n].i = i, c[n].d = 0;
      same += gender && b->clips[i].gender == gender;
      ++n;
    }
  if (same) {  /* cand = same or cand */
    int m = 0;
    for (int k = 0; k < n; ++k)
      if (b->clips[c[k].i].gender == gender) c[m++] = c[k];
    n = m;
  }
  if (f0 > 0 && is_nonverbal(type)) {
    int m = 0;
    for (int k = 0; k < n; ++k)
      if (b->clips[c[k].i].f0 > 0) c[m].i = c[k].i, c[m].d = fabs(log(b->clips[c[k].i].f0 / f0)), ++m;
    if (m) {
      qsort(c, (size_t)m, sizeof *c, cmp_cand);
      n = m < 2 ? m : 2;
    }
  }
  int r = -1;
  if (n) {
    int64_t k = seed % n;   /* Python's modulo is never negative */
    r = c[k < 0 ? k + n : k].i;
  } else {
    st_seterr("no event clip for [%s]", type);
  }
  st_free(c);
  return r;
}

int st_events_mel(const st_events* b, const char* type, char gender, double f0, int has_ref, float ref_loud,
                  int64_t seed, float** mel, int* T) {
  *mel = NULL;
  *T = 0;
  int i = pick(b, type, gender, f0, seed);
  if (i < 0) return -1;
  const clip_t* c = &b->clips[i];
  int gap = (int)(k_gap_s * SR / HOP);
  int n = c->n, tot = n + 2 * gap;
  float* m = (float*)st_malloc(sizeof(float) * (size_t)tot * N_MELS);
  if (!m) return -1;
  float* x = m + (size_t)gap * N_MELS;
  const uint8_t* q = b->q + (size_t)c->start * N_MELS;
  for (size_t k = 0; k < (size_t)n * N_MELS; ++k) x[k] = b->lo[i] + (float)q[k] * b->step[i];
  if (is_nonverbal(type) && f0 > 0 && c->f0 > 0) {
    double st = 12.0 * log2(f0 / c->f0);
    if (st > k_max_shift) st = k_max_shift;
    if (st < -k_max_shift) st = -k_max_shift;
    pitch_shift(x, n, pow(2.0, st / 12.0));
  }
  if (has_ref) {
    double g = k_gain_db[is_nonverbal(type) ? 0 : 1];
    float off = (float)(ref_loud + g / 20.0 * log(10.0) - st_events_loudness(x, n));
    for (size_t k = 0; k < (size_t)n * N_MELS; ++k) x[k] += off;
  }
  float floor_ = (float)log(1e-7);
  for (int t = 0; t < gap; ++t)
    for (int k = 0; k < N_MELS; ++k) m[(size_t)t * N_MELS + k] = m[(size_t)(gap + n + t) * N_MELS + k] = floor_;
  *mel = m;
  *T = tot;
  return 0;
}
