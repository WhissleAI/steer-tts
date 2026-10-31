/* Vocos backbone + ISTFTHead (vocos.models.VocosBackbone, vocos.heads.ISTFTHead with padding "center"):
 *   x = LN(embed(mel));  per block: x += gamma * pwconv2(gelu(pwconv1(LN(dwconv7(x)))));  x = LN(x);
 *   [log|mag|, phase] = head.out(x);  S = min(exp(log|mag|), 100) * (cos phase + i sin phase);  wav = istft(S). */
#include "st_vocoder.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "st_dsp.h"

#define N_MELS 100

typedef struct {
  float* dw_t;   /* depthwise weights transposed to [7, dim] (owned) */
  const float *dw_w, *dw_b, *ln_w, *ln_b, *gamma, *b1, *b2;
  const st_mat *w1, *w2;
} cnx;

struct st_vocoder {
  int dim, inter, layers, n_fft, hop;
  const st_mat *embed, *out;
  const float *embed_b, *ln_w, *ln_b, *fln_w, *fln_b, *out_b;
  cnx* blk;
};

static const float* vec(st_weights* w, const char* name, int n) { return st_weights_f32(w, name, n); }

st_vocoder* st_vocoder_load(st_weights* w, int dim, int inter, int layers, int n_fft, int hop) {
  if (dim <= 0 || inter <= 0 || layers <= 0 || hop <= 0 || n_fft < 2 || (n_fft & (n_fft - 1))) {
    st_seterr("vocoder: bad shape (dim %d inter %d layers %d n_fft %d hop %d)", dim, inter, layers, n_fft, hop);
    return NULL;
  }
  st_vocoder* v = (st_vocoder*)st_calloc(sizeof *v);
  if (!v) return NULL;
  v->dim = dim, v->inter = inter, v->layers = layers, v->n_fft = n_fft, v->hop = hop;
  v->blk = (cnx*)st_calloc(sizeof(cnx) * (size_t)layers);
  if (!v->blk) goto fail;
  if (!(v->embed = st_weights_mat(w, "backbone.embed.weight", dim, N_MELS * 7)) ||
      !(v->embed_b = vec(w, "backbone.embed.bias", dim)) || !(v->ln_w = vec(w, "backbone.norm.weight", dim)) ||
      !(v->ln_b = vec(w, "backbone.norm.bias", dim)) || !(v->fln_w = vec(w, "backbone.final_layer_norm.weight", dim)) ||
      !(v->fln_b = vec(w, "backbone.final_layer_norm.bias", dim)) ||
      !(v->out = st_weights_mat(w, "head.out.weight", n_fft + 2, dim)) || !(v->out_b = vec(w, "head.out.bias", n_fft + 2)))
    goto fail;
  for (int i = 0; i < layers; ++i) {
    cnx* b = &v->blk[i];
#define P(s) st_name("backbone.convnext.%d.%s", i, s)
    if (!(b->dw_w = vec(w, P("dwconv.weight"), dim * 7)) || !(b->dw_b = vec(w, P("dwconv.bias"), dim)) ||
        !(b->ln_w = vec(w, P("norm.weight"), dim)) || !(b->ln_b = vec(w, P("norm.bias"), dim)) ||
        !(b->gamma = vec(w, P("gamma"), dim)) || !(b->w1 = st_weights_mat(w, P("pwconv1.weight"), inter, dim)) ||
        !(b->b1 = vec(w, P("pwconv1.bias"), inter)) || !(b->w2 = st_weights_mat(w, P("pwconv2.weight"), dim, inter)) ||
        !(b->b2 = vec(w, P("pwconv2.bias"), dim)))
      goto fail;
#undef P
    if (!(b->dw_t = (float*)st_malloc(sizeof(float) * 7 * (size_t)dim))) goto fail;
    for (int c = 0; c < dim; ++c)
      for (int k = 0; k < 7; ++k) b->dw_t[(size_t)k * dim + c] = b->dw_w[(size_t)c * 7 + k];
  }
  return v;
fail:
  st_vocoder_free(v);
  return NULL;
}

void st_vocoder_free(st_vocoder* v) {
  if (!v) return;
  if (v->blk)
    for (int i = 0; i < v->layers; ++i) st_free(v->blk[i].dw_t);
  st_free(v->blk);
  st_free(v);
}

/* depthwise conv (k 7, pad 3) split over time ranges; w transposed [7, C] so the channel loop is contiguous */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(ST_NO_VEC)
typedef float v8f __attribute__((vector_size(32)));
#define ST_VEC8 1
#endif

typedef struct {
  const float *x, *w, *b;
  float* y;
  int T, C;
} dw_job;

static void dw_task(void* ctx, int task, int n_tasks) {
  const dw_job* j = (const dw_job*)ctx;
  int t0 = (int)((long long)j->T * task / n_tasks), t1 = (int)((long long)j->T * (task + 1) / n_tasks);
  int C = j->C;
  for (int t = t0; t < t1; ++t) {
    float* yr = j->y + (size_t)t * C;
    memcpy(yr, j->b, sizeof(float) * (size_t)C);
    for (int k = 0; k < 7; ++k) {
      int ti = t + k - 3;
      if (ti < 0 || ti >= j->T) continue;
      const float *xr = j->x + (size_t)ti * C, *w = j->w + (size_t)k * C;
      int c = 0;
#ifdef ST_VEC8
      for (; c + 8 <= C; c += 8) {
        v8f a, b, o;
        memcpy(&a, xr + c, sizeof a);
        memcpy(&b, w + c, sizeof b);
        memcpy(&o, yr + c, sizeof o);
        o += a * b;
        memcpy(yr + c, &o, sizeof o);
      }
#endif
      for (; c < C; ++c) yr[c] += xr[c] * w[c];
    }
  }
}

/* layer norm / GELU / head nonlinearity, split over rows */
typedef struct {
  float* x;
  int T, C;
  const float *w, *b;
  int op;   /* 0: LN eps 1e-6, 1: GELU, 2: head -> re / im */
  float *re, *im;
  int nb;
} row_job;

static void row_task(void* ctx, int task, int n_tasks) {
  const row_job* j = (const row_job*)ctx;
  int t0 = (int)((long long)j->T * task / n_tasks), t1 = (int)((long long)j->T * (task + 1) / n_tasks);
  if (t1 <= t0) return;
  float* r = j->x + (size_t)t0 * j->C;
  if (j->op == 0) st_layernorm(r, t1 - t0, j->C, j->C, j->w, j->b, 1e-6f);
  else if (j->op == 1) st_gelu(r, (size_t)(t1 - t0) * j->C);
  else
    for (int t = t0; t < t1; ++t) {
      const float* h = j->x + (size_t)t * j->C;
      float *re = j->re + (size_t)t * j->nb, *im = j->im + (size_t)t * j->nb;
      for (int k = 0; k < j->nb; ++k) {
        float m = expf(h[k]);
        m = m > 1e2f ? 1e2f : m;   /* torch.clip(max=1e2); NaN propagates */
        float ph = h[j->nb + k];
        re[k] = m * cosf(ph);
        im[k] = m * sinf(ph);
      }
    }
}

static void rows(st_pool* p, row_job* j) {
  int tasks = st_pool_size(p) * 2;
  if (tasks > j->T) tasks = j->T;
  st_parallel_for(tasks > 1 ? p : NULL, tasks, row_task, j);
}

int st_vocoder_decode(st_vocoder* v, st_pool* p, const float* mel, int T, float** wav, size_t* n) {
  *wav = NULL;
  *n = 0;
  if (T <= 0) {
    st_seterr("vocoder: empty mel");
    return -1;
  }
  int D = v->dim, I = v->inter, nb = v->n_fft / 2 + 1, rc = -1;
  size_t ho = (size_t)T * (size_t)(v->n_fft + 2);
  size_t scratch = (size_t)T * I > ho ? (size_t)T * I : ho;
  float* x = (float*)st_malloc(sizeof(float) * (size_t)T * D);
  float* y = (float*)st_malloc(sizeof(float) * (size_t)T * D);
  float* s = (float*)st_malloc(sizeof(float) * scratch);
  float* ri = (float*)st_malloc(sizeof(float) * (size_t)T * nb * 2);
  float* out = NULL;
  if (!x || !y || !s || !ri) goto done;
  if (st_conv1d(p, mel, T, N_MELS, N_MELS, v->embed, 7, 1, 1, 3, 3, v->embed_b, x, D) < 0) goto done;
  row_job ln = {x, T, D, v->ln_w, v->ln_b, 0, NULL, NULL, 0};
  rows(p, &ln);
  for (int i = 0; i < v->layers; ++i) {
    const cnx* b = &v->blk[i];
    dw_job dj = {x, b->dw_t, b->dw_b, y, T, D};
    int tasks = st_pool_size(p) * 2;
    if (tasks > T) tasks = T;
    st_parallel_for(tasks > 1 ? p : NULL, tasks, dw_task, &dj);
    row_job l2 = {y, T, D, b->ln_w, b->ln_b, 0, NULL, NULL, 0};
    rows(p, &l2);
    st_linear(p, y, T, D, b->w1, b->b1, s, I, 0);
    row_job g = {s, T, I, NULL, NULL, 1, NULL, NULL, 0};
    rows(p, &g);
    st_linear(p, s, T, I, b->w2, b->b2, y, D, 0);
    for (int t = 0; t < T; ++t) {
      float* xr = x + (size_t)t * D;
      const float* yr = y + (size_t)t * D;
      for (int c = 0; c < D; ++c) xr[c] += b->gamma[c] * yr[c];
    }
  }
  row_job fl = {x, T, D, v->fln_w, v->fln_b, 0, NULL, NULL, 0};
  rows(p, &fl);
  st_linear(p, x, T, D, v->out, v->out_b, s, v->n_fft + 2, 0);
  row_job hd = {s, T, v->n_fft + 2, NULL, NULL, 2, ri, ri + (size_t)T * nb, nb};
  rows(p, &hd);
  size_t len = st_istft_len(T, v->hop);
  out = (float*)st_malloc(sizeof(float) * (len ? len : 1));
  if (!out) goto done;
  if (st_istft(p, ri, ri + (size_t)T * nb, nb, T, v->n_fft, v->hop, out)) goto done;
  *wav = out;
  *n = len;
  out = NULL;
  rc = 0;
done:
  st_free(x);
  st_free(y);
  st_free(s);
  st_free(ri);
  st_free(out);
  return rc;
}
