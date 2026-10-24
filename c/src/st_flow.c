/* Flow-matching mel refiner / prompted flow (steertts/flow.py), inference only.
 *
 * Flow.sample(mu, g, steps, temp, seed[, ctx, cm, cfg]) with the midpoint method:
 *   mu_n = (mu - mean) / std  (prompted flow: minus its per-band mean over the target frames; context frames
 *          replaced by the normalised prompt);  x = temp * randn(seed, [1, 100, N]);  ts = linspace(0, 1, steps + 1)
 *   per step: v = net(x, t0); v = net(x + dt/2 v, t0 + dt/2); x += dt v;   out = x * std + mean
 * net (FlowNet): c = temb(t) + gemb(g) [+ ctx_out(masked mean of ctx_enc(ctx))];
 *   h = inp(cat[x, mu_n (, ctx, cm)]);  12 x ConvNeXt block (dw dilated conv -> LN -> AdaLN-zero (scale, shift) ->
 *   pw1 -> GELU -> pw2 -> gate), AttnBlock after the blocks in attn_at;  v = out(LN(h)).
 *
 * Evaluation layout (all time-major [N, C], N = P + T):
 *   - everything that does not depend on x is computed once per call: gemb(g), the context term, and the inp conv
 *     over the constant input channels (mu_n, ctx, cm) with the bias; per evaluation only the x columns of the inp
 *     weight are applied (an owned copy of those columns, still int8 when the weights are);
 *   - c depends on t only, so temb and every block's modulation (mod(c) = scale, shift, gate) are computed for all
 *     2 * steps evaluations up front as small GEMMs;
 *   - depthwise conv + LayerNorm + modulation is one fused row pass; GELU is a vectorised erf (the same
 *     Abramowitz-Stegun 7.1.26 form PyTorch's AVX2 Vectorized<float>::erf uses); the GEMMs (st_linear) dominate. */
#include "st_flow.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "st_rng.h"

#ifdef __clang__
#pragma STDC FP_CONTRACT OFF
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define ST_X86 1
#include <immintrin.h>
#endif

enum { MAXL = 64, NMEL = 100, KW = 7, HEADS = 6 };

typedef struct {
  float* wT;   /* [KW, d] depthwise weight, tap-major (owned) */
  const float* dwb;
  int dil;
  const st_mat *mod, *pw1, *pw2;
  const float *modb, *pw1b, *pw2b;
} fblock;

typedef struct {
  const st_mat *mod, *in, *out;
  const float *modb, *inb, *outb;
} fattn;

struct st_flow {
  int d, n_layers, g_dim, prompt, mu_cmn, cin;
  float mean, std;
  st_mat inp_x, inp_c;   /* inp weight split: x columns [0, 300), constant columns [300, 3 * cin) (owned) */
  void *inp_x_buf, *inp_c_buf;
  const float* inp_b;
  const st_mat *t1, *t2, *g1, *g2, *ce1, *ce2, *co, *outw;
  const float *t1b, *t2b, *g1b, *g2b, *ce1b, *ce2b, *cob, *outb;
  float* freq;   /* [d/2] TimeEmb frequencies */
  fblock blk[MAXL];
  fattn att[MAXL];
  int has_att[MAXL];
};

#ifdef ST_X86
static int use_avx2(void) { return st_cpu_avx2(); }   /* cached atomically there */
#endif

/* ------------------------------------------------------------------ loading */
static int sub_cols(const st_mat* m, int k0, int k1, st_mat* out, void** buf) {
  int K = k1 - k0;
  *out = *m;
  out->K = K;
  if (m->type == ST_W_F32) {
    float* b = (float*)st_malloc(sizeof(float) * (size_t)m->O * K);
    if (!b) return -1;
    for (int o = 0; o < m->O; ++o) memcpy(b + (size_t)o * K, m->w + (size_t)o * m->K + k0, sizeof(float) * K);
    out->w = b;
    *buf = b;
  } else if (m->type == ST_W_I8) {   /* per-row scale: a column subset keeps the row's scale */
    int8_t* b = (int8_t*)st_malloc((size_t)m->O * K);
    if (!b) return -1;
    for (int o = 0; o < m->O; ++o) memcpy(b + (size_t)o * K, m->q8 + (size_t)o * m->K + k0, (size_t)K);
    out->q8 = b;
    *buf = b;
  } else {
    st_seterr("flow: int4 weights are not supported for the inp conv");
    return -1;
  }
  return 0;
}

static int load_lin(st_weights* w, const char* base, int O, int K, const st_mat** m, const float** b) {
  *m = st_weights_mat(w, st_name("%s.weight", base), O, K);
  if (!*m) return -1;
  *b = st_weights_f32(w, st_name("%s.bias", base), O);
  return *b ? 0 : -1;
}

static int cfg_int_list(const st_json* a, int* out, int max) {
  if (!a || a->type != ST_JARR) return -1;
  int n = a->n < max ? a->n : max;
  for (int i = 0; i < n; ++i) out[i] = (int)st_json_num(st_json_at(a, i), 0);
  return n;
}

st_flow* st_flow_load(st_weights* w, const st_json* cfg) {
  st_flow* f = (st_flow*)st_calloc(sizeof *f);
  if (!f) return NULL;
  f->d = (int)st_json_num(st_json_get(cfg, "d"), 384);
  f->n_layers = (int)st_json_num(st_json_get(cfg, "n_layers"), 12);
  f->g_dim = (int)st_json_num(st_json_get(cfg, "g_dim"), 128);
  f->prompt = (int)st_json_num(st_json_get(cfg, "prompt"), 0);
  f->mu_cmn = (int)st_json_num(st_json_get(cfg, "mu_cmn"), 0);
  int n_mel = (int)st_json_num(st_json_get(cfg, "n_mel"), NMEL), d = f->d, L = f->n_layers;
  int tmp[MAXL];
  if (n_mel != NMEL || L < 1 || L > MAXL || d % HEADS || d % 2) {
    st_seterr("flow: unsupported cfg (n_mel %d, d %d, n_layers %d)", n_mel, d, L);
    goto fail;
  }
  if (cfg_int_list(st_json_get(cfg, "rope_at"), tmp, MAXL) > 0 || st_json_num(st_json_get(cfg, "n_ptok"), 0) > 0 ||
      st_json_num(st_json_get(cfg, "skips"), 0) != 0) {
    st_seterr("flow: rope_at / n_ptok / skips (optional flow blocks) are not supported");
    goto fail;
  }
  int dils[MAXL], nd = cfg_int_list(st_json_get(cfg, "dils"), dils, MAXL);
  if (nd <= 0) {
    nd = 12;
    for (int i = 0; i < 12; ++i) dils[i] = 1 << (i % 3);
  }
  const float* mean = st_weights_f32(w, "mean", 1);
  const float* sd = st_weights_f32(w, "std", 1);
  if (!mean || !sd) goto fail;
  f->mean = *mean, f->std = *sd;

  f->cin = f->prompt ? 3 * NMEL + 1 : 2 * NMEL;
  const st_mat* inp = st_weights_mat(w, "net.inp.weight", d, 3 * f->cin);
  f->inp_b = st_weights_f32(w, "net.inp.bias", d);
  if (!inp || !f->inp_b) goto fail;
  if (sub_cols(inp, 0, 3 * NMEL, &f->inp_x, &f->inp_x_buf) || sub_cols(inp, 3 * NMEL, 3 * f->cin, &f->inp_c, &f->inp_c_buf))
    goto fail;
  if (load_lin(w, "net.temb.mlp.0", 4 * d, d, &f->t1, &f->t1b) || load_lin(w, "net.temb.mlp.2", d, 4 * d, &f->t2, &f->t2b) ||
      load_lin(w, "net.gemb.0", d, f->g_dim, &f->g1, &f->g1b) || load_lin(w, "net.gemb.2", d, d, &f->g2, &f->g2b))
    goto fail;
  f->outw = st_weights_mat(w, "net.out.weight", NMEL, d);
  f->outb = st_weights_f32(w, "net.out.bias", NMEL);
  if (!f->outw || !f->outb) goto fail;
  for (int i = 0; i < L; ++i) {
    fblock* b = &f->blk[i];
    b->dil = dils[i % nd];
    const float* dw = st_weights_f32(w, st_name("net.blocks.%d.dw.weight", i), (int64_t)d * KW);
    b->dwb = st_weights_f32(w, st_name("net.blocks.%d.dw.bias", i), d);
    if (!dw || !b->dwb) goto fail;
    b->wT = (float*)st_malloc(sizeof(float) * KW * d);
    if (!b->wT) goto fail;
    for (int c = 0; c < d; ++c)
      for (int k = 0; k < KW; ++k) b->wT[k * d + c] = dw[c * KW + k];
    char nm[64];
    snprintf(nm, sizeof nm, "net.blocks.%d.mod", i);
    if (load_lin(w, nm, 3 * d, d, &b->mod, &b->modb)) goto fail;
    snprintf(nm, sizeof nm, "net.blocks.%d.pw1", i);
    if (load_lin(w, nm, 3 * d, d, &b->pw1, &b->pw1b)) goto fail;
    snprintf(nm, sizeof nm, "net.blocks.%d.pw2", i);
    if (load_lin(w, nm, d, 3 * d, &b->pw2, &b->pw2b)) goto fail;
  }
  if (f->prompt) {
    int at[MAXL], na = cfg_int_list(st_json_get(cfg, "attn_at"), at, MAXL);
    if (na < 0) at[0] = 3, at[1] = 7, at[2] = 11, na = 3;
    for (int j = 0; j < na; ++j) {
      int i = at[j];
      if (i < 0 || i >= L) continue;
      fattn* a = &f->att[i];
      char nm[64];
      snprintf(nm, sizeof nm, "net.attn.%d.mod", i);
      if (load_lin(w, nm, 3 * d, d, &a->mod, &a->modb)) goto fail;
      a->in = st_weights_mat(w, st_name("net.attn.%d.att.in_proj_weight", i), 3 * d, d);
      a->inb = st_weights_f32(w, st_name("net.attn.%d.att.in_proj_bias", i), 3 * d);
      if (!a->in || !a->inb) goto fail;
      snprintf(nm, sizeof nm, "net.attn.%d.att.out_proj", i);
      if (load_lin(w, nm, d, d, &a->out, &a->outb)) goto fail;
      f->has_att[i] = 1;
    }
    if (load_lin(w, "net.ctx_enc.0", d, 3 * NMEL, &f->ce1, &f->ce1b) || load_lin(w, "net.ctx_enc.2", d, 3 * d, &f->ce2, &f->ce2b) ||
        load_lin(w, "net.ctx_out", d, d, &f->co, &f->cob))
      goto fail;
  }
  f->freq = (float*)st_malloc(sizeof(float) * (size_t)(d / 2));
  if (!f->freq) goto fail;
  /* torch: exp(-log(10000) * arange(half) / half) in float32 */
  for (int i = 0; i < d / 2; ++i) {
    float a = ((float)-9.210340371976184 * (float)i) / (float)(d / 2);
    f->freq[i] = (float)exp((double)a);
  }
  return f;
fail:
  st_flow_free(f);
  return NULL;
}

void st_flow_free(st_flow* f) {
  if (!f) return;
  for (int i = 0; i < MAXL; ++i) st_free(f->blk[i].wT);
  st_free(f->inp_x_buf);
  st_free(f->inp_c_buf);
  st_free(f->freq);
  st_free(f);
}

int st_flow_is_prompted(const st_flow* f) { return f && f->prompt; }

/* ------------------------------------------------------------------ row kernels */
typedef struct {
  const float* h;   /* [N, d] */
  float* out;       /* [N, d] */
  int N, d, dil;
  const float* wT;  /* NULL: no depthwise conv (plain LN) */
  const float* b;
  const float* mod; /* [3d] scale, shift, gate; NULL: no modulation */
} dwln_job;

static void dwln_rows_c(const dwln_job* j, int t0, int t1) {
  int d = j->d, pad = j->dil * (KW / 2);
  for (int t = t0; t < t1; ++t) {
    float* o = j->out + (size_t)t * d;
    if (j->wT) {
      for (int c = 0; c < d; ++c) o[c] = j->b[c];
      for (int k = 0; k < KW; ++k) {
        int ti = t + k * j->dil - pad;
        if (ti < 0 || ti >= j->N) continue;
        const float *x = j->h + (size_t)ti * d, *w = j->wT + (size_t)k * d;
        for (int c = 0; c < d; ++c) o[c] += x[c] * w[c];
      }
    } else {
      memcpy(o, j->h + (size_t)t * d, sizeof(float) * d);
    }
    float s[8] = {0}, q[8] = {0};
    for (int c = 0; c < d; ++c) s[c & 7] += o[c];
    float mean = ((s[0] + s[1]) + (s[2] + s[3]) + ((s[4] + s[5]) + (s[6] + s[7]))) / (float)d;
    for (int c = 0; c < d; ++c) {
      float v = o[c] - mean;
      q[c & 7] += v * v;
    }
    float var = ((q[0] + q[1]) + (q[2] + q[3]) + ((q[4] + q[5]) + (q[6] + q[7]))) / (float)d;
    float inv = 1.f / sqrtf(var + 1e-5f);
    if (j->mod) {
      const float *sc = j->mod, *sh = j->mod + d;
      for (int c = 0; c < d; ++c) o[c] = (o[c] - mean) * inv * (1.f + sc[c]) + sh[c];
    } else {
      for (int c = 0; c < d; ++c) o[c] = (o[c] - mean) * inv;
    }
  }
}

#ifdef ST_X86
__attribute__((target("avx2,fma"))) static float hsum256(__m256 v) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_add_ps(a, _mm_movehl_ps(a, a));
  a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
  return _mm_cvtss_f32(a);
}

__attribute__((target("avx2,fma"))) static void dwln_rows_avx2(const dwln_job* j, int t0, int t1) {
  int d = j->d, pad = j->dil * (KW / 2);
  for (int t = t0; t < t1; ++t) {
    float* o = j->out + (size_t)t * d;
    __m256 s = _mm256_setzero_ps();
    if (j->wT) {
      const float* xr[KW];
      const float* wr[KW];
      int nk = 0;
      for (int k = 0; k < KW; ++k) {
        int ti = t + k * j->dil - pad;
        if (ti < 0 || ti >= j->N) continue;
        xr[nk] = j->h + (size_t)ti * d, wr[nk] = j->wT + (size_t)k * d, ++nk;
      }
      for (int c = 0; c < d; c += 8) {
        __m256 acc = _mm256_loadu_ps(j->b + c);
        for (int k = 0; k < nk; ++k) acc = _mm256_fmadd_ps(_mm256_loadu_ps(xr[k] + c), _mm256_loadu_ps(wr[k] + c), acc);
        _mm256_storeu_ps(o + c, acc);
        s = _mm256_add_ps(s, acc);
      }
    } else {
      const float* x = j->h + (size_t)t * d;
      for (int c = 0; c < d; c += 8) {
        __m256 v = _mm256_loadu_ps(x + c);
        _mm256_storeu_ps(o + c, v);
        s = _mm256_add_ps(s, v);
      }
    }
    float mean = hsum256(s) / (float)d;
    __m256 mv = _mm256_set1_ps(mean), q = _mm256_setzero_ps();
    for (int c = 0; c < d; c += 8) {
      __m256 v = _mm256_sub_ps(_mm256_loadu_ps(o + c), mv);
      q = _mm256_fmadd_ps(v, v, q);
    }
    float var = hsum256(q) / (float)d;
    __m256 inv = _mm256_set1_ps(1.f / sqrtf(var + 1e-5f)), one = _mm256_set1_ps(1.f);
    if (j->mod) {
      const float *sc = j->mod, *sh = j->mod + d;
      for (int c = 0; c < d; c += 8) {
        __m256 v = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(o + c), mv), inv);
        v = _mm256_fmadd_ps(v, _mm256_add_ps(one, _mm256_loadu_ps(sc + c)), _mm256_loadu_ps(sh + c));
        _mm256_storeu_ps(o + c, v);
      }
    } else {
      for (int c = 0; c < d; c += 8) _mm256_storeu_ps(o + c, _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(o + c), mv), inv));
    }
  }
}
#endif

static void dwln_task(void* ctx, int task, int n_tasks) {
  const dwln_job* j = (const dwln_job*)ctx;
  int t0 = (int)((long long)j->N * task / n_tasks), t1 = (int)((long long)j->N * (task + 1) / n_tasks);
#ifdef ST_X86
  if (use_avx2() && j->d % 8 == 0) {
    dwln_rows_avx2(j, t0, t1);
    return;
  }
#endif
  dwln_rows_c(j, t0, t1);
}

/* GELU(x) = x/2 (1 + erf(x/sqrt2)), erf as Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7), PyTorch's AVX2 form */
static float gelu1(float x) {
  float z = x * 0.70710678118654752f, a = fabsf(z);
  float t = 1.f / (0.3275911f * a + 1.f);
  float r = (((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f;
  float e = 1.f - r * t * expf(-z * z);
  return x * 0.5f * (1.f + (z < 0 ? -e : e));
}

#ifdef ST_X86
/* Cephes-style expf for x <= 0 (exp(-z^2)), relative error ~1 ulp */
__attribute__((target("avx2,fma"))) static __m256 exp_neg256(__m256 x) {
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

__attribute__((target("avx2,fma"))) static void gelu_avx2(float* x, size_t n) {
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
    __m256 e = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_xor_ps(sgn, ex), t), r, one);
    e = _mm256_xor_ps(s, e);
    _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(v, half), _mm256_add_ps(one, e)));
  }
  for (; i < n; ++i) x[i] = gelu1(x[i]);
}
#endif

typedef struct {
  float* x;
  size_t n;
} gelu_job;

static void gelu_task(void* ctx, int task, int n_tasks) {
  const gelu_job* g = (const gelu_job*)ctx;
  size_t i0 = g->n * (size_t)task / (size_t)n_tasks, i1 = g->n * (size_t)(task + 1) / (size_t)n_tasks;
  i0 &= ~(size_t)7;
  if (task + 1 < n_tasks) i1 &= ~(size_t)7;
#ifdef ST_X86
  if (use_avx2()) {
    gelu_avx2(g->x + i0, i1 - i0);
    return;
  }
#endif
  for (size_t i = i0; i < i1; ++i) g->x[i] = gelu1(g->x[i]);
}

/* h[t, c] += gate[c] * v[t, c] */
typedef struct {
  float* h;
  const float *v, *gate;
  int N, d;
} gate_job;

static void gate_task(void* ctx, int task, int n_tasks) {
  const gate_job* g = (const gate_job*)ctx;
  int t0 = (int)((long long)g->N * task / n_tasks), t1 = (int)((long long)g->N * (task + 1) / n_tasks);
  for (int t = t0; t < t1; ++t) {
    float* h = g->h + (size_t)t * g->d;
    const float* v = g->v + (size_t)t * g->d;
    for (int c = 0; c < g->d; ++c) h[c] += v[c] * g->gate[c];
  }
}

static int n_row_tasks(st_pool* p, int N) {
  int n = st_pool_size(p) * 2;
  if (n > N) n = N;
  return n < 1 ? 1 : n;
}

/* ------------------------------------------------------------------ helpers */
/* torch.linspace(0, 1, n) on CPU, float32: float step, the products evaluated in double (as the 2.x kernel ends up
 * doing), first half counted from 0, second half from 1 (checked against torch for n = 2 .. 300) */
static void linspace01(int n, float* ts) {
  float step = 1.0f / (float)(n - 1);
  int half = n / 2;
  for (int i = 0; i < n; ++i)
    ts[i] = i < half ? (float)((double)step * i) : (float)(1.0 - (double)step * (n - 1 - i));
}

/* im2col for a k=3, pad=1 conv over C channels: col[t, c*3 + j] = x[t + j - 1, c] (zero outside) */
static void im2col3(const float* x, int N, int C, float* col) {
  for (int t = 0; t < N; ++t) {
    float* r = col + (size_t)t * 3 * C;
    for (int j = 0; j < 3; ++j) {
      int ti = t + j - 1;
      if (ti < 0 || ti >= N)
        for (int c = 0; c < C; ++c) r[c * 3 + j] = 0.f;
      else {
        const float* xr = x + (size_t)ti * C;
        for (int c = 0; c < C; ++c) r[c * 3 + j] = xr[c];
      }
    }
  }
}

typedef struct {
  float *h, *hn, *u, *v, *col, *qkv, *att;
} fbufs;

/* one network evaluation: x [N,100] -> vel [N,100]. hc: constant inp part [N,d] (bias included);
 * mods: per layer [3d] block modulation and [3d] attention modulation for this evaluation (stride given) */
static int net_eval(const st_flow* f, st_pool* p, const float* x, int N, const float* hc, const float* bmod,
                    const float* amod, size_t mstride, fbufs* B, float* vel) {
  int d = f->d, nt = n_row_tasks(p, N);
  im2col3(x, N, NMEL, B->col);
  memcpy(B->h, hc, sizeof(float) * (size_t)N * d);
  st_linear(p, B->col, N, 3 * NMEL, &f->inp_x, NULL, B->h, d, 1);
  for (int l = 0; l < f->n_layers; ++l) {
    const fblock* b = &f->blk[l];
    const float* m = bmod + (size_t)l * mstride;
    dwln_job dj = {B->h, B->hn, N, d, b->dil, b->wT, b->dwb, m};
    st_parallel_for(p, nt, dwln_task, &dj);
    st_linear(p, B->hn, N, d, b->pw1, b->pw1b, B->u, 3 * d, 0);
    gelu_job gj = {B->u, (size_t)N * 3 * d};
    st_parallel_for(p, nt, gelu_task, &gj);
    st_linear(p, B->u, N, 3 * d, b->pw2, b->pw2b, B->v, d, 0);
    gate_job gt = {B->h, B->v, m + 2 * d, N, d};
    st_parallel_for(p, nt, gate_task, &gt);
    if (f->has_att[l]) {
      const fattn* a = &f->att[l];
      const float* am = amod + (size_t)l * mstride;
      dwln_job lj = {B->h, B->hn, N, d, 0, NULL, NULL, am};
      st_parallel_for(p, nt, dwln_task, &lj);
      st_linear(p, B->hn, N, d, a->in, a->inb, B->qkv, 3 * d, 0);
      if (st_attention(p, B->qkv, 3 * d, B->qkv + d, 3 * d, B->qkv + 2 * d, 3 * d, B->att, d, N, N, HEADS, d / HEADS,
                       NULL, NULL, 0.f))
        return -1;
      st_linear(p, B->att, N, d, a->out, a->outb, B->v, d, 0);
      gate_job ga = {B->h, B->v, am + 2 * d, N, d};
      st_parallel_for(p, nt, gate_task, &ga);
    }
  }
  dwln_job fj = {B->h, B->hn, N, d, 0, NULL, NULL, NULL};
  st_parallel_for(p, nt, dwln_task, &fj);
  st_linear(p, B->hn, N, d, f->outw, f->outb, vel, NMEL, 0);
  return 0;
}

/* c rows for every evaluation time: temb(t) + base, then each layer's modulation -> mods[l][e][3d] */
static int cond_tables(const st_flow* f, st_pool* p, const float* tv, int ne, const float* base, float* bmod,
                       float* amod) {
  int d = f->d, h2 = d / 2;
  float* emb = (float*)st_malloc(sizeof(float) * (size_t)ne * d);
  float* hid = (float*)st_malloc(sizeof(float) * (size_t)ne * 4 * d);
  float* c = (float*)st_malloc(sizeof(float) * (size_t)ne * d);
  if (!emb || !hid || !c) {
    st_free(emb), st_free(hid), st_free(c);
    return -1;
  }
  for (int e = 0; e < ne; ++e) {
    float tt = 1000.0f * tv[e];
    for (int i = 0; i < h2; ++i) {
      float a = tt * f->freq[i];
      emb[(size_t)e * d + i] = sinf(a);
      emb[(size_t)e * d + h2 + i] = cosf(a);
    }
  }
  st_linear(p, emb, ne, d, f->t1, f->t1b, hid, 4 * d, 0);
  st_silu(hid, (size_t)ne * 4 * d);
  st_linear(p, hid, ne, 4 * d, f->t2, f->t2b, c, d, 0);
  for (int e = 0; e < ne; ++e)
    for (int i = 0; i < d; ++i) c[(size_t)e * d + i] += base[i];
  size_t ms = (size_t)ne * 3 * d;
  for (int l = 0; l < f->n_layers; ++l) {
    st_linear(p, c, ne, d, f->blk[l].mod, f->blk[l].modb, bmod + (size_t)l * ms, 3 * d, 0);
    if (f->has_att[l]) st_linear(p, c, ne, d, f->att[l].mod, f->att[l].modb, amod + (size_t)l * ms, 3 * d, 0);
  }
  st_free(emb), st_free(hid), st_free(c);
  return 0;
}

/* constant inp part [N,d] = W_c * im2col(rest) + bias, rest = [mu_n (, ctx, cm)] */
static int const_inp(const st_flow* f, st_pool* p, const float* mun, const float* ctx, const float* cm, int N, float* hc) {
  int C = f->cin - NMEL;
  float* rest = (float*)st_malloc(sizeof(float) * (size_t)N * C);
  float* col = (float*)st_malloc(sizeof(float) * (size_t)N * 3 * C);
  if (!rest || !col) {
    st_free(rest), st_free(col);
    return -1;
  }
  for (int t = 0; t < N; ++t) {
    float* r = rest + (size_t)t * C;
    memcpy(r, mun + (size_t)t * NMEL, sizeof(float) * NMEL);
    if (f->prompt) {
      if (ctx) memcpy(r + NMEL, ctx + (size_t)t * NMEL, sizeof(float) * NMEL);
      else memset(r + NMEL, 0, sizeof(float) * NMEL);
      r[2 * NMEL] = cm ? cm[t] : 0.f;
    }
  }
  im2col3(rest, N, C, col);
  st_linear(p, col, N, 3 * C, &f->inp_c, f->inp_b, hc, f->d, 0);
  st_free(rest), st_free(col);
  return 0;
}

/* ctx_out(sum_t ctx_enc(ctx)[t] cm[t] / max(sum cm, 1)) for the first P frames (ctx is zero elsewhere) */
static int ctx_term(const st_flow* f, st_pool* p, const float* ctx, int N, int P, float* out) {
  int d = f->d;
  float* e1 = (float*)st_malloc(sizeof(float) * (size_t)N * d);
  float* e2 = (float*)st_malloc(sizeof(float) * (size_t)N * d);
  float* mean = (float*)st_calloc(sizeof(float) * (size_t)d);
  int rc = -1;
  if (!e1 || !e2 || !mean) goto done;
  if (st_conv1d(p, ctx, N, NMEL, NMEL, f->ce1, 3, 1, 1, 1, 1, f->ce1b, e1, d) < 0) goto done;
  st_silu(e1, (size_t)N * d);
  if (st_conv1d(p, e1, N, d, d, f->ce2, 3, 1, 1, 1, 1, f->ce2b, e2, d) < 0) goto done;
  for (int c = 0; c < d; ++c) {
    double s = 0;
    for (int t = 0; t < P; ++t) s += e2[(size_t)t * d + c];
    mean[c] = (float)s / (float)(P > 1 ? P : 1);
  }
  st_linear(p, mean, 1, d, f->co, f->cob, out, d, 0);
  rc = 0;
done:
  st_free(e1), st_free(e2), st_free(mean);
  return rc;
}

/* ------------------------------------------------------------------ sampling */
int st_flow_sample(st_flow* f, st_pool* p, const float* mu, int T, const float* g, int steps, float temp, uint64_t seed,
                   const float* prompt, int P, float cfg, float* out) {
  if (!f || !mu || !g || !out || T <= 0 || steps < 1) {
    st_seterr("flow: bad arguments (T %d, steps %d)", T, steps);
    return -1;
  }
  if (!prompt || P <= 0) prompt = NULL, P = 0;
  if (prompt && !f->prompt) {
    st_seterr("flow: a prompt was given to the (unprompted) refiner");
    return -1;
  }
  int d = f->d, N = P + T, ne = 2 * steps;
  int guided = f->prompt && prompt && cfg != 0.f;
  size_t nd = (size_t)N * d, nm = (size_t)N * NMEL, ms = (size_t)ne * 3 * d;
  size_t nb = (size_t)f->n_layers * ms;
  int rc = -1;
  float *mun = NULL, *ctx = NULL, *cm = NULL, *x = NULL, *xm = NULL, *v = NULL, *vu = NULL, *hc = NULL, *hcu = NULL;
  float *bmod = NULL, *amod = NULL, *bmodu = NULL, *amodu = NULL, *noise = NULL, *ts = NULL, *tv = NULL;
  float *base = NULL, *baseu = NULL, *tmp = NULL;
  fbufs B = {0};
  mun = (float*)st_malloc(sizeof(float) * nm);
  x = (float*)st_malloc(sizeof(float) * nm);
  xm = (float*)st_malloc(sizeof(float) * nm);
  v = (float*)st_malloc(sizeof(float) * nm);
  hc = (float*)st_malloc(sizeof(float) * nd);
  bmod = (float*)st_malloc(sizeof(float) * nb);
  amod = (float*)st_malloc(sizeof(float) * nb);
  noise = (float*)st_malloc(sizeof(float) * nm);
  ts = (float*)st_malloc(sizeof(float) * (size_t)(steps + 1));
  tv = (float*)st_malloc(sizeof(float) * (size_t)ne);
  base = (float*)st_malloc(sizeof(float) * d);
  tmp = (float*)st_malloc(sizeof(float) * d);
  B.h = (float*)st_malloc(sizeof(float) * nd);
  B.hn = (float*)st_malloc(sizeof(float) * nd);
  B.v = (float*)st_malloc(sizeof(float) * nd);
  B.u = (float*)st_malloc(sizeof(float) * nd * 3);
  B.col = (float*)st_malloc(sizeof(float) * nm * 3);
  if (!mun || !x || !xm || !v || !hc || !bmod || !amod || !noise || !ts || !tv || !base || !tmp || !B.h || !B.hn ||
      !B.v || !B.u || !B.col)
    goto done;
  if (f->prompt) {
    B.qkv = (float*)st_malloc(sizeof(float) * nd * 3);
    B.att = (float*)st_malloc(sizeof(float) * nd);
    if (!B.qkv || !B.att) goto done;
  }
  if (guided) {
    vu = (float*)st_malloc(sizeof(float) * nm);
    hcu = (float*)st_malloc(sizeof(float) * nd);
    bmodu = (float*)st_malloc(sizeof(float) * nb);
    amodu = (float*)st_malloc(sizeof(float) * nb);
    baseu = (float*)st_malloc(sizeof(float) * d);
    if (!vu || !hcu || !bmodu || !amodu || !baseu) goto done;
  }

  /* normalised mu (prompt frames first), per-band mean removal over the target frames, context */
  for (int t = 0; t < N; ++t) {
    const float* src = t < P ? prompt + (size_t)t * NMEL : mu + (size_t)(t - P) * NMEL;
    for (int b = 0; b < NMEL; ++b) mun[(size_t)t * NMEL + b] = (src[b] - f->mean) / f->std;
  }
  if (f->mu_cmn) {
    for (int b = 0; b < NMEL; ++b) {
      double s = 0;
      for (int t = P; t < N; ++t) s += mun[(size_t)t * NMEL + b];
      float mb = (float)s / (float)(N - P);
      for (int t = 0; t < N; ++t) mun[(size_t)t * NMEL + b] -= mb;
    }
  }
  if (prompt) {
    ctx = (float*)st_calloc(sizeof(float) * nm);
    cm = (float*)st_calloc(sizeof(float) * (size_t)N);
    if (!ctx || !cm) goto done;
    for (int t = 0; t < P; ++t) {
      cm[t] = 1.f;
      for (int b = 0; b < NMEL; ++b) {
        float c = (prompt[(size_t)t * NMEL + b] - f->mean) / f->std;
        ctx[(size_t)t * NMEL + b] = c;
        mun[(size_t)t * NMEL + b] = c;
      }
    }
  }

  /* conditioning: gemb(g) (+ context term); the prompt-free branch of the prompted net has ctx_out(0) = bias */
  st_linear(p, g, 1, f->g_dim, f->g1, f->g1b, tmp, d, 0);
  st_silu(tmp, (size_t)d);
  st_linear(p, tmp, 1, d, f->g2, f->g2b, base, d, 0);
  if (f->prompt) {
    if (guided) for (int i = 0; i < d; ++i) baseu[i] = base[i] + f->cob[i];
    if (prompt) {
      if (ctx_term(f, p, ctx, N, P, tmp)) goto done;
      for (int i = 0; i < d; ++i) base[i] += tmp[i];
    } else {
      for (int i = 0; i < d; ++i) base[i] += f->cob[i];
    }
  }

  /* evaluation times: t0 and t0 + dt/2 per step */
  linspace01(steps + 1, ts);
  for (int i = 0; i < steps; ++i) {
    float dt = ts[i + 1] - ts[i];
    tv[2 * i] = ts[i];
    tv[2 * i + 1] = ts[i] + 0.5f * dt;
  }
  if (cond_tables(f, p, tv, ne, base, bmod, amod)) goto done;
  if (const_inp(f, p, mun, ctx, cm, N, hc)) goto done;
  if (guided && (cond_tables(f, p, tv, ne, baseu, bmodu, amodu) || const_inp(f, p, mun, NULL, NULL, N, hcu))) goto done;

  /* x0 = temp * randn([1, 100, N]) (channel-major in torch) */
  st_randn_torch(seed, noise, nm);
  for (int t = 0; t < N; ++t)
    for (int b = 0; b < NMEL; ++b) x[(size_t)t * NMEL + b] = temp * noise[(size_t)b * N + t];

  size_t mstr = ms;
  for (int i = 0; i < steps; ++i) {
    float dt = ts[i + 1] - ts[i], hdt = 0.5f * dt;
    for (int s = 0; s < 2; ++s) {
      int e = 2 * i + s;
      const float* xin = s == 0 ? x : xm;
      size_t off = (size_t)e * 3 * d;
      if (net_eval(f, p, xin, N, hc, bmod + off, amod + off, mstr, &B, v)) goto done;
      if (guided) {
        if (net_eval(f, p, xin, N, hcu, bmodu + off, amodu + off, mstr, &B, vu)) goto done;
        for (size_t k = 0; k < nm; ++k) v[k] = v[k] + cfg * (v[k] - vu[k]);
      }
      if (s == 0)
        for (size_t k = 0; k < nm; ++k) xm[k] = x[k] + hdt * v[k];
      else
        for (size_t k = 0; k < nm; ++k) x[k] = x[k] + dt * v[k];
    }
  }
  for (int t = 0; t < T; ++t)
    for (int b = 0; b < NMEL; ++b) out[(size_t)t * NMEL + b] = x[(size_t)(P + t) * NMEL + b] * f->std + f->mean;
  rc = 0;
done:
  st_free(mun), st_free(ctx), st_free(cm), st_free(x), st_free(xm), st_free(v), st_free(vu), st_free(hc), st_free(hcu);
  st_free(bmod), st_free(amod), st_free(bmodu), st_free(amodu), st_free(noise), st_free(ts), st_free(tv);
  st_free(base), st_free(baseu), st_free(tmp);
  st_free(B.h), st_free(B.hn), st_free(B.u), st_free(B.v), st_free(B.col), st_free(B.qkv), st_free(B.att);
  return rc;
}
