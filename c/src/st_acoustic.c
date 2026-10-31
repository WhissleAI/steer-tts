/* SteerTTS acoustic model, inference path of steertts/model.py (SteerTTSModel.infer), batch 1, no padding.
 *
 * Weight storage is whatever st_weights gives (fp32, int8 rows, int4 groups); compute is fp32. AdaLN scale / shift
 * depend only on g, so all of them are computed once per conditioning vector ("tables"). The sequence-mixing layers
 * run the step-by-step recurrence instead of the chunked form the Python code uses (steertts/seqmix.py); the two
 * are mathematically identical. The backward branch of a batch-1 unpadded sequence is the same recurrence run from
 * the last step to the first. */
#include "st_acoustic.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* keep every float op rounded separately (torch semantics for the duration arithmetic) */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

typedef struct {
  const st_mat* W;
  const float* b;
} lin;

typedef struct {
  int H, N, P;
  const float *norm_w, *kn_w, *qn_w, *k_bias, *q_bias;
  float* decay;   /* -exp(decay_log), [H] (owned) */
  lin in, k, q, v, step, rot, mix, out;
} seqmix;

typedef struct {
  int ada1, ada_seq, ada2;   /* indices into the AdaLN table */
  int seq;                   /* hybrid: a second, sequence-mixing branch */
  lin qkv, o;
  seqmix seq_f, seq_b;
  lin c1, c2;
} block;

typedef struct {
  int n;
  block* l;
  int ada_out;
} stack;

typedef struct {
  lin c1, c2, out;
  int ada1, ada2;
} varpred;

struct st_acoustic {
  int d, heads, ffn, k, g_dim, spk_dim, tag_dim, n_sym, n_tags, spk_enc, state_dim;
  const st_mat *emb, *tag;
  lin spk_proj, cond0, cond2;
  stack enc, dec;
  varpred dur, pitch, energy;
  lin pitch_emb, energy_emb, to_mel;
  int n_ada;
  lin* ada;   /* [n_ada] proj: g -> 2d */
};

/* ------------------------------------------------------------------ loading */
static int load_lin(st_weights* w, lin* l, const char* pfx, int O, int K) {
  char nm[640];   /* pfx: up to 600 bytes (add_ada) */
  snprintf(nm, sizeof nm, "%s.weight", pfx);
  l->W = st_weights_mat(w, nm, O, K);
  if (!l->W) return -1;
  snprintf(nm, sizeof nm, "%s.bias", pfx);
  l->b = st_weights_f32(w, nm, l->W->O);
  return l->b ? 0 : -1;
}

static const float* load_vec(st_weights* w, const char* pfx, const char* name, int64_t n) {
  char nm[1200];   /* st_name buffers are 512 bytes */
  snprintf(nm, sizeof nm, "%s.%s", pfx, name);
  return st_weights_f32(w, nm, n);
}

static int add_ada(st_acoustic* a, st_weights* w, const char* pfx, int cap) {
  char nm[600];   /* st_name buffers are 512 bytes */
  if (a->n_ada >= cap) {
    st_seterr("acoustic: AdaLN table overflow");
    return -1;
  }
  snprintf(nm, sizeof nm, "%s.proj", pfx);
  if (load_lin(w, &a->ada[a->n_ada], nm, 2 * a->d, a->g_dim)) return -1;
  return a->n_ada++;
}

static int load_seqmix(st_acoustic* a, st_weights* w, seqmix* m, const char* pfx) {
  char nm[256];
  int d = a->d;
  m->P = 32;   /* BiSeqMix head_dim */
  m->N = a->state_dim;
  m->H = d / m->P;
  if (d % m->P || m->N % 2 || m->N > 64) {
    st_seterr("acoustic: d=%d / state_dim=%d unsupported", d, m->N);
    return -1;
  }
  int H = m->H, N = m->N;
#define LL(field, sub, O)                       \
  snprintf(nm, sizeof nm, "%s.%s", pfx, sub);   \
  if (load_lin(w, &m->field, nm, O, d)) return -1;
  LL(in, "in_proj", d)
  LL(k, "k_proj", H * N)
  LL(q, "q_proj", H * N)
  LL(v, "v_proj", d)
  LL(step, "step_proj", H)
  LL(rot, "rot_proj", H * N / 2)
  LL(mix, "mix_proj", H)
  LL(out, "out_proj", d)
#undef LL
  if (!(m->norm_w = load_vec(w, pfx, "norm.weight", d)) || !(m->kn_w = load_vec(w, pfx, "k_norm.weight", N)) ||
      !(m->qn_w = load_vec(w, pfx, "q_norm.weight", N)) || !(m->k_bias = load_vec(w, pfx, "k_bias", H * N)) ||
      !(m->q_bias = load_vec(w, pfx, "q_bias", H * N)))
    return -1;
  const float* dl = load_vec(w, pfx, "decay_log", H);
  if (!dl) return -1;
  m->decay = (float*)st_malloc(sizeof(float) * (size_t)H);
  if (!m->decay) return -1;
  for (int h = 0; h < H; ++h) m->decay[h] = -expf(dl[h]);
  return 0;
}

static int load_stack(st_acoustic* a, st_weights* w, stack* s, const char* name, int n, const char* mixer, int cap) {
  char pfx[256], nm[512];
  int hyb = !strcmp(mixer, "hybrid");
  if (strcmp(mixer, "attn") && !hyb) {
    st_seterr("acoustic: unknown mixer %s", mixer);
    return -1;
  }
  s->n = n;
  s->l = (block*)st_calloc(sizeof(block) * (size_t)(n > 0 ? n : 1));
  if (!s->l) return -1;
  for (int i = 0; i < n; ++i) {
    block* b = &s->l[i];
    snprintf(pfx, sizeof pfx, "%s.layers.%d", name, i);
    b->seq = hyb;
    snprintf(nm, sizeof nm, "%s.n1", pfx);
    if ((b->ada1 = add_ada(a, w, nm, cap)) < 0) return -1;
    snprintf(nm, sizeof nm, "%s.att.in_proj_weight", pfx);
    if (!(b->qkv.W = st_weights_mat(w, nm, 3 * a->d, a->d))) return -1;
    snprintf(nm, sizeof nm, "%s.att.in_proj_bias", pfx);
    if (!(b->qkv.b = st_weights_f32(w, nm, 3 * a->d))) return -1;
    snprintf(nm, sizeof nm, "%s.att.out_proj", pfx);
    if (load_lin(w, &b->o, nm, a->d, a->d)) return -1;
    if (hyb) {
      snprintf(nm, sizeof nm, "%s.n_seq", pfx);
      if ((b->ada_seq = add_ada(a, w, nm, cap)) < 0) return -1;
      snprintf(nm, sizeof nm, "%s.seq.fwd", pfx);
      if (load_seqmix(a, w, &b->seq_f, nm)) return -1;
      snprintf(nm, sizeof nm, "%s.seq.bwd", pfx);
      if (load_seqmix(a, w, &b->seq_b, nm)) return -1;
    }
    snprintf(nm, sizeof nm, "%s.n2", pfx);
    if ((b->ada2 = add_ada(a, w, nm, cap)) < 0) return -1;
    snprintf(nm, sizeof nm, "%s.c1", pfx);
    if (load_lin(w, &b->c1, nm, a->ffn, a->d * a->k)) return -1;
    snprintf(nm, sizeof nm, "%s.c2", pfx);
    if (load_lin(w, &b->c2, nm, a->d, a->ffn)) return -1;
  }
  snprintf(nm, sizeof nm, "%s.out_norm", name);
  return (s->ada_out = add_ada(a, w, nm, cap)) < 0 ? -1 : 0;
}

static int load_varpred(st_acoustic* a, st_weights* w, varpred* v, const char* name, int cap) {
  char nm[256];
  snprintf(nm, sizeof nm, "%s.c1", name);
  if (load_lin(w, &v->c1, nm, a->d, a->d * 3)) return -1;
  snprintf(nm, sizeof nm, "%s.c2", name);
  if (load_lin(w, &v->c2, nm, a->d, a->d * 3)) return -1;
  snprintf(nm, sizeof nm, "%s.out", name);
  if (load_lin(w, &v->out, nm, 1, a->d)) return -1;
  snprintf(nm, sizeof nm, "%s.n1", name);
  if ((v->ada1 = add_ada(a, w, nm, cap)) < 0) return -1;
  snprintf(nm, sizeof nm, "%s.n2", name);
  return (v->ada2 = add_ada(a, w, nm, cap)) < 0 ? -1 : 0;
}

static int arch_int(const st_json* arch, const char* key, int dflt) {
  return (int)st_json_num(st_json_get(arch, key), dflt);
}

st_acoustic* st_acoustic_load(st_weights* w, const st_json* arch) {
  st_acoustic* a = (st_acoustic*)st_calloc(sizeof *a);
  if (!a) return NULL;
  a->d = arch_int(arch, "d", 192);
  a->heads = arch_int(arch, "heads", 2);
  a->ffn = arch_int(arch, "ffn", 384);
  a->k = arch_int(arch, "k", 3);
  a->g_dim = arch_int(arch, "g_dim", 128);
  a->spk_dim = arch_int(arch, "spk_dim", 64);
  a->tag_dim = arch_int(arch, "tag_dim", 32);
  a->state_dim = arch_int(arch, "state_dim", 32);
  a->spk_enc = arch_int(arch, "spk_enc_dim", 0);
  int enc_layers = arch_int(arch, "enc_layers", 4), dec_layers = arch_int(arch, "dec_layers", 4);
  const char* enc_mixer = st_json_str(st_json_get(arch, "enc_mixer"), "attn");
  const char* dec_mixer = st_json_str(st_json_get(arch, "dec_mixer"), "attn");
  if (a->d % a->heads || a->k % 2 == 0 || a->d % 2) {
    st_seterr("acoustic: unsupported arch (d=%d heads=%d k=%d)", a->d, a->heads, a->k);
    goto fail;
  }
  if (!(a->emb = st_weights_mat(w, "emb.weight", 0, a->d))) goto fail;
  a->n_sym = a->emb->O;
  int ns = arch_int(arch, "n_symbols", 0);
  if (ns > 0 && ns != a->n_sym) {
    st_seterr("acoustic: emb.weight has %d rows, arch n_symbols = %d", a->n_sym, ns);
    goto fail;
  }
  if (!(a->tag = st_weights_mat(w, "tag.weight", 0, a->tag_dim))) goto fail;
  a->n_tags = a->tag->O;
  if (a->spk_enc > 0 && load_lin(w, &a->spk_proj, "spk_proj", a->spk_dim, a->spk_enc)) goto fail;
  if (load_lin(w, &a->cond0, "cond.0", 256, a->spk_dim + a->tag_dim + 2 * ST_N_CTRL) ||
      load_lin(w, &a->cond2, "cond.2", a->g_dim, 256))
    goto fail;
  int cap = 4 * (enc_layers + dec_layers) + 2 + 6;
  a->ada = (lin*)st_calloc(sizeof(lin) * (size_t)cap);
  if (!a->ada) goto fail;
  if (load_stack(a, w, &a->enc, "encoder", enc_layers, enc_mixer, cap) ||
      load_stack(a, w, &a->dec, "decoder", dec_layers, dec_mixer, cap) ||
      load_varpred(a, w, &a->dur, "dur_pred", cap) || load_varpred(a, w, &a->pitch, "pitch_pred", cap) ||
      load_varpred(a, w, &a->energy, "energy_pred", cap) || load_lin(w, &a->pitch_emb, "pitch_emb", a->d, 3) ||
      load_lin(w, &a->energy_emb, "energy_emb", a->d, 3) || load_lin(w, &a->to_mel, "to_mel", ST_N_MELS, a->d))
    goto fail;
  return a;
fail:
  st_acoustic_free(a);
  return NULL;
}

static void free_stack(stack* s) {
  if (!s->l) return;
  for (int i = 0; i < s->n; ++i) {
    st_free(s->l[i].seq_f.decay);
    st_free(s->l[i].seq_b.decay);
  }
  st_free(s->l);
}

void st_acoustic_free(st_acoustic* a) {
  if (!a) return;
  free_stack(&a->enc);
  free_stack(&a->dec);
  st_free(a->ada);
  st_free(a);
}

int st_acoustic_g_dim(const st_acoustic* a) { return a->g_dim; }
int st_acoustic_n_symbols(const st_acoustic* a) { return a->n_sym; }
int st_acoustic_n_tags(const st_acoustic* a) { return a->n_tags; }
int st_acoustic_spk_enc_dim(const st_acoustic* a) { return a->spk_proj.W ? a->spk_enc : 0; }

/* ------------------------------------------------------------------ conditioning */
int st_acoustic_condition(const st_acoustic* a, const float* spk64, const float* spk_vec, int tag, const float* ctrl,
                          const float* cmask, float* g) {
  int ci = a->spk_dim + a->tag_dim + 2 * ST_N_CTRL;
  float in[512], hid[256];
  if (ci > 512) {
    st_seterr("acoustic: conditioning input too wide");
    return -1;
  }
  if (tag < 0 || tag >= a->n_tags) {
    st_seterr("acoustic: tag index %d out of range [0, %d)", tag, a->n_tags);
    return -1;
  }
  if (spk_vec) {
    if (!a->spk_proj.W) {
      st_seterr("acoustic: this model has no speaker-vector projection");
      return -1;
    }
    st_linear(NULL, spk_vec, 1, a->spk_enc, a->spk_proj.W, a->spk_proj.b, in, a->spk_dim, 0);
  } else if (spk64) {
    memcpy(in, spk64, sizeof(float) * (size_t)a->spk_dim);
  } else {
    st_seterr("acoustic: no speaker given");
    return -1;
  }
  st_mat_row(a->tag, tag, in + a->spk_dim);
  float* c = in + a->spk_dim + a->tag_dim;
  for (int i = 0; i < ST_N_CTRL; ++i) {
    float v = ctrl ? ctrl[i] : 0.f, m = cmask ? cmask[i] : 0.f;
    if (isnan(v)) v = 0.f;
    else if (isinf(v)) v = v > 0 ? FLT_MAX : -FLT_MAX;
    c[i] = v * m;
    c[ST_N_CTRL + i] = m;
  }
  st_linear(NULL, in, 1, ci, a->cond0.W, a->cond0.b, hid, 256, 0);
  st_silu(hid, 256);
  st_linear(NULL, hid, 1, 256, a->cond2.W, a->cond2.b, g, a->g_dim, 0);
  return 0;
}

/* AdaLN tables for one g: row i = [1 + s (d), b (d)] */
static float* make_tables(const st_acoustic* a, const float* g) {
  int d = a->d;
  float* t = (float*)st_malloc(sizeof(float) * (size_t)a->n_ada * 2 * (size_t)d);
  if (!t) return NULL;
  for (int i = 0; i < a->n_ada; ++i) {
    float* r = t + (size_t)i * 2 * d;
    st_linear(NULL, g, 1, a->g_dim, a->ada[i].W, a->ada[i].b, r, 2 * d, 0);
    for (int c = 0; c < d; ++c) r[c] = 1.f + r[c];
  }
  return t;
}

/* ------------------------------------------------------------------ row-parallel helpers */
typedef void (*rows_fn)(void* ctx, int t0, int t1);
typedef struct {
  rows_fn fn;
  void* ctx;
  int T;
} rows_job;

static void rows_task(void* c, int task, int n_tasks) {
  const rows_job* j = (const rows_job*)c;
  int t0 = (int)((long long)j->T * task / n_tasks), t1 = (int)((long long)j->T * (task + 1) / n_tasks);
  if (t1 > t0) j->fn(j->ctx, t0, t1);
}

/* fn over [0, T) split into contiguous row ranges of at least min_rows */
static void par_rows(st_pool* p, int T, int min_rows, rows_fn fn, void* ctx) {
  int tasks = st_pool_size(p);
  if (tasks > T / min_rows) tasks = T / min_rows;
  if (tasks <= 1) {
    if (T > 0) fn(ctx, 0, T);
    return;
  }
  rows_job j = {fn, ctx, T};
  st_parallel_for(p, tasks, rows_task, &j);
}

typedef struct {
  const float* x;
  float* y;
  int d;
  const float* tab;
} adaln_job;

/* y = LayerNorm(x) * (1 + s) + b  (LayerNorm without affine, eps 1e-5) */
static void adaln_rows(void* ctx, int t0, int t1) {
  const adaln_job* j = (const adaln_job*)ctx;
  int d = j->d;
  const float *opl = j->tab, *b = j->tab + d;
  for (int t = t0; t < t1; ++t) {
    const float* r = j->x + (size_t)t * d;
    float* o = j->y + (size_t)t * d;
    double m = 0, v = 0;
    for (int c = 0; c < d; ++c) m += r[c];
    m /= d;
    for (int c = 0; c < d; ++c) {
      double e = r[c] - m;
      v += e * e;
    }
    v /= d;
    float inv = (float)(1.0 / sqrt(v + 1e-5)), mf = (float)m;
    for (int c = 0; c < d; ++c) {
      float u = (r[c] - mf) * inv;
      u = u * opl[c];
      o[c] = u + b[c];
    }
  }
}

static void adaln(st_pool* p, const float* x, int T, int d, const float* tab, float* y) {
  adaln_job j = {x, y, d, tab};
  par_rows(p, T, 16, adaln_rows, &j);
}

typedef struct {
  float* x;
  int C;
} gelu_job;

static void gelu_rows(void* ctx, int t0, int t1) {
  const gelu_job* j = (const gelu_job*)ctx;
  st_gelu(j->x + (size_t)t0 * j->C, (size_t)(t1 - t0) * j->C);
}

/* ------------------------------------------------------------------ sequence mixing */
typedef struct {
  const seqmix* m;
  int T, rev;
  float *nu, *gate, *k, *q, *v, *step, *rot, *mix, *y;
} mdir;

static size_t mdir_floats(const st_acoustic* a, int T) {
  int H = a->d / 32, N = a->state_dim;
  return (size_t)T * (size_t)(5 * a->d + 2 * H * N + H * N / 2 + 2 * H);
}

static void mdir_bind(mdir* md, const seqmix* m, int T, int rev, int d, float* buf) {
  int H = m->H, N = m->N;
  md->m = m, md->T = T, md->rev = rev;
  md->nu = buf, buf += (size_t)T * d;
  md->gate = buf, buf += (size_t)T * d;
  md->v = buf, buf += (size_t)T * d;
  md->y = buf, buf += (size_t)T * d;
  md->k = buf, buf += (size_t)T * H * N;
  md->q = buf, buf += (size_t)T * H * N;
  md->rot = buf, buf += (size_t)T * H * N / 2;
  md->step = buf, buf += (size_t)T * H;
  md->mix = buf;
  (void)buf;
}

static void seqmix_norm_rows(void* ctx, int t0, int t1) {
  const mdir* md = (const mdir*)ctx;
  int d = md->m->H * md->m->P;
  st_rmsnorm(md->nu + (size_t)t0 * d, t1 - t0, d, d, md->m->norm_w, FLT_EPSILON);
}

/* step = softplus(.), mix = sigmoid(.), rot *= step, k / q = RMSNorm(.) * w + bias */
static void seqmix_coef_rows(void* ctx, int t0, int t1) {
  const mdir* md = (const mdir*)ctx;
  const seqmix* m = md->m;
  int H = m->H, N = m->N;
  for (int t = t0; t < t1; ++t) {
    float* kr = md->k + (size_t)t * H * N;
    float* qr = md->q + (size_t)t * H * N;
    st_rmsnorm(kr, H, N, N, m->kn_w, FLT_EPSILON);
    st_rmsnorm(qr, H, N, N, m->qn_w, FLT_EPSILON);
    for (int i = 0; i < H * N; ++i) kr[i] += m->k_bias[i], qr[i] += m->q_bias[i];
    for (int h = 0; h < H; ++h) {
      float v = md->step[(size_t)t * H + h];
      float sp = v > 20.f ? v : log1pf(expf(v));   /* F.softplus, threshold 20 */
      md->step[(size_t)t * H + h] = sp;
      float l = md->mix[(size_t)t * H + h];
      md->mix[(size_t)t * H + h] = 1.f / (1.f + expf(-l));
      float* ang = md->rot + ((size_t)t * H + h) * (N / 2);
      for (int i = 0; i < N / 2; ++i) ang[i] *= sp;
    }
  }
}

/* Layer.norm, the projections and the per-step coefficients (layer.coeffs) */
static void seqmix_prepare(st_pool* p, mdir* md, const float* u, int d) {
  const seqmix* m = md->m;
  int T = md->T, H = m->H, N = m->N;
  memcpy(md->nu, u, sizeof(float) * (size_t)T * d);
  par_rows(p, T, 16, seqmix_norm_rows, md);
  st_linear(p, md->nu, T, d, m->in.W, m->in.b, md->gate, d, 0);
  st_linear(p, md->nu, T, d, m->k.W, m->k.b, md->k, H * N, 0);
  st_linear(p, md->nu, T, d, m->q.W, m->q.b, md->q, H * N, 0);
  st_linear(p, md->nu, T, d, m->v.W, m->v.b, md->v, d, 0);
  st_linear(p, md->nu, T, d, m->step.W, m->step.b, md->step, H, 0);
  st_linear(p, md->nu, T, d, m->rot.W, m->rot.b, md->rot, H * N / 2, 0);
  st_linear(p, md->nu, T, d, m->mix.W, m->mix.b, md->mix, H, 0);
  par_rows(p, T, 8, seqmix_coef_rows, md);
}

/* One head of the recurrence: h_t = R_t(a_t h_{t-1} + b_t k_{t-1} v_{t-1}^T) + c_t k_t v_t^T,
 * y_t = q_t^T h_t; state [N, P], R_t rotates the (2i, 2i+1) state rows by rot_t[i]. rev: run from T-1 down to 0
 * (the backward branch). The P axis is vectorised with GCC / Clang vector extensions where available. */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(ST_NO_VEC)
typedef float v8f __attribute__((vector_size(32)));
#define SPLAT8(a) ((v8f){(a), (a), (a), (a), (a), (a), (a), (a)})
#define ST_VEC8 1
#endif

static void seqmix_scan_head(const mdir* md, int h) {
  const seqmix* m = md->m;
  int T = md->T, H = m->H, N = m->N, P = m->P, d = H * P;
  float cs[32], sn[32];
  float dk = m->decay[h];
#ifdef ST_VEC8
  if (P % 8 == 0 && P <= 64 && N <= 64) {
    int PV = P / 8;
    v8f st[64 * 8], vpv[8], vcv[8], acc[8];
    for (int s = 0; s < T; ++s) {
      int t = md->rev ? T - 1 - s : s, tp = md->rev ? t + 1 : t - 1;
      float stp = md->step[(size_t)t * H + h], mx = md->mix[(size_t)t * H + h];
      float ak = expf(stp * dk), ck = mx * stp, bk = (1.f - mx) * stp * ak;
      const float* kc = md->k + ((size_t)t * H + h) * N;
      const float* qc = md->q + ((size_t)t * H + h) * N;
      const float* ang = md->rot + ((size_t)t * H + h) * (N / 2);
      memcpy(vcv, md->v + (size_t)t * d + (size_t)h * P, sizeof(float) * (size_t)P);
      for (int i = 0; i < N / 2; ++i) cs[i] = cosf(ang[i]), sn[i] = sinf(ang[i]);
      if (s > 0) {
        const float* kp = md->k + ((size_t)tp * H + h) * N;
        memcpy(vpv, md->v + (size_t)tp * d + (size_t)h * P, sizeof(float) * (size_t)P);
        v8f av = SPLAT8(ak);
        for (int i = 0; i < N / 2; ++i) {
          v8f *h0 = st + (size_t)(2 * i) * PV, *h1 = h0 + PV;
          v8f b0 = SPLAT8(bk * kp[2 * i]), b1 = SPLAT8(bk * kp[2 * i + 1]);
          v8f g0 = SPLAT8(ck * kc[2 * i]), g1 = SPLAT8(ck * kc[2 * i + 1]);
          v8f c = SPLAT8(cs[i]), sv = SPLAT8(sn[i]);
          for (int k = 0; k < PV; ++k) {
            v8f u0 = av * h0[k] + b0 * vpv[k], u1 = av * h1[k] + b1 * vpv[k];
            h0[k] = c * u0 - sv * u1 + g0 * vcv[k];
            h1[k] = sv * u0 + c * u1 + g1 * vcv[k];
          }
        }
      } else {
        for (int n = 0; n < N; ++n) {   /* h_{-1} = 0 and no previous write: h_0 = ck B_0 x_0^T */
          v8f g0 = SPLAT8(ck * kc[n]);
          for (int k = 0; k < PV; ++k) st[(size_t)n * PV + k] = g0 * vcv[k];
        }
      }
      for (int k = 0; k < PV; ++k) acc[k] = SPLAT8(0.f);
      for (int n = 0; n < N; ++n) {
        v8f c = SPLAT8(qc[n]);
        for (int k = 0; k < PV; ++k) acc[k] += c * st[(size_t)n * PV + k];
      }
      memcpy(md->y + (size_t)t * d + (size_t)h * P, acc, sizeof(float) * (size_t)P);
    }
    return;
  }
#endif
  float st[64 * 64];
  if (N > 64 || P > 64) return;   /* checked at load */
  for (int s = 0; s < T; ++s) {
    int t = md->rev ? T - 1 - s : s, tp = md->rev ? t + 1 : t - 1;
    float stp = md->step[(size_t)t * H + h], mx = md->mix[(size_t)t * H + h];
    float ak = expf(stp * dk), ck = mx * stp, bk = (1.f - mx) * stp * ak;
    const float* kc = md->k + ((size_t)t * H + h) * N;
    const float* qc = md->q + ((size_t)t * H + h) * N;
    const float* vc = md->v + (size_t)t * d + (size_t)h * P;
    const float* ang = md->rot + ((size_t)t * H + h) * (N / 2);
    for (int i = 0; i < N / 2; ++i) cs[i] = cosf(ang[i]), sn[i] = sinf(ang[i]);
    if (s > 0) {
      const float* kp = md->k + ((size_t)tp * H + h) * N;
      const float* vp = md->v + (size_t)tp * d + (size_t)h * P;
      for (int i = 0; i < N / 2; ++i) {
        float* h0 = st + (size_t)(2 * i) * P;
        float* h1 = h0 + P;
        float b0 = bk * kp[2 * i], b1 = bk * kp[2 * i + 1], g0 = ck * kc[2 * i], g1 = ck * kc[2 * i + 1];
        float c = cs[i], sv = sn[i];
        for (int q = 0; q < P; ++q) {
          float u0 = ak * h0[q] + b0 * vp[q], u1 = ak * h1[q] + b1 * vp[q];
          h0[q] = c * u0 - sv * u1 + g0 * vc[q];
          h1[q] = sv * u0 + c * u1 + g1 * vc[q];
        }
      }
    } else {
      for (int n = 0; n < N; ++n)
        for (int q = 0; q < P; ++q) st[(size_t)n * P + q] = ck * kc[n] * vc[q];
    }
    float* y = md->y + (size_t)t * d + (size_t)h * P;
    for (int q = 0; q < P; ++q) y[q] = 0.f;
    for (int n = 0; n < N; ++n) {
      float c = qc[n];
      const float* r = st + (size_t)n * P;
      for (int q = 0; q < P; ++q) y[q] += c * r[q];
    }
  }
}

typedef struct {
  mdir* dirs;
  int H;
} scan_job;

static void scan_task(void* ctx, int task, int n_tasks) {
  (void)n_tasks;
  scan_job* j = (scan_job*)ctx;
  seqmix_scan_head(&j->dirs[task / j->H], task % j->H);
}

static void seqmix_gate_rows(void* ctx, int t0, int t1) {
  const mdir* md = (const mdir*)ctx;
  size_t d = (size_t)md->m->H * md->m->P, a = (size_t)t0 * d, n = (size_t)(t1 - t0) * d;
  st_silu(md->gate + a, n);
  for (size_t i = 0; i < n; ++i) md->y[a + i] *= md->gate[a + i];
}

/* x += BiSeqMix(u): forward and backward layers on the AdaLN output u [T, d]. buf: 2 * mdir_floats. */
static void biseqmix(const st_acoustic* a, st_pool* p, const seqmix* f, const seqmix* b, const float* u, int T, float* x,
                    float* buf) {
  int d = a->d;
  mdir md[2];
  mdir_bind(&md[0], f, T, 0, d, buf);
  mdir_bind(&md[1], b, T, 1, d, buf + mdir_floats(a, T));
  seqmix_prepare(p, &md[0], u, d);
  seqmix_prepare(p, &md[1], u, d);
  scan_job j = {md, f->H};
  st_parallel_for(p, 2 * f->H, scan_task, &j);
  for (int k = 0; k < 2; ++k) {
    /* gate_and_proj: out_proj(y * silu(in_proj(nu))) */
    par_rows(p, T, 16, seqmix_gate_rows, &md[k]);
    st_linear(p, md[k].y, T, d, md[k].m->out.W, md[k].m->out.b, x, d, 1);
  }
}

/* ------------------------------------------------------------------ stacks */
static void add_pos(float* x, int T, int d) {
  float sc = (float)(-log(10000.0) / d);
  for (int i = 0; i < d / 2; ++i) {
    float div = expf((float)(2 * i) * sc);
    for (int t = 0; t < T; ++t) {
      float ang = (float)t * div;
      x[(size_t)t * d + 2 * i] += sinf(ang);
      x[(size_t)t * d + 2 * i + 1] += cosf(ang);
    }
  }
}

/* Stack.forward: x [T, d] in place -> out_norm(...) */
static int run_stack(const st_acoustic* a, st_pool* p, const stack* s, const float* tab, float* x, int T) {
  int d = a->d, ffn = a->ffn;
  /* h [T, d] + one scratch region reused by attention (qkv + out), the sequence-mixing branches and the FFN */
  size_t rest_n = 2 * mdir_floats(a, T);
  if ((size_t)T * 4 * d > rest_n) rest_n = (size_t)T * 4 * d;
  if ((size_t)T * ffn > rest_n) rest_n = (size_t)T * ffn;
  float* ws = (float*)st_malloc(sizeof(float) * ((size_t)T * d + rest_n));
  if (!ws) return -1;
  float* h = ws;
  float* rest = ws + (size_t)T * d;
  add_pos(x, T, d);
  for (int i = 0; i < s->n; ++i) {
    const block* b = &s->l[i];
    adaln(p, x, T, d, tab + (size_t)b->ada1 * 2 * d, h);
    float* qkv = rest;
    float* att = rest + (size_t)T * 3 * d;
    st_linear(p, h, T, d, b->qkv.W, b->qkv.b, qkv, 3 * d, 0);
    if (st_attention(p, qkv, 3 * d, qkv + d, 3 * d, qkv + 2 * d, 3 * d, att, d, T, T, a->heads, d / a->heads, NULL,
                     NULL, 0.f)) {
      st_free(ws);
      return -1;
    }
    st_linear(p, att, T, d, b->o.W, b->o.b, x, d, 1);
    if (b->seq) {
      adaln(p, x, T, d, tab + (size_t)b->ada_seq * 2 * d, h);
      biseqmix(a, p, &b->seq_f, &b->seq_b, h, T, x, rest);
    }
    adaln(p, x, T, d, tab + (size_t)b->ada2 * 2 * d, h);
    float* f = rest;
    if (st_conv1d(p, h, T, d, d, b->c1.W, a->k, 1, 1, a->k / 2, a->k / 2, b->c1.b, f, ffn) < 0) {
      st_free(ws);
      return -1;
    }
    gelu_job gj = {f, ffn};
    par_rows(p, T, 8, gelu_rows, &gj);
    st_linear(p, f, T, ffn, b->c2.W, b->c2.b, x, d, 1);
  }
  adaln(p, x, T, d, tab + (size_t)s->ada_out * 2 * d, h);
  memcpy(x, h, sizeof(float) * (size_t)T * d);
  st_free(ws);
  return 0;
}

/* VariancePredictor.forward: h [N, d] -> out [N] */
static int run_varpred(const st_acoustic* a, st_pool* p, const varpred* v, const float* tab, const float* h, int N,
                       float* out) {
  int d = a->d;
  float* ws = (float*)st_malloc(sizeof(float) * (size_t)N * d * 2);
  if (!ws) return -1;
  float *u = ws, *z = ws + (size_t)N * d;
  if (st_conv1d(p, h, N, d, d, v->c1.W, 3, 1, 1, 1, 1, v->c1.b, u, d) < 0) goto fail;
  st_relu(u, (size_t)N * d);
  adaln(p, u, N, d, tab + (size_t)v->ada1 * 2 * d, z);
  if (st_conv1d(p, z, N, d, d, v->c2.W, 3, 1, 1, 1, 1, v->c2.b, u, d) < 0) goto fail;
  st_relu(u, (size_t)N * d);
  adaln(p, u, N, d, tab + (size_t)v->ada2 * 2 * d, z);
  st_linear(p, z, N, d, v->out.W, v->out.b, out, 1, 0);
  st_free(ws);
  return 0;
fail:
  st_free(ws);
  return -1;
}

/* encoder output h [N, d] and the three predictor outputs [N] each */
static int encode_predict(const st_acoustic* a, st_pool* p, const int* ids, int N, const float* tab, float* h,
                          float* logdur, float* pitch, float* energy) {
  int d = a->d;
  for (int i = 0; i < N; ++i) st_mat_row(a->emb, ids[i], h + (size_t)i * d);
  ST_CHECK(run_stack(a, p, &a->enc, tab, h, N));
  ST_CHECK(run_varpred(a, p, &a->dur, tab, h, N, logdur));
  ST_CHECK(run_varpred(a, p, &a->pitch, tab, h, N, pitch));
  ST_CHECK(run_varpred(a, p, &a->energy, tab, h, N, energy));
  return 0;
}

/* length_regulate(h + pe) then decoder and to_mel -> mel [T, 100] */
static int decode(const st_acoustic* a, st_pool* p, const float* h, const float* pe, const int* dur, int N, int T,
                  const float* tab, float* mel) {
  int d = a->d;
  float* x = (float*)st_malloc(sizeof(float) * (size_t)T * d);
  if (!x) return -1;
  int f = 0;
  for (int i = 0; i < N; ++i)
    for (int r = 0; r < dur[i]; ++r, ++f) {
      float* o = x + (size_t)f * d;
      const float *hi = h + (size_t)i * d, *pi = pe + (size_t)i * d;
      for (int c = 0; c < d; ++c) o[c] = hi[c] + pi[c];
    }
  if (run_stack(a, p, &a->dec, tab, x, T)) {
    st_free(x);
    return -1;
  }
  st_linear(p, x, T, d, a->to_mel.W, a->to_mel.b, mel, ST_N_MELS, 0);
  st_free(x);
  return 0;
}

int st_acoustic_infer(st_acoustic* a, st_pool* p, const int* ids, const int* widx, int N, const float* g,
                      const float* g0, float guidance, float pace, const st_shift* shifts, int n_words, float** mel,
                      int* T_out, int** dur_out) {
  int d = a->d, rc = -1;
  *mel = NULL;
  *T_out = 0;
  if (dur_out) *dur_out = NULL;
  if (N <= 0) {
    st_seterr("acoustic: empty input");
    return -1;
  }
  for (int i = 0; i < N; ++i)
    if (ids[i] < 0 || ids[i] >= a->n_sym) {
      st_seterr("acoustic: symbol id %d out of range [0, %d)", ids[i], a->n_sym);
      return -1;
    }
  if (!(pace > 0.f)) {
    st_seterr("acoustic: pace must be > 0");
    return -1;
  }
  float *tab = NULL, *tab0 = NULL, *h = NULL, *h0 = NULL, *pr = NULL, *pe = NULL, *mel0 = NULL, *out = NULL;
  int* dur = NULL;
  tab = make_tables(a, g);
  if (!tab) goto done;
  if (g0 && !(tab0 = make_tables(a, g0))) goto done;
  h = (float*)st_malloc(sizeof(float) * (size_t)N * d * 2);
  pr = (float*)st_malloc(sizeof(float) * (size_t)N * 6);
  pe = (float*)st_malloc(sizeof(float) * (size_t)N * d * 2);
  dur = (int*)st_malloc(sizeof(int) * (size_t)N);
  if (!h || !pr || !pe || !dur) goto done;
  h0 = h + (size_t)N * d;
  float *ld = pr, *pit = pr + N, *en = pr + 2 * N, *ld0 = pr + 3 * N, *pit0 = pr + 4 * N, *en0 = pr + 5 * N;
  if (encode_predict(a, p, ids, N, tab, h, ld, pit, en)) goto done;
  if (g0) {
    if (encode_predict(a, p, ids, N, tab0, h0, ld0, pit0, en0)) goto done;
    for (int i = 0; i < N; ++i) {   /* mix = a0 + s * (a - a0) */
      float t;
      t = ld[i] - ld0[i], t = guidance * t, ld[i] = ld0[i] + t;
      t = pit[i] - pit0[i], t = guidance * t, pit[i] = pit0[i] + t;
      t = en[i] - en0[i], t = guidance * t, en[i] = en0[i] + t;
    }
  }
  /* dur = round(clamp(exp(logdur) - 1, 0) / pace * word factor), round half to even; dur[0] >= 1 */
  long long Tsum = 0;
  for (int i = 0; i < N; ++i) {
    float v = expf(ld[i]) - 1.f;
    if (v < 0.f) v = 0.f;
    v = v / pace;
    if (shifts && widx && widx[i] >= 0 && widx[i] < n_words) {
      const st_shift* s = &shifts[widx[i]];
      pit[i] += s->pitch;
      en[i] += s->energy;
      v *= s->dur;
    }
    float r = nearbyintf(v);
    if (!(r > 0.f)) r = 0.f;   /* NaN, <= 0 */
    if (i == 0 && r < 1.f) r = 1.f;
    if (r > (float)(1 << 20)) {   /* checked before the conversion: an overflowed exp is +inf */
      st_seterr("acoustic: a token duration is out of range");
      goto done;
    }
    dur[i] = (int)r;
    Tsum += dur[i];
    if (Tsum > ST_MAX_FRAMES) break;
  }
  if (Tsum > ST_MAX_FRAMES) {   /* attention and scratch grow with the length of one pass */
    st_seterr("this text is too long to render in one pass (over %d s of audio); use stream() or markup to render it "
              "sentence by sentence", (int)(ST_MAX_FRAMES * 256LL / 24000));
    goto done;
  }
  int T = (int)Tsum;
  /* pe = pitch_emb(pitch) + energy_emb(energy): Conv1d(1, d, 3, padding 1) each */
  float* pe2 = pe + (size_t)N * d;
  if (st_conv1d(p, pit, N, 1, 1, a->pitch_emb.W, 3, 1, 1, 1, 1, a->pitch_emb.b, pe, d) < 0 ||
      st_conv1d(p, en, N, 1, 1, a->energy_emb.W, 3, 1, 1, 1, 1, a->energy_emb.b, pe2, d) < 0)
    goto done;
  for (size_t i = 0; i < (size_t)N * d; ++i) pe[i] += pe2[i];
  out = (float*)st_malloc(sizeof(float) * (size_t)T * ST_N_MELS);
  if (!out) goto done;
  if (decode(a, p, h, pe, dur, N, T, tab, out)) goto done;
  if (g0) {
    mel0 = (float*)st_malloc(sizeof(float) * (size_t)T * ST_N_MELS);
    if (!mel0 || decode(a, p, h0, pe, dur, N, T, tab0, mel0)) goto done;
    for (size_t i = 0; i < (size_t)T * ST_N_MELS; ++i) {
      float t = out[i] - mel0[i];
      t = guidance * t;
      out[i] = mel0[i] + t;
    }
  }
  *mel = out;
  out = NULL;
  *T_out = T;
  if (dur_out) {
    *dur_out = dur;
    dur = NULL;
  }
  rc = 0;
done:
  st_free(tab);
  st_free(tab0);
  st_free(h);
  st_free(pr);
  st_free(pe);
  st_free(mel0);
  st_free(out);
  st_free(dur);
  return rc;
}
