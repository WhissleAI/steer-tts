/* Acoustic model encoder (+ variance predictors) and decoder on a ggml device. Same network as st_acoustic.c; the
 * host keeps the embedding lookup, positions, AdaLN tables (they depend on g only), durations and the length
 * regulation, so durations are decided by the same code as on the CPU.
 *
 * Sequence mixing (BiSeqMix, one direction; reversed rows for the backward one) maps onto ggml's Mamba-2 scan:
 *   CPU recurrence:  h_t = R_t (a_t h_{t-1} + b_t k_{t-1} v_{t-1}^T) + c_t k_t v_t^T,   y_t = q_t^T h_t
 *   with R_t rotating state row pairs (2i, 2i+1) by rot_t[i] * step_t (R_0 unused), a_t = exp(step_t * decay),
 *   b_t = (1 - mix_t) step_t a_t, c_t = mix_t step_t.
 * 2-D rotations commute, so with Theta_t = sum_{s=1..t} angle_s and h'_t = R(-Theta_t) h_t:
 *   h'_t = a_t h'_{t-1} + b_t k'_{t-1} v_{t-1}^T + c_t k'_t v_t^T,   y_t = q'_t^T h'_t,   k' / q' = R(-Theta_t) k / q.
 * That is a scalar-decay (Mamba-2) scan with two rank-1 inputs; by linearity it is two scans with the same decay and
 * readout, run as one scan over 2H heads: heads [0, H) get (B = mix k', x = v), heads [H, 2H) get
 * (B = (1 - mix) a k'_{t-1}, x = v_{t-1}); both with dt = raw step (the scan applies softplus and multiplies x by it)
 * and A = decay; the two halves of y are summed. The state size is zero-padded from 32 to 64 (ggml-cuda's smallest
 * Mamba-2 kernel); K = 2 keeps ggml-cuda on its fp32 sequential scan (its chunked SSD path uses fp16 GEMMs). */
#include "st_acoustic_impl.h"
#include "st_gpu.h"

#ifdef STEERTTS_GGML
#include <float.h>
#include <string.h>

#include "st_ggml.h"

#define NS_PAD 64   /* padded state size */

typedef struct {
  struct ggml_tensor* W;
  struct ggml_tensor* b;
} glin;

typedef struct {
  glin in, k, q, v, step, rot, mix, out;
  struct ggml_tensor *norm_w, *kn_w, *qn_w, *k_bias, *q_bias, *A2;
} gseq;

typedef struct {
  glin qkv, o, c1, c2;
  gseq f, b;
} gblock;

typedef struct {
  glin c1, c2, out;
} gvp;

typedef struct {
  st_gpu* g;
  st_gw w;
  st_gr r;
  gblock *enc, *dec;
  gvp dur, pitch, energy;
  glin to_mel;
} gac;

static void gac_free(gac* q) {
  if (!q) return;
  st_gr_free(&q->r);
  st_gw_free(&q->w);
  st_free(q->enc);
  st_free(q->dec);
  st_free(q);
}

static int up_lin(st_gw* w, glin* o, const lin* l) {
  o->W = st_gw_mat(w, l->W);
  o->b = l->b ? st_gw_f32(w, l->b, l->W->O, 1, 1, 0) : NULL;
  return o->W && (o->b || !l->b) ? 0 : -1;
}

static int up_seq(st_gw* w, gseq* o, const seqmix* m) {
  int H = m->H, N = m->N, d = H * m->P;
  if (up_lin(w, &o->in, &m->in) || up_lin(w, &o->k, &m->k) || up_lin(w, &o->q, &m->q) || up_lin(w, &o->v, &m->v) ||
      up_lin(w, &o->step, &m->step) || up_lin(w, &o->rot, &m->rot) || up_lin(w, &o->mix, &m->mix) ||
      up_lin(w, &o->out, &m->out))
    return -1;
  float A2[2 * 64];
  if (H > 64) return -1;
  for (int h = 0; h < H; ++h) A2[h] = A2[H + h] = m->decay[h];
  o->norm_w = st_gw_f32(w, m->norm_w, d, 1, 1, 0);
  o->kn_w = st_gw_f32(w, m->kn_w, N, 1, 1, 0);
  o->qn_w = st_gw_f32(w, m->qn_w, N, 1, 1, 0);
  o->k_bias = st_gw_f32(w, m->k_bias, N, H, 1, 0);
  o->q_bias = st_gw_f32(w, m->q_bias, N, H, 1, 0);
  o->A2 = st_gw_f32(w, A2, 1, 2 * H, 1, 1);
  return o->norm_w && o->kn_w && o->qn_w && o->k_bias && o->q_bias && o->A2 ? 0 : -1;
}

static int up_stack(st_gw* w, gblock** out, const stack* s) {
  *out = (gblock*)st_calloc(sizeof(gblock) * (size_t)(s->n > 0 ? s->n : 1));
  if (!*out) return -1;
  for (int i = 0; i < s->n; ++i) {
    const block* b = &s->l[i];
    gblock* o = &(*out)[i];
    if (up_lin(w, &o->qkv, &b->qkv) || up_lin(w, &o->o, &b->o) || up_lin(w, &o->c1, &b->c1) ||
        up_lin(w, &o->c2, &b->c2))
      return -1;
    if (b->seq && (up_seq(w, &o->f, &b->seq_f) || up_seq(w, &o->b, &b->seq_b))) return -1;
  }
  return 0;
}

static int up_vp(st_gw* w, gvp* o, const varpred* v) {
  return up_lin(w, &o->c1, &v->c1) || up_lin(w, &o->c2, &v->c2) || up_lin(w, &o->out, &v->out) ? -1 : 0;
}

int st_acoustic_gpu_attach(st_acoustic* a, st_gpu* g) {
  if (!a || !g) return -1;
  st_acoustic_gpu_detach(a);
  if (a->state_dim > NS_PAD || a->d % 32) {
    st_seterr("acoustic: state_dim %d / d %d not supported on the GPU path", a->state_dim, a->d);
    return -1;
  }
  gac* q = (gac*)st_calloc(sizeof *q);
  if (!q) return -1;
  q->g = g;
  int nl = a->enc.n + a->dec.n;
  if (st_gw_init(&q->w, 64 + nl * (8 + 2 * 22)) || st_gr_init(&q->r, g, 8192)) goto fail;
  if (up_stack(&q->w, &q->enc, &a->enc) || up_stack(&q->w, &q->dec, &a->dec) || up_vp(&q->w, &q->dur, &a->dur) ||
      up_vp(&q->w, &q->pitch, &a->pitch) || up_vp(&q->w, &q->energy, &a->energy) ||
      up_lin(&q->w, &q->to_mel, &a->to_mel))
    goto fail;
  if (st_gw_upload(&q->w, g)) goto fail;
  a->gpu = q;
  return 0;
fail:
  gac_free(q);
  return -1;
}

void st_acoustic_gpu_detach(st_acoustic* a) {
  if (!a || !a->gpu) return;
  gac_free((gac*)a->gpu);
  a->gpu = NULL;
}

/* ------------------------------------------------------------------ graph pieces */
typedef struct {
  struct ggml_context* c;
  struct ggml_tensor* tab;   /* {2d, n_ada} */
  struct ggml_tensor* rev;   /* {T} i32: T-1 .. 0 */
  struct ggml_tensor* s0;    /* {NS_PAD, P, 2H} zeros */
  struct ggml_tensor* ids;   /* {1} i32 = 0 */
  int d;
} gctx;

static struct ggml_tensor* lin_(gctx* G, struct ggml_tensor* x, const glin* l) { return st_g_linear(G->c, x, l->W, l->b); }

/* LayerNorm (eps 1e-5, no affine) * (1 + s) + b, table row i */
static struct ggml_tensor* adaln(gctx* G, struct ggml_tensor* x, int i) {
  struct ggml_context* c = G->c;
  size_t row = (size_t)i * 2 * G->d * sizeof(float);
  struct ggml_tensor* opl = ggml_view_1d(c, G->tab, G->d, row);
  struct ggml_tensor* b = ggml_view_1d(c, G->tab, G->d, row + (size_t)G->d * sizeof(float));
  return ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), opl), b);
}

/* t -> t - 1 along dim 2 of {n0, n1, T}, zero at t = 0 */
static struct ggml_tensor* shift_t(struct ggml_context* c, struct ggml_tensor* x) {
  int64_t T = x->ne[2];
  if (T == 1) return ggml_scale(c, x, 0.0f);
  struct ggml_tensor* v = ggml_view_3d(c, x, x->ne[0], x->ne[1], T - 1, x->nb[1], x->nb[2], 0);
  return ggml_pad_ext(c, v, 0, 0, 0, 0, 1, 0, 0, 0);
}

/* pairs (2i, 2i+1) of x {N, H, T} rotated by -theta {N/2, H, T}: (c x0 + s x1, c x1 - s x0) */
static struct ggml_tensor* rot_neg(struct ggml_context* c, struct ggml_tensor* x, struct ggml_tensor* cs,
                                   struct ggml_tensor* sn) {
  int64_t N = x->ne[0], H = x->ne[1], T = x->ne[2];
  struct ggml_tensor* x4 = ggml_reshape_4d(c, x, 2, N / 2, H, T);
  struct ggml_tensor* x0 = ggml_cont(c, ggml_view_4d(c, x4, 1, N / 2, H, T, x4->nb[1], x4->nb[2], x4->nb[3], 0));
  struct ggml_tensor* x1 =
      ggml_cont(c, ggml_view_4d(c, x4, 1, N / 2, H, T, x4->nb[1], x4->nb[2], x4->nb[3], sizeof(float)));
  struct ggml_tensor* c4 = ggml_reshape_4d(c, cs, 1, N / 2, H, T);
  struct ggml_tensor* s4 = ggml_reshape_4d(c, sn, 1, N / 2, H, T);
  struct ggml_tensor* y0 = ggml_add(c, ggml_mul(c, x0, c4), ggml_mul(c, x1, s4));
  struct ggml_tensor* y1 = ggml_sub(c, ggml_mul(c, x1, c4), ggml_mul(c, x0, s4));
  return ggml_reshape_3d(c, ggml_concat(c, y0, y1, 0), N, H, T);
}

/* one direction of BiSeqMix on u {d, T} (already in scan order) -> out_proj(y * silu(in_proj(nu))) {d, T} */
static struct ggml_tensor* seqdir(gctx* G, const gseq* m, struct ggml_tensor* u, int H, int N) {
  struct ggml_context* c = G->c;
  int d = G->d, P = d / H;
  int64_t T = u->ne[1];
  struct ggml_tensor* nu = ggml_mul(c, ggml_rms_norm(c, u, FLT_EPSILON), m->norm_w);
  struct ggml_tensor* gate = lin_(G, nu, &m->in);
  struct ggml_tensor* k = ggml_reshape_3d(c, lin_(G, nu, &m->k), N, H, T);
  struct ggml_tensor* q = ggml_reshape_3d(c, lin_(G, nu, &m->q), N, H, T);
  struct ggml_tensor* v = ggml_reshape_3d(c, lin_(G, nu, &m->v), P, H, T);
  struct ggml_tensor* st = lin_(G, nu, &m->step);   /* {H, T} raw */
  struct ggml_tensor* rot = ggml_reshape_3d(c, lin_(G, nu, &m->rot), N / 2, H, T);
  struct ggml_tensor* mx = ggml_reshape_3d(c, ggml_sigmoid(c, lin_(G, nu, &m->mix)), 1, H, T);
  k = ggml_add(c, ggml_mul(c, ggml_rms_norm(c, k, FLT_EPSILON), m->kn_w), m->k_bias);
  q = ggml_add(c, ggml_mul(c, ggml_rms_norm(c, q, FLT_EPSILON), m->qn_w), m->q_bias);
  struct ggml_tensor* sp = ggml_reshape_3d(c, ggml_softplus(c, st), 1, H, T);
  /* Theta_t = sum_{s=1..t} rot_s * step_s: angles with t = 0 zeroed, cumulative sum along time */
  struct ggml_tensor* ang = ggml_mul(c, rot, sp);   /* {N/2, H, T} */
  if (T > 1)
    ang = ggml_pad_ext(c, ggml_view_3d(c, ang, N / 2, H, T - 1, ang->nb[1], ang->nb[2], ang->nb[2]), 0, 0, 0, 0, 1, 0,
                       0, 0);
  else
    ang = ggml_scale(c, ang, 0.0f);
  struct ggml_tensor* angT = ggml_cont(c, ggml_permute(c, ang, 1, 2, 0, 3));   /* {T, N/2, H} */
  struct ggml_tensor* th = ggml_cont(c, ggml_permute(c, ggml_cumsum(c, angT), 2, 0, 1, 3));   /* {N/2, H, T} */
  struct ggml_tensor* cs = ggml_cos(c, th);
  struct ggml_tensor* sn = ggml_sin(c, th);
  struct ggml_tensor* kr = rot_neg(c, k, cs, sn);
  struct ggml_tensor* qr = rot_neg(c, q, cs, sn);
  /* inputs of the two scans */
  struct ggml_tensor* dk = ggml_view_2d(c, m->A2, 1, H, m->A2->nb[1], 0);           /* {1, H} decay */
  struct ggml_tensor* a = ggml_exp(c, ggml_mul(c, sp, ggml_reshape_3d(c, dk, 1, H, 1)));   /* {1, H, T} */
  struct ggml_tensor* b2 = ggml_mul(c, ggml_scale_bias(c, mx, -1.0f, 1.0f), a);           /* (1 - mix) a */
  struct ggml_tensor* B1 = ggml_mul(c, kr, mx);
  struct ggml_tensor* B2 = ggml_mul(c, shift_t(c, kr), b2);
  struct ggml_tensor* Bc = ggml_pad_ext(c, ggml_concat(c, B1, B2, 1), 0, NS_PAD - N, 0, 0, 0, 0, 0, 0);
  struct ggml_tensor* Cc = ggml_pad_ext(c, ggml_concat(c, qr, qr, 1), 0, NS_PAD - N, 0, 0, 0, 0, 0, 0);
  struct ggml_tensor* xs = ggml_concat(c, v, shift_t(c, v), 1);   /* {P, 2H, T} */
  struct ggml_tensor* dt = ggml_reshape_3d(c, ggml_concat(c, st, st, 0), 2 * H, T, 1);
  struct ggml_tensor* y = ggml_ssm_scan(c, G->s0, ggml_reshape_4d(c, xs, P, 2 * H, T, 1), dt, m->A2,
                                        ggml_reshape_4d(c, Bc, NS_PAD, 2 * H, T, 1),
                                        ggml_reshape_4d(c, Cc, NS_PAD, 2 * H, T, 1), G->ids, 2);
  size_t es = sizeof(float);
  struct ggml_tensor* y1 = ggml_view_3d(c, y, P, H, T, P * es, (size_t)2 * H * P * es, 0);
  struct ggml_tensor* y2 = ggml_view_3d(c, y, P, H, T, P * es, (size_t)2 * H * P * es, (size_t)H * P * es);
  struct ggml_tensor* ys = ggml_reshape_2d(c, ggml_add(c, y1, y2), d, T);
  return lin_(G, ggml_mul(c, ys, ggml_silu(c, gate)), &m->out);
}

static struct ggml_tensor* run_stack_g(gctx* G, const st_acoustic* a, const stack* s, const gblock* gb,
                                       struct ggml_tensor* x) {
  struct ggml_context* c = G->c;
  int d = a->d, H = d / 32, N = a->state_dim;
  for (int i = 0; i < s->n; ++i) {
    const block* b = &s->l[i];
    const gblock* o = &gb[i];
    struct ggml_tensor* h = adaln(G, x, b->ada1);
    struct ggml_tensor* att = st_g_mha(c, lin_(G, h, &o->qkv), d, a->heads);
    x = ggml_add(c, x, lin_(G, att, &o->o));
    if (b->seq) {
      h = adaln(G, x, b->ada_seq);
      struct ggml_tensor* yf = seqdir(G, &o->f, h, H, N);
      struct ggml_tensor* yb = ggml_get_rows(c, seqdir(G, &o->b, ggml_get_rows(c, h, G->rev), H, N), G->rev);
      x = ggml_add(c, ggml_add(c, x, yf), yb);
    }
    h = adaln(G, x, b->ada2);
    struct ggml_tensor* f = ggml_gelu_erf(c, st_g_conv1d(c, h, o->c1.W, o->c1.b, d, a->k, a->k / 2));
    x = ggml_add(c, x, lin_(G, f, &o->c2));
  }
  return adaln(G, x, s->ada_out);
}

static struct ggml_tensor* varpred_g(gctx* G, const st_acoustic* a, const varpred* v, const gvp* o,
                                     struct ggml_tensor* h) {
  struct ggml_context* c = G->c;
  int d = a->d;
  struct ggml_tensor* u = ggml_relu(c, st_g_conv1d(c, h, o->c1.W, o->c1.b, d, 3, 1));
  u = adaln(G, u, v->ada1);
  u = ggml_relu(c, st_g_conv1d(c, u, o->c2.W, o->c2.b, d, 3, 1));
  u = adaln(G, u, v->ada2);
  return lin_(G, u, &o->out);   /* {1, N} */
}

/* graph inputs shared by encode / decode; T rows */
static int begin(gac* q, const st_acoustic* a, int T, gctx* G, struct ggml_tensor** x) {
  memset(G, 0, sizeof *G);
  G->c = st_gr_begin(&q->r);
  if (!G->c) {
    st_seterr("ggml: cannot start a graph");
    return -1;
  }
  G->d = a->d;
  int H = a->d / 32, P = 32;
  *x = st_gr_input(&q->r, GGML_TYPE_F32, a->d, T, 1);
  G->tab = st_gr_input(&q->r, GGML_TYPE_F32, 2 * a->d, a->n_ada, 1);
  G->rev = st_gr_input(&q->r, GGML_TYPE_I32, T, 1, 1);
  G->s0 = st_gr_input(&q->r, GGML_TYPE_F32, NS_PAD, P, 2 * H);
  G->ids = st_gr_input(&q->r, GGML_TYPE_I32, 1, 1, 1);
  return 0;
}

static int set_common(const st_acoustic* a, const gctx* G, struct ggml_tensor* x, const float* xdata, int T,
                      const float* tab) {
  int H = a->d / 32;
  int* rev = (int*)st_malloc(sizeof(int) * (size_t)T);
  float* z = (float*)st_calloc(sizeof(float) * NS_PAD * 32 * 2 * (size_t)H);
  if (!rev || !z) {
    st_free(rev), st_free(z);
    return -1;
  }
  for (int t = 0; t < T; ++t) rev[t] = T - 1 - t;
  int zero = 0;
  st_gr_set(x, xdata, sizeof(float) * (size_t)T * a->d);
  st_gr_set(G->tab, tab, sizeof(float) * (size_t)a->n_ada * 2 * a->d);
  st_gr_set(G->rev, rev, sizeof(int) * (size_t)T);
  st_gr_set(G->s0, z, sizeof(float) * NS_PAD * 32 * 2 * (size_t)H);
  st_gr_set(G->ids, &zero, sizeof zero);
  st_free(rev), st_free(z);
  return 0;
}

int st_acoustic_gpu_encode(st_acoustic* a, float* h, int N, const float* tab, float* logdur, float* pitch,
                           float* energy) {
  gac* q = (gac*)a->gpu;
  gctx G;
  struct ggml_tensor* x;
  if (begin(q, a, N, &G, &x)) return -1;
  struct ggml_tensor* henc = run_stack_g(&G, a, &a->enc, q->enc, ggml_reshape_2d(G.c, x, a->d, N));
  struct ggml_tensor* ld = varpred_g(&G, a, &a->dur, &q->dur, henc);
  struct ggml_tensor* pt = varpred_g(&G, a, &a->pitch, &q->pitch, henc);
  struct ggml_tensor* en = varpred_g(&G, a, &a->energy, &q->energy, henc);
  st_gr_expand(&q->r, henc);
  st_gr_expand(&q->r, ld);
  st_gr_expand(&q->r, pt);
  st_gr_expand(&q->r, en);
  if (st_gr_alloc(&q->r) || set_common(a, &G, x, h, N, tab) || st_gr_compute(&q->r)) return -1;
  st_gr_get(henc, h, sizeof(float) * (size_t)N * a->d);
  st_gr_get(ld, logdur, sizeof(float) * (size_t)N);
  st_gr_get(pt, pitch, sizeof(float) * (size_t)N);
  st_gr_get(en, energy, sizeof(float) * (size_t)N);
  return 0;
}

int st_acoustic_gpu_decode(st_acoustic* a, const float* xin, int T, const float* tab, float* mel) {
  gac* q = (gac*)a->gpu;
  gctx G;
  struct ggml_tensor* x;
  if (begin(q, a, T, &G, &x)) return -1;
  struct ggml_tensor* y = run_stack_g(&G, a, &a->dec, q->dec, ggml_reshape_2d(G.c, x, a->d, T));
  y = lin_(&G, y, &q->to_mel);   /* {100, T} */
  st_gr_expand(&q->r, y);
  if (st_gr_alloc(&q->r) || set_common(a, &G, x, xin, T, tab) || st_gr_compute(&q->r)) return -1;
  st_gr_get(y, mel, sizeof(float) * (size_t)T * ST_N_MELS);
  return 0;
}

#else

int st_acoustic_gpu_encode(st_acoustic* a, float* h, int N, const float* tab, float* logdur, float* pitch,
                           float* energy) {
  (void)a, (void)h, (void)N, (void)tab, (void)logdur, (void)pitch, (void)energy;
  st_seterr("built without GPU support");
  return -1;
}

int st_acoustic_gpu_decode(st_acoustic* a, const float* x, int T, const float* tab, float* mel) {
  (void)a, (void)x, (void)T, (void)tab, (void)mel;
  st_seterr("built without GPU support");
  return -1;
}

#endif
