/* Flow refiner / prompted flow on a ggml device: the whole midpoint ODE (2 * steps network evaluations, plus the
 * unconditional ones under prompt CFG) is one graph, so x stays on the device from the noise to the result. Everything
 * that does not depend on x (time / g / context conditioning, the constant part of the input conv, the noise) is
 * computed by st_flow_sample on the CPU exactly as before and uploaded. Network: see st_flow.c (FlowNet). */
#include "st_flow_impl.h"
#include "st_gpu.h"

#ifdef STEERTTS_GGML
#include <string.h>

#include "st_ggml.h"

typedef struct {
  struct ggml_tensor *dw, *dwb, *pw1, *pw1b, *pw2, *pw2b;
  struct ggml_tensor *in, *inb, *out, *outb;   /* attention (prompted flow) */
} gfblk;

typedef struct {
  st_gpu* g;
  st_gw w;
  st_gr r;
  int r_nodes;
  struct ggml_tensor *inp_x, *outw, *outb;
  gfblk blk[MAXL];
} gflow;

static void gflow_free(gflow* q) {
  if (!q) return;
  st_gr_free(&q->r);
  st_gw_free(&q->w);
  st_free(q);
}

int st_flow_gpu_attach(st_flow* f, st_gpu* g) {
  if (!f || !g) return -1;
  st_flow_gpu_detach(f);
  gflow* q = (gflow*)st_calloc(sizeof *q);
  float* dw = NULL;
  if (!q) return -1;
  q->g = g;
  int d = f->d, L = f->n_layers;
  if (d % 128) {   /* ggml-cuda's depthwise conv works on 128-channel blocks */
    st_seterr("flow: d = %d is not a multiple of 128", d);
    goto fail;
  }
  if (st_gw_init(&q->w, 8 + 10 * L)) goto fail;
  st_gw* w = &q->w;
  q->inp_x = st_gw_mat(w, &f->inp_x);
  q->outw = st_gw_mat(w, f->outw);
  q->outb = st_gw_f32(w, f->outb, NMEL, 1, 1, 0);
  if (!q->inp_x || !q->outw || !q->outb) goto fail;
  dw = (float*)st_malloc(sizeof(float) * KW * (size_t)d);
  if (!dw) goto fail;
  for (int l = 0; l < L; ++l) {
    const fblock* b = &f->blk[l];
    gfblk* o = &q->blk[l];
    for (int c = 0; c < d; ++c)   /* back to torch's [C, 1, 7] = ne {7, C} */
      for (int k = 0; k < KW; ++k) dw[c * KW + k] = b->wT[k * d + c];
    o->dw = st_gw_f32(w, dw, KW, d, 1, 1);
    o->dwb = st_gw_f32(w, b->dwb, d, 1, 1, 0);
    o->pw1 = st_gw_mat(w, b->pw1);
    o->pw1b = st_gw_f32(w, b->pw1b, 3 * d, 1, 1, 0);
    o->pw2 = st_gw_mat(w, b->pw2);
    o->pw2b = st_gw_f32(w, b->pw2b, d, 1, 1, 0);
    if (!o->dw || !o->dwb || !o->pw1 || !o->pw1b || !o->pw2 || !o->pw2b) goto fail;
    if (f->has_att[l]) {
      const fattn* a = &f->att[l];
      o->in = st_gw_mat(w, a->in);
      o->inb = st_gw_f32(w, a->inb, 3 * d, 1, 1, 0);
      o->out = st_gw_mat(w, a->out);
      o->outb = st_gw_f32(w, a->outb, d, 1, 1, 0);
      if (!o->in || !o->inb || !o->out || !o->outb) goto fail;
    }
  }
  if (st_gw_upload(w, g)) goto fail;
  st_free(dw);
  f->gpu = q;
  return 0;
fail:
  st_free(dw);
  gflow_free(q);
  return -1;
}

void st_flow_gpu_detach(st_flow* f) {
  if (!f || !f->gpu) return;
  gflow_free((gflow*)f->gpu);
  f->gpu = NULL;
}

/* one row of a modulation table: [3d] at layer l, evaluation e -> scale, shift, gate views */
static struct ggml_tensor* mrow(struct ggml_context* c, struct ggml_tensor* tab, int l, int e, int ne, int d, int part) {
  size_t off = (((size_t)l * ne + e) * 3 + part) * (size_t)d * sizeof(float);
  return ggml_view_1d(c, tab, d, off);
}

/* LayerNorm (eps 1e-5, no affine) then x * (1 + scale) + shift (the table holds 1 + scale) */
static struct ggml_tensor* ln_mod(struct ggml_context* c, struct ggml_tensor* x, struct ggml_tensor* tab, int l, int e,
                                  int ne, int d) {
  x = ggml_norm(c, x, 1e-5f);
  x = ggml_mul(c, x, mrow(c, tab, l, e, ne, d, 0));
  return ggml_add(c, x, mrow(c, tab, l, e, ne, d, 1));
}

static struct ggml_tensor* net(const st_flow* f, gflow* q, struct ggml_context* c, struct ggml_tensor* x,
                               struct ggml_tensor* hc, struct ggml_tensor* bm, struct ggml_tensor* am, int e, int ne) {
  int d = f->d;
  struct ggml_tensor* h = ggml_add(c, hc, st_g_conv1d(c, x, q->inp_x, NULL, NMEL, 3, 1));
  for (int l = 0; l < f->n_layers; ++l) {
    const gfblk* b = &q->blk[l];
    struct ggml_tensor* hn = st_g_dwconv(c, h, b->dw, b->dwb, KW, f->blk[l].dil);
    hn = ln_mod(c, hn, bm, l, e, ne, d);
    struct ggml_tensor* u = ggml_gelu_erf(c, st_g_linear(c, hn, b->pw1, b->pw1b));
    struct ggml_tensor* v = st_g_linear(c, u, b->pw2, b->pw2b);
    h = ggml_add(c, h, ggml_mul(c, v, mrow(c, bm, l, e, ne, d, 2)));
    if (f->has_att[l]) {
      hn = ln_mod(c, h, am, l, e, ne, d);
      struct ggml_tensor* qkv = st_g_linear(c, hn, b->in, b->inb);
      struct ggml_tensor* att = st_g_mha(c, qkv, d, HEADS);
      v = st_g_linear(c, att, b->out, b->outb);
      h = ggml_add(c, h, ggml_mul(c, v, mrow(c, am, l, e, ne, d, 2)));
    }
  }
  return st_g_linear(c, ggml_norm(c, h, 1e-5f), q->outw, q->outb);
}

/* the tables with scale -> 1 + scale (the CPU kernels add the 1 per element; same float result) */
static float* one_plus(const float* tab, int L, int ne, int d) {
  size_t n = (size_t)L * ne * 3 * d;
  float* t = (float*)st_malloc(sizeof(float) * n);
  if (!t) return NULL;
  memcpy(t, tab, sizeof(float) * n);
  for (size_t r = 0; r < (size_t)L * ne; ++r)
    for (int i = 0; i < d; ++i) t[r * 3 * d + i] = 1.f + t[r * 3 * d + i];
  return t;
}

int st_flow_gpu_run(st_flow* f, float* x, int N, const float* hc, const float* bmod, const float* amod, const float* hcu,
                    const float* bmodu, const float* amodu, int guided, float cfg, const float* ts, int steps) {
  gflow* q = (gflow*)f->gpu;
  int d = f->d, L = f->n_layers, ne = 2 * steps, n_att = 0, rc = -1;
  for (int l = 0; l < L; ++l) n_att += f->has_att[l];
  int nodes = ne * (guided ? 2 : 1) * (40 * L + 60 * n_att + 64) + 256;
  if (nodes > q->r_nodes) {
    st_gr_free(&q->r);
    q->r_nodes = 0;
    if (st_gr_init(&q->r, q->g, nodes)) return -1;
    q->r_nodes = nodes;
  }
  float *tb = NULL, *ta = NULL, *tbu = NULL, *tau = NULL;
  struct ggml_context* c = st_gr_begin(&q->r);
  if (!c) {
    st_seterr("ggml: cannot start a graph");
    return -1;
  }
  struct ggml_tensor* x0 = st_gr_input(&q->r, GGML_TYPE_F32, NMEL, N, 1);
  struct ggml_tensor* ghc = st_gr_input(&q->r, GGML_TYPE_F32, d, N, 1);
  struct ggml_tensor* bm = st_gr_input(&q->r, GGML_TYPE_F32, 3 * d, ne, L);
  struct ggml_tensor* am = n_att ? st_gr_input(&q->r, GGML_TYPE_F32, 3 * d, ne, L) : NULL;
  struct ggml_tensor *ghcu = NULL, *bmu = NULL, *amu = NULL;
  if (guided) {
    ghcu = st_gr_input(&q->r, GGML_TYPE_F32, d, N, 1);
    bmu = st_gr_input(&q->r, GGML_TYPE_F32, 3 * d, ne, L);
    amu = n_att ? st_gr_input(&q->r, GGML_TYPE_F32, 3 * d, ne, L) : NULL;
  }
  struct ggml_tensor* xc = ggml_reshape_2d(c, x0, NMEL, N);
  for (int i = 0; i < steps; ++i) {
    float dt = ts[i + 1] - ts[i], hdt = 0.5f * dt;
    struct ggml_tensor* xm = xc;
    for (int s = 0; s < 2; ++s) {
      int e = 2 * i + s;
      struct ggml_tensor* xin = s == 0 ? xc : xm;
      struct ggml_tensor* v = net(f, q, c, xin, ggml_reshape_2d(c, ghc, d, N), bm, am, e, ne);
      if (guided) {
        struct ggml_tensor* vu = net(f, q, c, xin, ggml_reshape_2d(c, ghcu, d, N), bmu, amu, e, ne);
        v = ggml_add(c, v, ggml_scale(c, ggml_sub(c, v, vu), cfg));   /* v + cfg * (v - vu) */
      }
      if (s == 0) xm = ggml_add(c, xc, ggml_scale(c, v, hdt));
      else xc = ggml_add(c, xc, ggml_scale(c, v, dt));
    }
  }
  st_gr_expand(&q->r, xc);
  if (st_gr_alloc(&q->r)) return -1;
  size_t tn = (size_t)L * ne * 3 * d;
  if (!(tb = one_plus(bmod, L, ne, d)) || (n_att && !(ta = one_plus(amod, L, ne, d)))) goto done;
  if (guided && (!(tbu = one_plus(bmodu, L, ne, d)) || (n_att && !(tau = one_plus(amodu, L, ne, d))))) goto done;
  st_gr_set(x0, x, sizeof(float) * (size_t)N * NMEL);
  st_gr_set(ghc, hc, sizeof(float) * (size_t)N * d);
  st_gr_set(bm, tb, sizeof(float) * tn);
  if (am) st_gr_set(am, ta, sizeof(float) * tn);
  if (guided) {
    st_gr_set(ghcu, hcu, sizeof(float) * (size_t)N * d);
    st_gr_set(bmu, tbu, sizeof(float) * tn);
    if (amu) st_gr_set(amu, tau, sizeof(float) * tn);
  }
  if (st_gr_compute(&q->r)) goto done;
  st_gr_get(xc, x, sizeof(float) * (size_t)N * NMEL);
  rc = 0;
done:
  st_free(tb), st_free(ta), st_free(tbu), st_free(tau);
  return rc;
}

#else

int st_flow_gpu_run(st_flow* f, float* x, int N, const float* hc, const float* bmod, const float* amod, const float* hcu,
                    const float* bmodu, const float* amodu, int guided, float cfg, const float* ts, int steps) {
  (void)f, (void)x, (void)N, (void)hc, (void)bmod, (void)amod, (void)hcu, (void)bmodu, (void)amodu, (void)guided;
  (void)cfg, (void)ts, (void)steps;
  st_seterr("built without GPU support");
  return -1;
}

#endif
