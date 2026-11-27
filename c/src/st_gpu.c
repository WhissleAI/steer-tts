/* Optional ggml backend: device selection, weight upload, graph evaluation and the ops the three networks share.
 * Without STEERTTS_GGML only the stubs at the end are compiled. */
#include "st_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef STEERTTS_GGML
#include <math.h>

#include "ggml-cpu.h"
#include "st_ggml.h"

int st_gpu_built(void) { return 1; }

static void quiet_log(enum ggml_log_level level, const char* text, void* user) {
  (void)user;
  if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN || getenv("STEERTTS_GGML_VERBOSE"))
    fputs(text, stderr);
}

static int ieq(const char* a, const char* b) {
  for (; *a && *b; ++a, ++b) {
    char x = *a >= 'A' && *a <= 'Z' ? (char)(*a + 32) : *a, y = *b >= 'A' && *b <= 'Z' ? (char)(*b + 32) : *b;
    if (x != y) return 0;
  }
  return *a == *b;
}

st_gpu* st_gpu_open(const char* device, int threads) {
  if (!device || !*device) device = "gpu";
  /* strict fp32 GEMMs: ggml-cuda asks cuBLAS for TF32 tensor-core math (10-bit mantissa); NVIDIA_TF32_OVERRIDE=0
   * turns that off for the process. STEERTTS_GPU_TF32=1 keeps TF32 (faster, ~1e-3 relative error per GEMM). An
   * application that set NVIDIA_TF32_OVERRIDE itself keeps its value. */
  const char* tf = getenv("STEERTTS_GPU_TF32");
  if (!(tf && *tf && strcmp(tf, "0"))) {
#ifdef _WIN32
    if (!getenv("NVIDIA_TF32_OVERRIDE")) _putenv("NVIDIA_TF32_OVERRIDE=0");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 0);
#endif
  }
  ggml_log_set(quiet_log, NULL);
  ggml_backend_dev_t dev = NULL;
  int want_cpu = ieq(device, "ggml-cpu") || ieq(device, "ggml");
  size_t n = ggml_backend_dev_count();
  for (size_t i = 0; i < n && !dev; ++i) {
    ggml_backend_dev_t d = ggml_backend_dev_get(i);
    enum ggml_backend_dev_type ty = ggml_backend_dev_type(d);
    const char* reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(d));
    if (want_cpu) {
      if (ty == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
    } else if (ieq(device, "gpu") || ieq(device, "auto")) {
      if (ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU) dev = d;
    } else if (ieq(device, ggml_backend_dev_name(d)) ||
               ((ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU) && ieq(device, reg))) {
      dev = d;
    }
  }
  if (!dev) {
    char list[512] = "";
    for (size_t i = 0; i < n; ++i) {
      size_t l = strlen(list);
      snprintf(list + l, sizeof list - l, "%s%s", l ? ", " : "", ggml_backend_dev_name(ggml_backend_dev_get(i)));
    }
    st_seterr("no ggml device matches '%s' (devices: %s)", device, *list ? list : "none");
    return NULL;
  }
  st_gpu* g = (st_gpu*)st_calloc(sizeof *g);
  if (!g) return NULL;
  g->be = ggml_backend_dev_init(dev, NULL);
  if (!g->be) {
    st_seterr("ggml: cannot initialise %s", ggml_backend_dev_name(dev));
    st_free(g);
    return NULL;
  }
  g->is_cpu = ggml_backend_is_cpu(g->be);
  if (g->is_cpu) ggml_backend_cpu_set_n_threads(g->be, threads > 0 ? threads : 1);
  snprintf(g->name, sizeof g->name, "%s (%s)", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
  return g;
}

void st_gpu_close(st_gpu* g) {
  if (!g) return;
  ggml_backend_free(g->be);
  st_free(g);
}

const char* st_gpu_name(const st_gpu* g) { return g ? g->name : "cpu"; }
size_t st_gpu_weight_bytes(const st_gpu* g) { return g ? g->weight_bytes : 0; }

/* ------------------------------------------------------------------ weights */
int st_gw_init(st_gw* w, int max_tensors) {
  memset(w, 0, sizeof *w);
  struct ggml_init_params ip = {ggml_tensor_overhead() * (size_t)max_tensors + 1024, NULL, true};
  w->ctx = ggml_init(ip);
  if (!w->ctx) {
    st_seterr("ggml: cannot create a weight context");
    return -1;
  }
  w->cap = max_tensors;
  w->p = (st_gw_pend*)st_calloc(sizeof(st_gw_pend) * (size_t)max_tensors);
  return w->p ? 0 : -1;
}

static struct ggml_tensor* gw_add(st_gw* w, struct ggml_tensor* t, const st_mat* m, const float* f, float* owned) {
  if (!t || w->n >= w->cap) {
    st_seterr("ggml: weight table full");
    st_free(owned);
    return NULL;
  }
  w->p[w->n++] = (st_gw_pend){t, m, f, owned};
  return t;
}

struct ggml_tensor* st_gw_mat(st_gw* w, const st_mat* m) {
  if (!m) return NULL;
  struct ggml_tensor* t = ggml_new_tensor_2d(w->ctx, GGML_TYPE_F32, m->K, m->O);
  return gw_add(w, t, m, NULL, NULL);
}

struct ggml_tensor* st_gw_f32(st_gw* w, const float* data, int64_t n0, int64_t n1, int64_t n2, int copy) {
  if (!data) return NULL;
  struct ggml_tensor* t = n2 > 1   ? ggml_new_tensor_3d(w->ctx, GGML_TYPE_F32, n0, n1, n2)
                          : n1 > 1 ? ggml_new_tensor_2d(w->ctx, GGML_TYPE_F32, n0, n1)
                                   : ggml_new_tensor_1d(w->ctx, GGML_TYPE_F32, n0);
  float* own = NULL;
  if (copy) {
    size_t nb = sizeof(float) * (size_t)(n0 * n1 * n2);
    own = (float*)st_malloc(nb);
    if (!own) return NULL;
    memcpy(own, data, nb);
    data = own;
  }
  return gw_add(w, t, NULL, data, own);
}

int st_gw_upload(st_gw* w, st_gpu* g) {
  w->buf = ggml_backend_alloc_ctx_tensors(w->ctx, g->be);
  if (!w->buf) {
    st_seterr("ggml: cannot allocate %s weights on %s", "the model's", g->name);
    return -1;
  }
  ggml_backend_buffer_set_usage(w->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g->weight_bytes += ggml_backend_buffer_get_size(w->buf);
  float* tmp = NULL;
  size_t tmp_n = 0;
  for (int i = 0; i < w->n; ++i) {
    st_gw_pend* p = &w->p[i];
    if (p->m) {
      const st_mat* m = p->m;
      if (m->type == ST_W_F32) {
        ggml_backend_tensor_set(p->t, m->w, 0, sizeof(float) * (size_t)m->O * m->K);
      } else {
        size_t need = (size_t)m->O * m->K;
        if (need > tmp_n) {
          st_free(tmp);
          tmp = (float*)st_malloc(sizeof(float) * need);
          tmp_n = tmp ? need : 0;
          if (!tmp) return -1;
        }
        for (int r = 0; r < m->O; ++r) st_mat_row(m, r, tmp + (size_t)r * m->K);
        ggml_backend_tensor_set(p->t, tmp, 0, sizeof(float) * need);
      }
    } else {
      ggml_backend_tensor_set(p->t, p->f, 0, ggml_nbytes(p->t));
    }
    st_free(p->owned);
    p->owned = NULL;
  }
  st_free(tmp);
  st_free(w->p);
  w->p = NULL;
  w->n = w->cap = 0;
  return 0;
}

void st_gw_free(st_gw* w) {
  if (!w) return;
  if (w->p)
    for (int i = 0; i < w->n; ++i) st_free(w->p[i].owned);
  st_free(w->p);
  if (w->buf) ggml_backend_buffer_free(w->buf);
  if (w->ctx) ggml_free(w->ctx);
  memset(w, 0, sizeof *w);
}

/* ------------------------------------------------------------------ graphs */
int st_gr_init(st_gr* r, st_gpu* g, int max_nodes) {
  memset(r, 0, sizeof *r);
  r->g = g;
  r->max_nodes = max_nodes;
  r->meta_size = ggml_tensor_overhead() * (size_t)max_nodes * 2 + ggml_graph_overhead_custom((size_t)max_nodes, false);
  r->meta = st_malloc(r->meta_size);
  r->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g->be));
  if (!r->meta || !r->alloc) {
    st_seterr("ggml: cannot create a graph allocator");
    return -1;
  }
  return 0;
}

struct ggml_context* st_gr_begin(st_gr* r) {
  if (r->ctx) ggml_free(r->ctx);
  struct ggml_init_params ip = {r->meta_size, r->meta, true};
  r->ctx = ggml_init(ip);
  r->gf = r->ctx ? ggml_new_graph_custom(r->ctx, (size_t)r->max_nodes, false) : NULL;
  return r->gf ? r->ctx : NULL;
}

struct ggml_tensor* st_gr_input(st_gr* r, enum ggml_type type, int64_t n0, int64_t n1, int64_t n2) {
  struct ggml_tensor* t = ggml_new_tensor_3d(r->ctx, type, n0, n1, n2);
  ggml_set_input(t);
  return t;
}

void st_gr_expand(st_gr* r, struct ggml_tensor* out) {
  ggml_set_output(out);
  ggml_build_forward_expand(r->gf, out);
}

int st_gr_alloc(st_gr* r) {
  if (!ggml_gallocr_alloc_graph(r->alloc, r->gf)) {
    st_seterr("ggml: cannot allocate the compute buffers on %s (out of device memory?)", r->g->name);
    return -1;
  }
  return 0;
}

int st_gr_compute(st_gr* r) {
  if (ggml_backend_graph_compute(r->g->be, r->gf) != GGML_STATUS_SUCCESS) {
    st_seterr("ggml: graph evaluation failed on %s", r->g->name);
    return -1;
  }
  return 0;
}

void st_gr_free(st_gr* r) {
  if (!r) return;
  if (r->ctx) ggml_free(r->ctx);
  if (r->alloc) ggml_gallocr_free(r->alloc);
  st_free(r->meta);
  memset(r, 0, sizeof *r);
}

void st_gr_set(struct ggml_tensor* t, const void* data, size_t nbytes) { ggml_backend_tensor_set(t, data, 0, nbytes); }
void st_gr_get(const struct ggml_tensor* t, void* data, size_t nbytes) { ggml_backend_tensor_get(t, data, 0, nbytes); }

/* ------------------------------------------------------------------ ops */
struct ggml_tensor* st_g_linear(struct ggml_context* c, struct ggml_tensor* x, struct ggml_tensor* W,
                                struct ggml_tensor* b) {
  struct ggml_tensor* y = ggml_mul_mat(c, W, x);
  return b ? ggml_add(c, y, b) : y;
}

struct ggml_tensor* st_g_layernorm(struct ggml_context* c, struct ggml_tensor* x, struct ggml_tensor* w,
                                   struct ggml_tensor* b, float eps) {
  struct ggml_tensor* y = ggml_norm(c, x, eps);
  if (w) y = ggml_mul(c, y, w);
  if (b) y = ggml_add(c, y, b);
  return y;
}

struct ggml_tensor* st_g_conv1d(struct ggml_context* c, struct ggml_tensor* x, struct ggml_tensor* W,
                                struct ggml_tensor* b, int C, int k, int pad) {
  int64_t T = x->ne[1], O = W->ne[1];
  struct ggml_tensor* xt = ggml_cont(c, ggml_transpose(c, x));   /* {T, C} */
  xt = ggml_reshape_3d(c, xt, T, C, 1);
  struct ggml_tensor* kern = ggml_reshape_3d(c, W, k, C, O);     /* shape only */
  struct ggml_tensor* col = ggml_im2col(c, kern, xt, 1, 0, pad, 0, 1, 0, false, GGML_TYPE_F32);   /* {C*k, T', 1} */
  col = ggml_reshape_2d(c, col, (int64_t)C * k, col->ne[1]);
  return st_g_linear(c, col, W, b);
}

struct ggml_tensor* st_g_dwconv(struct ggml_context* c, struct ggml_tensor* x, struct ggml_tensor* w,
                                struct ggml_tensor* b, int k, int dil) {
  /* A dilated depthwise conv is `dil` interleaved dense ones: split time into phases t = m * dil + ph, run ggml's
   * depthwise (causal, time-contiguous) ssm_conv on each phase as a separate sequence over the padded input, and
   * interleave the outputs back. */
  int64_t C = x->ne[0], T = x->ne[1], r = dil, half = (int64_t)(k / 2) * r;
  int64_t L0 = T + 2 * half, e = (r - L0 % r) % r, L = L0 + e, M = L / r;
  struct ggml_tensor* xp = ggml_pad_ext(c, x, 0, 0, (int)half, (int)(half + e), 0, 0, 0, 0);   /* {C, L} */
  xp = ggml_reshape_3d(c, xp, C, r, M);
  struct ggml_tensor* sx = ggml_cont(c, ggml_permute(c, xp, 1, 2, 0, 3));   /* {M, C, r} */
  struct ggml_tensor* y = ggml_ssm_conv(c, sx, w);                          /* {C, M - k + 1, r} */
  if (r > 1) y = ggml_cont(c, ggml_permute(c, y, 0, 2, 1, 3));              /* {C, r, M - k + 1} */
  y = ggml_reshape_2d(c, y, C, r * (M - k + 1));
  if (y->ne[1] != T) y = ggml_view_2d(c, y, C, T, y->nb[1], 0);
  return b ? ggml_add(c, y, b) : y;
}

struct ggml_tensor* st_g_mha(struct ggml_context* c, struct ggml_tensor* qkv, int d, int heads) {
  int64_t T = qkv->ne[1], dh = d / heads;
  size_t es = ggml_element_size(qkv);
  struct ggml_tensor* q = ggml_view_3d(c, qkv, dh, heads, T, dh * es, qkv->nb[1], 0);
  struct ggml_tensor* k = ggml_view_3d(c, qkv, dh, heads, T, dh * es, qkv->nb[1], (size_t)d * es);
  struct ggml_tensor* v = ggml_view_3d(c, qkv, dh, heads, T, dh * es, qkv->nb[1], (size_t)2 * d * es);
  q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));    /* {dh, T, H} */
  k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
  struct ggml_tensor* vt = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));   /* {T, dh, H} */
  float scale = 1.0f / sqrtf((float)dh);
  /* scores {Tk, Tq, H} in fp32; long inputs are split over the queries so a chunk stays under ~128 MB */
  int64_t qc = T, budget = (int64_t)32 << 20;   /* floats */
  if (T * T * heads > budget) qc = budget / (T * heads) > 64 ? budget / (T * heads) : 64;
  struct ggml_tensor* out = NULL;
  for (int64_t q0 = 0; q0 < T; q0 += qc) {
    int64_t nq = T - q0 < qc ? T - q0 : qc;
    struct ggml_tensor* qq = qc == T ? q : ggml_view_3d(c, q, dh, nq, heads, q->nb[1], q->nb[2], (size_t)q0 * q->nb[1]);
    struct ggml_tensor* kq = ggml_mul_mat(c, k, qq);   /* {Tk, nq, H} */
    kq = ggml_soft_max_ext(c, kq, NULL, scale, 0.0f);
    struct ggml_tensor* o = ggml_mul_mat(c, vt, kq);   /* {dh, nq, H} */
    o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));   /* {dh, H, nq} */
    o = ggml_reshape_2d(c, o, d, nq);
    out = out ? ggml_concat(c, out, o, 1) : o;
  }
  return out;
}

#else /* !STEERTTS_GGML */

int st_gpu_built(void) { return 0; }
st_gpu* st_gpu_open(const char* device, int threads) {
  (void)threads;
  st_seterr("device '%s': this build has no GPU support (rebuild with -DSTEERTTS_GGML=ON)", device ? device : "");
  return NULL;
}
void st_gpu_close(st_gpu* g) { (void)g; }
const char* st_gpu_name(const st_gpu* g) {
  (void)g;
  return "cpu";
}
size_t st_gpu_weight_bytes(const st_gpu* g) {
  (void)g;
  return 0;
}
int st_vocoder_gpu_attach(struct st_vocoder* v, st_gpu* g) {
  (void)v, (void)g;
  st_seterr("built without GPU support");
  return -1;
}
int st_flow_gpu_attach(struct st_flow* f, st_gpu* g) {
  (void)f, (void)g;
  st_seterr("built without GPU support");
  return -1;
}
int st_acoustic_gpu_attach(struct st_acoustic* a, st_gpu* g) {
  (void)a, (void)g;
  st_seterr("built without GPU support");
  return -1;
}
void st_vocoder_gpu_detach(struct st_vocoder* v) { (void)v; }
void st_flow_gpu_detach(struct st_flow* f) { (void)f; }
void st_acoustic_gpu_detach(struct st_acoustic* a) { (void)a; }

#endif
