/* ggml helpers shared by the GPU graphs (st_gpu_*.c). Only compiled with STEERTTS_GGML.
 *
 * Layout: an activation [T, C] (time-major, as everywhere in the C runtime) is a ggml tensor with ne = {C, T}, so the
 * bytes are the same; a weight st_mat [O, K] is ne = {K, O} and ggml_mul_mat(W, x) gives ne = {O, T}. */
#ifndef ST_GGML_H
#define ST_GGML_H
#ifdef STEERTTS_GGML
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "st_base.h"
#include "st_gpu.h"

struct st_gpu {
  ggml_backend_t be;
  char name[192];
  size_t weight_bytes;
  int is_cpu;
};

/* ------------------------------------------------------------------ weights on the device */
typedef struct {
  struct ggml_tensor* t;
  const st_mat* m;   /* source matrix (dequantized row by row), or */
  const float* f;    /* source floats (nelements of t) */
  float* owned;      /* f owned by the store (freed after the upload) */
} st_gw_pend;

typedef struct {
  struct ggml_context* ctx;
  ggml_backend_buffer_t buf;
  st_gw_pend* p;
  int n, cap;
} st_gw;

int st_gw_init(st_gw* w, int max_tensors);
/* matrix [O, K] -> ne {K, O} f32 */
struct ggml_tensor* st_gw_mat(st_gw* w, const st_mat* m);
/* floats -> ne {n0, n1, n2} f32 (n1, n2 may be 1); copy = 1 copies the data now (caller may free it) */
struct ggml_tensor* st_gw_f32(st_gw* w, const float* data, int64_t n0, int64_t n1, int64_t n2, int copy);
int st_gw_upload(st_gw* w, st_gpu* g);   /* allocate on the device, copy, drop the host copies */
void st_gw_free(st_gw* w);

/* ------------------------------------------------------------------ one graph evaluation */
typedef struct {
  st_gpu* g;
  ggml_gallocr_t alloc;
  struct ggml_context* ctx;
  struct ggml_cgraph* gf;
  void* meta;
  size_t meta_size;
  int max_nodes;
} st_gr;

int st_gr_init(st_gr* r, st_gpu* g, int max_nodes);
struct ggml_context* st_gr_begin(st_gr* r);     /* fresh context for building a graph */
struct ggml_tensor* st_gr_input(st_gr* r, enum ggml_type type, int64_t n0, int64_t n1, int64_t n2);
/* expands the graph from `out` (may be called for several outputs), then allocates and runs it */
void st_gr_expand(st_gr* r, struct ggml_tensor* out);
int st_gr_alloc(st_gr* r);                       /* after the graph is complete; then set inputs */
int st_gr_compute(st_gr* r);
void st_gr_free(st_gr* r);
void st_gr_set(struct ggml_tensor* t, const void* data, size_t nbytes);
void st_gr_get(const struct ggml_tensor* t, void* data, size_t nbytes);

/* ------------------------------------------------------------------ ops on time-major activations */
/* y = W x + b (b may be NULL) */
struct ggml_tensor* st_g_linear(struct ggml_context* c, struct ggml_tensor* x, struct ggml_tensor* W,
                                struct ggml_tensor* b);
/* LayerNorm over C with optional affine */
struct ggml_tensor* st_g_layernorm(struct ggml_context* c, struct ggml_tensor* x, struct ggml_tensor* w,
                                   struct ggml_tensor* b, float eps);
/* dense Conv1d, stride 1: x {C, T}, W {C*k, O} (torch [O, C, k] flattened), symmetric padding pad, dilation 1 */
struct ggml_tensor* st_g_conv1d(struct ggml_context* c, struct ggml_tensor* x, struct ggml_tensor* W,
                                struct ggml_tensor* b, int C, int k, int pad);
/* depthwise Conv1d k = 7 (any odd k ggml's ssm_conv supports), "same" padding, dilation dil: w {k, C} */
struct ggml_tensor* st_g_dwconv(struct ggml_context* c, struct ggml_tensor* x, struct ggml_tensor* w,
                                struct ggml_tensor* b, int k, int dil);
/* multi-head self-attention, batch 1, no mask: qkv {3d, T} (q | k | v) -> {d, T} */
struct ggml_tensor* st_g_mha(struct ggml_context* c, struct ggml_tensor* qkv, int d, int heads);

#endif
#endif
