/* Vocos backbone + ISTFTHead spectra as one ggml graph (the iSTFT stays on the CPU, st_dsp.c). Same network as
 * st_vocoder.c: x = LN(embed(mel)); per block x += gamma * pw2(gelu(pw1(LN(dwconv7(x))))); x = LN(x);
 * [log|mag|, phase] = out(x); re / im = min(exp(log|mag|), 100) * (cos, sin)(phase). */
#include "st_gpu.h"
#include "st_vocoder_impl.h"

#ifdef STEERTTS_GGML
#include <math.h>

#include "st_ggml.h"

typedef struct {
  struct ggml_tensor *dw, *dw_b, *ln_w, *ln_b, *gamma, *w1, *b1, *w2, *b2;
} gblk;

typedef struct {
  st_gpu* g;
  st_gw w;
  st_gr r;
  struct ggml_tensor *embed, *embed_b, *ln_w, *ln_b, *fln_w, *fln_b, *out, *out_b;
  gblk* blk;
} gvoc;

static void gvoc_free(gvoc* q) {
  if (!q) return;
  st_gr_free(&q->r);
  st_gw_free(&q->w);
  st_free(q->blk);
  st_free(q);
}

int st_vocoder_gpu_attach(st_vocoder* v, st_gpu* g) {
  if (!v || !g) return -1;
  st_vocoder_gpu_detach(v);
  gvoc* q = (gvoc*)st_calloc(sizeof *q);
  if (!q) return -1;
  q->g = g;
  int D = v->dim;
  q->blk = (gblk*)st_calloc(sizeof(gblk) * (size_t)v->layers);
  if (!q->blk || st_gw_init(&q->w, 16 + 9 * v->layers) || st_gr_init(&q->r, g, 512 + 40 * v->layers)) goto fail;
  st_gw* w = &q->w;
  q->embed = st_gw_mat(w, v->embed);
  q->embed_b = st_gw_f32(w, v->embed_b, D, 1, 1, 0);
  q->ln_w = st_gw_f32(w, v->ln_w, D, 1, 1, 0);
  q->ln_b = st_gw_f32(w, v->ln_b, D, 1, 1, 0);
  q->fln_w = st_gw_f32(w, v->fln_w, D, 1, 1, 0);
  q->fln_b = st_gw_f32(w, v->fln_b, D, 1, 1, 0);
  q->out = st_gw_mat(w, v->out);
  q->out_b = st_gw_f32(w, v->out_b, v->n_fft + 2, 1, 1, 0);
  if (!q->embed || !q->embed_b || !q->ln_w || !q->ln_b || !q->fln_w || !q->fln_b || !q->out || !q->out_b) goto fail;
  for (int i = 0; i < v->layers; ++i) {
    const cnx* b = &v->blk[i];
    gblk* o = &q->blk[i];
    o->dw = st_gw_f32(w, b->dw_w, 7, D, 1, 0);   /* torch [C, 1, 7] = ne {7, C} */
    o->dw_b = st_gw_f32(w, b->dw_b, D, 1, 1, 0);
    o->ln_w = st_gw_f32(w, b->ln_w, D, 1, 1, 0);
    o->ln_b = st_gw_f32(w, b->ln_b, D, 1, 1, 0);
    o->gamma = st_gw_f32(w, b->gamma, D, 1, 1, 0);
    o->w1 = st_gw_mat(w, b->w1);
    o->b1 = st_gw_f32(w, b->b1, v->inter, 1, 1, 0);
    o->w2 = st_gw_mat(w, b->w2);
    o->b2 = st_gw_f32(w, b->b2, D, 1, 1, 0);
    if (!o->dw || !o->dw_b || !o->ln_w || !o->ln_b || !o->gamma || !o->w1 || !o->b1 || !o->w2 || !o->b2) goto fail;
  }
  if (st_gw_upload(w, g)) goto fail;
  v->gpu = q;
  return 0;
fail:
  gvoc_free(q);
  return -1;
}

void st_vocoder_gpu_detach(st_vocoder* v) {
  if (!v || !v->gpu) return;
  gvoc_free((gvoc*)v->gpu);
  v->gpu = NULL;
}

int st_vocoder_gpu_spectra(st_vocoder* v, const float* mel, int T, float* ri) {
  gvoc* q = (gvoc*)v->gpu;
  int nb = v->n_fft / 2 + 1;
  struct ggml_context* c = st_gr_begin(&q->r);
  if (!c) {
    st_seterr("ggml: cannot start a graph");
    return -1;
  }
  struct ggml_tensor* in = st_gr_input(&q->r, GGML_TYPE_F32, N_MELS, T, 1);
  struct ggml_tensor* x = st_g_conv1d(c, ggml_reshape_2d(c, in, N_MELS, T), q->embed, q->embed_b, N_MELS, 7, 3);
  x = st_g_layernorm(c, x, q->ln_w, q->ln_b, 1e-6f);
  for (int i = 0; i < v->layers; ++i) {
    const gblk* b = &q->blk[i];
    struct ggml_tensor* y = st_g_dwconv(c, x, b->dw, b->dw_b, 7, 1);
    y = st_g_layernorm(c, y, b->ln_w, b->ln_b, 1e-6f);
    y = ggml_gelu_erf(c, st_g_linear(c, y, b->w1, b->b1));
    y = st_g_linear(c, y, b->w2, b->b2);
    x = ggml_add(c, x, ggml_mul(c, y, b->gamma));
  }
  x = st_g_layernorm(c, x, q->fln_w, q->fln_b, 1e-6f);
  struct ggml_tensor* h = st_g_linear(c, x, q->out, q->out_b);   /* {n_fft + 2, T} */
  struct ggml_tensor* lm = ggml_cont(c, ggml_view_2d(c, h, nb, T, h->nb[1], 0));
  struct ggml_tensor* ph = ggml_cont(c, ggml_view_2d(c, h, nb, T, h->nb[1], (size_t)nb * sizeof(float)));
  struct ggml_tensor* mag = ggml_clamp(c, ggml_exp(c, lm), -INFINITY, 1e2f);
  struct ggml_tensor* out = ggml_concat(c, ggml_mul(c, mag, ggml_cos(c, ph)), ggml_mul(c, mag, ggml_sin(c, ph)), 1);
  st_gr_expand(&q->r, out);
  if (st_gr_alloc(&q->r)) return -1;
  st_gr_set(in, mel, sizeof(float) * (size_t)T * N_MELS);
  if (st_gr_compute(&q->r)) return -1;
  st_gr_get(out, ri, sizeof(float) * (size_t)T * nb * 2);
  return 0;
}

#else

int st_vocoder_gpu_spectra(st_vocoder* v, const float* mel, int T, float* ri) {
  (void)v, (void)mel, (void)T, (void)ri;
  st_seterr("built without GPU support");
  return -1;
}

#endif
