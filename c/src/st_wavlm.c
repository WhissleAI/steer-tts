/* WavLM-base-plus-sv x-vector forward (transformers 5.x WavLMForXVector, eval, batch 1, no padding).
 *
 *   wav16 -> 7 strided convs (no bias; GroupNorm(512 groups) after the first; GELU)            [T, 512], T ~ n / 320
 *         -> LayerNorm + Linear 512 -> 768
 *         -> + GELU(grouped pos conv k 128, 16 groups, pad 64, last frame dropped) -> LayerNorm    = hidden state 0
 *         -> 12 post-LN layers: x = LN(x + MHA(x, bias = gate(x) * relbias)); x = LN(x + FF(x))  = hidden states 1..12
 *         -> softmax(layer_weights)-weighted sum of the 13 hidden states -> projector 768 -> 512
 *         -> TDNN (Linear over k frames, dilated, ReLU) x 5 -> [mean, unbiased std] over time -> Linear 3000 -> 512
 *
 * Relative position bias: bucket(j - i) as in WavLMAttention._relative_positions_bucket (computed once from layer 0's
 * rel_attn_embed); every layer gates it per (head, query) with gru_rel_pos_linear on its own input.
 *
 * Convolutions run as one GEMM on the activations in place: a conv weight [O, C, k] is re-laid out at load to
 * [O, k, C], so the receptive field of output frame t is the contiguous slice x[t*stride .. t*stride + k) of the
 * time-major input (row stride stride*C, K = k*C) and no im2col copy is needed. The grouped positional conv does the
 * same on a group-major copy of its padded input. */
#include "st_wavlm.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "st_audio.h"

#define WL_MAXCONV 8
#define WL_MAXTDNN 8
#define WL_MAXLAYER 24
#define WL_SR 16000
#define WL_SR24 24000
#define WL_MAX_S 8
#define WL_MIN_N 8000

typedef struct {
  const st_mat *q, *k, *v, *o, *ff1, *ff2;
  const float *qb, *kb, *vb, *ob, *ff1b, *ff2b;
  const float *ln1w, *ln1b, *ln2w, *ln2b;
  const float *gw, *gb, *gconst;   /* gru_rel_pos_linear [8, dh], [8]; gru_rel_pos_const [H] */
} wl_layer;

/* a weight matrix re-laid out from [O, C, k] to [O, k, C] (owned storage; int8 scales point into st_weights) */
typedef struct {
  st_mat m;
  void* data;
} wl_mat;

struct st_wavlm {
  int nconv, conv_dim[WL_MAXCONV], conv_k[WL_MAXCONV], conv_s[WL_MAXCONV];
  int D, H, dh, nlayer, inter, pos_k, pos_g, nbucket, maxdist;
  int ntdnn, tdnn_dim[WL_MAXTDNN], tdnn_k[WL_MAXTDNN], tdnn_d[WL_MAXTDNN], xdim;
  float eps;
  const st_mat* conv0;               /* [512, 1*k0]: already [O, k, C] for C = 1 */
  wl_mat conv[WL_MAXCONV];           /* layers 1.. */
  const float *gn_w, *gn_b;
  const float *fp_lnw, *fp_lnb, *fp_b;
  const st_mat* fp;
  wl_mat pos;                        /* [D, pos_k, D / pos_g] */
  const float *pos_b, *enc_lnw, *enc_lnb;
  const float* relemb;               /* [nbucket, H] */
  wl_layer layer[WL_MAXLAYER];
  float lw[WL_MAXLAYER + 1];         /* softmax(layer_weights) */
  const st_mat *proj, *tdnn[WL_MAXTDNN], *fe;
  const float *proj_b, *tdnn_b[WL_MAXTDNN], *fe_b;
};

/* ------------------------------------------------------------------ local kernels
 * GELU and exp dominate the elementwise time (~41 M GELUs per 8 s clip: the CNN outputs and the FF layers; ~23 M
 * softmax exps), and libm's erff / expf do not vectorise without -ffast-math. On x86-64 with AVX2+FMA they run here as
 * 8-wide polynomials: exp as Cephes expf (range reduction by ln 2, degree-6 polynomial, ~1 ulp) and
 * GELU(x) = x/2 * (1 + erf(x/sqrt2)) with erfc from the Numerical Recipes Chebyshev fit (fractional error < 1.2e-7
 * everywhere), so 1 + erf is accurate to ~1e-7 relative on both sides. Elsewhere: libm. */
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__)
#define WL_X86 1
#include <immintrin.h>

__attribute__((target("avx2,fma"))) static inline __m256 exp8(__m256 x) {
  x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
  x = _mm256_min_ps(x, _mm256_set1_ps(88.3f));
  __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
  r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
  __m256 p = _mm256_set1_ps(1.9875691500e-4f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
  p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.f)));
  /* 2^n for n in [-126, 127]: below that the clamped input already gives ~1e-38 */
  __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

__attribute__((target("avx2,fma"))) static void gelu_avx2(float* x, size_t n) {
  const __m256 half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.f), two = _mm256_set1_ps(2.f);
  const __m256 rs2 = _mm256_set1_ps(0.70710678118654752f), sign = _mm256_set1_ps(-0.f);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(x + i);
    __m256 z = _mm256_andnot_ps(sign, _mm256_mul_ps(v, rs2));
    __m256 t = _mm256_div_ps(one, _mm256_fmadd_ps(half, z, one));
    __m256 p = _mm256_set1_ps(0.17087277f);
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.82215223f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.48851587f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-1.13520398f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.27886807f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.18628806f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.09678418f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.37409196f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.00002368f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-1.26551223f));
    __m256 erfc = _mm256_mul_ps(t, exp8(_mm256_fnmadd_ps(z, z, p)));
    /* 1 + erf(v / sqrt2) = 2 - erfc(z) for v >= 0, erfc(z) for v < 0 */
    __m256 neg = _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_LT_OQ);
    __m256 s = _mm256_blendv_ps(_mm256_sub_ps(two, erfc), erfc, neg);
    _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(half, v), s));
  }
  if (i < n) st_gelu(x + i, n - i);
}

/* x[i] = exp(x[i] - mx), returns the sum */
__attribute__((target("avx2,fma"))) static float exp_sum_avx2(float* x, int n, float mx) {
  __m256 m = _mm256_set1_ps(mx), acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 e = exp8(_mm256_sub_ps(_mm256_loadu_ps(x + i), m));
    _mm256_storeu_ps(x + i, e);
    acc = _mm256_add_ps(acc, e);
  }
  float b[8], s = 0;
  _mm256_storeu_ps(b, acc);
  for (int k = 0; k < 8; ++k) s += b[k];
  for (; i < n; ++i) s += (x[i] = expf(x[i] - mx));
  return s;
}

static int wl_simd(void) { return st_cpu_avx2(); }   /* cached atomically there */
#endif

static void gelu(float* x, size_t n) {
#ifdef WL_X86
  if (wl_simd()) {
    gelu_avx2(x, n);
    return;
  }
#endif
  st_gelu(x, n);
}

static float exp_sum(float* x, int n, float mx) {
#ifdef WL_X86
  if (wl_simd()) return exp_sum_avx2(x, n, mx);
#endif
  float s = 0;
  for (int i = 0; i < n; ++i) s += (x[i] = expf(x[i] - mx));
  return s;
}

/* ------------------------------------------------------------------ load */
static int relayout(const st_mat* src, int C, int k, wl_mat* dst) {
  int O = src->O;
  size_t K = (size_t)C * k;
  memset(dst, 0, sizeof *dst);
  dst->m = *src;
  if (src->type == ST_W_I8) {
    int8_t* q = (int8_t*)st_malloc((size_t)O * K);
    if (!q) return -1;
    for (int o = 0; o < O; ++o)
      for (int c = 0; c < C; ++c)
        for (int j = 0; j < k; ++j) q[(size_t)o * K + (size_t)j * C + c] = src->q8[(size_t)o * K + (size_t)c * k + j];
    dst->m.q8 = q;
    dst->data = q;
    return 0;
  }
  float* w = (float*)st_malloc(sizeof(float) * (size_t)O * K);
  float* row = (float*)st_malloc(sizeof(float) * K);
  if (!w || !row) {
    st_free(w);
    st_free(row);
    return -1;
  }
  for (int o = 0; o < O; ++o) {
    st_mat_row(src, o, row);   /* f32 or dequantized int4 */
    for (int c = 0; c < C; ++c)
      for (int j = 0; j < k; ++j) w[(size_t)o * K + (size_t)j * C + c] = row[(size_t)c * k + j];
  }
  st_free(row);
  memset(&dst->m, 0, sizeof dst->m);
  dst->m.type = ST_W_F32;
  dst->m.O = O;
  dst->m.K = (int)K;
  dst->m.w = w;
  dst->data = w;
  return 0;
}

#define NEED(x)          \
  do {                   \
    if (!(x)) goto fail; \
  } while (0)

st_wavlm* st_wavlm_load(st_weights* w) {
  st_wavlm* m = (st_wavlm*)st_calloc(sizeof *m);
  if (!m) return NULL;
  static const int strides[] = {5, 2, 2, 2, 2, 2, 2};
  static const int dil[] = {1, 2, 3, 1, 1};
  int64_t sh[8];
  /* feature encoder */
  while (m->nconv < WL_MAXCONV &&
         st_weights_shape(w, st_name("wavlm.feature_extractor.conv_layers.%d.conv.weight", m->nconv), sh) == 3)
    ++m->nconv;
  if (m->nconv != 7) {
    st_seterr("wavlm: expected 7 feature-encoder convs, found %d", m->nconv);
    goto fail;
  }
  int cin = 1;
  for (int i = 0; i < m->nconv; ++i) {
    const char* n = st_name("wavlm.feature_extractor.conv_layers.%d.conv.weight", i);
    st_weights_shape(w, n, sh);
    if (sh[1] != cin) {
      st_seterr("wavlm: %s has %lld input channels, expected %d", n, (long long)sh[1], cin);
      goto fail;
    }
    m->conv_dim[i] = (int)sh[0];
    m->conv_k[i] = (int)sh[2];
    m->conv_s[i] = strides[i];
    const st_mat* cw = st_weights_mat(w, n, m->conv_dim[i], cin * m->conv_k[i]);
    NEED(cw);
    if (i == 0) m->conv0 = cw;
    else if (relayout(cw, cin, m->conv_k[i], &m->conv[i])) goto fail;
    cin = m->conv_dim[i];
  }
  if (st_weights_has(w, "wavlm.feature_extractor.conv_layers.0.conv.bias") ||
      st_weights_has(w, "wavlm.feature_extractor.conv_layers.1.layer_norm.weight")) {
    st_seterr("wavlm: only the conv_bias = false, feat_extract_norm = group variant is supported");
    goto fail;
  }
  int C0 = m->conv_dim[0], Cc = cin;
  NEED(m->gn_w = st_weights_f32(w, "wavlm.feature_extractor.conv_layers.0.layer_norm.weight", C0));
  NEED(m->gn_b = st_weights_f32(w, "wavlm.feature_extractor.conv_layers.0.layer_norm.bias", C0));
  NEED(m->fp_lnw = st_weights_f32(w, "wavlm.feature_projection.layer_norm.weight", Cc));
  NEED(m->fp_lnb = st_weights_f32(w, "wavlm.feature_projection.layer_norm.bias", Cc));
  NEED(m->fp = st_weights_mat(w, "wavlm.feature_projection.projection.weight", 0, Cc));
  m->D = m->fp->O;
  int D = m->D;
  NEED(m->fp_b = st_weights_f32(w, "wavlm.feature_projection.projection.bias", D));
  /* positional conv */
  if (st_weights_shape(w, "wavlm.encoder.pos_conv_embed.conv.weight", sh) != 3 || sh[0] != D || D % sh[1]) {
    st_seterr("wavlm: missing or bad wavlm.encoder.pos_conv_embed.conv.weight (export it with c/tools/export_wavlm.py)");
    goto fail;
  }
  m->pos_g = D / (int)sh[1];
  m->pos_k = (int)sh[2];
  {
    const st_mat* pw = st_weights_mat(w, "wavlm.encoder.pos_conv_embed.conv.weight", D, (int)sh[1] * m->pos_k);
    NEED(pw);
    if (relayout(pw, (int)sh[1], m->pos_k, &m->pos)) goto fail;
  }
  NEED(m->pos_b = st_weights_f32(w, "wavlm.encoder.pos_conv_embed.conv.bias", D));
  NEED(m->enc_lnw = st_weights_f32(w, "wavlm.encoder.layer_norm.weight", D));
  NEED(m->enc_lnb = st_weights_f32(w, "wavlm.encoder.layer_norm.bias", D));
  /* transformer */
  if (st_weights_shape(w, "wavlm.encoder.layers.0.attention.rel_attn_embed.weight", sh) != 2) {
    st_seterr("wavlm: missing rel_attn_embed");
    goto fail;
  }
  m->nbucket = (int)sh[0];
  m->H = (int)sh[1];
  m->dh = D / m->H;
  m->maxdist = 800;   /* config.max_bucket_distance */
  m->eps = 1e-5f;
  NEED(m->relemb = st_weights_f32(w, "wavlm.encoder.layers.0.attention.rel_attn_embed.weight", (int64_t)m->nbucket * m->H));
  while (m->nlayer < WL_MAXLAYER && st_weights_has(w, st_name("wavlm.encoder.layers.%d.attention.q_proj.weight", m->nlayer)))
    ++m->nlayer;
  for (int l = 0; l < m->nlayer; ++l) {
    wl_layer* L = &m->layer[l];
#define LW(f) st_name("wavlm.encoder.layers.%d.%s", l, f)
    NEED(L->q = st_weights_mat(w, LW("attention.q_proj.weight"), D, D));
    NEED(L->k = st_weights_mat(w, LW("attention.k_proj.weight"), D, D));
    NEED(L->v = st_weights_mat(w, LW("attention.v_proj.weight"), D, D));
    NEED(L->o = st_weights_mat(w, LW("attention.out_proj.weight"), D, D));
    NEED(L->qb = st_weights_f32(w, LW("attention.q_proj.bias"), D));
    NEED(L->kb = st_weights_f32(w, LW("attention.k_proj.bias"), D));
    NEED(L->vb = st_weights_f32(w, LW("attention.v_proj.bias"), D));
    NEED(L->ob = st_weights_f32(w, LW("attention.out_proj.bias"), D));
    NEED(L->gw = st_weights_f32(w, LW("attention.gru_rel_pos_linear.weight"), 8 * m->dh));
    NEED(L->gb = st_weights_f32(w, LW("attention.gru_rel_pos_linear.bias"), 8));
    NEED(L->gconst = st_weights_f32(w, LW("attention.gru_rel_pos_const"), m->H));
    NEED(L->ln1w = st_weights_f32(w, LW("layer_norm.weight"), D));
    NEED(L->ln1b = st_weights_f32(w, LW("layer_norm.bias"), D));
    NEED(L->ff1 = st_weights_mat(w, LW("feed_forward.intermediate_dense.weight"), 0, D));
    m->inter = L->ff1->O;
    NEED(L->ff1b = st_weights_f32(w, LW("feed_forward.intermediate_dense.bias"), m->inter));
    NEED(L->ff2 = st_weights_mat(w, LW("feed_forward.output_dense.weight"), D, m->inter));
    NEED(L->ff2b = st_weights_f32(w, LW("feed_forward.output_dense.bias"), D));
    NEED(L->ln2w = st_weights_f32(w, LW("final_layer_norm.weight"), D));
    NEED(L->ln2b = st_weights_f32(w, LW("final_layer_norm.bias"), D));
#undef LW
  }
  {
    const float* lw = st_weights_f32(w, "layer_weights", m->nlayer + 1);
    NEED(lw);
    float mx = lw[0], s = 0;
    for (int i = 1; i <= m->nlayer; ++i) mx = lw[i] > mx ? lw[i] : mx;
    for (int i = 0; i <= m->nlayer; ++i) s += (m->lw[i] = expf(lw[i] - mx));
    for (int i = 0; i <= m->nlayer; ++i) m->lw[i] /= s;
  }
  /* x-vector head */
  NEED(m->proj = st_weights_mat(w, "projector.weight", 0, D));
  NEED(m->proj_b = st_weights_f32(w, "projector.bias", m->proj->O));
  cin = m->proj->O;
  while (m->ntdnn < WL_MAXTDNN && st_weights_has(w, st_name("tdnn.%d.kernel.weight", m->ntdnn))) {
    int i = m->ntdnn;
    const st_mat* t = st_weights_mat(w, st_name("tdnn.%d.kernel.weight", i), 0, 0);
    NEED(t);
    if (i >= 5 || t->K % cin) {
      st_seterr("wavlm: unexpected TDNN layer %d", i);
      goto fail;
    }
    m->tdnn[i] = t;
    m->tdnn_dim[i] = t->O;
    m->tdnn_k[i] = t->K / cin;
    m->tdnn_d[i] = dil[i];
    NEED(m->tdnn_b[i] = st_weights_f32(w, st_name("tdnn.%d.kernel.bias", i), t->O));
    cin = t->O;
    ++m->ntdnn;
  }
  NEED(m->fe = st_weights_mat(w, "feature_extractor.weight", 512, 2 * cin));
  NEED(m->fe_b = st_weights_f32(w, "feature_extractor.bias", 512));
  m->xdim = 512;
  if (m->ntdnn != 5 || m->nlayer < 1 || m->H * m->dh != D) {
    st_seterr("wavlm: unexpected architecture (%d TDNN layers, %d transformer layers)", m->ntdnn, m->nlayer);
    goto fail;
  }
  return m;
fail:
  st_wavlm_free(m);
  return NULL;
}

void st_wavlm_free(st_wavlm* m) {
  if (!m) return;
  for (int i = 0; i < WL_MAXCONV; ++i) st_free(m->conv[i].data);
  st_free(m->pos.data);
  st_free(m);
}

/* ------------------------------------------------------------------ parallel elementwise helpers */
typedef struct {
  float* x;
  size_t rows, C;
  const double *mean, *inv;   /* group norm: per channel */
  const float *g, *b;
  double* part;               /* per task partial sums [n_tasks, C] */
  int op;
} ew_job;

enum { EW_GELU, EW_GN_SUM, EW_GN_VAR, EW_GN_APPLY_GELU };

static void ew_task(void* ctx, int task, int n_tasks) {
  ew_job* j = (ew_job*)ctx;
  size_t r0 = j->rows * (size_t)task / (size_t)n_tasks, r1 = j->rows * (size_t)(task + 1) / (size_t)n_tasks;
  size_t C = j->C;
  if (j->op == EW_GELU) {
    gelu(j->x + r0 * C, (r1 - r0) * C);
  } else if (j->op == EW_GN_SUM || j->op == EW_GN_VAR) {
    double* s = j->part + (size_t)task * C;
    for (size_t c = 0; c < C; ++c) s[c] = 0;
    for (size_t r = r0; r < r1; ++r) {
      const float* x = j->x + r * C;
      if (j->op == EW_GN_SUM)
        for (size_t c = 0; c < C; ++c) s[c] += x[c];
      else
        for (size_t c = 0; c < C; ++c) {
          double d = x[c] - j->mean[c];
          s[c] += d * d;
        }
    }
  } else {
    for (size_t r = r0; r < r1; ++r) {
      float* x = j->x + r * C;
      for (size_t c = 0; c < C; ++c) x[c] = (float)((x[c] - j->mean[c]) * j->inv[c]) * j->g[c] + j->b[c];
      gelu(x, C);
    }
  }
}

static int ew_tasks(st_pool* p, size_t rows) {
  int t = st_pool_size(p) * 4;
  if ((size_t)t > rows) t = (int)rows;
  return t < 1 ? 1 : t;
}

static void par_gelu(st_pool* p, float* x, size_t rows, size_t C) {
  ew_job j = {x, rows, C, NULL, NULL, NULL, NULL, NULL, EW_GELU};
  st_parallel_for(p, ew_tasks(p, rows), ew_task, &j);
}

/* GroupNorm with one channel per group on time-major [T, C] (stats over time, biased variance), then GELU */
static int groupnorm_gelu(st_pool* p, float* x, size_t T, size_t C, const float* g, const float* b, float eps) {
  int nt = ew_tasks(p, T);
  double* part = (double*)st_malloc(sizeof(double) * (size_t)nt * C);
  double* mean = (double*)st_malloc(sizeof(double) * C);
  double* inv = (double*)st_malloc(sizeof(double) * C);
  if (!part || !mean || !inv) {
    st_free(part);
    st_free(mean);
    st_free(inv);
    return -1;
  }
  ew_job j = {x, T, C, mean, inv, g, b, part, EW_GN_SUM};
  st_parallel_for(p, nt, ew_task, &j);
  for (size_t c = 0; c < C; ++c) {
    double s = 0;
    for (int t = 0; t < nt; ++t) s += part[(size_t)t * C + c];
    mean[c] = s / (double)T;
  }
  j.op = EW_GN_VAR;
  st_parallel_for(p, nt, ew_task, &j);
  for (size_t c = 0; c < C; ++c) {
    double s = 0;
    for (int t = 0; t < nt; ++t) s += part[(size_t)t * C + c];
    inv[c] = 1.0 / sqrt(s / (double)T + eps);
  }
  j.op = EW_GN_APPLY_GELU;
  st_parallel_for(p, nt, ew_task, &j);
  st_free(part);
  st_free(mean);
  st_free(inv);
  return 0;
}

/* ------------------------------------------------------------------ positional conv (grouped) */
typedef struct {
  const st_wavlm* m;
  const float* xg;   /* [G, T + pos_k, Cg] group-major zero-padded input */
  float* y;          /* [T, D] */
  int T;
} pos_job;

static void pos_task(void* ctx, int g, int n_tasks) {
  (void)n_tasks;
  const pos_job* j = (const pos_job*)ctx;
  const st_wavlm* m = j->m;
  int Cg = m->D / m->pos_g;
  st_mat w;
  st_mat_rows(&m->pos.m, g * Cg, Cg, &w);
  /* output t (after dropping the last of T + 1 frames) reads padded rows t .. t + k - 1: contiguous, lda = Cg */
  st_linear(NULL, j->xg + (size_t)g * (size_t)(j->T + m->pos_k) * Cg, j->T, Cg, &w, m->pos_b + (size_t)g * Cg,
            j->y + (size_t)g * Cg, m->D, 0);
}

/* ------------------------------------------------------------------ relative position bias */
static int rel_bucket(int rel, int nbucket, int maxdist) {
  int nb = nbucket / 2, me = nb / 2, b = rel > 0 ? nb : 0, a = rel < 0 ? -rel : rel;
  if (a < me) return b + a;
  /* float32 as torch: log(a / me) / log(maxdist / me) * (nb - me), + me, truncated */
  float v = logf((float)a / (float)me) / (float)log((double)maxdist / me) * (float)(nb - me);
  int l = (int)((float)me + v);
  return b + (l < nb - 1 ? l : nb - 1);
}

/* ------------------------------------------------------------------ attention with the gated relative bias
 * One task per (head, block of 64 queries): gate g[h, i] from the layer input, S = Q K^T * scale + g[h, i] * rel[h, j - i],
 * softmax, S V. rel is tabulated per offset ([H, 2T - 1]), so bias row i is the contiguous slice starting at T - 1 - i
 * and the [H, T, T] bias tensor is never built. */
enum { WL_QB = 64 };
typedef struct {
  const st_wavlm* m;
  const wl_layer* L;
  const float *x, *q, *k, *v;   /* [T, D] */
  const float* rel;             /* [H, 2T - 1] */
  float* out;                   /* [T, D] */
  int T, nqb;
  int* fail;
} attn_job;

static void attn_task(void* ctx, int task, int n_tasks) {
  (void)n_tasks;
  const attn_job* j = (const attn_job*)ctx;
  const st_wavlm* m = j->m;
  const wl_layer* L = j->L;
  int T = j->T, D = m->D, dh = m->dh, h = task / j->nqb, q0 = (task % j->nqb) * WL_QB;
  int nq = T - q0 < WL_QB ? T - q0 : WL_QB;
  float* s = (float*)st_malloc(sizeof(float) * (size_t)WL_QB * (size_t)T);
  if (!s) {
    *j->fail = 1;
    return;
  }
  st_gemm_nt(NULL, nq, T, dh, j->q + (size_t)q0 * D + (size_t)h * dh, D, j->k + (size_t)h * dh, D, s, T, NULL, 0);
  float scale = 1.f / sqrtf((float)dh);
  for (int r = 0; r < nq; ++r) {
    int t = q0 + r;
    /* gate: gru_rel_pos_linear on this head's slice of the layer input, viewed (2, 4) and summed */
    const float* xh = j->x + (size_t)t * D + (size_t)h * dh;
    float pr[8];
    for (int i = 0; i < 8; ++i) {
      float a = 0;
      for (int kk = 0; kk < dh; ++kk) a += xh[kk] * L->gw[i * dh + kk];
      pr[i] = a + L->gb[i];
    }
    float ga = 1.f / (1.f + expf(-(((pr[0] + pr[1]) + pr[2]) + pr[3])));
    float gb = 1.f / (1.f + expf(-(((pr[4] + pr[5]) + pr[6]) + pr[7])));
    float gate = ga * (gb * L->gconst[h] - 1.f) + 2.f;
    const float* rb = j->rel + (size_t)h * (2 * T - 1) + (T - 1 - t);
    float* row = s + (size_t)r * T;
    float mx = -INFINITY;
    for (int c = 0; c < T; ++c) {
      row[c] = row[c] * scale + gate * rb[c];
      mx = row[c] > mx ? row[c] : mx;
    }
    float inv = 1.f / exp_sum(row, T, mx);
    for (int c = 0; c < T; ++c) row[c] *= inv;
  }
  st_gemm_nn(NULL, nq, dh, T, s, T, j->v + (size_t)h * dh, D, j->out + (size_t)q0 * D + (size_t)h * dh, D, NULL, 0);
  st_free(s);
}

/* ------------------------------------------------------------------ forward */
static int forward16(st_wavlm* m, st_pool* p, const float* x, size_t n, float* out) {
  int rc = -1;
  int D = m->D, H = m->H, FF = m->inter;
  float *a = NULL, *b = NULL, *h = NULL, *acc = NULL, *q = NULL, *k = NULL, *v = NULL, *ctx = NULL, *ff = NULL;
  float *rel = NULL, *xg = NULL, *t0 = NULL, *t1 = NULL;
  /* feature encoder */
  int Tl[WL_MAXCONV + 1];
  Tl[0] = (int)n;
  for (int i = 0; i < m->nconv; ++i) {
    Tl[i + 1] = Tl[i] < m->conv_k[i] ? 0 : (Tl[i] - m->conv_k[i]) / m->conv_s[i] + 1;
    if (Tl[i + 1] < 1) {
      st_seterr("wavlm: input of %zu samples is too short", n);
      return -1;
    }
  }
  a = (float*)st_malloc(sizeof(float) * (size_t)Tl[1] * m->conv_dim[0]);
  b = (float*)st_malloc(sizeof(float) * (size_t)Tl[2] * m->conv_dim[1]);
  if (!a || !b) goto done;
  st_linear(p, x, Tl[1], m->conv_s[0], m->conv0, NULL, a, m->conv_dim[0], 0);
  if (groupnorm_gelu(p, a, (size_t)Tl[1], (size_t)m->conv_dim[0], m->gn_w, m->gn_b, 1e-5f)) goto done;
  {
    float *src = a, *dst = b;
    for (int i = 1; i < m->nconv; ++i) {
      int C = m->conv_dim[i - 1];
      st_linear(p, src, Tl[i + 1], m->conv_s[i] * C, &m->conv[i].m, NULL, dst, m->conv_dim[i], 0);
      par_gelu(p, dst, (size_t)Tl[i + 1], (size_t)m->conv_dim[i]);
      float* t = src;
      src = dst;
      dst = t;
    }
    if (src != a) {   /* keep the result in `a` */
      float* t = a;
      a = b;
      b = t;
    }
  }
  st_free(b);
  b = NULL;
  int T = Tl[m->nconv], Cc = m->conv_dim[m->nconv - 1];
  /* feature projection */
  st_layernorm(a, T, Cc, Cc, m->fp_lnw, m->fp_lnb, m->eps);
  size_t TD = (size_t)T * D;
  h = (float*)st_malloc(sizeof(float) * TD);
  acc = (float*)st_malloc(sizeof(float) * TD);
  q = (float*)st_malloc(sizeof(float) * TD);
  k = (float*)st_malloc(sizeof(float) * TD);
  v = (float*)st_malloc(sizeof(float) * TD);
  ctx = (float*)st_malloc(sizeof(float) * TD);
  ff = (float*)st_malloc(sizeof(float) * (size_t)T * (FF > D ? FF : D));
  rel = (float*)st_malloc(sizeof(float) * (size_t)H * (2 * T - 1));
  int Cg = D / m->pos_g, Tp = T + m->pos_k;
  xg = (float*)st_calloc(sizeof(float) * (size_t)m->pos_g * Tp * Cg);
  if (!h || !acc || !q || !k || !v || !ctx || !ff || !rel || !xg) goto done;
  st_linear(p, a, T, Cc, m->fp, m->fp_b, h, D, 0);
  st_free(a);
  a = NULL;
  /* positional conv embedding: input padded by pos_k / 2 on the left (and right), output frame T dropped */
  for (int g = 0; g < m->pos_g; ++g)
    for (int t = 0; t < T; ++t)
      memcpy(xg + ((size_t)g * Tp + (size_t)(t + m->pos_k / 2)) * Cg, h + (size_t)t * D + (size_t)g * Cg,
             sizeof(float) * (size_t)Cg);
  {
    pos_job pj = {m, xg, ctx, T};
    st_parallel_for(p, m->pos_g, pos_task, &pj);
  }
  par_gelu(p, ctx, (size_t)T, (size_t)D);
  for (size_t i = 0; i < TD; ++i) h[i] += ctx[i];
  st_layernorm(h, T, D, D, m->enc_lnw, m->enc_lnb, m->eps);
  for (size_t i = 0; i < TD; ++i) acc[i] = m->lw[0] * h[i];
  /* relative position bias per offset d = j - i in [-(T-1), T-1]: rel[h, d + T - 1] */
  for (int d = -(T - 1); d <= T - 1; ++d) {
    const float* e = m->relemb + (size_t)rel_bucket(d, m->nbucket, m->maxdist) * H;
    for (int hh = 0; hh < H; ++hh) rel[(size_t)hh * (2 * T - 1) + (size_t)(d + T - 1)] = e[hh];
  }
  /* transformer */
  for (int l = 0; l < m->nlayer; ++l) {
    const wl_layer* L = &m->layer[l];
    st_linear(p, h, T, D, L->q, L->qb, q, D, 0);
    st_linear(p, h, T, D, L->k, L->kb, k, D, 0);
    st_linear(p, h, T, D, L->v, L->vb, v, D, 0);
    int fail = 0;
    attn_job aj = {m, L, h, q, k, v, rel, ctx, T, (T + WL_QB - 1) / WL_QB, &fail};
    st_parallel_for(p, H * aj.nqb, attn_task, &aj);
    if (fail) {
      st_seterr("out of memory");
      goto done;
    }
    st_linear(p, ctx, T, D, L->o, L->ob, h, D, 1);   /* residual: h += attn */
    st_layernorm(h, T, D, D, L->ln1w, L->ln1b, m->eps);
    st_linear(p, h, T, D, L->ff1, L->ff1b, ff, FF, 0);
    par_gelu(p, ff, (size_t)T, (size_t)FF);
    st_linear(p, ff, T, FF, L->ff2, L->ff2b, h, D, 1);
    st_layernorm(h, T, D, D, L->ln2w, L->ln2b, m->eps);
    float wl = m->lw[l + 1];
    for (size_t i = 0; i < TD; ++i) acc[i] += wl * h[i];
  }
  st_free(rel);
  st_free(xg);
  rel = xg = NULL;
  /* x-vector head */
  int maxd = m->proj->O;
  for (int i = 0; i < m->ntdnn; ++i) maxd = m->tdnn_dim[i] > maxd ? m->tdnn_dim[i] : maxd;
  t0 = (float*)st_malloc(sizeof(float) * (size_t)T * maxd);
  t1 = (float*)st_malloc(sizeof(float) * (size_t)T * maxd);
  if (!t0 || !t1) goto done;
  st_linear(p, acc, T, D, m->proj, m->proj_b, t0, m->proj->O, 0);
  int C = m->proj->O, Tc = T;
  for (int i = 0; i < m->ntdnn; ++i) {
    int kk = m->tdnn_k[i], d = m->tdnn_d[i], To = Tc - d * (kk - 1), O = m->tdnn_dim[i];
    if (To < 2) {
      st_seterr("wavlm: input too short for the TDNN stack");
      goto done;
    }
    if (d == 1 || kk == 1) {
      st_linear(p, t0, To, C, m->tdnn[i], m->tdnn_b[i], t1, O, 0);   /* frames t .. t+k-1 are contiguous */
    } else {
      /* dilated: gather the k frames of each output row (ff is free here and large enough: T * FF >= To * k * C) */
      size_t KC = (size_t)kk * C;
      if ((size_t)To * KC > (size_t)T * (FF > D ? FF : D)) {
        st_seterr("wavlm: TDNN scratch too small");
        goto done;
      }
      for (int t = 0; t < To; ++t)
        for (int j = 0; j < kk; ++j)
          memcpy(ff + (size_t)t * KC + (size_t)j * C, t0 + (size_t)(t + j * d) * C, sizeof(float) * (size_t)C);
      st_linear(p, ff, To, (int)KC, m->tdnn[i], m->tdnn_b[i], t1, O, 0);
    }
    st_relu(t1, (size_t)To * O);
    float* tt = t0;
    t0 = t1;
    t1 = tt;
    C = O;
    Tc = To;
  }
  /* statistics pooling: mean and unbiased std over time */
  float* stats = t1;   /* reuse: 2 * C floats */
  for (int c = 0; c < C; ++c) {
    double s = 0, s2 = 0;
    for (int t = 0; t < Tc; ++t) s += t0[(size_t)t * C + c];
    double mu = s / Tc;
    for (int t = 0; t < Tc; ++t) {
      double dd = t0[(size_t)t * C + c] - mu;
      s2 += dd * dd;
    }
    stats[c] = (float)mu;
    stats[C + c] = (float)sqrt(s2 / (Tc - 1));
  }
  st_linear(p, stats, 1, 2 * C, m->fe, m->fe_b, out, m->xdim, 0);
  {
    double nn = 0;
    for (int i = 0; i < m->xdim; ++i) nn += (double)out[i] * out[i];
    double nrm = sqrt(nn);
    float inv = (float)(1.0 / (nrm > 1e-12 ? nrm : 1e-12));
    for (int i = 0; i < m->xdim; ++i) out[i] *= inv;
  }
  rc = 0;
done:
  st_free(a);
  st_free(b);
  st_free(h);
  st_free(acc);
  st_free(q);
  st_free(k);
  st_free(v);
  st_free(ctx);
  st_free(ff);
  st_free(rel);
  st_free(xg);
  st_free(t0);
  st_free(t1);
  return rc;
}

int st_wavlm_xvector16(st_wavlm* m, st_pool* p, const float* wav16, size_t n, float out[512]) {
  if (!m || !wav16 || n == 0) {
    st_seterr("wavlm: empty input");
    return -1;
  }
  if (n > (size_t)WL_MAX_S * WL_SR) n = (size_t)WL_MAX_S * WL_SR;
  if (n >= WL_MIN_N) return forward16(m, p, wav16, n, out);
  /* XVec: np.tile(y, ceil(8000 / len(y))) */
  size_t reps = (WL_MIN_N + n - 1) / n;
  float* t = (float*)st_malloc(sizeof(float) * n * reps);
  if (!t) return -1;
  for (size_t r = 0; r < reps; ++r) memcpy(t + r * n, wav16, sizeof(float) * n);
  int rc = forward16(m, p, t, n * reps, out);
  st_free(t);
  return rc;
}

int st_wavlm_xvector(st_wavlm* m, st_pool* p, const float* wav24, size_t n, float out[512]) {
  if (!m || !wav24 || n == 0) {
    st_seterr("wavlm: empty input");
    return -1;
  }
  if (n > (size_t)WL_MAX_S * WL_SR24) n = (size_t)WL_MAX_S * WL_SR24;
  float* y;
  size_t ny;
  if (st_resample(wav24, n, WL_SR24, WL_SR, &y, &ny)) return -1;
  int rc = ny ? st_wavlm_xvector16(m, p, y, ny, out) : -1;
  if (!ny) st_seterr("wavlm: input too short");
  st_free(y);
  return rc;
}
