/* SteerTTS prompt mapper: all-MiniLM-L6-v2 (BertModel: word + position + token-type-0 embeddings, LayerNorm, 6
 * post-LN encoder layers with erf GELU) mean-pooled and L2-normalised (F.normalize, eps 1e-12), then the Adapter
 * (Linear 384->512, GELU, Linear 512->512, GELU; heads tag / spec / val) and PromptMapper's rounding / top-3. */
#include "st_prompt.h"

#include <math.h>
#include <string.h>

#include "st_wordpiece.h"

enum { D = ST_MINILM_DIM, NL = 6, NH = 12, DFF = 1536, MAXPOS = 512, MAXTOK = 256, AH = 512, NC = ST_PROMPT_NCTRL };
static const float LN_EPS = 1e-12f;

typedef struct {
  const st_mat *q, *k, *v, *o, *up, *down;
  const float *bq, *bk, *bv, *bo, *bup, *bdown, *ln1_g, *ln1_b, *ln2_g, *ln2_b;
} bert_layer;

struct st_prompt {
  st_wordpiece* wp;
  const st_mat *wemb, *pemb, *temb;
  const float *eln_g, *eln_b;
  bert_layer L[NL];
  const st_mat *a0, *a3, *atag, *aspec, *aval;
  const float *b0, *b3, *btag, *bspec, *bval;
  int n_tags;
  char** tags;
};

#define GET_MAT(dst, w, nm, O, K) \
  do {                            \
    (dst) = st_weights_mat((w), (nm), (O), (K)); \
    if (!(dst)) goto fail;        \
  } while (0)
#define GET_F32(dst, w, nm, n) \
  do {                         \
    (dst) = st_weights_f32((w), (nm), (n)); \
    if (!(dst)) goto fail;     \
  } while (0)

void st_prompt_free(st_prompt* p) {
  if (!p) return;
  st_wordpiece_free(p->wp);
  if (p->tags)
    for (int i = 0; i < p->n_tags; ++i) st_free(p->tags[i]);
  st_free(p->tags);
  st_free(p);
}

st_prompt* st_prompt_load(st_weights* mw, const char* vocab_path, st_weights* aw, const char* const* tags, int n_tags) {
  st_prompt* p = (st_prompt*)st_calloc(sizeof *p);
  if (!p) return NULL;
  if (!(p->wp = st_wordpiece_load(vocab_path))) goto fail;
  int V = st_wordpiece_vocab_size(p->wp);
  GET_MAT(p->wemb, mw, "embeddings.word_embeddings.weight", V, D);
  GET_MAT(p->pemb, mw, "embeddings.position_embeddings.weight", 0, D);
  GET_MAT(p->temb, mw, "embeddings.token_type_embeddings.weight", 0, D);
  if (p->pemb->O < MAXTOK) {
    st_seterr("%s: %d position embeddings < %d", st_weights_path(mw), p->pemb->O, MAXTOK);
    goto fail;
  }
  GET_F32(p->eln_g, mw, "embeddings.LayerNorm.weight", D);
  GET_F32(p->eln_b, mw, "embeddings.LayerNorm.bias", D);
  for (int l = 0; l < NL; ++l) {
    bert_layer* L = &p->L[l];
#define LN(s) st_name("encoder.layer.%d.%s", l, s)
    GET_MAT(L->q, mw, LN("attention.self.query.weight"), D, D);
    GET_MAT(L->k, mw, LN("attention.self.key.weight"), D, D);
    GET_MAT(L->v, mw, LN("attention.self.value.weight"), D, D);
    GET_MAT(L->o, mw, LN("attention.output.dense.weight"), D, D);
    GET_MAT(L->up, mw, LN("intermediate.dense.weight"), DFF, D);
    GET_MAT(L->down, mw, LN("output.dense.weight"), D, DFF);
    GET_F32(L->bq, mw, LN("attention.self.query.bias"), D);
    GET_F32(L->bk, mw, LN("attention.self.key.bias"), D);
    GET_F32(L->bv, mw, LN("attention.self.value.bias"), D);
    GET_F32(L->bo, mw, LN("attention.output.dense.bias"), D);
    GET_F32(L->bup, mw, LN("intermediate.dense.bias"), DFF);
    GET_F32(L->bdown, mw, LN("output.dense.bias"), D);
    GET_F32(L->ln1_g, mw, LN("attention.output.LayerNorm.weight"), D);
    GET_F32(L->ln1_b, mw, LN("attention.output.LayerNorm.bias"), D);
    GET_F32(L->ln2_g, mw, LN("output.LayerNorm.weight"), D);
    GET_F32(L->ln2_b, mw, LN("output.LayerNorm.bias"), D);
#undef LN
  }
  GET_MAT(p->a0, aw, "body.0.weight", AH, D);
  GET_MAT(p->a3, aw, "body.3.weight", AH, AH);
  GET_MAT(p->atag, aw, "tag.weight", n_tags, AH);
  GET_MAT(p->aspec, aw, "spec.weight", NC, AH);
  GET_MAT(p->aval, aw, "val.weight", NC, AH);
  GET_F32(p->b0, aw, "body.0.bias", AH);
  GET_F32(p->b3, aw, "body.3.bias", AH);
  GET_F32(p->btag, aw, "tag.bias", n_tags);
  GET_F32(p->bspec, aw, "spec.bias", NC);
  GET_F32(p->bval, aw, "val.bias", NC);
  if (n_tags < 3) {
    st_seterr("prompt adapter: %d tags < 3", n_tags);
    goto fail;
  }
  p->tags = (char**)st_calloc(sizeof(char*) * (size_t)n_tags);
  if (!p->tags) goto fail;
  p->n_tags = n_tags;
  for (int i = 0; i < n_tags; ++i)
    if (!(p->tags[i] = st_strdup(tags[i]))) goto fail;
  return p;
fail:
  st_prompt_free(p);
  return NULL;
}

int st_prompt_n_tags(const st_prompt* p) { return p->n_tags; }
const char* st_prompt_tag(const st_prompt* p, int i) { return i >= 0 && i < p->n_tags ? p->tags[i] : NULL; }

/* ------------------------------------------------------------------ MiniLM */
int st_minilm_embed(st_prompt* p, st_pool* pool, const char* text, float out[D]) {
  int32_t ids[MAXTOK];
  int T = st_wordpiece_encode(p->wp, text, ids, MAXTOK);
  if (T < 0) return -1;
  size_t TD = (size_t)T * D;
  float* x = (float*)st_malloc(sizeof(float) * (TD * 5 + (size_t)T * DFF + 2 * D));
  if (!x) return -1;
  float *q = x + TD, *k = q + TD, *v = k + TD, *ctx = v + TD, *ff = ctx + TD, *row = ff + (size_t)T * DFF,
        *t0 = row + D;
  st_mat_row(p->temb, 0, t0);
  for (int t = 0; t < T; ++t) {
    float* xr = x + (size_t)t * D;
    st_mat_row(p->wemb, ids[t], xr);
    st_mat_row(p->pemb, t, row);
    for (int c = 0; c < D; ++c) xr[c] = (xr[c] + t0[c]) + row[c];   /* (word + token_type) + position, as BERT */
  }
  st_layernorm(x, T, D, D, p->eln_g, p->eln_b, LN_EPS);
  for (int l = 0; l < NL; ++l) {
    const bert_layer* L = &p->L[l];
    st_linear(pool, x, T, D, L->q, L->bq, q, D, 0);
    st_linear(pool, x, T, D, L->k, L->bk, k, D, 0);
    st_linear(pool, x, T, D, L->v, L->bv, v, D, 0);
    if (st_attention(pool, q, D, k, D, v, D, ctx, D, T, T, NH, D / NH, NULL, NULL, 0.f)) {
      st_free(x);
      return -1;
    }
    st_linear(pool, ctx, T, D, L->o, L->bo, x, D, 1);   /* x += attn_out (residual) */
    st_layernorm(x, T, D, D, L->ln1_g, L->ln1_b, LN_EPS);
    st_linear(pool, x, T, D, L->up, L->bup, ff, DFF, 0);
    st_gelu(ff, (size_t)T * DFF);
    st_linear(pool, ff, T, DFF, L->down, L->bdown, x, D, 1);
    st_layernorm(x, T, D, D, L->ln2_g, L->ln2_b, LN_EPS);
  }
  /* mean pooling over the (all-ones) attention mask, then F.normalize(p=2, eps=1e-12) */
  double nrm = 0;
  for (int c = 0; c < D; ++c) {
    double s = 0;
    for (int t = 0; t < T; ++t) s += x[(size_t)t * D + c];
    out[c] = (float)(s / T);
    nrm += (double)out[c] * out[c];
  }
  float n = (float)sqrt(nrm);
  n = n > 1e-12f ? n : 1e-12f;
  for (int c = 0; c < D; ++c) out[c] /= n;
  st_free(x);
  return 0;
}

/* ------------------------------------------------------------------ rounding */
long long st_pyround_scaled(double x, int nd) {
  double S = 1;
  for (int i = 0; i < nd; ++i) S *= 10;
  double pr = x * S, e = fma(x, S, -pr);   /* x * S == pr + e exactly */
  double t = floor(pr), d = (pr - t) - 0.5; /* exact for |pr| < 2^52 */
  long long k = (long long)t;
  if (d > -e) ++k;
  else if (d == -e && (k & 1)) ++k;          /* an exact tie: to even */
  return k;
}

/* ------------------------------------------------------------------ adapter */
int st_prompt_map(st_prompt* p, st_pool* pool, const char* prompt, st_prompt_result* out) {
  float e[D], h0[AH], h1[AH], lt[256], ls[NC], lv[NC];
  float* tag = p->n_tags <= 256 ? lt : (float*)st_malloc(sizeof(float) * (size_t)p->n_tags);
  if (!tag) return -1;
  if (st_minilm_embed(p, pool, prompt, e) != 0) {
    if (tag != lt) st_free(tag);
    return -1;
  }
  st_linear(pool, e, 1, D, p->a0, p->b0, h0, AH, 0);
  st_gelu(h0, AH);
  st_linear(pool, h0, 1, AH, p->a3, p->b3, h1, AH, 0);
  st_gelu(h1, AH);
  st_linear(pool, h1, 1, AH, p->atag, p->btag, tag, p->n_tags, 0);
  st_linear(pool, h1, 1, AH, p->aspec, p->bspec, ls, NC, 0);
  st_linear(pool, h1, 1, AH, p->aval, p->bval, lv, NC, 0);
  st_softmax_rows(tag, 1, p->n_tags, p->n_tags);
  memset(out, 0, sizeof *out);
  /* torch.topk(3): largest first; equal values keep the lower index first */
  for (int j = 0; j < 3; ++j) {
    int best = -1;
    for (int i = 0; i < p->n_tags; ++i) {
      int used = 0;
      for (int u = 0; u < j; ++u) used |= out->top_tag[u] == i;
      if (!used && (best < 0 || tag[i] > tag[best])) best = i;
    }
    long long m = st_pyround_scaled((double)tag[best], 3);
    out->top_tag[j] = best;
    out->top_p_milli[j] = (int)m;
    out->top_p[j] = (float)((double)m / 1000.0);
  }
  out->n_top = 3;
  for (int i = 0; i < NC; ++i) {
    float sp = 1.f / (1.f + expf(-ls[i]));   /* torch.sigmoid in fp32 */
    if (!(sp > 0.5f)) continue;
    long long c = st_pyround_scaled((double)lv[i], 2);
    out->ctrl_dim[out->n_ctrl] = i;
    out->ctrl_z_centi[out->n_ctrl] = (int)c;
    out->ctrl_z[out->n_ctrl++] = (float)((double)c / 100.0);
  }
  if (tag != lt) st_free(tag);
  return 0;
}
