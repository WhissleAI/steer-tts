/* Port of steertts_edge/edge_front: backend.EdgeG2p (g2p_en.G2p.__call__ replacement), lexicon.Lexicon (residual
 * CMUdict + OOV cache), gru_np.GruG2P (tier B int8 GRU), hetero.Hetero, tok.word_tokenize.
 * Token stream of g2p_call: symbol ids (phones and the punctuation , . ? ! -) and TOK_SPACE between words, i.e.
 * g2p_en's output list with every token to_phonemes would skip removed. */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")   /* no FMA contraction: the GRU arithmetic is identical on every target */
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if !defined(ST_G2P_SCALAR) && (defined(__SSE2__) || defined(_M_X64))   /* -DST_G2P_SCALAR: portable path */
#include <emmintrin.h>
#define ST_G2P_SSE2 1
#elif !defined(ST_G2P_SCALAR) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define ST_G2P_NEON 1
#endif
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "st_front_int.h"

static u32 le32(const void* p) {
  const unsigned char* q = (const unsigned char*)p;
  return (u32)q[0] | ((u32)q[1] << 8) | ((u32)q[2] << 16) | ((u32)q[3] << 24);
}

/* ------------------------------------------------------------------ Lexicon */
static int lex_load(st_lexicon* L, const char* path) {
  memset(L, 0, sizeof *L);
  if (st_read_file(path, &L->buf, &L->len)) {
    st_seterr("steertts front: cannot open %s", path);
    return -1;
  }
  const char* b = L->buf;
  if (L->len < 20 || memcmp(b, "STLX", 4) != 0) {
    st_seterr("steertts front: bad lexicon %s", path);
    goto fail;
  }
  if (le32(b + 4) != 1) {
    st_seterr("steertts front: bad lexicon version");
    goto fail;
  }
  L->n = le32(b + 8);
  u32 wb = le32(b + 12), pb = le32(b + 16);
  size_t need = 20 + 8 * ((size_t)L->n + 1) + wb + pb;
  if ((size_t)L->n > L->len || need > L->len) {
    st_seterr("steertts front: truncated asset");
    goto fail;
  }
  L->wo = b + 20;
  L->po = L->wo + 4 * ((size_t)L->n + 1);
  L->words = L->po + 4 * ((size_t)L->n + 1);
  L->phones = (const uint8_t*)(L->words + wb);
  for (u32 k = 0; k < L->n; ++k) {  /* offsets must be monotonic and inside the blobs */
    if (le32(L->wo + 4 * k) > le32(L->wo + 4 * (k + 1)) || le32(L->po + 4 * k) > le32(L->po + 4 * (k + 1))) {
      st_seterr("steertts front: bad lexicon %s", path);
      goto fail;
    }
  }
  if (le32(L->wo + 4 * (size_t)L->n) > wb || le32(L->po + 4 * (size_t)L->n) > pb) {
    st_seterr("steertts front: bad lexicon %s", path);
    goto fail;
  }
  return 0;
fail:
  st_free(L->buf);
  memset(L, 0, sizeof *L);
  return -1;
}

/* 1 if w is a key; phones (symbol ids, up to *nph) may be empty (member-only entry / cached empty prediction) */
static int lex_get(const st_lexicon* L, const char* w, int* ph, int* nph) {
  size_t lo = 0, hi = L->n, wl = strlen(w);
  while (lo < hi) {  /* keys are sorted bytewise (== Python str order for ascii) */
    size_t mid = (lo + hi) / 2;
    u32 a = le32(L->wo + 4 * mid), e = le32(L->wo + 4 * (mid + 1));
    size_t kl = e - a, m = kl < wl ? kl : wl;
    int c = memcmp(L->words + a, w, m);
    if (c == 0) c = kl < wl ? -1 : kl > wl;
    if (c == 0) {
      if (ph) {
        u32 p0 = le32(L->po + 4 * mid), p1 = le32(L->po + 4 * (mid + 1));
        int n = 0;
        for (u32 p = p0; p < p1 && n < 64; ++p) ph[n++] = L->phones[p];
        *nph = n;
      }
      return 1;
    }
    if (c < 0)
      lo = mid + 1;
    else
      hi = mid;
  }
  return 0;
}

/* ------------------------------------------------------------------ GRU */
static void gru_free(st_gru* g) {
  st_free(g->enc_x); st_free(g->dec_x); st_free(g->enc_s); st_free(g->enc_b); st_free(g->dec_s);
  st_free(g->dec_b); st_free(g->fc_s); st_free(g->fc_b); st_free(g->enc_q); st_free(g->dec_q);
  st_free(g->fc_q); st_free(g->phon2sym);
  memset(g, 0, sizeof *g);
}

typedef struct {
  const char* b;
  size_t n, o;
  int bad;
} grd;
static float* rd_f32(grd* r, size_t k) {
  if (r->bad || k > (r->n - r->o) / 4) {
    r->bad = 1;
    return NULL;
  }
  float* v = st_malloc(k * sizeof(float) + 1);
  if (!v) {
    r->bad = 2;
    return NULL;
  }
  for (size_t i = 0; i < k; ++i) {
    u32 u = le32(r->b + r->o + 4 * i);
    memcpy(&v[i], &u, 4);
  }
  r->o += 4 * k;
  return v;
}
static int8_t* rd_i8(grd* r, size_t k) {
  if (r->bad || k > r->n - r->o) {
    r->bad = 1;
    return NULL;
  }
  int8_t* v = st_malloc(k + 1);
  if (!v) {
    r->bad = 2;
    return NULL;
  }
  memcpy(v, r->b + r->o, k);
  r->o += k;
  return v;
}

static int gru_load(st_gru* g, const char* path) {
  memset(g, 0, sizeof *g);
  char* buf;
  size_t len;
  if (st_read_file(path, &buf, &len)) {
    st_seterr("steertts front: cannot open %s", path);
    return -1;
  }
  if (len < 20 || memcmp(buf, "STGR", 4) != 0) {
    st_free(buf);
    st_seterr("steertts front: bad gru.bin");
    return -1;
  }
  if (le32(buf + 4) != 1) {
    st_free(buf);
    st_seterr("steertts front: bad gru.bin version");
    return -1;
  }
  g->H = (int)le32(buf + 8);
  g->NG = (int)le32(buf + 12);
  g->NP = (int)le32(buf + 16);
  if (g->H <= 0 || g->H > 512 || g->H % 32 || g->NG <= 2 || g->NG > 4096 || g->NP <= 3 || g->NP > 4096) {
    st_free(buf);
    st_seterr("steertts front: unsupported GRU size");
    return -1;
  }
  size_t H = (size_t)g->H, H3 = 3 * H;
  grd r = {buf, len, 20, 0};
  g->enc_x = rd_f32(&r, (size_t)g->NG * H3);
  g->dec_x = rd_f32(&r, (size_t)g->NP * H3);
  g->enc_q = rd_i8(&r, H3 * H);
  g->enc_s = rd_f32(&r, H3);
  g->enc_b = rd_f32(&r, H3);
  g->dec_q = rd_i8(&r, H3 * H);
  g->dec_s = rd_f32(&r, H3);
  g->dec_b = rd_f32(&r, H3);
  g->fc_q = rd_i8(&r, (size_t)g->NP * H);
  g->fc_s = rd_f32(&r, (size_t)g->NP);
  g->fc_b = rd_f32(&r, (size_t)g->NP);
  float* p2s = rd_f32(&r, (size_t)g->NP);  /* int32 payload, read as raw 32-bit words */
  if (!r.bad) {
    g->phon2sym = st_malloc((size_t)g->NP * sizeof(int));
    if (!g->phon2sym) r.bad = 2;
    for (int k = 0; !r.bad && k < g->NP; ++k) {
      int32_t v;
      memcpy(&v, &p2s[k], 4);
      g->phon2sym[k] = v < ST_N_SYMBOLS ? v : -1;
    }
  }
  st_free(p2s);
  st_free(buf);
  if (r.bad) {
    gru_free(g);
    st_seterr(r.bad == 2 ? "steertts front: out of memory" : "steertts front: truncated asset");
    return -1;
  }
  return 0;
}

/* sum_i q[i] * h[i] (the row scale is applied by the caller). numpy computes h @ W in float32 (OpenBLAS sgemv, its
 * own summation order), so any implementation differs from it by float32 rounding noise (~1e-6 on the logits); only
 * a top-2 logit gap below that could flip an argmax (the golden G2P set's smallest gap is 3.4e-5).
 * Canonical arithmetic (identical on every path): 32 float lanes, lane k accumulates float(q[i]) * h[i] for
 * i = k (mod 32) in increasing i; lanes are combined as 4-lane vectors t = ((A0+A1)+(A2+A3))+((A4+A5)+(A6+A7))
 * (A_m = lanes 4m..4m+3) in float, then (t0+t1)+(t2+t3) in double. */
static double reduce_lanes(const float* A) {
  float t[4];
  for (int l = 0; l < 4; ++l)
    t[l] = ((A[l] + A[4 + l]) + (A[8 + l] + A[12 + l])) + ((A[16 + l] + A[20 + l]) + (A[24 + l] + A[28 + l]));
  return ((double)t[0] + t[1]) + ((double)t[2] + t[3]);
}
#if defined(ST_G2P_SSE2)
static void widen16(const int8_t* q, __m128* f0, __m128* f1, __m128* f2, __m128* f3) {  /* 16 x int8 -> 4 x float4 */
  __m128i b = _mm_loadu_si128((const __m128i*)(const void*)q);
  __m128i lo = _mm_srai_epi16(_mm_unpacklo_epi8(b, b), 8);  /* sign-extend int8 -> int16 (SSE2 only) */
  __m128i hi = _mm_srai_epi16(_mm_unpackhi_epi8(b, b), 8);
  *f0 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(lo, lo), 16));
  *f1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(lo, lo), 16));
  *f2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(hi, hi), 16));
  *f3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(hi, hi), 16));
}
static double dot_q(const int8_t* q, const float* h, int n) {  /* n % 32 == 0; accumulators spelled out (registers) */
  __m128 a0 = _mm_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
  for (int i = 0; i < n; i += 32) {
    __m128 w0, w1, w2, w3, w4, w5, w6, w7;
    widen16(q + i, &w0, &w1, &w2, &w3);
    widen16(q + i + 16, &w4, &w5, &w6, &w7);
    const float* x = h + i;
    a0 = _mm_add_ps(a0, _mm_mul_ps(w0, _mm_loadu_ps(x)));
    a1 = _mm_add_ps(a1, _mm_mul_ps(w1, _mm_loadu_ps(x + 4)));
    a2 = _mm_add_ps(a2, _mm_mul_ps(w2, _mm_loadu_ps(x + 8)));
    a3 = _mm_add_ps(a3, _mm_mul_ps(w3, _mm_loadu_ps(x + 12)));
    a4 = _mm_add_ps(a4, _mm_mul_ps(w4, _mm_loadu_ps(x + 16)));
    a5 = _mm_add_ps(a5, _mm_mul_ps(w5, _mm_loadu_ps(x + 20)));
    a6 = _mm_add_ps(a6, _mm_mul_ps(w6, _mm_loadu_ps(x + 24)));
    a7 = _mm_add_ps(a7, _mm_mul_ps(w7, _mm_loadu_ps(x + 28)));
  }
  float A[32];
  _mm_storeu_ps(A, a0);
  _mm_storeu_ps(A + 4, a1);
  _mm_storeu_ps(A + 8, a2);
  _mm_storeu_ps(A + 12, a3);
  _mm_storeu_ps(A + 16, a4);
  _mm_storeu_ps(A + 20, a5);
  _mm_storeu_ps(A + 24, a6);
  _mm_storeu_ps(A + 28, a7);
  return reduce_lanes(A);
}
#elif defined(ST_G2P_NEON)
static double dot_q(const int8_t* q, const float* h, int n) {  /* n % 32 == 0 */
  float32x4_t acc[8];
  for (int m = 0; m < 8; ++m) acc[m] = vdupq_n_f32(0.0f);
  for (int i = 0; i < n; i += 32) {
    for (int half = 0; half < 2; ++half) {
      int8x16_t b = vld1q_s8(q + i + 16 * half);
      int16x8_t lo = vmovl_s8(vget_low_s8(b)), hi = vmovl_s8(vget_high_s8(b));
      int32x4_t w[4] = {vmovl_s16(vget_low_s16(lo)), vmovl_s16(vget_high_s16(lo)), vmovl_s16(vget_low_s16(hi)),
                        vmovl_s16(vget_high_s16(hi))};
      for (int m = 0; m < 4; ++m) {
        float32x4_t x = vld1q_f32(h + i + 16 * half + 4 * m);
        acc[4 * half + m] = vaddq_f32(acc[4 * half + m], vmulq_f32(vcvtq_f32_s32(w[m]), x));  /* no fused mla */
      }
    }
  }
  float A[32];
  for (int m = 0; m < 8; ++m) vst1q_f32(A + 4 * m, acc[m]);
  return reduce_lanes(A);
}
#else
static double dot_q(const int8_t* q, const float* h, int n) {
  float A[32] = {0};
  for (int i = 0; i < n; i += 32)
    for (int k = 0; k < 32; ++k) A[k] += (float)q[i + k] * h[i + k];
  return reduce_lanes(A);
}
#endif
static double sig(double x) { return 1.0 / (1.0 + exp(-x)); }

/* gru_np.GruG2P._cell: hp = h @ W_hh.T + b_hh; r, z = sigmoid; n = tanh(xn + r * hn); h = (1 - z) n + z h */
static void cell(const st_gru* g, const float* xp, double* h, const int8_t* q, const float* s, const float* b) {
  const int H = g->H;
  double hp[3 * 512];
  float hf[512];
  for (int j = 0; j < H; ++j) hf[j] = (float)h[j];
  for (int j = 0; j < 3 * H; ++j) {
    double d = (double)s[j] * dot_q(q + (size_t)j * H, hf, H);
    hp[j] = d + b[j];
  }
  for (int j = 0; j < H; ++j) {
    double r = sig((double)xp[j] + hp[j]);
    double z = sig((double)xp[H + j] + hp[H + j]);
    double rn = r * hp[2 * H + j];
    double nn = tanh((double)xp[2 * H + j] + rn);
    double a = (1 - z) * nn, c = z * h[j];
    h[j] = a + c;
  }
}

int gru_predict_raw(const st_gru* g, const char* word, int* out, double* margin) {
  const int H = g->H;
  double h[512] = {0}, lo[4096];
  float hf[512];
  for (const char* c = word;; ++c) {  /* _G2I.get(c, 1), then </s> = 2 */
    int id = !*c ? 2 : (*c >= 'a' && *c <= 'z') ? *c - 'a' + 3 : 1;
    if (id >= g->NG) id = 1;
    cell(g, g->enc_x + (size_t)id * 3 * H, h, g->enc_q, g->enc_s, g->enc_b);
    if (!*c) break;
  }
  int prev = 2, n = 0;
  double gap = 1e9;
  for (int step = 0; step < 20; ++step) {
    cell(g, g->dec_x + (size_t)prev * 3 * H, h, g->dec_q, g->dec_s, g->dec_b);
    int p = 0;
    for (int j = 0; j < H; ++j) hf[j] = (float)h[j];
    for (int k = 0; k < g->NP; ++k) {
      double d = (double)g->fc_s[k] * dot_q(g->fc_q + (size_t)k * H, hf, H);
      lo[k] = d + g->fc_b[k];
      if (lo[k] > lo[p]) p = k;  /* numpy argmax: first maximum */
    }
    if (margin) {
      double second = -1e300;
      for (int k = 0; k < g->NP; ++k)
        if (k != p && lo[k] > second) second = lo[k];
      if (lo[p] - second < gap) gap = lo[p] - second;
    }
    if (p == 3) break;
    out[n++] = p;
    prev = p;
  }
  if (margin) *margin = gap;
  return n;
}

/* ------------------------------------------------------------------ prediction cache (like Python's lru_cache) */
#define CACHE_MAX 16384
#define CACHE_SLOTS 32768
typedef struct {
  char* w;
  signed char ph[20];
  int n;
} centry;
struct st_g2p_cache {
#ifdef _WIN32
  SRWLOCK lock;
#else
  pthread_mutex_t lock;
#endif
  centry* slot;
  int count;
};
static void c_lock(st_g2p_cache* c) {
#ifdef _WIN32
  AcquireSRWLockExclusive(&c->lock);
#else
  pthread_mutex_lock(&c->lock);
#endif
}
static void c_unlock(st_g2p_cache* c) {
#ifdef _WIN32
  ReleaseSRWLockExclusive(&c->lock);
#else
  pthread_mutex_unlock(&c->lock);
#endif
}
static uint32_t c_hash(const char* s) {
  uint32_t h = 2166136261u;
  while (*s) h = (h ^ (unsigned char)*s++) * 16777619u;
  return h;
}
static st_g2p_cache* cache_new(void) {
  st_g2p_cache* c = calloc(1, sizeof *c);  /* the table itself is allocated on the first insertion */
  if (!c) return NULL;
#ifdef _WIN32
  InitializeSRWLock(&c->lock);
#else
  pthread_mutex_init(&c->lock, NULL);
#endif
  return c;
}
static void cache_clear(st_g2p_cache* c) {
  for (int k = 0; c->slot && k < CACHE_SLOTS; ++k) {
    free(c->slot[k].w);
    c->slot[k].w = NULL;
  }
  c->count = 0;
}
static void cache_free(st_g2p_cache* c) {
  if (!c) return;
  cache_clear(c);
#ifndef _WIN32
  pthread_mutex_destroy(&c->lock);
#endif
  free(c->slot);
  free(c);
}

/* GRU prediction as symbol ids (outputs to_phonemes would drop are removed), cached */
static int predict_cached(const st_g2p* g, const char* w, int* out) {
  st_g2p_cache* c = g->cache;
  uint32_t h = c_hash(w);
  c_lock(c);
  for (uint32_t k = h % CACHE_SLOTS; c->slot && c->slot[k].w; k = (k + 1) % CACHE_SLOTS) {
    if (!strcmp(c->slot[k].w, w)) {
      int n = c->slot[k].n;
      for (int i = 0; i < n; ++i) out[i] = c->slot[k].ph[i];
      c_unlock(c);
      return n;
    }
  }
  c_unlock(c);
  int raw[20], nr = gru_predict_raw(&g->gru, w, raw, NULL), n = 0;
  for (int i = 0; i < nr; ++i)
    if (g->gru.phon2sym[raw[i]] >= 0) out[n++] = g->gru.phon2sym[raw[i]];
  size_t wl = strlen(w);
  char* key = malloc(wl + 1);
  if (!key) return n;  /* not cached: still a correct answer */
  memcpy(key, w, wl + 1);
  c_lock(c);
  if (!c->slot && !(c->slot = calloc(CACHE_SLOTS, sizeof *c->slot))) {
    c_unlock(c);
    free(key);
    return n;
  }
  if (c->count >= CACHE_MAX) cache_clear(c);
  uint32_t k = h % CACHE_SLOTS;
  for (; c->slot[k].w; k = (k + 1) % CACHE_SLOTS)
    if (!strcmp(c->slot[k].w, w)) break;
  if (c->slot[k].w) {
    free(key);  /* another thread inserted it meanwhile */
  } else {
    c->slot[k].w = key;
    c->slot[k].n = n;
    for (int i = 0; i < n; ++i) c->slot[k].ph[i] = (signed char)out[i];
    c->count++;
  }
  c_unlock(c);
  return n;
}

/* ------------------------------------------------------------------ Hetero */
static int cmp_kv(const void* a, const void* b) {
  const st_kv *x = (const st_kv*)a, *y = (const st_kv*)b;
  int c = strcmp(x->k, y->k);
  return c ? c : (x->idx < y->idx ? -1 : x->idx > y->idx);  /* tie: file order */
}
static int cmp_hom(const void* a, const void* b) {
  const st_hom *x = (const st_hom*)a, *y = (const st_hom*)b;
  int c = strcmp(x->w, y->w);
  return c ? c : (x->n1 < y->n1 ? -1 : 1);  /* never equal after dedupe; order irrelevant */
}
static const st_kv* kv_find(const st_kv* v, int n, const char* k) {
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2, c = strcmp(v[mid].k, k);
    if (!c) return &v[mid];
    if (c < 0)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return NULL;
}
static const st_hom* hom_find(const st_hetero* h, const char* w) {
  int lo = 0, hi = h->nhom - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2, c = strcmp(h->hom[mid].w, w);
    if (!c) return &h->hom[mid];
    if (c < 0)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return NULL;
}

static void het_free(st_hetero* h) {
  for (int k = 0; k < h->nhom; ++k) {
    free(h->hom[k].w);
    free(h->hom[k].p1);
    free(h->hom[k].p2);
    free(h->hom[k].pos);
  }
  for (int k = 0; k < h->nw; ++k) free(h->w[k].k);
  for (int k = 0; k < h->ntd; ++k) {
    free(h->td[k].k);
    free(h->td[k].s);
  }
  free(h->hom);
  free(h->w);
  free(h->td);
  memset(h, 0, sizeof *h);
}

static char* dupn(const char* s, size_t n) {
  char* d = malloc(n + 1);
  if (d) {
    memcpy(d, s, n);
    d[n] = 0;
  }
  return d;
}
static int parse_phones(const st_front* f, char* s, int** out, int* n) {
  int cap = 0;
  *out = NULL;
  *n = 0;
  for (char* t = s; *t;) {
    while (*t == ' ') ++t;
    if (!*t) break;
    char* e = t;
    while (*e && *e != ' ') ++e;
    char sv = *e;
    *e = 0;
    int id = st_front_symbol_id(f, t);
    *e = sv;
    if (id < 0) return -1;
    if (*n == cap) {
      cap = cap ? 2 * cap : 8;
      int* p = realloc(*out, (size_t)cap * sizeof(int));
      if (!p) return -1;
      *out = p;
    }
    (*out)[(*n)++] = id;
    t = e;
  }
  return 0;
}

/* entries with equal keys: keep the last (dict assignment); arrays are sorted with file order as tie-break */
static int het_load(st_hetero* h, const char* path, const st_front* f) {
  memset(h, 0, sizeof *h);
  char* buf;
  size_t len;
  if (st_read_file(path, &buf, &len)) {
    st_seterr("steertts front: cannot open %s", path);
    return -1;
  }
  int caph = 0, capw = 0, capt = 0, rc = 0;
  h->has_td = 1;
  for (char* line = buf; line < buf + len && !rc;) {
    char* nl = memchr(line, '\n', (size_t)(buf + len - line));
    char* end = nl ? nl : buf + len;
    *end = 0;
    if (end > line) {
      char* fld[6];
      int nf = 0;
      for (char* p = line; nf < 6;) {
        fld[nf++] = p;
        char* t = strchr(p, '\t');
        if (!t) break;
        *t = 0;
        p = t + 1;
      }
      if (!strcmp(fld[0], "H") && nf == 5) {
        if (h->nhom == caph) {
          caph = caph ? 2 * caph : 256;
          st_hom* v = realloc(h->hom, (size_t)caph * sizeof *v);
          if (!v) { rc = -2; break; }
          h->hom = v;
        }
        st_hom* e = &h->hom[h->nhom];
        memset(e, 0, sizeof *e);
        e->w = dupn(fld[1], strlen(fld[1]));
        e->pos = dupn(fld[4], strlen(fld[4]));
        h->nhom++;
        if (!e->w || !e->pos) { rc = -2; break; }
        if (parse_phones(f, fld[2], &e->p1, &e->n1) || parse_phones(f, fld[3], &e->p2, &e->n2)) rc = -1;
      } else if (!strcmp(fld[0], "W") && nf == 3) {
        if (h->nw == capw) {
          capw = capw ? 2 * capw : 1024;
          st_kv* v = realloc(h->w, (size_t)capw * sizeof *v);
          if (!v) { rc = -2; break; }
          h->w = v;
        }
        st_kv* e = &h->w[h->nw];
        e->k = dupn(fld[1], strlen(fld[1]));
        e->s = NULL;
        e->v = atoi(fld[2]);
        e->idx = h->nw;
        h->nw++;
        if (!e->k) { rc = -2; break; }
      } else if (!strcmp(fld[0], "T") && nf == 3) {
        if (h->ntd == capt) {
          capt = capt ? 2 * capt : 1024;
          st_kv* v = realloc(h->td, (size_t)capt * sizeof *v);
          if (!v) { rc = -2; break; }
          h->td = v;
        }
        st_kv* e = &h->td[h->ntd];
        e->k = dupn(fld[1], strlen(fld[1]));
        e->s = dupn(fld[2], strlen(fld[2]));
        e->v = 0;
        e->idx = h->ntd;
        h->ntd++;
        if (!e->k || !e->s) { rc = -2; break; }
      } else if (!strcmp(fld[0], "N")) {
        h->has_td = 0;
      } else {
        rc = -1;
      }
    }
    line = end + 1;
  }
  st_free(buf);
  if (rc) {
    het_free(h);
    st_seterr(rc == -2 ? "steertts front: out of memory" : "steertts front: bad hetero.tsv line");
    return -1;
  }
  /* W: sort by (key, file index); keep the last of each key */
  {
    qsort(h->w, (size_t)h->nw, sizeof *h->w, cmp_kv);
    int m = 0;
    for (int k = 0; k < h->nw; ++k) {
      if (k + 1 < h->nw && !strcmp(h->w[k].k, h->w[k + 1].k)) {
        free(h->w[k].k);
        continue;
      }
      h->w[m++] = h->w[k];
    }
    h->nw = m;
  }
  {
    qsort(h->td, (size_t)h->ntd, sizeof *h->td, cmp_kv);
    int m = 0;
    for (int k = 0; k < h->ntd; ++k) {
      if (k + 1 < h->ntd && !strcmp(h->td[k].k, h->td[k + 1].k)) {
        free(h->td[k].k);
        free(h->td[k].s);
        continue;
      }
      h->td[m++] = h->td[k];
    }
    h->ntd = m;
  }
  {
    /* H: keep the last entry per word (sort is not stable, so drop earlier duplicates first, O(n^2) on 371 rows) */
    int m = 0;
    for (int k = 0; k < h->nhom; ++k) {
      int later = 0;
      for (int j = k + 1; j < h->nhom && !later; ++j) later = !strcmp(h->hom[k].w, h->hom[j].w);
      if (later) {
        free(h->hom[k].w);
        free(h->hom[k].p1);
        free(h->hom[k].p2);
        free(h->hom[k].pos);
        continue;
      }
      h->hom[m++] = h->hom[k];
    }
    h->nhom = m;
    qsort(h->hom, (size_t)h->nhom, sizeof *h->hom, cmp_hom);
  }
  return 0;
}

static int isalpha_ascii(const char* t) {
  if (!*t) return 0;
  for (; *t; ++t)
    if (!((*t >= 'a' && *t <= 'z') || (*t >= 'A' && *t <= 'Z'))) return 0;
  return 1;
}
static void last2(fx* x, const char* t, st_sb* o) {
  size_t n = strlen(t);
  sb_cat(x, o, n > 2 ? t + n - 2 : t);
}
static void suf(fx* x, const char* t, st_sb* o) {
  if (isalpha_ascii(t) && strlen(t) > 2)
    last2(x, t, o);
  else
    sb_cat(x, o, t);
}
static void het_tag(fx* x, const st_hetero* h, const char* t, st_sb* o) {
  const st_kv* e = kv_find(h->td, h->ntd, t);
  if (e) {
    sb_cat(x, o, e->s);
    return;
  }
  if (!strcmp(t, "<s>") || !strcmp(t, "</s>")) {
    sb_cat(x, o, t);
    return;
  }
  int any = 0;
  for (const char* c = t; *c; ++c)
    if ((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z')) any = 1;
  if (!any) {
    sb_cat(x, o, "P");
    return;
  }
  sb_ch(x, o, 'U');
  last2(x, t, o);
}

/* averaged-perceptron homograph disambiguation: 1 = first pronunciation */
static int het_first(fx* x, const st_hetero* h, const st_hom* hom, char** toks, int ntok, int j) {
  const char* w = toks[j];
#define AT(k) ((k) < 0 ? "<s>" : (k) >= ntok ? "</s>" : toks[(k)])
  const char *p1 = AT(j - 1), *p2 = AT(j - 2), *n1 = AT(j + 1), *n2 = AT(j + 2);
#undef AT
  st_sb F[21] = {{0}};
  int nf = 0;
  sb_cat(x, &F[nf++], "b");
  sb_cat(x, &F[nf], "p1="); sb_cat(x, &F[nf++], p1);
  sb_cat(x, &F[nf], "n1="); sb_cat(x, &F[nf++], n1);
  sb_cat(x, &F[nf], "p2="); sb_cat(x, &F[nf++], p2);
  sb_cat(x, &F[nf], "n2="); sb_cat(x, &F[nf++], n2);
  sb_cat(x, &F[nf], "p1s="); suf(x, p1, &F[nf++]);
  sb_cat(x, &F[nf], "n1s="); suf(x, n1, &F[nf++]);
  sb_cat(x, &F[nf], "p1n1="); sb_cat(x, &F[nf], p1); sb_ch(x, &F[nf], '_'); sb_cat(x, &F[nf++], n1);
  sb_cat(x, &F[nf], "p2p1="); sb_cat(x, &F[nf], p2); sb_ch(x, &F[nf], '_'); sb_cat(x, &F[nf++], p1);
  sb_cat(x, &F[nf], "n1n2="); sb_cat(x, &F[nf], n1); sb_ch(x, &F[nf], '_'); sb_cat(x, &F[nf++], n2);
  sb_cat(x, &F[nf], "p2s="); suf(x, p2, &F[nf++]);
  sb_cat(x, &F[nf], "n2s="); suf(x, n2, &F[nf++]);
  if (h->has_td) {
    st_sb A = {0}, B = {0}, C = {0}, D = {0};
    het_tag(x, h, p2, &A);
    het_tag(x, h, p1, &B);
    het_tag(x, h, n1, &C);
    het_tag(x, h, n2, &D);
    const char *a = sb_cstr(&A), *b = sb_cstr(&B), *c = sb_cstr(&C), *d = sb_cstr(&D);
    sb_cat(x, &F[nf], "p1t="); sb_cat(x, &F[nf++], b);
    sb_cat(x, &F[nf], "n1t="); sb_cat(x, &F[nf++], c);
    sb_cat(x, &F[nf], "p2t="); sb_cat(x, &F[nf++], a);
    sb_cat(x, &F[nf], "n2t="); sb_cat(x, &F[nf++], d);
    sb_cat(x, &F[nf], "p1tn1t="); sb_cat(x, &F[nf], b); sb_ch(x, &F[nf], '_'); sb_cat(x, &F[nf++], c);
    sb_cat(x, &F[nf], "p2tp1t="); sb_cat(x, &F[nf], a); sb_ch(x, &F[nf], '_'); sb_cat(x, &F[nf++], b);
    sb_cat(x, &F[nf], "n1tn2t="); sb_cat(x, &F[nf], c); sb_ch(x, &F[nf], '_'); sb_cat(x, &F[nf++], d);
    sb_cat(x, &F[nf], "p1n1t="); sb_cat(x, &F[nf], p1); sb_ch(x, &F[nf], '_'); sb_cat(x, &F[nf++], c);
    sb_cat(x, &F[nf], "p1tn1="); sb_cat(x, &F[nf], b); sb_ch(x, &F[nf], '_'); sb_cat(x, &F[nf++], n1);
    sb_free(&A); sb_free(&B); sb_free(&C); sb_free(&D);
  }
  long s = 0;
  st_sb key = {0};
  for (int k = 0; k < nf; ++k) {
    key.n = 0;
    sb_cat(x, &key, "w:"); sb_cat(x, &key, w); sb_ch(x, &key, '|'); sb_cat(x, &key, sb_cstr(&F[k]));
    const st_kv* e = x->err ? NULL : kv_find(h->w, h->nw, sb_cstr(&key));
    if (e) s += e->v;
    key.n = 0;
    sb_cat(x, &key, "g:"); sb_cat(x, &key, hom->pos); sb_ch(x, &key, '|'); sb_cat(x, &key, sb_cstr(&F[k]));
    e = x->err ? NULL : kv_find(h->w, h->nw, sb_cstr(&key));
    if (e) s += e->v;
  }
  sb_free(&key);
  for (int k = 0; k < nf; ++k) sb_free(&F[k]);
  return s > 0;
}

/* ------------------------------------------------------------------ EdgeG2p */
int g2p_init(st_g2p* g, const char* dir, const st_ucd* u, const st_front* f) {
  memset(g, 0, sizeof *g);
  g->u = u;
  size_t dl = strlen(dir);
  char* path = malloc(dl + 32);
  if (!path) {
    st_seterr("steertts front: out of memory");
    return -1;
  }
#define PATH(name) (memcpy(path, dir, dl), strcpy(path + dl, "/" name), path)
  int ok = numwords_init(&g->nw, u) == 0 && lex_load(&g->cmu, PATH("lexicon.bin")) == 0 &&
           lex_load(&g->oov, PATH("oov.bin")) == 0 && gru_load(&g->gru, PATH("gru.bin")) == 0 &&
           het_load(&g->het, PATH("hetero.tsv"), f) == 0;
#undef PATH
  free(path);
  if (ok) {
    g->hang = st_rx_compile(u, "([^a-z0-9])\\1{3,}", 0);
    g->word = st_rx_compile(u,
                            "(?<!@)[a-z0-9]+(?:[.\\-][a-z0-9]+){0,126}[.](?:[a-z]{2,13})\\b/?(?!@)"
                            "|[a-z](?:[a-z]|['\\-_])+[a-z]"
                            "|[a-z0-9_]+"
                            "|\\.(?:\\s*\\.)+"
                            "|\\S",
                            0);
    ok = g->hang && g->word;
  }
  if (ok) {
    g->cache = cache_new();
    if (!g->cache) {
      st_seterr("steertts front: out of memory");
      ok = 0;
    }
  }
  if (ok) {
    for (int c = 0; c < 128; ++c) g->punct_id[c] = -2;
    for (const char* p = ",.?!;:-"; *p; ++p) {
      char s[2] = {*p, 0};
      g->punct_id[(int)*p] = st_front_symbol_id(f, s);
    }
  }
  if (!ok) {
    g2p_free(g);
    return -1;
  }
  return 0;
}

void g2p_free(st_g2p* g) {
  numwords_free(&g->nw);
  st_free(g->cmu.buf);
  st_free(g->oov.buf);
  gru_free(&g->gru);
  het_free(&g->het);
  st_rx_free(g->hang);
  st_rx_free(g->word);
  cache_free(g->cache);
  memset(g, 0, sizeof *g);
}

int g2p_in_cmu(const st_g2p* g, const char* w) {
  for (const char* c = w; *c; ++c)
    if ((unsigned char)*c >= 128) return 0;
  return lex_get(&g->cmu, w, NULL, NULL);
}

static void repl_triple(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) {
  for (int k = 0; k < 3; ++k) rxm_put(x, out, m, s, 1);
}

/* tok.word_tokenize on the cleaned ascii text; tokens are malloc'd */
static void tokenize(fx* x, const st_g2p* g, const char* a, char*** toks, int* ntok) {
  st_u32s A = {0}, t = {0};
  u32s_ascii(x, &A, a);
  st_rx_sub(x, g->hang, A.p, A.n, repl_triple, NULL, &t);
  u32s_free(&A);
  int pos = 0, cap = 0;
  st_rxm m;
  *toks = NULL;
  *ntok = 0;
  while (st_rx_next(x, g->word, t.p, t.n, &pos, &m)) {
    if (*ntok == cap) {
      cap = cap ? 2 * cap : 32;
      char** v = realloc(*toks, (size_t)cap * sizeof *v);
      if (!v) {
        fx_fail(x, "MemoryError: out of memory");
        break;
      }
      *toks = v;
    }
    int len = m.end - m.start;
    char* w = malloc((size_t)len + 1);
    if (!w) {
      fx_fail(x, "MemoryError: out of memory");
      break;
    }
    for (int i = 0; i < len; ++i) w[i] = (char)t.p[m.start + i];
    w[len] = 0;
    (*toks)[(*ntok)++] = w;
  }
  u32s_free(&t);
}

/* pronunciation of toks[j] as symbol ids (at most 64) */
static int pron(const st_g2p* g, fx* x, char** toks, int ntok, int j, int* ph) {
  const char* w = toks[j];
  int has_letter = 0, n = 0;
  for (const char* c = w; *c; ++c)
    if (*c >= 'a' && *c <= 'z') has_letter = 1;
  if (!has_letter) {  /* non-alpha token -> itself; only single punctuation marks are symbols (tokens are ascii) */
    if (w[0] && !w[1] && (unsigned char)w[0] < 128 && g->punct_id[(int)w[0]] >= 0) {
      ph[0] = g->punct_id[(int)w[0]];
      return 1;
    }
    return 0;
  }
  const st_hom* h = hom_find(&g->het, w);
  if (h) {
    int first = het_first(x, &g->het, h, toks, ntok, j);
    const int* p = first ? h->p1 : h->p2;
    n = first ? h->n1 : h->n2;
    if (n > 64) n = 64;
    memcpy(ph, p, (size_t)n * sizeof(int));
    return n;
  }
  if (lex_get(&g->cmu, w, ph, &n)) return n ? n : predict_cached(g, w, ph);  /* member-only entry -> model */
  if (lex_get(&g->oov, w, ph, &n)) return n;                                  /* cached prediction (may be empty) */
  return predict_cached(g, w, ph);
}

static void replace_all(fx* x, st_sb* s, const char* from, const char* to) {
  if (!s->p || !strstr(s->p, from)) return;
  st_sb o = {0};
  size_t fl = strlen(from);
  const char* p = s->p;
  for (const char* e; (e = strstr(p, from)) != NULL; p = e + fl) {
    sb_put(x, &o, p, (int)(e - p));
    sb_cat(x, &o, to);
  }
  sb_cat(x, &o, p);
  sb_free(s);
  *s = o;
}

void g2p_call(fx* x, const st_g2p* g, const u32* text, int n, int** out, int* nout) {
  st_u32s t = {0};
  st_sb a = {0};
  *out = NULL;
  *nout = 0;
  nw_normalize_numbers(x, &g->nw, text, n, &t);
  for (int i = 0; i < t.n && !x->err; ++i) {  /* NFD, drop Mn, lower, keep [ a-z'.,?!-] */
    int l;
    const char* c = ucd_g2p_clean(g->u, t.p[i], &l);
    sb_put(x, &a, c, l);
  }
  u32s_free(&t);
  replace_all(x, &a, "i.e.", "that is");
  replace_all(x, &a, "e.g.", "for example");
  char** toks = NULL;
  int ntok = 0, cap = 0;
  if (!x->err) tokenize(x, g, sb_cstr(&a), &toks, &ntok);
  sb_free(&a);
  int* o = NULL;
  int no = 0;
  for (int j = 0; j < ntok && !x->err; ++j) {
    int ph[64];
    int np = pron(g, x, toks, ntok, j, ph);
    if (no + np + 1 > cap) {
      cap = 2 * (no + np + 1) + 32;
      int* v = realloc(o, (size_t)cap * sizeof(int));
      if (!v) {
        fx_fail(x, "MemoryError: out of memory");
        break;
      }
      o = v;
    }
    memcpy(o + no, ph, (size_t)np * sizeof(int));
    no += np;
    o[no++] = TOK_SPACE;
  }
  if (no) --no;  /* drop the trailing separator */
  for (int j = 0; j < ntok; ++j) free(toks[j]);
  free(toks);
  if (x->err) {
    free(o);
    return;
  }
  *out = o;
  *nout = no;
}
