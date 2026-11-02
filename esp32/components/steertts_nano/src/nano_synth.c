/* SteerTTS Nano synthesis: conditioning, encoder, predictors, length regulator, chunked decoder + vocoder, iSTFT.
 *
 * Streaming: every conv stack keeps, per layer, the last k-1 input rows of the previous chunk (the halo) and emits
 * its output k/2 rows late, so chunked output is identical to running the whole utterance at once (zero padding at
 * both ends). Each chunk is one pass over the decoder and vocoder weights (batched int8 GEMM over all frames of the
 * chunk), which is what makes weights in memory-mapped flash affordable. */
#include <math.h>

#include "nano_internal.h"
#include "nano_tables.h"

#define HEAD_BATCH 16   /* vocoder head frames per weight pass (its 1026-wide output is the largest activation) */

typedef struct {
  const nano_blockw* w;
  int d, ff, k, act;
  int8_t* hist;
  int fill;
  const float *g, *b;   /* norm affine: AdaLN (1 + scale, shift) or LayerNorm (gamma, beta) */
} nstage;

typedef struct {
  const nano_qlayer* q;
  int8_t* hist;
  int fill;
} cstage;

struct nano_ctx {
  const nano_model* m;
  nano_arena ar;
  size_t mark;   /* end of the persistent part */
  int chunk, first_chunk, hr, hp;
  /* persistent per utterance */
  float* cond;
  int8_t* H;      /* (maxph + 2 hp) x enc_dim; later the decoder inputs P (maxph x dec_dim) */
  uint16_t* dur;
  float *f0, *en;
  float* dec_ada;   /* dec_layers x 2 x dec_dim */
  /* phase buffers */
  int8_t *A, *B, *T1, *FF, *MEL, *HO;
  float *Z, *OLA, *lut_mag, *lut_cos, *lut_sin;
  int16_t* pcm;
  nstage st[NANO_MAX_LAYERS];
  nstage vst[NANO_MAX_LAYERS];
  cstage emb;
  int8_t gelu[256];
  /* iSTFT state */
  long frame_idx, total_frames, emitted;
  nano_pcm_fn fn;
  void* user;
  int stop;
  nano_profile prof;
};

const char* nano_stage_name(int s) {
  static const char* n[NANO_ST_COUNT] = {"cond", "encoder", "predictors", "pitch/energy + proj", "decoder",
                                         "mel proj", "vocoder backbone", "vocoder head", "istft"};
  return s >= 0 && s < NANO_ST_COUNT ? n[s] : "?";
}
const char* nano_op_name(int o) {
  static const char* n[NANO_OP_COUNT] = {"int8 gemm", "depthwise", "norm", "elementwise", "fft"};
  return o >= 0 && o < NANO_OP_COUNT ? n[o] : "?";
}

void nano_request_default(nano_request* r) {
  memset(r, 0, sizeof *r);
  r->pace = 1.0f;
}

const nano_profile* nano_ctx_profile(const nano_ctx* c) { return &c->prof; }
void nano_ctx_profile_reset(nano_ctx* c) {
  size_t pk = c->prof.arena_peak;
  memset(&c->prof, 0, sizeof c->prof);
  c->prof.arena_peak = pk;
  c->prof.hash_enc = c->prof.hash_p = c->prof.hash_mel = c->prof.hash_head = 2166136261u;
}

/* ------------------------------------------------------------------ layout */
typedef struct {
  size_t ctx, persist, enc, pred, dv, total;
} layout;

static size_t al16(size_t n) { return (n + 15) & ~(size_t)15; }
static int imax(int a, int b) { return a > b ? a : b; }

static void compute_layout(const nano_model* m, int chunk, int first, layout* L) {
  const int* c = m->cfg;
  int C = imax(chunk, first);
  int hr = imax(imax(c[NANO_CFG_ENC_K], c[NANO_CFG_DEC_K]), imax(c[NANO_CFG_VOC_K], c[NANO_CFG_VOC_EMBED_K])) - 1;
  int hp = imax(c[NANO_CFG_PRED_K], c[NANO_CFG_PE_K]) / 2;
  int mp = c[NANO_CFG_MAX_PHONEMES], de = c[NANO_CFG_ENC_DIM], dd = c[NANO_CFG_DEC_DIM], dv = c[NANO_CFG_VOC_DIM];
  int dp = c[NANO_CFG_PRED_DIM], dc = c[NANO_CFG_D_COND], nm = c[NANO_CFG_N_MELS], nfft = c[NANO_CFG_N_FFT];
  L->ctx = al16(sizeof(nano_ctx));
  L->persist = al16(dc * 4) + al16((size_t)(mp + 2 * hp) * imax(de, dd)) + al16(mp * 2) + 2 * al16(mp * 4) +
               al16((size_t)c[NANO_CFG_DEC_LAYERS] * 2 * dd * 4);
  int sp_e = c[NANO_CFG_ENC_LAYERS] * (c[NANO_CFG_ENC_K] / 2);
  int sp_dv = c[NANO_CFG_DEC_LAYERS] * (c[NANO_CFG_DEC_K] / 2) + c[NANO_CFG_VOC_EMBED_K] / 2 +
              c[NANO_CFG_VOC_LAYERS] * (c[NANO_CFG_VOC_K] / 2);
  int nre = imax(C, sp_e);   /* the flush (zero rows appended) runs in its own call when it would not fit */
  L->enc = al16((size_t)c[NANO_CFG_ENC_LAYERS] * 2 * de * 4) + 2 * al16((size_t)(hr + nre) * de) +
           al16((size_t)nre * de) + al16((size_t)nre * c[NANO_CFG_ENC_FF]) +
           (size_t)c[NANO_CFG_ENC_LAYERS] * al16((size_t)(c[NANO_CFG_ENC_K] - 1) * de);
  L->pred = 2 * al16((size_t)(mp + 2 * hp) * imax(dp, de));
  int nr = imax(C, sp_dv), dm = imax(dd, dv), fm = imax(c[NANO_CFG_DEC_FF], c[NANO_CFG_VOC_FF]);
  L->dv = 2 * al16((size_t)(hr + nr) * dm) + al16((size_t)(hr + nr) * imax(nm, dv)) + al16((size_t)nr * dm) +
          al16((size_t)nr * fm) + (size_t)c[NANO_CFG_DEC_LAYERS] * al16((size_t)(c[NANO_CFG_DEC_K] - 1) * dd) +
          al16((size_t)(c[NANO_CFG_VOC_EMBED_K] - 1) * nm) +
          (size_t)c[NANO_CFG_VOC_LAYERS] * al16((size_t)(c[NANO_CFG_VOC_K] - 1) * dv) +
          al16((size_t)HEAD_BATCH * (nfft + 2)) + 2 * al16((size_t)nfft * 4) + 3 * al16(256 * 4) +
          al16((size_t)c[NANO_CFG_HOP] * 2);
  size_t ph = L->enc > L->pred ? L->enc : L->pred;
  if (L->dv > ph) ph = L->dv;
  L->total = L->ctx + L->persist + ph + 16;
}

static void resolve_chunks(const nano_model* m, int chunk, int first, int* C, int* F) {
  *C = chunk > 0 ? chunk : m->cfg[NANO_CFG_CHUNK];
  *F = first > 0 ? first : (m->cfg[NANO_CFG_FIRST_CHUNK] > 0 ? m->cfg[NANO_CFG_FIRST_CHUNK] : *C);
  if (*F > *C) *F = *C;
}

size_t nano_arena_bytes(const nano_model* m, int chunk) {
  int C, F;
  resolve_chunks(m, chunk, 0, &C, &F);
  layout L;
  compute_layout(m, C, F, &L);
  return L.total;
}

nano_ctx* nano_ctx_create(const nano_model* m, void* arena, size_t bytes) {
  nano_arena ar = {(uint8_t*)arena, bytes, 0, 0};
  size_t pad = (16 - ((uintptr_t)arena & 15)) & 15;
  ar.top = pad;
  nano_ctx* c = (nano_ctx*)nano_alloc(&ar, sizeof(nano_ctx));
  if (!c) {
    nano_seterr("arena too small");
    return NULL;
  }
  memset(c, 0, sizeof *c);
  c->prof.hash_enc = c->prof.hash_p = c->prof.hash_mel = c->prof.hash_head = 2166136261u;
  c->m = m;
  c->ar = ar;
  /* the arena size is checked per request (nano_synthesize), since it depends on the chunk size */
  resolve_chunks(m, 0, 0, &c->chunk, &c->first_chunk);
  return c;
}

/* ------------------------------------------------------------------ helpers */
#define PROF_ST(c, s, ...)                               \
  do {                                                   \
    uint64_t t0s_ = nano_cycles();                       \
    __VA_ARGS__;                                         \
    (c)->prof.cycles[s] += nano_cycles() - t0s_;         \
  } while (0)

static void gemm(nano_ctx* c, int st, const int8_t* in, int in_stride, int rows, const nano_qlayer* q, int8_t* out,
                 int out_stride, int lo) {
  if (rows <= 0) return;
  int K = q->k * q->in;
  NANO_PROF_OP(c, NANO_OP_GEMM, nano_gemm_s8(in, in_stride, rows, q->w, K, q->b, q->m, q->s, out, out_stride, q->out,
                                             lo, 127));
  c->prof.macs[st] += (uint64_t)rows * q->out * K;
  c->prof.weight_bytes[st] += (uint64_t)q->out * K + 12u * q->out;
  c->prof.passes[st]++;
}

static void* xalloc(nano_ctx* c, size_t n) {
  void* p = nano_alloc(&c->ar, n);
  if (p) memset(p, 0, n);
  return p;
}

/* AdaLN scale / shift from the conditioning vector: out[0..d) = 1 + s, out[d..2d) = shift */
static void ada_params(const nano_model* m, const nano_blockw* w, int d, const float* cond, float* out) {
  int dc = m->cfg[NANO_CFG_D_COND];
  for (int o = 0; o < 2 * d; o++) {
    const int8_t* wr = w->ada_w + (size_t)o * dc;
    float acc = 0.f;
    for (int i = 0; i < dc; i++) acc += (float)wr[i] * cond[i];
    float v = acc * w->ada_ws[o] + w->ada_b[o];
    out[o] = o < d ? 1.0f + v : v;
  }
}

/* one streaming block: in rows [hr - fill, hr + n) of buffer `in` (history copied in front), output rows -> out */
static int run_block(nano_ctx* c, int stg, nstage* s, int8_t* in, int n, int flush, int8_t* out, int out_stride) {
  const nano_blockw* w = s->w;
  int d = s->d, k = s->k, p = k / 2, h = k - 1;
  int8_t* X = in + (size_t)(c->hr - s->fill) * d;
  memcpy(X, s->hist, (size_t)s->fill * d);
  int tot = s->fill + n;
  if (flush) {
    memset(X + (size_t)tot * d, 0, (size_t)p * d);
    tot += p;
  }
  int nout = tot - h;
  if (nout > 0) {
    NANO_PROF_OP(c, NANO_OP_DW, nano_dw_s8(X, nout, d, w->dw.w, k, w->dw.b, w->dw.m, w->dw.s, c->T1, -128, 127));
    c->prof.macs[stg] += (uint64_t)nout * d * k;
    c->prof.weight_bytes[stg] += (uint64_t)d * k + 12u * d;
    NANO_PROF_OP(c, NANO_OP_NORM, nano_norm_rows(c->T1, d, nout, d, w->dw.sc[1], s->g, s->b,
                                                 c->m->fcfg[NANO_FCFG_LN_EPS], w->pw1.sc[0], c->T1, d));
    gemm(c, stg, c->T1, d, nout, &w->pw1, c->FF, s->ff, s->act == NANO_ACT_RELU ? 0 : -128);
    if (s->act == NANO_ACT_GELU)
      NANO_PROF_OP(c, NANO_OP_ELEM, nano_gelu_lut(w->pw1.sc[1], w->pw2.sc[0], c->gelu);
                   nano_lut_rows(c->FF, nout * s->ff, c->gelu));
    gemm(c, stg, c->FF, s->ff, nout, &w->pw2, c->T1, d, -128);
    NANO_PROF_OP(c, NANO_OP_ELEM, nano_residual(X + (size_t)p * d, d, c->T1, nout, d, w->dw.sc[0], w->pw2.sc[1],
                                                w->res, out, out_stride));
  } else {
    nout = 0;
  }
  int keep = tot < h ? tot : h;
  memmove(s->hist, X + (size_t)(tot - keep) * d, (size_t)keep * d);
  s->fill = keep;
  return nout;
}

/* a stack of blocks of width d over ping-pong buffers; input rows at A + hr*d; the last block writes to `out` */
static int run_stack(nano_ctx* c, int stg, nstage* st, int L, int n, int flush, int8_t* out, int out_stride) {
  int8_t *cur = c->A, *nxt = c->B;
  for (int i = 0; i < L; i++) {
    int d = st[i].d;
    int last = i == L - 1;
    n = run_block(c, stg, &st[i], cur, n, flush, last ? out : nxt + (size_t)c->hr * d, last ? out_stride : d);
    int8_t* t = cur;
    cur = nxt;
    nxt = t;
  }
  return n;
}

/* full conv stage with history (vocoder input conv): in rows at in + hr*d_in */
static int run_conv(nano_ctx* c, int stg, cstage* s, int8_t* in, int n, int flush, int8_t* out, int out_stride) {
  const nano_qlayer* q = s->q;
  int d = q->in, k = q->k, p = k / 2, h = k - 1;
  int8_t* X = in + (size_t)(c->hr - s->fill) * d;
  memcpy(X, s->hist, (size_t)s->fill * d);
  int tot = s->fill + n;
  if (flush) {
    memset(X + (size_t)tot * d, 0, (size_t)p * d);
    tot += p;
  }
  int nout = tot - h;
  if (nout > 0)
    gemm(c, stg, X, d, nout, q, out, out_stride, -128);
  else
    nout = 0;
  int keep = tot < h ? tot : h;
  memmove(s->hist, X + (size_t)(tot - keep) * d, (size_t)keep * d);
  s->fill = keep;
  return nout;
}

static void init_stage(nano_ctx* c, nstage* s, const nano_blockw* w, int d, int ff, int k, int act, const float* g,
                       const float* b) {
  s->w = w;
  s->d = d;
  s->ff = ff;
  s->k = k;
  s->act = act;
  s->g = g;
  s->b = b;
  s->hist = (int8_t*)xalloc(c, (size_t)(k - 1) * d);
  s->fill = k / 2;   /* k/2 zero rows: the left zero padding */
}

static uint32_t fnv8(uint32_t h, const int8_t* p, size_t n) {
  for (size_t i = 0; i < n; i++) h = (h ^ (uint8_t)p[i]) * 16777619u;
  return h;
}

static int rint_i(float v) { return (int)((v + 12582912.0f) - 12582912.0f); }

/* ------------------------------------------------------------------ iSTFT */
NANO_INLINE float hann(const nano_ctx* c, int n) {
  int nfft = c->m->cfg[NANO_CFG_N_FFT], M = nfft / 2, stride = 2048 / nfft;
  if (n == M) return 1.0f;
  int idx = n < M ? n : nfft - n;
  return 0.5f - 0.5f * nano_cs2048[2 * idx * stride];
}

/* count <= hop samples from OLA[off..], at absolute position n_abs0 */
static void emit(nano_ctx* c, long n_abs0, int off, int count) {
  const int* cf = c->m->cfg;
  int nfft = cf[NANO_CFG_N_FFT], hop = cf[NANO_CFG_HOP], pad = (nfft - hop) / 2;
  long total = c->total_frames * hop;
  int np = 0;
  for (int i = 0; i < count; i++) {
    long na = n_abs0 + i, o = na - pad;
    if (o < 0 || o >= total) continue;
    long fhi = na / hop;
    if (fhi > c->total_frames - 1) fhi = c->total_frames - 1;
    long flo = na - nfft + 1 <= 0 ? 0 : (na - nfft + 1 + hop - 1) / hop;
    float env = 0.f;
    for (long f = flo; f <= fhi; f++) {
      float wv = hann(c, (int)(na - f * hop));
      env += wv * wv;
    }
    float y = env > 1e-11f ? c->OLA[off + i] / env : 0.f;
    float v = y * 32767.0f;
    if (v > 32767.f) v = 32767.f;
    if (v < -32768.f) v = -32768.f;
    c->pcm[np++] = (int16_t)rint_i(v);
  }
  if (np && !c->stop) {
    c->prof.samples += np;
    c->emitted += np;
    if (c->fn && c->fn(c->user, c->pcm, np)) c->stop = 1;
  }
}

static void istft_frame(nano_ctx* c, const int8_t* q) {
  const int* cf = c->m->cfg;
  int nfft = cf[NANO_CFG_N_FFT], hop = cf[NANO_CFG_HOP], M = nfft / 2, stride = 2048 / nfft;
  const int8_t* qm = q;           /* log-magnitude, M + 1 bins */
  const int8_t* qp = q + M + 1;   /* phase */
  float* Z = c->Z;
  const float *lm = c->lut_mag, *lc = c->lut_cos, *ls = c->lut_sin;
  uint64_t t0 = nano_cycles();
  for (int k = 0; k < M; k++) {
    int j = M - k;
    float ak = lm[(uint8_t)qm[k]], aj = lm[(uint8_t)qm[j]];
    float xr = ak * lc[(uint8_t)qp[k]], xi = k == 0 ? 0.f : ak * ls[(uint8_t)qp[k]];
    float br = aj * lc[(uint8_t)qp[j]], bi = j == M ? 0.f : -(aj * ls[(uint8_t)qp[j]]);
    float er = (xr + br) * 0.5f, ei = (xi + bi) * 0.5f;
    float dr = (xr - br) * 0.5f, di = (xi - bi) * 0.5f;
    float ct = nano_cs2048[2 * k * stride], st = nano_cs2048[2 * k * stride + 1];
    float orr = dr * ct - di * st, oi = dr * st + di * ct;
    Z[2 * k] = er - oi;
    Z[2 * k + 1] = -(ei + orr);   /* conjugate: inverse FFT via the forward FFT */
  }
  uint64_t t1 = nano_cycles();
  nano_fft_c(Z, M);
  uint64_t t2 = nano_cycles();
  c->prof.op_cycles[NANO_OP_FFT] += t2 - t1;
  float invM = 1.0f / (float)M;
  float* ola = c->OLA;
  for (int n = 0; n < M; n++) {
    ola[2 * n] += Z[2 * n] * invM * hann(c, 2 * n);
    ola[2 * n + 1] += -Z[2 * n + 1] * invM * hann(c, 2 * n + 1);
  }
  emit(c, c->frame_idx * hop, 0, hop);
  memmove(ola, ola + hop, (size_t)(nfft - hop) * sizeof(float));
  memset(ola + nfft - hop, 0, (size_t)hop * sizeof(float));
  c->frame_idx++;
  c->prof.op_cycles[NANO_OP_ELEM] += (t1 - t0) + (nano_cycles() - t2);
}

/* head over rows of T1 (already normalized), then iSTFT per frame */
static void head_and_istft(nano_ctx* c, int rows) {
  const nano_model* m = c->m;
  int nf = m->cfg[NANO_CFG_N_FFT] + 2, dv = m->cfg[NANO_CFG_VOC_DIM];
  for (int r0 = 0; r0 < rows && !c->stop; r0 += HEAD_BATCH) {
    int nb = rows - r0 < HEAD_BATCH ? rows - r0 : HEAD_BATCH;
    PROF_ST(c, NANO_ST_HEAD, gemm(c, NANO_ST_HEAD, c->T1 + (size_t)r0 * dv, dv, nb, &m->voc_head, c->HO, nf, -128));
    c->prof.hash_head = fnv8(c->prof.hash_head, c->HO, (size_t)nb * nf);
    PROF_ST(c, NANO_ST_ISTFT, for (int i = 0; i < nb && !c->stop; i++) istft_frame(c, c->HO + (size_t)i * nf));
  }
}

/* vocoder on n new mel rows at MEL + hr*n_mels */
static void vocode(nano_ctx* c, int n, int flush) {
  const nano_model* m = c->m;
  const int* cf = m->cfg;
  int dv = cf[NANO_CFG_VOC_DIM];
  int ne = 0, nv = 0;
  PROF_ST(c, NANO_ST_VOC, {
    ne = run_conv(c, NANO_ST_VOC, &c->emb, c->MEL, n, flush, c->T1, dv);
    NANO_PROF_OP(c, NANO_OP_NORM, nano_norm_rows(c->T1, dv, ne, dv, m->voc_embed.sc[1], m->voc_ln0_g, m->voc_ln0_b,
                                                 m->fcfg[NANO_FCFG_LN_EPS], m->voc[0].dw.sc[0],
                                                 c->A + (size_t)c->hr * dv, dv));
    nv = run_stack(c, NANO_ST_VOC, c->vst, cf[NANO_CFG_VOC_LAYERS], ne, flush, c->MEL + (size_t)c->hr * cf[NANO_CFG_N_MELS], dv);
    NANO_PROF_OP(c, NANO_OP_NORM, nano_norm_rows(c->MEL + (size_t)c->hr * cf[NANO_CFG_N_MELS], dv, nv, dv,
                                                 m->voc[cf[NANO_CFG_VOC_LAYERS] - 1].res, m->voc_lnf_g, m->voc_lnf_b,
                                                 m->fcfg[NANO_FCFG_LN_EPS], m->voc_head.sc[0], c->T1, dv));
  });
  head_and_istft(c, nv);
}

/* ------------------------------------------------------------------ main entry */
long nano_synthesize(nano_ctx* c, const nano_request* r, nano_pcm_fn fn, void* user) {
  const nano_model* m = c->m;
  const int* cf = m->cfg;
  int T = r->n_ids;
  if (T < 1 || T > cf[NANO_CFG_MAX_PHONEMES] || !r->ids) {
    nano_seterr("phoneme count %d outside 1..%d", T, cf[NANO_CFG_MAX_PHONEMES]);
    return -1;
  }
  if (!(r->pace > 0.f) || r->pace > 100.f) {
    nano_seterr("pace %g outside (0, 100]", (double)r->pace);
    return -1;
  }
  if (r->voice < 0 || r->voice >= cf[NANO_CFG_N_VOICES] || r->tag < 0 || r->tag >= cf[NANO_CFG_N_TAGS]) {
    nano_seterr("voice or tag out of range");
    return -1;
  }
  for (int i = 0; i < T; i++)
    if (r->ids[i] < 0 || r->ids[i] >= cf[NANO_CFG_N_SYMBOLS]) {
      nano_seterr("phoneme id %d out of range", r->ids[i]);
      return -1;
    }
  int C, F;
  resolve_chunks(m, r->chunk, r->first_chunk, &C, &F);
  layout Lo;
  compute_layout(m, C, F, &Lo);
  if (Lo.total > c->ar.cap) {
    nano_seterr("arena too small for chunk %d: %u bytes needed, %u given", C, (unsigned)Lo.total,
                (unsigned)c->ar.cap);
    return -1;
  }
  c->chunk = C;
  c->first_chunk = F;
  c->hr = imax(imax(cf[NANO_CFG_ENC_K], cf[NANO_CFG_DEC_K]), imax(cf[NANO_CFG_VOC_K], cf[NANO_CFG_VOC_EMBED_K])) - 1;
  c->hp = imax(cf[NANO_CFG_PRED_K], cf[NANO_CFG_PE_K]) / 2;
  c->fn = fn;
  c->user = user;
  c->stop = 0;
  int de = cf[NANO_CFG_ENC_DIM], dd = cf[NANO_CFG_DEC_DIM], dv = cf[NANO_CFG_VOC_DIM], dp = cf[NANO_CFG_PRED_DIM];
  int dc = cf[NANO_CFG_D_COND], mp = cf[NANO_CFG_MAX_PHONEMES], hp = c->hp, hr = c->hr, nm = cf[NANO_CFG_N_MELS];
  int Le = cf[NANO_CFG_ENC_LAYERS], Ld = cf[NANO_CFG_DEC_LAYERS], Lv = cf[NANO_CFG_VOC_LAYERS];
  float eps = m->fcfg[NANO_FCFG_LN_EPS];

  /* persistent allocations */
  c->ar.top = (size_t)((uint8_t*)c - c->ar.base) + sizeof(nano_ctx);
  c->cond = (float*)xalloc(c, (size_t)dc * 4);
  c->H = (int8_t*)xalloc(c, (size_t)(mp + 2 * hp) * imax(de, dd));
  c->dur = (uint16_t*)xalloc(c, (size_t)mp * 2);
  c->f0 = (float*)xalloc(c, (size_t)mp * 4);
  c->en = (float*)xalloc(c, (size_t)mp * 4);
  c->dec_ada = (float*)xalloc(c, (size_t)Ld * 2 * dd * 4);
  c->mark = c->ar.top;
  c->prof.phonemes += T;
  c->prof.sample_rate = cf[NANO_CFG_SAMPLE_RATE];

  /* ---- conditioning */
  PROF_ST(c, NANO_ST_COND, {
    int nc = cf[NANO_CFG_N_CTRL];
    float cin[NANO_N_CTRL_MAX];   /* 12: values; 24: [value * set, set] */
    for (int i = 0; i < NANO_N_CTRL; i++) {
      int set = r->ctrl && (!r->ctrl_set || r->ctrl_set[i]);
      cin[i] = set ? r->ctrl[i] : 0.f;
      if (nc == 2 * NANO_N_CTRL) cin[NANO_N_CTRL + i] = set ? 1.f : 0.f;
    }
    for (int o = 0; o < dc; o++) {
      float v = m->voice[(size_t)r->voice * dc + o] + m->tag[(size_t)r->tag * dc + o];
      float a = m->ctrl_b[o];
      for (int i = 0; i < nc; i++) a += m->ctrl_w[(size_t)o * nc + i] * cin[i];
      v += a;
      if (cf[NANO_CFG_COND_ACT] == 1) v = v / (1.0f + nano_expf(-v));
      c->cond[o] = v;
    }
    for (int i = 0; i < Ld; i++) ada_params(m, &m->dec[i], dd, c->cond, c->dec_ada + (size_t)i * 2 * dd);
  });

  /* ---- encoder (chunks of C phonemes, one weight pass each) */
  c->ar.top = c->mark;
  float* enc_ada = (float*)xalloc(c, (size_t)Le * 2 * de * 4);
  int sp_e = Le * (cf[NANO_CFG_ENC_K] / 2), nre = imax(C, sp_e);
  c->A = (int8_t*)xalloc(c, (size_t)(hr + nre) * de);
  c->B = (int8_t*)xalloc(c, (size_t)(hr + nre) * de);
  c->T1 = (int8_t*)xalloc(c, (size_t)nre * de);
  c->FF = (int8_t*)xalloc(c, (size_t)nre * cf[NANO_CFG_ENC_FF]);
  PROF_ST(c, NANO_ST_COND, for (int i = 0; i < Le; i++) ada_params(m, &m->enc[i], de, c->cond, enc_ada + (size_t)i * 2 * de));
  for (int i = 0; i < Le; i++)
    init_stage(c, &c->st[i], &m->enc[i], de, cf[NANO_CFG_ENC_FF], cf[NANO_CFG_ENC_K], cf[NANO_CFG_ENC_ACT],
               enc_ada + (size_t)i * 2 * de, enc_ada + (size_t)i * 2 * de + de);
  if (!c->st[Le - 1].hist || !c->FF) goto oom;
  memset(c->H, 0, (size_t)(mp + 2 * hp) * imax(de, dd));
  PROF_ST(c, NANO_ST_ENC, {
    int done = 0, outrow = 0, flushed = 0;
    while (!flushed) {
      int n = T - done < C ? T - done : C;
      for (int i = 0; i < n; i++)
        memcpy(c->A + (size_t)(hr + i) * de, m->emb + (size_t)r->ids[done + i] * de, (size_t)de);
      done += n;
      flushed = done == T && (n == 0 || n + sp_e <= C);
      outrow += run_stack(c, NANO_ST_ENC, c->st, Le, n, flushed, c->H + (size_t)(hp + outrow) * de, de);
    }
  });

  c->prof.hash_enc = fnv8(c->prof.hash_enc, c->H + (size_t)hp * de, (size_t)T * de);
  /* ---- predictors (whole sentence; H has hp zero rows on both sides) */
  c->ar.top = c->mark;
  int wy = imax(dp, de);
  int8_t* Y1 = (int8_t*)xalloc(c, (size_t)(mp + 2 * hp) * wy);
  int8_t* Y2 = (int8_t*)xalloc(c, (size_t)(mp + 2 * hp) * wy);
  if (!Y2) goto oom;
  float sH = m->enc[Le - 1].res;
  PROF_ST(c, NANO_ST_PRED, {
    float* outs[3] = {NULL, c->f0, c->en};
    for (int pi = 0; pi < 3; pi++) {
      const nano_predw* P = &m->pred[pi];
      const int8_t* x = c->H;
      int din = de;
      int8_t* y = Y1;
      int pk = cf[NANO_CFG_PRED_K], pp = pk / 2, Lp = cf[NANO_CFG_PRED_LAYERS];
      for (int j = 0; j < Lp; j++) {
        memset(y, 0, (size_t)(T + 2 * hp) * dp);
        gemm(c, NANO_ST_PRED, x + (size_t)(hp - pp) * din, din, T, &P->c[j], y + (size_t)hp * dp, dp, 0);
        if (j + 1 < Lp) {
          NANO_PROF_OP(c, NANO_OP_NORM, nano_norm_rows(y + (size_t)hp * dp, dp, T, dp, P->c[j].sc[1], P->ln_g[j],
                                                       P->ln_b[j], eps, P->c[j + 1].sc[0], y + (size_t)hp * dp, dp));
          x = y;
          din = dp;
          y = y == Y1 ? Y2 : Y1;
        }
      }
      /* last layer: LayerNorm in float, then the scalar output */
      uint64_t t0 = nano_cycles();
      float s_in = P->c[Lp - 1].sc[1], eps_q = eps / (s_in * s_in), inv_d = 1.0f / (float)dp;
      for (int t = 0; t < T; t++) {
        const int8_t* q = y + (size_t)(hp + t) * dp;
        int32_t S = 0, SS = 0;
        for (int i = 0; i < dp; i++) {
          S += q[i];
          SS += (int32_t)q[i] * q[i];
        }
        float mean = (float)S * inv_d, var = (float)SS * inv_d - mean * mean;
        if (var < 0.f) var = 0.f;
        float rs = 1.0f / sqrtf(var + eps_q), acc = P->out_b;
        for (int i = 0; i < dp; i++)
          acc += ((((float)q[i] - mean) * rs) * P->ln_g[Lp - 1][i] + P->ln_b[Lp - 1][i]) * P->out_w[i];
        if (pi == 0)
          c->f0[t] = acc;   /* temporarily: log-durations */
        else
          outs[pi][t] = acc;
      }
      c->prof.op_cycles[NANO_OP_NORM] += nano_cycles() - t0;
      if (pi == 0) {
        /* durations now, so f0 can take its slot */
        float off = m->fcfg[NANO_FCFG_DUR_OFFSET];
        for (int t = 0; t < T; t++) {
          float f = (nano_expf(c->f0[t]) - off) / r->pace;
          if (r->widx && r->ws)
            for (int k = 0; k < r->n_ws; k++)
              if (r->ws[k].word == r->widx[t]) f *= r->ws[k].dur;
          int fr = r->fixed_dur > 0 ? r->fixed_dur : rint_i(f < 0.f ? 0.f : f > 60000.f ? 60000.f : f);
          if (fr > cf[NANO_CFG_MAX_DUR]) fr = cf[NANO_CFG_MAX_DUR];
          c->dur[t] = (uint16_t)fr;
        }
      }
    }
    if (r->widx && r->ws)
      for (int t = 0; t < T; t++)
        for (int k = 0; k < r->n_ws; k++)
          if (r->ws[k].word == r->widx[t]) {
            c->f0[t] += r->ws[k].pitch / m->fcfg[NANO_FCFG_WS_PITCH_DIV];
            c->en[t] += r->ws[k].energy / m->fcfg[NANO_FCFG_WS_ENERGY_DIV];
          }
  });

  /* ---- pitch / energy embeddings + projection to the decoder width (P overwrites H) */
  PROF_ST(c, NANO_ST_PROJ, {
    int pk = cf[NANO_CFG_PE_K], pp = pk / 2;
    float inv = 1.0f / m->dec_in.sc[0];
    const float* sig[2] = {c->f0, c->en};
    for (int t = 0; t < T; t++) {
      const int8_t* h = c->H + (size_t)(hp + t) * de;
      int8_t* o = Y1 + (size_t)t * de;
      for (int ch = 0; ch < de; ch++) {
        float v = (float)h[ch] * sH;
        for (int e = 0; e < 2; e++) {
          float a = m->pe_b[e][ch];
          for (int j = 0; j < pk; j++) {
            int tt = t + j - pp;
            if (tt >= 0 && tt < T) a += m->pe_w[e][j * de + ch] * sig[e][tt];
          }
          v += a;
        }
        o[ch] = (int8_t)nano_q8(v * inv);
      }
    }
    gemm(c, NANO_ST_PROJ, Y1, de, T, &m->dec_in, c->H, dd, -128);
  });
  c->prof.hash_p = fnv8(c->prof.hash_p, c->H, (size_t)T * dd);
  for (int t = 0; t < T; t++) c->prof.hash_p = (c->prof.hash_p ^ c->dur[t]) * 16777619u;

  /* ---- decoder + vocoder, chunk by chunk */
  long Tf = 0;
  for (int t = 0; t < T; t++) Tf += c->dur[t];
  c->total_frames = Tf;
  c->frame_idx = 0;
  c->emitted = 0;
  c->prof.frames += Tf;
  c->ar.top = c->mark;
  {
    int sp_dv = Ld * (cf[NANO_CFG_DEC_K] / 2) + cf[NANO_CFG_VOC_EMBED_K] / 2 + Lv * (cf[NANO_CFG_VOC_K] / 2);
    int nr = imax(C, sp_dv);
    int dm = imax(dd, dv), fm = imax(cf[NANO_CFG_DEC_FF], cf[NANO_CFG_VOC_FF]), nfft = cf[NANO_CFG_N_FFT];
    c->A = (int8_t*)xalloc(c, (size_t)(hr + nr) * dm);
    c->B = (int8_t*)xalloc(c, (size_t)(hr + nr) * dm);
    c->MEL = (int8_t*)xalloc(c, (size_t)(hr + nr) * imax(nm, dv));
    c->T1 = (int8_t*)xalloc(c, (size_t)nr * dm);
    c->FF = (int8_t*)xalloc(c, (size_t)nr * fm);
    for (int i = 0; i < Ld; i++)
      init_stage(c, &c->st[i], &m->dec[i], dd, cf[NANO_CFG_DEC_FF], cf[NANO_CFG_DEC_K], cf[NANO_CFG_DEC_ACT],
                 c->dec_ada + (size_t)i * 2 * dd, c->dec_ada + (size_t)i * 2 * dd + dd);
    c->emb.q = &m->voc_embed;
    c->emb.hist = (int8_t*)xalloc(c, (size_t)(cf[NANO_CFG_VOC_EMBED_K] - 1) * nm);
    c->emb.fill = cf[NANO_CFG_VOC_EMBED_K] / 2;
    for (int i = 0; i < Lv; i++)
      init_stage(c, &c->vst[i], &m->voc[i], dv, cf[NANO_CFG_VOC_FF], cf[NANO_CFG_VOC_K], cf[NANO_CFG_VOC_ACT],
                 m->voc[i].ln_g, m->voc[i].ln_b);
    c->HO = (int8_t*)xalloc(c, (size_t)HEAD_BATCH * (nfft + 2));
    c->Z = (float*)xalloc(c, (size_t)nfft * 4);
    c->OLA = (float*)xalloc(c, (size_t)nfft * 4);
    c->lut_mag = (float*)xalloc(c, 256 * 4);
    c->lut_cos = (float*)xalloc(c, 256 * 4);
    c->lut_sin = (float*)xalloc(c, 256 * 4);
    c->pcm = (int16_t*)xalloc(c, (size_t)cf[NANO_CFG_HOP] * 2);
    if (!c->pcm) goto oom;
    float smag = m->voc_head.sc[1], sph = m->voc_head.sc[2], mmax = m->fcfg[NANO_FCFG_MAG_MAX];
    for (int q = -128; q < 128; q++) {
      float e = nano_expf((float)q * smag);
      c->lut_mag[(uint8_t)q] = e < mmax ? e : mmax;
      nano_sincosf((float)q * sph, &c->lut_sin[(uint8_t)q], &c->lut_cos[(uint8_t)q]);
    }
    if (c->ar.peak > c->prof.arena_peak) c->prof.arena_peak = c->ar.peak;
    /* frame cursor over the phonemes */
    int ph = 0, left = T > 0 ? c->dur[0] : 0;
    long fed = 0;
    int first = 1;
    while (!c->stop) {
      int want = first ? F : C;
      int n = 0;
      int8_t* dst = c->A + (size_t)hr * dd;
      while (n < want && fed < Tf) {
        while (left == 0) left = c->dur[++ph];
        memcpy(dst + (size_t)n * dd, c->H + (size_t)ph * dd, (size_t)dd);
        n++;
        left--;
        fed++;
      }
      int flush = fed == Tf && (n == 0 || n + sp_dv <= C);   /* else the flush gets its own (n = 0) call */
      int nd = 0;
      /* the last decoder block writes to FF (free after its pw2), the mel rows go to MEL + hr*nm */
      PROF_ST(c, NANO_ST_DEC, nd = run_stack(c, NANO_ST_DEC, c->st, Ld, n, flush, c->FF, dd));
      PROF_ST(c, NANO_ST_MEL, gemm(c, NANO_ST_MEL, c->FF, dd, nd, &m->dec_mel, c->MEL + (size_t)hr * nm, nm, -128));
      c->prof.hash_mel = fnv8(c->prof.hash_mel, c->MEL + (size_t)hr * nm, (size_t)nd * nm);
      vocode(c, nd, flush);
      first = 0;
      if (flush) break;
    }
    /* tail: overlap-add positions past the last frame */
    if (!c->stop && Tf > 0) {
      int nfft2 = cf[NANO_CFG_N_FFT], hop = cf[NANO_CFG_HOP];
      int tail = (nfft2 - hop) / 2;
      PROF_ST(c, NANO_ST_ISTFT, for (int o = 0; o < tail; o += hop)
                                    emit(c, c->frame_idx * hop + o, o, tail - o < hop ? tail - o : hop));
    }
  }
  if (c->ar.peak > c->prof.arena_peak) c->prof.arena_peak = c->ar.peak;
  return c->emitted;
oom:
  nano_seterr("arena too small");
  return -1;
}
