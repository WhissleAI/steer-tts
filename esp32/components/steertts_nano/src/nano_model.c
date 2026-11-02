/* nano.bin loader: header + tensor directory checks, tensor pointers into the (memory-mapped) image. */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "nano_internal.h"

static char g_err[160];

void nano_seterr(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_err, sizeof g_err, fmt, ap);
  va_end(ap);
}
const char* nano_last_error(void) { return g_err; }

typedef struct {
  const nano_model* m;
  int bad;
} ld;

static const nano_bin_entry* find(ld* L, const char* name) {
  const nano_bin_header* h = L->m->hdr;
  const nano_bin_entry* e = (const nano_bin_entry*)(L->m->base + h->dir_offset);
  for (uint32_t i = 0; i < h->n_tensors; i++)
    if (!strncmp(e[i].name, name, NANO_BIN_NAME_LEN)) return &e[i];
  if (!L->bad) nano_seterr("nano.bin: missing tensor %s", name);
  L->bad = 1;
  return NULL;
}

/* tensor of dtype dt with exactly n elements */
static const void* get(ld* L, const char* name, int dt, uint32_t n) {
  const nano_bin_entry* e = find(L, name);
  if (!e) return NULL;
  uint32_t sz = dt == NANO_I8 ? 1 : 4;
  uint32_t cnt = e->shape[0] * e->shape[1] * e->shape[2];
  if (e->dtype != dt || cnt != n || e->nbytes != n * sz || e->offset % NANO_BIN_ALIGN ||
      (size_t)e->offset + e->nbytes > L->m->len) {
    if (!L->bad) nano_seterr("nano.bin: tensor %s has the wrong type, size or offset", name);
    L->bad = 1;
    return NULL;
  }
  return L->m->base + e->offset;
}

static const void* getf(ld* L, const char* fmt, const char* a, int dt, uint32_t n) {
  char name[NANO_BIN_NAME_LEN + 24];
  snprintf(name, sizeof name, fmt, a);
  return get(L, name, dt, n);
}

static void qlayer(ld* L, nano_qlayer* q, const char* name, int out, int k, int in, int depthwise, int nsc) {
  q->out = out;
  q->k = k;
  q->in = in;
  q->w = (const int8_t*)getf(L, "%s.w", name, NANO_I8, (uint32_t)(depthwise ? k * out : out * k * in));
  q->b = (const int32_t*)getf(L, "%s.b", name, NANO_I32, (uint32_t)out);
  q->m = (const int32_t*)getf(L, "%s.m", name, NANO_I32, (uint32_t)out);
  q->s = (const int32_t*)getf(L, "%s.s", name, NANO_I32, (uint32_t)out);
  getf(L, "%s.ws", name, NANO_F32, (uint32_t)out);
  q->sc = (const float*)getf(L, "%s.sc", name, NANO_F32, (uint32_t)nsc);
  if (!L->bad && (((uintptr_t)q->w) & 15)) L->bad = 1;
}

static void block(ld* L, nano_blockw* B, const char* p, int d, int ff, int k, int dc, int adaptive) {
  char n[40];
  snprintf(n, sizeof n, "%s.dw", p);
  qlayer(L, &B->dw, n, d, k, d, 1, 2);
  if (adaptive) {
    snprintf(n, sizeof n, "%s.ada", p);
    B->ada_w = (const int8_t*)getf(L, "%s.w", n, NANO_I8, (uint32_t)(2 * d * dc));
    B->ada_ws = (const float*)getf(L, "%s.ws", n, NANO_F32, (uint32_t)(2 * d));
    B->ada_b = (const float*)getf(L, "%s.b", n, NANO_F32, (uint32_t)(2 * d));
  } else {
    snprintf(n, sizeof n, "%s.ln", p);
    B->ln_g = (const float*)getf(L, "%s.g", n, NANO_F32, (uint32_t)d);
    B->ln_b = (const float*)getf(L, "%s.b", n, NANO_F32, (uint32_t)d);
  }
  snprintf(n, sizeof n, "%s.pw1", p);
  qlayer(L, &B->pw1, n, ff, 1, d, 0, 2);
  snprintf(n, sizeof n, "%s.pw2", p);
  qlayer(L, &B->pw2, n, d, 1, ff, 0, 2);
  const float* r = (const float*)getf(L, "%s.res", p, NANO_F32, 1);
  B->res = r ? r[0] : 0.f;
}

static int same(float a, float b) { return fabsf(a - b) <= 1e-6f * fabsf(b); }

int nano_model_open(nano_model* m, const void* data, size_t len) {
  memset(m, 0, sizeof *m);
  const nano_bin_header* h = (const nano_bin_header*)data;
  if (!data || len < NANO_BIN_HEADER_BYTES || memcmp(h->magic, NANO_BIN_MAGIC, 4)) {
    nano_seterr("not a nano.bin image");
    return -1;
  }
  if (h->version != NANO_BIN_VERSION || h->header_bytes != NANO_BIN_HEADER_BYTES) {
    nano_seterr("nano.bin version %u not supported (runtime reads version %d)", (unsigned)h->version,
                NANO_BIN_VERSION);
    return -1;
  }
  if (h->file_bytes > len || h->dir_offset + (size_t)h->n_tensors * NANO_BIN_ENTRY_BYTES > len) {
    nano_seterr("nano.bin truncated (%u bytes expected, %u present)", (unsigned)h->file_bytes, (unsigned)len);
    return -1;
  }
  if (((uintptr_t)data) & 15) {
    nano_seterr("nano.bin image must be 16-byte aligned");
    return -1;
  }
  m->base = (const uint8_t*)data;
  m->len = len;
  m->hdr = h;
  memcpy(m->cfg, h->cfg, sizeof m->cfg);
  memcpy(m->fcfg, h->fcfg, sizeof m->fcfg);
  const int* c = m->cfg;
  int dc = c[NANO_CFG_D_COND], de = c[NANO_CFG_ENC_DIM], dd = c[NANO_CFG_DEC_DIM], dv = c[NANO_CFG_VOC_DIM];
  int dp = c[NANO_CFG_PRED_DIM];
  if (c[NANO_CFG_ENC_LAYERS] < 1 || c[NANO_CFG_ENC_LAYERS] > NANO_MAX_LAYERS || c[NANO_CFG_DEC_LAYERS] < 1 ||
      c[NANO_CFG_DEC_LAYERS] > NANO_MAX_LAYERS || c[NANO_CFG_VOC_LAYERS] < 1 ||
      c[NANO_CFG_VOC_LAYERS] > NANO_MAX_LAYERS || c[NANO_CFG_PRED_LAYERS] < 1 || c[NANO_CFG_PRED_LAYERS] > 4) {
    nano_seterr("nano.bin: unsupported layer counts");
    return -1;
  }
  /* SIMD kernels work on 16-channel groups */
  int dims[] = {de, dd, dv, dp, c[NANO_CFG_ENC_FF], c[NANO_CFG_DEC_FF], c[NANO_CFG_VOC_FF], c[NANO_CFG_N_MELS]};
  for (unsigned i = 0; i < sizeof dims / sizeof dims[0]; i++)
    if (dims[i] <= 0 || dims[i] % 16 || dims[i] > NANO_MAX_DIM) {
      nano_seterr("nano.bin: widths must be multiples of 16 and <= %d", NANO_MAX_DIM);
      return -1;
    }
  int ks[] = {c[NANO_CFG_ENC_K], c[NANO_CFG_DEC_K], c[NANO_CFG_VOC_K], c[NANO_CFG_PRED_K], c[NANO_CFG_PE_K],
              c[NANO_CFG_VOC_EMBED_K]};
  for (unsigned i = 0; i < sizeof ks / sizeof ks[0]; i++)
    if (ks[i] < 1 || ks[i] % 2 == 0 || ks[i] > NANO_MAX_K) {
      nano_seterr("nano.bin: kernel sizes must be odd and <= %d", NANO_MAX_K);
      return -1;
    }
  int nfft = c[NANO_CFG_N_FFT];
  if (nfft != 1024 && nfft != 512 && nfft != 2048) {
    nano_seterr("nano.bin: n_fft must be 512, 1024 or 2048");
    return -1;
  }
  if (c[NANO_CFG_WIN] != nfft || c[NANO_CFG_HOP] <= 0 || nfft % c[NANO_CFG_HOP] || c[NANO_CFG_CHUNK] < 1 ||
      c[NANO_CFG_MAX_PHONEMES] < 1 || (c[NANO_CFG_N_CTRL] != NANO_N_CTRL && c[NANO_CFG_N_CTRL] != 2 * NANO_N_CTRL) || dc > NANO_MAX_DIM || dc < 1) {
    nano_seterr("nano.bin: unsupported configuration");
    return -1;
  }
  ld L = {m, 0};
  m->voice = (const float*)get(&L, "cond.voice", NANO_F32, (uint32_t)(c[NANO_CFG_N_VOICES] * dc));
  m->tag = (const float*)get(&L, "cond.tag", NANO_F32, (uint32_t)(c[NANO_CFG_N_TAGS] * dc));
  m->ctrl_w = (const float*)get(&L, "cond.ctrl.w", NANO_F32, (uint32_t)(dc * c[NANO_CFG_N_CTRL]));
  m->ctrl_b = (const float*)get(&L, "cond.ctrl.b", NANO_F32, (uint32_t)dc);
  m->emb = (const int8_t*)get(&L, "enc.emb", NANO_I8, (uint32_t)(c[NANO_CFG_N_SYMBOLS] * de));
  char p[32];
  for (int i = 0; i < c[NANO_CFG_ENC_LAYERS]; i++) {
    snprintf(p, sizeof p, "enc.%d", i);
    block(&L, &m->enc[i], p, de, c[NANO_CFG_ENC_FF], c[NANO_CFG_ENC_K], dc, 1);
  }
  static const char* pn[3] = {"dur", "f0", "en"};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < c[NANO_CFG_PRED_LAYERS]; j++) {
      snprintf(p, sizeof p, "pred.%s.c%d", pn[i], j);
      qlayer(&L, &m->pred[i].c[j], p, dp, c[NANO_CFG_PRED_K], j ? dp : de, 0, 2);
      m->pred[i].ln_g[j] = (const float*)getf(&L, "%s.ln.g", p, NANO_F32, (uint32_t)dp);
      m->pred[i].ln_b[j] = (const float*)getf(&L, "%s.ln.b", p, NANO_F32, (uint32_t)dp);
    }
    snprintf(p, sizeof p, "pred.%s.out", pn[i]);
    m->pred[i].out_w = (const float*)getf(&L, "%s.w", p, NANO_F32, (uint32_t)dp);
    const float* ob = (const float*)getf(&L, "%s.b", p, NANO_F32, 1);
    m->pred[i].out_b = ob ? ob[0] : 0.f;
  }
  for (int i = 0; i < 2; i++) {
    snprintf(p, sizeof p, "pe.%s", pn[i + 1]);
    m->pe_w[i] = (const float*)getf(&L, "%s.w", p, NANO_F32, (uint32_t)(c[NANO_CFG_PE_K] * de));
    m->pe_b[i] = (const float*)getf(&L, "%s.b", p, NANO_F32, (uint32_t)de);
  }
  qlayer(&L, &m->dec_in, "dec.in", dd, 1, de, 0, 2);
  for (int i = 0; i < c[NANO_CFG_DEC_LAYERS]; i++) {
    snprintf(p, sizeof p, "dec.%d", i);
    block(&L, &m->dec[i], p, dd, c[NANO_CFG_DEC_FF], c[NANO_CFG_DEC_K], dc, 1);
  }
  qlayer(&L, &m->dec_mel, "dec.mel", c[NANO_CFG_N_MELS], 1, dd, 0, 2);
  qlayer(&L, &m->voc_embed, "voc.embed", dv, c[NANO_CFG_VOC_EMBED_K], c[NANO_CFG_N_MELS], 0, 2);
  m->voc_ln0_g = (const float*)get(&L, "voc.ln0.g", NANO_F32, (uint32_t)dv);
  m->voc_ln0_b = (const float*)get(&L, "voc.ln0.b", NANO_F32, (uint32_t)dv);
  for (int i = 0; i < c[NANO_CFG_VOC_LAYERS]; i++) {
    snprintf(p, sizeof p, "voc.%d", i);
    block(&L, &m->voc[i], p, dv, c[NANO_CFG_VOC_FF], c[NANO_CFG_VOC_K], dc, 0);
  }
  m->voc_lnf_g = (const float*)get(&L, "voc.lnf.g", NANO_F32, (uint32_t)dv);
  m->voc_lnf_b = (const float*)get(&L, "voc.lnf.b", NANO_F32, (uint32_t)dv);
  qlayer(&L, &m->voc_head, "voc.head", nfft + 2, 1, dv, 0, 3);
  if (L.bad) return -1;

  /* scale chain: each int8 tensor handed from one layer to the next is in one agreed scale */
  struct { float a, b; const char* what; } chk[64];
  int nc = 0;
#define CHK(x, y, w) do { if (nc < 64) { chk[nc].a = (x); chk[nc].b = (y); chk[nc].what = (w); nc++; } } while (0)
  const nano_blockw* blks[3] = {m->enc, m->dec, m->voc};
  int nl[3] = {c[NANO_CFG_ENC_LAYERS], c[NANO_CFG_DEC_LAYERS], c[NANO_CFG_VOC_LAYERS]};
  int act[3] = {c[NANO_CFG_ENC_ACT], c[NANO_CFG_DEC_ACT], c[NANO_CFG_VOC_ACT]};
  for (int s = 0; s < 3; s++)
    for (int i = 0; i < nl[s]; i++) {
      const nano_blockw* B = &blks[s][i];
      if (i + 1 < nl[s]) CHK(B->res, blks[s][i + 1].dw.sc[0], "block output -> next block input");
      if (act[s] == NANO_ACT_RELU) CHK(B->pw1.sc[1], B->pw2.sc[0], "pw1 output -> pw2 input (ReLU)");
    }
  float sH = m->enc[nl[0] - 1].res;
  for (int i = 0; i < 3; i++) CHK(m->pred[i].c[0].sc[0], sH, "encoder output -> predictor input");
  CHK(m->dec_in.sc[1], m->dec[0].dw.sc[0], "dec.in output -> decoder input");
  CHK(m->dec[nl[1] - 1].res, m->dec_mel.sc[0], "decoder output -> mel input");
  CHK(m->dec_mel.sc[1], m->voc_embed.sc[0], "mel -> vocoder input");
#undef CHK
  for (int i = 0; i < nc; i++)
    if (!same(chk[i].a, chk[i].b)) {
      nano_seterr("nano.bin: scale mismatch (%s): %g vs %g", chk[i].what, (double)chk[i].a, (double)chk[i].b);
      return -1;
    }
  if (c[NANO_CFG_ENC_ACT] > 1 || c[NANO_CFG_DEC_ACT] > 1 || c[NANO_CFG_VOC_ACT] > 1) {
    nano_seterr("nano.bin: unknown activation");
    return -1;
  }
  const nano_bin_entry* e = (const nano_bin_entry*)(m->base + h->dir_offset);
  for (uint32_t i = 0; i < h->n_tensors; i++) {
    m->weight_bytes += e[i].nbytes;
    if (e[i].dtype == NANO_I8) m->n_params_i8 += e[i].nbytes;
  }
  return 0;
}
