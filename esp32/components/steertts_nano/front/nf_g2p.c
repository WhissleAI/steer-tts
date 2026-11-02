/* G2P (steertts_edge EdgeG2p): g2p_en's cleaning and tokenizer, CMUdict lookup, averaged-perceptron heteronym
 * choice, the cached OOV table and the GRU sequence-to-sequence model for unknown words, all read from front.bin.
 *
 * The GRU runs with int8 weights from flash and an int16 hidden state (split into two int8 halves so the int8 SIMD
 * dot product computes the exact int16 x int8 sum); gates in float32. The reference runs the same int8 weights in
 * float64, so a few near-tie predictions can differ (measured on the golden G2P set: see docs/FRONT_END.md). */
#include <stdio.h>

#include "nf_int.h"
#include "steertts_nano.h"   /* nano_dot_s8 */

float nano_expf(float x);

/* ------------------------------------------------------------------ string tables */
/* het W: u32 n; u32 koff[n+1]; i32 val[n]; keys */
static int het_w(const nano_front_ctx* c, const char* k, int32_t* v) {
  const uint8_t* t = c->d.het_w;
  int n = (int)rd32(t);
  const uint8_t* ko = t + 4;
  const uint8_t* val = ko + 4 * (n + 1);
  const char* pool = (const char*)(val + 4 * n);
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2, r = strcmp(pool + rd32(ko + 4 * mid), k);
    if (!r) {
      *v = (int32_t)rd32(val + 4 * mid);
      return 1;
    }
    if (r < 0)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return 0;
}
/* het T: u32 n; u32 koff[n+1]; u32 voff[n+1]; u32 key_pool_len; keys; values */
static const char* het_t(const nano_front_ctx* c, const char* k) {
  const uint8_t* t = c->d.het_t;
  int n = (int)rd32(t);
  const uint8_t* ko = t + 4;
  const uint8_t* vo = ko + 4 * (n + 1);
  u32 kpl = rd32(vo + 4 * (n + 1));
  const char* kp = (const char*)(vo + 4 * (n + 1) + 4);
  const char* vp = kp + kpl;
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2, r = strcmp(kp + rd32(ko + 4 * mid), k);
    if (!r) return vp + rd32(vo + 4 * mid);
    if (r < 0)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return NULL;
}
/* het H: u32 n; u32 off[n+1]; entries word\0 pos\0 n1 p1 n2 p2 */
static const uint8_t* het_h(const nano_front_ctx* c, const char* w) {
  const uint8_t* t = c->d.het_h;
  int n = (int)rd32(t);
  const uint8_t* offs = t + 4;
  const uint8_t* pool = offs + 4 * (n + 1);
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    const uint8_t* e = pool + rd32(offs + 4 * mid);
    int r = strcmp((const char*)e, w);
    if (!r) return e;
    if (r < 0)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return NULL;
}

int nf_in_cmu(const nano_front_ctx* c, const char* w) {
  for (const char* q = w; *q; ++q)
    if ((unsigned char)*q >= 128) return 0;
  return nf_lex_get(c->d.lex, w, NULL, NULL);
}

/* ------------------------------------------------------------------ heteronyms (averaged perceptron) */
static int isalpha_ascii(const char* t) {
  if (!*t) return 0;
  for (; *t; ++t)
    if (!((*t >= 'a' && *t <= 'z') || (*t >= 'A' && *t <= 'Z'))) return 0;
  return 1;
}
static const char* last2(const char* t) {
  size_t n = strlen(t);
  return n > 2 ? t + n - 2 : t;
}
static void suf(const char* t, char* o, size_t cap) { snprintf(o, cap, "%s", isalpha_ascii(t) && strlen(t) > 2 ? last2(t) : t); }
static void het_tag(const nano_front_ctx* c, const char* t, char* o, size_t cap) {
  const char* e = het_t(c, t);
  if (e) {
    snprintf(o, cap, "%s", e);
    return;
  }
  if (!strcmp(t, "<s>") || !strcmp(t, "</s>")) {
    snprintf(o, cap, "%s", t);
    return;
  }
  int any = 0;
  for (const char* q = t; *q; ++q)
    if ((*q >= 'a' && *q <= 'z') || (*q >= 'A' && *q <= 'Z')) any = 1;
  if (!any) {
    snprintf(o, cap, "P");
    return;
  }
  snprintf(o, cap, "U%s", last2(t));
}

/* score of the first pronunciation: features generated one at a time (no feature table on the stack) */
static int het_first(nano_front_ctx* c, const char* pos, char toks[][48], int j, int ntok) {
  const char* w = toks[2];
  const char *p1 = toks[1], *p2 = toks[0], *n1 = toks[3], *n2 = toks[4];
  (void)j;
  (void)ntok;
  char a[48], A[48], B[48], C[48], D[48];
  if (c->d.has_td) {
    het_tag(c, p2, A, sizeof A);
    het_tag(c, p1, B, sizeof B);
    het_tag(c, n1, C, sizeof C);
    het_tag(c, n2, D, sizeof D);
  }
  int nf = c->d.has_td ? 21 : 12;
  long s = 0;
  char F[112], key[176];
  int32_t v;
  for (int k = 0; k < nf; ++k) {
    switch (k) {
      case 0: snprintf(F, sizeof F, "b"); break;
      case 1: snprintf(F, sizeof F, "p1=%s", p1); break;
      case 2: snprintf(F, sizeof F, "n1=%s", n1); break;
      case 3: snprintf(F, sizeof F, "p2=%s", p2); break;
      case 4: snprintf(F, sizeof F, "n2=%s", n2); break;
      case 5: suf(p1, a, sizeof a); snprintf(F, sizeof F, "p1s=%s", a); break;
      case 6: suf(n1, a, sizeof a); snprintf(F, sizeof F, "n1s=%s", a); break;
      case 7: snprintf(F, sizeof F, "p1n1=%s_%s", p1, n1); break;
      case 8: snprintf(F, sizeof F, "p2p1=%s_%s", p2, p1); break;
      case 9: snprintf(F, sizeof F, "n1n2=%s_%s", n1, n2); break;
      case 10: suf(p2, a, sizeof a); snprintf(F, sizeof F, "p2s=%s", a); break;
      case 11: suf(n2, a, sizeof a); snprintf(F, sizeof F, "n2s=%s", a); break;
      case 12: snprintf(F, sizeof F, "p1t=%s", B); break;
      case 13: snprintf(F, sizeof F, "n1t=%s", C); break;
      case 14: snprintf(F, sizeof F, "p2t=%s", A); break;
      case 15: snprintf(F, sizeof F, "n2t=%s", D); break;
      case 16: snprintf(F, sizeof F, "p1tn1t=%s_%s", B, C); break;
      case 17: snprintf(F, sizeof F, "p2tp1t=%s_%s", A, B); break;
      case 18: snprintf(F, sizeof F, "n1tn2t=%s_%s", C, D); break;
      case 19: snprintf(F, sizeof F, "p1n1t=%s_%s", p1, C); break;
      default: snprintf(F, sizeof F, "p1tn1=%s_%s", B, n1); break;
    }
    snprintf(key, sizeof key, "w:%s|%s", w, F);
    if (het_w(c, key, &v)) s += v;
    snprintf(key, sizeof key, "g:%s|%s", pos, F);
    if (het_w(c, key, &v)) s += v;
  }
  return s > 0;
}

/* ------------------------------------------------------------------ GRU (int8 weights from flash) */
typedef struct {
  int H, NG, NP;
  const float *enc_x, *dec_x, *enc_s, *enc_b, *dec_s, *dec_b, *fc_s, *fc_b;
  const int8_t *enc_q, *dec_q, *fc_q;
  const int32_t* p2s;
} gru_t;
static void gru_get(const nano_front_ctx* c, gru_t* g) {
  const uint8_t* b = c->d.gru;
  g->H = (int)rd32(b + 8);
  g->NG = (int)rd32(b + 12);
  g->NP = (int)rd32(b + 16);
  const uint8_t* o = b + 20;
#define AT(k) (b + rd32(o + 4 * (k)))
  g->enc_x = (const float*)AT(0);
  g->dec_x = (const float*)AT(1);
  g->enc_q = (const int8_t*)AT(2);
  g->enc_s = (const float*)AT(3);
  g->enc_b = (const float*)AT(4);
  g->dec_q = (const int8_t*)AT(5);
  g->dec_s = (const float*)AT(6);
  g->dec_b = (const float*)AT(7);
  g->fc_q = (const int8_t*)AT(8);
  g->fc_s = (const float*)AT(9);
  g->fc_b = (const float*)AT(10);
  g->p2s = (const int32_t*)AT(11);
#undef AT
}

#define H16 32639.0f   /* h in [-1, 1] -> int16 with |hi| <= 127 */
/* state in the idle normalisation buffer a: h, t (float[H]), hi, lo (int8[H]) = 10 H bytes <= sizeof a */
#define GH(c) ((float*)(void*)(c)->a)
#define GT(c) (GH(c) + NF_GRU_MAX_H)
#define GHI(c) ((int8_t*)(void*)(GT(c) + NF_GRU_MAX_H))
#define GLO(c) (GHI(c) + NF_GRU_MAX_H)
static void h_split(nano_front_ctx* c, int H) {
  for (int j = 0; j < H; j++) {
    float v = GH(c)[j] * H16;
    int q = (int)((v + 12582912.0f) - 12582912.0f);
    if (q > 32639) q = 32639;
    if (q < -32640) q = -32640;
    int lo = ((q + 128) & 255) - 128;
    GLO(c)[j] = (int8_t)lo;
    GHI(c)[j] = (int8_t)((q - lo) / 256);
  }
}
static float hdot(const nano_front_ctx* c, const int8_t* row, int H) {
  int32_t a = nano_dot_s8(row, GHI(c), H), b = nano_dot_s8(row, GLO(c), H);
  return (float)(a * 256 + b) * (1.0f / H16);
}
static float sigm(float x) { return 1.0f / (1.0f + nano_expf(-x)); }
static float tanh_f(float x) {
  if (x > 15.f) return 1.f;
  if (x < -15.f) return -1.f;
  float e = nano_expf(2.0f * x);
  return (e - 1.0f) / (e + 1.0f);
}
/* gru_np.GruG2P._cell */
static void cell(nano_front_ctx* c, const gru_t* g, const float* xp, const int8_t* q, const float* s, const float* b) {
  int H = g->H;
  h_split(c, H);
  for (int j = 0; j < H; j++) {
    float hr = s[j] * hdot(c, q + (size_t)j * H, H) + b[j];
    float hz = s[H + j] * hdot(c, q + (size_t)(H + j) * H, H) + b[H + j];
    float hn = s[2 * H + j] * hdot(c, q + (size_t)(2 * H + j) * H, H) + b[2 * H + j];
    float r = sigm(xp[j] + hr), z = sigm(xp[H + j] + hz);
    float n = tanh_f(xp[2 * H + j] + r * hn);
    GT(c)[j] = (1.0f - z) * n + z * GH(c)[j];
  }
  memcpy(GH(c), GT(c), (size_t)H * sizeof(float));
}

int nf_gru_predict(nano_front_ctx* c, const char* word, int* out) {
  gru_t g;
  gru_get(c, &g);
  int H = g.H;
  memset(GH(c), 0, (size_t)H * sizeof(float));
  for (const char* q = word;; ++q) {
    int id = !*q ? 2 : (*q >= 'a' && *q <= 'z') ? *q - 'a' + 3 : 1;
    if (id >= g.NG) id = 1;
    cell(c, &g, g.enc_x + (size_t)id * 3 * H, g.enc_q, g.enc_s, g.enc_b);
    if (!*q) break;
  }
  int prev = 2, n = 0;
  for (int step = 0; step < 20; ++step) {
    cell(c, &g, g.dec_x + (size_t)prev * 3 * H, g.dec_q, g.dec_s, g.dec_b);
    h_split(c, H);
    int p = 0;
    float best = 0;
    for (int k = 0; k < g.NP; ++k) {
      float v = g.fc_s[k] * hdot(c, g.fc_q + (size_t)k * H, H) + g.fc_b[k];
      if (k == 0 || v > best) {
        best = v;
        p = k;
      }
    }
    if (p == 3) break;
    out[n++] = p;
    prev = p;
  }
  return n;
}
int nf_gru_symbol(const nano_front_ctx* c, int raw) {
  gru_t g;
  gru_get(c, &g);
  int v = raw >= 0 && raw < g.NP ? g.p2s[raw] : -1;
  return v >= 0 && v < 118 ? v : -1;
}

/* ------------------------------------------------------------------ tokenizer (edge_front tok.word_tokenize) */
static int tw(int ch) { return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_'; }
static int tal(int ch) { return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9'); }
static int tsp(int ch) { return ch == ' ' || (ch >= 9 && ch <= 13) || (ch >= 28 && ch <= 31); }

/* token match at p in s (length n): end or -1 */
static int tok_at(const char* s, int n, int p) {
#define C(i) ((i) >= 0 && (i) < n ? (unsigned char)s[(i)] : -1)
  /* (?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+){0,126}[.](?:[a-z]{2,13})\b/?(?!@) */
  if (tal(C(p)) && C(p - 1) != '@') {
    int k = p;
    while (tal(C(k))) k++;
    int sep[128], ng = 0;
    while (ng < 126 && (C(k) == '.' || C(k) == '-') && tal(C(k + 1))) {
      sep[ng++] = k;
      k++;
      while (tal(C(k))) k++;
    }
    for (int g = ng; g >= 0; g--) {
      int q = g == ng ? k : sep[g];
      if (C(q) != '.') continue;
      int r = 0;
      while (r < 13 && C(q + 1 + r) >= 'a' && C(q + 1 + r) <= 'z') r++;
      if (r < 2 || tw(C(q + 1 + r))) continue;
      int e = q + 1 + r;
      if (C(e) == '/') e++;
      if (C(e) == '@') continue;
      return e;
    }
  }
  /* [a-z](?:[a-z]|['\-_])+[a-z] */
  if (C(p) >= 'a' && C(p) <= 'z') {
    int j = p + 1;
    while ((C(j) >= 'a' && C(j) <= 'z') || C(j) == '\'' || C(j) == '-' || C(j) == '_') j++;
    int L = j - 1;
    while (L >= p + 2 && !(C(L) >= 'a' && C(L) <= 'z')) L--;
    if (L >= p + 2) return L + 1;
  }
  /* [a-z0-9_]+ */
  if (tw(C(p))) {
    int j = p;
    while (tw(C(j))) j++;
    return j;
  }
  /* \.(?:\s*\.)+ */
  if (C(p) == '.') {
    int k = p + 1, cnt = 0;
    for (;;) {
      int q = k;
      while (tsp(C(q))) q++;
      if (C(q) != '.') break;
      k = q + 1;
      cnt++;
    }
    if (cnt) return k;
  }
  /* \S */
  if (C(p) >= 0 && !tsp(C(p))) return p + 1;
  return -1;
#undef C
}

/* replace every `from` by `to` (str.replace) from src into dst */
static int replace_all(const char* src, const char* from, const char* to, char* dst, int cap) {
  int fl = (int)strlen(from), tl = (int)strlen(to), o = 0;
  for (const char* p = src; *p;) {
    if (!strncmp(p, from, (size_t)fl)) {
      if (o + tl >= cap) return -1;
      memcpy(dst + o, to, (size_t)tl);
      o += tl;
      p += fl;
    } else {
      if (o + 1 >= cap) return -1;
      dst[o++] = *p++;
    }
  }
  dst[o] = 0;
  return o;
}

/* pronunciation of token j (symbol ids, <= 64) */
static int pron(nano_front_ctx* c, int j, int* ph) {
  char toks[5][48];   /* p2 p1 w n1 n2 for the heteronym features */
  for (int k = 0; k < 5; k++) {
    int t = j - 2 + k;
    if (t < 0)
      strcpy(toks[k], "<s>");
    else if (t >= c->ntok)
      strcpy(toks[k], "</s>");
    else {
      int L = c->tok_n[t] < 47 ? c->tok_n[t] : 47;
      memcpy(toks[k], c->asc + c->tok_s[t], (size_t)L);
      toks[k][L] = 0;
    }
  }
  char w[256];
  {
    int L = c->tok_n[j];
    memcpy(w, c->asc + c->tok_s[j], (size_t)L);
    w[L] = 0;
  }
  int has_letter = 0;
  for (const char* q = w; *q; ++q)
    if (*q >= 'a' && *q <= 'z') has_letter = 1;
  if (!has_letter) {
    static const char P[] = ",.?!;:-";
    const char* q = w[0] && !w[1] ? strchr(P, w[0]) : NULL;
    if (q && *q) {
      ph[0] = 2 + (int)(q - P);
      return 1;
    }
    return 0;
  }
  const uint8_t* h = strlen(w) < 48 ? het_h(c, w) : NULL;
  if (h) {
    const char* pos = (const char*)h + strlen((const char*)h) + 1;
    const uint8_t* p1 = (const uint8_t*)pos + strlen(pos) + 1;
    const uint8_t* p2 = p1 + 1 + p1[0];
    int first = het_first(c, pos, toks, j, c->ntok);
    const uint8_t* p = first ? p1 : p2;
    int n = p[0] < 64 ? p[0] : 64;
    for (int i = 0; i < n; i++) ph[i] = p[1 + i];
    return n;
  }
  uint8_t pb[64];
  int n = 0;
  int ascii = 1;
  for (const char* q = w; *q; ++q)
    if ((unsigned char)*q >= 128) ascii = 0;
  if (ascii && nf_lex_get(c->d.lex, w, pb, &n) && n) {
    for (int i = 0; i < n; i++) ph[i] = pb[i];
    return n;
  }
  if (!(ascii && nf_lex_get(c->d.lex, w, NULL, NULL)) && nf_lex_get(c->d.oov, w, pb, &n)) {
    for (int i = 0; i < n; i++) ph[i] = pb[i];
    return n;
  }
  int raw[20], nr = nf_gru_predict(c, w, raw), m = 0;
  for (int i = 0; i < nr; i++) {
    int s = nf_gru_symbol(c, raw[i]);
    if (s >= 0) ph[m++] = s;
  }
  return m;
}

/* EdgeG2p(text): normalised text in x (= c->a) -> ids with -1 between words */
int nf_g2p(nano_front_ctx* c, const u32* text, int n, int* out, int cap) {
  nfs X = {c->a, 0, NF_MAX_CP, c}, Y = {c->b, 0, NF_MAX_CP, c};
  if (text != c->a) memmove(c->a, text, (size_t)n * sizeof(u32));
  X.n = n;
  nf_normalize_numbers2(c, &X, &Y);
  if (c->fail) return -1;
  /* NFD, drop Mn, lower, keep [ a-z'.,?!-] */
  int an = 0;
  for (int i = 0; i < X.n; i++) {
    int l;
    const char* s = nf_g2p_clean(c, X.p[i], &l);
    if (an + l >= NF_MAX_CP) {
      nf_fail(c, "ValueError: text too long for the device front end");
      return -1;
    }
    memcpy(c->asc + an, s, (size_t)l);
    an += l;
  }
  c->asc[an] = 0;
  char* tmp = (char*)c->b;   /* b is free again: 4 * NF_MAX_CP bytes */
  int cap2 = (int)sizeof c->b;
  if (replace_all(c->asc, "i.e.", "that is", tmp, cap2) < 0 ||
      replace_all(tmp, "e.g.", "for example", c->asc, NF_MAX_CP) < 0) {
    nf_fail(c, "ValueError: text too long for the device front end");
    return -1;
  }
  /* ([^a-z0-9])\1{3,} -> \1\1\1 */
  {
    int o = 0, L = (int)strlen(c->asc);
    for (int i = 0; i < L;) {
      int ch = (unsigned char)c->asc[i];
      int j = i + 1;
      while (j < L && (unsigned char)c->asc[j] == ch) j++;
      if (!tal(ch) && j - i >= 4) {
        for (int k = 0; k < 3; k++) tmp[o++] = (char)ch;
      } else {
        memcpy(tmp + o, c->asc + i, (size_t)(j - i));
        o += j - i;
      }
      i = j;
    }
    tmp[o] = 0;
    memcpy(c->asc, tmp, (size_t)o + 1);
  }
  /* tokens */
  c->ntok = 0;
  int L = (int)strlen(c->asc);
  for (int p = 0; p < L;) {
    int e = tok_at(c->asc, L, p);
    if (e < 0) {
      p++;
      continue;
    }
    if (c->ntok >= NF_MAX_TOK || e - p > 255) {
      nf_fail(c, "ValueError: text too long for the device front end");
      return -1;
    }
    c->tok_s[c->ntok] = (uint16_t)p;
    c->tok_n[c->ntok] = (uint8_t)(e - p);
    c->ntok++;
    p = e;
  }
  int no = 0;
  for (int j = 0; j < c->ntok && !c->fail; ++j) {
    int ph[64];
    int np = pron(c, j, ph);
    if (no + np + 1 > cap) {
      nf_fail(c, "ValueError: text too long for the device front end");
      return -1;
    }
    memcpy(out + no, ph, (size_t)np * sizeof(int));
    no += np;
    out[no++] = -1;
  }
  if (no) --no;
  return c->fail ? -1 : no;
}
