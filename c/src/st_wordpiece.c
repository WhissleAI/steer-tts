/* BERT WordPiece tokenizer (see st_wordpiece.h): vocabulary hash, UTF-8 decoding, the BertNormalizer with NFD's
 * canonical reordering, the BertPreTokenizer split and greedy longest-match-first WordPiece. */
#include "st_wordpiece.h"

#include <stdlib.h>
#include <string.h>

#include "st_base.h"
#include "st_unicode.h"

enum { MAX_WORD_CHARS = 100 };

struct st_wordpiece {
  char* text;            /* the vocab file, lines NUL-terminated in place */
  const char** tok;      /* [n] */
  int* len;              /* [n] */
  int n;
  int32_t* hash;         /* [hmask + 1], -1 = empty */
  uint32_t hmask;
  int unk, cls, sep;
  int special[5];        /* [PAD] [UNK] [CLS] [SEP] [MASK] ids (-1 if absent) */
};

static const char* const SPECIAL[5] = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"};

static uint32_t fnv1a(const char* s, size_t n) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) h = (h ^ (uint8_t)s[i]) * 16777619u;
  return h;
}

int st_wordpiece_id(const st_wordpiece* wp, const char* tok, size_t len) {
  for (uint32_t h = fnv1a(tok, len) & wp->hmask;; h = (h + 1) & wp->hmask) {
    int32_t i = wp->hash[h];
    if (i < 0) return -1;
    if ((size_t)wp->len[i] == len && memcmp(wp->tok[i], tok, len) == 0) return i;
  }
}

int st_wordpiece_vocab_size(const st_wordpiece* wp) { return wp->n; }

void st_wordpiece_free(st_wordpiece* wp) {
  if (!wp) return;
  st_free(wp->text);
  st_free((void*)wp->tok);
  st_free(wp->len);
  st_free(wp->hash);
  st_free(wp);
}

st_wordpiece* st_wordpiece_load(const char* vocab_path) {
  st_wordpiece* wp = (st_wordpiece*)st_calloc(sizeof *wp);
  if (!wp) return NULL;
  size_t len;
  if (st_read_file(vocab_path, &wp->text, &len) != 0) goto fail;
  int n = 0;
  for (size_t i = 0; i < len; ++i) n += wp->text[i] == '\n';
  if (len && wp->text[len - 1] != '\n') ++n;
  wp->tok = (const char**)st_malloc(sizeof(char*) * (size_t)(n + 1));
  wp->len = (int*)st_malloc(sizeof(int) * (size_t)(n + 1));
  uint32_t hs = 1;
  while (hs < 2u * (uint32_t)n + 2u) hs <<= 1;
  wp->hash = (int32_t*)st_malloc(sizeof(int32_t) * hs);
  if (!wp->tok || !wp->len || !wp->hash) goto fail;
  memset(wp->hash, 0xff, sizeof(int32_t) * hs);
  wp->hmask = hs - 1;
  char* s = wp->text;
  for (int i = 0; i < n; ++i) {
    char* e = strchr(s, '\n');
    if (!e) e = s + strlen(s);
    char* next = *e ? e + 1 : e;
    if (e > s && e[-1] == '\r') --e;
    *e = 0;
    wp->tok[i] = s;
    wp->len[i] = (int)(e - s);
    wp->n = i + 1;
    if (st_wordpiece_id(wp, s, (size_t)wp->len[i]) < 0) {   /* first occurrence wins */
      uint32_t h = fnv1a(s, (size_t)wp->len[i]) & wp->hmask;
      while (wp->hash[h] >= 0) h = (h + 1) & wp->hmask;
      wp->hash[h] = i;
    }
    s = next;
  }
  for (int k = 0; k < 5; ++k) wp->special[k] = st_wordpiece_id(wp, SPECIAL[k], strlen(SPECIAL[k]));
  wp->unk = wp->special[1];
  wp->cls = wp->special[2];
  wp->sep = wp->special[3];
  if (wp->unk < 0 || wp->cls < 0 || wp->sep < 0) {
    st_seterr("%s: no [UNK] / [CLS] / [SEP] token", vocab_path);
    goto fail;
  }
  return wp;
fail:
  st_wordpiece_free(wp);
  return NULL;
}

/* ------------------------------------------------------------------ UTF-8 */
static size_t utf8_decode(const unsigned char* s, size_t n, uint32_t* out) {
  static const uint32_t min_of[4] = {0, 0x80, 0x800, 0x10000};
  size_t i = 0, m = 0;
  while (i < n) {
    uint32_t c = s[i];
    if (c < 0x80) {
      out[m++] = c;
      ++i;
      continue;
    }
    int k = (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : 0, ok = k > 0 && i + (size_t)k < n;
    c &= 0x3Fu >> k;
    for (int j = 1; ok && j <= k; ++j) {
      ok = (s[i + (size_t)j] & 0xC0) == 0x80;
      c = (c << 6) | (s[i + (size_t)j] & 0x3F);
    }
    if (!ok || c < min_of[k] || c > 0x10FFFF || (c >= 0xD800 && c < 0xE000)) {
      out[m++] = 0xFFFD;   /* invalid: one replacement character per bad lead byte */
      ++i;
      continue;
    }
    out[m++] = c;
    i += (size_t)k + 1;
  }
  return m;
}

static int utf8_encode(uint32_t c, char* o) {
  if (c < 0x80) {
    o[0] = (char)c;
    return 1;
  }
  if (c < 0x800) {
    o[0] = (char)(0xC0 | (c >> 6));
    o[1] = (char)(0x80 | (c & 0x3F));
    return 2;
  }
  if (c < 0x10000) {
    o[0] = (char)(0xE0 | (c >> 12));
    o[1] = (char)(0x80 | ((c >> 6) & 0x3F));
    o[2] = (char)(0x80 | (c & 0x3F));
    return 3;
  }
  o[0] = (char)(0xF0 | (c >> 18));
  o[1] = (char)(0x80 | ((c >> 12) & 0x3F));
  o[2] = (char)(0x80 | ((c >> 6) & 0x3F));
  o[3] = (char)(0x80 | (c & 0x3F));
  return 4;
}

/* ------------------------------------------------------------------ encoding */
typedef struct {
  const st_wordpiece* wp;
  int32_t* ids;
  int n, cap;
} sink;

static int push(sink* s, int id) {
  if (s->n >= s->cap) return 1;
  s->ids[s->n++] = id;
  return s->n >= s->cap;
}

/* one pre-tokenized word (code points) -> pieces; returns 1 once the sink is full */
static int wordpiece(sink* s, const uint32_t* w, int n) {
  if (n > MAX_WORD_CHARS) return push(s, s->wp->unk);
  char buf[2 + 4 * MAX_WORD_CHARS + 1], key[2 + 4 * MAX_WORD_CHARS + 1];
  int off[MAX_WORD_CHARS + 1], nb = 0;
  for (int i = 0; i < n; ++i) {
    off[i] = nb;
    nb += utf8_encode(w[i], buf + nb);
  }
  off[n] = nb;
  int32_t pieces[MAX_WORD_CHARS];
  int np = 0, start = 0;
  key[0] = key[1] = '#';
  while (start < n) {
    int id = -1, end = n;
    for (; end > start; --end) {
      size_t L = (size_t)(off[end] - off[start]);
      if (start == 0) {
        id = st_wordpiece_id(s->wp, buf, L);
      } else {
        memcpy(key + 2, buf + off[start], L);
        id = st_wordpiece_id(s->wp, key, L + 2);
      }
      if (id >= 0) break;
    }
    if (id < 0) return push(s, s->wp->unk);
    pieces[np++] = id;
    start = end;
  }
  for (int i = 0; i < np; ++i)
    if (push(s, pieces[i])) return 1;
  return 0;
}

/* a span without special tokens: normalize, pre-tokenize, WordPiece. 1 = sink full, -1 = error */
static int encode_span(sink* s, const char* text, size_t len) {
  if (!len) return 0;
  uint32_t* cps = (uint32_t*)st_malloc(sizeof(uint32_t) * len * (1 + ST_UNI_MAXOUT));
  uint8_t* cls = (uint8_t*)st_malloc(len * ST_UNI_MAXOUT + 1);
  if (!cps || !cls) {
    st_free(cps);
    st_free(cls);
    return -1;
  }
  size_t n = utf8_decode((const unsigned char*)text, len, cps);
  uint32_t* nrm = cps + len;   /* normalized code points */
  size_t m = 0, run = 0;       /* run: start of the current run of reorderable marks */
  for (size_t i = 0; i < n; ++i) {
    uint32_t o[ST_UNI_MAXOUT];
    int k = st_uni_norm(cps[i], o);
    if (!k) {
      if (st_uni_reorder_class(cps[i]) != 255) run = m;   /* a removed ccc-0 mark ends the run */
      continue;
    }
    for (int j = 0; j < k; ++j) {
      int c = st_uni_reorder_class(o[j]);
      c = c == 255 ? 0 : c;
      if (!c) {
        nrm[m] = o[j];
        cls[m++] = 0;
        run = m;
        continue;
      }
      /* stable insertion into the sorted run [run, m) */
      size_t p = m;
      while (p > run && cls[p - 1] > c) {
        nrm[p] = nrm[p - 1];
        cls[p] = cls[p - 1];
        --p;
      }
      nrm[p] = o[j];
      cls[p] = (uint8_t)c;
      ++m;
    }
  }
  int r = 0;
  size_t i = 0;
  while (i < m && !r) {
    uint32_t c = nrm[i];
    if (st_uni_is_space(c)) {
      ++i;
      continue;
    }
    if (st_uni_is_punct(c)) {
      r = wordpiece(s, nrm + i, 1);
      ++i;
      continue;
    }
    size_t j = i;
    while (j < m && !st_uni_is_space(nrm[j]) && !st_uni_is_punct(nrm[j])) ++j;
    r = wordpiece(s, nrm + i, (int)(j - i > MAX_WORD_CHARS ? MAX_WORD_CHARS + 1 : j - i));
    i = j;
  }
  st_free(cps);
  st_free(cls);
  return r;
}

int st_wordpiece_encode(const st_wordpiece* wp, const char* text, int32_t* ids, int max_len) {
  if (max_len < 2) {
    st_seterr("st_wordpiece_encode: max_len %d < 2", max_len);
    return -1;
  }
  sink s = {wp, ids + 1, 0, max_len - 2};
  size_t L = strlen(text), a = 0, i = 0;
  int r = s.cap == 0;
  while (i < L && !r) {
    int sp = -1;
    if (text[i] == '[')
      for (int k = 0; k < 5; ++k) {
        size_t kl = strlen(SPECIAL[k]);
        if (wp->special[k] >= 0 && i + kl <= L && memcmp(text + i, SPECIAL[k], kl) == 0) {
          sp = k;
          break;
        }
      }
    if (sp < 0) {
      ++i;
      continue;
    }
    r = encode_span(&s, text + a, i - a);
    if (r < 0) return -1;
    if (!r) r = push(&s, wp->special[sp]);
    i += strlen(SPECIAL[sp]);
    a = i;
  }
  if (!r && encode_span(&s, text + a, L - a) < 0) return -1;
  ids[0] = wp->cls;
  ids[s.n + 1] = wp->sep;
  return s.n + 2;
}
