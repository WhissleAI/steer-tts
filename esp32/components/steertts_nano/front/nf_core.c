/* Front end core: front.bin sections, Unicode tables, UTF-8, lexicon lookup, errors. */
#include <stdarg.h>
#include <stdio.h>

#include "nf_int.h"

void nf_fail(nano_front_ctx* c, const char* fmt, ...) {
  if (c->fail) return;   /* keep the first error, like a raised exception */
  c->fail = 1;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(c->err, sizeof c->err, fmt, ap);
  va_end(ap);
}

enum {
  SEC_UCD_WORD = 10, SEC_UCD_SPACE, SEC_UCD_DIGIT, SEC_UCD_G2P_ASCII, SEC_UCD_G2P, SEC_UCD_FOLD, SEC_UCD_LOWER,
  SEC_LEX = 20, SEC_OOV, SEC_GRU = 30, SEC_HET_W = 40, SEC_HET_T, SEC_HET_H, SEC_HET_FLAGS
};

int nano_front_open(nano_front_ctx* c, const void* data, size_t len) {
  memset(&c->d, 0, sizeof c->d);
  c->fail = 0;
  c->err[0] = 0;
  const uint8_t* b = (const uint8_t*)data;
  if (!b || len < 16 || memcmp(b, "STFB", 4) || rd32(b + 4) != 1) {
    nf_fail(c, "not a front.bin image");
    return -1;
  }
  u32 n = rd32(b + 8);
  if (12 + 12 * (size_t)n > len) {
    nf_fail(c, "front.bin truncated");
    return -1;
  }
  c->d.base = b;
  c->d.len = len;
  for (u32 i = 0; i < n; i++) {
    const uint8_t* e = b + 12 + 12 * i;
    u32 id = rd32(e), off = rd32(e + 4), sz = rd32(e + 8);
    if ((size_t)off + sz > len || (off & 3)) {
      nf_fail(c, "front.bin: bad section table");
      return -1;
    }
    const uint8_t* p = b + off;
    switch (id) {
      case SEC_UCD_WORD: c->d.word = p; break;
      case SEC_UCD_SPACE: c->d.space = p; break;
      case SEC_UCD_DIGIT: c->d.digit = p; break;
      case SEC_UCD_G2P_ASCII: c->d.g2p_ascii = p; break;
      case SEC_UCD_G2P: c->d.g2p = p; break;
      case SEC_UCD_FOLD: c->d.fold = p; break;
      case SEC_UCD_LOWER: c->d.lower = p; break;
      case SEC_LEX: c->d.lex = p; break;
      case SEC_OOV: c->d.oov = p; break;
      case SEC_GRU: c->d.gru = p; break;
      case SEC_HET_W: c->d.het_w = p; break;
      case SEC_HET_T: c->d.het_t = p; break;
      case SEC_HET_H: c->d.het_h = p; break;
      case SEC_HET_FLAGS: c->d.has_td = (int)rd32(p); break;
      default: break;
    }
  }
  if (!c->d.word || !c->d.space || !c->d.digit || !c->d.g2p_ascii || !c->d.g2p || !c->d.fold || !c->d.lower ||
      !c->d.lex || !c->d.oov || !c->d.gru || !c->d.het_w || !c->d.het_t || !c->d.het_h) {
    nf_fail(c, "front.bin: missing sections");
    return -1;
  }
  if (memcmp(c->d.gru, "STGR", 4) || rd32(c->d.gru + 4) != 2 || rd32(c->d.gru + 8) > NF_GRU_MAX_H ||
      rd32(c->d.gru + 8) % 32 || rd32(c->d.gru + 16) > 128) {
    nf_fail(c, "front.bin: unsupported G2P model");
    return -1;
  }
  return 0;
}

/* ------------------------------------------------------------------ Unicode (CPython tables) */
int nf_is_word(const nano_front_ctx* c, u32 ch) {
  if (ch < 128) return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_';
  const uint8_t* t = c->d.word;
  int lo = 0, hi = (int)rd32(t);   /* last range with start <= ch */
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (rd32(t + 4 + 8 * mid) <= ch)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo > 0 && ch <= rd32(t + 4 + 8 * (lo - 1) + 4);
}

int nf_is_space(const nano_front_ctx* c, u32 ch) {
  if (ch < 128) return ch == ' ' || (ch >= 9 && ch <= 13) || (ch >= 28 && ch <= 31);
  const uint8_t* t = c->d.space;
  int n = (int)rd32(t);
  for (int k = 0; k < n; k++)
    if (rd32(t + 4 + 4 * k) == ch) return 1;
  return 0;
}

int nf_digit(const nano_front_ctx* c, u32 ch) {
  if (ch >= '0' && ch <= '9') return (int)(ch - '0');
  if (ch < 128) return -1;
  const uint8_t* t = c->d.digit;
  int n = (int)rd32(t);
  for (int k = 0; k < n; k++) {
    const uint8_t* e = t + 4 + 12 * k;
    if (ch >= rd32(e) && ch <= rd32(e + 4)) return (int)(rd32(e + 8) + (ch - rd32(e)));
  }
  return -1;
}

u32 nf_fold(const nano_front_ctx* c, u32 ch) {
  if (ch >= 'A' && ch <= 'Z') return ch + 32;
  if (ch < 128) return ch;
  const uint8_t* t = c->d.fold;
  int n = (int)rd32(t);
  for (int k = 0; k < n; k++)
    if (rd32(t + 4 + 8 * k) == ch) return rd32(t + 8 + 8 * k);
  return ch;
}

int nf_lower(const nano_front_ctx* c, u32 ch, u32* out) {
  if (ch >= 'A' && ch <= 'Z') {
    out[0] = ch + 32;
    return 1;
  }
  if (ch >= 128) {
    const uint8_t* t = c->d.lower;
    int n = (int)rd32(t);
    for (int k = 0; k < n; k++) {
      const uint8_t* e = t + 4 + 20 * k;
      if (rd32(e) == ch) {
        int m = (int)rd32(e + 4);
        for (int i = 0; i < m; i++) out[i] = rd32(e + 8 + 4 * i);
        return m;
      }
    }
  }
  out[0] = ch;
  return 1;
}

const char* nf_g2p_clean(const nano_front_ctx* c, u32 ch, int* len) {
  if (ch < 128) {
    const uint8_t* e = c->d.g2p_ascii + 4 * ch;
    *len = e[0];
    return (const char*)e + 1;
  }
  const uint8_t* t = c->d.g2p;
  int n = (int)rd32(t), lo = 0, hi = n - 1;
  const uint8_t* pool = t + 4 + 8 * n;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    const uint8_t* e = t + 4 + 8 * mid;
    u32 cp = rd32(e);
    if (cp == ch) {
      *len = e[6];
      return (const char*)pool + (e[4] | (e[5] << 8));
    }
    if (cp < ch)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  *len = 0;
  return "";
}

/* ------------------------------------------------------------------ UTF-8 (lenient: invalid -> U+FFFD) */
int nf_utf8_decode(const char* s0, size_t n, nfs* b) {
  const unsigned char* s = (const unsigned char*)s0;
  size_t i = 0;
#define CONT(k) ((k) < n && (s[k] & 0xC0) == 0x80)
  while (i < n) {
    unsigned c = s[i];
    u32 cp = 0xFFFD;
    size_t len = 1;
    if (c < 0x80) {
      cp = c;
    } else if ((c & 0xE0) == 0xC0 && CONT(i + 1)) {
      cp = ((c & 0x1Fu) << 6) | (s[i + 1] & 0x3Fu);
      len = 2;
      if (cp < 0x80) cp = 0xFFFD;
    } else if ((c & 0xF0) == 0xE0 && CONT(i + 1) && CONT(i + 2)) {
      cp = ((c & 0x0Fu) << 12) | ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3Fu);
      len = 3;
      if (cp < 0x800 || (cp >= 0xD800 && cp < 0xE000)) cp = 0xFFFD;
    } else if ((c & 0xF8) == 0xF0 && CONT(i + 1) && CONT(i + 2) && CONT(i + 3)) {
      cp = ((c & 0x07u) << 18) | ((s[i + 1] & 0x3Fu) << 12) | ((s[i + 2] & 0x3Fu) << 6) | (s[i + 3] & 0x3Fu);
      len = 4;
      if (cp < 0x10000 || cp > 0x10FFFF) cp = 0xFFFD;
    }
#undef CONT
    nfs_push(b, cp);
    i += len;
  }
  return b->c->fail ? -1 : 0;
}

int nf_utf8_encode(const u32* s, int n, char* o0, int cap) {
  int o = 0;
  for (int i = 0; i < n; i++) {
    u32 c = s[i];
    char t[4];
    int k;
    if (c < 0x80) {
      t[0] = (char)c;
      k = 1;
    } else if (c < 0x800) {
      t[0] = (char)(0xC0 | (c >> 6));
      t[1] = (char)(0x80 | (c & 0x3F));
      k = 2;
    } else if (c < 0x10000) {
      t[0] = (char)(0xE0 | (c >> 12));
      t[1] = (char)(0x80 | ((c >> 6) & 0x3F));
      t[2] = (char)(0x80 | (c & 0x3F));
      k = 3;
    } else {
      t[0] = (char)(0xF0 | (c >> 18));
      t[1] = (char)(0x80 | ((c >> 12) & 0x3F));
      t[2] = (char)(0x80 | ((c >> 6) & 0x3F));
      t[3] = (char)(0x80 | (c & 0x3F));
      k = 4;
    }
    if (o + k >= cap) return -1;
    memcpy(o0 + o, t, (size_t)k);
    o += k;
  }
  if (o >= cap) return -1;
  o0[o] = 0;
  return o;
}

/* ------------------------------------------------------------------ lexicon
 * header u32 n, nblocks, block; u32 boff[nblocks + 1]; per entry: u8 prefix, u8 suffix_len, suffix, u8 nph, phones */
static int cmp_bytes(const uint8_t* a, int na, const char* w, int nw) {
  int m = na < nw ? na : nw;
  int r = memcmp(a, w, (size_t)m);
  if (r) return r;
  return na < nw ? -1 : na > nw;
}

int nf_lex_get(const uint8_t* sec, const char* w, uint8_t* ph, int* nph) {
  u32 nb = rd32(sec + 4);
  const uint8_t* boff = sec + 12;
  const uint8_t* data = boff + 4 * (nb + 1);
  int wl = (int)strlen(w);
  /* last block whose first word <= w */
  int lo = 0, hi = (int)nb;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    const uint8_t* e = data + rd32(boff + 4 * mid);
    if (cmp_bytes(e + 2, e[1], w, wl) <= 0)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == 0) return 0;
  const uint8_t* e = data + rd32(boff + 4 * (lo - 1));
  const uint8_t* end = data + rd32(boff + 4 * lo);
  char cur[256];
  int cl = 0;
  while (e < end) {
    int pre = e[0], sl = e[1];
    if (pre > cl) return 0;
    memcpy(cur + pre, e + 2, (size_t)sl);
    cl = pre + sl;
    const uint8_t* p = e + 2 + sl;
    int np = p[0];
    int r = cmp_bytes((const uint8_t*)cur, cl, w, wl);
    if (r == 0) {
      if (ph) {
        int k = np < 64 ? np : 64;
        memcpy(ph, p + 1, (size_t)k);
        *nph = k;
      }
      return 1;
    }
    if (r > 0) return 0;
    e = p + 1 + np;
  }
  return 0;
}

/* sound-event symbols: [laugh] = 78 ... [clock] = 104, [reserved0..12] = 105..117 */
int nf_event_symbol(const char* nm, int n) {
  static const char* ev[] = {"laugh",  "cry",   "scream", "yell",     "cheer", "cough", "sneeze",    "yawn",   "throat",
                             "breath", "sigh",  "sniff",  "dog",      "cat",   "bird",  "knock",     "door",   "phone",
                             "bell",   "horn",  "applause", "glass",  "siren", "footsteps", "typing", "thunder", "clock"};
  for (int k = 0; k < 27; k++)
    if ((int)strlen(ev[k]) == n && !memcmp(ev[k], nm, (size_t)n)) return 78 + k;
  if (n >= 9 && n <= 10 && !memcmp(nm, "reserved", 8)) {
    int v = 0;
    for (int i = 8; i < n; i++) {
      if (nm[i] < '0' || nm[i] > '9') return -1;
      v = 10 * v + (nm[i] - '0');
    }
    if (n == 10 && nm[8] == '0') return -1;   /* "reserved01" is not a symbol */
    if (v <= 12) return 105 + v;
  }
  return -1;
}
