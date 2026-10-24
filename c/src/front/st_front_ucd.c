/* Front end utilities: error context, growable buffers, UTF-8, and the CPython character tables (unicode.bin, built
 * by tools/build_native_assets.py). */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "st_front_int.h"

void fx_fail(fx* x, const char* fmt, ...) {
  if (x->err) return;  /* keep the first error, like a raised exception */
  x->err = 1;
  va_list ap;
  va_start(ap, fmt);
  char buf[1024];
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  st_seterr("%s", buf);
}

/* ------------------------------------------------------------------ buffers */
static int grow(fx* x, void** p, int* cap, int need, size_t el) {
  if (need <= *cap) return 0;
  int c = *cap ? *cap : 16;
  while (c < need) c = c > (1 << 29) ? need : c * 2;
  void* q = realloc(*p, (size_t)c * el);
  if (!q) {
    fx_fail(x, "MemoryError: out of memory");
    return -1;
  }
  *p = q;
  *cap = c;
  return 0;
}

int u32s_reserve(fx* x, st_u32s* b, int n) { return grow(x, (void**)&b->p, &b->cap, b->n + n, sizeof(u32)); }
void u32s_put(fx* x, st_u32s* b, const u32* s, int n) {
  if (n <= 0 || u32s_reserve(x, b, n)) return;
  memcpy(b->p + b->n, s, (size_t)n * sizeof(u32));
  b->n += n;
}
void u32s_push(fx* x, st_u32s* b, u32 c) {
  if (u32s_reserve(x, b, 1)) return;
  b->p[b->n++] = c;
}
void u32s_ascii(fx* x, st_u32s* b, const char* s) {
  int n = (int)strlen(s);
  if (u32s_reserve(x, b, n)) return;
  for (int i = 0; i < n; ++i) b->p[b->n++] = (unsigned char)s[i];
}
void u32s_free(st_u32s* b) {
  free(b->p);
  b->p = NULL;
  b->n = b->cap = 0;
}
void u32s_move(st_u32s* dst, st_u32s* src) {
  free(dst->p);
  *dst = *src;
  src->p = NULL;
  src->n = src->cap = 0;
}
int u32s_eq(const u32* a, int na, const u32* b, int nb) {
  return na == nb && (na == 0 || memcmp(a, b, (size_t)na * sizeof(u32)) == 0);
}
int u32_find(const u32* s, int n, u32 c) {
  for (int i = 0; i < n; ++i)
    if (s[i] == c) return i;
  return -1;
}

void sb_put(fx* x, st_sb* b, const char* s, int n) {
  if (n < 0 || grow(x, (void**)&b->p, &b->cap, b->n + n + 1, 1)) return;
  if (n) memcpy(b->p + b->n, s, (size_t)n);
  b->n += n;
  b->p[b->n] = 0;
}
void sb_cat(fx* x, st_sb* b, const char* s) { sb_put(x, b, s, (int)strlen(s)); }
void sb_ch(fx* x, st_sb* b, char c) { sb_put(x, b, &c, 1); }
void sb_fmt(fx* x, st_sb* b, const char* fmt, ...) {
  char buf[128];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  if (n >= 0) sb_put(x, b, buf, n < (int)sizeof buf ? n : (int)sizeof buf - 1);
}
const char* sb_cstr(const st_sb* b) { return b->p ? b->p : ""; }
void sb_free(st_sb* b) {
  free(b->p);
  b->p = NULL;
  b->n = b->cap = 0;
}
void sb_from_u32(fx* x, st_sb* b, const u32* s, int n) {
  if (grow(x, (void**)&b->p, &b->cap, b->n + n + 1, 1)) return;
  for (int i = 0; i < n; ++i) b->p[b->n++] = (char)s[i];
  b->p[b->n] = 0;
}

/* ------------------------------------------------------------------ UTF-8 */
void sb_utf8(fx* x, st_sb* b, const u32* s, int n) {
  if (grow(x, (void**)&b->p, &b->cap, b->n + 4 * n + 1, 1)) return;
  char* o = b->p + b->n;
  for (int i = 0; i < n; ++i) {
    u32 c = s[i];
    if (c < 0x80) {
      *o++ = (char)c;
    } else if (c < 0x800) {
      *o++ = (char)(0xC0 | (c >> 6));
      *o++ = (char)(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
      *o++ = (char)(0xE0 | (c >> 12));
      *o++ = (char)(0x80 | ((c >> 6) & 0x3F));
      *o++ = (char)(0x80 | (c & 0x3F));
    } else {
      *o++ = (char)(0xF0 | (c >> 18));
      *o++ = (char)(0x80 | ((c >> 12) & 0x3F));
      *o++ = (char)(0x80 | ((c >> 6) & 0x3F));
      *o++ = (char)(0x80 | (c & 0x3F));
    }
  }
  b->n = (int)(o - b->p);
  b->p[b->n] = 0;
}

/* Lenient decoder: invalid sequences become U+FFFD (Python strings are always valid). */
void u32s_utf8(fx* x, st_u32s* b, const char* s0, size_t n) {
  const unsigned char* s = (const unsigned char*)s0;
  if (u32s_reserve(x, b, (int)n)) return;
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
    b->p[b->n++] = cp;
    i += len;
  }
}

/* ------------------------------------------------------------------ unicode.bin */
typedef struct {
  const unsigned char* p;
  size_t n, o;
  int bad;
} rd;
static u32 rd32(rd* r) {
  if (r->o + 4 > r->n) {
    r->bad = 1;
    return 0;
  }
  const unsigned char* q = r->p + r->o;
  r->o += 4;
  return (u32)q[0] | ((u32)q[1] << 8) | ((u32)q[2] << 16) | ((u32)q[3] << 24);  /* assets are little endian */
}
static int rd8(rd* r) {
  if (r->o + 1 > r->n) {
    r->bad = 1;
    return 0;
  }
  return r->p[r->o++];
}
static const unsigned char* rdbytes(rd* r, size_t k) {
  if (r->o + k > r->n) {
    r->bad = 1;
    return NULL;
  }
  const unsigned char* q = r->p + r->o;
  r->o += k;
  return q;
}
/* entry count, rejecting counts the remaining bytes cannot hold */
static int rdcount(rd* r, size_t min_bytes) {
  u32 k = rd32(r);
  if (r->bad || (size_t)k * min_bytes > r->n - r->o) {
    r->bad = 1;
    return 0;
  }
  return (int)k;
}

static int cmp_str_entry(const void* a, const void* b) {
  const ucd_str *x = (const ucd_str*)a, *y = (const ucd_str*)b;
  return x->c < y->c ? -1 : x->c > y->c ? 1 : (x->idx < y->idx ? -1 : x->idx > y->idx);
}
static int cmp_u32_entry(const void* a, const void* b) {
  const ucd_u32 *x = (const ucd_u32*)a, *y = (const ucd_u32*)b;
  return x->c < y->c ? -1 : x->c > y->c ? 1 : (x->idx < y->idx ? -1 : x->idx > y->idx);
}

int ucd_load(st_ucd* u, const char* path) {
  memset(u, 0, sizeof *u);
  char* data;
  size_t len;
  if (st_read_file(path, &data, &len)) {
    st_seterr("steertts front: cannot open %s", path);
    return -1;
  }
  rd r = {(const unsigned char*)data, len, 0, 0};
  fx x = {0};
  const unsigned char* magic = rdbytes(&r, 4);
  if (!magic || memcmp(magic, "STUC", 4) != 0 || rd32(&r) != 1) {
    st_free(data);
    st_seterr("steertts front: bad unicode.bin");
    return -1;
  }
  int n = rdcount(&r, 8);
  u->word = calloc((size_t)n + 1, sizeof *u->word);
  if (!u->word) goto oom;
  for (int k = 0; k < n; ++k) {
    u->word[k][0] = rd32(&r);
    u->word[k][1] = rd32(&r);
  }
  u->nword = n;
  n = rdcount(&r, 4);
  u->space = calloc((size_t)n + 1, sizeof(u32));
  if (!u->space) goto oom;
  for (int k = 0; k < n; ++k) {
    u32 c = rd32(&r);
    if (c >= 128) u->space[u->nspace++] = c;
  }
  n = rdcount(&r, 12);
  u->digit = calloc((size_t)n + 1, sizeof *u->digit);
  if (!u->digit) goto oom;
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < 3; ++j) u->digit[k][j] = rd32(&r);
  u->ndigit = n;
  n = rdcount(&r, 5);
  u->g2p = calloc((size_t)n + 1, sizeof *u->g2p);
  if (!u->g2p) goto oom;
  for (int k = 0; k < n && !r.bad; ++k) {
    u32 c = rd32(&r);
    int l = rd8(&r);
    const unsigned char* s = rdbytes(&r, (size_t)l);
    if (!s) break;
    char* d = malloc((size_t)l + 1);
    if (!d) goto oom;
    memcpy(d, s, (size_t)l);
    d[l] = 0;
    if (c < 128) {  /* later entries win (dict assignment) */
      free(u->ascii_g2p[c]);
      u->ascii_g2p[c] = d;
      u->ascii_g2p_n[c] = l;
    } else {
      u->g2p[u->ng2p].c = c;
      u->g2p[u->ng2p].s = d;
      u->g2p[u->ng2p].n = l;
      u->g2p[u->ng2p].idx = u->ng2p;
      u->ng2p++;
    }
  }
  n = rdcount(&r, 8);
  u->fold = calloc((size_t)n + 1, sizeof *u->fold);
  if (!u->fold) goto oom;
  for (int k = 0; k < n; ++k) {
    u->fold[k][0] = rd32(&r);
    u->fold[k][1] = rd32(&r);
  }
  u->nfold = n;
  n = rdcount(&r, 5);
  u->lower = calloc((size_t)n + 1, sizeof *u->lower);
  if (!u->lower) goto oom;
  for (int k = 0; k < n && !r.bad; ++k) {
    u32 c = rd32(&r);
    int l = rd8(&r);
    const unsigned char* s = rdbytes(&r, (size_t)l);
    if (!s) break;
    st_u32s t = {0};
    u32s_utf8(&x, &t, (const char*)s, (size_t)l);
    if (x.err) goto oom;
    u->lower[u->nlower].c = c;
    u->lower[u->nlower].s = t.p;
    u->lower[u->nlower].n = t.n;
    u->lower[u->nlower].idx = u->nlower;
    u->nlower++;
  }
  if (r.bad) {
    ucd_free(u);
    st_free(data);
    st_seterr("steertts front: truncated asset");
    return -1;
  }
  st_free(data);
  /* sort by code point, keeping the last duplicate (dict semantics) */
  qsort(u->g2p, (size_t)u->ng2p, sizeof *u->g2p, cmp_str_entry);
  int m = 0;
  for (int k = 0; k < u->ng2p; ++k) {
    if (k + 1 < u->ng2p && u->g2p[k + 1].c == u->g2p[k].c) {
      free(u->g2p[k].s);
      continue;
    }
    u->g2p[m++] = u->g2p[k];
  }
  u->ng2p = m;
  qsort(u->lower, (size_t)u->nlower, sizeof *u->lower, cmp_u32_entry);
  m = 0;
  for (int k = 0; k < u->nlower; ++k) {
    if (k + 1 < u->nlower && u->lower[k + 1].c == u->lower[k].c) {
      free(u->lower[k].s);
      continue;
    }
    u->lower[m++] = u->lower[k];
  }
  u->nlower = m;
  return 0;
oom:
  ucd_free(u);
  st_free(data);
  st_seterr("steertts front: out of memory");
  return -1;
}

void ucd_free(st_ucd* u) {
  free(u->word);
  free(u->space);
  free(u->digit);
  for (int c = 0; c < 128; ++c) free(u->ascii_g2p[c]);
  for (int k = 0; k < u->ng2p; ++k) free(u->g2p[k].s);
  free(u->g2p);
  free(u->fold);
  for (int k = 0; k < u->nlower; ++k) free(u->lower[k].s);
  free(u->lower);
  memset(u, 0, sizeof *u);
}

int ucd_is_word(const st_ucd* u, u32 c) {
  if (c < 128) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
  int lo = 0, hi = u->nword;  /* last range with start <= c */
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (u->word[mid][0] <= c)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo > 0 && c <= u->word[lo - 1][1];
}

int ucd_is_space(const st_ucd* u, u32 c) {
  if (c < 128) return c == ' ' || (c >= 9 && c <= 13) || (c >= 28 && c <= 31);
  for (int k = 0; k < u->nspace; ++k)
    if (u->space[k] == c) return 1;
  return 0;
}

int ucd_digit(const st_ucd* u, u32 c) {
  if (c >= '0' && c <= '9') return (int)(c - '0');
  if (c < 128) return -1;
  for (int k = 0; k < u->ndigit; ++k)
    if (c >= u->digit[k][0] && c <= u->digit[k][1]) return (int)(u->digit[k][2] + (c - u->digit[k][0]));
  return -1;
}

u32 ucd_fold(const st_ucd* u, u32 c) {
  if (c >= 'A' && c <= 'Z') return c + 32;
  if (c < 128) return c;
  for (int k = 0; k < u->nfold; ++k)
    if (u->fold[k][0] == c) return u->fold[k][1];
  return c;
}

/* str.lower() for ascii and the few non-ascii code points markup tag names can contain */
void ucd_lower_small(fx* x, const st_ucd* u, const u32* s, int n, st_u32s* out) {
  for (int i = 0; i < n; ++i) {
    u32 c = s[i];
    if (c >= 'A' && c <= 'Z') {
      u32s_push(x, out, c + 32);
    } else if (c < 128) {
      u32s_push(x, out, c);
    } else {
      int lo = 0, hi = u->nlower - 1, f = -1;
      while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (u->lower[mid].c == c) {
          f = mid;
          break;
        }
        if (u->lower[mid].c < c)
          lo = mid + 1;
        else
          hi = mid - 1;
      }
      if (f >= 0)
        u32s_put(x, out, u->lower[f].s, u->lower[f].n);
      else
        u32s_push(x, out, c);
    }
  }
}

/* g2p_en clean step for one code point: NFD, drop Mn, lower, keep [ a-z'.,?!-] */
const char* ucd_g2p_clean(const st_ucd* u, u32 c, int* len) {
  if (c < 128) {
    *len = u->ascii_g2p_n[c];
    return u->ascii_g2p[c] ? u->ascii_g2p[c] : "";
  }
  int lo = 0, hi = u->ng2p - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    if (u->g2p[mid].c == c) {
      *len = u->g2p[mid].n;
      return u->g2p[mid].s;
    }
    if (u->g2p[mid].c < c)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  *len = 0;
  return "";
}
