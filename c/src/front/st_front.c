/* Port of steertts_edge/text.py (v1 edge path: normalize -> _expand -> EdgeG2p -> to_phonemes) and the public API.
 * The "v2 front-end" section of text.py is not used by to_phonemes / normalize and is not ported. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "st_front_int.h"

/* ------------------------------------------------------------------ symbols */
static int make_symbols(st_front* f) {
  static const char* arpa[] = {"AA", "AE", "AH", "AO", "AW", "AY", "B",  "CH", "D",  "DH", "EH", "ER", "EY",
                               "F",  "G",  "HH", "IH", "IY", "JH", "K",  "L",  "M",  "N",  "NG", "OW", "OY",
                               "P",  "R",  "S",  "SH", "T",  "TH", "UH", "UW", "V",  "W",  "Y",  "Z",  "ZH"};
  static const char* ev[] = {"laugh", "cry",   "scream", "yell",  "cheer",    "cough",  "sneeze",  "yawn",  "throat",
                             "breath", "sigh", "sniff",  "dog",   "cat",      "bird",   "knock",   "door",  "phone",
                             "bell",  "horn",  "applause", "glass", "siren",  "footsteps", "typing", "thunder",
                             "clock"};
  const char* vowels = " AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW ";
  char b[32];
  int n = 0;
  f->symbols[n++] = st_strdup("<pad>");
  f->symbols[n++] = st_strdup("<sp>");
  for (const char* p = ",.?!;:-"; *p; ++p) {
    b[0] = *p;
    b[1] = 0;
    f->symbols[n++] = st_strdup(b);
  }
  for (size_t k = 0; k < sizeof arpa / sizeof arpa[0]; ++k) {
    snprintf(b, sizeof b, " %s ", arpa[k]);
    if (strstr(vowels, b)) {
      for (int st = 0; st < 3; ++st) {
        snprintf(b, sizeof b, "%s%d", arpa[k], st);
        f->symbols[n++] = st_strdup(b);
      }
    } else {
      f->symbols[n++] = st_strdup(arpa[k]);
    }
  }
  for (size_t k = 0; k < sizeof ev / sizeof ev[0]; ++k) {
    snprintf(b, sizeof b, "[%s]", ev[k]);
    f->symbols[n++] = st_strdup(b);
  }
  for (int k = 0; k < 13; ++k) {
    snprintf(b, sizeof b, "[reserved%d]", k);
    f->symbols[n++] = st_strdup(b);
  }
  if (n != ST_N_SYMBOLS) {
    st_seterr("steertts front: symbol table has %d entries", n);
    return -1;
  }
  for (int k = 0; k < n; ++k)
    if (!f->symbols[k]) return -1;
  /* insertion sort by name (118 entries) */
  for (int k = 0; k < n; ++k) {
    int id = k, j = k;
    while (j > 0 && strcmp(f->symbols[f->sym_sorted[j - 1]], f->symbols[id]) > 0) {
      f->sym_sorted[j] = f->sym_sorted[j - 1];
      --j;
    }
    f->sym_sorted[j] = id;
  }
  return 0;
}

int st_front_symbol_id(const st_front* f, const char* s) {
  int lo = 0, hi = ST_N_SYMBOLS - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2, c = strcmp(f->symbols[f->sym_sorted[mid]], s);
    if (!c) return f->sym_sorted[mid];
    if (c < 0)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return -1;
}

const char* st_front_symbol(const st_front* f, int id) {
  return id >= 0 && id < ST_N_SYMBOLS ? f->symbols[id] : NULL;
}

/* ------------------------------------------------------------------ load / free */
void st_front_free(st_front* f) {
  if (!f) return;
  g2p_free(&f->g2p);
  st_rx* all[] = {f->quotes, f->range,       f->by,      f->compound, f->currency, f->percent, f->ordinal,
                  f->comma_num, f->decimal,  f->long_digits, f->camel_acr, f->acronym, f->ws,    f->event,   f->emph,
                  f->token,  f->sent_end};
  for (size_t k = 0; k < sizeof all / sizeof all[0]; ++k) st_rx_free(all[k]);
  for (int k = 0; k < 15; ++k) st_rx_free(f->units[k]);
  for (int k = 0; k < ST_N_SYMBOLS; ++k) st_free(f->symbols[k]);
  ucd_free(&f->ucd);
  st_free(f);
}

st_front* st_front_load(const char* dir) {
  if (!dir || !*dir) {
    st_seterr("steertts front: no asset directory");
    return NULL;
  }
  st_front* f = st_calloc(sizeof *f);
  if (!f) return NULL;
  size_t dl = strlen(dir);
  char* path = malloc(dl + 16);
  if (!path) {
    st_free(f);
    st_seterr("steertts front: out of memory");
    return NULL;
  }
  memcpy(path, dir, dl);
  strcpy(path + dl, "/unicode.bin");
  int rc = ucd_load(&f->ucd, path);
  free(path);
  if (rc || make_symbols(f) || g2p_init(&f->g2p, dir, &f->ucd, f)) goto fail;
  const st_ucd* u = &f->ucd;
  struct {
    st_rx** r;
    const char* p;
    int icase;
  } t[] = {
      {&f->quotes, "[\\\"\xe2\x80\x9c\xe2\x80\x9d\xe2\x80\x98\xe2\x80\x99`]", 0},   /* ["“”‘’`] */
      {&f->range, "(\\d)\\s*[-\xe2\x80\x93]\\s*(\\d)", 0},                         /* en dash */
      {&f->by, "(\\d+)\\s*[x\xc3\x97]\\s*(\\d+)", 0},                              /* × */
      {&f->compound, "(?<=[A-Za-z0-9])[-\xe2\x80\x90/](?=[A-Za-z0-9])", 0},        /* hyphen U+2010 */
      {&f->currency,
       "([$\xc2\xa3\xe2\x82\xac])\\s?(\\d(?:[\\d,]*\\d)?(?:\\.\\d+)?)(?:\\s*(million|billion|thousand)\\b)?", 0},
      {&f->percent, "(\\d[\\d,]*(?:\\.\\d+)?)\\s?%", 0},
      {&f->ordinal, "\\b(\\d+)(st|nd|rd|th)\\b", 0},
      {&f->comma_num, "\\b\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?\\b", 0},
      {&f->decimal, "\\b\\d+\\.\\d+\\b", 0},
      {&f->long_digits, "\\b\\d{5,}\\b", 0},
      {&f->camel_acr, "(?<=[a-z])([A-Z]{2,5}s?)\\b", 0},                         /* SteerTTS -> Steer TTS */
      {&f->acronym, "\\b[A-Z]{2,5}s?\\b", 0},
      {&f->ws, "\\s+", 0},
      {&f->event, "\\[\\s*([A-Za-z_]+)\\s*\\]", 0},
      {&f->emph, "\\*([^*\\n]+)\\*", 0},
      {&f->token,
       "(<\\s*(/?)\\s*([A-Za-z_]+)(?:\\s+prompt\\s*=\\s*\"([^\"]*)\")?\\s*>)"
       "|(\\[\\s*([A-Za-z_]+)(?:\\s+([\\d.]+)\\s*(ms|s)?)?\\s*\\])",
       1},
      {&f->sent_end, "(?<=[.!?])\\s+", 0},
  };
  for (size_t k = 0; k < sizeof t / sizeof t[0]; ++k)
    if (!(*t[k].r = st_rx_compile(u, t[k].p, t[k].icase))) goto fail;
  /* text._UNITS in dict order, then sorted(key=len, reverse=True) (stable); '/' matches "[ /]" */
  static const char* units[][2] = {
      {"km/h", "kilometers per hour"}, {"km/hr", "kilometers per hour"}, {"m/s", "meters per second"},
      {"mph", "miles per hour"},       {"km", "kilometers"},             {"kg", "kilograms"},
      {"cm", "centimeters"},           {"mm", "millimeters"},            {"ft", "feet"},
      {"lbs", "pounds"},               {"lb", "pounds"},                 {"kHz", "kilohertz"},
      {"Hz", "hertz"},                 {"GB", "gigabytes"},              {"MB", "megabytes"}};
  int order[15], n = 0;
  for (int len = 8; len >= 1; --len)
    for (int k = 0; k < 15; ++k)
      if ((int)strlen(units[k][0]) == len) order[n++] = k;
  for (int k = 0; k < 15; ++k) {
    char pat[64] = "(\\d)\\s?";
    for (const char* c = units[order[k]][0]; *c; ++c) strcat(pat, *c == '/' ? "[ /]" : (char[2]){*c, 0});
    strcat(pat, "\\b");
    f->unit_words[k] = units[order[k]][1];
    if (!(f->units[k] = st_rx_compile(u, pat, 0))) goto fail;
  }
  return f;
fail:
  st_front_free(f);
  return NULL;
}

/* ------------------------------------------------------------------ helpers */
void front_strip(const st_front* f, const u32* s, int n, int* a, int* b) {
  int i = 0, j = n;
  while (i < j && ucd_is_space(&f->ucd, s[i])) ++i;
  while (j > i && ucd_is_space(&f->ucd, s[j - 1])) --j;
  *a = i;
  *b = j;
}

static void repl_space(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) { u32s_push(x, out, ' '); }

void front_collapse_ws(fx* x, const st_front* f, const u32* s, int n, st_u32s* out) {
  st_rx_sub(x, f->ws, s, n, repl_space, NULL, out);
}

static void sub_inplace(fx* x, const st_rx* r, st_u32s* s, st_rx_repl fn, void* ud) {
  st_u32s t = {0};
  st_rx_sub(x, r, s->p, s->n, fn, ud, &t);
  u32s_move(s, &t);
}

/* text._num: num2words(...).replace("-", " ").replace(",", "") appended as code points */
static void put_num_words(fx* x, const st_sb* w, st_u32s* out) {
  for (int i = 0; i < w->n; ++i) {
    char c = w->p[i];
    if (c == '-')
      u32s_push(x, out, ' ');
    else if (c != ',')
      u32s_push(x, out, (unsigned char)c);
  }
}
static void num_int(fx* x, const st_front* f, const u32* s, int n, int ord, st_u32s* out) {
  st_sb d = {0}, w = {0};
  py_int_digits(x, &f->ucd, s, n, &d);
  if (!x->err) {
    if (ord)
      nw_ordinal(x, sb_cstr(&d), &w);
    else
      nw_cardinal(x, sb_cstr(&d), &w);
  }
  if (!x->err) put_num_words(x, &w, out);
  sb_free(&d);
  sb_free(&w);
}
/* text._num on float(s): Python float() accepts any Unicode decimal digits */
static void num_float(fx* x, const st_front* f, const u32* s, int n, st_u32s* out) {
  st_sb a = {0}, w = {0};
  for (int i = 0; i < n; ++i) {
    int d = ucd_digit(&f->ucd, s[i]);
    sb_ch(x, &a, d >= 0 ? (char)('0' + d) : (char)s[i]);
  }
  if (!x->err) nw_cardinal_float(x, st_front_strtod(sb_cstr(&a)), &w);
  if (!x->err) put_num_words(x, &w, out);
  sb_free(&a);
  sb_free(&w);
}

/* ------------------------------------------------------------------ _expand */
static void r_range(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  rxm_put(x, o, m, s, 1);
  u32s_ascii(x, o, " to ");
  rxm_put(x, o, m, s, 2);
}
static void r_by(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  rxm_put(x, o, m, s, 1);
  u32s_ascii(x, o, " by ");
  rxm_put(x, o, m, s, 2);
}
static void r_currency(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  u32 c = s[m->caps[2]];
  rxm_put(x, o, m, s, 2);
  u32s_push(x, o, ' ');
  if (rxm_has(m, 3)) {
    rxm_put(x, o, m, s, 3);
    u32s_push(x, o, ' ');
  }
  u32s_ascii(x, o, c == '$' ? "dollars" : c == 0xA3 ? "pounds" : "euros");
}
static void r_percent(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  rxm_put(x, o, m, s, 1);
  u32s_ascii(x, o, " percent");
}
static void r_unit(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  rxm_put(x, o, m, s, 1);
  u32s_push(x, o, ' ');
  u32s_ascii(x, o, (const char*)ud);
}
static void r_ordinal(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  num_int(x, (const st_front*)ud, rxm_ptr(m, s, 1), rxm_len(m, 1), 1, o);
}
static void r_comma_num(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  st_u32s nc = {0};
  int dot = 0;
  for (int i = m->start; i < m->end; ++i) {
    if (s[i] != ',') u32s_push(x, &nc, s[i]);
    if (s[i] == '.') dot = 1;
  }
  if (dot)
    num_float(x, (const st_front*)ud, nc.p, nc.n, o);
  else
    num_int(x, (const st_front*)ud, nc.p, nc.n, 0, o);
  u32s_free(&nc);
}
static void r_decimal(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  num_float(x, (const st_front*)ud, s + m->start, m->end - m->start, o);
}
static void r_long_digits(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  num_int(x, (const st_front*)ud, s + m->start, m->end - m->start, 0, o);
}
static void r_acronym(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  const st_front* f = (const st_front*)ud;
  char g[8], low[8];  /* [A-Z]{2,5}s? : at most 6 ascii characters */
  int n = m->end - m->start;
  for (int i = 0; i < n; ++i) {
    g[i] = (char)s[m->start + i];
    low[i] = g[i] >= 'A' && g[i] <= 'Z' ? (char)(g[i] + 32) : g[i];
  }
  g[n] = low[n] = 0;
  if (g2p_in_cmu(&f->g2p, low)) {
    u32s_ascii(x, o, g);
    return;
  }
  int b = n;
  while (b > 0 && g[b - 1] == 's') --b;  /* rstrip("s") */
  for (int i = 0; i < b; ++i) {
    if (i) u32s_push(x, o, ' ');
    u32s_push(x, o, (unsigned char)g[i]);
  }
  if (g[n - 1] == 's' && n > 2) u32s_ascii(x, o, "'s");
}

static void r_camel_acr(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  u32s_push(x, o, ' ');
  for (int i = m->caps[2]; i < m->caps[3]; ++i) u32s_push(x, o, s[i]);
}

static void expand(fx* x, const st_front* f, st_u32s* t) {
  void* F = (void*)f;
  sub_inplace(x, f->range, t, r_range, NULL);
  sub_inplace(x, f->by, t, r_by, NULL);
  sub_inplace(x, f->compound, t, repl_space, NULL);
  sub_inplace(x, f->currency, t, r_currency, NULL);
  sub_inplace(x, f->percent, t, r_percent, NULL);
  for (int k = 0; k < 15; ++k) sub_inplace(x, f->units[k], t, r_unit, (void*)f->unit_words[k]);
  sub_inplace(x, f->ordinal, t, r_ordinal, F);
  sub_inplace(x, f->comma_num, t, r_comma_num, F);
  sub_inplace(x, f->decimal, t, r_decimal, F);
  sub_inplace(x, f->long_digits, t, r_long_digits, F);
  sub_inplace(x, f->camel_acr, t, r_camel_acr, NULL);
  sub_inplace(x, f->acronym, t, r_acronym, F);
}

static void r_quote(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  u32 c = s[m->start];
  if (c == 0x2018 || c == 0x2019 || c == '`') u32s_push(x, o, '\'');
}

static void replace_char(fx* x, st_u32s* t, u32 from, const char* to) {
  if (u32_find(t->p, t->n, from) < 0) return;
  st_u32s o = {0};
  for (int i = 0; i < t->n; ++i) {
    if (t->p[i] == from)
      u32s_ascii(x, &o, to);
    else
      u32s_push(x, &o, t->p[i]);
  }
  u32s_move(t, &o);
}

void front_normalize(fx* x, const st_front* f, const u32* s, int n, st_u32s* out) {
  st_u32s t = {0}, c = {0};
  st_rx_sub(x, f->quotes, s, n, r_quote, NULL, &t);
  expand(x, f, &t);  /* first: it reads digit-dash-digit as a range before dashes become pauses */
  replace_char(x, &t, 0x2014, ", ");  /* — */
  replace_char(x, &t, 0x2013, ", ");  /* – */
  replace_char(x, &t, 0x2026, ". ");  /* … */
  front_collapse_ws(x, f, t.p, t.n, &c);
  int a, b;
  front_strip(f, c.p, c.n, &a, &b);
  u32s_put(x, out, c.p + a, b - a);
  u32s_free(&t);
  u32s_free(&c);
}

/* ------------------------------------------------------------------ to_phonemes */
typedef struct {
  int *ids, *widx;
  int n, cap;
} phbuf;
static void ph_push(fx* x, phbuf* p, int id, int w) {
  if (p->n == p->cap) {
    int c = p->cap ? 2 * p->cap : 64;
    int* a = realloc(p->ids, (size_t)c * sizeof(int));
    if (!a) {
      fx_fail(x, "MemoryError: out of memory");
      return;
    }
    p->ids = a;
    int* b = realloc(p->widx, (size_t)c * sizeof(int));
    if (!b) {
      fx_fail(x, "MemoryError: out of memory");
      return;
    }
    p->widx = b;
    p->cap = c;
  }
  p->ids[p->n] = id;
  p->widx[p->n] = w;
  p->n++;
}

static void to_ph(fx* x, const st_front* f, const u32* s, int n, phbuf* r) {
  st_u32s text = {0};
  front_normalize(x, f, s, n, &text);
  if (x->err) goto done;
  if (u32_find(text.p, text.n, '[') >= 0) {
    st_spans parts = {0};
    st_rx_split(x, f->event, text.p, text.n, &parts);  /* text, name, text, name, ... */
    if (!x->err && parts.n > 1) {
      ph_push(x, r, ST_SYM_SP, -1);
      int nw = 0;
      for (int k = 0; k < parts.n && !x->err; ++k) {
        const u32* p = text.p + parts.p[k].a;
        int pn = parts.p[k].b - parts.p[k].a;
        if (k % 2) {
          char tok[64];
          int tl = 0;
          tok[tl++] = '[';
          for (int i = 0; i < pn && tl < 60; ++i) {
            u32 c = p[i];
            tok[tl++] = (char)(c >= 'A' && c <= 'Z' ? c + 32 : c);
          }
          tok[tl++] = ']';
          tok[tl] = 0;
          int id = pn < 58 ? st_front_symbol_id(f, tok) : -1;  /* longer names are never symbols */
          if (id >= 0) {
            ph_push(x, r, id, -1);
            ph_push(x, r, ST_SYM_SP, -1);
          }
        } else {
          int a = 0, b = pn;  /* p.strip(" []") */
#define STRIPCH(c) ((c) == ' ' || (c) == '[' || (c) == ']')
          while (a < b && STRIPCH(p[a])) ++a;
          while (b > a && STRIPCH(p[b - 1])) --b;
#undef STRIPCH
          if (a == b) continue;
          st_u32s q = {0};
          u32s_put(x, &q, p, pn);
          for (int i = 0; i < q.n; ++i)
            if (q.p[i] == '[' || q.p[i] == ']') q.p[i] = ' ';
          phbuf sub = {0};
          if (!x->err) to_ph(x, f, q.p, q.n, &sub);
          for (int i = 1; i < sub.n; ++i) ph_push(x, r, sub.ids[i], sub.widx[i] >= 0 ? sub.widx[i] + nw : -1);
          free(sub.ids);
          free(sub.widx);
          u32s_free(&q);
          int mx = -1;
          for (int i = 0; i < r->n; ++i)
            if (r->widx[i] > mx) mx = r->widx[i];
          nw = mx + 1;
        }
      }
      free(parts.p);
      goto done;
    }
    free(parts.p);
    if (x->err) goto done;
  }
  {
    int* raw = NULL;
    int nraw = 0;
    g2p_call(x, &f->g2p, text.p, text.n, &raw, &nraw);
    if (x->err) goto done;
    ph_push(x, r, ST_SYM_SP, -1);
    int wi = -1, ncur = 0;  /* phones of the current word in cur[0 .. ncur) */
    int* cur = malloc(((size_t)nraw + 1) * sizeof(int));
    if (!cur) fx_fail(x, "MemoryError: out of memory");
    for (int k = 0; k <= nraw && !x->err; ++k) {
      int tok = k < nraw ? raw[k] : TOK_SPACE;
      if (tok == TOK_SPACE) {
        if (ncur) {
          ++wi;
          for (int i = 0; i < ncur; ++i) ph_push(x, r, cur[i], wi);
          ncur = 0;
          ph_push(x, r, ST_SYM_SP, -1);
        }
        continue;
      }
      if (tok >= 2 && tok <= 8) {  /* _PUNCT */
        if (ncur) {
          ++wi;
          for (int i = 0; i < ncur; ++i) ph_push(x, r, cur[i], wi);
          ncur = 0;
        }
        ph_push(x, r, tok, -1);
      } else {
        cur[ncur++] = tok;
      }
    }
    free(cur);
    free(raw);
  }
done:
  u32s_free(&text);
}

int front_to_phonemes(fx* x, const st_front* f, const u32* s, int n, st_phonemes* out) {
  phbuf r = {0};
  to_ph(x, f, s, n, &r);
  if (x->err) {
    free(r.ids);
    free(r.widx);
    memset(out, 0, sizeof *out);
    return -1;
  }
  out->ids = r.ids;
  out->widx = r.widx;
  out->n = r.n;
  return 0;
}

/* ------------------------------------------------------------------ public API */
int st_front_phonemes(st_front* f, const char* utf8, st_phonemes* out) {
  fx x = {0};
  st_u32s s = {0};
  memset(out, 0, sizeof *out);
  if (!f || !utf8) {
    st_seterr("steertts front: NULL argument");
    return -1;
  }
  u32s_utf8(&x, &s, utf8, strlen(utf8));
  int rc = x.err ? -1 : front_to_phonemes(&x, f, s.p, s.n, out);
  u32s_free(&s);
  return rc;
}

void st_phonemes_free(st_phonemes* p) {
  if (!p) return;
  free(p->ids);
  free(p->widx);
  memset(p, 0, sizeof *p);
}

char* st_front_normalize(st_front* f, const char* utf8) {
  fx x = {0};
  if (!f || !utf8) {
    st_seterr("steertts front: NULL argument");
    return NULL;
  }
  st_u32s s = {0}, t = {0};
  st_sb o = {0};
  u32s_utf8(&x, &s, utf8, strlen(utf8));
  front_normalize(&x, f, s.p, s.n, &t);
  sb_utf8(&x, &o, t.p, t.n);
  char* r = x.err ? NULL : st_strdup(sb_cstr(&o));
  u32s_free(&s);
  u32s_free(&t);
  sb_free(&o);
  return r;
}

int st_front_has_speech(const st_phonemes* p) {
  for (int i = 0; i < p->n; ++i)
    if (p->ids[i] >= ST_FIRST_PHONE) return 1;
  return 0;
}
