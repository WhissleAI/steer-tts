/* Markup (steertts_edge markup.MarkupSynth.parse), streamed one segment at a time, no heap.
 *
 * Pipeline of the reference, kept in order: *x* -> <emph>x</emph>; the token expression
 *   (<\s*(/?)\s*([A-Za-z_]+)(?:\s+prompt\s*=\s*"([^"]*)")?\s*>)|(\[\s*([A-Za-z_]+)(?:\s+([\d.]+)\s*(ms|s)?)?\s*\])
 * (IGNORECASE) cuts the text into runs with a style / prosody; runs are split at sentence ends; consecutive pieces
 * with the same style form a segment. Here the scanner works on the caller's UTF-8 text directly and emits each
 * segment as soon as its end is known, so memory does not grow with the text length. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "nf_int.h"

static const struct {
  const char* name;
  int has_pitch, has_energy, has_dur;
  double pitch, energy, dur;   /* double: the factors multiply as Python floats */
} PROS[] = {{"emph", 1, 1, 1, 2.5, 4.0, 1.2}, {"slow", 0, 0, 1, 0, 0, 1.4}, {"fast", 0, 0, 1, 0, 0, 0.75},
            {"high", 1, 0, 0, 3.0, 0, 0},     {"low", 1, 0, 0, -3.0, 0, 0}, {"loud", 0, 1, 0, 0, 6.0, 0},
            {"soft", 0, 1, 0, 0, -6.0, 0}};
#define N_PROS 7

enum { R_NONE, R_TEXT, R_PAUSE, R_EVENT };

/* ------------------------------------------------------------------ UTF-8 helpers on the source */
static u32 cp_at(const nf_markup* m, int p, int* next) {
  const unsigned char* s = (const unsigned char*)m->src;
  if (p >= m->len) {
    *next = p;
    return 0;
  }
  unsigned c = s[p];
  int n = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
  if (p + n > m->len) n = 1;
  for (int k = 1; k < n; k++)
    if ((s[p + k] & 0xC0) != 0x80) n = 1;
  u32 v = n == 1 ? (c < 0x80 ? c : 0xFFFD) : n == 2 ? ((c & 0x1Fu) << 6) | (s[p + 1] & 0x3Fu)
          : n == 3 ? ((c & 0x0Fu) << 12) | ((s[p + 1] & 0x3Fu) << 6) | (s[p + 2] & 0x3Fu)
                   : ((c & 0x07u) << 18) | ((s[p + 1] & 0x3Fu) << 12) | ((s[p + 2] & 0x3Fu) << 6) | (s[p + 3] & 0x3Fu);
  *next = p + n;
  return v;
}
static int prev_pos(const char* s, int p) {   /* start of the code point before byte p */
  int q = p - 1;
  while (q > 0 && (((unsigned char)s[q]) & 0xC0) == 0x80) q--;
  return q;
}
static int sp_at(nano_front_ctx* c, int p, int* next) {
  int n;
  u32 ch = cp_at(&c->mk, p, &n);
  if (p < c->mk.len && nf_is_space(c, ch)) {
    *next = n;
    return 1;
  }
  return 0;
}
static int skip_sp(nano_front_ctx* c, int p) {
  int n;
  while (sp_at(c, p, &n)) p = n;
  return p;
}
/* [A-Za-z_] under IGNORECASE: ASCII letters, '_' and the four code points that fold to ASCII letters */
static int name_ch(u32 ch) {
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_' || ch == 0x130 || ch == 0x131 ||
         ch == 0x17F || ch == 0x212A;
}
static int name_end(nano_front_ctx* c, int p) {
  int n;
  while (p < c->mk.len && name_ch(cp_at(&c->mk, p, &n))) p = n;
  return p;
}
static void lower_name(nano_front_ctx* c, int a, int b, char* out, int cap) {
  int o = 0, n;
  for (int p = a; p < b; p = n) {
    u32 ch = cp_at(&c->mk, p, &n), lo[4];
    int k = nf_lower(c, ch, lo);
    char tmp[16];
    int L = nf_utf8_encode(lo, k, tmp, (int)sizeof tmp);
    if (L < 0 || o + L >= cap) break;
    memcpy(out + o, tmp, (size_t)L);
    o += L;
  }
  out[o] = 0;
}
static int ci_lit(nano_front_ctx* c, int p, const char* w, int* end) {   /* IGNORECASE literal */
  int n;
  for (; *w; w++) {
    u32 ch = cp_at(&c->mk, p, &n);
    if (p >= c->mk.len || nf_fold(c, ch) != (u32)(unsigned char)*w) return 0;
    p = n;
  }
  *end = p;
  return 1;
}

/* ------------------------------------------------------------------ styles */
static int known(const nf_markup* m, const char* nm) {
  for (int k = 0; k < m->ntags; k++)
    if (m->tags[k] && !strcmp(m->tags[k], nm)) return 1;
  return 0;
}
static int style_eq(const nf_style* a, const nf_style* b) {
  if (strcmp(a->tag, b->tag) || a->n != b->n) return 0;
  for (int k = 0; k < a->n; k++)
    if (strcmp(a->name[k], b->name[k]) || a->val[k] != b->val[k]) return 0;
  return 1;
}
static int add_style(nano_front_ctx* c, const char* tag, const char* const* names, const float* vals, int n) {
  nf_markup* m = &c->mk;
  nf_style s;
  memset(&s, 0, sizeof s);
  snprintf(s.tag, sizeof s.tag, "%s", tag);
  for (int k = 0; k < n; k++) {   /* dict: later duplicates win */
    const char* nm = names[k] ? names[k] : "";
    int j = 0;
    while (j < s.n && strcmp(s.name[j], nm)) j++;
    if (j < s.n) {
      s.val[j] = vals[k];
      continue;
    }
    if (s.n >= NF_MAX_CTRL) break;
    snprintf(s.name[s.n], sizeof s.name[0], "%s", nm);
    s.val[s.n++] = vals[k];
  }
  for (int k = 1; k < s.n; k++) {   /* sorted by name */
    char nm[16];
    float v = s.val[k];
    memcpy(nm, s.name[k], sizeof nm);
    int j = k;
    while (j > 0 && strcmp(s.name[j - 1], nm) > 0) {
      memcpy(s.name[j], s.name[j - 1], sizeof nm);
      s.val[j] = s.val[j - 1];
      j--;
    }
    memcpy(s.name[j], nm, sizeof nm);
    s.val[j] = v;
  }
  for (int k = 0; k < m->nst; k++)
    if (style_eq(&m->st[k], &s)) return k;
  if (m->nst >= NF_MAX_STYLES) {
    nf_fail(c, "ValueError: too many styles for the device front end");
    return 0;
  }
  m->st[m->nst] = s;
  return m->nst++;
}

/* ------------------------------------------------------------------ token scanner */
/* emph: \*([^*\n]+)\* starting at p -> closing '*' position or -1 */
static int emph_from(const nf_markup* m, int p) {
  if (m->src[p] != '*') return -1;
  int q = p + 1;
  while (q < m->len && m->src[q] != '*' && m->src[q] != '\n') q++;
  if (q >= m->len || m->src[q] != '*' || q == p + 1) return -1;
  return q;
}

typedef struct {
  int a, b;        /* token span */
  int kind;        /* 0 tag, 1 bracket, 2 emph open, 3 emph close */
  int closing, na, nb, pa, pb, va, vb, ua, ub;   /* groups: name, prompt, value, unit (-1: absent) */
} token;

static int tag_at(nano_front_ctx* c, int p, token* t) {
  int n;
  u32 ch = cp_at(&c->mk, p, &n);
  if (p >= c->mk.len) return 0;
  memset(t, 0, sizeof *t);
  t->pa = t->va = t->ua = -1;
  if (ch == '<') {
    int k = skip_sp(c, n);
    if (k < c->mk.len && c->mk.src[k] == '/') {
      t->closing = 1;
      k++;
    }
    k = skip_sp(c, k);
    int a = k, b = name_end(c, k);
    if (b == a) return 0;
    t->na = a;
    t->nb = b;
    int q = b, e;
    /* (?:\s+prompt\s*=\s*"([^"]*)")? */
    int r = skip_sp(c, q);
    if (r > q && ci_lit(c, r, "prompt", &e)) {
      e = skip_sp(c, e);
      if (e < c->mk.len && c->mk.src[e] == '=') {
        e = skip_sp(c, e + 1);
        if (e < c->mk.len && c->mk.src[e] == '"') {
          int z = e + 1;
          while (z < c->mk.len && c->mk.src[z] != '"') z++;
          if (z < c->mk.len) {
            int z2 = skip_sp(c, z + 1);
            if (z2 < c->mk.len && c->mk.src[z2] == '>') {
              t->pa = e + 1;
              t->pb = z;
              t->a = p;
              t->b = z2 + 1;
              t->kind = 0;
              return 1;
            }
          }
        }
      }
    }
    q = skip_sp(c, q);
    if (q < c->mk.len && c->mk.src[q] == '>') {
      t->a = p;
      t->b = q + 1;
      t->kind = 0;
      return 1;
    }
    return 0;
  }
  if (ch == '[') {
    int k = skip_sp(c, n);
    int a = k, b = name_end(c, k);
    if (b == a) return 0;
    t->na = a;
    t->nb = b;
    t->kind = 1;
    /* (?:\s+([\d.]+)\s*(ms|s)?)?\s*\] */
    int r = skip_sp(c, b);
    if (r > b) {
      int v = r, nn;
      while (v < c->mk.len) {
        u32 x = cp_at(&c->mk, v, &nn);
        if (x == '.' || nf_digit(c, x) >= 0)
          v = nn;
        else
          break;
      }
      if (v > r) {
        int s2 = skip_sp(c, v), e;
        int ends[3][3], ne = 0;   /* unit start, unit end, end */
        if (ci_lit(c, s2, "ms", &e)) {
          int z = skip_sp(c, e);
          if (z < c->mk.len && c->mk.src[z] == ']') {
            ends[ne][0] = s2, ends[ne][1] = e, ends[ne][2] = z + 1;
            ne++;
          }
        }
        if (ci_lit(c, s2, "s", &e)) {
          int z = skip_sp(c, e);
          if (z < c->mk.len && c->mk.src[z] == ']') {
            ends[ne][0] = s2, ends[ne][1] = e, ends[ne][2] = z + 1;
            ne++;
          }
        }
        if (s2 < c->mk.len && c->mk.src[s2] == ']') {
          ends[ne][0] = -1, ends[ne][1] = -1, ends[ne][2] = s2 + 1;
          ne++;
        }
        if (ne) {
          t->va = r;
          t->vb = v;
          t->ua = ends[0][0];
          t->ub = ends[0][1];
          t->a = p;
          t->b = ends[0][2];
          return 1;
        }
      }
    }
    int z = skip_sp(c, b);
    if (z < c->mk.len && c->mk.src[z] == ']') {
      t->a = p;
      t->b = z + 1;
      return 1;
    }
    return 0;
  }
  return 0;
}

/* next token at or after pos: 1 with *t, 0 if none */
static int next_token(nano_front_ctx* c, token* t) {
  nf_markup* m = &c->mk;
  for (int p = m->pos; p < m->len; p++) {
    if (((unsigned char)m->src[p] & 0xC0) == 0x80) continue;
    if (p == m->emph_close) {
      memset(t, 0, sizeof *t);
      t->a = p;
      t->b = p + 1;
      t->kind = 3;
      return 1;
    }
    if (m->src[p] == '*' && m->emph_close < p) {
      int q = emph_from(m, p);
      if (q > 0) {
        memset(t, 0, sizeof *t);
        t->a = p;
        t->b = p + 1;
        t->kind = 2;
        t->na = q;   /* its closing '*' (armed when the token is consumed) */
        return 1;
      }
    }
    if ((m->src[p] == '<' || m->src[p] == '[') && tag_at(c, p, t)) return 1;
  }
  return 0;
}

/* ------------------------------------------------------------------ runs -> pieces -> segments */
static unsigned pros_now(const nf_markup* m) {
  unsigned r = 0;
  for (int k = 0; k < m->nps; k++) r |= 1u << m->pstack[k];
  return r;
}

/* produce the next run into m->run_*; 0 when the text is exhausted */
static int next_run(nano_front_ctx* c) {
  nf_markup* m = &c->mk;
  for (;;) {
    if (m->pos >= m->len) return 0;
    token t;
    int have = next_token(c, &t);
    int end = have ? t.a : m->len;
    if (end > m->pos) {   /* the text before the token */
      m->run_kind = R_TEXT;
      m->run_a = m->pos;
      m->run_b = end;
      m->run_syn = 0;
      m->run_style = m->sstack[m->nss - 1];
      m->run_pros = pros_now(m);
      m->run_split = m->run_a;
      m->pos = end;
      return 1;
    }
    m->pos = t.b;
    if (t.kind == 2 || t.kind == 3) {   /* <emph> / </emph> */
      if (t.kind == 2) {
        m->emph_close = t.na;
        if (m->nps < 32) m->pstack[m->nps++] = 0;
      } else {
        for (int k = 0; k < m->nps; k++)
          if (m->pstack[k] == 0) {
            memmove(m->pstack + k, m->pstack + k + 1, (size_t)(m->nps - k - 1) * sizeof(int));
            m->nps--;
            break;
          }
        m->emph_close = -1;
      }
      continue;
    }
    char nm[48];
    lower_name(c, t.na, t.nb, nm, (int)sizeof nm);
    if (t.kind == 1) {   /* [pause ...] / [event] */
      if (!strcmp(nm, "pause")) {
        int ms = 400;
        if (t.va >= 0) {
          char a[64];
          int an = 0, dots = 0, digits = 0, nn;
          for (int p = t.va; p < t.vb && an < 60; p = nn) {
            u32 x = cp_at(m, p, &nn);
            int d = nf_digit(c, x);
            if (d >= 0) {
              a[an++] = (char)('0' + d);
              digits++;
            } else {
              a[an++] = '.';
              dots++;
            }
          }
          a[an] = 0;
          int is_s = 0;
          if (t.ua >= 0) {
            char un[8];
            lower_name(c, t.ua, t.ub, un, (int)sizeof un);
            is_s = !strcmp(un, "s");
          }
          if (dots > 1 || digits == 0) {
            char raw[64];
            int L = t.vb - t.va < 60 ? t.vb - t.va : 60;
            memcpy(raw, m->src + t.va, (size_t)L);
            raw[L] = 0;
            nf_fail(c, "ValueError: could not convert string to float: '%s'", raw);
            return 0;
          }
          double val = nf_strtod(a) * (is_s ? 1000.0 : 1.0);
          if (isinf(val)) {
            nf_fail(c, "OverflowError: cannot convert float infinity to integer");
            return 0;
          }
          double tr = trunc(val);
          ms = tr > 5000 ? 5000 : (int)tr;
        }
        m->run_kind = R_PAUSE;
        m->run_ms = ms < 5000 ? ms : 5000;
        return 1;
      }
      if (m->is_event && m->is_event(m->ev_user, nm)) {
        m->run_kind = R_EVENT;
        snprintf(m->run_buf, sizeof m->run_buf, "%s", nm);
        return 1;
      }
      /* an event the bank lacks (or unknown) stays in the TTS text */
      m->run_kind = R_TEXT;
      snprintf(m->run_buf, sizeof m->run_buf, " [%s] ", nm);
      m->run_syn = 1;
      m->run_a = 0;
      m->run_b = (int)strlen(m->run_buf);
      m->run_split = 0;
      m->run_style = m->sstack[m->nss - 1];
      m->run_pros = pros_now(m);
      return 1;
    }
    int pi = -1;
    for (int k = 0; k < N_PROS; k++)
      if (!strcmp(nm, PROS[k].name)) pi = k;
    if (pi >= 0) {
      if (t.closing) {
        for (int k = 0; k < m->nps; k++)
          if (m->pstack[k] == pi) {
            memmove(m->pstack + k, m->pstack + k + 1, (size_t)(m->nps - k - 1) * sizeof(int));
            m->nps--;
            break;
          }
      } else if (m->nps < 32) {
        m->pstack[m->nps++] = pi;
      }
    } else if (!strcmp(nm, "style") || known(m, nm)) {
      if (t.closing) {
        if (m->nss > 1) m->nss--;
      } else if (!strcmp(nm, "style")) {
        if (t.pa >= 0 && t.pb > t.pa) {
          nf_fail(c, "ValueError: free-text prompts are resolved on the host (the prompt adapter is not on device): "
                     "send the resolved tag + ctrl");
          return 0;
        }
        int idx = add_style(c, "none", NULL, NULL, 0);
        if (m->nss < 32) m->sstack[m->nss++] = idx;
      } else {
        int idx = add_style(c, nm, NULL, NULL, 0);   /* a named style tag resets the default controls */
        if (m->nss < 32) m->sstack[m->nss++] = idx;
      }
    }
    /* unknown tag: dropped, its inner text is spoken normally */
  }
}

static const char* run_text(const nf_markup* m) { return m->run_syn ? m->run_buf : m->src; }
static int is_lit_range(const char* s, int a, int b, const char* w) {
  int n = (int)strlen(w);
  return b - a == n && !memcmp(s + a, w, (size_t)n);
}

/* sentence end inside the current run at byte p (start of a space run): match end or -1.
 * (?<=[.!?])(?<!\bDr\.)...(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\b[ap]\.m\.)(?<!\b[A-Z]\.)\s+ */
static int word_before(nano_front_ctx* c, const char* s, int lo, int p) {   /* \b before byte p (inside [lo, ..)) */
  if (p <= lo) return 1;   /* start of the run: previous is "nothing" -> non-word */
  int q = prev_pos(s, p), n;
  nf_markup tmp = c->mk;
  tmp.src = s;
  tmp.len = p;
  u32 ch = cp_at(&tmp, q, &n);
  return !nf_is_word(c, ch);
}
static int ends_with(nano_front_ctx* c, const char* s, int lo, int p, const char* w) {
  int n = (int)strlen(w);
  if (p - lo < n || memcmp(s + p - n, w, (size_t)n)) return 0;
  return word_before(c, s, lo, p - n);   /* \b before the word: the char before it is not a word char */
}
static int sent_end_at(nano_front_ctx* c, int lo, int hi, int p) {
  const char* s = run_text(&c->mk);
  if (p <= lo) return -1;
  char pc = s[p - 1];
  if (!(pc == '.' || pc == '!' || pc == '?')) return -1;
  static const char* AB[] = {"Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "St.", "Jr.", "Sr.", "vs.", "No.", "Fig.",
                             "Vol.", "pp.", "approx.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.",
                             "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "e.g.", "i.e.", "a.m.", "p.m."};
  for (int k = 0; k < (int)(sizeof AB / sizeof AB[0]); k++)
    if (ends_with(c, s, lo, p, AB[k])) return -1;
  if (p - lo >= 2 && s[p - 1] == '.' && s[p - 2] >= 'A' && s[p - 2] <= 'Z' && word_before(c, s, lo, p - 2)) return -1;
  /* \s+ */
  nf_markup tmp = c->mk;
  tmp.src = s;
  tmp.len = hi;
  int q = p, n;
  while (q < hi) {
    u32 ch = cp_at(&tmp, q, &n);
    if (!nf_is_space(c, ch)) break;
    q = n;
  }
  return q > p ? q : -1;
}

/* ------------------------------------------------------------------ segment building */
static int seg_append(nano_front_ctx* c, nf_segment* seg, const char* s, int a, int b, unsigned pros) {
  nf_markup* m = &c->mk;
  int before = 0;
  if (pros) {
    before = nf_count_words(c, seg->text, (size_t)m->cur_n);
    if (before < 0) return -1;
  }
  if (m->cur_n + (b - a) >= (int)sizeof seg->text) {
    nf_fail(c, "ValueError: sentence too long for the device front end");
    return -1;
  }
  memcpy(seg->text + m->cur_n, s + a, (size_t)(b - a));
  m->cur_n += b - a;
  seg->text[m->cur_n] = 0;
  for (int k = a; k < b; k++)
    if (!(s[k] == ' ' || (s[k] >= 9 && s[k] <= 13))) m->cur_nonspace = 1;   /* refined at flush */
  if (!pros) return 0;
  int after = nf_count_words(c, seg->text, (size_t)m->cur_n);
  if (after < 0) return -1;
  float pitch = 0, energy = 0, dur = 1.0f;
  double dp = 0, de = 0, dd = 1.0;
  for (int k = 0; k < N_PROS; k++) {
    if (!(pros & (1u << k))) continue;
    if (PROS[k].has_pitch) dp += PROS[k].pitch;
    if (PROS[k].has_energy) de += PROS[k].energy;
    if (PROS[k].has_dur) dd *= PROS[k].dur;
  }
  pitch = (float)dp;
  energy = (float)de;
  dur = (float)dd;
  for (int w = before; w < after; w++) {
    int j = 0;
    while (j < seg->n_ws && seg->ws[j].word != w) j++;
    if (j == seg->n_ws) {
      if (seg->n_ws >= NF_MAX_WS) {
        nf_fail(c, "ValueError: too many emphasised words for the device front end");
        return -1;
      }
      seg->n_ws++;
    }
    seg->ws[j].word = w;
    seg->ws[j].pitch = pitch;
    seg->ws[j].energy = energy;
    seg->ws[j].dur = dur;
  }
  return 0;
}

static void seg_start(nano_front_ctx* c, nf_segment* seg, int style) {
  nf_markup* m = &c->mk;
  memset(seg, 0, offsetof(nf_segment, text) + 1);
  seg->text[0] = 0;
  seg->n_ws = 0;
  seg->pause_ms = seg->skip = 0;
  seg->event[0] = 0;
  const nf_style* s = &m->st[style];
  snprintf(seg->tag, sizeof seg->tag, "%s", s->tag);
  seg->n_ctrl = s->n;
  for (int k = 0; k < s->n; k++) {
    memcpy(seg->ctrl_name[k], s->name[k], sizeof seg->ctrl_name[k]);
    seg->ctrl_val[k] = s->val[k];
  }
  m->cur_n = 0;
  m->cur_style = style;
  m->cur_nonspace = 0;
}

/* finish the segment: collapse whitespace and strip; 1 if it has text */
static int seg_finish(nano_front_ctx* c, nf_segment* seg) {
  nf_markup* m = &c->mk;
  nfs X = {c->a, 0, NF_MAX_CP, c};
  if (nf_utf8_decode(seg->text, (size_t)m->cur_n, &X)) return -1;
  int any = 0;
  for (int i = 0; i < X.n && !any; i++) any = !nf_is_space(c, X.p[i]);
  nf_collapse_strip(c, &X);
  if (nf_utf8_encode(X.p, X.n, seg->text, (int)sizeof seg->text) < 0) {
    nf_fail(c, "ValueError: sentence too long for the device front end");
    return -1;
  }
  /* sort word shifts by word */
  for (int k = 1; k < seg->n_ws; k++) {
    nf_word_shift w = seg->ws[k];
    int j = k;
    while (j > 0 && seg->ws[j - 1].word > w.word) {
      seg->ws[j] = seg->ws[j - 1];
      j--;
    }
    seg->ws[j] = w;
  }
  return any;
}

int nano_front_markup_begin(nano_front_ctx* c, const char* text, const char* default_tag, const char* const* ctrl_names,
                            const float* ctrl_vals, int n_ctrl, const char* const* known_tags, int n_tags,
                            nf_event_fn is_event, void* event_user) {
  nf_markup* m = &c->mk;
  c->fail = 0;
  c->err[0] = 0;
  memset(m, 0, sizeof *m);
  m->src = text;
  m->len = (int)strlen(text);
  m->emph_close = -1;
  m->tags = known_tags;
  m->ntags = n_tags;
  m->is_event = is_event;
  m->ev_user = event_user;
  const char* d_tag = "none";
  if (default_tag && *default_tag && known(m, default_tag)) d_tag = default_tag;
  m->sstack[m->nss++] = add_style(c, d_tag, ctrl_names, ctrl_vals, n_ctrl > 0 ? n_ctrl : 0);
  m->run_kind = R_NONE;
  m->cur_style = -1;
  return c->fail ? -1 : 0;
}

/* next piece: 1 TEXT (range in run_text), 2 BREAK, 3 PAUSE, 4 EVENT, 5 literal PAUSE/EVENT text; 0 at the end */
static int next_piece(nano_front_ctx* c, int* a, int* b) {
  nf_markup* m = &c->mk;
  for (;;) {
    if (m->run_kind == R_NONE) {
      if (!next_run(c)) return c->fail ? -1 : 0;
      if (m->run_kind == R_PAUSE) {
        m->run_kind = R_NONE;
        return 3;
      }
      if (m->run_kind == R_EVENT) {
        m->run_kind = R_NONE;
        return 4;
      }
      const char* s = run_text(m);
      if (is_lit_range(s, m->run_a, m->run_b, "PAUSE") || is_lit_range(s, m->run_a, m->run_b, "EVENT")) {
        m->run_kind = R_NONE;
        *a = m->run_a;
        *b = m->run_b;
        return 5;
      }
    }
    if (m->run_split < 0) {   /* a BREAK is due between two parts */
      m->run_split = -m->run_split - 1;
      return 2;
    }
    const char* s = run_text(m);
    int lo = m->run_split, hi = m->run_b, p = lo, e = -1;
    for (; p < hi; p++) {
      if (((unsigned char)s[p] & 0xC0) == 0x80) continue;
      if ((e = sent_end_at(c, m->run_a, hi, p)) > 0) break;
    }
    *a = lo;
    if (p < hi && e > 0) {
      *b = p;
      m->run_split = -e - 1;   /* next call: BREAK, then continue at e */
    } else {
      *b = hi;
      m->run_kind = R_NONE;
    }
    return 1;
  }
}

int nano_front_markup_next(nano_front_ctx* c, nf_segment* seg) {
  nf_markup* m = &c->mk;
  if (c->fail) return -1;
  for (;;) {
    if (m->queued) {   /* a pause / event segment behind the one just returned */
      m->queued = 0;
      memset(seg, 0, offsetof(nf_segment, text) + 1);
      seg->text[0] = 0;
      snprintf(seg->tag, sizeof seg->tag, "none");
      seg->n_ctrl = seg->n_ws = 0;
      seg->pause_ms = m->q_ms;
      seg->skip = m->q_skip;
      snprintf(seg->event, sizeof seg->event, "%s", m->q_event);
      return 1;
    }
    if (m->pend) {   /* the piece that ended the previous segment by a style change starts this one */
      m->pend = 0;
      seg_start(c, seg, m->pend_style);
      if (seg_append(c, seg, run_text(m), m->pend_a, m->pend_b, m->pend_pros)) return -1;
      continue;
    }
    if (m->done) return 0;
    int a = 0, b = 0;
    int k = next_piece(c, &a, &b);
    if (k < 0) return -1;
    const char* s = run_text(m);
    int txt_lit = k == 1 && (is_lit_range(s, a, b, "PAUSE") || is_lit_range(s, a, b, "EVENT") ||
                             is_lit_range(s, a, b, "BREAK"));
    int boundary = k != 1 || txt_lit;
    int restyle = !boundary && m->cur_style >= 0 && !style_eq(&m->st[m->run_style], &m->st[m->cur_style]);
    int emitted = 0;
    if ((boundary || restyle) && m->cur_style >= 0) {
      int any = seg_finish(c, seg);
      if (any < 0) return -1;
      m->cur_style = -1;
      emitted = any;
    }
    if (k == 0) {
      m->done = 1;
      if (emitted) return 1;
      return 0;
    }
    if (boundary) {
      int special = k == 3 || k == 4 || k == 5 || (k == 1 && (is_lit_range(s, a, b, "PAUSE") || is_lit_range(s, a, b, "EVENT")));
      if (special) {
        m->q_ms = k == 3 ? m->run_ms : 0;
        m->q_skip = 0;
        m->q_event[0] = 0;
        if (k == 4) snprintf(m->q_event, sizeof m->q_event, "%s", m->run_buf);
        if (k == 5 || k == 1) {   /* Python quirk: the run's style tuple lands in pause_ms / event */
          m->q_skip = 1;
          m->q_ms = is_lit_range(s, a, b, "PAUSE") ? -1 : 0;
        }
        m->queued = 1;
      }
      if (emitted) return 1;
      continue;
    }
    if (emitted) {   /* style change: return the finished segment, this piece starts the next one */
      m->pend = 1;
      m->pend_a = a;
      m->pend_b = b;
      m->pend_style = m->run_style;
      m->pend_pros = m->run_pros;
      return 1;
    }
    if (m->cur_style < 0) seg_start(c, seg, m->run_style);
    if (seg_append(c, seg, s, a, b, m->run_pros)) return -1;
  }
}
