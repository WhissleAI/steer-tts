/* Port of markup.py MarkupSynth.parse / _segment / _n_words: the edge build (prompt_fn == NULL: <style prompt> raises)
 * and the v1 build (prompt_fn resolves the prompt like _style_of_prompt). */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "st_front_int.h"

/* markup.PROSODY in its dict order (bit i of a piece's pros set = PROS[i]) */
static const struct {
  const char* name;
  int has_pitch, has_energy, has_dur;
  double pitch, energy, dur;
} PROS[] = {{"emph", 1, 1, 1, 2.5, 4.0, 1.2}, {"slow", 0, 0, 1, 0, 0, 1.4}, {"fast", 0, 0, 1, 0, 0, 0.75},
            {"high", 1, 0, 0, 3.0, 0, 0},     {"low", 1, 0, 0, -3.0, 0, 0}, {"loud", 0, 1, 0, 0, 6.0, 0},
            {"soft", 0, 1, 0, 0, -6.0, 0}};
#define N_PROS 7

typedef struct {
  char* tag;
  int n;
  char** name;   /* sorted */
  float* val;
} mstyle;

enum { P_TEXT, P_PAUSE, P_BREAK, P_EVENT };
typedef struct {
  int kind;
  st_u32s text;
  int style;     /* index into the style table */
  unsigned pros;
  int ms;
} piece;

typedef struct {
  fx* x;
  const st_front* f;
  mstyle* st;
  int nst, capst;
  piece* pc;
  int npc, cappc;
  /* _n_words one-entry memo */
  st_u32s memo;
  int memo_n;
} mk;

static void style_free(mstyle* s) {
  free(s->tag);
  for (int k = 0; k < s->n; ++k) free(s->name[k]);
  free(s->name);
  free(s->val);
}

static int style_eq(const mstyle* a, const mstyle* b) {
  if (strcmp(a->tag, b->tag) || a->n != b->n) return 0;
  for (int k = 0; k < a->n; ++k)
    if (strcmp(a->name[k], b->name[k]) || a->val[k] != b->val[k]) return 0;
  return 1;
}

static char* dup(fx* x, const char* s) {
  size_t n = strlen(s);
  char* d = malloc(n + 1);
  if (!d)
    fx_fail(x, "MemoryError: out of memory");
  else
    memcpy(d, s, n + 1);
  return d;
}

/* add a style from (tag, ctrl pairs with dict semantics: later duplicates win), ctrl sorted by name; returns index */
static int add_style(mk* M, const char* tag, const char* const* names, const float* vals, int n) {
  fx* x = M->x;
  if (M->nst == M->capst) {
    int c = M->capst ? 2 * M->capst : 8;
    mstyle* v = realloc(M->st, (size_t)c * sizeof *v);
    if (!v) {
      fx_fail(x, "MemoryError: out of memory");
      return 0;
    }
    M->st = v;
    M->capst = c;
  }
  mstyle* s = &M->st[M->nst++];
  memset(s, 0, sizeof *s);
  s->tag = dup(x, tag);
  if (n > 0) {
    s->name = calloc((size_t)n, sizeof *s->name);
    s->val = calloc((size_t)n, sizeof *s->val);
    if (!s->name || !s->val) {
      fx_fail(x, "MemoryError: out of memory");
      return 0;
    }
  }
  for (int k = 0; k < n && !x->err; ++k) {
    const char* nm = names[k] ? names[k] : "";
    int j = 0;
    while (j < s->n && strcmp(s->name[j], nm)) ++j;
    if (j < s->n) {
      s->val[j] = vals[k];
      continue;
    }
    s->name[s->n] = dup(x, nm);
    s->val[s->n] = vals[k];
    s->n++;
  }
  for (int k = 1; k < s->n; ++k) {  /* insertion sort by name */
    char* nm = s->name[k];
    float v = s->val[k];
    int j = k;
    while (j > 0 && strcmp(s->name[j - 1], nm) > 0) {
      s->name[j] = s->name[j - 1];
      s->val[j] = s->val[j - 1];
      --j;
    }
    s->name[j] = nm;
    s->val[j] = v;
  }
  return M->nst - 1;
}

static piece* add_piece(mk* M, int kind) {
  if (M->npc == M->cappc) {
    int c = M->cappc ? 2 * M->cappc : 16;
    piece* v = realloc(M->pc, (size_t)c * sizeof *v);
    if (!v) {
      fx_fail(M->x, "MemoryError: out of memory");
      return NULL;
    }
    M->pc = v;
    M->cappc = c;
  }
  piece* p = &M->pc[M->npc++];
  memset(p, 0, sizeof *p);
  p->kind = kind;
  return p;
}

static int is_event_name(fx* x, st_event_fn fn, void* user, const st_u32s* name) {
  st_sb nm = {0};
  sb_utf8(x, &nm, name->p, name->n);
  int r = !x->err && fn(user, sb_cstr(&nm));
  sb_free(&nm);
  return r;
}

static int is_lit(const u32* s, int n, const char* lit) {
  int k = 0;
  for (; lit[k]; ++k)
    if (k >= n || s[k] != (u32)(unsigned char)lit[k]) return 0;
  return k == n;
}

static int n_words(mk* M, const u32* t, int n) {
  if (M->memo_n >= 0 && u32s_eq(t, n, M->memo.p, M->memo.n)) return M->memo_n;
  int a, b, r = 0;
  front_strip(M->f, t, n, &a, &b);
  if (a < b) {
    st_phonemes p;
    if (front_to_phonemes(M->x, M->f, t, n, &p)) return 0;
    int mx = -1;
    for (int i = 0; i < p.n; ++i)
      if (p.widx[i] > mx) mx = p.widx[i];
    r = mx >= 0 ? mx + 1 : 0;
    st_phonemes_free(&p);
  }
  M->memo.n = 0;
  u32s_put(M->x, &M->memo, t, n);
  M->memo_n = r;
  return r;
}

static char* utf8_dup(fx* x, const u32* s, int n) {
  st_sb b = {0};
  sb_utf8(x, &b, s, n);
  char* r = x->err ? NULL : st_strdup(sb_cstr(&b));
  if (!r) fx_fail(x, "MemoryError: out of memory");
  sb_free(&b);
  return r;
}

static int seg_style(fx* x, st_segment* seg, const mstyle* s) {
  seg->tag = st_strdup(s->tag);
  if (!seg->tag) return -1;
  if (s->n) {
    seg->ctrl_name = st_calloc((size_t)s->n * sizeof(char*));
    seg->ctrl_val = st_malloc((size_t)s->n * sizeof(float));
    if (!seg->ctrl_name || !seg->ctrl_val) return -1;
    for (int k = 0; k < s->n; ++k) {
      seg->ctrl_name[k] = st_strdup(s->name[k]);
      if (!seg->ctrl_name[k]) return -1;
      seg->ctrl_val[k] = s->val[k];
      seg->n_ctrl = k + 1;
    }
  }
  return 0;
}

/* _segment: pieces [p0, p1) share a style */
static void make_segment(mk* M, int p0, int p1, st_segment* seg) {
  fx* x = M->x;
  st_u32s t = {0}, c = {0};
  st_word_shift* ws = NULL;   /* dense by word, ws[w].word < 0: unset */
  int nws = 0;
  for (int i = p0; i < p1 && !x->err; ++i) {
    const piece* p = &M->pc[i];
    int before = n_words(M, t.p, t.n);
    u32s_put(x, &t, p->text.p, p->text.n);
    if (!p->pros || x->err) continue;
    int after = n_words(M, t.p, t.n);
    double pitch = 0, energy = 0, dur = 1.0;
    for (int k = 0; k < N_PROS; ++k) {
      if (!(p->pros & (1u << k))) continue;
      if (PROS[k].has_pitch) pitch += PROS[k].pitch;
      if (PROS[k].has_energy) energy += PROS[k].energy;
      if (PROS[k].has_dur) dur *= PROS[k].dur;
    }
    if (after > nws) {
      st_word_shift* v = realloc(ws, (size_t)after * sizeof *v);
      if (!v) {
        fx_fail(x, "MemoryError: out of memory");
        break;
      }
      ws = v;
      for (int w = nws; w < after; ++w) ws[w].word = -1;
      nws = after;
    }
    for (int w = before; w < after; ++w) {
      ws[w].word = w;
      ws[w].pitch = (float)pitch;
      ws[w].energy = (float)energy;
      ws[w].dur = (float)dur;
    }
  }
  if (!x->err) {
    front_collapse_ws(x, M->f, t.p, t.n, &c);
    int a, b;
    front_strip(M->f, c.p, c.n, &a, &b);
    seg->text = utf8_dup(x, c.p + a, b - a);
  }
  if (!x->err && seg_style(x, seg, &M->st[M->pc[p0].style])) fx_fail(x, "MemoryError: out of memory");
  int cnt = 0;
  for (int w = 0; w < nws; ++w) cnt += ws[w].word >= 0;
  if (!x->err && cnt) {
    seg->ws = st_malloc((size_t)cnt * sizeof *seg->ws);
    if (!seg->ws) {
      fx_fail(x, "MemoryError: out of memory");
    } else {
      for (int w = 0; w < nws; ++w)
        if (ws[w].word >= 0) seg->ws[seg->n_ws++] = ws[w];
    }
  }
  free(ws);
  u32s_free(&t);
  u32s_free(&c);
}

void st_segments_free(st_segment* s, int n) {
  if (!s) return;
  for (int i = 0; i < n; ++i) {
    st_free(s[i].text);
    st_free(s[i].tag);
    for (int k = 0; k < s[i].n_ctrl; ++k) st_free(s[i].ctrl_name[k]);
    st_free(s[i].ctrl_name);
    st_free(s[i].ctrl_val);
    st_free(s[i].ws);
    st_free(s[i].event);
  }
  st_free(s);
}

typedef struct {
  st_segment* v;
  int n, cap;
} segvec;
static st_segment* add_seg(fx* x, segvec* S) {
  if (S->n == S->cap) {
    int c = S->cap ? 2 * S->cap : 8;
    st_segment* v = st_malloc((size_t)c * sizeof *v);
    if (!v) {
      fx_fail(x, "MemoryError: out of memory");
      return NULL;
    }
    if (S->n) memcpy(v, S->v, (size_t)S->n * sizeof *v);
    st_free(S->v);
    S->v = v;
    S->cap = c;
  }
  st_segment* s = &S->v[S->n++];
  memset(s, 0, sizeof *s);
  return s;
}

static int is_known(const char* name, const char* const* tags, int n) {
  for (int k = 0; k < n; ++k)
    if (tags[k] && !strcmp(tags[k], name)) return 1;
  return 0;
}

static void r_emph(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* o) {
  u32s_ascii(x, o, "<emph>");
  rxm_put(x, o, m, s, 1);
  u32s_ascii(x, o, "</emph>");
}

/* text run with the current style / prosody */
static void text_run(mk* M, const u32* s, int n, int style, unsigned pros) {
  piece* p = add_piece(M, P_TEXT);
  if (!p) return;
  u32s_put(M->x, &p->text, s, n);
  p->style = style;
  p->pros = pros;
}

static void resolve_prompt(mk* M, st_prompt_style_fn fn, void* user, const char* prompt, int* idx) {
  char* tag = NULL;
  char** names = NULL;
  float* vals = NULL;
  int n = 0;
  if (fn(user, prompt, &tag, &n, &names, &vals) != 0) {
    M->x->err = 1;  /* the callback set the message */
  } else {
    *idx = add_style(M, tag ? tag : "none", (const char* const*)names, vals, n);
  }
  if (names)
    for (int k = 0; k < n; ++k) st_free(names[k]);
  st_free(names);
  st_free(vals);
  st_free(tag);
}

int st_front_parse_markup(st_front* f, const char* text_utf8, const char* default_tag, const char* const* ctrl_names,
                          const float* ctrl_vals, int n_ctrl, const char* const* known_tags, int n_tags,
                          st_prompt_style_fn prompt_fn, void* prompt_user, st_segment** segs_out, int* n_segs) {
  return st_front_parse_markup_ev(f, text_utf8, default_tag, ctrl_names, ctrl_vals, n_ctrl, known_tags, n_tags,
                                  prompt_fn, prompt_user, NULL, NULL, segs_out, n_segs);
}

int st_front_parse_markup_ev(st_front* f, const char* text_utf8, const char* default_tag, const char* const* ctrl_names,
                             const float* ctrl_vals, int n_ctrl, const char* const* known_tags, int n_tags,
                             st_prompt_style_fn prompt_fn, void* prompt_user, st_event_fn is_event, void* event_user,
                             st_segment** segs_out, int* n_segs) {
  *segs_out = NULL;
  *n_segs = 0;
  if (!f || !text_utf8) {
    st_seterr("steertts front: NULL argument");
    return -1;
  }
  fx X = {0}, *x = &X;
  mk M;
  memset(&M, 0, sizeof M);
  M.x = x;
  M.f = f;
  M.memo_n = -1;
  const st_ucd* u = &f->ucd;
  segvec S = {0};
  st_u32s raw = {0}, text = {0};
  int* style_stack = NULL;
  int nss = 0, *pros_stack = NULL, nps = 0;
  int* stack_mem = NULL;

  /* d_tag, d_ctrl = _style_of_prompt(None) -> ("none", {}); tag / ctrl override */
  const char* d_tag = "none";
  if (default_tag && *default_tag && is_known(default_tag, known_tags, n_tags)) d_tag = default_tag;
  int d_style = add_style(&M, d_tag, ctrl_names, ctrl_vals, n_ctrl > 0 ? n_ctrl : 0);

  u32s_utf8(x, &raw, text_utf8, strlen(text_utf8));
  st_rx_sub(x, f->emph, raw.p, raw.n, r_emph, NULL, &text);
  u32s_free(&raw);
  /* style / prosody stack depths <= number of tags + 1 */
  stack_mem = malloc(2 * ((size_t)text.n + 2) * sizeof(int));
  if (!stack_mem) fx_fail(x, "MemoryError: out of memory");
  style_stack = stack_mem;
  pros_stack = stack_mem ? stack_mem + text.n + 2 : NULL;
  if (!x->err) style_stack[nss++] = d_style;
  unsigned pros_cur = 0;
  int pos = 0, it = 0;
  st_rxm m;
  st_u32s tmp = {0};
  while (!x->err && st_rx_next(x, f->token, text.p, text.n, &it, &m)) {
    pros_cur = 0;
    for (int k = 0; k < nps; ++k) pros_cur |= 1u << pros_stack[k];
    if (m.start > pos) text_run(&M, text.p + pos, m.start - pos, style_stack[nss - 1], pros_cur);
    pos = m.end;
    if (rxm_has(&m, 5)) {  /* [pause ...] / [event] */
      tmp.n = 0;
      ucd_lower_small(x, u, rxm_ptr(&m, text.p, 6), rxm_len(&m, 6), &tmp);
      if (is_lit(tmp.p, tmp.n, "pause")) {
        int ms = 400;
        if (rxm_has(&m, 7)) {
          const u32* v = rxm_ptr(&m, text.p, 7);
          int vn = rxm_len(&m, 7), dots = 0, digits = 0, is_s = 0;
          if (rxm_has(&m, 8)) {
            st_u32s un = {0};
            ucd_lower_small(x, u, rxm_ptr(&m, text.p, 8), rxm_len(&m, 8), &un);
            is_s = is_lit(un.p, un.n, "s");
            u32s_free(&un);
          }
          st_sb a = {0};
          for (int i = 0; i < vn; ++i) {
            int d = ucd_digit(u, v[i]);
            if (d >= 0) {
              sb_ch(x, &a, (char)('0' + d));
              ++digits;
            } else {
              sb_ch(x, &a, '.');
              ++dots;
            }
          }
          if (dots > 1 || digits == 0) {
            st_sb vs = {0};
            sb_utf8(x, &vs, v, vn);
            fx_fail(x, "ValueError: could not convert string to float: '%s'", sb_cstr(&vs));
            sb_free(&vs);
          } else {
            double val = st_front_strtod(sb_cstr(&a)) * (is_s ? 1000.0 : 1.0);
            if (isinf(val)) {
              fx_fail(x, "OverflowError: cannot convert float infinity to integer");
            } else {
              double tr = trunc(val);
              ms = tr > 5000 ? 5000 : (int)tr;
            }
          }
          sb_free(&a);
        }
        piece* p = add_piece(&M, P_PAUSE);
        if (p) p->ms = ms < 5000 ? ms : 5000;
      } else if (is_event && is_event_name(x, is_event, event_user, &tmp)) {  /* a recorded event: its own chunk */
        piece* p = add_piece(&M, P_EVENT);
        if (p) u32s_put(x, &p->text, tmp.p, tmp.n);
      } else {  /* event token the bank lacks (or unknown: dropped by to_phonemes) stays in the TTS text */
        st_u32s ev = {0};
        u32s_ascii(x, &ev, " [");
        u32s_put(x, &ev, tmp.p, tmp.n);
        u32s_ascii(x, &ev, "] ");
        text_run(&M, ev.p, ev.n, style_stack[nss - 1], pros_cur);
        u32s_free(&ev);
      }
      continue;
    }
    int closing = rxm_len(&m, 2) > 0;  /* bool(m.group(2)) */
    tmp.n = 0;
    ucd_lower_small(x, u, rxm_ptr(&m, text.p, 3), rxm_len(&m, 3), &tmp);
    st_sb name = {0};
    sb_utf8(x, &name, tmp.p, tmp.n);
    const char* nm = sb_cstr(&name);
    int pi = -1;
    for (int k = 0; k < N_PROS; ++k)
      if (!strcmp(nm, PROS[k].name)) pi = k;
    if (pi >= 0) {
      if (closing) {
        for (int k = 0; k < nps; ++k) {
          if (pros_stack[k] == pi) {  /* list.remove: first occurrence */
            memmove(pros_stack + k, pros_stack + k + 1, (size_t)(nps - k - 1) * sizeof(int));
            --nps;
            break;
          }
        }
      } else {
        pros_stack[nps++] = pi;
      }
    } else if (!strcmp(nm, "style") || is_known(nm, known_tags, n_tags)) {
      if (closing) {
        if (nss > 1) --nss;
      } else if (!strcmp(nm, "style")) {
        int idx = -1;
        if (rxm_len(&m, 4) > 0) {  /* non-empty prompt */
          if (!prompt_fn) {
            fx_fail(x, "ValueError: free-text prompts are resolved on the host (the prompt adapter is not on "
                       "device): send the resolved tag + ctrl");
          } else {
            char* prompt = utf8_dup(x, rxm_ptr(&m, text.p, 4), rxm_len(&m, 4));
            if (prompt) resolve_prompt(&M, prompt_fn, prompt_user, prompt, &idx);
            st_free(prompt);
          }
        } else {
          idx = add_style(&M, "none", NULL, NULL, 0);
        }
        if (!x->err) style_stack[nss++] = idx;
      } else {
        style_stack[nss++] = add_style(&M, nm, NULL, NULL, 0);  /* a named style tag resets the default controls */
      }
    }
    /* unknown tag: the marker is dropped, its inner text is spoken normally */
    sb_free(&name);
  }
  u32s_free(&tmp);
  if (!x->err && pos < text.n) {
    pros_cur = 0;
    for (int k = 0; k < nps; ++k) pros_cur |= 1u << pros_stack[k];
    text_run(&M, text.p + pos, text.n - pos, style_stack[nss - 1], pros_cur);
  }

  /* split runs at sentence ends so each sentence is its own (streamable) segment */
  piece* runs = M.pc;
  int nruns = M.npc;
  M.pc = NULL;
  M.npc = M.cappc = 0;
  for (int r = 0; r < nruns && !x->err; ++r) {
    piece* R = &runs[r];
    if (R->kind == P_PAUSE || R->kind == P_EVENT || is_lit(R->text.p, R->text.n, "PAUSE") ||
        is_lit(R->text.p, R->text.n, "EVENT")) {
      piece* p = add_piece(&M, R->kind);
      if (p) {
        *p = *R;
        R->text.p = NULL;  /* moved */
        R->text.n = R->text.cap = 0;
      }
      continue;
    }
    st_spans parts = {0};
    st_rx_split(x, f->sent_end, R->text.p, R->text.n, &parts);
    for (int i = 0; i < parts.n && !x->err; ++i) {
      piece* p = add_piece(&M, P_TEXT);
      if (!p) break;
      u32s_put(x, &p->text, R->text.p + parts.p[i].a, parts.p[i].b - parts.p[i].a);
      p->style = R->style;
      p->pros = R->pros;
      if (i + 1 < parts.n) add_piece(&M, P_BREAK);
    }
    free(parts.p);
  }
  for (int r = 0; r < nruns; ++r) u32s_free(&runs[r].text);
  free(runs);

  /* group consecutive pieces with the same style into segments */
  int cur0 = 0, ncur = 0, have_style = 0;
  for (int i = 0; i <= M.npc && !x->err; ++i) {
    const piece* p = i < M.npc ? &M.pc[i] : NULL;
    int txt = p && p->kind == P_TEXT;
    int txt_pause = txt && is_lit(p->text.p, p->text.n, "PAUSE");
    int txt_event = txt && is_lit(p->text.p, p->text.n, "EVENT");
    int txt_break = txt && is_lit(p->text.p, p->text.n, "BREAK");
    int boundary = !p || !txt || txt_pause || txt_event || txt_break;
    int restyle = !boundary && have_style && !style_eq(&M.st[p->style], &M.st[M.pc[cur0].style]);
    if ((boundary || restyle) && ncur) {  /* flush */
      int a, b, any = 0;
      for (int k = cur0; k < cur0 + ncur && !any; ++k) {
        front_strip(f, M.pc[k].text.p, M.pc[k].text.n, &a, &b);
        any = a < b;
      }
      if (any) {  /* "".join(texts).strip() is non-empty iff some piece has a non-space character */
        st_segment* s = add_seg(x, &S);
        if (s) make_segment(&M, cur0, cur0 + ncur, s);
      }
      ncur = 0;
    }
    if (!p) break;
    if (boundary) {
      have_style = 0;
      if (p->kind == P_PAUSE || p->kind == P_EVENT || txt_pause || txt_event) {
        st_segment* s = add_seg(x, &S);
        if (!s) break;
        s->text = st_strdup("");
        s->tag = st_strdup("none");
        if (!s->text || !s->tag) fx_fail(x, "MemoryError: out of memory");
        if (p->kind == P_PAUSE) {
          s->pause_ms = p->ms;
        } else if (p->kind == P_EVENT) {
          st_sb nm = {0};
          sb_utf8(x, &nm, p->text.p, p->text.n);
          if (!(s->event = st_strdup(sb_cstr(&nm)))) fx_fail(x, "MemoryError: out of memory");
          sb_free(&nm);
        } else {  /* Python bug: pause_ms / event = the run's style tuple */
          s->skip = 1;
          s->pause_ms = txt_pause ? -1 : 0;
        }
      }
      continue;
    }
    if (!ncur) cur0 = i;
    have_style = 1;
    ++ncur;
  }

  for (int i = 0; i < M.npc; ++i) u32s_free(&M.pc[i].text);
  free(M.pc);
  for (int i = 0; i < M.nst; ++i) style_free(&M.st[i]);
  free(M.st);
  u32s_free(&M.memo);
  u32s_free(&text);
  free(stack_mem);
  if (x->err) {
    st_segments_free(S.v, S.n);
    return -1;
  }
  *segs_out = S.v;
  *n_segs = S.n;
  return 0;
}
