/* A small backtracking regex engine with Python `re` (sre) semantics for str patterns, covering exactly the syntax the
 * front end uses: literals, `.`, classes with ranges / negation / \d \s \w \D \S \W, groups (capturing, (?:...)),
 * alternation (ordered), greedy and lazy quantifiers * + ? {m} {m,} {m,n}, anchors ^ $ \A \Z, \b \B, lookahead
 * (?= ) (?! ), fixed-width lookbehind (?<= ) (?<! ), back references \1..\9 and IGNORECASE. Unicode classes and
 * case folding come from the CPython tables (st_ucd).
 * Like sre it is a plain depth-first backtracker (continuation passing), so it returns the match sre returns. Two
 * exact speed-ups: single-character repeats count their run and only try lengths whose next character can start the
 * rest of the pattern, and repeats of a group with at most one match per position iterate instead of recursing. */
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "st_front_int.h"

enum { CHR, ANY, CLS, BOL, EOL, EOS, WB, NWB, GRP, ALT, REP, LOOK, BREF };

typedef struct rxn rxn;
typedef struct {
  rxn** v;
  int n;
} rxseq;
typedef struct {
  u32 (*rg)[2];
  int nrg;
  unsigned char neg, d, D, s, S, w, W;
} rxcls;
struct rxn {
  int t;
  u32 c;
  rxcls cls;
  int grp;              /* GRP: capture index or -1; BREF: group */
  rxseq** alts;         /* ALT */
  int nalts;
  rxseq* body;          /* GRP / REP / LOOK */
  int mn, mx, greedy, behind, neg, width;
  int det;              /* REP whose body has at most one match per start position */
  /* single-character REP: the rest of its sequence must start with a character matching one of these nodes
   * (nfollow == 0: no pruning) */
  const rxn** follow;
  int nfollow;
};
static int single_node(const rxn* n) { return n->t == CHR || n->t == ANY || n->t == CLS; }

struct st_rx {
  const st_ucd* u;
  int icase, ngroups;
  rxseq* root;
  const rxn* first;     /* a single-character node every match starts with, or NULL */
  void** mem;           /* every allocation, freed together */
  int nmem, capmem;
};

/* ------------------------------------------------------------------ allocation */
static int track(st_rx* R, void* p) {
  if (R->nmem == R->capmem) {
    int c = R->capmem ? 2 * R->capmem : 64;
    void** m = realloc(R->mem, (size_t)c * sizeof *m);
    if (!m) return -1;
    R->mem = m;
    R->capmem = c;
  }
  R->mem[R->nmem++] = p;
  return 0;
}
static void* rx_new(st_rx* R, size_t n) {
  void* p = calloc(1, n ? n : 1);
  if (p && track(R, p)) {
    free(p);
    p = NULL;
  }
  return p;
}
/* realloc a tracked block (p == NULL: new block) */
static void* rx_grow(st_rx* R, void* p, size_t n) {
  if (!p) return rx_new(R, n);
  for (int k = R->nmem; k-- > 0;) {
    if (R->mem[k] == p) {
      void* q = realloc(p, n);
      if (q) R->mem[k] = q;
      return q;
    }
  }
  return NULL;
}

void st_rx_free(st_rx* r) {
  if (!r) return;
  for (int k = 0; k < r->nmem; ++k) free(r->mem[k]);
  free(r->mem);
  free(r);
}
int st_rx_groups(const st_rx* r) { return r->ngroups; }

/* ------------------------------------------------------------------ parser */
typedef struct {
  st_rx* R;
  u32* p;
  int np, i, err;
} ps;

static void ps_fail(ps* P, const char* why) {
  if (P->err) return;
  P->err = 1;
  fx x = {0};
  st_sb pat = {0};
  sb_utf8(&x, &pat, P->p, P->np);
  st_seterr("steertts front regex: %s in %s", why, sb_cstr(&pat));
  sb_free(&pat);
}
static int more(const ps* P) { return P->i < P->np; }
static rxn* node(ps* P, int t) {
  rxn* n = rx_new(P->R, sizeof *n);
  if (!n) {
    ps_fail(P, "out of memory");
    return NULL;
  }
  n->t = t;
  n->grp = -1;
  n->greedy = 1;
  return n;
}
static rxseq* seq_new(ps* P) {
  rxseq* s = rx_new(P->R, sizeof *s);
  if (!s) ps_fail(P, "out of memory");
  return s;
}
static void seq_push(ps* P, rxseq* s, rxn* n) {
  rxn** v = rx_grow(P->R, s->v, (size_t)(s->n + 1) * sizeof *v);
  if (!v) {
    ps_fail(P, "out of memory");
    return;
  }
  s->v = v;
  s->v[s->n++] = n;
}

static rxseq* parse_alt(ps* P);
static rxn* parse_atom(ps* P);

static int parse_int(ps* P, int* v) {
  int j = P->i;
  *v = 0;
  while (j < P->np && P->p[j] >= '0' && P->p[j] <= '9' && *v < 100000000) *v = *v * 10 + (int)(P->p[j++] - '0');
  int ok = j > P->i;
  P->i = j;
  return ok;
}

/* ascii-only membership test (Python semantics for ascii), usable at compile time */
static int asp(u32 x) { return x == ' ' || (x >= 9 && x <= 13) || (x >= 28 && x <= 31); }
static int adg(u32 x) { return x >= '0' && x <= '9'; }
static int awd(u32 x) { return adg(x) || (x >= 'a' && x <= 'z') || (x >= 'A' && x <= 'Z') || x == '_'; }
static int ascii_in(const rxn* n, u32 c) {
  if (n->t == CHR) return n->c == c;
  if (n->t == ANY) return c != '\n';
  const rxcls* k = &n->cls;
  int r = 0;
  for (int i = 0; i < k->nrg; ++i) r = r || (c >= k->rg[i][0] && c <= k->rg[i][1]);
  r = r || (k->d && adg(c)) || (k->D && !adg(c)) || (k->s && asp(c)) || (k->S && !asp(c)) || (k->w && awd(c)) ||
      (k->W && !awd(c));
  return k->neg ? !r : r;
}
/* (?:x|y|...) where every alternative is one single-character node: always consumes exactly one character */
static int one_char_alt(const rxn* n) {
  if (n->t != ALT) return 0;
  for (int a = 0; a < n->nalts; ++a)
    if (n->alts[a]->n != 1 || !single_node(n->alts[a]->v[0])) return 0;
  return 1;
}
/* A non-capturing group body has at most one match from any start position (so greedy repetition of it never needs
 * to backtrack into an iteration) if it is a sequence of single-character nodes (or alternations of them) and greedy
 * single-character repeats that are either fixed-count or followed by an ascii literal they cannot match (e.g.
 * ",\d{3}" or "\s*\."). It must also consume at least one character. */
static int deterministic(const ps* P, const rxseq* b) {
  if (P->R->icase) return 0;
  int consumes = 0;
  for (int k = 0; k < b->n; ++k) {
    const rxn* n = b->v[k];
    if (single_node(n) || one_char_alt(n)) {
      consumes = 1;
      continue;
    }
    if (n->t != REP || !n->greedy || n->body->n != 1 || !single_node(n->body->v[0])) return 0;
    if (n->mn > 0) consumes = 1;
    if (n->mn == n->mx) continue;
    if (k + 1 >= b->n || b->v[k + 1]->t != CHR || b->v[k + 1]->c >= 128) return 0;
    if (ascii_in(n->body->v[0], b->v[k + 1]->c)) return 0;
  }
  return consumes;
}

static rxn* parse_quant(ps* P, rxn* a) {
  if (!a || !more(P)) return a;
  int mn, mx, save = P->i;
  u32 q = P->p[P->i];
  if (q == '*') {
    mn = 0, mx = INT_MAX, ++P->i;
  } else if (q == '+') {
    mn = 1, mx = INT_MAX, ++P->i;
  } else if (q == '?') {
    mn = 0, mx = 1, ++P->i;
  } else if (q == '{') {
    ++P->i;
    int a1 = 0, a2 = 0;
    int h1 = parse_int(P, &a1);
    if (more(P) && P->p[P->i] == '}') {
      if (!h1) {
        P->i = save;
        return a;
      }
      mn = mx = a1;
      ++P->i;
    } else if (more(P) && P->p[P->i] == ',') {
      ++P->i;
      int h2 = parse_int(P, &a2);
      if (!more(P) || P->p[P->i] != '}') {
        P->i = save;
        return a;
      }
      ++P->i;
      mn = h1 ? a1 : 0;
      mx = h2 ? a2 : INT_MAX;
    } else {
      P->i = save;
      return a;  /* literal '{' (parse_atom sees it next) */
    }
  } else {
    return a;
  }
  rxn* r = node(P, REP);
  rxseq* body = seq_new(P);
  if (P->err) return NULL;
  r->mn = mn;
  r->mx = mx;
  r->body = body;
  seq_push(P, body, a);
  if (more(P) && P->p[P->i] == '?') {
    r->greedy = 0;
    ++P->i;
  }
  if (a->t == GRP && a->grp < 0) r->det = deterministic(P, a->body);
  if (a->t == BREF) r->det = 1;  /* \1{3,}: copies of a fixed string, one way to match each */
  return r;
}

static u32 escape_char(ps* P, u32 e) {
  switch (e) {
    case 'n': return '\n';
    case 't': return '\t';
    case 'r': return '\r';
    case 'f': return '\f';
    case 'v': return '\v';
    case 'a': return '\a';
    default:
      if ((e >= 'a' && e <= 'z') || (e >= 'A' && e <= 'Z') || (e >= '0' && e <= '9')) ps_fail(P, "unsupported escape");
      return e;
  }
}

static rxn* parse_class(ps* P) {
  rxn* n = node(P, CLS);
  if (!n) return NULL;
  rxcls* c = &n->cls;
  if (more(P) && P->p[P->i] == '^') {
    c->neg = 1;
    ++P->i;
  }
  int first = 1;
  while (!P->err) {
    if (!more(P)) {
      ps_fail(P, "unterminated class");
      break;
    }
    u32 ch = P->p[P->i], lo;
    if (ch == ']' && !first) {
      ++P->i;
      break;
    }
    first = 0;
    if (ch == '\\') {
      if (P->i + 1 >= P->np) {
        ps_fail(P, "bad escape");
        break;
      }
      u32 e = P->p[P->i + 1];
      P->i += 2;
      switch (e) {
        case 'd': c->d = 1; continue;
        case 'D': c->D = 1; continue;
        case 's': c->s = 1; continue;
        case 'S': c->S = 1; continue;
        case 'w': c->w = 1; continue;
        case 'W': c->W = 1; continue;
        default: lo = escape_char(P, e);
      }
    } else {
      lo = ch;
      ++P->i;
    }
    u32 hi = lo;
    if (P->i + 1 < P->np && P->p[P->i] == '-' && P->p[P->i + 1] != ']') {
      ++P->i;
      u32 h = P->p[P->i];
      if (h == '\\') {
        if (P->i + 1 >= P->np) {
          ps_fail(P, "bad escape");
          break;
        }
        h = escape_char(P, P->p[P->i + 1]);
        P->i += 2;
      } else {
        ++P->i;
      }
      hi = h;
      if (hi < lo) ps_fail(P, "bad range");
    }
    u32(*rg)[2] = rx_grow(P->R, c->rg, (size_t)(c->nrg + 1) * sizeof *rg);
    if (!rg) {
      ps_fail(P, "out of memory");
      break;
    }
    c->rg = rg;
    c->rg[c->nrg][0] = lo;
    c->rg[c->nrg][1] = hi;
    c->nrg++;
  }
  return n;
}

static int width(ps* P, const rxseq* s) {
  int w = 0;
  for (int k = 0; k < s->n; ++k) {
    const rxn* n = s->v[k];
    switch (n->t) {
      case CHR: case ANY: case CLS: w += 1; break;
      case GRP: w += width(P, n->body); break;
      case ALT: {
        int a0 = width(P, n->alts[0]);
        for (int a = 0; a < n->nalts; ++a)
          if (width(P, n->alts[a]) != a0) ps_fail(P, "variable-width lookbehind");
        w += a0;
        break;
      }
      case REP:
        if (n->mn != n->mx) ps_fail(P, "variable-width lookbehind");
        w += n->mn * width(P, n->body);
        break;
      case BREF: ps_fail(P, "backref in lookbehind"); break;
      default: break;
    }
  }
  return w;
}

static rxseq* parse_seq(ps* P) {
  rxseq* s = seq_new(P);
  while (!P->err && more(P) && P->p[P->i] != '|' && P->p[P->i] != ')') {
    rxn* a = parse_quant(P, parse_atom(P));
    if (P->err) break;
    seq_push(P, s, a);
  }
  return s;
}

static rxseq* parse_alt(ps* P) {
  rxseq* first = parse_seq(P);
  if (P->err || !more(P) || P->p[P->i] != '|') return first;
  rxn* a = node(P, ALT);
  if (!a) return NULL;
  a->alts = rx_grow(P->R, NULL, sizeof(rxseq*));
  if (!a->alts) {
    ps_fail(P, "out of memory");
    return NULL;
  }
  a->alts[a->nalts++] = first;
  while (!P->err && more(P) && P->p[P->i] == '|') {
    ++P->i;
    rxseq* s = parse_seq(P);
    rxseq** v = rx_grow(P->R, a->alts, (size_t)(a->nalts + 1) * sizeof *v);
    if (!v) {
      ps_fail(P, "out of memory");
      return NULL;
    }
    a->alts = v;
    a->alts[a->nalts++] = s;
  }
  rxseq* s = seq_new(P);
  if (!P->err) seq_push(P, s, a);
  return s;
}

static rxn* parse_atom(ps* P) {
  u32 ch = P->p[P->i++];
  rxn* n;
  switch (ch) {
    case '(': {
      rxn* g;
      if (more(P) && P->p[P->i] == '?') {
        ++P->i;
        if (more(P) && P->p[P->i] == ':') {
          ++P->i;
          g = node(P, GRP);
        } else if (more(P) && (P->p[P->i] == '=' || P->p[P->i] == '!')) {
          g = node(P, LOOK);
          if (g) g->neg = P->p[P->i] == '!';
          ++P->i;
        } else if (P->i + 1 < P->np && P->p[P->i] == '<' && (P->p[P->i + 1] == '=' || P->p[P->i + 1] == '!')) {
          g = node(P, LOOK);
          if (g) {
            g->behind = 1;
            g->neg = P->p[P->i + 1] == '!';
          }
          P->i += 2;
        } else {
          ps_fail(P, "unsupported group syntax");
          return NULL;
        }
      } else {
        g = node(P, GRP);
        if (g) {
          g->grp = ++P->R->ngroups;
          if (g->grp > ST_RX_MAXG) ps_fail(P, "too many groups");
        }
      }
      if (!g) return NULL;
      g->body = parse_alt(P);
      if (P->err) return NULL;
      if (!more(P) || P->p[P->i] != ')') {
        ps_fail(P, "missing )");
        return NULL;
      }
      ++P->i;
      if (g->t == LOOK && g->behind) g->width = width(P, g->body);
      return g;
    }
    case '[': return parse_class(P);
    case '.': return node(P, ANY);
    case '^': return node(P, BOL);
    case '$': return node(P, EOL);
    case '\\': {
      if (!more(P)) {
        ps_fail(P, "trailing backslash");
        return NULL;
      }
      u32 e = P->p[P->i++];
      switch (e) {
        case 'd': case 'D': case 's': case 'S': case 'w': case 'W':
          n = node(P, CLS);
          if (n) {
            n->cls.d = e == 'd'; n->cls.D = e == 'D'; n->cls.s = e == 's';
            n->cls.S = e == 'S'; n->cls.w = e == 'w'; n->cls.W = e == 'W';
          }
          return n;
        case 'b': return node(P, WB);
        case 'B': return node(P, NWB);
        case 'Z': return node(P, EOS);
        case 'A': return node(P, BOL);
        default:
          if (e >= '1' && e <= '9') {
            n = node(P, BREF);
            if (n) n->grp = (int)(e - '0');
            return n;
          }
          n = node(P, CHR);
          if (n) n->c = escape_char(P, e);
          return n;
      }
    }
    default:
      n = node(P, CHR);
      if (n) n->c = ch;
      return n;
  }
}

/* ------------------------------------------------------------------ follow sets */
typedef struct {
  const rxn** v;
  int n, cap, oom;
} nodelist;
static void nl_push(nodelist* l, const rxn* n) {
  if (l->n == l->cap) {
    int c = l->cap ? 2 * l->cap : 8;
    const rxn** v = realloc((void*)l->v, (size_t)c * sizeof *v);
    if (!v) {
      l->oom = 1;
      return;
    }
    l->v = v;
    l->cap = c;
  }
  l->v[l->n++] = n;
}

/* FIRST set of q[i..]: single-character nodes one of which must match the first consumed character; returns 0 if
 * the suffix can match without consuming (or contains a back reference), i.e. no usable restriction. */
static int first_set(const rxseq* q, int i, nodelist* out) {
  for (; i < q->n; ++i) {
    const rxn* n = q->v[i];
    switch (n->t) {
      case CHR: case ANY: case CLS:
        nl_push(out, n);
        return 1;
      case BOL: case EOL: case EOS: case WB: case NWB: case LOOK:
        continue;  /* zero width: ignoring them only makes the set larger (still a valid necessary condition) */
      case GRP:
        if (first_set(n->body, 0, out)) return 1;
        continue;
      case ALT: {
        int all = 1;
        for (int a = 0; a < n->nalts; ++a) all = first_set(n->alts[a], 0, out) && all;
        if (all) return 1;
        continue;
      }
      case REP:
        if (first_set(n->body, 0, out) && n->mn > 0) return 1;
        continue;
      case BREF:
        return 0;
    }
  }
  return 0;
}

/* Assign REP::follow. outer / outer_ok: what must follow the end of q (ok = 0: anything / nothing, e.g. the end of
 * the pattern, a lookaround body or a repeated group body - then no pruning past q's end). */
static int follow_walk(st_rx* R, const rxseq* q, const rxn** outer, int nouter, int outer_ok) {
  for (int i = 0; i < q->n; ++i) {
    rxn* n = q->v[i];
    nodelist f = {0};
    int ok = first_set(q, i + 1, &f);
    if (!ok && outer_ok) {
      for (int k = 0; k < nouter; ++k) nl_push(&f, outer[k]);
      ok = 1;
    }
    int rc = f.oom ? -1 : 0;
    if (!rc) {
      if (n->t == REP && n->body->n == 1 && single_node(n->body->v[0])) {
        if (ok && f.n) {
          n->follow = rx_new(R, (size_t)f.n * sizeof *n->follow);
          if (!n->follow) rc = -1;
          else memcpy((void*)n->follow, f.v, (size_t)f.n * sizeof *f.v);
          n->nfollow = f.n;
        }
      } else if (n->t == GRP) {
        rc = follow_walk(R, n->body, f.v, f.n, ok);
      } else if (n->t == ALT) {
        for (int a = 0; a < n->nalts && !rc; ++a) rc = follow_walk(R, n->alts[a], f.v, f.n, ok);
      } else if (n->t == REP || n->t == LOOK) {
        rc = follow_walk(R, n->body, NULL, 0, 0);
      }
    }
    free((void*)f.v);
    if (rc) return -1;
  }
  return 0;
}

/* a single-character node every match must start with (descends into groups and x+ / x{m,} with m >= 1), or NULL */
static const rxn* first_char(const rxseq* q) {
  if (q->n == 0) return NULL;
  const rxn* n = q->v[0];
  if (single_node(n)) return n;
  if (n->t == GRP) return first_char(n->body);
  if (n->t == REP && n->mn >= 1) return first_char(n->body);
  return NULL;
}

st_rx* st_rx_compile(const st_ucd* u, const char* pat, int icase) {
  st_rx* R = calloc(1, sizeof *R);
  if (!R) {
    st_seterr("steertts front: out of memory");
    return NULL;
  }
  R->u = u;
  R->icase = icase;
  fx x = {0};
  st_u32s p = {0};
  u32s_utf8(&x, &p, pat, strlen(pat));
  ps P = {R, p.p, p.n, 0, x.err};
  if (P.err) st_seterr("steertts front: out of memory");
  if (!P.err) R->root = parse_alt(&P);
  if (!P.err && more(&P)) ps_fail(&P, "unbalanced )");
  if (!P.err && follow_walk(R, R->root, NULL, 0, 0)) {
    P.err = 1;
    st_seterr("steertts front: out of memory");
  }
  u32s_free(&p);
  if (P.err) {
    st_rx_free(R);
    return NULL;
  }
  R->first = first_char(R->root);
  return R;
}

/* ------------------------------------------------------------------ matcher */
typedef struct rxk rxk;
struct rxk {
  int (*fn)(const rxk* k, int pos);
};
typedef struct {
  const st_rx* R;
  const u32* s;
  int n;
  int* caps;
  fx* x;
} mt;

static int base_cls(const mt* m, const rxcls* c, u32 ch) {
  for (int i = 0; i < c->nrg; ++i)
    if (ch >= c->rg[i][0] && ch <= c->rg[i][1]) return 1;
  const st_ucd* u = m->R->u;
  if ((c->d || c->D) && (ucd_digit(u, ch) >= 0 ? c->d : c->D)) return 1;
  if ((c->s || c->S) && (ucd_is_space(u, ch) ? c->s : c->S)) return 1;
  if ((c->w || c->W) && (ucd_is_word(u, ch) ? c->w : c->W)) return 1;
  return 0;
}
static int cls_match(const mt* m, const rxcls* c, u32 ch) {
  int r = base_cls(m, c, ch);
  if (!r && m->R->icase) {
    u32 f = ucd_fold(m->R->u, ch);
    if (f != ch) r = base_cls(m, c, f);
    if (!r && f >= 'a' && f <= 'z') r = base_cls(m, c, f - 32);
  }
  return c->neg ? !r : r;
}
static int single(const mt* m, const rxn* nd, u32 ch) {
  switch (nd->t) {
    case CHR: return m->R->icase ? ucd_fold(m->R->u, ch) == ucd_fold(m->R->u, nd->c) : ch == nd->c;
    case ANY: return ch != '\n';
    default: return cls_match(m, &nd->cls, ch);
  }
}
static int word_at(const mt* m, int p) { return p >= 0 && p < m->n && ucd_is_word(m->R->u, m->s[p]); }

static int mseq(mt* m, const rxseq* q, int i, int pos, const rxk* k);
static int mrep(mt* m, const rxn* nd, int count, int pos, const rxk* k);

typedef struct {
  rxk k;
} k_true;
static int k_true_fn(const rxk* k, int p) { (void)k; (void)p; return 1; }
typedef struct {
  rxk k;
  int at;
} k_at;
static int k_at_fn(const rxk* k, int p) { return p == ((const k_at*)k)->at; }
typedef struct {
  rxk k;
  int* out;
} k_end;
static int k_end_fn(const rxk* k, int p) {
  *((const k_end*)k)->out = p;
  return 1;
}
typedef struct {
  rxk k;
  mt* m;
  const rxseq* q;
  int i;
  const rxk* next;
} k_rest;
static int k_rest_fn(const rxk* k, int p) {
  const k_rest* r = (const k_rest*)k;
  return mseq(r->m, r->q, r->i, p, r->next);
}
typedef struct {
  rxk k;
  mt* m;
  int g, start;
  const rxk* next;
} k_grp;
static int k_grp_fn(const rxk* k, int p) {
  const k_grp* r = (const k_grp*)k;
  int* caps = r->m->caps;
  int o0 = caps[2 * r->g], o1 = caps[2 * r->g + 1];
  caps[2 * r->g] = r->start;
  caps[2 * r->g + 1] = p;
  if (r->next->fn(r->next, p)) return 1;
  caps[2 * r->g] = o0;
  caps[2 * r->g + 1] = o1;
  return 0;
}
typedef struct {
  rxk k;
  mt* m;
  const rxn* nd;
  int count, start;
  const rxk* next;
} k_rep;
static int k_rep_fn(const rxk* k, int p) {
  const k_rep* r = (const k_rep*)k;
  if (p == r->start) return 0;  /* an empty iteration ends the loop (sre does the same) */
  return mrep(r->m, r->nd, r->count, p, r->next);
}

static int mrep(mt* m, const rxn* nd, int count, int pos, const rxk* k) {
  if (nd->greedy) {
    if (count < nd->mx) {
      k_rep kr = {{k_rep_fn}, m, nd, count + 1, pos, k};
      if (mseq(m, nd->body, 0, pos, &kr.k)) return 1;
    }
    return count >= nd->mn && k->fn(k, pos);
  }
  if (count >= nd->mn && k->fn(k, pos)) return 1;
  if (count < nd->mx) {
    k_rep kr = {{k_rep_fn}, m, nd, count + 1, pos, k};
    return mseq(m, nd->body, 0, pos, &kr.k);
  }
  return 0;
}

/* can the rest of the sequence start at p? (exact pruning) */
static int viable(const mt* m, const rxn* nd, int p) {
  if (!nd->nfollow) return 1;
  if (p >= m->n) return 0;
  for (int f = 0; f < nd->nfollow; ++f)
    if (single(m, nd->follow[f], m->s[p])) return 1;
  return 0;
}

static int mseq(mt* m, const rxseq* q, int i, int pos, const rxk* k) {
  const st_ucd* u = m->R->u;
  const u32* s = m->s;
  const int n = m->n;
  for (; i < q->n; ++i) {
    const rxn* nd = q->v[i];
    switch (nd->t) {
      case CHR: case ANY: case CLS:
        if (pos < n && single(m, nd, s[pos])) {
          ++pos;
          continue;
        }
        return 0;
      case BOL:
        if (pos != 0) return 0;
        continue;
      case EOL:
        if (!(pos == n || (pos == n - 1 && s[pos] == '\n'))) return 0;
        continue;
      case EOS:
        if (pos != n) return 0;
        continue;
      case WB: case NWB: {
        int b = word_at(m, pos - 1) != word_at(m, pos);
        if (b != (nd->t == WB)) return 0;
        continue;
      }
      case LOOK: {
        int r;
        if (!nd->behind) {
          k_true kt = {{k_true_fn}};
          r = mseq(m, nd->body, 0, pos, &kt.k);
        } else {
          int st = pos - nd->width;
          k_at ka = {{k_at_fn}, pos};
          r = st >= 0 && mseq(m, nd->body, 0, st, &ka.k);
        }
        if (r == nd->neg) return 0;
        continue;
      }
      case BREF: {
        int g = nd->grp;
        if (m->caps[2 * g] < 0) return 0;
        int a = m->caps[2 * g], len = m->caps[2 * g + 1] - a;
        if (pos + len > n) return 0;
        for (int t = 0; t < len; ++t) {
          u32 x = s[a + t], y = s[pos + t];
          if (m->R->icase ? ucd_fold(u, x) != ucd_fold(u, y) : x != y) return 0;
        }
        pos += len;
        continue;
      }
      case GRP: {
        k_rest rest = {{k_rest_fn}, m, q, i + 1, k};
        if (nd->grp < 0) return mseq(m, nd->body, 0, pos, &rest.k);
        k_grp kg = {{k_grp_fn}, m, nd->grp, pos, &rest.k};
        return mseq(m, nd->body, 0, pos, &kg.k);
      }
      case ALT: {
        k_rest rest = {{k_rest_fn}, m, q, i + 1, k};
        for (int a = 0; a < nd->nalts; ++a)
          if (mseq(m, nd->alts[a], 0, pos, &rest.k)) return 1;
        return 0;
      }
      case REP: {
        k_rest rest = {{k_rest_fn}, m, q, i + 1, k};
        const rxn* b = nd->body->v[0];
        if (nd->body->n == 1 && single_node(b)) {  /* x*, [..]{m,n}: count the run, then try lengths */
          int mx = nd->mx, c = 0;
          if (nd->greedy) {
            while (c < mx && pos + c < n && single(m, b, s[pos + c])) ++c;
            for (; c >= nd->mn; --c)
              if (viable(m, nd, pos + c) && k_rest_fn(&rest.k, pos + c)) return 1;
            return 0;
          }
          while (c < nd->mn) {
            if (pos + c < n && single(m, b, s[pos + c]))
              ++c;
            else
              return 0;
          }
          for (;;) {
            if (viable(m, nd, pos + c) && k_rest_fn(&rest.k, pos + c)) return 1;
            if (c >= mx || pos + c >= n || !single(m, b, s[pos + c])) return 0;
            ++c;
          }
        }
        if (nd->det && nd->greedy) {  /* unique body match per position: collect iterations, then back off */
          int stack_ps[64], *ps_ = stack_ps, cap = 64, np = 0, p = pos, ok = 0;
          ps_[np++] = pos;
          while (np - 1 < nd->mx) {
            int e = -1;
            k_end ke = {{k_end_fn}, &e};
            if (!mseq(m, nd->body, 0, p, &ke.k) || e == p) break;
            if (np == cap) {
              int* q2 = malloc((size_t)cap * 2 * sizeof(int));
              if (!q2) {
                fx_fail(m->x, "MemoryError: out of memory");
                if (ps_ != stack_ps) free(ps_);
                return 0;
              }
              memcpy(q2, ps_, (size_t)np * sizeof(int));
              if (ps_ != stack_ps) free(ps_);
              ps_ = q2;
              cap *= 2;
            }
            ps_[np++] = e;
            p = e;
          }
          for (int c = np - 1; c >= nd->mn; --c)
            if (k_rest_fn(&rest.k, ps_[c])) {
              ok = 1;
              break;
            }
          if (ps_ != stack_ps) free(ps_);
          return ok;
        }
        return mrep(m, nd, 0, pos, &rest.k);
      }
    }
  }
  return k->fn(k, pos);
}

static void caps_reset(const st_rx* r, st_rxm* m) {
  for (int g = 0; g < 2 * (r->ngroups + 1); ++g) m->caps[g] = -1;
}

int st_rx_match_at(fx* x, const st_rx* r, const u32* s, int n, int pos, st_rxm* m) {
  caps_reset(r, m);
  mt t = {r, s, n, m->caps, x};
  int end = -1;
  k_end ke = {{k_end_fn}, &end};
  if (!mseq(&t, r->root, 0, pos, &ke.k)) return 0;
  m->start = m->caps[0] = pos;
  m->end = m->caps[1] = end;
  return 1;
}

int st_rx_search(fx* x, const st_rx* r, const u32* s, int n, int pos, st_rxm* m) {
  mt t = {r, s, n, m->caps, x};
  for (int st = pos; st <= n && !x->err; ++st) {
    if (r->first && (st >= n || !single(&t, r->first, s[st]))) continue;
    int end = -1;
    k_end ke = {{k_end_fn}, &end};
    caps_reset(r, m);
    if (mseq(&t, r->root, 0, st, &ke.k)) {
      m->start = m->caps[0] = st;
      m->end = m->caps[1] = end;
      return 1;
    }
  }
  return 0;
}

int st_rx_next(fx* x, const st_rx* r, const u32* s, int n, int* pos, st_rxm* m) {
  if (*pos > n || x->err || !st_rx_search(x, r, s, n, *pos, m)) return 0;
  *pos = m->end > m->start ? m->end : m->end + 1;
  return 1;
}

void rxm_put(fx* x, st_u32s* out, const st_rxm* m, const u32* s, int g) {
  if (rxm_has(m, g)) u32s_put(x, out, s + m->caps[2 * g], m->caps[2 * g + 1] - m->caps[2 * g]);
}

void st_rx_sub(fx* x, const st_rx* r, const u32* s, int n, st_rx_repl f, void* ud, st_u32s* out) {
  int pos = 0, last = 0;
  st_rxm m;
  while (pos <= n && !x->err && st_rx_search(x, r, s, n, pos, &m)) {
    u32s_put(x, out, s + last, m.start - last);
    f(x, ud, &m, s, out);
    last = m.end;
    pos = m.end > m.start ? m.end : m.end + 1;
    if (m.end == m.start && m.start < n) {  /* (never happens with the front end's patterns) */
      u32s_push(x, out, s[m.start]);
      last = m.start + 1;
    }
  }
  if (last < n) u32s_put(x, out, s + last, n - last);
}

static void spans_push(fx* x, st_spans* o, int a, int b) {
  if (o->n == o->cap) {
    int c = o->cap ? 2 * o->cap : 8;
    st_span* p = realloc(o->p, (size_t)c * sizeof *p);
    if (!p) {
      fx_fail(x, "MemoryError: out of memory");
      return;
    }
    o->p = p;
    o->cap = c;
  }
  o->p[o->n].a = a;
  o->p[o->n].b = b;
  o->n++;
}

void st_rx_split(fx* x, const st_rx* r, const u32* s, int n, st_spans* out) {
  int pos = 0, last = 0;
  st_rxm m;
  while (pos <= n && !x->err && st_rx_search(x, r, s, n, pos, &m)) {
    if (m.end == m.start) {  /* (never happens with the front end's patterns) */
      pos = m.end + 1;
      continue;
    }
    spans_push(x, out, last, m.start);
    for (int g = 1; g <= r->ngroups; ++g) {
      if (rxm_has(&m, g))
        spans_push(x, out, m.caps[2 * g], m.caps[2 * g + 1]);
      else
        spans_push(x, out, 0, 0);
    }
    last = pos = m.end;
  }
  spans_push(x, out, last, n);
}
