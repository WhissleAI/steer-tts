/* Text normalisation (steertts_edge text.normalize / _expand / _pre) without a regex engine.
 *
 * Every pass is one re.sub of the reference: a matcher tried at each position from left to right (leftmost match,
 * non-overlapping), writing its replacement on success. The expression each matcher implements is quoted above it;
 * where Python's backtracking can pick a shorter alternative, the matcher tries the alternatives in the same order. */
#include <stdio.h>
#include <stdlib.h>

#include "nf_int.h"

typedef struct {
  nano_front_ctx* c;
  const u32* s;
  int n;
} M;

/* ------------------------------------------------------------------ character classes */
static int at(const M* m, int i) { return i >= 0 && i < m->n ? (int)m->s[i] : -1; }
static int isw(const M* m, int i) { return i >= 0 && i < m->n && nf_is_word(m->c, m->s[i]); }
static int bnd(const M* m, int i) { return isw(m, i - 1) != isw(m, i); }   /* \b */
static int isd(const M* m, int i) { return i >= 0 && i < m->n && nf_digit(m->c, m->s[i]) >= 0; }   /* \d */
static int iss(const M* m, int i) { return i >= 0 && i < m->n && nf_is_space(m->c, m->s[i]); }     /* \s */
static int d09(const M* m, int i) { int ch = at(m, i); return ch >= '0' && ch <= '9'; }
static int AZ(const M* m, int i) { int ch = at(m, i); return ch >= 'A' && ch <= 'Z'; }
static int az(const M* m, int i) { int ch = at(m, i); return ch >= 'a' && ch <= 'z'; }
static int alpha(const M* m, int i) { return AZ(m, i) || az(m, i); }
static int alnum(const M* m, int i) { return alpha(m, i) || d09(m, i); }
static int lit(const M* m, int i, const char* w) {   /* literal at i: length or 0 */
  int k = 0;
  for (; w[k]; k++)
    if (at(m, i + k) != (unsigned char)w[k]) return 0;
  return k;
}
static int spaces(const M* m, int i) {   /* \s* : end */
  while (iss(m, i)) i++;
  return i;
}

typedef int (*matcher)(M* m, int i, nfs* out, const void* ud);

/* re.sub(pattern, repl, s): matcher returns the match end (> i) after writing the replacement, or -1 */
static void sub(nano_front_ctx* c, const nfs* in, nfs* out, matcher f, const void* ud) {
  M m = {c, in->p, in->n};
  out->n = 0;
  for (int i = 0; i < in->n && !c->fail;) {
    int save = out->n;
    int e = f(&m, i, out, ud);
    if (e > i) {
      i = e;
    } else {
      out->n = save;
      nfs_push(out, in->p[i++]);
    }
  }
}

/* ------------------------------------------------------------------ number helpers (text._num) */
static void put_num_words(const nfa* w, nfs* out) {
  for (int i = 0; i < w->n; ++i) {
    char ch = w->p[i];
    if (ch == '-')
      nfs_push(out, ' ');
    else if (ch != ',')
      nfs_push(out, (unsigned char)ch);
  }
}
static void num_int(nano_front_ctx* c, const u32* s, int n, int ord, nfs* out) {
  char db[200], wb[NF_MAX_CP];
  nfa d = {db, 0, (int)sizeof db, c}, w = {wb, 0, (int)sizeof wb, c};
  db[0] = wb[0] = 0;
  if (n >= (int)sizeof db - 1) {   /* > 198 digits: num2words raises for > 102 anyway */
    char lim[64];
    for (int i = 0; i < 60; i++) lim[i] = (char)('0' + nf_digit(c, s[i]));
    lim[60] = 0;
    nf_fail(c, "OverflowError: abs(%s...) must be less than 10**102.", lim);
    return;
  }
  nf_py_int(c, s, n, &d);
  if (c->fail) return;
  if (ord)
    nf_ordinal(c, db, &w);
  else
    nf_cardinal(c, db, &w);
  if (!c->fail) put_num_words(&w, out);
}
static void num_float(nano_front_ctx* c, const u32* s, int n, nfs* out) {
  char ab[128], wb[NF_MAX_CP];
  nfa a = {ab, 0, (int)sizeof ab, c}, w = {wb, 0, (int)sizeof wb, c};
  ab[0] = wb[0] = 0;
  for (int i = 0; i < n; ++i) {
    int d = nf_digit(c, s[i]);
    nfa_ch(&a, d >= 0 ? (char)('0' + d) : (char)s[i]);
  }
  if (!c->fail) nf_cardinal_float(c, nf_strtod(ab), &w);
  if (!c->fail) put_num_words(&w, out);
}
static void put_letter(nfs* o, u32 ch) {   /* a lone "A" is the article: spell it "ay" */
  if (ch == 'A')
    nfs_ascii(o, "ay");
  else
    nfs_push(o, ch);
}
static void put_int_ascii(nfs* o, const u32* s, int n) {   /* str(int(s)) for ASCII digits */
  int i = 0;
  while (i + 1 < n && s[i] == '0') ++i;
  nfs_put(o, s + i, n - i);
}
static void put_ampm(nfs* o, u32 ch) { nfs_ascii(o, ch == 'A' || ch == 'a' ? " ay M" : " P M"); }

/* ------------------------------------------------------------------ quotes */
/* ["“”‘’`] -> "'" for ‘ ’ `, else "" */
static int m_quote(M* m, int i, nfs* o, const void* ud) {
  u32 ch = m->s[i];
  if (ch == '"' || ch == 0x201C || ch == 0x201D) return i + 1;
  if (ch == 0x2018 || ch == 0x2019 || ch == '`') {
    nfs_push(o, '\'');
    return i + 1;
  }
  return -1;
}

/* ------------------------------------------------------------------ text._pre */
/* (?<![0-9-])(?:\(([0-9]{3})\)\s?|([0-9]{3})[-.])([0-9]{3})[-.]([0-9]{4})(?![0-9-]) */
static int d3(const M* m, int i) { return d09(m, i) && d09(m, i + 1) && d09(m, i + 2); }
static int m_phone(M* m, int i, nfs* o, const void* ud) {
  if (d09(m, i - 1) || at(m, i - 1) == '-') return -1;
  int g1 = -1, k;
  if (at(m, i) == '(' && d3(m, i + 1) && at(m, i + 4) == ')') {
    g1 = i + 1;
    k = i + 5;
    if (iss(m, k) && d09(m, k + 1)) k++;
  } else if (d3(m, i) && (at(m, i + 3) == '-' || at(m, i + 3) == '.')) {
    g1 = i;
    k = i + 4;
  } else {
    return -1;
  }
  if (!d3(m, k) || !(at(m, k + 3) == '-' || at(m, k + 3) == '.')) return -1;
  int g3 = k, g4 = k + 4;
  for (int t = 0; t < 4; t++)
    if (!d09(m, g4 + t)) return -1;
  int e = g4 + 4;
  if (d09(m, e) || at(m, e) == '-') return -1;
  int st[3] = {g1, g3, g4}, ln[3] = {3, 3, 4};
  for (int g = 0; g < 3; g++) {
    if (g) nfs_ascii(o, ", ");
    for (int t = 0; t < ln[g]; t++) {
      if (t) nfs_push(o, ' ');
      nfs_push(o, m->s[st[g] + t]);
    }
  }
  return e;
}

/* \s?([AaPp])(?:\.[Mm]\.?|[Mm])(?![A-Za-z]) at q: end or -1; *ch = the [AaPp] position */
static int ampm_at(const M* m, int q, int* ch) {
  for (int sp = iss(m, q) ? 1 : 0; sp >= 0; sp--) {
    int k = q + sp, a = at(m, k);
    if (!(a == 'A' || a == 'a' || a == 'P' || a == 'p')) continue;
    int m1 = at(m, k + 2);
    if (at(m, k + 1) == '.' && (m1 == 'M' || m1 == 'm')) {
      if (at(m, k + 3) == '.' && !alpha(m, k + 4)) {
        *ch = k;
        return k + 4;
      }
      if (!alpha(m, k + 3)) {
        *ch = k;
        return k + 3;
      }
    }
    int m2 = at(m, k + 1);
    if ((m2 == 'M' || m2 == 'm') && !alpha(m, k + 2)) {
      *ch = k;
      return k + 2;
    }
  }
  return -1;
}

/* (?<![0-9:])\b([01]?[0-9]|2[0-3]):([0-5][0-9])(?!:[0-9])(?:\s?([AaPp])(?:\.[Mm]\.?|[Mm])(?![A-Za-z]))? */
static int m_time(M* m, int i, nfs* o, const void* ud) {
  if (d09(m, i - 1) || at(m, i - 1) == ':' || !bnd(m, i)) return -1;
  int cand[3], nc = 0;
  if ((at(m, i) == '0' || at(m, i) == '1') && d09(m, i + 1)) cand[nc++] = 2;
  if (d09(m, i)) cand[nc++] = 1;
  if (at(m, i) == '2' && at(m, i + 1) >= '0' && at(m, i + 1) <= '3') cand[nc++] = 2;
  for (int k = 0; k < nc; k++) {
    int h = i + cand[k];
    if (at(m, h) != ':' || !(at(m, h + 1) >= '0' && at(m, h + 1) <= '5') || !d09(m, h + 2)) continue;
    if (at(m, h + 3) == ':' && d09(m, h + 4)) continue;
    int e = h + 3, ach = -1, ae = ampm_at(m, e, &ach);
    const u32 *H = m->s + i, *mm = m->s + h + 1;
    int hn = cand[k], hv = 0;
    for (int t = 0; t < hn; t++) hv = 10 * hv + (int)(H[t] - '0');
    put_int_ascii(o, H, hn);
    if (mm[0] != '0' || mm[1] != '0') {
      if (mm[0] == '0') {
        nfs_ascii(o, " oh ");
        nfs_push(o, mm[1]);
      } else {
        nfs_push(o, ' ');
        nfs_put(o, mm, 2);
      }
    } else if (ae < 0) {
      nfs_ascii(o, hv > 12 || hv == 0 || H[0] == '0' ? " hundred" : " o'clock");
    }
    if (ae >= 0) {
      put_ampm(o, m->s[ach]);
      return ae;
    }
    return e;
  }
  return -1;
}

/* (?<=[0-9]):(?=[0-9]) -> " " */
static int m_colon(M* m, int i, nfs* o, const void* ud) {
  if (at(m, i) != ':' || !d09(m, i - 1) || !d09(m, i + 1)) return -1;
  nfs_push(o, ' ');
  return i + 1;
}

/* \b([0-9]{1,2})\s?([AaPp])(?:\.[Mm]\.?|[Mm])(?![A-Za-z]) */
static int m_ampm(M* m, int i, nfs* o, const void* ud) {
  if (!d09(m, i) || !bnd(m, i)) return -1;
  int d = d09(m, i + 1) ? 2 : 1;
  for (; d >= 1; d--) {
    int ch = -1, e = ampm_at(m, i + d, &ch);
    if (e < 0) continue;
    put_int_ascii(o, m->s + i, d);
    put_ampm(o, m->s[ch]);
    return e;
  }
  return -1;
}

/* text._ABBREV, applied one by one */
enum { A_TITLE, A_DOT, A_ETC, A_DOTNUM, A_MON, A_AFTER, A_SEPT };
typedef struct {
  const char* w;
  int kind;
  const char* out;
} abbrev;
static const abbrev ABBREV[] = {
    {"Dr", A_TITLE, "Doctor"},       {"Mrs", A_TITLE, "Missus"},   {"Mr", A_TITLE, "Mister"},
    {"Ms", A_TITLE, "Mizz"},         {"Prof", A_TITLE, "Professor"}, {"St", A_TITLE, "Saint"},
    {"Jr", A_DOT, "Junior"},         {"Sr", A_DOT, "Senior"},      {"vs", A_DOT, "versus"},
    {"e.g", A_DOT, "for example"},   {"i.e", A_DOT, "that is"},    {"etc", A_ETC, "et cetera"},
    {"etc", A_DOT, "et cetera."},    {"approx", A_DOT, "approximately"}, {"No", A_DOTNUM, "number"},
    {"Fig", A_DOTNUM, "figure"},     {"Vol", A_DOTNUM, "volume"},  {"pp", A_DOTNUM, "pages"},
    {"Jan", A_MON, "January "},      {"Feb", A_MON, "February "},  {"Mar", A_MON, "March "},
    {"Apr", A_MON, "April "},        {"Jun", A_MON, "June "},      {"Jul", A_MON, "July "},
    {"Aug", A_MON, "August "},       {"Sept", A_MON, "September "}, {"Sep", A_MON, "September "},
    {"Oct", A_MON, "October "},      {"Nov", A_MON, "November "},  {"Dec", A_MON, "December "},
    {"Jan", A_AFTER, "January"},     {"Feb", A_AFTER, "February"}, {"Mar", A_AFTER, "March"},
    {"Apr", A_AFTER, "April"},       {"Jun", A_AFTER, "June"},     {"Jul", A_AFTER, "July"},
    {"Aug", A_AFTER, "August"},      {"Sept", A_AFTER, "September"}, {"Sep", A_AFTER, "September"},
    {"Oct", A_AFTER, "October"},     {"Nov", A_AFTER, "November"}, {"Dec", A_AFTER, "December"},
    {"Jan", A_DOT, "January"},       {"Feb", A_DOT, "February"},   {"Mar", A_DOTNUM, "March"},
    {"Apr", A_DOT, "April"},         {"Jun", A_DOTNUM, "June"},    {"Jul", A_DOTNUM, "July"},
    {"Aug", A_DOT, "August"},        {"Sep", A_SEPT, "September"}, {"Oct", A_DOT, "October"},
    {"Nov", A_DOT, "November"},      {"Dec", A_DOT, "December"},
};
#define N_ABBREV ((int)(sizeof ABBREV / sizeof ABBREV[0]))

static int m_abbrev(M* m, int i, nfs* o, const void* ud) {
  const abbrev* a = (const abbrev*)ud;
  int L, k;
  if (a->kind == A_AFTER) {   /* (?<=[0-9] )W\b */
    if (!(d09(m, i - 2) && at(m, i - 1) == ' ')) return -1;
    if (!(L = lit(m, i, a->w)) || !bnd(m, i + L)) return -1;
    nfs_ascii(o, a->out);
    return i + L;
  }
  if (!bnd(m, i) || !(L = lit(m, i, a->w))) return -1;   /* \bW */
  k = i + L;
  switch (a->kind) {
    case A_TITLE: {   /* \.?(?=\s+[A-Z]) */
      int q = at(m, k) == '.' ? k + 1 : k;
      int r = spaces(m, q);
      if (r == q || !AZ(m, r)) return -1;
      nfs_ascii(o, a->out);
      return q;
    }
    case A_DOT:   /* \. */
      if (at(m, k) != '.') return -1;
      nfs_ascii(o, a->out);
      return k + 1;
    case A_SEPT:   /* t?\. */
      if (at(m, k) == 't') k++;
      if (at(m, k) != '.') return -1;
      nfs_ascii(o, a->out);
      return k + 1;
    case A_ETC: {   /* \.(?=\s*[,;:!?)]|\s+[a-z]) */
      if (at(m, k) != '.') return -1;
      int r = spaces(m, k + 1), ch = at(m, r);
      int ok = ch == ',' || ch == ';' || ch == ':' || ch == '!' || ch == '?' || ch == ')' || (r > k + 1 && az(m, r));
      if (!ok) return -1;
      nfs_ascii(o, a->out);
      return k + 1;
    }
    case A_DOTNUM: {   /* \.(?=\s*[0-9]) */
      if (at(m, k) != '.' || !d09(m, spaces(m, k + 1))) return -1;
      nfs_ascii(o, a->out);
      return k + 1;
    }
    case A_MON: {   /* (?:\.\s*|\s+)(?=[0-9]) */
      int e;
      if (at(m, k) == '.' && d09(m, (e = spaces(m, k + 1)))) {
      } else if ((e = spaces(m, k)) > k && d09(m, e)) {
      } else {
        return -1;
      }
      nfs_ascii(o, a->out);
      return e;
    }
  }
  return -1;
}

/* ([0-9])\s?°\s?([CF])(?![A-Za-z]) */
static int m_deg_cf(M* m, int i, nfs* o, const void* ud) {
  if (!d09(m, i)) return -1;
  int k = i + 1;
  if (iss(m, k) && at(m, k + 1) == 0xB0) k++;
  if (at(m, k) != 0xB0) return -1;
  k++;
  if (iss(m, k) && (at(m, k + 1) == 'C' || at(m, k + 1) == 'F')) k++;
  int u = at(m, k);
  if (!(u == 'C' || u == 'F') || alpha(m, k + 1)) return -1;
  nfs_push(o, m->s[i]);
  nfs_ascii(o, u == 'C' ? " degrees Celsius" : " degrees Fahrenheit");
  return k + 1;
}
/* ([0-9])\s?° */
static int m_deg(M* m, int i, nfs* o, const void* ud) {
  if (!d09(m, i)) return -1;
  int k = i + 1;
  if (iss(m, k) && at(m, k + 1) == 0xB0) k++;
  if (at(m, k) != 0xB0) return -1;
  nfs_push(o, m->s[i]);
  nfs_ascii(o, " degrees");
  return k + 1;
}
/* \b([A-Z])\s?&\s?([A-Z])\b */
static int m_letter_amp(M* m, int i, nfs* o, const void* ud) {
  if (!AZ(m, i) || !bnd(m, i)) return -1;
  int k = i + 1;
  if (iss(m, k) && at(m, k + 1) == '&') k++;
  if (at(m, k) != '&') return -1;
  k++;
  if (iss(m, k) && AZ(m, k + 1)) k++;
  if (!AZ(m, k) || !bnd(m, k + 1)) return -1;
  put_letter(o, m->s[i]);
  nfs_ascii(o, " and ");
  put_letter(o, m->s[k]);
  return k + 1;
}
/* \s*&\s* -> " and " */
static int m_amp(M* m, int i, nfs* o, const void* ud) {
  int k = spaces(m, i);
  if (at(m, k) != '&') return -1;
  nfs_ascii(o, " and ");
  return spaces(m, k + 1);
}
/* ~\s?(?=[0-9]) -> "about " */
static int m_tilde(M* m, int i, nfs* o, const void* ud) {
  if (at(m, i) != '~') return -1;
  int k = i + 1;
  if (iss(m, k) && d09(m, k + 1))
    k++;
  else if (!d09(m, k))
    return -1;
  nfs_ascii(o, "about ");
  return k;
}
static int cur_sym(int ch) { return ch == '$' || ch == 0xA3 || ch == 0x20AC; }
/* ([$£€]\s?[0-9]+(?:\.[0-9]+)?)\s?([KkMmBb])n?\b */
static int m_scale(M* m, int i, nfs* o, const void* ud) {
  if (!cur_sym(at(m, i))) return -1;
  int k = i + 1;
  if (iss(m, k) && d09(m, k + 1)) k++;
  if (!d09(m, k)) return -1;
  while (d09(m, k)) k++;
  if (at(m, k) == '.' && d09(m, k + 1)) {
    k++;
    while (d09(m, k)) k++;
  }
  int g1e = k;
  if (iss(m, k)) k++;
  int u = at(m, k);
  if (!(u == 'K' || u == 'k' || u == 'M' || u == 'm' || u == 'B' || u == 'b')) return -1;
  int e = k + 1;
  if (at(m, e) == 'n' && bnd(m, e + 1))
    e++;
  else if (!bnd(m, e))
    return -1;
  nfs_put(o, m->s + i, g1e - i);
  nfs_ascii(o, u == 'K' || u == 'k' ? " thousand" : u == 'M' || u == 'm' ? " million" : " billion");
  return e;
}
/* (?<![0-9])[0-9]+(?:\.[0-9]+){2,}(?![0-9]) -> "." as " point " */
static int m_version(M* m, int i, nfs* o, const void* ud) {
  if (!d09(m, i) || d09(m, i - 1)) return -1;
  int k = i;
  while (d09(m, k)) k++;
  int groups = 0;
  while (at(m, k) == '.' && d09(m, k + 1)) {
    k++;
    while (d09(m, k)) k++;
    groups++;
  }
  if (groups < 2) return -1;
  for (int t = i; t < k; t++) {
    if (m->s[t] == '.')
      nfs_ascii(o, " point ");
    else
      nfs_push(o, m->s[t]);
  }
  return k;
}

/* \b(Chapter|Part|...)\s+(?=[IVXLCDM])(M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\b */
static int run_of(const M* m, int i, int ch, int mx) {
  int k = 0;
  while (k < mx && at(m, i + k) == ch) k++;
  return k;
}
/* one roman digit group (?:A9|A4|A5?A1{0,3}) at p: the alternatives' end positions in regex order */
static int roman_alts(const M* m, int p, int one, int five, int ten, int* ends) {
  int n = 0;
  if (at(m, p) == one && at(m, p + 1) == ten) ends[n++] = p + 2;
  if (at(m, p) == one && at(m, p + 1) == five) ends[n++] = p + 2;
  if (at(m, p) == five) {
    int r = run_of(m, p + 1, one, 3);
    for (int k = r; k >= 0; k--) ends[n++] = p + 1 + k;
  }
  int r = run_of(m, p, one, 3);
  for (int k = r; k >= 0; k--) ends[n++] = p + k;
  return n;
}
static int roman_val(int ch) {
  switch (ch) {
    case 'I': return 1;
    case 'V': return 5;
    case 'X': return 10;
    case 'L': return 50;
    case 'C': return 100;
    case 'D': return 500;
    default: return 1000;
  }
}
static int m_roman(M* m, int i, nfs* o, const void* ud) {
  static const char* W[] = {"Chapter", "Part", "Book", "Volume", "Act", "Scene", "Section", "Article", "Phase",
                            "Season", "Episode", "Level", "Stage", "Round"};
  if (!bnd(m, i)) return -1;
  int L = 0;
  for (int k = 0; k < 14 && !L; k++) L = lit(m, i, W[k]);
  if (!L) return -1;
  int s0 = i + L, r0 = spaces(m, s0);
  if (r0 == s0) return -1;
  int c0 = at(m, r0);
  if (!(c0 == 'I' || c0 == 'V' || c0 == 'X' || c0 == 'L' || c0 == 'C' || c0 == 'D' || c0 == 'M')) return -1;
  int mm = run_of(m, r0, 'M', 3);
  for (int a = mm; a >= 0; a--) {
    int e1[8], n1 = roman_alts(m, r0 + a, 'C', 'D', 'M', e1);
    for (int b = 0; b < n1; b++) {
      int e2[8], n2 = roman_alts(m, e1[b], 'X', 'L', 'C', e2);
      for (int cc = 0; cc < n2; cc++) {
        int e3[8], n3 = roman_alts(m, e2[cc], 'I', 'V', 'X', e3);
        for (int d = 0; d < n3; d++) {
          int e = e3[d];
          if (!bnd(m, e)) continue;
          int v = 0;
          for (int t = r0; t < e; t++) {
            int x = roman_val(at(m, t));
            v += (t + 1 < e && x < roman_val(at(m, t + 1))) ? -x : x;
          }
          char b2[16];
          snprintf(b2, sizeof b2, " %d", v);
          nfs_put(o, m->s + i, L);
          nfs_ascii(o, b2);
          return e;
        }
      }
    }
  }
  return -1;
}
/* \bWW(II|I|1|2)\b */
static int m_ww(M* m, int i, nfs* o, const void* ud) {
  if (!bnd(m, i) || !lit(m, i, "WW")) return -1;
  static const char* alt[] = {"II", "I", "1", "2"};
  for (int k = 0; k < 4; k++) {
    int L = lit(m, i + 2, alt[k]);
    if (!L || !bnd(m, i + 2 + L)) continue;
    nfs_ascii(o, (L == 2 || alt[k][0] == '2') ? "World War 2" : "World War 1");
    return i + 2 + L;
  }
  return -1;
}
/* (?<![0-9/])([1-9])/(10|[2-9])(?![0-9/]) */
static int m_fraction(M* m, int i, nfs* o, const void* ud) {
  static const char* one[] = {"", "", "half", "third", "quarter", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"};
  static const char* many[] = {"", "", "halves", "thirds", "quarters", "fifths", "sixths", "sevenths", "eighths",
                               "ninths", "tenths"};
  if (d09(m, i - 1) || at(m, i - 1) == '/') return -1;
  int nu = at(m, i);
  if (!(nu >= '1' && nu <= '9') || at(m, i + 1) != '/') return -1;
  int k, e;
  if (lit(m, i + 2, "10") && !(d09(m, i + 4) || at(m, i + 4) == '/')) {
    k = 10;
    e = i + 4;
  } else if (at(m, i + 2) >= '2' && at(m, i + 2) <= '9' && !(d09(m, i + 3) || at(m, i + 3) == '/')) {
    k = at(m, i + 2) - '0';
    e = i + 3;
  } else {
    return -1;
  }
  nfs_push(o, (u32)nu);
  nfs_push(o, ' ');
  nfs_ascii(o, nu == '1' ? one[k] : many[k]);
  return e;
}
static const char* tens_plural(int d) {
  static const char* t[] = {"", "tens", "twenties", "thirties", "forties", "fifties", "sixties", "seventies",
                            "eighties", "nineties"};
  return t[d - '0'];
}
/* '?s\b after p: end or -1 */
static int qs_end(const M* m, int p) {
  if (at(m, p) == '\'' && at(m, p + 1) == 's' && bnd(m, p + 2)) return p + 2;
  if (at(m, p) == 's' && bnd(m, p + 1)) return p + 1;
  return -1;
}
/* \b(1[1-9]|20)([0-9])0'?s\b */
static int m_decade4(M* m, int i, nfs* o, const void* ud) {
  if (!bnd(m, i)) return -1;
  int a = at(m, i), b = at(m, i + 1);
  if (!((a == '1' && b >= '1' && b <= '9') || (a == '2' && b == '0'))) return -1;
  if (!d09(m, i + 2) || at(m, i + 3) != '0') return -1;
  int e = qs_end(m, i + 4);
  if (e < 0) return -1;
  int t = at(m, i + 2);
  if (t != '0') {
    nfs_put(o, m->s + i, 2);
    nfs_push(o, ' ');
    nfs_ascii(o, tens_plural(t));
  } else if (a == '2' && b == '0') {
    nfs_ascii(o, "two thousands");
  } else {
    nfs_put(o, m->s + i, 2);
    nfs_ascii(o, " hundreds");
  }
  return e;
}
/* (?<![0-9A-Za-z])'?([1-9])0'?s\b */
static int m_decade2(M* m, int i, nfs* o, const void* ud) {
  if (alnum(m, i - 1)) return -1;
  int k = at(m, i) == '\'' ? i + 1 : i;
  int d = at(m, k);
  if (!(d >= '1' && d <= '9') || at(m, k + 1) != '0') return -1;
  int e = qs_end(m, k + 2);
  if (e < 0) return -1;
  nfs_ascii(o, tens_plural(d));
  return e;
}

/* ------------------------------------------------------------------ text._expand */
/* (\b(?:[Nn]umber|[Cc]all|[Pp]hone|[Tt]el|[Dd]ial|[Ff]ax)\.?:?\s+)?(?<![0-9.,-])([0-9]+(?:-[0-9]+)+)(?![0-9-]|[.,][0-9]) */
static int dgroups(const M* m, int p) {   /* [0-9]+(?:-[0-9]+)+ with the lookahead: end or -1 */
  if (!d09(m, p)) return -1;
  int k = p, ng = 0;
  while (d09(m, k)) k++;
  while (at(m, k) == '-' && d09(m, k + 1)) {
    k++;
    while (d09(m, k)) k++;
    ng++;
  }
  if (!ng) return -1;
  if (d09(m, k) || at(m, k) == '-' || ((at(m, k) == '.' || at(m, k) == ',') && d09(m, k + 1))) return -1;
  return k;
}
static int lb_ok(const M* m, int p) {
  int ch = at(m, p - 1);
  return !(d09(m, p - 1) || ch == '.' || ch == ',' || ch == '-');
}
static void id_digits(const M* m, int g1s, int g1e, int a, int b, nfs* o) {
  const u32* s = m->s;
  int ng = 1, lead0 = 0, len[3] = {0, 0, 0}, gl = 0;
  for (int i = a; i < b; ++i) {
    if (s[i] == '-') {
      if (ng <= 3) len[ng - 1] = gl;
      ++ng;
      gl = 0;
    } else {
      if (gl == 0 && s[i] == '0' && i + 1 < b && s[i + 1] != '-') lead0 = 1;
      ++gl;
    }
  }
  if (ng <= 3) len[ng - 1] = gl;
  int date = ng == 3 && ((len[2] == 4 && len[0] <= 2 && len[1] <= 2) || (len[0] == 4 && len[1] <= 2 && len[2] <= 2));
  if (!(g1s >= 0 || (!date && lead0))) {
    nfs_put(o, s + (g1s >= 0 ? g1s : a), b - (g1s >= 0 ? g1s : a));
    return;
  }
  if (g1s >= 0) nfs_put(o, s + g1s, g1e - g1s);
  for (int i = a; i < b; ++i) {
    if (s[i] == '-') {
      nfs_ascii(o, ", ");
    } else {
      if (i > a && s[i - 1] != '-') nfs_push(o, ' ');
      nfs_push(o, s[i]);
    }
  }
}
static int m_digit_groups(M* m, int i, nfs* o, const void* ud) {
  static const char* W[] = {"umber", "all", "hone", "el", "ial", "ax"};
  static const char F[] = {'n', 'c', 'p', 't', 'd', 'f'};
  if (bnd(m, i)) {
    int ch = at(m, i), L = 0;
    for (int k = 0; k < 6 && !L; k++)
      if (ch == F[k] || ch == F[k] - 32) L = lit(m, i + 1, W[k]) ? 1 + (int)strlen(W[k]) : 0;
    if (L) {
      int k = i + L;
      if (at(m, k) == '.') k++;
      if (at(m, k) == ':') k++;
      int r = spaces(m, k);
      if (r > k && lb_ok(m, r)) {
        int e = dgroups(m, r);
        if (e > 0) {
          id_digits(m, i, r, r, e, o);
          return e;
        }
      }
    }
  }
  if (!lb_ok(m, i)) return -1;
  int e = dgroups(m, i);
  if (e < 0) return -1;
  id_digits(m, -1, -1, i, e, o);
  return e;
}
/* (\d)\s*[-–]\s*(\d) -> \1 to \2 */
static int m_range(M* m, int i, nfs* o, const void* ud) {
  if (!isd(m, i)) return -1;
  int k = spaces(m, i + 1), ch = at(m, k);
  if (!(ch == '-' || ch == 0x2013)) return -1;
  k = spaces(m, k + 1);
  if (!isd(m, k)) return -1;
  nfs_push(o, m->s[i]);
  nfs_ascii(o, " to ");
  nfs_push(o, m->s[k]);
  return k + 1;
}
/* (\d+)\s*[x×]\s*(\d+) -> \1 by \2 */
static int m_by(M* m, int i, nfs* o, const void* ud) {
  if (!isd(m, i)) return -1;
  int a = i;
  while (isd(m, a)) a++;
  int k = spaces(m, a), ch = at(m, k);
  if (!(ch == 'x' || ch == 0xD7)) return -1;
  int b = spaces(m, k + 1);
  if (!isd(m, b)) return -1;
  int e = b;
  while (isd(m, e)) e++;
  nfs_put(o, m->s + i, a - i);
  nfs_ascii(o, " by ");
  nfs_put(o, m->s + b, e - b);
  return e;
}
/* (?<![^\s(])([+−-])(?=[0-9]) */
static int m_sign(M* m, int i, nfs* o, const void* ud) {
  int ch = at(m, i);
  if (!(ch == '+' || ch == 0x2212 || ch == '-') || !d09(m, i + 1)) return -1;
  if (i > 0 && !(iss(m, i - 1) || at(m, i - 1) == '(')) return -1;
  nfs_ascii(o, ch == '+' ? "plus " : "minus ");
  return i + 1;
}
/* (?<=[A-Za-z0-9])[-‐/](?=[A-Za-z0-9]) -> " " */
static int m_compound(M* m, int i, nfs* o, const void* ud) {
  int ch = at(m, i);
  if (!(ch == '-' || ch == 0x2010 || ch == '/') || !alnum(m, i - 1) || !alnum(m, i + 1)) return -1;
  nfs_push(o, ' ');
  return i + 1;
}
/* ([$£€])\s?(\d(?:[\d,]*\d)?(?:\.\d+)?)(?:\s*(million|billion|thousand)\b)? */
static int m_currency(M* m, int i, nfs* o, const void* ud) {
  int cs = at(m, i);
  if (!cur_sym(cs)) return -1;
  int k = i + 1;
  if (iss(m, k) && isd(m, k + 1)) k++;
  if (!isd(m, k)) return -1;
  int g2 = k, e = k + 1, r = e;
  while (isd(m, r) || at(m, r) == ',') r++;
  while (r > e && !isd(m, r - 1)) r--;
  e = r;
  if (at(m, e) == '.' && isd(m, e + 1)) {
    e++;
    while (isd(m, e)) e++;
  }
  int g2e = e, g3 = -1, g3e = -1;
  int q = spaces(m, e);
  static const char* SC[] = {"million", "billion", "thousand"};
  for (int t = 0; t < 3; t++) {
    int L = lit(m, q, SC[t]);
    if (L && bnd(m, q + L)) {
      g3 = q;
      g3e = q + L;
      break;
    }
  }
  nfs_put(o, m->s + g2, g2e - g2);
  nfs_push(o, ' ');
  if (g3 >= 0) {
    nfs_put(o, m->s + g3, g3e - g3);
    nfs_push(o, ' ');
    e = g3e;
  }
  nfs_ascii(o, cs == '$' ? "dollars" : cs == 0xA3 ? "pounds" : "euros");
  return e;
}
/* (\d[\d,]*(?:\.\d+)?)\s?% -> \1 percent */
static int m_percent(M* m, int i, nfs* o, const void* ud) {
  if (!isd(m, i)) return -1;
  int k = i + 1;
  while (isd(m, k) || at(m, k) == ',') k++;
  int ends[2], ne = 0;
  if (at(m, k) == '.' && isd(m, k + 1)) {
    int d = k + 1;
    while (isd(m, d)) d++;
    ends[ne++] = d;
  }
  ends[ne++] = k;
  for (int t = 0; t < ne; t++) {
    int g = ends[t], p = g;
    if (iss(m, p) && at(m, p + 1) == '%') p++;
    if (at(m, p) != '%') continue;
    nfs_put(o, m->s + i, g - i);
    nfs_ascii(o, " percent");
    return p + 1;
  }
  return -1;
}
/* (\d)\s?UNIT\b, "/" in a unit matching [ /]; units longest first */
static const char* UNITS[][2] = {{"km/hr", "kilometers per hour"}, {"km/h", "kilometers per hour"},
                                 {"m/s", "meters per second"},    {"mph", "miles per hour"},
                                 {"lbs", "pounds"},               {"kHz", "kilohertz"},
                                 {"kph", "kilometers per hour"},  {"km", "kilometers"},
                                 {"kg", "kilograms"},             {"cm", "centimeters"},
                                 {"mm", "millimeters"},           {"ft", "feet"},
                                 {"lb", "pounds"},                {"Hz", "hertz"},
                                 {"GB", "gigabytes"},             {"MB", "megabytes"}};
static int unit_lit(const M* m, int p, const char* u) {
  int k = 0;
  for (; u[k]; k++) {
    int ch = at(m, p + k);
    if (u[k] == '/' ? !(ch == ' ' || ch == '/') : ch != (unsigned char)u[k]) return 0;
  }
  return k;
}
static int m_unit(M* m, int i, nfs* o, const void* ud) {
  const char* const* u = (const char* const*)ud;
  if (!isd(m, i)) return -1;
  for (int sp = iss(m, i + 1) ? 1 : 0; sp >= 0; sp--) {
    int k = i + 1 + sp, L = unit_lit(m, k, u[0]);
    if (!L || !bnd(m, k + L)) continue;
    nfs_push(o, m->s[i]);
    nfs_push(o, ' ');
    nfs_ascii(o, u[1]);
    return k + L;
  }
  return -1;
}
/* \b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*[0-9])[A-Z0-9]+\b : Q4 -> Q 4, MP3 -> M P 3 */
static int m_letter_digit(M* m, int i, nfs* o, const void* ud) {
  if (!(AZ(m, i) || d09(m, i)) || !bnd(m, i)) return -1;
  int k = i, L = 0, D = 0;
  while (AZ(m, k) || d09(m, k)) {
    if (AZ(m, k)) L = 1; else D = 1;
    k++;
  }
  if (!L || !D || !bnd(m, k)) return -1;
  for (int t = i; t < k; t++) {
    int dig = d09(m, t);
    if (t > i && !(dig && d09(m, t - 1))) nfs_push(o, ' ');
    if (dig)
      nfs_push(o, m->s[t]);
    else
      put_letter(o, m->s[t]);
  }
  return k;
}
/* \b(\d+)(st|nd|rd|th)\b */
static int m_ordinal(M* m, int i, nfs* o, const void* ud) {
  if (!isd(m, i) || !bnd(m, i)) return -1;
  int k = i;
  while (isd(m, k)) k++;
  if (!(lit(m, k, "st") || lit(m, k, "nd") || lit(m, k, "rd") || lit(m, k, "th")) || !bnd(m, k + 2)) return -1;
  num_int(m->c, m->s + i, k - i, 1, o);
  return k + 2;
}
/* \b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b */
static int m_comma_num(M* m, int i, nfs* o, const void* ud) {
  if (!isd(m, i) || !bnd(m, i)) return -1;
  int run = 0;
  while (run < 3 && isd(m, i + run)) run++;
  for (int d = run; d >= 1; d--) {
    int ge[64], ng = 0, k = i + d;   /* ends after 1, 2, ... groups */
    while (ng < 64 && at(m, k) == ',' && isd(m, k + 1) && isd(m, k + 2) && isd(m, k + 3)) {
      k += 4;
      ge[ng++] = k;
    }
    for (int g = ng; g >= 1; g--) {
      int p = ge[g - 1], ends[2], ne = 0;
      if (at(m, p) == '.' && isd(m, p + 1)) {
        int q = p + 1;
        while (isd(m, q)) q++;
        ends[ne++] = q;
      }
      ends[ne++] = p;
      for (int t = 0; t < ne; t++) {
        int e = ends[t];
        if (!bnd(m, e)) continue;
        u32 nc[128];
        int nn = 0, dot = 0;
        if (e - i > 128) {
          nf_fail(m->c, "ValueError: text too long for the device front end");
          return -1;
        }
        for (int x = i; x < e; x++) {
          if (m->s[x] != ',') nc[nn++] = m->s[x];
          if (m->s[x] == '.') dot = 1;
        }
        if (dot)
          num_float(m->c, nc, nn, o);
        else
          num_int(m->c, nc, nn, 0, o);
        return e;
      }
    }
  }
  return -1;
}
/* \b\d+\.\d+\b */
static int m_decimal(M* m, int i, nfs* o, const void* ud) {
  if (!isd(m, i) || !bnd(m, i)) return -1;
  int k = i;
  while (isd(m, k)) k++;
  if (at(m, k) != '.' || !isd(m, k + 1)) return -1;
  k++;
  while (isd(m, k)) k++;
  if (!bnd(m, k)) return -1;
  num_float(m->c, m->s + i, k - i, o);
  return k;
}
/* \b\d{5,}\b */
static int m_long_digits(M* m, int i, nfs* o, const void* ud) {
  if (!isd(m, i) || !bnd(m, i)) return -1;
  int k = i;
  while (isd(m, k)) k++;
  if (k - i < 5 || !bnd(m, k)) return -1;
  num_int(m->c, m->s + i, k - i, 0, o);
  return k;
}
/* [A-Z]{2,5}s?\b at i: end, trying the regex's backtracking order */
static int acr_end(const M* m, int i) {
  int r = 0;
  while (r < 5 && AZ(m, i + r)) r++;
  for (int k = r; k >= 2; k--) {
    if (at(m, i + k) == 's' && bnd(m, i + k + 1)) return i + k + 1;
    if (bnd(m, i + k)) return i + k;
  }
  return -1;
}
/* (?<=[a-z])([A-Z]{2,5}s?)\b -> " \1" */
static int m_camel_acr(M* m, int i, nfs* o, const void* ud) {
  if (!az(m, i - 1)) return -1;
  int e = acr_end(m, i);
  if (e < 0) return -1;
  nfs_push(o, ' ');
  nfs_put(o, m->s + i, e - i);
  return e;
}
/* \b[A-Z]{2,5}s?\b : a known word stays, else spelled */
static int m_acronym(M* m, int i, nfs* o, const void* ud) {
  if (!bnd(m, i)) return -1;
  int e = acr_end(m, i);
  if (e < 0) return -1;
  char g[8], low[8];
  int n = e - i;
  for (int t = 0; t < n; ++t) {
    g[t] = (char)m->s[i + t];
    low[t] = g[t] >= 'A' && g[t] <= 'Z' ? (char)(g[t] + 32) : g[t];
  }
  g[n] = low[n] = 0;
  if (nf_in_cmu(m->c, low)) {
    nfs_ascii(o, g);
    return e;
  }
  int b = n;
  while (b > 0 && g[b - 1] == 's') --b;
  for (int t = 0; t < b; ++t) {
    if (t) nfs_push(o, ' ');
    put_letter(o, (unsigned char)g[t]);
  }
  if (g[n - 1] == 's' && n > 2) nfs_ascii(o, "'s");
  return e;
}
/* (?<=[a-z])-(?=[a-z]) -> " " */
static int m_num_hyphen(M* m, int i, nfs* o, const void* ud) {
  if (at(m, i) != '-' || !az(m, i - 1) || !az(m, i + 1)) return -1;
  nfs_push(o, ' ');
  return i + 1;
}
/* \b(thousand|million|billion|trillion),(?= [a-z]) -> \1 */
static int m_num_comma(M* m, int i, nfs* o, const void* ud) {
  if (!bnd(m, i)) return -1;
  static const char* W[] = {"thousand", "million", "billion", "trillion"};
  for (int k = 0; k < 4; k++) {
    int L = lit(m, i, W[k]);
    if (L && at(m, i + L) == ',' && at(m, i + L + 1) == ' ' && az(m, i + L + 2)) {
      nfs_put(o, m->s + i, L);
      return i + L + 1;
    }
  }
  return -1;
}
/* — – … */
static int m_dashes(M* m, int i, nfs* o, const void* ud) {
  int ch = at(m, i);
  if (ch == 0x2014 || ch == 0x2013) {
    nfs_ascii(o, ", ");
    return i + 1;
  }
  return -1;
}
static int m_ellipsis(M* m, int i, nfs* o, const void* ud) {
  if (at(m, i) != 0x2026) return -1;
  nfs_ascii(o, ". ");
  return i + 1;
}

/* ------------------------------------------------------------------ driver */
typedef struct {
  matcher f;
  const void* ud;
} pass;

void nf_collapse_strip(nano_front_ctx* c, nfs* t) {
  int o = 0, i = 0, n = t->n;
  while (i < n) {
    if (nf_is_space(c, t->p[i])) {
      while (i < n && nf_is_space(c, t->p[i])) i++;
      t->p[o++] = ' ';
    } else {
      t->p[o++] = t->p[i++];
    }
  }
  int a = 0, b = o;
  while (a < b && t->p[a] == ' ') a++;
  while (b > a && t->p[b - 1] == ' ') b--;
  memmove(t->p, t->p + a, (size_t)(b - a) * sizeof(u32));
  t->n = b - a;
}

/* normalise s into out; the two context buffers ping-pong (s may be anywhere, out must be c->a) */
void nf_normalize(nano_front_ctx* c, const u32* s, int n, nfs* out) {
  nfs X = {c->a, 0, NF_MAX_CP, c}, Y = {c->b, 0, NF_MAX_CP, c};
  nfs src = {(u32*)s, n, n, c};
  sub(c, &src, &Y, m_quote, NULL);   /* s may alias c->a: read it once into Y */
  nfs *cur = &Y, *nxt = &X;
#define RUN(f, ud)                       \
  do {                                   \
    sub(c, cur, nxt, (f), (ud));         \
    nfs* t_ = cur;                       \
    cur = nxt;                           \
    nxt = t_;                            \
  } while (0)
  RUN(m_phone, NULL);
  RUN(m_time, NULL);
  RUN(m_colon, NULL);
  RUN(m_ampm, NULL);
  for (int k = 0; k < N_ABBREV; k++) RUN(m_abbrev, &ABBREV[k]);
  RUN(m_deg_cf, NULL);
  RUN(m_deg, NULL);
  RUN(m_letter_amp, NULL);
  RUN(m_amp, NULL);
  RUN(m_tilde, NULL);
  RUN(m_scale, NULL);
  RUN(m_version, NULL);
  RUN(m_roman, NULL);
  RUN(m_ww, NULL);
  RUN(m_fraction, NULL);
  RUN(m_decade4, NULL);
  RUN(m_decade2, NULL);
  RUN(m_digit_groups, NULL);
  RUN(m_range, NULL);
  RUN(m_by, NULL);
  RUN(m_sign, NULL);
  RUN(m_compound, NULL);
  RUN(m_currency, NULL);
  RUN(m_percent, NULL);
  for (int k = 0; k < 16; k++) RUN(m_unit, UNITS[k]);
  RUN(m_letter_digit, NULL);
  RUN(m_ordinal, NULL);
  RUN(m_comma_num, NULL);
  RUN(m_decimal, NULL);
  RUN(m_long_digits, NULL);
  RUN(m_camel_acr, NULL);
  RUN(m_acronym, NULL);
  nf_normalize_numbers2(c, cur, nxt);   /* g2p_en's number pass, result back in cur */
  RUN(m_num_hyphen, NULL);
  RUN(m_num_comma, NULL);
  RUN(m_dashes, NULL);
  RUN(m_ellipsis, NULL);
#undef RUN
  nf_collapse_strip(c, cur);
  if (cur != out) {
    memcpy(out->p, cur->p, (size_t)cur->n * sizeof(u32));
    out->n = cur->n;
  }
}
