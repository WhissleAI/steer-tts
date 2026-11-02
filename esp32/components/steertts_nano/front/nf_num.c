/* Number words: num2words cardinal / ordinal / float (the edge front end's subset), inflect.number_to_words and
 * g2p_en's normalize_numbers, with Python's exceptions where the reference raises. ASCII strings, fixed buffers. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "nf_int.h"

#define NW 640   /* number word buffers: a longer result would not fit the sentence buffer anyway */

static const char* ONES[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                             "eighteen", "nineteen"};
static const char* TENS[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
static const char* HIGH[] = {"m", "b", "tr", "quadr", "quint", "sext", "sept", "oct", "non", "dec", "undec", "duodec",
                             "tredec", "quattuordec", "quindec", "sexdec", "septendec", "octodec", "novemdec",
                             "vigint", "unvigint", "duovigint", "trevigint", "quattuorvigint", "quinvigint",
                             "sexvigint", "septenvigint", "octovigint", "novemvigint", "trigint", "untrigint",
                             "duotrigint"};

static int is_sp(char c) { return c == ' ' || (c >= 9 && c <= 13) || (c >= 28 && c <= 31); }

static void below100(nfa* o, int n) {
  if (n < 20) {
    nfa_cat(o, ONES[n]);
    return;
  }
  nfa_cat(o, TENS[n / 10]);
  if (n % 10) {
    nfa_ch(o, '-');
    nfa_cat(o, ONES[n % 10]);
  }
}
static void below1000(nfa* o, int n) {
  int h = n / 100, r = n % 100;
  if (!h) {
    below100(o, r);
    return;
  }
  nfa_cat(o, ONES[h]);
  nfa_cat(o, " hundred");
  if (r) {
    nfa_cat(o, " and ");
    below100(o, r);
  }
}

double nf_strtod(const char* a) { return strtod(a, NULL); }   /* the device C library ignores locales */

static void double_int_digits(double v, nfa* o) {
  char b[24];
  if (v < 9.2e18) {
    snprintf(b, sizeof b, "%llu", (unsigned long long)v);
    nfa_cat(o, b);
    return;
  }
  int e;
  double m = frexp(v, &e);
  unsigned long long mant = (unsigned long long)ldexp(m, 53);
  e -= 53;
  uint32_t big[40];
  int nb = 0;
  while (mant) {
    big[nb++] = (uint32_t)(mant % 1000000000u);
    mant /= 1000000000u;
  }
  for (int k = 0; k < e && nb < 40; ++k) {
    unsigned long long carry = 0;
    for (int j = 0; j < nb; ++j) {
      unsigned long long t = (unsigned long long)big[j] * 2 + carry;
      big[j] = (uint32_t)(t % 1000000000u);
      carry = t / 1000000000u;
    }
    if (carry && nb < 40) big[nb++] = (uint32_t)carry;
  }
  snprintf(b, sizeof b, "%u", (unsigned)big[nb - 1]);
  nfa_cat(o, b);
  for (int k = nb - 1; k-- > 0;) {
    snprintf(b, sizeof b, "%09u", (unsigned)big[k]);
    nfa_cat(o, b);
  }
}

void nf_py_int(nano_front_ctx* c, const u32* s, int n, nfa* out) {
  for (int i = 0; i < n; ++i) {
    if (nf_digit(c, s[i]) < 0) {
      char lit[220];
      int k = nf_utf8_encode(s, n < 50 ? n : 50, lit, (int)sizeof lit);
      if (k < 0) lit[0] = 0;
      nf_fail(c, "ValueError: invalid literal for int() with base 10: '%s'", lit);
      return;
    }
  }
  if (n > 4300) {
    nf_fail(c, "ValueError: Exceeds the limit (4300 digits) for integer string conversion");
    return;
  }
  int start = out->n;
  for (int i = 0; i < n; ++i) {
    int d = nf_digit(c, s[i]);
    if (out->n == start && d == 0) continue;
    nfa_ch(out, (char)('0' + d));
  }
  if (out->n == start) nfa_ch(out, '0');
}

void nf_cardinal(nano_front_ctx* c, const char* nd, nfa* out) {
  int n = (int)strlen(nd);
  if (n > 102) {
    nf_fail(c, "OverflowError: abs(%.60s...) must be less than 10**102.", nd);
    return;
  }
  if (!strcmp(nd, "0")) {
    nfa_cat(out, "zero");
    return;
  }
  int g[35], ng = 0;
  for (int end = n; end > 0; end -= 3) {
    int st = end >= 3 ? end - 3 : 0, v = 0;
    for (int i = st; i < end; ++i) v = v * 10 + (nd[i] - '0');
    g[ng++] = v;
  }
  int first = 1;
  for (int k = ng; k-- > 0;) {
    if (!g[k]) continue;
    if (!first) nfa_cat(out, (k == 0 && g[0] < 100) ? " and " : ", ");
    below1000(out, g[k]);
    if (k) {
      nfa_ch(out, ' ');
      if (k == 1) {
        nfa_cat(out, "thousand");
      } else {
        nfa_cat(out, HIGH[k - 2]);
        nfa_cat(out, "illion");
      }
    }
    first = 0;
  }
}

void nf_ordinal(nano_front_ctx* c, const char* nd, nfa* out) {
  char cb[NW];
  nfa cs = {cb, 0, (int)sizeof cb, c};
  cb[0] = 0;
  nf_cardinal(c, nd, &cs);
  if (c->fail) return;
  const char* sp = strrchr(cb, ' ');
  const char* last = sp ? sp + 1 : cb;
  const char* hy = strrchr(last, '-');
  const char* w = hy ? hy + 1 : last;
  nfa_put(out, cb, (int)(w - cb));
  static const char* ORD[][2] = {{"one", "first"}, {"two", "second"}, {"three", "third"}, {"four", "fourth"},
                                 {"five", "fifth"}, {"six", "sixth"}, {"seven", "seventh"}, {"eight", "eighth"},
                                 {"nine", "ninth"}, {"ten", "tenth"}, {"eleven", "eleventh"}, {"twelve", "twelfth"}};
  for (size_t k = 0; k < sizeof ORD / sizeof ORD[0]; ++k)
    if (!strcmp(w, ORD[k][0])) {
      nfa_cat(out, ORD[k][1]);
      return;
    }
  int wl = (int)strlen(w);
  if (wl && w[wl - 1] == 'y') {
    nfa_put(out, w, wl - 1);
    nfa_cat(out, "ie");
  } else {
    nfa_put(out, w, wl);
  }
  nfa_cat(out, "th");
}

void nf_cardinal_float(nano_front_ctx* c, double value, nfa* out) {
  if (isinf(value)) {
    nf_fail(c, "OverflowError: cannot convert float infinity to integer");
    return;
  }
  if (isnan(value)) {
    nf_fail(c, "ValueError: cannot convert float NaN to integer");
    return;
  }
  if (fabs(value) >= 1e102) {   /* num2words raises; also keeps the digit buffers small */
    nf_fail(c, "OverflowError: abs(%.0f) must be less than 10**102.", fabs(value));
    return;
  }
  char db[120];
  nfa d = {db, 0, (int)sizeof db, c};
  db[0] = 0;
  if (value == trunc(value)) {
    double_int_digits(fabs(value), &d);
    if (value < 0) nfa_cat(out, "minus ");
    nf_cardinal(c, db, out);
    return;
  }
  double pre = trunc(value);
  char buf[64];
  for (int p = 0; p < 17; ++p) {
    snprintf(buf, sizeof buf, "%.*e", p, value);
    if (strtod(buf, NULL) == value) break;
  }
  const char* e = strchr(buf, 'e');
  int k = 0;
  for (const char* q = buf; q < e; ++q)
    if (*q >= '0' && *q <= '9') ++k;
  int ex = atoi(e + 1);
  int precision = abs(k - 1 - ex);
  if (precision > 308) {
    nf_fail(c, "OverflowError: int too large to convert to float");
    return;
  }
  char p10s[16];
  snprintf(p10s, sizeof p10s, "1e%d", precision);
  double p10 = strtod(p10s, NULL);
  double post = fabs(value - pre) * p10;
  double rp = nearbyint(post);
  double pi = fabs(rp - post) < 0.01 ? rp : floor(post);
  char pb[340];
  nfa ps = {pb, 0, (int)sizeof pb, c};
  pb[0] = 0;
  double_int_digits(pi, &ps);
  if (fabs(pre) == 0) {
    nfa_cat(out, "zero");
  } else {
    if (pre < 0) nfa_cat(out, "minus ");
    double_int_digits(fabs(pre), &d);
    nf_cardinal(c, db, out);
  }
  if (precision) nfa_cat(out, " point");
  int pad = precision > ps.n ? precision - ps.n : 0;
  for (int i = 0; i < precision && !c->fail; ++i) {
    char one[2] = {i < pad ? '0' : pb[i - pad], 0};
    nfa_ch(out, ' ');
    nf_cardinal(c, one, out);
  }
}

/* ------------------------------------------------------------------ inflect.number_to_words subset */
static const char* IONES[] = {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
static const char* ITEENS[] = {"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                               "eighteen", "nineteen"};
static const char* IMILL[] = {" ", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion",
                              " sextillion", " septillion", " octillion", " nonillion", " decillion"};
static void imill(nano_front_ctx* c, int k, nfa* o) {
  if (k < 0 || k >= 12) {
    nf_fail(c, "IndexError: list index out of range");
    return;
  }
  nfa_cat(o, IMILL[k]);
}
static void i_tenfn(nano_front_ctx* c, int tens, int units, int mindex, nfa* o) {
  if (tens != 1) {
    if (tens) nfa_cat(o, TENS[tens]);
    if (tens && units) nfa_ch(o, '-');
    nfa_cat(o, IONES[units]);
    if (tens || units) imill(c, mindex, o);
    return;
  }
  nfa_cat(o, ITEENS[units]);
  imill(c, mindex, o);
}
static void i_hundfn(nano_front_ctx* c, int h, int t, int u, int mindex, const char* andword, nfa* o) {
  if (h) {
    nfa_cat(o, IONES[h]);
    nfa_cat(o, " hundred");
    if (t || u) nfa_cat(o, andword);
    i_tenfn(c, t, u, 0, o);
    imill(c, mindex, o);
    nfa_cat(o, ", ");
  } else if (t || u) {
    i_tenfn(c, t, u, 0, o);
    imill(c, mindex, o);
    nfa_cat(o, ", ");
  }
}

/* inflect _enword (group 0 or 2) on an ASCII digit string */
static void i_enword(nano_front_ctx* c, const char* num, int group, const char* zero, const char* one,
                     const char* andword, nfa* out) {
  if (group == 2) {
    /* re.sub(r"(\d)(\d)", pair, num) then re.sub(r"(\d)$", last, ...) */
    int n = (int)strlen(num), i = 0;
    char tb[NW];
    nfa t = {tb, 0, (int)sizeof tb, c};
    tb[0] = 0;
    while (i < n) {
      if (i + 1 < n && num[i] >= '0' && num[i] <= '9' && num[i + 1] >= '0' && num[i + 1] <= '9') {
        int a = num[i] - '0', b = num[i + 1] - '0';
        if (a) {
          i_tenfn(c, a, b, 0, &t);
        } else {
          nfa_cat(&t, zero);
          nfa_ch(&t, ' ');
          nfa_cat(&t, b ? IONES[b] : zero);
        }
        nfa_cat(&t, ", ");
        i += 2;
      } else {
        nfa_ch(&t, num[i++]);
      }
    }
    if (t.n && tb[t.n - 1] >= '0' && tb[t.n - 1] <= '9') {
      int a = tb[t.n - 1] - '0';
      t.n--;
      tb[t.n] = 0;
      nfa_cat(&t, a ? IONES[a] : zero);
      nfa_cat(&t, ", ");
    }
    nfa_cat(out, tb);
    return;
  }
  /* int(num) == 0 / 1 */
  int nz = 0, val1 = 0, all_dig = 1;
  const char* q = num;
  while (is_sp(*q)) q++;
  for (const char* r = q; *r; r++) {
    if (*r < '0' || *r > '9') all_dig = 0;
  }
  if (!all_dig) {
    nf_fail(c, "ValueError: invalid literal for int() with base 10: '%.60s'", num);
    return;
  }
  while (*q == '0') q++;
  nz = !*q;
  val1 = !strcmp(q, "1");
  if (nz || val1) {
    nfa_cat(out, nz ? zero : one);
    return;
  }
  int len = (int)strlen(q), mill = (len - 1) / 3;
  int head = len - 3 * mill;
  char trip[3];
  for (int k = 0; k < 3; ++k) trip[k] = k < 3 - head ? '0' : q[k - (3 - head)];
  i_hundfn(c, trip[0] - '0', trip[1] - '0', trip[2] - '0', mill, andword, out);
  for (int g = mill - 1, p = head; g >= 0 && !c->fail; --g, p += 3)
    i_hundfn(c, q[p] - '0', q[p + 1] - '0', q[p + 2] - '0', g, andword, out);
}

static void strip_into(const char* s, int n, nfa* out) {
  int a = 0, b = n;
  while (a < b && is_sp(s[a])) ++a;
  while (b > a && is_sp(s[b - 1])) --b;
  nfa_put(out, s + a, b - a);
}

/* (\S)\s+, -> \1,   then \s+, -> ,   then ,\s*$ -> ""   then \s+ -> " " */
static void clean_commas(const char* s, nfa* o) {
  int n = (int)strlen(s), i = 0;
  char tb[640];
  int tn = 0;
  /* pass 1 and 2 together: a whitespace run followed by ',' is dropped */
  while (i < n && tn < (int)sizeof tb - 1) {
    if (is_sp(s[i])) {
      int j = i;
      while (j < n && is_sp(s[j])) j++;
      if (j < n && s[j] == ',') {
        i = j;   /* "(\S)\s+," and "\s+," both drop the spaces */
        continue;
      }
      while (i < j && tn < (int)sizeof tb - 1) tb[tn++] = s[i++];
      continue;
    }
    tb[tn++] = s[i++];
  }
  /* ,\s*$ */
  int e = tn;
  while (e > 0 && is_sp(tb[e - 1])) e--;
  if (e > 0 && tb[e - 1] == ',') tn = e - 1;
  /* \s+ -> " " */
  for (int k = 0; k < tn;) {
    if (is_sp(tb[k])) {
      while (k < tn && is_sp(tb[k])) k++;
      nfa_ch(o, ' ');
    } else {
      nfa_ch(o, tb[k++]);
    }
  }
}

static void ordinal_words(const char* words, nfa* out) {
  /* (\w+)$: the final run of word characters */
  int n = (int)strlen(words), b = n;
  while (b > 0) {
    char ch = words[b - 1];
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_')
      b--;
    else
      break;
  }
  if (b == n) {
    nfa_cat(out, words);
    return;
  }
  const char* w = words + b;
  int wl = n - b;
  nfa_put(out, words, b);
  static const char* O[][2] = {{"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"},
                               {"eight", "eighth"}, {"nine", "ninth"}, {"twelve", "twelfth"}};
  for (size_t k = 0; k < sizeof O / sizeof O[0]; ++k)
    if (!strcmp(w, O[k][0])) {
      nfa_cat(out, O[k][1]);
      return;
    }
  if (wl >= 2 && !strcmp(w + wl - 2, "ty")) {
    nfa_put(out, w, wl - 1);
    nfa_cat(out, "ieth");
  } else {
    nfa_cat(out, w);
    nfa_cat(out, "th");
  }
}


/* \s+ -> " " then strip, src -> dst */
static void ws_strip(const char* s, int n, nfa* o) {
  char last = ' ';
  int start = o->n;
  for (int i = 0; i < n;) {
    if (is_sp(s[i])) {
      while (i < n && is_sp(s[i])) i++;
      if (last != ' ') {
        nfa_ch(o, ' ');
        last = ' ';
      }
    } else {
      nfa_ch(o, s[i]);
      last = s[i++];
    }
  }
  while (o->n > start && o->p[o->n - 1] == ' ') o->p[--o->n] = 0;
}

/* inflect.engine().number_to_words(s, andword, zero, group); s = digits or digits + st/nd/rd/th */
static void inflect_words(nano_front_ctx* c, const char* s0, const char* andword, const char* zero, int group,
                          nfa* res) {
  char sb[160], ord[4] = {0};
  int n = (int)strlen(s0);
  int nd = 0;   /* ^(\d+)(st|nd|rd|th)$ */
  while (nd < n && s0[nd] >= '0' && s0[nd] <= '9') nd++;
  int is_ord = nd > 0 && nd == n - 2 &&
               (!strcmp(s0 + nd, "st") || !strcmp(s0 + nd, "nd") || !strcmp(s0 + nd, "rd") || !strcmp(s0 + nd, "th"));
  int keep = is_ord ? nd : n;
  if (keep >= (int)sizeof sb) {
    nf_fail(c, "IndexError: list index out of range");
    return;
  }
  memcpy(sb, s0, (size_t)keep);
  sb[keep] = 0;
  if (is_ord) memcpy(ord, s0 + nd, 3);
  char aw[16];
  if (*andword)
    snprintf(aw, sizeof aw, " %s ", andword);
  else
    strcpy(aw, " ");
  char A[NW], B[NW];
  nfa a = {A, 0, NW, c}, b = {B, 0, NW, c};
  A[0] = B[0] = 0;
  i_enword(c, sb, group, zero, "one", aw, &a);   /* raw */
  if (c->fail) return;
  clean_commas(A, &b);                          /* (\S)\s+, / \s+, / ,\s*$ / \s+ */
  a.n = 0;
  A[0] = 0;
  strip_into(B, b.n, &a);                       /* chunks in A */
  b.n = 0;
  B[0] = 0;
  if (group) {
    const char* p = A;
    for (;;) {
      const char* e = strchr(p, ',');
      int len = e ? (int)(e - p) : (int)strlen(p);
      int x = 0, y = len;
      while (x < y && is_sp(p[x])) x++;
      while (y > x && is_sp(p[y - 1])) y--;
      if (y > x) {
        if (b.n) nfa_cat(&b, ", ");
        nfa_put(&b, p + x, y - x);
      }
      if (!e) break;
      p = e + 1;
    }
  } else {
    /* , (\S+)\Z -> " and \1", then \s+ -> " ", strip */
    int L = a.n, k = L;
    while (k > 0 && !is_sp(A[k - 1])) k--;
    if (k < L && k >= 2 && A[k - 1] == ' ' && A[k - 2] == ',') {
      char T[NW];
      nfa t = {T, 0, NW, c};
      T[0] = 0;
      nfa_put(&t, A, k - 2);
      nfa_ch(&t, ' ');
      nfa_cat(&t, andword);
      nfa_ch(&t, ' ');
      nfa_put(&t, A + k, L - k);
      ws_strip(T, t.n, &b);
    } else {
      ws_strip(A, L, &b);
    }
  }
  if (ord[0])
    ordinal_words(B, res);
  else
    nfa_cat(res, B);
}

/* ------------------------------------------------------------------ normalize_numbers */
static int dg(u32 ch) { return ch >= '0' && ch <= '9'; }

static void expand_dollars(nano_front_ctx* c, const u32* m, int n, nfs* out) {
  int dots = 0, d1 = -1;
  for (int i = 0; i < n; ++i)
    if (m[i] == '.') {
      if (d1 < 0) d1 = i;
      ++dots;
    }
  if (dots > 1) {
    nfs_put(out, m, n);
    nfs_ascii(out, " dollars");
    return;
  }
  int dn = d1 < 0 ? n : d1;
  char db[128], cb2[128];
  nfa dol = {db, 0, (int)sizeof db, c}, cen = {cb2, 0, (int)sizeof cb2, c};
  db[0] = cb2[0] = 0;
  if (dn == 0)
    nfa_ch(&dol, '0');
  else
    nf_py_int(c, m, dn, &dol);
  if (d1 >= 0 && n - d1 - 1 > 0)
    nf_py_int(c, m + d1 + 1, n - d1 - 1, &cen);
  else
    nfa_ch(&cen, '0');
  if (c->fail) return;
  int d = strcmp(db, "0") != 0, cc = strcmp(cb2, "0") != 0;
  if (d) {
    nfs_ascii(out, db);
    nfs_ascii(out, strcmp(db, "1") ? " dollars" : " dollar");
  }
  if (d && cc) nfs_ascii(out, ", ");
  if (cc) {
    nfs_ascii(out, cb2);
    nfs_ascii(out, strcmp(cb2, "1") ? " cents" : " cent");
  }
  if (!d && !cc) nfs_ascii(out, "zero dollars");
}

static void expand_number(nano_front_ctx* c, const u32* m, int n, nfs* out) {
  char nb[200];
  nfa num = {nb, 0, (int)sizeof nb, c};
  nb[0] = 0;
  nf_py_int(c, m, n, &num);
  if (c->fail) return;
  char wb[NW];
  nfa w = {wb, 0, (int)sizeof wb, c};
  wb[0] = 0;
  if (num.n == 4 && strcmp(nb, "1000") > 0 && strcmp(nb, "3000") < 0) {
    int v = atoi(nb);
    char b[16];
    if (v == 2000) {
      nfa_cat(&w, "two thousand");
    } else if (v > 2000 && v < 2010) {
      nfa_cat(&w, "two thousand ");
      snprintf(b, sizeof b, "%d", v % 100);
      inflect_words(c, b, "and", "zero", 0, &w);
    } else if (v % 100 == 0) {
      snprintf(b, sizeof b, "%d", v / 100);
      inflect_words(c, b, "and", "zero", 0, &w);
      nfa_cat(&w, " hundred");
    } else {
      char tb[160];   /* four digits, grouped */
      nfa t = {tb, 0, (int)sizeof tb, c};
      tb[0] = 0;
      inflect_words(c, nb, "", "oh", 2, &t);
      for (int i = 0; i < t.n; ++i) {
        if (tb[i] == ',' && i + 1 < t.n && tb[i + 1] == ' ') {
          nfa_ch(&w, ' ');
          ++i;
        } else {
          nfa_ch(&w, tb[i]);
        }
      }
    }
  } else {
    inflect_words(c, nb, "", "zero", 0, &w);
  }
  nfs_ascii(out, wb);
}

/* one normalize_numbers pass: src -> dst */
static void nn_pass(nano_front_ctx* c, int pass, const nfs* src, nfs* dst) {
  const u32* p = src->p;
  int m = src->n;
  dst->n = 0;
  for (int i = 0; i < m && !c->fail;) {
    u32 ch = p[i];
    if (pass == 0 && dg(ch)) {   /* ([0-9][0-9\,]+[0-9]) -> the digits without commas */
      int j = i + 1;
      while (j < m && (dg(p[j]) || p[j] == ',')) j++;
      int L = j - 1;
      while (L > i + 1 && !dg(p[L])) L--;
      if (L >= i + 2) {
        for (int k = i; k <= L; k++)
          if (p[k] != ',') nfs_push(dst, p[k]);
        i = L + 1;
        continue;
      }
    } else if (pass == 1 && ch == 0xA3) {   /* £([0-9\,]*[0-9]+) -> \1 pounds */
      int j = i + 1;
      while (j < m && (dg(p[j]) || p[j] == ',')) j++;
      int L = j;
      while (L > i + 1 && !dg(p[L - 1])) L--;
      if (L > i + 1) {
        nfs_put(dst, p + i + 1, L - i - 1);
        nfs_ascii(dst, " pounds");
        i = L;
        continue;
      }
    } else if (pass == 2 && ch == '$') {   /* \$([0-9\.\,]*[0-9]+) -> dollars and cents */
      int j = i + 1;
      while (j < m && (dg(p[j]) || p[j] == ',' || p[j] == '.')) j++;
      int L = j;
      while (L > i + 1 && !dg(p[L - 1])) L--;
      if (L > i + 1) {
        expand_dollars(c, p + i + 1, L - i - 1, dst);
        i = L;
        continue;
      }
    } else if (pass == 3 && dg(ch)) {   /* ([0-9]+\.[0-9]+) -> " point " */
      int j = i;
      while (j < m && dg(p[j])) j++;
      if (j + 1 < m && p[j] == '.' && dg(p[j + 1])) {
        int k = j + 1;
        while (k < m && dg(p[k])) k++;
        nfs_put(dst, p + i, j - i);
        nfs_ascii(dst, " point ");
        nfs_put(dst, p + j + 1, k - j - 1);
        i = k;
        continue;
      }
      nfs_put(dst, p + i, j - i);   /* no match can start inside this digit run */
      i = j;
      continue;
    } else if (pass == 4 && dg(ch)) {   /* [0-9]+(st|nd|rd|th) -> ordinal words */
      int j = i;
      while (j < m && dg(p[j])) j++;
      if (j + 1 < m && ((p[j] == 's' && p[j + 1] == 't') || (p[j] == 'n' && p[j + 1] == 'd') ||
                        (p[j] == 'r' && p[j + 1] == 'd') || (p[j] == 't' && p[j + 1] == 'h'))) {
        if (j + 2 - i >= 160) {
          nf_fail(c, "IndexError: list index out of range");
          return;
        }
        char ab[160];
        int an = 0;
        for (int k = i; k < j + 2; k++) ab[an++] = (char)p[k];
        ab[an] = 0;
        char wb[NW];
        nfa w = {wb, 0, (int)sizeof wb, c};
        wb[0] = 0;
        inflect_words(c, ab, "and", "zero", 0, &w);
        nfs_ascii(dst, wb);
        i = j + 2;
        continue;
      }
      nfs_put(dst, p + i, j - i);
      i = j;
      continue;
    } else if (pass == 5 && dg(ch)) {   /* [0-9]+ -> number words */
      int j = i;
      while (j < m && dg(p[j])) j++;
      expand_number(c, p + i, j - i, dst);
      i = j;
      continue;
    }
    nfs_push(dst, ch);
    i++;
  }
}

/* g2p_en normalize_numbers on x (result in x; y is scratch) */
void nf_normalize_numbers2(nano_front_ctx* c, nfs* x, nfs* y) {
  for (int pass = 0; pass < 6 && !c->fail; pass += 2) {
    nn_pass(c, pass, x, y);
    if (!c->fail) nn_pass(c, pass + 1, y, x);
  }
}
