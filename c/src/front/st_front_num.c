/* Ports of steertts_edge/edge_front/numwords.py (num2words / inflect subsets) and normnum.py (g2p_en
 * normalize_numbers). Integers are decimal digit strings (Python ints are unbounded); Python exceptions become
 * fx_fail("ValueError: ...") etc. with Python's message. */
#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "st_front_int.h"

static const char* ONES[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                             "eighteen", "nineteen"};
static const char* TENS[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
static const char* HIGH[] = {"m", "b", "tr", "quadr", "quint", "sext", "sept", "oct", "non", "dec", "undec", "duodec",
                             "tredec", "quattuordec", "quindec", "sexdec", "septendec", "octodec", "novemdec",
                             "vigint", "unvigint", "duovigint", "trevigint", "quattuorvigint", "quinvigint",
                             "sexvigint", "septenvigint", "octovigint", "novemvigint", "trigint", "untrigint",
                             "duotrigint"};
/* scales: "", thousand, 32 x -illion; MAXVAL = 10**102 */

static int is_sp(char c) { return c == ' ' || (c >= 9 && c <= 13) || (c >= 28 && c <= 31); }

static void below100(fx* x, int n, st_sb* o) {
  if (n < 20) {
    sb_cat(x, o, ONES[n]);
    return;
  }
  sb_cat(x, o, TENS[n / 10]);
  if (n % 10) {
    sb_ch(x, o, '-');
    sb_cat(x, o, ONES[n % 10]);
  }
}
static void below1000(fx* x, int n, st_sb* o) {
  int h = n / 100, r = n % 100;
  if (!h) {
    below100(x, r, o);
    return;
  }
  sb_cat(x, o, ONES[h]);
  sb_cat(x, o, " hundred");
  if (r) {
    sb_cat(x, o, " and ");
    below100(x, r, o);
  }
}

double st_front_strtod(const char* a) {
  /* Python float() ignores the C locale: swap '.' for the locale's decimal point */
  const char* dp = localeconv()->decimal_point;
  if (!dp || (dp[0] == '.' && !dp[1])) return strtod(a, NULL);
  size_t n = strlen(a), dl = strlen(dp);
  char buf[512];
  char* t = n * dl + 1 <= sizeof buf ? buf : malloc(n * dl + 1);
  if (!t) return strtod(a, NULL);
  char* o = t;
  for (size_t i = 0; i < n; ++i) {
    if (a[i] == '.') {
      memcpy(o, dp, dl);
      o += dl;
    } else {
      *o++ = a[i];
    }
  }
  *o = 0;
  double v = strtod(t, NULL);
  if (t != buf) free(t);
  return v;
}

/* exact decimal digits of a non-negative integral double (Python int(float)) */
static void double_int_digits(fx* x, double v, st_sb* o) {
  if (v < 9.2e18) {
    sb_fmt(x, o, "%llu", (unsigned long long)v);
    return;
  }
  int e;
  double m = frexp(v, &e);  /* v = m * 2^e, m in [0.5, 1) */
  unsigned long long mant = (unsigned long long)ldexp(m, 53);
  e -= 53;
  uint32_t big[40];  /* base 1e9, little endian; DBL_MAX < 1e309 = 35 limbs */
  int nb = 0;
  while (mant) {
    big[nb++] = (uint32_t)(mant % 1000000000u);
    mant /= 1000000000u;
  }
  for (int k = 0; k < e; ++k) {
    unsigned long long carry = 0;
    for (int j = 0; j < nb; ++j) {
      unsigned long long t = (unsigned long long)big[j] * 2 + carry;
      big[j] = (uint32_t)(t % 1000000000u);
      carry = t / 1000000000u;
    }
    if (carry) big[nb++] = (uint32_t)carry;
  }
  sb_fmt(x, o, "%u", (unsigned)big[nb - 1]);
  for (int k = nb - 1; k-- > 0;) sb_fmt(x, o, "%09u", (unsigned)big[k]);
}

void py_int_digits(fx* x, const st_ucd* u, const u32* s, int n, st_sb* out) {
  for (int i = 0; i < n; ++i) {  /* CPython reports an invalid literal before the digit-count limit */
    if (ucd_digit(u, s[i]) < 0) {
      st_sb r = {0};
      sb_ch(x, &r, '\'');
      sb_utf8(x, &r, s, n);
      sb_ch(x, &r, '\'');
      if (r.n > 200) r.p[200] = 0;  /* r[:200] on bytes is fine: the golden inputs are ascii here */
      fx_fail(x, "ValueError: invalid literal for int() with base 10: %s", sb_cstr(&r));
      sb_free(&r);
      return;
    }
  }
  if (n > 4300) {
    fx_fail(x, "ValueError: Exceeds the limit (4300 digits) for integer string conversion: value has %d digits; "
               "use sys.set_int_max_str_digits() to increase the limit", n);
    return;
  }
  int start = out->n;
  for (int i = 0; i < n; ++i) {
    int d = ucd_digit(u, s[i]);
    if (out->n == start && d == 0) continue;
    sb_ch(x, out, (char)('0' + d));
  }
  if (out->n == start) sb_ch(x, out, '0');
}

void nw_cardinal(fx* x, const char* nd, st_sb* out) {
  int n = (int)strlen(nd);
  if (n > 102) {
    fx_fail(x, "OverflowError: abs(%s) must be less than 1%s.", nd,
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
    return;
  }
  if (!strcmp(nd, "0")) {
    sb_cat(x, out, "zero");
    return;
  }
  int g[35], ng = 0;  /* groups of 3 digits, lowest first; g[k] has scale k */
  for (int end = n; end > 0; end -= 3) {
    int st = end >= 3 ? end - 3 : 0, v = 0;
    for (int i = st; i < end; ++i) v = v * 10 + (nd[i] - '0');
    g[ng++] = v;
  }
  int first = 1;
  for (int k = ng; k-- > 0;) {
    if (!g[k]) continue;
    if (!first) sb_cat(x, out, (k == 0 && g[0] < 100) ? " and " : ", ");
    below1000(x, g[k], out);
    if (k) {
      sb_ch(x, out, ' ');
      if (k == 1) {
        sb_cat(x, out, "thousand");
      } else {
        sb_cat(x, out, HIGH[k - 2]);
        sb_cat(x, out, "illion");
      }
    }
    first = 0;
  }
}

void nw_ordinal(fx* x, const char* nd, st_sb* out) {
  st_sb c = {0};
  nw_cardinal(x, nd, &c);
  if (x->err) {
    sb_free(&c);
    return;
  }
  const char* cs = sb_cstr(&c);
  const char* sp = strrchr(cs, ' ');
  const char* last = sp ? sp + 1 : cs;
  const char* hy = strrchr(last, '-');
  const char* w = hy ? hy + 1 : last;
  sb_put(x, out, cs, (int)(w - cs));  /* head + last word's hyphen prefix */
  static const char* ORD[][2] = {{"one", "first"}, {"two", "second"}, {"three", "third"}, {"four", "fourth"},
                                 {"five", "fifth"}, {"six", "sixth"}, {"seven", "seventh"}, {"eight", "eighth"},
                                 {"nine", "ninth"}, {"ten", "tenth"}, {"eleven", "eleventh"}, {"twelve", "twelfth"}};
  for (size_t k = 0; k < sizeof ORD / sizeof ORD[0]; ++k) {
    if (!strcmp(w, ORD[k][0])) {
      sb_cat(x, out, ORD[k][1]);
      sb_free(&c);
      return;
    }
  }
  int wl = (int)strlen(w);
  if (wl && w[wl - 1] == 'y') {
    sb_put(x, out, w, wl - 1);
    sb_cat(x, out, "ie");
  } else {
    sb_put(x, out, w, wl);
  }
  sb_cat(x, out, "th");
  sb_free(&c);
}

void nw_cardinal_float(fx* x, double value, st_sb* out) {
  if (isinf(value)) {
    fx_fail(x, "OverflowError: cannot convert float infinity to integer");
    return;
  }
  if (isnan(value)) {
    fx_fail(x, "ValueError: cannot convert float NaN to integer");
    return;
  }
  st_sb d = {0};
  if (value == trunc(value)) {
    double_int_digits(x, fabs(value), &d);
    if (value < 0) sb_cat(x, out, "minus ");
    if (!x->err) nw_cardinal(x, sb_cstr(&d), out);
    sb_free(&d);
    return;
  }
  double pre = trunc(value);
  /* precision = -Decimal(repr(value)).as_tuple().exponent = (#significant digits - 1) - decimal exponent, from the
   * shortest round-tripping repr: the fewest digits whose correctly rounded %e form parses back to value */
  char buf[64];
  for (int p = 0; p < 17; ++p) {
    snprintf(buf, sizeof buf, "%.*e", p, value);
    if (strtod(buf, NULL) == value) break;
  }
  const char* e = strchr(buf, 'e');
  int k = 0;
  for (const char* c = buf; c < e; ++c)
    if (*c >= '0' && *c <= '9') ++k;
  int ex = atoi(e + 1);
  int precision = abs(k - 1 - ex);
  if (precision > 308) {
    fx_fail(x, "OverflowError: int too large to convert to float");
    return;
  }
  char p10s[16];
  snprintf(p10s, sizeof p10s, "1e%d", precision);
  double p10 = strtod(p10s, NULL);  /* float(10 ** precision) */
  double post = fabs(value - pre) * p10;
  double rp = nearbyint(post);  /* Python round(): half to even */
  double pi = fabs(rp - post) < 0.01 ? rp : floor(post);
  st_sb ps = {0};
  double_int_digits(x, pi, &ps);
  if (fabs(pre) == 0) {
    sb_cat(x, out, "zero");
  } else {
    if (pre < 0) sb_cat(x, out, "minus ");
    double_int_digits(x, fabs(pre), &d);
    if (!x->err) nw_cardinal(x, sb_cstr(&d), out);
  }
  if (precision) sb_cat(x, out, " point");
  const char* pd = sb_cstr(&ps);
  int pad = precision > ps.n ? precision - ps.n : 0;
  for (int i = 0; i < precision && !x->err; ++i) {
    char one[2] = {i < pad ? '0' : pd[i - pad], 0};
    sb_ch(x, out, ' ');
    nw_cardinal(x, one, out);
  }
  sb_free(&ps);
  sb_free(&d);
}

/* ------------------------------------------------------------------ inflect.number_to_words subset */
static const char* IONES[] = {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
static const char* ITEENS[] = {"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                               "eighteen", "nineteen"};
static const char* IMILL[] = {" ", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion",
                              " sextillion", " septillion", " octillion", " nonillion", " decillion"};
static void imill(fx* x, int k, st_sb* o) {
  if (k < 0 || k >= 12) {
    fx_fail(x, "IndexError: list index out of range");
    return;
  }
  sb_cat(x, o, IMILL[k]);
}
static void i_tenfn(fx* x, int tens, int units, int mindex, st_sb* o) {
  if (tens != 1) {
    if (tens) sb_cat(x, o, TENS[tens]);
    if (tens && units) sb_ch(x, o, '-');
    sb_cat(x, o, IONES[units]);
    if (tens || units) imill(x, mindex, o);
    return;
  }
  sb_cat(x, o, ITEENS[units]);
  imill(x, mindex, o);
}
static void i_hundfn(fx* x, int h, int t, int u, int mindex, const char* andword, st_sb* o) {
  if (h) {
    sb_cat(x, o, IONES[h]);
    sb_cat(x, o, " hundred");
    if (t || u) sb_cat(x, o, andword);
    i_tenfn(x, t, u, 0, o);
    imill(x, mindex, o);
    sb_cat(x, o, ", ");
  } else if (t || u) {
    i_tenfn(x, t, u, 0, o);
    imill(x, mindex, o);
    sb_cat(x, o, ", ");
  }
}

typedef struct {
  const char* zero;
} grp2_ud;
static void repl_pair(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) {
  const char* zero = ((grp2_ud*)ud)->zero;
  int a = (int)(s[m->caps[2]] - '0'), b = (int)(s[m->caps[4]] - '0');
  st_sb r = {0};
  if (a) {
    i_tenfn(x, a, b, 0, &r);
  } else {
    sb_cat(x, &r, zero);
    sb_ch(x, &r, ' ');
    sb_cat(x, &r, b ? IONES[b] : zero);
  }
  sb_cat(x, &r, ", ");
  u32s_ascii(x, out, sb_cstr(&r));
  sb_free(&r);
}
static void repl_last_digit(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) {
  int a = (int)(s[m->caps[2]] - '0');
  u32s_ascii(x, out, a ? IONES[a] : ((grp2_ud*)ud)->zero);
  u32s_ascii(x, out, ", ");
}
static void to_u32(fx* x, const char* a, st_u32s* o) {
  o->n = 0;
  u32s_ascii(x, o, a);
}

/* inflect _enword (group 0 or 2), appended to out */
static void i_enword(fx* x, const st_numwords* nw, const char* num, int group, const char* zero, const char* one,
                     const char* andword, st_sb* out) {
  if (group == 2) {
    st_u32s a = {0}, b = {0};
    grp2_ud ud = {zero};
    to_u32(x, num, &a);
    st_rx_sub(x, nw->pair, a.p, a.n, repl_pair, &ud, &b);
    a.n = 0;
    st_rx_sub(x, nw->last_digit, b.p, b.n, repl_last_digit, &ud, &a);
    sb_from_u32(x, out, a.p, a.n);
    u32s_free(&a);
    u32s_free(&b);
    return;
  }
  st_u32s nu = {0};
  st_sb iv = {0};
  to_u32(x, num, &nu);
  py_int_digits(x, nw->u, nu.p, nu.n, &iv);
  u32s_free(&nu);
  if (x->err) {
    sb_free(&iv);
    return;
  }
  if (!strcmp(sb_cstr(&iv), "0") || !strcmp(sb_cstr(&iv), "1")) {
    sb_cat(x, out, !strcmp(sb_cstr(&iv), "0") ? zero : one);
    sb_free(&iv);
    return;
  }
  sb_free(&iv);
  while (is_sp(*num)) ++num;
  while (*num == '0') ++num;
  int len = (int)strlen(num), mill = (len - 1) / 3;
  /* triplets from the most significant (zero-padded) group down; mill index counts from the right */
  int head = len - 3 * mill;
  char trip[3];
  for (int k = 0; k < 3; ++k) trip[k] = k < 3 - head ? '0' : num[k - (3 - head)];
  i_hundfn(x, trip[0] - '0', trip[1] - '0', trip[2] - '0', mill, andword, out);
  for (int g = mill - 1, p = head; g >= 0 && !x->err; --g, p += 3)
    i_hundfn(x, num[p] - '0', num[p + 1] - '0', num[p + 2] - '0', g, andword, out);
}

static void inflect_ordinal_words(fx* x, const st_numwords* nw, const char* words, st_sb* out) {
  st_u32s W = {0};
  st_rxm m;
  to_u32(x, words, &W);
  if (x->err || !st_rx_search(x, nw->last_word, W.p, W.n, 0, &m)) {
    sb_cat(x, out, words);
    u32s_free(&W);
    return;
  }
  st_sb w = {0};
  sb_from_u32(x, &w, rxm_ptr(&m, W.p, 1), rxm_len(&m, 1));
  sb_put(x, out, words, m.start);
  static const char* O[][2] = {{"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"},
                               {"eight", "eighth"}, {"nine", "ninth"}, {"twelve", "twelfth"}};
  const char* ws = sb_cstr(&w);
  int done = 0;
  for (size_t k = 0; k < sizeof O / sizeof O[0] && !done; ++k)
    if (!strcmp(ws, O[k][0])) {
      sb_cat(x, out, O[k][1]);
      done = 1;
    }
  if (!done) {
    if (w.n >= 2 && !strcmp(ws + w.n - 2, "ty")) {
      sb_put(x, out, ws, w.n - 1);
      sb_cat(x, out, "ieth");
    } else {
      sb_cat(x, out, ws);
      sb_cat(x, out, "th");
    }
  }
  sb_free(&w);
  u32s_free(&W);
}

static void repl_cl1(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) {
  rxm_put(x, out, m, s, 1);
  u32s_push(x, out, ',');
}
static void repl_comma(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) { u32s_push(x, out, ','); }
static void repl_empty(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) {}
static void repl_space(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) { u32s_push(x, out, ' '); }
static void repl_final_one(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) {
  u32s_push(x, out, ' ');
  u32s_ascii(x, out, (const char*)ud);
  u32s_push(x, out, ' ');
  rxm_put(x, out, m, s, 1);
}

static void sub_inplace(fx* x, const st_rx* r, st_u32s* s, st_rx_repl f, void* ud) {
  st_u32s t = {0};
  st_rx_sub(x, r, s->p, s->n, f, ud, &t);
  u32s_move(s, &t);
}

static void strip_into(fx* x, const char* s, int n, st_sb* out) {
  int a = 0, b = n;
  while (a < b && is_sp(s[a])) ++a;
  while (b > a && is_sp(s[b - 1])) --b;
  sb_put(x, out, s + a, b - a);
}

/* inflect.engine().number_to_words(s, andword, zero, group) as numwords.inflect_words (s: str(int) or "21st") */
static void inflect_words(fx* x, const st_numwords* nw, const char* s0, const char* andword, const char* zero,
                          int group, st_sb* res) {
  st_u32s S = {0}, c = {0};
  st_sb s = {0}, ordsuf = {0}, raw = {0}, chunks = {0}, out = {0}, aw = {0};
  st_rxm m;
  to_u32(x, s0, &S);
  if (!x->err && st_rx_match_at(x, nw->ord_suffix, S.p, S.n, 0, &m)) {
    sb_from_u32(x, &s, rxm_ptr(&m, S.p, 1), rxm_len(&m, 1));
    sb_from_u32(x, &ordsuf, rxm_ptr(&m, S.p, 2), rxm_len(&m, 2));
  } else {
    sb_cat(x, &s, s0);
  }
  if (*andword) {
    sb_ch(x, &aw, ' ');
    sb_cat(x, &aw, andword);
    sb_ch(x, &aw, ' ');
  } else {
    sb_ch(x, &aw, ' ');
  }
  i_enword(x, nw, sb_cstr(&s), group, zero, "one", sb_cstr(&aw), &raw);
  if (x->err) goto done;
  to_u32(x, sb_cstr(&raw), &c);
  sub_inplace(x, nw->cl1, &c, repl_cl1, NULL);
  sub_inplace(x, nw->cl2, &c, repl_comma, NULL);
  sub_inplace(x, nw->cl3, &c, repl_empty, NULL);
  sub_inplace(x, nw->ws, &c, repl_space, NULL);
  {
    st_sb t = {0};
    sb_from_u32(x, &t, c.p, c.n);
    strip_into(x, sb_cstr(&t), t.n, &chunks);
    sb_free(&t);
  }
  if (group) {
    const char* p = sb_cstr(&chunks);
    for (;;) {
      const char* e = strchr(p, ',');
      int len = e ? (int)(e - p) : (int)strlen(p);
      st_sb piece = {0};
      strip_into(x, p, len, &piece);
      if (piece.n) {
        if (out.n) sb_cat(x, &out, ", ");
        sb_put(x, &out, piece.p, piece.n);
      }
      sb_free(&piece);
      if (!e) break;
      p = e + 1;
    }
  } else {
    to_u32(x, sb_cstr(&chunks), &c);
    sub_inplace(x, nw->final_one, &c, repl_final_one, (void*)andword);
    sub_inplace(x, nw->ws, &c, repl_space, NULL);
    st_sb t = {0};
    sb_from_u32(x, &t, c.p, c.n);
    strip_into(x, sb_cstr(&t), t.n, &out);
    sb_free(&t);
  }
  if (ordsuf.n)
    inflect_ordinal_words(x, nw, sb_cstr(&out), res);
  else
    sb_cat(x, res, sb_cstr(&out));
done:
  u32s_free(&S);
  u32s_free(&c);
  sb_free(&s);
  sb_free(&ordsuf);
  sb_free(&raw);
  sb_free(&chunks);
  sb_free(&out);
  sb_free(&aw);
}

/* ------------------------------------------------------------------ normnum.normalize_numbers */
static void expand_dollars(fx* x, const st_numwords* nw, const u32* m, int n, st_u32s* out) {
  int dots = 0, d1 = -1;
  for (int i = 0; i < n; ++i)
    if (m[i] == '.') {
      if (d1 < 0) d1 = i;
      ++dots;
    }
  if (dots > 1) {
    u32s_put(x, out, m, n);
    u32s_ascii(x, out, " dollars");
    return;
  }
  int dn = d1 < 0 ? n : d1;
  st_sb dol = {0}, cen = {0}, r = {0};
  if (dn == 0)
    sb_ch(x, &dol, '0');
  else
    py_int_digits(x, nw->u, m, dn, &dol);
  if (d1 >= 0 && n - d1 - 1 > 0)
    py_int_digits(x, nw->u, m + d1 + 1, n - d1 - 1, &cen);
  else
    sb_ch(x, &cen, '0');
  if (!x->err) {
    const char *ds = sb_cstr(&dol), *cs = sb_cstr(&cen);
    int d = strcmp(ds, "0") != 0, c = strcmp(cs, "0") != 0;
    if (d) {
      sb_cat(x, &r, ds);
      sb_cat(x, &r, strcmp(ds, "1") ? " dollars" : " dollar");
    }
    if (d && c) sb_cat(x, &r, ", ");
    if (c) {
      sb_cat(x, &r, cs);
      sb_cat(x, &r, strcmp(cs, "1") ? " cents" : " cent");
    }
    if (!d && !c) sb_cat(x, &r, "zero dollars");
    u32s_ascii(x, out, sb_cstr(&r));
  }
  sb_free(&dol);
  sb_free(&cen);
  sb_free(&r);
}

static void expand_number(fx* x, const st_numwords* nw, const u32* m, int n, st_u32s* out) {
  st_sb num = {0}, w = {0};
  py_int_digits(x, nw->u, m, n, &num);
  if (x->err) goto done;
  const char* ns = sb_cstr(&num);
  if (num.n == 4 && strcmp(ns, "1000") > 0 && strcmp(ns, "3000") < 0) {
    int v = atoi(ns);
    char b[16];
    if (v == 2000) {
      sb_cat(x, &w, "two thousand");
    } else if (v > 2000 && v < 2010) {
      sb_cat(x, &w, "two thousand ");
      snprintf(b, sizeof b, "%d", v % 100);
      inflect_words(x, nw, b, "and", "zero", 0, &w);
    } else if (v % 100 == 0) {
      snprintf(b, sizeof b, "%d", v / 100);
      inflect_words(x, nw, b, "and", "zero", 0, &w);
      sb_cat(x, &w, " hundred");
    } else {
      st_sb t = {0};
      inflect_words(x, nw, ns, "", "oh", 2, &t);
      const char* ts = sb_cstr(&t);
      for (int i = 0; i < t.n; ++i) {  /* .replace(", ", " ") */
        if (ts[i] == ',' && i + 1 < t.n && ts[i + 1] == ' ') {
          sb_ch(x, &w, ' ');
          ++i;
        } else {
          sb_ch(x, &w, ts[i]);
        }
      }
      sb_free(&t);
    }
  } else {
    inflect_words(x, nw, ns, "", "zero", 0, &w);
  }
  u32s_ascii(x, out, sb_cstr(&w));
done:
  sb_free(&num);
  sb_free(&w);
}

static void repl_drop_commas(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) {
  for (int i = m->caps[2]; i < m->caps[3]; ++i)
    if (s[i] != ',') u32s_push(x, out, s[i]);
}
static void repl_pounds(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) {
  rxm_put(x, out, m, s, 1);
  u32s_ascii(x, out, " pounds");
}
static void repl_dollars(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) {
  expand_dollars(x, (const st_numwords*)ud, rxm_ptr(m, s, 1), rxm_len(m, 1), out);
}
static void repl_decimal(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) {
  for (int i = m->caps[2]; i < m->caps[3]; ++i) {
    if (s[i] == '.')
      u32s_ascii(x, out, " point ");
    else
      u32s_push(x, out, s[i]);
  }
}
static void repl_ordinal(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) {
  st_sb a = {0}, w = {0};
  sb_from_u32(x, &a, s + m->start, m->end - m->start);
  inflect_words(x, (const st_numwords*)ud, sb_cstr(&a), "and", "zero", 0, &w);
  u32s_ascii(x, out, sb_cstr(&w));
  sb_free(&a);
  sb_free(&w);
}
static void repl_number(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out) {
  expand_number(x, (const st_numwords*)ud, s + m->start, m->end - m->start, out);
}

void nw_normalize_numbers(fx* x, const st_numwords* nw, const u32* s, int n, st_u32s* out) {
  st_u32s t = {0};
  st_rx_sub(x, nw->comma_number, s, n, repl_drop_commas, NULL, &t);
  sub_inplace(x, nw->pounds, &t, repl_pounds, NULL);
  sub_inplace(x, nw->dollars, &t, repl_dollars, (void*)nw);
  sub_inplace(x, nw->decimal, &t, repl_decimal, NULL);
  sub_inplace(x, nw->ordinal_re, &t, repl_ordinal, (void*)nw);
  st_rx_sub(x, nw->number, t.p, t.n, repl_number, (void*)nw, out);
  u32s_free(&t);
}

int numwords_init(st_numwords* nw, const st_ucd* u) {
  memset(nw, 0, sizeof *nw);
  nw->u = u;
  struct {
    st_rx** r;
    const char* p;
  } t[] = {{&nw->ord_suffix, "^(\\d+)(st|nd|rd|th)$"},
           {&nw->pair, "(\\d)(\\d)"},
           {&nw->last_digit, "(\\d)$"},
           {&nw->cl1, "(\\S)\\s+,"},
           {&nw->cl2, "\\s+,"},
           {&nw->cl3, ",\\s*$"},
           {&nw->ws, "\\s+"},
           {&nw->final_one, ", (\\S+)\\Z"},
           {&nw->last_word, "(\\w+)$"},
           {&nw->comma_number, "([0-9][0-9\\,]+[0-9])"},
           {&nw->pounds, "\xC2\xA3([0-9\\,]*[0-9]+)"},
           {&nw->dollars, "\\$([0-9\\.\\,]*[0-9]+)"},
           {&nw->decimal, "([0-9]+\\.[0-9]+)"},
           {&nw->ordinal_re, "[0-9]+(st|nd|rd|th)"},
           {&nw->number, "[0-9]+"}};
  for (size_t k = 0; k < sizeof t / sizeof t[0]; ++k) {
    *t[k].r = st_rx_compile(u, t[k].p, 0);
    if (!*t[k].r) {
      numwords_free(nw);
      return -1;
    }
  }
  return 0;
}

void numwords_free(st_numwords* nw) {
  st_rx* all[] = {nw->ord_suffix, nw->pair,         nw->last_digit, nw->cl1,     nw->cl2,        nw->cl3,
                  nw->ws,         nw->final_one,    nw->last_word,  nw->comma_number, nw->pounds, nw->dollars,
                  nw->decimal,    nw->ordinal_re,   nw->number};
  for (size_t k = 0; k < sizeof all / sizeof all[0]; ++k) st_rx_free(all[k]);
  memset(nw, 0, sizeof *nw);
}
