#include "numwords.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace steertts {
namespace detail {

namespace {

u32str A(const std::string& s) { return u32str(s.begin(), s.end()); }  // ascii -> u32
std::string N(const u32str& s) {                                        // ascii u32 -> narrow
  std::string o;
  o.reserve(s.size());
  for (char32_t c : s) o.push_back(static_cast<char>(c));
  return o;
}
bool sp(char c) { return c == ' ' || (c >= 9 && c <= 13) || (c >= 28 && c <= 31); }
std::string strip(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && sp(s[a])) ++a;
  while (b > a && sp(s[b - 1])) --b;
  return s.substr(a, b - a);
}

const char* ONES[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                      "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                      "nineteen"};
const char* TENS[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
const char* HIGH[] = {"m", "b", "tr", "quadr", "quint", "sext", "sept", "oct", "non", "dec", "undec", "duodec",
                      "tredec", "quattuordec", "quindec", "sexdec", "septendec", "octodec", "novemdec", "vigint",
                      "unvigint", "duovigint", "trevigint", "quattuorvigint", "quinvigint", "sexvigint",
                      "septenvigint", "octovigint", "novemvigint", "trigint", "untrigint", "duotrigint"};
constexpr int N_SCALES = 2 + 32;  // "", thousand, 32 x -illion; MAXVAL = 10**(3*(N_SCALES-1)+3) = 10**102
std::string scale(int k) { return k == 0 ? "" : k == 1 ? "thousand" : std::string(HIGH[k - 2]) + "illion"; }

std::string below100(int n) {
  if (n < 20) return ONES[n];
  int t = n / 10, o = n % 10;
  return std::string(TENS[t]) + (o ? std::string("-") + ONES[o] : "");
}
std::string below1000(int n) {
  int h = n / 100, r = n % 100;
  if (!h) return below100(r);
  return std::string(ONES[h]) + " hundred" + (r ? " and " + below100(r) : "");
}

const char* IONES[] = {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
const char* ITEENS[] = {"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                        "eighteen", "nineteen"};
const char* IMILL[] = {" ", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion",
                       " sextillion", " septillion", " octillion", " nonillion", " decillion"};
std::string imill(int k) {
  if (k < 0 || k >= 12) throw std::out_of_range("list index out of range");
  return IMILL[k];
}
std::string i_tenfn(int tens, int units, int mindex = 0) {
  if (tens != 1) {
    std::string tw = tens ? TENS[tens] : "";
    std::string hy = (tens && units) ? "-" : "";
    return tw + hy + IONES[units] + ((tens || units) ? imill(mindex) : std::string());
  }
  return std::string(ITEENS[units]) + imill(mindex);
}
std::string i_hundfn(int h, int t, int u, int mindex, const std::string& andword) {
  if (h) {
    std::string a = std::string(IONES[h]) + " hundred" + ((t || u) ? andword : "") + i_tenfn(t, u);
    return a + imill(mindex) + ", ";
  }
  if (t || u) {
    std::string a = i_tenfn(t, u);
    return a + imill(mindex) + ", ";
  }
  return "";
}

// exact decimal digits of a non-negative integral double (Python int(float))
std::string double_int_digits(double v) {
  if (v < 9.2e18) return std::to_string(static_cast<uint64_t>(v));
  int e;
  double m = std::frexp(v, &e);  // v = m * 2^e, m in [0.5, 1)
  uint64_t mant = static_cast<uint64_t>(std::ldexp(m, 53));
  e -= 53;
  std::vector<uint32_t> big;  // base 1e9, little endian
  while (mant) {
    big.push_back(static_cast<uint32_t>(mant % 1000000000u));
    mant /= 1000000000u;
  }
  for (int k = 0; k < e; ++k) {
    uint64_t carry = 0;
    for (auto& d : big) {
      uint64_t x = static_cast<uint64_t>(d) * 2 + carry;
      d = static_cast<uint32_t>(x % 1000000000u);
      carry = x / 1000000000u;
    }
    if (carry) big.push_back(static_cast<uint32_t>(carry));
  }
  std::string s = std::to_string(big.back());
  for (size_t k = big.size() - 1; k-- > 0;) {
    std::string part = std::to_string(big[k]);
    s += std::string(9 - part.size(), '0') + part;
  }
  return s;
}

}  // namespace

std::string py_int_digits(const u32str& s, const Ucd& u) {
  for (char32_t c : s) {  // CPython reports an invalid literal before the digit-count limit
    if (u.digit(c) < 0) {
      std::string r = "'" + utf8_encode(s) + "'";
      throw std::invalid_argument("invalid literal for int() with base 10: " + r.substr(0, 200));
    }
  }
  if (s.size() > 4300)
    throw std::invalid_argument("Exceeds the limit (4300 digits) for integer string conversion: value has " +
                                std::to_string(s.size()) +
                                " digits; use sys.set_int_max_str_digits() to increase the limit");
  std::string o;
  for (char32_t c : s) {
    int d = u.digit(c);
    if (o.empty() && d == 0) continue;
    o.push_back(static_cast<char>('0' + d));
  }
  return o.empty() ? "0" : o;
}

Numwords::Numwords(const Ucd* u)
    : u_(u),
      ord_suffix_(u, R"(^(\d+)(st|nd|rd|th)$)"),
      pair_(u, R"((\d)(\d))"),
      last_digit_(u, R"((\d)$)"),
      cl1_(u, R"((\S)\s+,)"),
      cl2_(u, R"(\s+,)"),
      cl3_(u, R"(,\s*$)"),
      ws_(u, R"(\s+)"),
      final_one_(u, R"(, (\S+)\Z)"),
      last_word_(u, R"((\w+)$)"),
      comma_number_(u, R"(([0-9][0-9\,]+[0-9]))"),
      pounds_(u, "\xC2\xA3([0-9\\,]*[0-9]+)"),
      dollars_(u, R"(\$([0-9\.\,]*[0-9]+))"),
      decimal_(u, R"(([0-9]+\.[0-9]+))"),
      ordinal_re_(u, R"([0-9]+(st|nd|rd|th))"),
      number_(u, R"([0-9]+)") {}

std::string Numwords::cardinal(const std::string& n) const {
  if (n.size() > 102)
    throw std::overflow_error("abs(" + n + ") must be less than 1" + std::string(102, '0') + ".");
  if (n == "0") return "zero";
  std::vector<std::pair<int, int>> groups;  // (value, k), lowest first
  for (int end = static_cast<int>(n.size()), k = 0; end > 0; end -= 3, ++k) {
    int st = end >= 3 ? end - 3 : 0;
    groups.push_back({std::atoi(n.substr(st, end - st).c_str()), k});
  }
  std::string out;
  bool first = true;
  for (size_t j = groups.size(); j-- > 0;) {
    int g = groups[j].first, k = groups[j].second;
    if (!g) continue;
    std::string s = below1000(g) + (k ? " " + scale(k) : "");
    if (first)
      out = s;
    else if (k == 0 && groups[0].first < 100)
      out += " and " + s;
    else
      out += ", " + s;
    first = false;
  }
  return out;
}

std::string Numwords::ordinal(const std::string& n) const {
  std::string c = cardinal(n);
  size_t sp = c.rfind(' ');
  std::string head = sp == std::string::npos ? "" : c.substr(0, sp + 1);
  std::string last = sp == std::string::npos ? c : c.substr(sp + 1);
  size_t hy = last.rfind('-');
  std::string lh = hy == std::string::npos ? "" : last.substr(0, hy + 1);
  std::string w = hy == std::string::npos ? last : last.substr(hy + 1);  // already lower case
  static const char* ORD[][2] = {{"one", "first"}, {"two", "second"}, {"three", "third"}, {"four", "fourth"},
                                 {"five", "fifth"}, {"six", "sixth"}, {"seven", "seventh"}, {"eight", "eighth"},
                                 {"nine", "ninth"}, {"ten", "tenth"}, {"eleven", "eleventh"}, {"twelve", "twelfth"}};
  std::string o;
  for (auto& p : ORD)
    if (w == p[0]) o = p[1];
  if (o.empty()) {
    if (!w.empty() && w.back() == 'y') w = w.substr(0, w.size() - 1) + "ie";
    o = w + "th";
  }
  return head + lh + o;
}

std::string Numwords::cardinal_float(double value) const {
  if (std::isinf(value)) throw std::overflow_error("cannot convert float infinity to integer");
  if (std::isnan(value)) throw std::invalid_argument("cannot convert float NaN to integer");
  if (value == std::trunc(value)) {
    double a = std::fabs(value);
    std::string d = double_int_digits(a);
    return value < 0 ? "minus " + cardinal(d) : cardinal(d);
  }
  double pre = std::trunc(value);
  // precision = -Decimal(repr(value)).as_tuple().exponent = (#significant digits - 1) - decimal exponent
  char buf[64];
  auto res = std::to_chars(buf, buf + sizeof buf, value, std::chars_format::scientific);
  std::string r(buf, res.ptr);
  size_t e = r.find('e');
  int k = 0;
  for (size_t i = 0; i < e; ++i)
    if (r[i] >= '0' && r[i] <= '9') ++k;
  int x = std::atoi(r.c_str() + e + 1);
  int precision = std::abs(k - 1 - x);
  if (precision > 308) throw std::overflow_error("int too large to convert to float");
  double p10 = std::strtod(("1e" + std::to_string(precision)).c_str(), nullptr);  // float(10 ** precision)
  double post = std::fabs(value - pre) * p10;
  double rp = std::nearbyint(post);  // Python round(): half to even
  double pi = std::fabs(rp - post) < 0.01 ? rp : std::floor(post);
  std::string ps = double_int_digits(pi);
  if (static_cast<int>(ps.size()) < precision) ps = std::string(precision - ps.size(), '0') + ps;
  std::string out = std::fabs(pre) == 0 ? "zero" : (pre < 0 ? "minus " : "") + cardinal(double_int_digits(std::fabs(pre)));
  if (precision) out += " point";
  for (int i = 0; i < precision; ++i) out += " " + cardinal(std::string(1, ps[i]));
  return out;
}

std::string Numwords::i_enword(std::string num, int group, const std::string& zero, const std::string& one,
                               const std::string& andword) const {
  if (group == 1) {
    throw std::logic_error("inflect group=1 is not used by the front end");
  }
  if (group == 2) {
    u32str s = pair_.sub(A(num), [&](const Regex::Match& m, const u32str& t) {
      int a = static_cast<int>(t[m.caps[2]] - '0'), b = static_cast<int>(t[m.caps[4]] - '0');
      std::string r = a ? i_tenfn(a, b) : zero + " " + (b ? IONES[b] : zero);
      return A(r + ", ");
    });
    s = last_digit_.sub(s, [&](const Regex::Match& m, const u32str& t) {
      int a = static_cast<int>(t[m.caps[2]] - '0');
      return A(std::string(a ? IONES[a] : zero) + ", ");
    });
    return N(s);
  }
  std::string iv = py_int_digits(A(num), *u_);
  if (iv == "0") return zero;
  if (iv == "1") return one;
  size_t a = 0;
  while (a < num.size() && sp(num[a])) ++a;
  while (a < num.size() && num[a] == '0') ++a;
  num = num.substr(a);
  std::vector<std::string> out;
  int mill = 0;
  while (num.size() > 3) {
    std::string trip = num.substr(num.size() - 3);
    num = num.substr(0, num.size() - 3);
    out.insert(out.begin(), i_hundfn(trip[0] - '0', trip[1] - '0', trip[2] - '0', mill, andword));
    ++mill;
  }
  std::string trip = std::string(3 - num.size(), '0') + num;
  out.insert(out.begin(), i_hundfn(trip[0] - '0', trip[1] - '0', trip[2] - '0', mill, andword));
  std::string r;
  for (auto& o : out) r += o;
  return r;
}

std::string Numwords::inflect_ordinal_words(const std::string& words) const {
  Regex::Match m;
  u32str W = A(words);
  if (!last_word_.search(W, 0, m)) return words;
  std::string w = N(m.str(W, 1)), nw;
  static const char* O[][2] = {{"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"},
                               {"eight", "eighth"}, {"nine", "ninth"}, {"twelve", "twelfth"}};
  for (auto& p : O)
    if (w == p[0]) nw = p[1];
  if (nw.empty()) {
    if (w.size() >= 2 && w.compare(w.size() - 2, 2, "ty") == 0)
      nw = w.substr(0, w.size() - 1) + "ieth";
    else
      nw = w + "th";
  }
  return words.substr(0, m.start) + nw;
}

std::string Numwords::inflect_words(const std::string& s0, const std::string& andword, const std::string& zero,
                                    int group) const {
  std::string s = s0, ordsuf;
  Regex::Match m;
  u32str S = A(s);
  if (ord_suffix_.match_at(S, 0, m)) {
    s = N(m.str(S, 1));
    ordsuf = N(m.str(S, 2));
  }
  std::string aw = andword.empty() ? " " : " " + andword + " ";
  u32str c = A(i_enword(s, group, zero, "one", aw));
  c = cl1_.sub(c, [](const Regex::Match& mm, const u32str& t) { return mm.str(t, 1) + U","; });
  c = cl2_.sub(c, [](const Regex::Match&, const u32str&) { return u32str(U","); });
  c = cl3_.sub(c, [](const Regex::Match&, const u32str&) { return u32str(); });
  c = ws_.sub(c, [](const Regex::Match&, const u32str&) { return u32str(U" "); });
  std::string chunks = strip(N(c)), out;
  if (group) {
    size_t st = 0;
    while (true) {
      size_t e = chunks.find(',', st);
      std::string piece = strip(chunks.substr(st, e == std::string::npos ? std::string::npos : e - st));
      if (!piece.empty()) out += (out.empty() ? "" : ", ") + piece;
      if (e == std::string::npos) break;
      st = e + 1;
    }
  } else {
    u32str t = final_one_.sub(A(chunks), [&](const Regex::Match& mm, const u32str& tt) {
      return A(" " + andword + " ") + mm.str(tt, 1);
    });
    t = ws_.sub(t, [](const Regex::Match&, const u32str&) { return u32str(U" "); });
    out = strip(N(t));
  }
  if (!ordsuf.empty()) out = inflect_ordinal_words(out);
  return out;
}

std::string Numwords::expand_dollars(const u32str& m) const {
  std::string match = N(m);
  std::vector<std::string> parts;
  size_t st = 0;
  while (true) {
    size_t e = match.find('.', st);
    parts.push_back(match.substr(st, e == std::string::npos ? std::string::npos : e - st));
    if (e == std::string::npos) break;
    st = e + 1;
  }
  if (parts.size() > 2) return match + " dollars";
  std::string dollars = parts[0].empty() ? "0" : py_int_digits(A(parts[0]), *u_);
  std::string cents = parts.size() > 1 && !parts[1].empty() ? py_int_digits(A(parts[1]), *u_) : "0";
  bool d = dollars != "0", c = cents != "0";
  if (d && c)
    return dollars + " " + (dollars == "1" ? "dollar" : "dollars") + ", " + cents + " " + (cents == "1" ? "cent" : "cents");
  if (d) return dollars + " " + (dollars == "1" ? "dollar" : "dollars");
  if (c) return cents + " " + (cents == "1" ? "cent" : "cents");
  return "zero dollars";
}

std::string Numwords::expand_number(const u32str& m) const {
  std::string num = py_int_digits(m, *u_);
  if (num.size() == 4 && num > "1000" && num < "3000") {
    int v = std::atoi(num.c_str());
    if (v == 2000) return "two thousand";
    if (v > 2000 && v < 2010) return "two thousand " + inflect_words(std::to_string(v % 100));
    if (v % 100 == 0) return inflect_words(std::to_string(v / 100)) + " hundred";
    std::string w = inflect_words(num, "", "oh", 2), o;
    for (size_t i = 0; i < w.size(); ++i) {  // .replace(", ", " ")
      if (w[i] == ',' && i + 1 < w.size() && w[i + 1] == ' ') {
        o += ' ';
        ++i;
      } else {
        o += w[i];
      }
    }
    return o;
  }
  return inflect_words(num, "");
}

u32str Numwords::normalize_numbers(const u32str& text) const {
  u32str t = comma_number_.sub(text, [](const Regex::Match& m, const u32str& s) {
    u32str g = m.str(s, 1), o;
    for (char32_t c : g)
      if (c != ',') o.push_back(c);
    return o;
  });
  t = pounds_.sub(t, [](const Regex::Match& m, const u32str& s) { return m.str(s, 1) + U" pounds"; });
  t = dollars_.sub(t, [&](const Regex::Match& m, const u32str& s) { return A(expand_dollars(m.str(s, 1))); });
  t = decimal_.sub(t, [](const Regex::Match& m, const u32str& s) {
    u32str g = m.str(s, 1), o;
    for (char32_t c : g) {
      if (c == '.')
        o += U" point ";
      else
        o.push_back(c);
    }
    return o;
  });
  t = ordinal_re_.sub(t, [&](const Regex::Match& m, const u32str& s) { return A(inflect_words(N(m.str(s, 0)))); });
  t = number_.sub(t, [&](const Regex::Match& m, const u32str& s) { return A(expand_number(m.str(s, 0))); });
  return t;
}

}  // namespace detail
}  // namespace steertts
