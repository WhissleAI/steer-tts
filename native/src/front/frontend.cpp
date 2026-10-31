// Port of steertts_edge/text.py (v1 edge path: normalize -> _expand -> EdgeG2p -> to_phonemes). The "v2 front-end"
// section of text.py is not used by to_phonemes / normalize and is not ported.
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "frontend_impl.h"

namespace steertts {

using detail::Regex;
using detail::u32str;

namespace {
u32str A(const std::string& s) { return u32str(s.begin(), s.end()); }  // ascii
std::string N(const u32str& s) {
  std::string o;
  for (char32_t c : s) o.push_back(static_cast<char>(c));
  return o;
}
u32str replace_all(const u32str& s, const u32str& from, const u32str& to) {
  u32str o;
  size_t st = 0;
  while (true) {
    size_t e = s.find(from, st);
    if (e == u32str::npos) break;
    o.append(s, st, e - st);
    o += to;
    st = e + from.size();
  }
  o.append(s, st, u32str::npos);
  return o;
}

std::vector<std::string> make_symbols() {
  std::vector<std::string> s = {"<pad>", "<sp>"};
  for (char c : std::string(",.?!;:-")) s.push_back(std::string(1, c));
  const char* arpa[] = {"AA", "AE", "AH", "AO", "AW", "AY", "B",  "CH", "D",  "DH", "EH", "ER", "EY",
                        "F",  "G",  "HH", "IH", "IY", "JH", "K",  "L",  "M",  "N",  "NG", "OW", "OY",
                        "P",  "R",  "S",  "SH", "T",  "TH", "UH", "UW", "V",  "W",  "Y",  "Z",  "ZH"};
  const std::string vowels = " AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW ";
  for (const char* p : arpa) {
    if (vowels.find(std::string(" ") + p + " ") != std::string::npos) {
      for (const char* st : {"0", "1", "2"}) s.push_back(std::string(p) + st);
    } else {
      s.push_back(p);
    }
  }
  const char* ev[] = {"laugh", "cry",  "scream", "yell", "cheer", "cough", "sneeze", "yawn",     "throat", "breath",
                      "sigh",  "sniff", "dog",   "cat",  "bird",  "knock", "door",   "phone",    "bell",   "horn",
                      "applause", "glass", "siren", "footsteps", "typing", "thunder", "clock"};
  for (const char* e : ev) s.push_back(std::string("[") + e + "]");
  for (int i = 0; i < 13; ++i) s.push_back("[reserved" + std::to_string(i) + "]");
  return s;
}

// markup._SENT_END: . ! ? and a space, but not after a title, a common abbreviation or an initial
std::string sent_end_pattern() {
  const char* abbr[] = {"Dr",  "Mr",  "Mrs", "Ms",  "Prof", "St",  "Jr",  "Sr",   "vs",  "No",  "Fig", "Vol", "pp",
                        "approx", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec"};
  std::string p = "(?<=[.!?])";
  for (const char* a : abbr) p += std::string("(?<!\\b") + a + "\\.)";
  return p + R"((?<!\be\.g\.)(?<!\bi\.e\.)(?<!\b[ap]\.m\.)(?<!\b[A-Z]\.)\s+)";
}

// text._roman
int roman_value(const std::string& r) {
  auto v = [](char c) {
    switch (c) {
      case 'I': return 1;
      case 'V': return 5;
      case 'X': return 10;
      case 'L': return 50;
      case 'C': return 100;
      case 'D': return 500;
      default: return 1000;
    }
  };
  int t = 0;
  for (size_t i = 0; i < r.size(); ++i) t += (i + 1 < r.size() && v(r[i]) < v(r[i + 1])) ? -v(r[i]) : v(r[i]);
  return t;
}

// spelled as letters (a lone "A" is the article, so it is written "ay")
const char* ampm_words(char32_t c) { return c == U'A' || c == U'a' ? " ay M" : " P M"; }

const char* tens_plural(char32_t d) {
  static const char* t[] = {"", "tens", "twenties", "thirties", "forties", "fifties", "sixties", "seventies", "eighties",
                            "nineties"};
  return t[d - U'0'];
}

// str(int(s)) for a string of ASCII digits
std::string int_str(const std::string& d) {
  size_t i = 0;
  while (i + 1 < d.size() && d[i] == '0') ++i;
  return d.substr(i);
}
}  // namespace

Frontend::Impl::Impl(const std::string& dir)
    : quotes(&ucd, "[\\\"“”‘’`]"),
      range(&ucd, "(\\d)\\s*[-–]\\s*(\\d)"),
      by(&ucd, "(\\d+)\\s*[x×]\\s*(\\d+)"),
      compound(&ucd, "(?<=[A-Za-z0-9])[-‐/](?=[A-Za-z0-9])"),
      currency(&ucd, "([$£€])\\s?(\\d(?:[\\d,]*\\d)?(?:\\.\\d+)?)(?:\\s*(million|billion|thousand)\\b)?"),
      percent(&ucd, R"((\d[\d,]*(?:\.\d+)?)\s?%)"),
      ordinal(&ucd, R"(\b(\d+)(st|nd|rd|th)\b)"),
      comma_num(&ucd, R"(\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b)"),
      decimal(&ucd, R"(\b\d+\.\d+\b)"),
      long_digits(&ucd, R"(\b\d{5,}\b)"),
      camel_acr(&ucd, R"((?<=[a-z])([A-Z]{2,5}s?)\b)"),
      acronym(&ucd, R"(\b[A-Z]{2,5}s?\b)"),
      ws(&ucd, R"(\s+)"),
      event(&ucd, R"(\[\s*([A-Za-z_]+)\s*\])"),
      emph(&ucd, R"(\*([^*\n]+)\*)"),
      token(&ucd,
            R"x((<\s*(/?)\s*([A-Za-z_]+)(?:\s+prompt\s*=\s*"([^"]*)")?\s*>))x"
            R"x(|(\[\s*([A-Za-z_]+)(?:\s+([\d.]+)\s*(ms|s)?)?\s*\]))x",
            /*icase=*/true),
      sent_end(&ucd, sent_end_pattern()) {
  ucd.load(dir + "/unicode.bin");
  symbols = make_symbols();
  for (size_t i = 0; i < symbols.size(); ++i) sym2id[symbols[i]] = static_cast<int>(i);
  g2p.reset(new detail::EdgeG2p(dir, &ucd, sym2id));
  // text._UNITS in dict order, then sorted(key=len, reverse=True) (stable)
  std::vector<std::pair<std::string, std::string>> u = {
      {"km/h", "kilometers per hour"}, {"km/hr", "kilometers per hour"}, {"m/s", "meters per second"},
      {"mph", "miles per hour"},       {"km", "kilometers"},             {"kg", "kilograms"},
      {"cm", "centimeters"},           {"mm", "millimeters"},            {"ft", "feet"},
      {"lbs", "pounds"},               {"lb", "pounds"},                 {"kHz", "kilohertz"},
      {"Hz", "hertz"},                 {"GB", "gigabytes"},              {"MB", "megabytes"},
      {"kph", "kilometers per hour"}};
  std::stable_sort(u.begin(), u.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  for (auto& e : u) {
    std::string pat;
    for (char c : e.first) pat += c == '/' ? std::string("[ /]") : std::string(1, c);  // re.escape is a no-op here
    units.emplace_back(Regex(&ucd, "(\\d)\\s?" + pat + "\\b"), e.second);
  }
  // text._pre
  phone = Regex(&ucd, R"((?<![0-9-])(?:\(([0-9]{3})\)\s?|([0-9]{3})[-.])([0-9]{3})[-.]([0-9]{4})(?![0-9-]))");
  time = Regex(&ucd, R"((?<![0-9:])\b([01]?[0-9]|2[0-3]):([0-5][0-9])(?!:[0-9])(?:\s?([AaPp])(?:\.[Mm]\.?|[Mm])(?![A-Za-z]))?)");
  colon = Regex(&ucd, R"((?<=[0-9]):(?=[0-9]))");
  ampm = Regex(&ucd, R"(\b([0-9]{1,2})\s?([AaPp])(?:\.[Mm]\.?|[Mm])(?![A-Za-z]))");
  const char* ab[][2] = {
      {R"(\bDr\.?(?=\s+[A-Z]))", "Doctor"},      {R"(\bMrs\.?(?=\s+[A-Z]))", "Missus"},
      {R"(\bMr\.?(?=\s+[A-Z]))", "Mister"},      {R"(\bMs\.?(?=\s+[A-Z]))", "Mizz"},
      {R"(\bProf\.?(?=\s+[A-Z]))", "Professor"}, {R"(\bSt\.?(?=\s+[A-Z]))", "Saint"},
      {R"(\bJr\.)", "Junior"},                   {R"(\bSr\.)", "Senior"},
      {R"(\bvs\.)", "versus"},                   {R"(\be\.g\.)", "for example"},
      {R"(\bi\.e\.)", "that is"},               {R"(\betc\.(?=\s*[,;:!?)]|\s+[a-z]))", "et cetera"},
      {R"(\betc\.)", "et cetera."},              {R"(\bapprox\.)", "approximately"},
      {R"(\bNo\.(?=\s*[0-9]))", "number"},          {R"(\bFig\.(?=\s*[0-9]))", "figure"},
      {R"(\bVol\.(?=\s*[0-9]))", "volume"},         {R"(\bpp\.(?=\s*[0-9]))", "pages"}};
  for (auto& a : ab) abbrev.emplace_back(Regex(&ucd, a[0]), a[1]);
  // month abbreviations before a day / year ("Nov 3", "Nov. 3, 2024"), then after a day ("3 Nov"; a final period stays)
  const char* months[][2] = {{"Jan", "January"}, {"Feb", "February"}, {"Mar", "March"},     {"Apr", "April"},
                             {"Jun", "June"},    {"Jul", "July"},     {"Aug", "August"},    {"Sept", "September"},
                             {"Sep", "September"}, {"Oct", "October"}, {"Nov", "November"}, {"Dec", "December"}};
  for (auto& m : months) abbrev.emplace_back(Regex(&ucd, std::string("\\b") + m[0] + R"((?:\.\s*|\s+)(?=[0-9]))"), std::string(m[1]) + " ");
  for (auto& m : months) abbrev.emplace_back(Regex(&ucd, std::string("(?<=[0-9] )") + m[0] + "\\b"), m[1]);
  const char* ab2[][2] = {
      {R"(\bJan\.)", "January"},                 {R"(\bFeb\.)", "February"},
      {R"(\bMar\.(?=\s*[0-9]))", "March"},          {R"(\bApr\.)", "April"},
      {R"(\bJun\.(?=\s*[0-9]))", "June"},           {R"(\bJul\.(?=\s*[0-9]))", "July"},
      {R"(\bAug\.)", "August"},                  {R"(\bSept?\.)", "September"},
      {R"(\bOct\.)", "October"},                 {R"(\bNov\.)", "November"},
      {R"(\bDec\.)", "December"}};
  for (auto& a : ab2) abbrev.emplace_back(Regex(&ucd, a[0]), a[1]);
  deg_cf = Regex(&ucd, "([0-9])\\s?°\\s?([CF])(?![A-Za-z])");
  deg = Regex(&ucd, "([0-9])\\s?°");
  letter_amp = Regex(&ucd, R"(\b([A-Z])\s?&\s?([A-Z])\b)");
  amp = Regex(&ucd, R"(\s*&\s*)");
  num_comma = Regex(&ucd, R"(\b(thousand|million|billion|trillion),(?= [a-z]))");
  tilde = Regex(&ucd, R"(~\s?(?=[0-9]))");
  scale = Regex(&ucd, "([$£€]\\s?[0-9]+(?:\\.[0-9]+)?)\\s?([KkMmBb])n?\\b");
  version = Regex(&ucd, R"((?<![0-9])[0-9]+(?:\.[0-9]+){2,}(?![0-9]))");
  roman = Regex(&ucd,
                R"(\b(Chapter|Part|Book|Volume|Act|Scene|Section|Article|Phase|Season|Episode|Level|Stage|Round)\s+)"
                R"((?=[IVXLCDM])(M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\b)");
  ww = Regex(&ucd, R"(\bWW(II|I|1|2)\b)");
  fraction = Regex(&ucd, R"((?<![0-9/])([1-9])/(10|[2-9])(?![0-9/]))");
  num_hyphen = Regex(&ucd, R"((?<=[a-z])-(?=[a-z]))");
  decade4 = Regex(&ucd, R"(\b(1[1-9]|20)([0-9])0'?s\b)");         // 1990s, 1800s, 2000s
  decade2 = Regex(&ucd, R"((?<![0-9A-Za-z])'?([1-9])0'?s\b)");    // '90s, 80s
  digit_groups = Regex(&ucd, R"((\b(?:[Nn]umber|[Cc]all|[Pp]hone|[Tt]el|[Dd]ial|[Ff]ax)\.?:?\s+)?(?<![0-9.,-])([0-9]+(?:-[0-9]+)+)(?![0-9-]|[.,][0-9]))");
  sign = Regex(&ucd, "(?<![^\\s(])([+−-])(?=[0-9])");           // +30, -5, U+2212 minus
  letter_digit = Regex(&ucd, R"(\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*[0-9])[A-Z0-9]+\b)");  // Q4, 5G, MP3
}

u32str Frontend::Impl::pre(const u32str& text0) const {
  u32str t = phone.sub(text0, [](const Regex::Match& m, const u32str& s) {
    u32str o;
    for (int g = 1; g <= 4; ++g) {
      if (!m.has(g)) continue;
      if (!o.empty()) o += U", ";
      u32str d = m.str(s, g);
      for (size_t i = 0; i < d.size(); ++i) {
        if (i) o += U' ';
        o += d[i];
      }
    }
    return o;
  });
  t = time.sub(t, [](const Regex::Match& m, const u32str& s) {
    std::string h = N(m.str(s, 1)), mm = N(m.str(s, 2)), w = int_str(h);
    if (mm != "00")
      w += mm[0] == '0' ? " oh " + mm.substr(1) : " " + mm;
    else if (!m.has(3))
      w += (std::atoi(h.c_str()) > 12 || std::atoi(h.c_str()) == 0 || h[0] == '0') ? " hundred" : " o'clock";
    if (m.has(3)) w += ampm_words(s[m.caps[6]]);
    return A(w);
  });
  t = colon.sub(t, [](const Regex::Match&, const u32str&) { return u32str(U" "); });  // durations, ratios
  t = ampm.sub(t, [](const Regex::Match& m, const u32str& s) {
    char32_t a = s[m.caps[4]];
    return A(int_str(N(m.str(s, 1))) + ampm_words(a));
  });
  for (auto& ab : abbrev) {
    const std::string& w = ab.second;
    t = ab.first.sub(t, [&](const Regex::Match&, const u32str&) { return A(w); });
  }
  t = deg_cf.sub(t, [](const Regex::Match& m, const u32str& s) {
    return m.str(s, 1) + (s[m.caps[4]] == U'C' ? U" degrees Celsius" : U" degrees Fahrenheit");
  });
  t = deg.sub(t, [](const Regex::Match& m, const u32str& s) { return m.str(s, 1) + U" degrees"; });
  t = letter_amp.sub(t, [](const Regex::Match& m, const u32str& s) {
    auto l = [&](int g) { return s[m.caps[2 * g]] == U'A' ? u32str(U"ay") : m.str(s, g); };
    return l(1) + U" and " + l(2);
  });
  t = amp.sub(t, [](const Regex::Match&, const u32str&) { return u32str(U" and "); });
  t = tilde.sub(t, [](const Regex::Match&, const u32str&) { return u32str(U"about "); });
  t = scale.sub(t, [](const Regex::Match& m, const u32str& s) {
    char32_t c = s[m.caps[4]];
    return m.str(s, 1) + (c == U'K' || c == U'k' ? U" thousand" : c == U'M' || c == U'm' ? U" million" : U" billion");
  });
  t = version.sub(t, [](const Regex::Match& m, const u32str& s) { return replace_all(m.str(s, 0), U".", U" point "); });
  t = roman.sub(t, [](const Regex::Match& m, const u32str& s) {
    return m.str(s, 1) + U" " + A(std::to_string(roman_value(N(m.str(s, 2)))));
  });
  t = ww.sub(t, [](const Regex::Match& m, const u32str& s) {
    u32str g = m.str(s, 1);
    return u32str(U"World War ") + (g == U"II" || g == U"2" ? U"2" : U"1");
  });
  t = fraction.sub(t, [](const Regex::Match& m, const u32str& s) {
    static const char* one[] = {"", "", "half", "third", "quarter", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"};
    static const char* many[] = {"", "", "halves", "thirds", "quarters", "fifths", "sixths", "sevenths", "eighths", "ninths",
                                 "tenths"};
    u32str n = m.str(s, 1), d = m.str(s, 2);
    int k = d == U"10" ? 10 : static_cast<int>(d[0] - U'0');
    return n + U" " + A(n == U"1" ? one[k] : many[k]);
  });
  t = decade4.sub(t, [](const Regex::Match& m, const u32str& s) {  // 1990s -> 19 nineties, 1800s -> 18 hundreds
    u32str c = m.str(s, 1);
    char32_t d = s[m.caps[4]];
    if (d != U'0') return c + U" " + A(tens_plural(d));
    return c == U"20" ? u32str(U"two thousands") : c + U" hundreds";
  });
  t = decade2.sub(t, [](const Regex::Match& m, const u32str& s) { return A(tens_plural(s[m.caps[2]])); });
  return t;
}

u32str Frontend::Impl::strip(const u32str& s) const {
  size_t a = 0, b = s.size();
  while (a < b && ucd.is_space(s[a])) ++a;
  while (b > a && ucd.is_space(s[b - 1])) --b;
  return s.substr(a, b - a);
}

u32str Frontend::Impl::collapse_ws(const u32str& s) const {
  return ws.sub(s, [](const Regex::Match&, const u32str&) { return u32str(U" "); });
}

std::string Frontend::Impl::num(const std::string& digits, bool ord) const {
  std::string w = ord ? g2p->numwords().ordinal(digits) : g2p->numwords().cardinal(digits), o;
  for (char c : w) {  // .replace("-", " ").replace(",", "")
    if (c == '-')
      o += ' ';
    else if (c != ',')
      o += c;
  }
  return o;
}

std::string Frontend::Impl::num_float(const u32str& s) const {
  std::string a;  // Python float() accepts any Unicode decimal digits
  for (char32_t c : s) {
    int d = ucd.digit(c);
    a.push_back(d >= 0 ? static_cast<char>('0' + d) : static_cast<char>(c));
  }
  double v = std::strtod(a.c_str(), nullptr);  // correctly rounded, like float()
  std::string w = g2p->numwords().cardinal_float(v), o;
  for (char c : w) {
    if (c == '-')
      o += ' ';
    else if (c != ',')
      o += c;
  }
  return o;
}

u32str Frontend::Impl::expand(const u32str& text0) const {
  const detail::Ucd& u = ucd;
  // hyphen-joined digit groups: an ID / phone number read digit by digit after "number" / "call" / "phone"
  // (No. 90-5551, call 555-1234) or when a group starts with 0 (555-0198); otherwise unchanged (a range, read next).
  // Dates (31-03-2018, 2023-10-22) are left as they are.
  u32str t = digit_groups.sub(pre(text0), [](const Regex::Match& m, const u32str& s) {
    std::vector<u32str> g(1);
    for (char32_t c : m.str(s, 2)) {
      if (c == U'-')
        g.emplace_back();
      else
        g.back() += c;
    }
    bool date = g.size() == 3 && ((g[2].size() == 4 && g[0].size() <= 2 && g[1].size() <= 2) ||
                                  (g[0].size() == 4 && g[1].size() <= 2 && g[2].size() <= 2));
    bool lead0 = false;
    for (auto& x : g) lead0 = lead0 || (x.size() > 1 && x[0] == U'0');
    if (!(m.has(1) || (!date && lead0))) return m.str(s, 0);
    u32str o = m.has(1) ? m.str(s, 1) : u32str();
    for (size_t k = 0; k < g.size(); ++k) {
      if (k) o += U", ";
      for (size_t i = 0; i < g[k].size(); ++i) {
        if (i) o += U' ';
        o += g[k][i];
      }
    }
    return o;
  });
  t = range.sub(t, [](const Regex::Match& m, const u32str& s) { return m.str(s, 1) + U" to " + m.str(s, 2); });
  t = by.sub(t, [](const Regex::Match& m, const u32str& s) { return m.str(s, 1) + U" by " + m.str(s, 2); });
  t = sign.sub(t, [](const Regex::Match& m, const u32str& s) { return u32str(s[m.start] == U'+' ? U"plus " : U"minus "); });
  t = compound.sub(t, [](const Regex::Match&, const u32str&) { return u32str(U" "); });
  t = currency.sub(t, [](const Regex::Match& m, const u32str& s) {
    char32_t c = s[m.caps[2]];
    u32str cur = c == U'$' ? U"dollars" : c == U'£' ? U"pounds" : U"euros";
    return m.str(s, 2) + U" " + (m.has(3) ? m.str(s, 3) + U" " : u32str()) + cur;
  });
  t = percent.sub(t, [](const Regex::Match& m, const u32str& s) { return m.str(s, 1) + U" percent"; });
  for (auto& un : units) {
    const std::string& words = un.second;
    t = un.first.sub(t, [&](const Regex::Match& m, const u32str& s) { return m.str(s, 1) + U" " + A(words); });
  }
  t = letter_digit.sub(t, [](const Regex::Match& m, const u32str& s) {  // after the units: 5GB is gigabytes
    u32str g = m.str(s, 0), o;  // Q4 -> Q 4, 5G -> 5 G, MP3 -> M P 3
    auto dig = [](char32_t c) { return c >= U'0' && c <= U'9'; };
    for (size_t i = 0; i < g.size(); ++i) {
      if (i && !(dig(g[i]) && dig(g[i - 1]))) o += U' ';
      if (g[i] == U'A')
        o += U"ay";
      else
        o += g[i];
    }
    return o;
  });
  t = ordinal.sub(t, [&](const Regex::Match& m, const u32str& s) {
    return A(num(detail::py_int_digits(m.str(s, 1), u), true));
  });
  t = comma_num.sub(t, [&](const Regex::Match& m, const u32str& s) {
    u32str g = m.str(s, 0), nc;
    for (char32_t c : g)
      if (c != ',') nc.push_back(c);
    if (g.find(U'.') != u32str::npos) return A(num_float(nc));
    return A(num(detail::py_int_digits(nc, u), false));
  });
  t = decimal.sub(t, [&](const Regex::Match& m, const u32str& s) { return A(num_float(m.str(s, 0))); });
  t = long_digits.sub(t, [&](const Regex::Match& m, const u32str& s) {
    return A(num(detail::py_int_digits(m.str(s, 0), u), false));
  });
  t = camel_acr.sub(t, [&](const Regex::Match& m, const u32str& s) {   // SteerTTS -> Steer TTS
    u32str o = U" ";
    o += m.str(s, 1);
    return o;
  });
  t = acronym.sub(t, [&](const Regex::Match& m, const u32str& s) {
    std::string g = N(m.str(s, 0)), low;
    for (char c : g) low.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c);
    if (g2p->in_cmu(low)) return A(g);
    std::string base = g;
    while (!base.empty() && base.back() == 's') base.pop_back();  // rstrip("s")
    std::string o;
    for (size_t i = 0; i < base.size(); ++i) {
      if (i) o += ' ';
      if (base[i] == 'A')
        o += "ay";  // a lone "A" is the article
      else
        o += base[i];
    }
    if (g.back() == 's' && g.size() > 2) o += "'s";
    return A(o);
  });
  // g2p_en's own number pass, without its hyphens ("seventy-two" goes to the neural guesser, "seventy two" does not)
  t = g2p->numwords().normalize_numbers(t);
  t = num_hyphen.sub(t, [](const Regex::Match&, const u32str&) { return u32str(U" "); });
  return num_comma.sub(t, [](const Regex::Match& m, const u32str& s) { return m.str(s, 1); });  // no pause inside a number
}

u32str Frontend::Impl::normalize(const u32str& text0) const {
  u32str t = quotes.sub(text0, [](const Regex::Match& m, const u32str& s) {
    char32_t c = s[m.start];
    return (c == U'‘' || c == U'’' || c == U'`') ? u32str(U"'") : u32str();
  });
  t = expand(t);  // first: it reads digit-dash-digit as a range before dashes become pauses
  t = replace_all(t, U"—", U", ");
  t = replace_all(t, U"–", U", ");
  t = replace_all(t, U"…", U". ");
  return strip(collapse_ws(t));
}

Phonemes Frontend::Impl::to_phonemes(const u32str& text0) const {
  const int SP = Frontend::kSp;
  u32str text = normalize(text0);
  if (text.find(U'[') != u32str::npos) {
    std::vector<u32str> parts = event.split(text);  // text, name, text, name, ...
    if (parts.size() > 1) {
      Phonemes r;
      r.ids.push_back(SP);
      r.word_idx.push_back(-1);
      int nw = 0;
      for (size_t k = 0; k < parts.size(); ++k) {
        const u32str& p = parts[k];
        if (k % 2) {
          std::string tok = "[";
          for (char32_t c : p) tok.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : static_cast<char>(c));
          tok += "]";
          auto it = sym2id.find(tok);
          if (it != sym2id.end()) {
            r.ids.push_back(it->second);
            r.ids.push_back(SP);
            r.word_idx.push_back(-1);
            r.word_idx.push_back(-1);
          }
        } else {
          size_t a = 0, b = p.size();  // p.strip(" []")
          auto strip_ch = [](char32_t c) { return c == U' ' || c == U'[' || c == U']'; };
          while (a < b && strip_ch(p[a])) ++a;
          while (b > a && strip_ch(p[b - 1])) --b;
          if (a == b) continue;
          u32str q = p;
          for (char32_t& c : q)
            if (c == U'[' || c == U']') c = U' ';
          Phonemes s = to_phonemes(q);
          for (size_t i = 1; i < s.ids.size(); ++i) {
            r.ids.push_back(s.ids[i]);
            r.word_idx.push_back(s.word_idx[i] >= 0 ? s.word_idx[i] + nw : -1);
          }
          int mx = -1;
          for (int32_t w : r.word_idx) mx = std::max(mx, static_cast<int>(w));
          nw = mx + 1;
        }
      }
      return r;
    }
  }
  std::vector<int> raw = g2p->call(text);
  raw.push_back(detail::TOK_SPACE);
  Phonemes r;
  r.ids.push_back(SP);
  r.word_idx.push_back(-1);
  std::vector<int> cur;
  int wi = -1;
  auto flush = [&]() {
    ++wi;
    for (int p : cur) {
      r.ids.push_back(p);
      r.word_idx.push_back(wi);
    }
    cur.clear();
  };
  for (int tok : raw) {
    if (tok == detail::TOK_SPACE) {
      if (!cur.empty()) {
        flush();
        r.ids.push_back(SP);
        r.word_idx.push_back(-1);
      }
      continue;
    }
    if (tok >= 2 && tok <= 8) {  // _PUNCT
      if (!cur.empty()) flush();
      r.ids.push_back(tok);
      r.word_idx.push_back(-1);
    } else {
      cur.push_back(tok);
    }
  }
  return r;
}

// ------------------------------------------------------------------------------------------------ public API
Frontend::Frontend(const std::string& asset_dir) : impl_(new Impl(asset_dir)) {}
Frontend::~Frontend() = default;

Phonemes Frontend::to_phonemes(const std::string& utf8) const { return impl_->to_phonemes(detail::utf8_decode(utf8)); }

std::string Frontend::normalize(const std::string& utf8) const {
  return detail::utf8_encode(impl_->normalize(detail::utf8_decode(utf8)));
}

const std::vector<std::string>& Frontend::symbols() const { return impl_->symbols; }

int Frontend::symbol_id(const std::string& s) const {
  auto it = impl_->sym2id.find(s);
  return it == impl_->sym2id.end() ? -1 : it->second;
}

bool Frontend::has_speech(const Phonemes& p) const {
  for (int64_t i : p.ids)
    if (i >= kFirstPhone) return true;
  return false;
}

}  // namespace steertts
