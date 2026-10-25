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
      sent_end(&ucd, R"((?<=[.!?])\s+)") {
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
      {"Hz", "hertz"},                 {"GB", "gigabytes"},              {"MB", "megabytes"}};
  std::stable_sort(u.begin(), u.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  for (auto& e : u) {
    std::string pat;
    for (char c : e.first) pat += c == '/' ? std::string("[ /]") : std::string(1, c);  // re.escape is a no-op here
    units.emplace_back(Regex(&ucd, "(\\d)\\s?" + pat + "\\b"), e.second);
  }
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
  u32str t = range.sub(text0, [](const Regex::Match& m, const u32str& s) { return m.str(s, 1) + U" to " + m.str(s, 2); });
  t = by.sub(t, [](const Regex::Match& m, const u32str& s) { return m.str(s, 1) + U" by " + m.str(s, 2); });
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
      o += base[i];
    }
    if (g.back() == 's' && g.size() > 2) o += "'s";
    return A(o);
  });
  return t;
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
