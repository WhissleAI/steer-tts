// Golden parity test + micro benchmark for the SteerTTS C++ front end.
//
//   front_golden <asset_dir> <golden.jsonl> [golden_g2p.jsonl] [--tags a,b,...] [--verbose N]
//       exact-match rates for normalize / ids / word_idx / markup segments (+ GRU G2P words), timing
//   front_golden --bench <asset_dir> <golden.jsonl>
//       process that only loads the front end and converts every to_phonemes text of the corpus:
//       us/sentence (cold cache, warm cache) and peak RSS
//
// The default known_tags list is pkg/config.json "tags" of steertts-v1-edge.
#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "g2p.h"
#include "frontend_impl.h"
#include "steertts/front.h"

namespace {

// ------------------------------------------------------------------------------------------------ tiny JSON
struct J {
  enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<J> a;
  std::vector<std::pair<std::string, J>> o;
  const J* get(const char* k) const {
    for (auto& kv : o)
      if (kv.first == k) return &kv.second;
    return nullptr;
  }
  bool has(const char* k) const { return get(k) != nullptr; }
  const J& operator[](const char* k) const {
    const J* j = get(k);
    if (!j) throw std::runtime_error(std::string("missing key ") + k);
    return *j;
  }
};

struct JP {
  const std::string& s;
  size_t i = 0;
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
  }
  static void put(std::string& o, uint32_t c) {
    if (c < 0x80) {
      o += static_cast<char>(c);
    } else if (c < 0x800) {
      o += static_cast<char>(0xC0 | (c >> 6));
      o += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
      o += static_cast<char>(0xE0 | (c >> 12));
      o += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      o += static_cast<char>(0x80 | (c & 0x3F));
    } else {
      o += static_cast<char>(0xF0 | (c >> 18));
      o += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
      o += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      o += static_cast<char>(0x80 | (c & 0x3F));
    }
  }
  uint32_t hex4() {
    uint32_t v = std::stoul(s.substr(i, 4), nullptr, 16);
    i += 4;
    return v;
  }
  std::string str() {
    std::string o;
    ++i;  // "
    while (s[i] != '"') {
      if (s[i] == '\\') {
        char e = s[i + 1];
        i += 2;
        switch (e) {
          case 'n': o += '\n'; break;
          case 't': o += '\t'; break;
          case 'r': o += '\r'; break;
          case 'b': o += '\b'; break;
          case 'f': o += '\f'; break;
          case 'u': {
            uint32_t c = hex4();
            if (c >= 0xD800 && c < 0xDC00 && s[i] == '\\' && s[i + 1] == 'u') {
              i += 2;
              uint32_t d = hex4();
              c = 0x10000 + ((c - 0xD800) << 10) + (d - 0xDC00);
            }
            put(o, c);
            break;
          }
          default: o += e;
        }
      } else {
        o += s[i++];
      }
    }
    ++i;
    return o;
  }
  J val() {
    ws();
    J j;
    char c = s[i];
    if (c == '{') {
      j.t = J::OBJ;
      ++i;
      ws();
      if (s[i] == '}') { ++i; return j; }
      while (true) {
        ws();
        std::string k = str();
        ws();
        ++i;  // :
        j.o.emplace_back(k, val());
        ws();
        if (s[i++] == '}') break;
      }
    } else if (c == '[') {
      j.t = J::ARR;
      ++i;
      ws();
      if (s[i] == ']') { ++i; return j; }
      while (true) {
        j.a.push_back(val());
        ws();
        if (s[i++] == ']') break;
      }
    } else if (c == '"') {
      j.t = J::STR;
      j.s = str();
    } else if (c == 'n') {
      i += 4;
    } else if (c == 't') {
      j.t = J::BOOL; j.b = true; i += 4;
    } else if (c == 'f') {
      j.t = J::BOOL; i += 5;
    } else {
      j.t = J::NUM;
      size_t e = i;
      while (e < s.size() && std::strchr("+-0123456789.eE", s[e])) ++e;
      j.n = std::strtod(s.substr(i, e - i).c_str(), nullptr);
      i = e;
    }
    return j;
  }
};
J parse(const std::string& line) {
  JP p{line};
  return p.val();
}

std::string err_name(const std::exception& e) {
  if (dynamic_cast<const std::invalid_argument*>(&e)) return "ValueError: ";
  if (dynamic_cast<const std::overflow_error*>(&e)) return "OverflowError: ";
  if (dynamic_cast<const std::out_of_range*>(&e)) return "IndexError: ";
  return "C++Exception: ";
}

std::string show(const std::string& s, size_t n = 160) {
  std::string o = s.size() > n ? s.substr(0, n) + "..." : s;
  for (auto& c : o)
    if (c == '\n') c = '|';
  return o;
}

template <class A>
bool same_ints(const std::vector<A>& a, const J& b) {
  if (a.size() != b.a.size()) return false;
  for (size_t k = 0; k < a.size(); ++k)
    if (static_cast<double>(a[k]) != b.a[k].n) return false;
  return true;
}

long peak_rss_kb() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss;
}
long cur_rss_kb() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmRSS:", 0) == 0) return std::atol(line.c_str() + 6);
  return -1;
}

const char* DEFAULT_TAGS =
    "none,adoration,amazement,amusement,anger,anxiety,apologetic,bored,confidence,confusion,contempt,contentment,"
    "cuteness,desire,determined,disappointment,disgust,distress,ecstasy,embarrassment,emphasis,enunciated,excited,"
    "fast,fear,frustration,guilt,happiness,hesitant,high_pitch,interest,irritation,laughing,loud,low_pitch,"
    "mysterious,neutral,nostalgia,pain,panic,pride,realization,relief,sadness,sarcasm,serenity,skeptical,slow,"
    "sympathy,tired,whisper";

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string x;
  while (std::getline(ss, x, ',')) v.push_back(x);
  return v;
}

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

int bench(const std::string& assets, const std::string& golden) {
  std::vector<std::string> texts;
  {
    std::ifstream f(golden);
    std::string line;
    while (std::getline(f, line)) {
      J j = parse(line);
      if (j["k"].s == "ph" && !j.has("error")) texts.push_back(j["text"].s);
    }
  }
  long rss0 = cur_rss_kb();
  auto t0 = Clock::now();
  steertts::Frontend fe(assets);
  auto t1 = Clock::now();
  long rss1 = cur_rss_kb();
  size_t n_ids = 0;
  double cold = 0, warm = 0;
  for (int pass = 0; pass < 2; ++pass) {
    auto a = Clock::now();
    for (auto& t : texts) n_ids += fe.to_phonemes(t).ids.size();
    auto b = Clock::now();
    (pass ? warm : cold) = secs(a, b);
  }
  size_t chars = 0;
  for (auto& t : texts) chars += t.size();
  std::printf("bench: %zu sentences (avg %.1f bytes), load %.1f ms\n", texts.size(),
              static_cast<double>(chars) / texts.size(), 1e3 * secs(t0, t1));
  std::printf("  to_phonemes cold cache: %.1f us/sentence   warm cache: %.1f us/sentence   (%zu ids)\n",
              1e6 * cold / texts.size(), 1e6 * warm / texts.size(), n_ids / 2);
  std::printf("  RSS: before load %ld kB, after load %ld kB (front end ~%ld kB), peak %ld kB\n", rss0, rss1,
              rss1 - rss0, peak_rss_kb());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 4 && std::string(argv[1]) == "--bench") return bench(argv[2], argv[3]);
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <asset_dir> <golden.jsonl> [golden_g2p.jsonl] [--tags a,b] [--verbose N]\n"
                         "       %s --bench <asset_dir> <golden.jsonl>\n", argv[0], argv[0]);
    return 2;
  }
  std::string assets = argv[1], golden = argv[2], g2pfile;
  std::vector<std::string> tags = split_csv(DEFAULT_TAGS);
  int verbose = 10;
  for (int k = 3; k < argc; ++k) {
    std::string a = argv[k];
    if (a == "--tags" && k + 1 < argc)
      tags = split_csv(argv[++k]);
    else if (a == "--verbose" && k + 1 < argc)
      verbose = std::atoi(argv[++k]);
    else
      g2pfile = a;
  }
  auto t0 = Clock::now();
  steertts::Frontend fe(assets);
  std::printf("front end loaded in %.1f ms\n", 1e3 * secs(t0, Clock::now()));

  std::ifstream f(golden);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", golden.c_str());
    return 2;
  }
  struct Stat {
    long n = 0, ok = 0;
  };
  std::map<std::string, Stat> st;
  int shown = 0;
  auto report = [&](const std::string& what, const std::string& text, const std::string& detail) {
    if (shown++ < verbose)
      std::printf("MISMATCH %-8s text=\"%s\"\n    %s\n", what.c_str(), show(text).c_str(), show(detail, 600).c_str());
  };
  std::string line;
  double t_ph = 0, t_mk = 0;
  long n_ph_timed = 0;
  while (std::getline(f, line)) {
    J j = parse(line);
    const std::string& text = j["text"].s;
    if (j["k"].s == "ph") {
      // normalize
      {
        std::string got, gerr;
        try {
          got = fe.normalize(text);
        } catch (const std::exception& e) {
          gerr = err_name(e) + e.what();
        }
        bool ok = j.has("norm_error") ? gerr == j["norm_error"].s : (gerr.empty() && got == j["norm"].s);
        auto& s = st["normalize"];
        ++s.n;
        s.ok += ok;
        if (!ok)
          report("norm", text, "py=\"" + (j.has("norm_error") ? j["norm_error"].s : j["norm"].s) + "\"\n    c++=\"" +
                                   (gerr.empty() ? got : gerr) + "\"");
      }
      steertts::Phonemes p;
      std::string gerr;
      auto a = Clock::now();
      try {
        p = fe.to_phonemes(text);
      } catch (const std::exception& e) {
        gerr = err_name(e) + e.what();
      }
      t_ph += secs(a, Clock::now());
      ++n_ph_timed;
      if (j.has("error")) {
        bool ok = gerr == j["error"].s;
        auto& s = st["errors(ph)"];
        ++s.n;
        s.ok += ok;
        if (!ok) report("error", text, "py=" + j["error"].s + "\n    c++=" + (gerr.empty() ? "<no error>" : gerr));
        continue;
      }
      bool ok_ids = gerr.empty() && same_ints(p.ids, j["ids"]);
      bool ok_w = gerr.empty() && same_ints(p.word_idx, j["widx"]);
      st["ids"].n++;
      st["ids"].ok += ok_ids;
      st["word_idx"].n++;
      st["word_idx"].ok += ok_w;
      if (!ok_ids || !ok_w) {
        std::string py, cc;
        for (auto& x : j["ids"].a) py += fe.symbols()[static_cast<size_t>(x.n)] + " ";
        for (auto x : p.ids) cc += fe.symbols()[static_cast<size_t>(x)] + " ";
        std::string pw, cw;
        for (auto& x : j["widx"].a) pw += std::to_string(static_cast<int>(x.n)) + " ";
        for (auto x : p.word_idx) cw += std::to_string(x) + " ";
        report(ok_ids ? "word_idx" : "ids", text,
               "py =" + py + "\n    c++=" + (gerr.empty() ? cc : gerr) + "\n    pyw=" + pw + "\n    c+w=" + cw);
      }
    } else if (j["k"].s == "mk") {
      std::string tag = j["tag"].t == J::STR ? j["tag"].s : "";
      std::vector<std::pair<std::string, float>> ctrl;
      if (j["ctrl"].t == J::ARR)
        for (auto& kv : j["ctrl"].a) ctrl.emplace_back(kv.a[0].s, static_cast<float>(kv.a[1].n));
      std::vector<steertts::Segment> segs;
      std::string gerr;
      auto a = Clock::now();
      try {
        segs = steertts::parse_markup(fe, text, tag, ctrl, tags);
      } catch (const std::exception& e) {
        gerr = err_name(e) + e.what();
      }
      t_mk += secs(a, Clock::now());
      auto& s = st["markup"];
      ++s.n;
      if (j.has("error")) {
        bool ok = gerr == j["error"].s;
        s.ok += ok;
        st["errors(mk)"].n++;
        st["errors(mk)"].ok += ok;
        if (!ok) report("mk-error", text, "py=" + j["error"].s + "\n    c++=" + (gerr.empty() ? "<no error>" : gerr));
        continue;
      }
      const J& py = j["segs"];
      std::string why;
      if (!gerr.empty()) why = "c++ raised " + gerr;
      else if (segs.size() != py.a.size()) why = "segment count py=" + std::to_string(py.a.size()) + " c++=" + std::to_string(segs.size());
      for (size_t k = 0; why.empty() && k < segs.size(); ++k) {
        const J& ps = py.a[k];
        const steertts::Segment& cs = segs[k];
        std::string ks = "seg " + std::to_string(k) + ": ";
        if (cs.text != ps["text"].s) why = ks + "text py=\"" + ps["text"].s + "\" c++=\"" + cs.text + "\"";
        else if (cs.tag != ps["tag"].s) why = ks + "tag py=" + ps["tag"].s + " c++=" + cs.tag;
        else if (cs.pause_ms != static_cast<int>(ps["pause_ms"].n)) why = ks + "pause_ms";
        else if (cs.event != ps["event"].s) why = ks + "event";
        else if (cs.ctrl.size() != ps["ctrl"].a.size()) why = ks + "ctrl size";
        else {
          for (size_t c = 0; why.empty() && c < cs.ctrl.size(); ++c)
            if (cs.ctrl[c].first != ps["ctrl"].a[c].a[0].s ||
                cs.ctrl[c].second != static_cast<float>(ps["ctrl"].a[c].a[1].n))
              why = ks + "ctrl";
          const J& ws = ps["ws"];
          if (why.empty() && cs.word_shift.size() != ws.a.size())
            why = ks + "word_shift keys py=" + std::to_string(ws.a.size()) + " c++=" + std::to_string(cs.word_shift.size());
          size_t c = 0;
          for (auto it = cs.word_shift.begin(); why.empty() && it != cs.word_shift.end(); ++it, ++c) {
            const J& w = ws.a[c];
            if (it->first != static_cast<int>(w.a[0].n) || it->second.pitch != static_cast<float>(w.a[1].n) ||
                it->second.energy != static_cast<float>(w.a[2].n) || it->second.dur != static_cast<float>(w.a[3].n))
              why = ks + "word_shift[" + std::to_string(it->first) + "]";
          }
        }
      }
      s.ok += why.empty();
      if (!why.empty()) report("markup", text, why);
    }
  }
  long n_g2p = 0, ok_g2p = 0;
  double min_gap_bad = 1e9;
  if (!g2pfile.empty()) {
    std::ifstream g(g2pfile);
    const steertts::detail::GruG2p& gru = fe.impl().g2p->gru();
    auto a = Clock::now();
    while (std::getline(g, line)) {
      J j = parse(line);
      double gap = 0;
      std::vector<int> p = gru.predict_raw(j["w"].s, &gap);
      bool ok = same_ints(p, j["p"]);
      ++n_g2p;
      ok_g2p += ok;
      if (!ok) {
        min_gap_bad = std::min(min_gap_bad, j["gap"].n);
        report("g2p", j["w"].s, "py gap=" + std::to_string(j["gap"].n) + " c++ gap=" + std::to_string(gap));
      }
    }
    std::printf("g2p: %.1f us/word (GRU only)\n", 1e6 * secs(a, Clock::now()) / std::max(1L, n_g2p));
  }
  std::printf("\n%-12s %8s %8s %9s\n", "field", "cases", "exact", "rate");
  bool all = true;
  for (auto& kv : st) {
    std::printf("%-12s %8ld %8ld %8.3f%%\n", kv.first.c_str(), kv.second.n, kv.second.ok,
                100.0 * kv.second.ok / std::max(1L, kv.second.n));
    all = all && kv.second.ok == kv.second.n;
  }
  if (n_g2p) {
    std::printf("%-12s %8ld %8ld %8.3f%%\n", "gru_g2p", n_g2p, ok_g2p, 100.0 * ok_g2p / n_g2p);
    all = all && ok_g2p == n_g2p;
  }
  std::printf("\ntiming (test process, includes first-use GRU runs): to_phonemes %.1f us/case, parse_markup %.1f us/case\n",
              1e6 * t_ph / std::max(1L, n_ph_timed), 1e6 * t_mk / std::max(1L, st["markup"].n));
  std::printf("peak RSS %ld kB\n", peak_rss_kb());
  std::printf(all ? "ALL EXACT\n" : "MISMATCHES PRESENT\n");
  return all ? 0 : 1;
}
