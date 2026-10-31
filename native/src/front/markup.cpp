// Port of steertts_edge/markup.py MarkupSynth.parse / _segment / _n_words (edge build: no prompt adapter).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <stdexcept>

#include "frontend_impl.h"

namespace steertts {

using detail::Regex;
using detail::u32str;

namespace {

// markup.PROSODY, in its dict order (bit i of a ProsSet = PROS[i])
struct Pros {
  const char* name;
  bool has_pitch, has_energy, has_dur;
  double pitch, energy, dur;
};
const Pros PROS[] = {{"emph", true, true, true, 2.5, 4.0, 1.2}, {"slow", false, false, true, 0, 0, 1.4},
                     {"fast", false, false, true, 0, 0, 0.75},  {"high", true, false, false, 3.0, 0, 0},
                     {"low", true, false, false, -3.0, 0, 0},   {"loud", false, true, false, 0, 6.0, 0},
                     {"soft", false, true, false, 0, -6.0, 0}};
constexpr int N_PROS = 7;

int pros_index(const std::string& n) {
  for (int i = 0; i < N_PROS; ++i)
    if (n == PROS[i].name) return i;
  return -1;
}

struct Style {
  std::string tag;
  std::vector<std::pair<std::string, float>> ctrl;  // sorted
  bool operator==(const Style& o) const { return tag == o.tag && ctrl == o.ctrl; }
  bool operator!=(const Style& o) const { return !(*this == o); }
};

struct Piece {
  enum Kind { TEXT, PAUSE, BREAK, EVENT } kind = TEXT;
  u32str text;
  Style style;
  unsigned pros = 0;  // frozenset(pros_stack) as a bit set
  int ms = 0;
};

std::string narrow_utf8(const u32str& s) { return detail::utf8_encode(s); }

bool is_lit(const u32str& s, const char* lit) {
  size_t n = 0;
  for (; lit[n]; ++n)
    if (n >= s.size() || s[n] != static_cast<char32_t>(static_cast<unsigned char>(lit[n]))) return false;
  return n == s.size();
}

}  // namespace

std::vector<Segment> parse_markup(const Frontend& fe, const std::string& text_utf8, const std::string& default_tag,
                                  const std::vector<std::pair<std::string, float>>& default_ctrl,
                                  const std::vector<std::string>& known_tags,
                                  const std::set<std::string>& bank_events) {
  const Frontend::Impl& I = fe.impl();
  const detail::Ucd& u = I.ucd;
  std::set<std::string> tag2i(known_tags.begin(), known_tags.end());
  const std::string prompt_err =
      "free-text prompts are resolved on the host (the prompt adapter is not on device): send the resolved tag + ctrl";

  // d_tag, d_ctrl = _style_of_prompt(None) -> ("none", {}); tag / ctrl override
  std::string d_tag = "none";
  if (!default_tag.empty() && tag2i.count(default_tag)) d_tag = default_tag;
  std::vector<std::pair<std::string, float>> d_ctrl;  // dict: insertion order, later keys overwrite
  for (auto& kv : default_ctrl) {
    auto it = std::find_if(d_ctrl.begin(), d_ctrl.end(), [&](const auto& e) { return e.first == kv.first; });
    if (it != d_ctrl.end())
      it->second = kv.second;
    else
      d_ctrl.push_back(kv);
  }
  std::sort(d_ctrl.begin(), d_ctrl.end());  // tuple(sorted(d_ctrl.items()))

  u32str text = I.emph.sub(detail::utf8_decode(text_utf8), [](const Regex::Match& m, const u32str& s) {
    return U"<emph>" + m.str(s, 1) + U"</emph>";
  });

  std::vector<Piece> runs;
  std::vector<Style> style_stack{Style{d_tag, d_ctrl}};
  std::vector<int> pros_stack;
  auto pros_set = [&]() {
    unsigned b = 0;
    for (int p : pros_stack) b |= 1u << p;
    return b;
  };
  auto text_run = [&](const u32str& t) {
    Piece p;
    p.text = t;
    p.style = style_stack.back();
    p.pros = pros_set();
    runs.push_back(p);
  };
  int pos = 0;
  for (const Regex::Match& m : I.token.finditer(text)) {
    if (m.start > pos) text_run(text.substr(pos, m.start - pos));
    pos = m.end;
    if (m.has(5)) {  // [pause ...] / [event]
      u32str name = u.lower_small(m.str(text, 6));
      if (is_lit(name, "pause")) {
        int ms = 400;
        if (m.has(7)) {
          u32str v = m.str(text, 7);
          u32str unit = m.has(8) ? u.lower_small(m.str(text, 8)) : u32str(U"ms");
          std::string a;
          int dots = 0, digits = 0;
          for (char32_t c : v) {
            int d = u.digit(c);
            if (d >= 0) {
              a.push_back(static_cast<char>('0' + d));
              ++digits;
            } else {
              a.push_back('.');
              ++dots;
            }
          }
          if (dots > 1 || digits == 0)
            throw std::invalid_argument("could not convert string to float: '" + narrow_utf8(v) + "'");
          double x = std::strtod(a.c_str(), nullptr) * (is_lit(unit, "s") ? 1000.0 : 1.0);
          if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");
          double tr = std::trunc(x);
          ms = tr > 5000 ? 5000 : static_cast<int>(tr);
        }
        Piece p;
        p.kind = Piece::PAUSE;
        p.ms = std::min(ms, 5000);
        runs.push_back(p);
      } else if (bank_events.count(narrow_utf8(name))) {  // a recorded event from the bank: its own chunk
        Piece p;
        p.kind = Piece::EVENT;
        p.text = name;
        runs.push_back(p);
      } else {  // event token the bank lacks (or unknown: dropped by to_phonemes) stays in the TTS text
        text_run(U" [" + name + U"] ");
      }
      continue;
    }
    bool closing = m.caps[2 * 2 + 1] > m.caps[2 * 2];  // bool(m.group(2))
    std::string name = narrow_utf8(u.lower_small(m.str(text, 3)));
    int pi = pros_index(name);
    if (pi >= 0) {
      if (closing) {
        auto it = std::find(pros_stack.begin(), pros_stack.end(), pi);
        if (it != pros_stack.end()) pros_stack.erase(it);  // list.remove: first occurrence
      } else {
        pros_stack.push_back(pi);
      }
    } else if (name == "style" || tag2i.count(name)) {
      if (closing) {
        if (style_stack.size() > 1) style_stack.pop_back();
      } else if (name == "style") {
        if (m.has(4) && m.caps[9] > m.caps[8]) throw std::invalid_argument(prompt_err);  // non-empty prompt
        style_stack.push_back(Style{"none", {}});
      } else {
        style_stack.push_back(Style{name, {}});  // a named style tag resets the default controls
      }
    }
    // unknown tag: the marker is dropped, its inner text is spoken normally
  }
  if (pos < static_cast<int>(text.size())) text_run(text.substr(pos));

  // split runs at sentence ends so each sentence is its own (streamable) segment
  std::vector<Piece> pieces;
  for (const Piece& r : runs) {
    if (r.kind == Piece::PAUSE || r.kind == Piece::EVENT || is_lit(r.text, "PAUSE") || is_lit(r.text, "EVENT")) {
      pieces.push_back(r);
      continue;
    }
    std::vector<u32str> parts = I.sent_end.split(r.text);
    for (size_t i = 0; i < parts.size(); ++i) {
      Piece p = r;
      p.text = parts[i];
      pieces.push_back(p);
      if (i + 1 < parts.size()) {
        Piece b;
        b.kind = Piece::BREAK;
        pieces.push_back(b);
      }
    }
  }

  // _n_words with a one-entry memo (the same prefix is often measured twice; results are deterministic)
  u32str memo_text;
  int memo_n = -1;
  auto n_words = [&](const u32str& t) {
    if (memo_n >= 0 && t == memo_text) return memo_n;
    int n = 0;
    if (!I.strip(t).empty()) {
      Phonemes p = I.to_phonemes(t);
      int mx = *std::max_element(p.word_idx.begin(), p.word_idx.end());
      n = mx >= 0 ? mx + 1 : 0;
    }
    memo_text = t;
    memo_n = n;
    return n;
  };
  auto make_segment = [&](const std::vector<Piece>& cur, const Style& style) {
    u32str t;
    Segment seg;
    for (const Piece& p : cur) {
      int before = n_words(t);
      t += p.text;
      if (p.pros) {
        int after = n_words(t);
        double pitch = 0, energy = 0, dur = 1.0;
        for (int i = 0; i < N_PROS; ++i) {
          if (!(p.pros & (1u << i))) continue;
          if (PROS[i].has_pitch) pitch += PROS[i].pitch;
          if (PROS[i].has_energy) energy += PROS[i].energy;
          if (PROS[i].has_dur) dur *= PROS[i].dur;
        }
        WordShift sh{static_cast<float>(pitch), static_cast<float>(energy), static_cast<float>(dur)};
        for (int w = before; w < after; ++w) seg.word_shift[w] = sh;
      }
    }
    seg.text = detail::utf8_encode(I.strip(I.collapse_ws(t)));
    seg.tag = style.tag;
    seg.ctrl = style.ctrl;
    return seg;
  };

  // group consecutive pieces with the same style into segments
  std::vector<Segment> segs;
  std::vector<Piece> cur;
  Style cur_style;
  bool have_style = false;
  auto flush = [&]() {
    if (cur.empty()) return;
    u32str all;
    for (auto& p : cur) all += p.text;
    if (!I.strip(all).empty()) segs.push_back(make_segment(cur, cur_style));
  };
  for (const Piece& p : pieces) {
    bool pause = p.kind == Piece::PAUSE, brk = p.kind == Piece::BREAK, event = p.kind == Piece::EVENT;
    bool txt_pause = p.kind == Piece::TEXT && is_lit(p.text, "PAUSE");
    bool txt_event = p.kind == Piece::TEXT && is_lit(p.text, "EVENT");
    bool txt_break = p.kind == Piece::TEXT && is_lit(p.text, "BREAK");
    if (pause || brk || event || txt_pause || txt_event || txt_break) {
      flush();
      cur.clear();
      have_style = false;
      if (pause || txt_pause) {
        Segment s;
        s.tag = "none";
        s.pause_ms = pause ? p.ms : Segment::kPauseIsStyleTuple;  // Python bug: pause_ms = the run's style tuple
        segs.push_back(s);
      } else if (event) {
        Segment s;
        s.tag = "none";
        s.event = narrow_utf8(p.text);
        segs.push_back(s);
      } else if (txt_event) {
        Segment s;
        s.tag = "none";
        s.event = Segment::kEventIsStyleTuple;  // Python bug: event = the run's style tuple
        segs.push_back(s);
      }
      continue;
    }
    if (have_style && p.style != cur_style) {
      flush();
      cur.clear();
    }
    cur_style = p.style;
    have_style = true;
    cur.push_back(p);
  }
  flush();
  return segs;
}

}  // namespace steertts
