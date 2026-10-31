// Unicode helpers for the SteerTTS front end: UTF-8 <-> UTF-32 and the Python character tables that
// tools/build_native_assets.py extracts from CPython into assets/front/unicode.bin (see that script for the format).
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace steertts {
namespace detail {

using u32str = std::u32string;

// Lenient decoder: invalid sequences become U+FFFD (Python strings are always valid; the golden corpus only has
// valid UTF-8).
u32str utf8_decode(const std::string& s);
std::string utf8_encode(const u32str& s);
void utf8_append(std::string& out, char32_t c);

class Ucd {
 public:
  Ucd() = default;
  void load(const std::string& path);

  bool is_word(char32_t c) const {  // re \w
    if (c < 128) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
    return in_ranges(word_, c);
  }
  bool is_space(char32_t c) const {  // re \s, str.isspace(), str.strip()
    if (c < 128) return c == ' ' || (c >= 9 && c <= 13) || (c >= 28 && c <= 31);
    for (char32_t s : space_)
      if (s == c) return true;
    return false;
  }
  int digit(char32_t c) const;  // re \d value (0-9), -1 if not a decimal digit
  bool is_digit(char32_t c) const { return digit(c) >= 0; }
  // re.IGNORECASE folding used by the regex engine: ascii letters -> lower case, the few non-ascii code points sre
  // treats as equal to an ascii letter (U+0130, U+0131, U+017F, U+212A) -> that letter, everything else unchanged.
  char32_t fold(char32_t c) const {
    if (c >= 'A' && c <= 'Z') return c + 32;
    if (c < 128) return c;
    for (auto& f : fold_)
      if (f.first == c) return f.second;
    return c;
  }
  // str.lower() for ascii and for the fold_ code points (the only non-ascii ones markup tag names can contain).
  u32str lower_small(const u32str& s) const;
  // g2p_en clean step for one code point: NFD, drop Mn, lower, keep [ a-z'.,?!-]. Returns the ascii result.
  const std::string& g2p_clean(char32_t c) const;

 private:
  static bool in_ranges(const std::vector<std::pair<char32_t, char32_t>>& r, char32_t c);
  std::vector<std::pair<char32_t, char32_t>> word_;
  std::vector<char32_t> space_;
  struct DR { char32_t lo, hi; uint32_t v0; };
  std::vector<DR> digit_;
  std::unordered_map<char32_t, std::string> g2p_;
  std::string ascii_g2p_[128];
  std::vector<std::pair<char32_t, char32_t>> fold_;
  std::unordered_map<char32_t, u32str> lower_;
  std::string empty_;
};

}  // namespace detail
}  // namespace steertts
