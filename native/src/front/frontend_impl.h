// Internal state of steertts::Frontend (port of steertts_edge/text.py: normalize, _expand, to_phonemes).
#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "g2p.h"
#include "pyre.h"
#include "steertts/front.h"
#include "ucd.h"

namespace steertts {

struct Frontend::Impl {
  explicit Impl(const std::string& dir);

  detail::Ucd ucd;
  std::vector<std::string> symbols;
  std::unordered_map<std::string, int> sym2id;
  std::unique_ptr<detail::EdgeG2p> g2p;

  detail::u32str normalize(const detail::u32str& text) const;
  detail::u32str expand(const detail::u32str& text) const;
  Phonemes to_phonemes(const detail::u32str& text) const;
  std::string num(const std::string& digits, bool ordinal) const;  // text._num on an int
  std::string num_float(const detail::u32str& s) const;            // text._num on float(s)
  detail::u32str strip(const detail::u32str& s) const;             // str.strip()
  detail::u32str collapse_ws(const detail::u32str& s) const;       // re.sub(r"\s+", " ", s)

  detail::Regex quotes, range, by, compound, currency, percent, ordinal, comma_num, decimal, long_digits, camel_acr, acronym, ws,
      event;
  std::vector<std::pair<detail::Regex, std::string>> units;
  // markup
  detail::Regex emph, token, sent_end;
};

}  // namespace steertts
