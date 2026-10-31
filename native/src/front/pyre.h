// A small backtracking regex engine with Python `re` (sre) semantics for str patterns, covering exactly the syntax the
// SteerTTS front end uses: literals, `.`, classes with ranges / negation / \d \s \w \D \S \W, groups (capturing,
// (?:...)), alternation (ordered, leftmost wins), greedy and lazy quantifiers * + ? {m} {m,} {m,n}, anchors ^ $ \Z,
// \b \B, lookahead (?= ) (?! ), fixed-width lookbehind (?<= ) (?<! ), back references \1..\9 and IGNORECASE.
// Unicode classes (\w \s \d) and case folding come from the CPython tables in unicode.bin (Ucd).
// Like sre it is a plain depth-first backtracker, so it returns the same match sre does (same leftmost start,
// same alternative / repetition choices) - no DFA tricks that could pick a different match.
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ucd.h"

namespace steertts {
namespace detail {

struct RxNode;

class Regex {
 public:
  struct Match {
    int start = -1, end = -1;
    std::vector<int> caps;  // 2 per group, group 0 = whole match; -1 = group did not participate
    bool has(int g) const { return caps[2 * g] >= 0; }
    u32str str(const u32str& s, int g) const {
      return has(g) ? s.substr(caps[2 * g], caps[2 * g + 1] - caps[2 * g]) : u32str();
    }
  };
  using Repl = std::function<u32str(const Match&, const u32str&)>;

  Regex();
  Regex(const Ucd* u, const std::string& utf8_pattern, bool icase = false);
  ~Regex();
  Regex(Regex&&) noexcept;
  Regex& operator=(Regex&&) noexcept;
  Regex(const Regex&) = delete;
  Regex& operator=(const Regex&) = delete;

  int groups() const { return ngroups_; }
  bool match_at(const u32str& s, int pos, Match& m) const;               // re.match at pos
  bool search(const u32str& s, int pos, Match& m) const;                 // re.search from pos
  std::vector<Match> finditer(const u32str& s) const;                    // re.finditer
  u32str sub(const u32str& s, const Repl& f) const;                      // re.sub(pattern, f, s)
  std::vector<u32str> split(const u32str& s) const;                      // re.split (groups included, None -> "")

 private:
  const Ucd* u_ = nullptr;
  bool icase_ = false;
  int ngroups_ = 0;
  std::vector<std::unique_ptr<RxNode>> pool_;
  std::vector<RxNode*>* root_ = nullptr;
  std::vector<std::unique_ptr<std::vector<RxNode*>>> seqs_;
  void follow_walk(const std::vector<RxNode*>& q, const std::vector<const RxNode*>& outer, bool outer_ok);
  friend struct RxParser;
  friend struct RxMatcher;
};

}  // namespace detail
}  // namespace steertts
