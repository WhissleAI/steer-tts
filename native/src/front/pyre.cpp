#include "pyre.h"

#include <climits>
#include <stdexcept>

namespace steertts {
namespace detail {

using Seq = std::vector<RxNode*>;

struct RxClass {
  bool neg = false;
  std::vector<std::pair<char32_t, char32_t>> ranges;
  bool d = false, D = false, s = false, S = false, w = false, W = false;
};

struct RxNode {
  enum T { CHR, ANY, CLS, BOL, EOL, EOS, WB, NWB, GRP, ALT, REP, LOOK, BREF } t;
  char32_t c = 0;
  RxClass cls;
  int grp = -1;
  std::vector<Seq*> alts;  // ALT
  Seq* body = nullptr;     // GRP / REP / LOOK
  int mn = 0, mx = 0;
  bool greedy = true;
  bool behind = false, neg = false;
  int width = 0;
  bool det = false;  // REP whose body has at most one match per start position (see RxParser::deterministic)
  // single-character REP: the rest of its sequence must start with a character matching one of these nodes
  // (computed by RxParser::follow_sets; empty when the rest can match empty / is not analysable -> no pruning)
  std::vector<const RxNode*> follow;
  bool single() const { return t == CHR || t == ANY || t == CLS; }
};

// ------------------------------------------------------------------------------------------------ parser
struct RxParser {
  Regex& R;
  u32str p;
  size_t i = 0;

  [[noreturn]] void fail(const char* why) {
    throw std::logic_error(std::string("steertts front regex: ") + why + " in " + utf8_encode(p));
  }
  RxNode* node(RxNode::T t) {
    R.pool_.emplace_back(new RxNode());
    R.pool_.back()->t = t;
    return R.pool_.back().get();
  }
  Seq* seq() {
    R.seqs_.emplace_back(new Seq());
    return R.seqs_.back().get();
  }
  bool more() const { return i < p.size(); }

  Seq* parse_alt() {
    std::vector<Seq*> alts{parse_seq()};
    while (more() && p[i] == '|') {
      ++i;
      alts.push_back(parse_seq());
    }
    if (alts.size() == 1) return alts[0];
    RxNode* a = node(RxNode::ALT);
    a->alts = alts;
    Seq* s = seq();
    s->push_back(a);
    return s;
  }
  Seq* parse_seq() {
    Seq* s = seq();
    while (more() && p[i] != '|' && p[i] != ')') s->push_back(parse_quant(parse_atom()));
    return s;
  }
  bool parse_int(int& v) {
    size_t j = i;
    v = 0;
    while (j < p.size() && p[j] >= '0' && p[j] <= '9') v = v * 10 + static_cast<int>(p[j++] - '0');
    bool ok = j > i;
    i = j;
    return ok;
  }
  RxNode* parse_quant(RxNode* a) {
    if (!more()) return a;
    int mn, mx;
    size_t save = i;
    char32_t q = p[i];
    if (q == '*') {
      mn = 0, mx = INT_MAX, ++i;
    } else if (q == '+') {
      mn = 1, mx = INT_MAX, ++i;
    } else if (q == '?') {
      mn = 0, mx = 1, ++i;
    } else if (q == '{') {
      ++i;
      int a1 = 0, a2 = 0;
      bool h1 = parse_int(a1);
      if (more() && p[i] == '}') {
        if (!h1) { i = save; return a; }
        mn = mx = a1;
        ++i;
      } else if (more() && p[i] == ',') {
        ++i;
        bool h2 = parse_int(a2);
        if (!more() || p[i] != '}') { i = save; return a; }
        ++i;
        mn = h1 ? a1 : 0;
        mx = h2 ? a2 : INT_MAX;
      } else {
        i = save;
        return a;  // literal '{' (parse_atom will see it next)
      }
    } else {
      return a;
    }
    RxNode* r = node(RxNode::REP);
    r->mn = mn;
    r->mx = mx;
    r->body = seq();
    r->body->push_back(a);
    if (more() && p[i] == '?') {
      r->greedy = false;
      ++i;
    }
    if (a->t == RxNode::GRP && a->grp < 0) r->det = deterministic(*a->body);
    if (a->t == RxNode::BREF) r->det = true;  // \1{3,}: copies of a fixed string, one way to match each
    return r;
  }
  // ascii-only membership test usable before the Unicode tables are loaded (Python semantics for ascii)
  static bool ascii_in(const RxNode* n, char32_t c) {
    auto sp = [](char32_t x) { return x == ' ' || (x >= 9 && x <= 13) || (x >= 28 && x <= 31); };
    auto dg = [](char32_t x) { return x >= '0' && x <= '9'; };
    auto wd = [&](char32_t x) { return dg(x) || (x >= 'a' && x <= 'z') || (x >= 'A' && x <= 'Z') || x == '_'; };
    if (n->t == RxNode::CHR) return n->c == c;
    if (n->t == RxNode::ANY) return c != '\n';
    const RxClass& k = n->cls;
    bool r = false;
    for (auto& rg : k.ranges) r = r || (c >= rg.first && c <= rg.second);
    r = r || (k.d && dg(c)) || (k.D && !dg(c)) || (k.s && sp(c)) || (k.S && !sp(c)) || (k.w && wd(c)) || (k.W && !wd(c));
    return k.neg ? !r : r;
  }
  // A non-capturing group body has at most one match from any start position (so greedy repetition of it never
  // needs to backtrack into an iteration, and can run as a loop instead of recursion) if it is a sequence of
  // single-character nodes (or alternations of them) and greedy single-character repeats that are either fixed-count or followed by an ascii
  // literal they cannot match (e.g. ",\d{3}" or "\s*\."). It must also consume at least one character.
  // (?:x|y|...) where every alternative is one single-character node: always consumes exactly one character
  static bool one_char_alt(const RxNode* n) {
    if (n->t != RxNode::ALT) return false;
    for (const Seq* a : n->alts)
      if (a->size() != 1 || !a->front()->single()) return false;
    return true;
  }
  bool deterministic(const Seq& b) {
    if (R.icase_) return false;
    bool consumes = false;
    for (size_t k = 0; k < b.size(); ++k) {
      const RxNode* n = b[k];
      if (n->single() || one_char_alt(n)) {
        consumes = true;
        continue;
      }
      if (n->t != RxNode::REP || !n->greedy || n->body->size() != 1 || !n->body->front()->single()) return false;
      if (n->mn > 0) consumes = true;
      if (n->mn == n->mx) continue;
      if (k + 1 >= b.size() || b[k + 1]->t != RxNode::CHR || b[k + 1]->c >= 128) return false;
      if (ascii_in(n->body->front(), b[k + 1]->c)) return false;
    }
    return consumes;
  }
  char32_t escape_char(char32_t e) {
    switch (e) {
      case 'n': return '\n';
      case 't': return '\t';
      case 'r': return '\r';
      case 'f': return '\f';
      case 'v': return '\v';
      case 'a': return '\a';
      default:
        if ((e >= 'a' && e <= 'z') || (e >= 'A' && e <= 'Z') || (e >= '0' && e <= '9')) fail("unsupported escape");
        return e;
    }
  }
  RxNode* parse_class() {
    RxNode* n = node(RxNode::CLS);
    RxClass& c = n->cls;
    if (more() && p[i] == '^') {
      c.neg = true;
      ++i;
    }
    bool first = true;
    while (true) {
      if (!more()) fail("unterminated class");
      char32_t ch = p[i];
      if (ch == ']' && !first) {
        ++i;
        break;
      }
      first = false;
      char32_t lo;
      if (ch == '\\') {
        if (i + 1 >= p.size()) fail("bad escape");
        char32_t e = p[i + 1];
        i += 2;
        switch (e) {
          case 'd': c.d = true; continue;
          case 'D': c.D = true; continue;
          case 's': c.s = true; continue;
          case 'S': c.S = true; continue;
          case 'w': c.w = true; continue;
          case 'W': c.W = true; continue;
          default: lo = escape_char(e);
        }
      } else {
        lo = ch;
        ++i;
      }
      char32_t hi = lo;
      if (i + 1 < p.size() && p[i] == '-' && p[i + 1] != ']') {
        ++i;
        char32_t h = p[i];
        if (h == '\\') {
          if (i + 1 >= p.size()) fail("bad escape");
          h = escape_char(p[i + 1]);
          i += 2;
        } else {
          ++i;
        }
        hi = h;
        if (hi < lo) fail("bad range");
      }
      c.ranges.push_back({lo, hi});
    }
    return n;
  }
  int width(const Seq& s) {
    int w = 0;
    for (const RxNode* n : s) {
      switch (n->t) {
        case RxNode::CHR: case RxNode::ANY: case RxNode::CLS: w += 1; break;
        case RxNode::GRP: w += width(*n->body); break;
        case RxNode::ALT: {
          int a0 = width(*n->alts[0]);
          for (Seq* a : n->alts)
            if (width(*a) != a0) fail("variable-width lookbehind");
          w += a0;
          break;
        }
        case RxNode::REP:
          if (n->mn != n->mx) fail("variable-width lookbehind");
          w += n->mn * width(*n->body);
          break;
        case RxNode::BREF: fail("backref in lookbehind");
        default: break;
      }
    }
    return w;
  }
  RxNode* parse_atom() {
    char32_t ch = p[i++];
    switch (ch) {
      case '(': {
        RxNode* g;
        if (more() && p[i] == '?') {
          ++i;
          if (more() && p[i] == ':') {
            ++i;
            g = node(RxNode::GRP);
          } else if (more() && (p[i] == '=' || p[i] == '!')) {
            g = node(RxNode::LOOK);
            g->neg = p[i] == '!';
            ++i;
          } else if (i + 1 < p.size() && p[i] == '<' && (p[i + 1] == '=' || p[i + 1] == '!')) {
            g = node(RxNode::LOOK);
            g->behind = true;
            g->neg = p[i + 1] == '!';
            i += 2;
          } else {
            fail("unsupported group syntax");
          }
        } else {
          g = node(RxNode::GRP);
          g->grp = ++R.ngroups_;
        }
        g->body = parse_alt();
        if (!more() || p[i] != ')') fail("missing )");
        ++i;
        if (g->t == RxNode::LOOK && g->behind) g->width = width(*g->body);
        return g;
      }
      case '[': return parse_class();
      case '.': return node(RxNode::ANY);
      case '^': return node(RxNode::BOL);
      case '$': return node(RxNode::EOL);
      case '\\': {
        if (!more()) fail("trailing backslash");
        char32_t e = p[i++];
        RxNode* n;
        switch (e) {
          case 'd': case 'D': case 's': case 'S': case 'w': case 'W':
            n = node(RxNode::CLS);
            n->cls.d = e == 'd'; n->cls.D = e == 'D'; n->cls.s = e == 's';
            n->cls.S = e == 'S'; n->cls.w = e == 'w'; n->cls.W = e == 'W';
            return n;
          case 'b': return node(RxNode::WB);
          case 'B': return node(RxNode::NWB);
          case 'Z': return node(RxNode::EOS);
          case 'A': return node(RxNode::BOL);
          default:
            if (e >= '1' && e <= '9') {
              n = node(RxNode::BREF);
              n->grp = static_cast<int>(e - '0');
              return n;
            }
            n = node(RxNode::CHR);
            n->c = escape_char(e);
            return n;
        }
      }
      default: {
        RxNode* n = node(RxNode::CHR);
        n->c = ch;
        return n;
      }
    }
  }
};

namespace {
// FIRST set of q[i..]: single-character nodes one of which must match the first consumed character; returns false
// if the suffix can match without consuming (or contains a back reference), i.e. no usable restriction.
bool first_set(const Seq& q, size_t i, std::vector<const RxNode*>& out) {
  for (; i < q.size(); ++i) {
    const RxNode* n = q[i];
    switch (n->t) {
      case RxNode::CHR: case RxNode::ANY: case RxNode::CLS:
        out.push_back(n);
        return true;
      case RxNode::BOL: case RxNode::EOL: case RxNode::EOS: case RxNode::WB: case RxNode::NWB: case RxNode::LOOK:
        continue;  // zero width: ignoring them only makes the set larger (still a valid necessary condition)
      case RxNode::GRP:
        if (first_set(*n->body, 0, out)) return true;
        continue;
      case RxNode::ALT: {
        bool all = true;
        for (const Seq* a : n->alts) all = first_set(*a, 0, out) && all;
        if (all) return true;
        continue;
      }
      case RxNode::REP:
        if (first_set(*n->body, 0, out) && n->mn > 0) return true;
        continue;
      case RxNode::BREF:
        return false;
    }
  }
  return false;
}
}  // namespace

Regex::Regex(const Ucd* u, const std::string& pat, bool icase) : u_(u), icase_(icase) {
  RxParser ps{*this, utf8_decode(pat)};
  root_ = ps.parse_alt();
  if (ps.more()) ps.fail("unbalanced )");
  follow_walk(*root_, {}, false);
}

// Assign REP::follow. `outer` / `outer_ok`: what must follow the end of `q` (ok = false: anything / nothing, e.g. the
// end of the whole pattern, a lookaround body or a repeated group body - then no pruning past q's end).
void Regex::follow_walk(const std::vector<RxNode*>& q, const std::vector<const RxNode*>& outer, bool outer_ok) {
  for (size_t i = 0; i < q.size(); ++i) {
    RxNode* n = q[i];
    std::vector<const RxNode*> f;
    bool ok = first_set(q, i + 1, f);
    if (!ok && outer_ok) {
      f.insert(f.end(), outer.begin(), outer.end());
      ok = true;
    }
    if (n->t == RxNode::REP && n->body->size() == 1 && n->body->front()->single()) {
      if (ok) n->follow = f;
    } else if (n->t == RxNode::GRP) {
      follow_walk(*n->body, f, ok);
    } else if (n->t == RxNode::ALT) {
      for (Seq* a : n->alts) follow_walk(*a, f, ok);
    } else if (n->t == RxNode::REP) {
      follow_walk(*n->body, {}, false);
    } else if (n->t == RxNode::LOOK) {
      follow_walk(*n->body, {}, false);
    }
  }
}
Regex::Regex() = default;
Regex::~Regex() = default;
Regex::Regex(Regex&&) noexcept = default;
Regex& Regex::operator=(Regex&&) noexcept = default;

// ------------------------------------------------------------------------------------------------ matcher
struct K {
  virtual bool operator()(int pos) const = 0;
 protected:
  ~K() = default;
};

struct RxMatcher {
  const Regex& R;
  const char32_t* s;
  int n;
  std::vector<int>& caps;

  bool base_cls(const RxClass& c, char32_t ch) const {
    for (auto& r : c.ranges)
      if (ch >= r.first && ch <= r.second) return true;
    const Ucd& u = *R.u_;
    if (c.d && u.is_digit(ch)) return true;
    if (c.D && !u.is_digit(ch)) return true;
    if (c.s && u.is_space(ch)) return true;
    if (c.S && !u.is_space(ch)) return true;
    if (c.w && u.is_word(ch)) return true;
    if (c.W && !u.is_word(ch)) return true;
    return false;
  }
  bool cls(const RxClass& c, char32_t ch) const {
    bool r = base_cls(c, ch);
    if (!r && R.icase_) {
      char32_t f = R.u_->fold(ch);
      if (f != ch) r = base_cls(c, f);
      if (!r && f >= 'a' && f <= 'z') r = base_cls(c, f - 32);
    }
    return c.neg ? !r : r;
  }
  bool single(const RxNode* nd, char32_t ch) const {
    switch (nd->t) {
      case RxNode::CHR: return R.icase_ ? R.u_->fold(ch) == R.u_->fold(nd->c) : ch == nd->c;
      case RxNode::ANY: return ch != '\n';
      default: return cls(nd->cls, ch);
    }
  }
  bool word_at(int p) const { return p >= 0 && p < n && R.u_->is_word(s[p]); }

  bool seq(const Seq& q, size_t i, int pos, const K& k);
  bool rep(const RxNode* nd, int count, int pos, const K& k);
};

namespace {
struct KTrue final : K {
  bool operator()(int) const override { return true; }
};
struct KAt final : K {
  int at;
  explicit KAt(int a) : at(a) {}
  bool operator()(int p) const override { return p == at; }
};
struct KEnd final : K {
  int* out;
  explicit KEnd(int* o) : out(o) {}
  bool operator()(int p) const override {
    *out = p;
    return true;
  }
};
struct KRest final : K {
  RxMatcher& m;
  const Seq& q;
  size_t i;
  const K& k;
  KRest(RxMatcher& m_, const Seq& q_, size_t i_, const K& k_) : m(m_), q(q_), i(i_), k(k_) {}
  bool operator()(int p) const override { return m.seq(q, i, p, k); }
};
struct KGrp final : K {
  RxMatcher& m;
  int g, start;
  const K& k;
  KGrp(RxMatcher& m_, int g_, int s_, const K& k_) : m(m_), g(g_), start(s_), k(k_) {}
  bool operator()(int p) const override {
    int o0 = m.caps[2 * g], o1 = m.caps[2 * g + 1];
    m.caps[2 * g] = start;
    m.caps[2 * g + 1] = p;
    if (k(p)) return true;
    m.caps[2 * g] = o0;
    m.caps[2 * g + 1] = o1;
    return false;
  }
};
struct KRep final : K {
  RxMatcher& m;
  const RxNode* nd;
  int count, start;
  const K& k;
  KRep(RxMatcher& m_, const RxNode* nd_, int c_, int s_, const K& k_) : m(m_), nd(nd_), count(c_), start(s_), k(k_) {}
  bool operator()(int p) const override {
    if (p == start) return false;  // an empty iteration ends the loop (sre does the same)
    return m.rep(nd, count, p, k);
  }
};
}  // namespace

bool RxMatcher::rep(const RxNode* nd, int count, int pos, const K& k) {
  if (nd->greedy) {
    if (count < nd->mx) {
      KRep kr(*this, nd, count + 1, pos, k);
      if (seq(*nd->body, 0, pos, kr)) return true;
    }
    return count >= nd->mn && k(pos);
  }
  if (count >= nd->mn && k(pos)) return true;
  if (count < nd->mx) {
    KRep kr(*this, nd, count + 1, pos, k);
    return seq(*nd->body, 0, pos, kr);
  }
  return false;
}

bool RxMatcher::seq(const Seq& q, size_t i, int pos, const K& k) {
  const Ucd& u = *R.u_;
  for (; i < q.size(); ++i) {
    const RxNode* nd = q[i];
    switch (nd->t) {
      case RxNode::CHR: case RxNode::ANY: case RxNode::CLS:
        if (pos < n && single(nd, s[pos])) {
          ++pos;
          continue;
        }
        return false;
      case RxNode::BOL:
        if (pos != 0) return false;
        continue;
      case RxNode::EOL:
        if (!(pos == n || (pos == n - 1 && s[pos] == '\n'))) return false;
        continue;
      case RxNode::EOS:
        if (pos != n) return false;
        continue;
      case RxNode::WB: case RxNode::NWB: {
        bool b = word_at(pos - 1) != word_at(pos);
        if (b != (nd->t == RxNode::WB)) return false;
        continue;
      }
      case RxNode::LOOK: {
        bool r;
        if (!nd->behind) {
          KTrue kt;
          r = seq(*nd->body, 0, pos, kt);
        } else {
          int st = pos - nd->width;
          KAt ka(pos);
          r = st >= 0 && seq(*nd->body, 0, st, ka);
        }
        if (r == nd->neg) return false;
        continue;
      }
      case RxNode::BREF: {
        int g = nd->grp;
        if (caps[2 * g] < 0) return false;
        int a = caps[2 * g], len = caps[2 * g + 1] - a;
        if (pos + len > n) return false;
        for (int t = 0; t < len; ++t) {
          char32_t x = s[a + t], y = s[pos + t];
          if (R.icase_ ? u.fold(x) != u.fold(y) : x != y) return false;
        }
        pos += len;
        continue;
      }
      case RxNode::GRP: {
        KRest rest(*this, q, i + 1, k);
        if (nd->grp < 0) return seq(*nd->body, 0, pos, rest);
        KGrp kg(*this, nd->grp, pos, rest);
        return seq(*nd->body, 0, pos, kg);
      }
      case RxNode::ALT: {
        KRest rest(*this, q, i + 1, k);
        for (const Seq* a : nd->alts)
          if (seq(*a, 0, pos, rest)) return true;
        return false;
      }
      case RxNode::REP: {
        KRest rest(*this, q, i + 1, k);
        const RxNode* b = nd->body->front();
        if (nd->body->size() == 1 && b->single()) {  // x*, [..]{m,n}: count the run, then try lengths
          int mx = nd->mx, c = 0;
          const auto& fol = nd->follow;
          auto viable = [&](int p) {  // can the rest of the sequence start at p? (exact pruning, sre does the same)
            if (fol.empty()) return true;
            if (p >= n) return false;
            for (const RxNode* f : fol)
              if (single(f, s[p])) return true;
            return false;
          };
          if (nd->greedy) {
            while (c < mx && pos + c < n && single(b, s[pos + c])) ++c;
            for (; c >= nd->mn; --c)
              if (viable(pos + c) && rest(pos + c)) return true;
            return false;
          }
          while (c < nd->mn) {
            if (pos + c < n && single(b, s[pos + c]))
              ++c;
            else
              return false;
          }
          while (true) {
            if (viable(pos + c) && rest(pos + c)) return true;
            if (c >= mx || pos + c >= n || !single(b, s[pos + c])) return false;
            ++c;
          }
        }
        if (nd->det && nd->greedy) {  // unique body match per position: collect iterations, then back off
          std::vector<int> ps{pos};
          int p = pos;
          while (static_cast<int>(ps.size()) - 1 < nd->mx) {
            int e = -1;
            KEnd ke(&e);
            if (!seq(*nd->body, 0, p, ke) || e == p) break;
            ps.push_back(e);
            p = e;
          }
          for (int c = static_cast<int>(ps.size()) - 1; c >= nd->mn; --c)
            if (rest(ps[c])) return true;
          return false;
        }
        return rep(nd, 0, pos, rest);
      }
    }
  }
  return k(pos);
}

bool Regex::match_at(const u32str& s, int pos, Match& m) const {
  m.caps.assign(2 * (ngroups_ + 1), -1);
  RxMatcher mt{*this, s.data(), static_cast<int>(s.size()), m.caps};
  int end = -1;
  KEnd ke(&end);
  if (!mt.seq(*root_, 0, pos, ke)) return false;
  m.start = pos;
  m.end = end;
  m.caps[0] = pos;
  m.caps[1] = end;
  return true;
}

namespace {
// a single-character node every match must start with (descends into groups and x+ / x{m,} with m >= 1), or null
const RxNode* first_char(const Seq& q) {
  if (q.empty()) return nullptr;
  const RxNode* n = q.front();
  if (n->single()) return n;
  if (n->t == RxNode::GRP) return first_char(*n->body);
  if (n->t == RxNode::REP && n->mn >= 1) return first_char(*n->body);
  return nullptr;
}
}  // namespace

bool Regex::search(const u32str& s, int pos, Match& m) const {
  int n = static_cast<int>(s.size());
  const RxNode* first = first_char(*root_);
  bool filt = first != nullptr;
  m.caps.assign(2 * (ngroups_ + 1), -1);
  RxMatcher mt{*this, s.data(), n, m.caps};
  for (int st = pos; st <= n; ++st) {
    if (filt && (st >= n || !mt.single(first, s[st]))) continue;
    int end = -1;
    KEnd ke(&end);
    std::fill(m.caps.begin(), m.caps.end(), -1);
    if (mt.seq(*root_, 0, st, ke)) {
      m.start = st;
      m.end = end;
      m.caps[0] = st;
      m.caps[1] = end;
      return true;
    }
  }
  return false;
}

std::vector<Regex::Match> Regex::finditer(const u32str& s) const {
  std::vector<Match> out;
  int pos = 0, n = static_cast<int>(s.size());
  Match m;
  while (pos <= n && search(s, pos, m)) {
    out.push_back(m);
    pos = m.end > m.start ? m.end : m.end + 1;
  }
  return out;
}

u32str Regex::sub(const u32str& s, const Repl& f) const {
  u32str out;
  int pos = 0, last = 0, n = static_cast<int>(s.size());
  Match m;
  while (pos <= n && search(s, pos, m)) {
    out.append(s, last, m.start - last);
    out += f(m, s);
    last = m.end;
    pos = m.end > m.start ? m.end : m.end + 1;
    if (m.end == m.start && m.start < n) {  // (never happens with the front end's patterns)
      out.push_back(s[m.start]);
      last = m.start + 1;
    }
  }
  if (last < n) out.append(s, last, n - last);
  return out;
}

std::vector<u32str> Regex::split(const u32str& s) const {
  std::vector<u32str> out;
  int pos = 0, last = 0, n = static_cast<int>(s.size());
  Match m;
  while (pos <= n && search(s, pos, m)) {
    if (m.end == m.start) {  // (never happens with the front end's patterns)
      pos = m.end + 1;
      continue;
    }
    out.push_back(s.substr(last, m.start - last));
    for (int g = 1; g <= ngroups_; ++g) out.push_back(m.str(s, g));
    last = pos = m.end;
  }
  out.push_back(s.substr(last));
  return out;
}

}  // namespace detail
}  // namespace steertts
