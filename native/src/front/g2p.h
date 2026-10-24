// Port of steertts_edge/edge_front: backend.EdgeG2p (g2p_en.G2p.__call__ replacement), lexicon.Lexicon (residual
// CMUdict + OOV cache), gru_np.GruG2P (tier B int8 GRU), hetero.Hetero, tok.word_tokenize, normnum.
// Token stream returned by EdgeG2p::call: SYMBOLS ids (phones and the punctuation , . ? ! -) and SPACE (-1) between
// words - i.e. g2p_en's output list with every token to_phonemes would skip (not in SYM2ID, <pad>, <sp>) removed.
#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "numwords.h"
#include "pyre.h"
#include "ucd.h"

namespace steertts {
namespace detail {

constexpr int TOK_SPACE = -1;

class Lexicon {
 public:
  void load(const std::string& path);
  // true if `w` is a key; phones (SYMBOLS ids) may be empty (member-only entry / cached empty prediction)
  bool get(const std::string& w, std::vector<int>* phones) const;
  size_t size() const { return n_; }

 private:
  uint32_t n_ = 0;
  std::vector<uint32_t> wo_, po_;
  std::string words_;
  std::vector<uint8_t> phones_;
};

class GruG2p {
 public:
  void load(const std::string& path);
  // g2p_en greedy decode; returns SYMBOLS ids (outputs to_phonemes would drop are removed). If `margin` is given it
  // receives the smallest top1-top2 logit gap over the decode (diagnostics).
  std::vector<int> predict(const std::string& word, double* margin = nullptr) const;
  std::vector<int> predict_raw(const std::string& word, double* margin = nullptr) const;  // GRU phoneme indices
  int phon2sym(int p) const { return phon2sym_[p]; }

 private:
  void cell(const float* xp, double* h, const int8_t* q, const float* s, const float* b) const;
  int H_ = 0, NG_ = 0, NP_ = 0;
  std::vector<float> enc_x_, dec_x_, enc_s_, enc_b_, dec_s_, dec_b_, fc_s_, fc_b_;
  std::vector<int8_t> enc_q_, dec_q_, fc_q_;  // int8 rows [3H][H] / [NP][H], W[j][i] = q[j][i] * scale[j]
  std::vector<int> phon2sym_;
};

class Hetero {
 public:
  struct Hom {
    std::vector<int> p1, p2;
    std::string pos;
  };
  void load(const std::string& path, const std::unordered_map<std::string, int>& sym2id);
  const Hom* find(const std::string& w) const {
    auto it = hom_.find(w);
    return it == hom_.end() ? nullptr : &it->second;
  }
  bool first(const std::vector<std::string>& toks, size_t j) const;

 private:
  std::string tag(const std::string& t) const;
  std::unordered_map<std::string, Hom> hom_;
  std::unordered_map<std::string, int> w_;
  std::unordered_map<std::string, std::string> td_;
  bool has_td_ = false;
};

class EdgeG2p {
 public:
  EdgeG2p(const std::string& dir, const Ucd* u, const std::unordered_map<std::string, int>& sym2id);
  std::vector<int> call(const u32str& text) const;      // g2p_en.G2p.__call__ (token stream, see top)
  bool in_cmu(const std::string& w) const;               // `w in cmu`
  std::vector<std::string> tokenize(const std::string& ascii) const;  // tok.word_tokenize
  const GruG2p& gru() const { return gru_; }
  const Numwords& numwords() const { return nw_; }

 private:
  std::vector<int> pron(const std::string& w, const std::vector<std::string>& toks, size_t j) const;
  std::vector<int> predict_cached(const std::string& w) const;
  const Ucd* u_;
  Numwords nw_;
  Lexicon cmu_, oov_;
  GruG2p gru_;
  Hetero het_;
  Regex hang_, word_;
  int punct_id_[128];
  mutable std::mutex mu_;
  mutable std::unordered_map<std::string, std::vector<int>> cache_;  // like Python's lru_cache on predict/member
};

}  // namespace detail
}  // namespace steertts
