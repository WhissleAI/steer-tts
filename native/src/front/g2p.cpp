#include "g2p.h"

#include <cmath>
#include <cstring>
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace steertts {
namespace detail {

namespace {
std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("steertts front: cannot open " + path);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
struct Rd {
  const std::string& b;
  size_t o = 0;
  void need(size_t n) const {
    if (o + n > b.size()) throw std::runtime_error("steertts front: truncated asset");
  }
  uint32_t u32() {
    need(4);
    uint32_t v;
    std::memcpy(&v, b.data() + o, 4);
    o += 4;
    return v;
  }
  template <class T>
  void arr(std::vector<T>& v, size_t n) {
    need(n * sizeof(T));
    v.resize(n);
    std::memcpy(v.data(), b.data() + o, n * sizeof(T));
    o += n * sizeof(T);
  }
};
u32str A(const std::string& s) { return u32str(s.begin(), s.end()); }
std::string N(const u32str& s) {
  std::string o;
  for (char32_t c : s) o.push_back(static_cast<char>(c));
  return o;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ Lexicon
void Lexicon::load(const std::string& path) {
  std::string b = read_file(path);
  Rd r{b};
  if (b.compare(0, 4, "STLX") != 0) throw std::runtime_error("steertts front: bad lexicon " + path);
  r.o = 4;
  if (r.u32() != 1) throw std::runtime_error("steertts front: bad lexicon version");
  n_ = r.u32();
  uint32_t wb = r.u32(), pb = r.u32();
  r.arr(wo_, n_ + 1);
  r.arr(po_, n_ + 1);
  r.need(wb + pb);
  words_.assign(b.data() + r.o, wb);
  r.o += wb;
  phones_.assign(reinterpret_cast<const uint8_t*>(b.data() + r.o), reinterpret_cast<const uint8_t*>(b.data() + r.o + pb));
}

bool Lexicon::get(const std::string& w, std::vector<int>* phones) const {
  size_t lo = 0, hi = n_;
  while (lo < hi) {  // keys are sorted bytewise (== Python str order for ascii)
    size_t mid = (lo + hi) / 2;
    int c = words_.compare(wo_[mid], wo_[mid + 1] - wo_[mid], w);
    if (c == 0) {
      if (phones) phones->assign(phones_.begin() + po_[mid], phones_.begin() + po_[mid + 1]);
      return true;
    }
    if (c < 0)
      lo = mid + 1;
    else
      hi = mid;
  }
  return false;
}

// ------------------------------------------------------------------------------------------------ GRU
void GruG2p::load(const std::string& path) {
  std::string b = read_file(path);
  if (b.compare(0, 4, "STGR") != 0) throw std::runtime_error("steertts front: bad gru.bin");
  Rd r{b};
  r.o = 4;
  if (r.u32() != 1) throw std::runtime_error("steertts front: bad gru.bin version");
  H_ = static_cast<int>(r.u32());
  NG_ = static_cast<int>(r.u32());
  NP_ = static_cast<int>(r.u32());
  size_t H3 = 3 * static_cast<size_t>(H_);
  r.arr(enc_x_, NG_ * H3);
  r.arr(dec_x_, NP_ * H3);
  r.arr(enc_q_, H3 * H_);
  r.arr(enc_s_, H3);
  r.arr(enc_b_, H3);
  r.arr(dec_q_, H3 * H_);
  r.arr(dec_s_, H3);
  r.arr(dec_b_, H3);
  r.arr(fc_q_, static_cast<size_t>(NP_) * H_);
  r.arr(fc_s_, NP_);
  r.arr(fc_b_, NP_);
  std::vector<int32_t> p2s;
  r.arr(p2s, NP_);
  phon2sym_.assign(p2s.begin(), p2s.end());
}

namespace {
// sum_i q[i] * h[i] (the row scale is applied by the caller). numpy computes h @ W in float32 (OpenBLAS sgemv, its
// own summation order), so any implementation differs from it by float32 rounding noise (~1e-6 on the logits);
// only a top-2 logit gap below that could flip an argmax. The golden G2P set (33k words) has a smallest numpy gap
// of 3.4e-5 and matches 100%.
// Canonical arithmetic (identical on every path, so results do not depend on the CPU): 32 float32 lanes, lane k
// accumulates float(q[i]) * h[i] for i = k (mod 32) in increasing i; then lanes are combined as 4-lane vectors
// t = ((A0+A1)+(A2+A3))+((A4+A5)+(A6+A7)) (A_m = lanes 4m..4m+3) in float, and (t0+t1)+(t2+t3) in double.
// Weights stay int8 in memory (0.4 MB, L2 resident). Build with -ffp-contract=off (CMakeLists does) so no FMA
// contraction changes the rounding.
inline double reduce_lanes(const float* A) {  // A[32]
  float t[4];
  for (int l = 0; l < 4; ++l)
    t[l] = ((A[l] + A[4 + l]) + (A[8 + l] + A[12 + l])) + ((A[16 + l] + A[20 + l]) + (A[24 + l] + A[28 + l]));
  return (static_cast<double>(t[0]) + t[1]) + (static_cast<double>(t[2]) + t[3]);
}
#if defined(__SSE2__) || defined(_M_X64)
inline void widen16(const int8_t* q, __m128& f0, __m128& f1, __m128& f2, __m128& f3) {  // 16 x int8 -> 4 x float4
  __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
  __m128i lo = _mm_srai_epi16(_mm_unpacklo_epi8(b, b), 8);  // sign-extend int8 -> int16 (SSE2 only)
  __m128i hi = _mm_srai_epi16(_mm_unpackhi_epi8(b, b), 8);
  f0 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(lo, lo), 16));
  f1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(lo, lo), 16));
  f2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(hi, hi), 16));
  f3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(hi, hi), 16));
}
inline double dot_q(const int8_t* q, const float* h, int n) {  // n % 32 == 0
  __m128 a0 = _mm_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
  for (int i = 0; i < n; i += 32) {
    __m128 w0, w1, w2, w3, w4, w5, w6, w7;
    widen16(q + i, w0, w1, w2, w3);
    widen16(q + i + 16, w4, w5, w6, w7);
    const float* x = h + i;
    a0 = _mm_add_ps(a0, _mm_mul_ps(w0, _mm_loadu_ps(x)));
    a1 = _mm_add_ps(a1, _mm_mul_ps(w1, _mm_loadu_ps(x + 4)));
    a2 = _mm_add_ps(a2, _mm_mul_ps(w2, _mm_loadu_ps(x + 8)));
    a3 = _mm_add_ps(a3, _mm_mul_ps(w3, _mm_loadu_ps(x + 12)));
    a4 = _mm_add_ps(a4, _mm_mul_ps(w4, _mm_loadu_ps(x + 16)));
    a5 = _mm_add_ps(a5, _mm_mul_ps(w5, _mm_loadu_ps(x + 20)));
    a6 = _mm_add_ps(a6, _mm_mul_ps(w6, _mm_loadu_ps(x + 24)));
    a7 = _mm_add_ps(a7, _mm_mul_ps(w7, _mm_loadu_ps(x + 28)));
  }
  alignas(16) float A[32];
  _mm_store_ps(A, a0);
  _mm_store_ps(A + 4, a1);
  _mm_store_ps(A + 8, a2);
  _mm_store_ps(A + 12, a3);
  _mm_store_ps(A + 16, a4);
  _mm_store_ps(A + 20, a5);
  _mm_store_ps(A + 24, a6);
  _mm_store_ps(A + 28, a7);
  return reduce_lanes(A);
}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
inline double dot_q(const int8_t* q, const float* h, int n) {  // n % 32 == 0
  float32x4_t acc[8];
  for (auto& a : acc) a = vdupq_n_f32(0.0f);
  for (int i = 0; i < n; i += 32) {
    for (int half = 0; half < 2; ++half) {
      int8x16_t b = vld1q_s8(q + i + 16 * half);
      int16x8_t lo = vmovl_s8(vget_low_s8(b)), hi = vmovl_s8(vget_high_s8(b));
      int32x4_t w[4] = {vmovl_s16(vget_low_s16(lo)), vmovl_s16(vget_high_s16(lo)), vmovl_s16(vget_low_s16(hi)),
                        vmovl_s16(vget_high_s16(hi))};
      for (int m = 0; m < 4; ++m) {
        float32x4_t x = vld1q_f32(h + i + 16 * half + 4 * m);
        acc[4 * half + m] = vaddq_f32(acc[4 * half + m], vmulq_f32(vcvtq_f32_s32(w[m]), x));  // no fused mla
      }
    }
  }
  float A[32];
  for (int m = 0; m < 8; ++m) vst1q_f32(A + 4 * m, acc[m]);
  return reduce_lanes(A);
}
#else
inline double dot_q(const int8_t* q, const float* h, int n) {
  float A[32] = {0};
  for (int i = 0; i < n; i += 32)
    for (int k = 0; k < 32; ++k) A[k] += static_cast<float>(q[i + k]) * h[i + k];
  return reduce_lanes(A);
}
#endif
inline double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }
}  // namespace

void GruG2p::cell(const float* xp, double* h, const int8_t* q, const float* s, const float* b) const {
  // gru_np.GruG2P._cell: hp = h @ W_hh.T + b_hh; r, z = sigmoid; n = tanh(xn + r * hn); h = (1 - z) n + z h
  const int H = H_;
  double hp[3 * 512];
  alignas(32) float hf[512];
  for (int j = 0; j < H; ++j) hf[j] = static_cast<float>(h[j]);
  for (int j = 0; j < 3 * H; ++j) hp[j] = static_cast<double>(s[j]) * dot_q(q + static_cast<size_t>(j) * H, hf, H) + b[j];
  for (int j = 0; j < H; ++j) {
    double r = sig(static_cast<double>(xp[j]) + hp[j]);
    double z = sig(static_cast<double>(xp[H + j]) + hp[H + j]);
    double nn = std::tanh(static_cast<double>(xp[2 * H + j]) + r * hp[2 * H + j]);
    h[j] = (1 - z) * nn + z * h[j];
  }
}

std::vector<int> GruG2p::predict_raw(const std::string& word, double* margin) const {
  const int H = H_;
  if (H > 512 || H % 32) throw std::runtime_error("steertts front: unsupported GRU size");
  std::vector<double> h(H, 0.0);
  std::vector<int> ids;
  for (char c : word) ids.push_back(c >= 'a' && c <= 'z' ? c - 'a' + 3 : 1);  // _G2I.get(c, 1)
  ids.push_back(2);
  for (int i : ids) cell(&enc_x_[static_cast<size_t>(i) * 3 * H], h.data(), enc_q_.data(), enc_s_.data(), enc_b_.data());
  int prev = 2;
  std::vector<int> preds;
  double gap = 1e9;
  std::vector<double> lo(NP_);
  std::vector<float> hf(H);
  for (int step = 0; step < 20; ++step) {
    cell(&dec_x_[static_cast<size_t>(prev) * 3 * H], h.data(), dec_q_.data(), dec_s_.data(), dec_b_.data());
    int p = 0;
    for (int j = 0; j < H; ++j) hf[j] = static_cast<float>(h[j]);
    for (int k = 0; k < NP_; ++k) {
      lo[k] = static_cast<double>(fc_s_[k]) * dot_q(&fc_q_[static_cast<size_t>(k) * H], hf.data(), H) + fc_b_[k];
      if (lo[k] > lo[p]) p = k;  // numpy argmax: first maximum
    }
    if (margin) {
      double second = -1e300;
      for (int k = 0; k < NP_; ++k)
        if (k != p && lo[k] > second) second = lo[k];
      gap = std::min(gap, lo[p] - second);
    }
    if (p == 3) break;
    preds.push_back(p);
    prev = p;
  }
  if (margin) *margin = gap;
  return preds;
}

std::vector<int> GruG2p::predict(const std::string& word, double* margin) const {
  std::vector<int> out;
  for (int p : predict_raw(word, margin))
    if (phon2sym_[p] >= 0) out.push_back(phon2sym_[p]);
  return out;
}

// ------------------------------------------------------------------------------------------------ Hetero
void Hetero::load(const std::string& path, const std::unordered_map<std::string, int>& sym2id) {
  std::istringstream in(read_file(path));
  std::string line;
  auto split = [](const std::string& s, char d) {
    std::vector<std::string> v;
    size_t st = 0;
    while (true) {
      size_t e = s.find(d, st);
      v.push_back(s.substr(st, e == std::string::npos ? std::string::npos : e - st));
      if (e == std::string::npos) break;
      st = e + 1;
    }
    return v;
  };
  auto phones = [&](const std::string& s) {
    std::vector<int> v;
    for (auto& p : split(s, ' '))
      if (!p.empty()) v.push_back(sym2id.at(p));
    return v;
  };
  has_td_ = true;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto f = split(line, '\t');
    if (f[0] == "H" && f.size() == 5) {
      hom_[f[1]] = Hom{phones(f[2]), phones(f[3]), f[4]};
    } else if (f[0] == "W" && f.size() == 3) {
      w_[f[1]] = std::stoi(f[2]);
    } else if (f[0] == "T" && f.size() == 3) {
      td_[f[1]] = f[2];
    } else if (f[0] == "N") {
      has_td_ = false;
    } else {
      throw std::runtime_error("steertts front: bad hetero.tsv line");
    }
  }
}

namespace {
bool isalpha_ascii(const std::string& t) {
  if (t.empty()) return false;
  for (char c : t)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) return false;
  return true;
}
std::string suf(const std::string& t) { return isalpha_ascii(t) && t.size() > 2 ? t.substr(t.size() - 2) : t; }
std::string last2(const std::string& t) { return t.size() > 2 ? t.substr(t.size() - 2) : t; }
}  // namespace

std::string Hetero::tag(const std::string& t) const {
  auto it = td_.find(t);
  if (it != td_.end()) return it->second;
  if (t == "<s>" || t == "</s>") return t;
  bool any = false;
  for (char c : t)
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) any = true;
  if (!any) return "P";
  return "U" + last2(t);
}

bool Hetero::first(const std::vector<std::string>& toks, size_t j) const {
  const std::string& w = toks[j];
  const std::string& pos1 = hom_.at(w).pos;
  auto at = [&](long k) -> std::string {
    if (k < 0) return "<s>";
    if (k >= static_cast<long>(toks.size())) return "</s>";
    return toks[k];
  };
  long J = static_cast<long>(j);
  std::string p1 = at(J - 1), p2 = at(J - 2), n1 = at(J + 1), n2 = at(J + 2);
  std::vector<std::string> f = {"b", "p1=" + p1, "n1=" + n1, "p2=" + p2, "n2=" + n2, "p1s=" + suf(p1),
                                "n1s=" + suf(n1), "p1n1=" + p1 + "_" + n1, "p2p1=" + p2 + "_" + p1,
                                "n1n2=" + n1 + "_" + n2, "p2s=" + suf(p2), "n2s=" + suf(n2)};
  if (has_td_) {
    std::string a = tag(p2), b = tag(p1), c = tag(n1), d = tag(n2);
    for (auto x : {"p1t=" + b, "n1t=" + c, "p2t=" + a, "n2t=" + d, "p1tn1t=" + b + "_" + c, "p2tp1t=" + a + "_" + b,
                   "n1tn2t=" + c + "_" + d, "p1n1t=" + p1 + "_" + c, "p1tn1=" + b + "_" + n1})
      f.push_back(x);
  }
  long s = 0;
  for (auto& x : f) {
    auto i1 = w_.find("w:" + w + "|" + x);
    if (i1 != w_.end()) s += i1->second;
    auto i2 = w_.find("g:" + pos1 + "|" + x);
    if (i2 != w_.end()) s += i2->second;
  }
  return s > 0;
}

// ------------------------------------------------------------------------------------------------ EdgeG2p
EdgeG2p::EdgeG2p(const std::string& dir, const Ucd* u, const std::unordered_map<std::string, int>& sym2id)
    : u_(u),
      nw_(u),
      hang_(u, R"(([^a-z0-9])\1{3,})"),
      word_(u,
            R"((?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+){0,126}[.](?:[a-z]{2,13})\b/?(?!@))"
            R"(|[a-z](?:[a-z]|['\-_])+[a-z])"
            R"(|[a-z0-9_]+)"
            R"(|\.(?:\s*\.)+)"
            R"(|\S)") {
  cmu_.load(dir + "/lexicon.bin");
  oov_.load(dir + "/oov.bin");
  gru_.load(dir + "/gru.bin");
  het_.load(dir + "/hetero.tsv", sym2id);
  for (int& p : punct_id_) p = -2;
  for (char c : std::string(",.?!;:-")) punct_id_[static_cast<int>(c)] = sym2id.at(std::string(1, c));
}

bool EdgeG2p::in_cmu(const std::string& w) const {
  for (char c : w)
    if (static_cast<unsigned char>(c) >= 128) return false;
  return cmu_.get(w, nullptr);
}

std::vector<std::string> EdgeG2p::tokenize(const std::string& a) const {
  u32str t = hang_.sub(A(a), [](const Regex::Match& m, const u32str& s) {
    u32str g = m.str(s, 1);
    return g + g + g;
  });
  std::vector<std::string> out;
  for (auto& m : word_.finditer(t)) out.push_back(N(m.str(t, 0)));
  return out;
}

std::vector<int> EdgeG2p::predict_cached(const std::string& w) const {
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = cache_.find(w);
    if (it != cache_.end()) return it->second;
  }
  std::vector<int> p = gru_.predict(w);
  std::lock_guard<std::mutex> lk(mu_);
  if (cache_.size() >= 16384) cache_.clear();
  cache_[w] = p;
  return p;
}

std::vector<int> EdgeG2p::pron(const std::string& w, const std::vector<std::string>& toks, size_t j) const {
  bool has_letter = false;
  for (char c : w)
    if (c >= 'a' && c <= 'z') has_letter = true;
  if (!has_letter) {  // non-alpha token -> itself; only single punctuation marks are symbols (tokens are ascii)
    if (w.size() == 1 && static_cast<unsigned char>(w[0]) < 128 && punct_id_[static_cast<int>(w[0])] >= 0)
      return {punct_id_[static_cast<int>(w[0])]};
    return {};
  }
  if (const Hetero::Hom* h = het_.find(w)) return het_.first(toks, j) ? h->p1 : h->p2;
  std::vector<int> ph;
  if (cmu_.get(w, &ph)) return ph.empty() ? predict_cached(w) : ph;  // member-only entry -> model
  if (oov_.get(w, &ph)) return ph;                                     // cached prediction (may be empty)
  return predict_cached(w);
}

std::vector<int> EdgeG2p::call(const u32str& text0) const {
  u32str t = nw_.normalize_numbers(text0);
  std::string a;
  for (char32_t c : t) a += u_->g2p_clean(c);  // NFD, drop Mn, lower, keep [ a-z'.,?!-]
  auto replace_all = [](std::string& s, const std::string& from, const std::string& to) {
    std::string o;
    size_t st = 0;
    while (true) {
      size_t e = s.find(from, st);
      if (e == std::string::npos) break;
      o.append(s, st, e - st);
      o += to;
      st = e + from.size();
    }
    o.append(s, st, std::string::npos);
    s.swap(o);
  };
  replace_all(a, "i.e.", "that is");
  replace_all(a, "e.g.", "for example");
  std::vector<std::string> words = tokenize(a);
  std::vector<int> out;
  for (size_t j = 0; j < words.size(); ++j) {
    for (int p : pron(words[j], words, j)) out.push_back(p);
    out.push_back(TOK_SPACE);
  }
  if (!out.empty()) out.pop_back();
  return out;
}

}  // namespace detail
}  // namespace steertts
