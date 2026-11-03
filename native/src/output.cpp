// Output level (see output.h); mirrors steertts_edge/output.py.
#include "output.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace steertts {
namespace output {

namespace {
constexpr double kPi = 3.14159265358979323846;
float db2lin(double db) { return float(std::pow(10.0, db / 20.0)); }
}  // namespace

Level parse_level(const std::string& name) {
  if (name == "auto" || name.empty()) return kAuto;
  if (name == "clip") return kClip;
  if (name == "off") return kOff;
  throw std::invalid_argument("level must be one of auto, clip, off, got '" + name + "'");
}

void check_gain(float g) {
  if (std::isnan(g)) return;
  if (!(g >= -6.0f && g <= 18.0f)) {
    std::ostringstream ss;
    ss << "gain_db=" << g << " is outside [-6.0, 18.0]";
    throw std::invalid_argument(ss.str());
  }
}

void Stage::load(const json::Value& cfg) {
  if (!cfg.has("output")) return;
  const json::Value& o = cfg["output"];
  if (o.has("target_lufs")) target_lufs = float(o["target_lufs"].num);
  if (o.has("true_peak_db")) true_peak_db = float(o["true_peak_db"].num);
  if (o.has("knee_db")) knee_db = float(o["knee_db"].num);
  if (o.has("gain_db")) gain_db = float(o["gain_db"].num);
  if (o.has("voice_gain_db"))
    for (const auto& kv : o["voice_gain_db"].obj) voice_gain_db[kv.first] = float(kv.second.num);
}

float Stage::gain(const std::string& voice) const {
  auto it = voice_gain_db.find(voice);
  return it == voice_gain_db.end() ? gain_db : it->second;
}

void soft_limit(std::vector<float>& y, float knee_db, float ceiling_db) {
  const float t = db2lin(knee_db), c = db2lin(ceiling_db), w = c - t;
  for (float& v : y) {
    float a = std::fabs(v);
    if (a > t) v = std::copysign(t + w * std::tanh((a - t) / w), v);
  }
}

double true_peak(const std::vector<float>& y) {
  constexpr int kOs = 4, kTaps = 48, kL = kTaps / kOs;
  double h[kOs][kL];
  for (int k = 0; k < kTaps; ++k) {   // the table steertts_edge/output.py builds
    double x = (k - (kTaps - 1) / 2.0) / kOs;
    double s = x == 0 ? 1.0 : std::sin(kPi * x) / (kPi * x);
    h[k % kOs][k / kOs] = s * (0.5 - 0.5 * std::cos(2 * kPi * (k + 0.5) / kTaps));
  }
  double p = 0;
  const size_t n = y.size();
  for (float v : y) p = std::max(p, double(std::fabs(v)));
  for (int ph = 0; ph < kOs; ++ph)
    for (size_t k = 0; k + 1 < n + kL; ++k) {
      double acc = 0;
      for (int j = 0; j < kL; ++j)
        if (k >= size_t(j) && k - j < n) acc += h[ph][j] * y[k - j];
      p = std::max(p, std::fabs(acc));
    }
  return p;
}

double lufs(const std::vector<float>& y, int sr) {
  const size_t n = y.size();
  if (!n) return -70.0;
  std::vector<double> x(y.begin(), y.end());
  auto biquad = [&](double b0, double b1, double b2, double a0, double a1, double a2) {
    b0 /= a0, b1 /= a0, b2 /= a0, a1 /= a0, a2 /= a0;
    double z1 = 0, z2 = 0;
    for (double& v : x) {
      double o = b0 * v + z1;
      z1 = b1 * v - a1 * o + z2;
      z2 = b2 * v - a2 * o;
      v = o;
    }
  };
  {
    double A = std::pow(10, 4.0 / 40), w = 2 * kPi * 1500.0 / sr, al = std::sin(w) / (2 / std::sqrt(2.0)), c = std::cos(w);
    biquad(A * ((A + 1) + (A - 1) * c + 2 * std::sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c),
           A * ((A + 1) + (A - 1) * c - 2 * std::sqrt(A) * al), (A + 1) - (A - 1) * c + 2 * std::sqrt(A) * al,
           2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - 2 * std::sqrt(A) * al);
    w = 2 * kPi * 38.0 / sr, al = std::sin(w) / (2 * 0.5), c = std::cos(w);
    biquad((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al);
  }
  const size_t blk = size_t(0.4 * sr), hop = size_t(0.1 * sr);
  std::vector<double> cs(n + 1, 0.0), z;
  for (size_t i = 0; i < n; ++i) cs[i + 1] = cs[i] + x[i] * x[i];
  if (n < blk) z.push_back(cs[n] / n);
  else
    for (size_t s = 0; s + blk <= n; s += hop) z.push_back((cs[s + blk] - cs[s]) / blk);
  auto L = [](double v) { return -0.691 + 10 * std::log10(v + 1e-12); };
  double s = 0;
  size_t m = 0;
  for (double v : z)
    if (L(v) > -70) s += v, ++m;
  if (!m) return -70.0;
  double rel = -0.691 + 10 * std::log10(s / m) - 10, s2 = 0;
  size_t m2 = 0;
  for (double v : z)
    if (L(v) > -70 && -0.691 + 10 * std::log10(v) > rel) s2 += v, ++m2;
  return -0.691 + 10 * std::log10(s2 / m2);
}

void Stage::chunk(std::vector<float>& y, Level level, const std::string& voice, float g) const {
  if (level == kOff || y.empty()) return;
  const float k = db2lin(std::isnan(g) ? gain(voice) : g);
  for (float& v : y) v *= k;
  soft_limit(y, knee_db, true_peak_db);
}

void Stage::clip(std::vector<float>& y, Level level, const std::string& voice, float g, int sr) const {
  if (level == kOff || y.empty()) return;
  double gd = g;
  if (std::isnan(g)) {
    if (level == kClip) {
      bool any = std::any_of(y.begin(), y.end(), [](float v) { return v != 0.0f; });
      gd = any ? std::min(std::max(target_lufs - lufs(y, sr), -6.0), 30.0) : 0.0;
    } else {
      gd = gain(voice);
    }
  }
  const float k = db2lin(gd);
  for (float& v : y) v *= k;
  soft_limit(y, knee_db, true_peak_db);
  double tp = true_peak(y), cap = db2lin(true_peak_db);
  if (tp > cap) {
    const float s = float(cap / tp);
    for (float& v : y) v *= s;
  }
}

}  // namespace output
}  // namespace steertts
