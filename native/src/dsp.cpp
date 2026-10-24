#include "dsp.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace steertts {
namespace dsp {

namespace {

const double kPi = 3.14159265358979323846;

// In-place iterative radix-2 inverse complex FFT (unnormalised), double precision.
struct InverseFFT {
  int n;
  std::vector<std::complex<double>> tw;
  std::vector<int> rev;
  explicit InverseFFT(int n_) : n(n_), tw(size_t(n_ / 2)), rev(size_t(n_)) {
    for (int k = 0; k < n / 2; ++k) tw[size_t(k)] = std::polar(1.0, 2 * kPi * k / n);
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    for (int i = 0; i < n; ++i) {
      int r = 0;
      for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
      rev[size_t(i)] = r;
    }
  }
  void run(std::vector<std::complex<double>>& a) const {
    for (int i = 0; i < n; ++i)
      if (i < rev[size_t(i)]) std::swap(a[size_t(i)], a[size_t(rev[size_t(i)])]);
    for (int len = 2; len <= n; len <<= 1) {
      int step = n / len;
      for (int i = 0; i < n; i += len)
        for (int j = 0; j < len / 2; ++j) {
          std::complex<double> u = a[size_t(i + j)], v = a[size_t(i + j + len / 2)] * tw[size_t(j * step)];
          a[size_t(i + j)] = u + v;
          a[size_t(i + j + len / 2)] = u - v;
        }
    }
  }
};

std::vector<float> hann(int n) {   // periodic, computed in double and rounded to float32 (np: astype(float32))
  std::vector<float> w(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) w[size_t(i)] = float(0.5 - 0.5 * std::cos(2 * kPi * i / n));
  return w;
}

}  // namespace

std::vector<float> istft(const float* re, const float* im, int T) {
  static const InverseFFT fft(kNfft);
  static const std::vector<float> win = hann(kNfft);
  const int nb = kNfft / 2 + 1;
  const size_t size = size_t(kNfft + kHop * (T - 1));
  std::vector<float> y(size, 0.0f), env(size, 0.0f), frame(kNfft);
  std::vector<std::complex<double>> a(kNfft);
  for (int t = 0; t < T; ++t) {
    // irfft: Hermitian-extend bins 0..N/2 (imaginary parts of the DC and Nyquist bins are ignored, as numpy)
    a[0] = {double(re[t]), 0.0};
    a[kNfft / 2] = {double(re[size_t(nb - 1) * T + t]), 0.0};
    for (int k = 1; k < kNfft / 2; ++k) {
      std::complex<double> v(re[size_t(k) * T + t], im[size_t(k) * T + t]);
      a[size_t(k)] = v;
      a[size_t(kNfft - k)] = std::conj(v);
    }
    fft.run(a);
    for (int i = 0; i < kNfft; ++i) frame[size_t(i)] = float(a[size_t(i)].real() / kNfft) * win[size_t(i)];
    float* yo = &y[size_t(t) * kHop];
    float* eo = &env[size_t(t) * kHop];
    for (int i = 0; i < kNfft; ++i) {
      yo[i] += frame[size_t(i)];
      eo[i] += win[size_t(i)] * win[size_t(i)];
    }
  }
  const size_t a0 = kNfft / 2, b0 = size - kNfft / 2;
  std::vector<float> out(b0 - a0);
  for (size_t i = a0; i < b0; ++i) out[i - a0] = y[i] / env[i];
  return out;
}

void fade_ends(std::vector<float>& y, int fade) {
  if (int(y.size()) <= 2 * fade) return;
  const double step = 1.0 / (fade - 1);   // np.linspace(0, 1, fade, dtype=float32)
  const size_t n = y.size();
  for (int i = 0; i < fade; ++i) {
    float r = (i == fade - 1) ? 1.0f : float(i * step);
    y[size_t(i)] *= r;
    y[n - 1 - size_t(i)] *= r;
  }
}

void save_wav(const std::vector<float>& y, int sr, const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  auto u32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
  auto u16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };
  const uint32_t bytes = uint32_t(y.size() * 2);
  std::fwrite("RIFF", 1, 4, f); u32(36 + bytes); std::fwrite("WAVE", 1, 4, f);
  std::fwrite("fmt ", 1, 4, f); u32(16); u16(1); u16(1); u32(uint32_t(sr)); u32(uint32_t(sr * 2)); u16(2); u16(16);
  std::fwrite("data", 1, 4, f); u32(bytes);
  std::vector<int16_t> pcm(y.size());
  for (size_t i = 0; i < y.size(); ++i) {
    float v = std::min(1.0f, std::max(-1.0f, y[i])) * 32767.0f;
    pcm[i] = int16_t(std::nearbyint(v));   // half to even, as numpy round
  }
  std::fwrite(pcm.data(), 2, pcm.size(), f);
  std::fclose(f);
}

}  // namespace dsp
}  // namespace steertts
