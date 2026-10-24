// Signal processing for the engine: iSTFT (== torch.istft(center=True) with a periodic Hann window, as the Python
// runtimes), segment fades, WAV output. No dependencies.
#pragma once
#include <string>
#include <vector>

namespace steertts {
namespace dsp {

constexpr int kNfft = 1024, kHop = 256;

// re, im: [kNfft/2+1, T] row-major spectra from the vocoder graph -> T*kHop - ... samples (as torch.istft, center=True)
std::vector<float> istft(const float* re, const float* im, int T);
// 10 ms linear fade in/out when the clip is longer than 2*fade (markup.MarkupSynth.stream)
void fade_ends(std::vector<float>& y, int fade);
void save_wav(const std::vector<float>& y, int sample_rate, const std::string& path);

}  // namespace dsp
}  // namespace steertts
