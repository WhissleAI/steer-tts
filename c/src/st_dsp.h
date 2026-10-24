/* Signal processing: iSTFT (== torch.istft(center=True), periodic Hann window, window-envelope normalisation),
 * segment fades, 16-bit PCM WAV output. */
#ifndef ST_DSP_H
#define ST_DSP_H
#include <stddef.h>

#include "st_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Inverse STFT of T frames, center=True: re / im [T, ld] time-major half spectra (n_fft/2+1 bins each; imaginary
 * parts of DC and Nyquist ignored as numpy / torch irfft), window = periodic Hann(n_fft), win_length = n_fft.
 * out: (T - 1) * hop samples (0 for T <= 1). p may be NULL. Returns 0, or -1 on allocation failure. */
int st_istft(st_pool* p, const float* re, const float* im, int ld, int T, int n_fft, int hop, float* out);
size_t st_istft_len(int T, int hop);
/* Linear fade in / out over `fade` samples when n > 2 * fade (markup segments: np.linspace(0, 1, fade) ramps). */
void st_fade_ends(float* y, size_t n, int fade);
/* 16-bit mono PCM WAV (clipped to [-1, 1], * 32767, round half to even). */
int st_wav_write(const char* path, const float* y, size_t n, int sample_rate);
/* the same file image in memory: 44-byte header + samples; st_free() it */
int st_wav_encode(const float* y, size_t n, int sample_rate, unsigned char** buf, size_t* len);

#ifdef __cplusplus
}
#endif
#endif
