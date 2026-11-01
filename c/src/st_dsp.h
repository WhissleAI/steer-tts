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
/* The same iSTFT frame chunk by frame chunk (streaming): begin with the total frame count, add the frames in order
 * (st_istft_add: nf more frames, re / im pointing at the first of them); each call writes the samples that are now
 * final into out (st_istft_len(T, hop) samples in all) and sets *ready to how many of them are written. Any split
 * gives exactly st_istft's samples. */
typedef struct st_istft_state st_istft_state;
st_istft_state* st_istft_begin(int T, int n_fft, int hop);
int st_istft_add(st_istft_state* s, st_pool* p, const float* re, const float* im, int ld, int nf, float* out,
                 size_t* ready);
void st_istft_end(st_istft_state* s);
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
