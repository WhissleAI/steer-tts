/* Vocos mel front end (vocos.feature_extractors.MelSpectrogramFeatures(24000, n_fft 1024, hop 256, 100 mels,
 * padding "center")) and SteerTTS.prompt_mel (the reference-clip context of the prompted flow). */
#ifndef ST_MEL_H
#define ST_MEL_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* wav [n] at 24 kHz -> *mel [T,100] (time-major, st_malloc'd; st_free it), T = 1 + n / 256.
 * torchaudio MelSpectrogram (power 1, center reflect pad, periodic Hann, HTK mel scale, no norm) then
 * log(clamp(x, 1e-7)) (vocos safe_log). n must exceed 512 (torch's reflect pad). */
int st_mel_spectrogram(const float* wav, size_t n, float** mel, int* T);

/* SteerTTS.prompt_mel: the log-mel of the reference clip's first 8 s (wav24: the clip at 24 kHz, already resampled
 * if needed; orig_sr: its ORIGINAL sample rate - a clip recorded below 24 kHz has its bands above 0.95 * orig_sr / 2
 * filled from render_mel's spectral shape), level-matched to render_mel [T,100] (the acoustic model's mel of this
 * request). Returns 0 with *pm = NULL, *P = 0 when the clip is shorter than 0.5 s (the pipeline then runs the
 * prompted flow without a prompt); otherwise *pm [P,100] (st_free it). Note: the Python pipeline only knows the
 * original rate for a file path; a raw array is taken as 24 kHz (pass orig_sr = 24000 to match that). */
int st_prompt_mel(const float* wav24, size_t n, int orig_sr, const float* render_mel, int T, float** pm, int* P);

#ifdef __cplusplus
}
#endif
#endif
