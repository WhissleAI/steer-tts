/* WAV reading and band-limited resampling (no dependencies beyond libc / libm). */
#ifndef ST_AUDIO_H
#define ST_AUDIO_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* RIFF WAVE (RF64 / big-endian RIFX are not supported): PCM 8 / 16 / 24 / 32-bit, IEEE float 32 / 64, WAVE_FORMAT_EXTENSIBLE with a
 * PCM or float sub-format. Channels are averaged to mono. Values equal soundfile.read(dtype="float32") followed by
 * numpy's float32 .mean(1): int / 2^(bits-1) (8-bit: (u - 128) / 128), floats as stored (float64 rounded to float32).
 * *y is st_malloc'ed (st_free it); *n frames; *sr the file's sample rate. 0 = ok, -1 = error (st_geterr). */
int st_wav_read(const char* path, float** y, size_t* n, int* sr);
int st_wav_read_mem(const void* data, size_t len, float** y, size_t* n, int* sr);

/* Band-limited resampling for any pair of positive integer rates, in the soxr "HQ" quality class (linear phase,
 * passband flat to 0.913 x the lower Nyquist, -6 dB at ~0.957, >= 125 dB stopband from the lower Nyquist on; Kaiser
 * windowed sinc, polyphase). The output is time aligned (no delay) and its length equals soxr.resample's:
 * ny = (size_t)((double)n / ((double)sr_in / sr_out) + 0.5). The signal is zero outside [0, n). sr_in == sr_out
 * copies. *y is st_malloc'ed (at least 1 element). 0 = ok, -1 = error. */
int st_resample(const float* x, size_t n, int sr_in, int sr_out, float** y, size_t* ny);
size_t st_resample_len(size_t n, int sr_in, int sr_out);

#ifdef __cplusplus
}
#endif
#endif
