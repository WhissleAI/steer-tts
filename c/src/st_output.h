/* Output level: the last stage between the vocoder and the caller. Same stage as the Python packages
 * (steertts/output.py, steertts_edge/output.py), the native engine (native/src/output.cpp) and the ESP32 firmware.
 *
 * config.json "output": {"target_lufs": -17, "true_peak_db": -1, "knee_db": -3, "gain_db": <fallback>,
 *                        "voice_gain_db": {voice: dB}}
 * Levels: AUTO (default): the voice's calibrated gain (neutral speech at target_lufs), a soft-knee limiter (tanh above
 * knee_db, ceiling true_peak_db) and, for a whole clip, a 4x-oversampled true-peak guard; streamed chunks get the same
 * fixed gain + limiter (memoryless: no look-ahead, no added latency). CLIP: a whole clip normalised to target_lufs
 * (BS.1770-4), then limiter + guard; streaming as AUTO. OFF: the raw model output. */
#ifndef ST_OUTPUT_H
#define ST_OUTPUT_H
#include <stddef.h>

#include "st_base.h"

enum { ST_LEVEL_OFF = 0, ST_LEVEL_AUTO = 1, ST_LEVEL_CLIP = 2 };
#define ST_GAIN_MIN (-6.0)
#define ST_GAIN_MAX 18.0

typedef struct {
  float target_lufs, true_peak_db, knee_db, gain_db;
  float* voice_gain;   /* [n_voices] dB; NaN = the fallback gain_db */
  int n_voices;
} st_output;

/* Reads cfg "output" (absent: target -17, cap -1, knee -3, gain 0 dB, i.e. the limiter only). 0 on success. */
int st_output_load(st_output* o, const st_json* cfg, char* const* voices, int n_voices);
void st_output_free(st_output* o);
/* "auto" / "clip" / "off" -> ST_LEVEL_*, -1 if unknown. */
int st_output_level(const char* name);
const char* st_output_level_name(int level);
/* dB gain of a voice (vi < 0 or uncalibrated: the fallback). */
float st_output_gain(const st_output* o, int vi);

void st_soft_limit(float* y, size_t n, float knee_db, float ceiling_db);
double st_true_peak(const float* y, size_t n);   /* linear, max(|y|, |4x oversampled y|) */
double st_lufs(const float* y, size_t n, int sr);   /* BS.1770-4 integrated loudness; -70 for silence */

/* gain_db: NaN = the level's own gain (AUTO: the voice's, CLIP: to the target). In place. */
void st_output_clip(const st_output* o, float* y, size_t n, int sr, int level, int vi, double gain_db);
void st_output_chunk(const st_output* o, float* y, size_t n, int level, int vi, double gain_db);

#endif
