/* Zero-shot cloning speaker encoder: WavLM-base-plus-sv x-vectors (transformers WavLMForXVector, eval mode), the
 * same 512-d L2-normalised vectors as steertts.speaker.XVec. Weights: c/tools/export_wavlm.py (wavlm_sv.safetensors,
 * fp32 or the packed int8 file). */
#ifndef ST_WAVLM_H
#define ST_WAVLM_H
#include <stddef.h>

#include "st_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_wavlm st_wavlm;

/* Reads (and for the convolutions re-lays out) the weights; w must outlive the model. NULL + st_seterr on error. */
st_wavlm* st_wavlm_load(st_weights* w);
void st_wavlm_free(st_wavlm* m);

/* == XVec.__call__([wav24])[0]: wav at 24 kHz (any length > 0), first 8 s, resampled to 16 kHz (st_resample), tiled
 * when shorter than 8000 samples -> L2-normalised 512-d vector. 0 = ok, -1 = error. Thread-safe for distinct calls on
 * one model (the model is read-only after load). */
int st_wavlm_xvector(st_wavlm* m, st_pool* p, const float* wav24, size_t n, float out[512]);
/* The same from 16 kHz samples (skips the resample): the first 128000 samples (8 s), tiled when shorter than 8000. */
int st_wavlm_xvector16(st_wavlm* m, st_pool* p, const float* wav16, size_t n, float out[512]);

#ifdef __cplusplus
}
#endif
#endif
