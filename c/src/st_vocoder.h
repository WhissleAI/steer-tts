/* Vocos-style vocoder (ConvNeXt backbone + iSTFT head): 100-band natural-log mel at 24 kHz, hop 256 -> waveform.
 * One implementation for charactr/vocos-mel-24khz (v1; tools/export_vocos.py, fp32 or packed int8) and the edge
 * vocoder (packed int4 / int8). State-dict names as vocos.Vocos (backbone.*, head.out.*). */
#ifndef ST_VOCODER_H
#define ST_VOCODER_H
#include <stddef.h>

#include "st_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_vocoder st_vocoder;
/* v1 Vocos: dim 512, inter 1536, 8 layers; edge: 384 / 896 / 8; both n_fft 1024, hop 256 (padding "center").
 * NULL + st_seterr on failure. */
st_vocoder* st_vocoder_load(st_weights* w, int dim, int inter, int layers, int n_fft, int hop);
void st_vocoder_free(st_vocoder* v);
/* mel [T, 100] -> wav ((T - 1) * hop samples, st_malloc'd, caller st_free's). */
int st_vocoder_decode(st_vocoder* v, st_pool* p, const float* mel, int T, float** wav, size_t* n);

#ifdef __cplusplus
}
#endif
#endif
