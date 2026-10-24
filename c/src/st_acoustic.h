/* SteerTTS acoustic model (steertts/model.py SteerTTSModel.infer; the edge build is the same network): phone ids ->
 * natural-log mel [T, 100], conditioned on g = cond([speaker slot, tag embedding, ctrl * cmask, cmask]).
 * Encoder / decoder blocks: AdaLN -> multi-head self-attention, AdaLN -> bidirectional sequence mixing, AdaLN -> conv FFN. */
#ifndef ST_ACOUSTIC_H
#define ST_ACOUSTIC_H
#include "st_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ST_N_CTRL 12
#define ST_N_MELS 100

typedef struct st_acoustic st_acoustic;

/* longest single pass, in mel frames (65536 frames = 699 s at 24 kHz / hop 256) */
#define ST_MAX_FRAMES (1 << 16)

/* w: model.safetensors (v1 fp32 / int8 packed) or the edge acoustic.safetensors; arch: config.json
 * "acoustic"."arch" (NULL: SteerTTSModel.__init__ defaults). NULL + st_seterr on failure. */
st_acoustic* st_acoustic_load(st_weights* w, const st_json* arch);
void st_acoustic_free(st_acoustic* a);
int st_acoustic_g_dim(const st_acoustic* a);
int st_acoustic_n_symbols(const st_acoustic* a);   /* embedding rows; ids >= this must be dropped (pipeline._render) */
int st_acoustic_n_tags(const st_acoustic* a);
int st_acoustic_spk_enc_dim(const st_acoustic* a);   /* 512 (spk_proj input), 0 if the model has no spk_proj */

/* g = cond([s, tag_emb, ctrl * cmask, cmask]); s = spk_proj(spk_vec) if spk_vec != NULL (spk_enc_dim floats), else
 * spk64 (a 64-d slot, e.g. a row of the edge voices.safetensors "s"). ctrl: 12 z values (NaN -> 0, +-inf -> +-FLT_MAX
 * as torch.nan_to_num), cmask: 12 x {0, 1}. tag out of range -> error. g: [g_dim]. */
int st_acoustic_condition(const st_acoustic* a, const float* spk64, const float* spk_vec, int tag, const float* ctrl,
                          const float* cmask, float* g);

/* Per-word shift in model units: pitch = semitones / PITCH_SCALE (3), energy = dB / 10, dur = factor (1 = none). */
typedef struct { float pitch, energy, dur; } st_shift;

/* model.infer for one utterance. ids / widx [N] (N >= 1; ids < n_symbols); g [g_dim]; g0 NULL or the base
 * conditioning for guidance (then every stage is base + guidance * (request - base), durations first, as
 * model.infer with base= and guidance != 1; pass g0 = NULL for guidance == 1); pace > 0 divides the durations;
 * shifts NULL or [n_words], applied to the tokens with widx == w (0 <= w < n_words; other widx values are not
 * shifted). Returns mel [T, 100] (time-major, st_malloc'd, caller st_free's) and T; dur_out (optional) gets the N
 * per-token frame counts (st_malloc'd). */
int st_acoustic_infer(st_acoustic* a, st_pool* p, const int* ids, const int* widx, int N, const float* g,
                      const float* g0, float guidance, float pace, const st_shift* shifts, int n_words, float** mel,
                      int* T, int** dur_out);

#ifdef __cplusplus
}
#endif
#endif
