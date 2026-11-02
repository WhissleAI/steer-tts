/* SteerTTS Nano: int8 streaming TTS runtime for microcontrollers (ESP32-S3 without PSRAM) and hosts.
 *
 * The model is one nano.bin file (format: docs/NANO_BIN.md) that is read in place: on the ESP32-S3 it is a
 * memory-mapped flash partition, so weights never take RAM. All working memory comes from one caller-provided arena
 * (no malloc inside the runtime); nano_arena_bytes() says how much a model needs.
 *
 * Input is a phoneme id sequence in the symbol table of the SteerTTS edge text front end (118 ids: 0 <pad>, 1 <sp>,
 * 2..8 punctuation, 9..77 ARPAbet phones, 78..117 event tokens), plus the steering inputs: voice, style tag, the 12
 * controls and optional per-word prosody shifts. Output is 16-bit mono PCM at the model's sample rate, delivered
 * through a callback in hop-sized blocks while the utterance is being synthesized.
 *
 *   nano_model m;  nano_model_open(&m, blob, blob_len);
 *   nano_ctx* c = nano_ctx_create(&m, arena, arena_len);
 *   nano_request r;  nano_request_default(&r);  r.ids = ids;  r.n_ids = n;
 *   nano_synthesize(c, &r, on_pcm, user);
 *
 * Not thread-safe per context; a model may be shared by several contexts. */
#ifndef STEERTTS_NANO_H
#define STEERTTS_NANO_H

#include <stddef.h>
#include <stdint.h>

#include "nano_bin.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NANO_N_CTRL_MAX 24   /* model inputs: 12 values, or 24 = [value * set (12), set (12)] */
#define NANO_N_CTRL 12       /* controls of the API */

/* ------------------------------------------------------------------ model (weights, read in place) */
typedef struct nano_qlayer {
  const int8_t* w;
  const int32_t* b;
  const int32_t* m;
  const int32_t* s;
  const float* sc;   /* sc[0] input scale, sc[1] output scale (sc[2]: phase scale of voc.head) */
  int out, k, in;    /* depthwise: out == in == channels */
} nano_qlayer;

typedef struct nano_blockw {
  nano_qlayer dw, pw1, pw2;
  const int8_t* ada_w;   /* adaptive: [2d][d_cond] */
  const float *ada_ws, *ada_b;
  const float *ln_g, *ln_b;   /* plain LayerNorm */
  float res;   /* output scale */
} nano_blockw;

typedef struct nano_predw {
  nano_qlayer c[4];
  const float *ln_g[4], *ln_b[4];
  const float* out_w;
  float out_b;
} nano_predw;

#define NANO_MAX_LAYERS 12

typedef struct nano_model {
  const uint8_t* base;
  size_t len;
  const nano_bin_header* hdr;
  int cfg[NANO_CFG_COUNT];
  float fcfg[NANO_FCFG_COUNT];
  const float *voice, *tag, *ctrl_w, *ctrl_b;
  const int8_t* emb;
  nano_blockw enc[NANO_MAX_LAYERS], dec[NANO_MAX_LAYERS], voc[NANO_MAX_LAYERS];
  nano_predw pred[3];   /* dur, f0, en */
  const float *pe_w[2], *pe_b[2];
  nano_qlayer dec_in, dec_mel, voc_embed, voc_head;
  const float *voc_ln0_g, *voc_ln0_b, *voc_lnf_g, *voc_lnf_b;
  uint32_t weight_bytes;    /* tensor data bytes */
  uint32_t n_params_i8;
} nano_model;

/* Parse and validate a nano.bin image (kept referenced, not copied). 0, or -1 with nano_last_error().
 * A nano_model is about 6 KB (tensor pointers): make it static or heap-allocated, not a local on a small stack. */
int nano_model_open(nano_model* m, const void* data, size_t len);
const char* nano_last_error(void);

/* ------------------------------------------------------------------ synthesis */
typedef struct nano_ctx nano_ctx;

typedef struct {
  int word;                      /* word index (nano_request.widx) */
  float pitch, energy, dur;      /* pitch += pitch / 3, energy += energy / 10, frames *= dur */
} nano_word_shift;

typedef struct {
  const int* ids;          /* phoneme ids (<= max_phonemes) */
  int n_ids;
  const int* widx;         /* per id: word index or -1; NULL if no word shifts */
  const nano_word_shift* ws;
  int n_ws;
  int voice;               /* 0 .. n_voices-1 */
  int tag;                 /* 0 = none */
  const float* ctrl;       /* the 12 z-scored controls (rate, loud, f0_med, f0_iqr, f0_end, pause, tilt, voiced,
                              flat, centroid, cpps, h1h2); NULL = none set */
  const uint8_t* ctrl_set; /* 12 flags, 1 = that control is set; NULL = all 12 set when ctrl != NULL. A model with
                              24 control inputs sees [value * set, set], so "not set" differs from "set to 0" */
  float pace;              /* speed, > 1 = faster (edge API): frames = round((exp(log_dur) - dur_offset) / pace *
                              word dur), see docs/NANO_BIN.md */
  int fixed_dur;           /* > 0: every phoneme gets this many frames (benchmarks) */
  int chunk;               /* frames per weight pass, 0 = model default */
  int first_chunk;         /* 0 = model default */
} nano_request;

void nano_request_default(nano_request* r);

/* Bytes of arena a context needs for this model and chunk size (0 = model default). */
size_t nano_arena_bytes(const nano_model* m, int chunk);
nano_ctx* nano_ctx_create(const nano_model* m, void* arena, size_t arena_bytes);   /* NULL if too small for the context */

/* PCM callback: n samples (a hop, or the tail); return nonzero to stop synthesis. */
typedef int (*nano_pcm_fn)(void* user, const int16_t* pcm, int n);

/* Synthesize one utterance. Returns the number of samples produced, or -1 (nano_last_error()). */
long nano_synthesize(nano_ctx* c, const nano_request* r, nano_pcm_fn fn, void* user);

/* ------------------------------------------------------------------ profiling */
enum nano_stage {
  NANO_ST_COND = 0,
  NANO_ST_ENC,        /* embedding + encoder blocks */
  NANO_ST_PRED,       /* duration / pitch / energy predictors */
  NANO_ST_PROJ,       /* pitch + energy embedding, projection to the decoder width */
  NANO_ST_DEC,        /* decoder blocks */
  NANO_ST_MEL,        /* mel projection */
  NANO_ST_VOC,        /* vocoder embedding + ConvNeXt blocks + final norm */
  NANO_ST_HEAD,       /* vocoder magnitude / phase head */
  NANO_ST_ISTFT,      /* spectrum, inverse FFT, window, overlap-add, PCM */
  NANO_ST_COUNT
};
enum nano_op {
  NANO_OP_GEMM = 0,   /* int8 matrix products (pointwise layers, full convs, heads) */
  NANO_OP_DW,         /* depthwise convs */
  NANO_OP_NORM,       /* LayerNorm / AdaLN + requantization */
  NANO_OP_ELEM,       /* residual adds, LUT activations, copies */
  NANO_OP_FFT,
  NANO_OP_COUNT
};
typedef struct {
  uint64_t cycles[NANO_ST_COUNT];
  uint64_t op_cycles[NANO_OP_COUNT];
  uint64_t macs[NANO_ST_COUNT];          /* int8 multiply-accumulates */
  uint64_t weight_bytes[NANO_ST_COUNT];  /* weight bytes streamed (counting every pass) */
  uint32_t passes[NANO_ST_COUNT];        /* weight passes */
  long frames, samples, phonemes;
  uint32_t hash_enc, hash_p, hash_mel, hash_head;   /* FNV-1a of int8 hand-over tensors (cross-platform checks) */
  int sample_rate;
  size_t arena_peak;                     /* high-water mark of the arena, bytes */
} nano_profile;

const nano_profile* nano_ctx_profile(const nano_ctx* c);
void nano_ctx_profile_reset(nano_ctx* c);
const char* nano_stage_name(int stage);
const char* nano_op_name(int op);
/* print RESULT| lines (cycles_per_us: 240 for CCOUNT at 240 MHz, 1000 for host nanoseconds) */
void nano_print_report(const nano_profile* p, double cycles_per_us, const char* tag);
/* benchmark helpers: random phoneme ids (xorshift32 state, start with 1), PCM hash (start with 2166136261) */
void nano_bench_ids(int* ids, int n, uint32_t* state);
uint32_t nano_hash_pcm(uint32_t h, const int16_t* pcm, int n);

/* ------------------------------------------------------------------ kernels (exposed for tests and benchmarks) */
/* out[r][o] = clamp(requant(sum_i in[r*in_stride + i] * w[o*K + i] + b[o], m[o], s[o]), lo, hi), r < rows.
 * One pass over the weights for all rows. */
void nano_gemm_s8(const int8_t* in, int in_stride, int rows, const int8_t* w, int K, const int32_t* b,
                  const int32_t* m, const int32_t* s, int8_t* out, int out_stride, int out_ch, int lo, int hi);
void nano_gemm_s8_ref(const int8_t* in, int in_stride, int rows, const int8_t* w, int K, const int32_t* b,
                      const int32_t* m, const int32_t* s, int8_t* out, int out_stride, int out_ch, int lo, int hi);
/* int8 dot product of n bytes (the SIMD path needs n % 16 == 0 and 16-byte aligned pointers). */
int32_t nano_dot_s8(const int8_t* a, const int8_t* b, int n);
int32_t nano_dot_s8_ref(const int8_t* a, const int8_t* b, int n);
/* Depthwise conv over time: in has rows + k - 1 rows of C channels; out has rows rows. */
void nano_dw_s8(const int8_t* in, int rows, int C, const int8_t* w, int k, const int32_t* b, const int32_t* m,
                const int32_t* s, int8_t* out, int lo, int hi);
void nano_dw_s8_ref(const int8_t* in, int rows, int C, const int8_t* w, int k, const int32_t* b, const int32_t* m,
                    const int32_t* s, int8_t* out, int lo, int hi);
int32_t nano_requant(int32_t acc, int32_t mult, int32_t shift);
const char* nano_kernel_impl(void);   /* "esp32s3-simd" or "c" */

/* platform cycle counter (CPU cycles on the ESP32-S3, nanoseconds on hosts) */
uint64_t nano_cycles(void);

#ifdef __cplusplus
}
#endif
#endif
