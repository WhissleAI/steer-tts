/* nano.bin: the SteerTTS Nano weight file (format version 1). Full description: docs/NANO_BIN.md.
 *
 * One flat little-endian file, read in place (memory-mapped flash on the ESP32-S3, mmap or a heap copy on a host):
 *
 *   [0, 256)            nano_bin_header: magic, version, layout offsets, the model configuration
 *   [dir_offset, ...)   n_tensors x nano_bin_entry (64 bytes each): name, dtype, shape, offset, size
 *   [data_offset, ...)  tensor data; every tensor starts on a 16-byte boundary (offsets are from the file start)
 *
 * Weights are int8 with per-output-channel requantization (TFLite convention: int32 multiplier + shift per output
 * channel, symmetric int8 activations with zero point 0), so the int8 dot products map onto esp-nn kernels and run
 * bit-exactly on any target. Small conditioning and norm tensors are float32. */
#ifndef NANO_BIN_H
#define NANO_BIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NANO_BIN_MAGIC "STNB"
#define NANO_BIN_VERSION 1
#define NANO_BIN_HEADER_BYTES 256
#define NANO_BIN_ENTRY_BYTES 64
#define NANO_BIN_NAME_LEN 40
#define NANO_BIN_ALIGN 16

enum nano_dtype {
  NANO_I8 = 1,
  NANO_I32 = 2,
  NANO_F32 = 3,
};

enum nano_act {
  NANO_ACT_RELU = 0,
  NANO_ACT_GELU = 1,   /* erf GELU, applied as a 256-entry int8 -> int8 table */
};

/* Model configuration: header.cfg[] indices (int32 values). */
enum nano_cfg {
  NANO_CFG_SAMPLE_RATE = 0,   /* 16000 */
  NANO_CFG_N_FFT,             /* 1024 (power of two) */
  NANO_CFG_HOP,               /* 256 */
  NANO_CFG_WIN,               /* 1024 (== n_fft, periodic Hann) */
  NANO_CFG_N_MELS,            /* 80 */
  NANO_CFG_N_SYMBOLS,         /* 118: the edge front end's symbol table */
  NANO_CFG_D_COND,            /* conditioning vector size */
  NANO_CFG_N_VOICES,
  NANO_CFG_N_CTRL,            /* 12 */
  NANO_CFG_N_TAGS,            /* 51 (tag 0 = "none") */
  NANO_CFG_ENC_LAYERS,
  NANO_CFG_ENC_DIM,
  NANO_CFG_ENC_FF,
  NANO_CFG_ENC_K,
  NANO_CFG_PRED_LAYERS,
  NANO_CFG_PRED_DIM,
  NANO_CFG_PRED_K,
  NANO_CFG_PE_K,              /* pitch / energy embedding conv kernel */
  NANO_CFG_DEC_LAYERS,
  NANO_CFG_DEC_DIM,
  NANO_CFG_DEC_FF,
  NANO_CFG_DEC_K,
  NANO_CFG_VOC_EMBED_K,
  NANO_CFG_VOC_LAYERS,
  NANO_CFG_VOC_DIM,
  NANO_CFG_VOC_FF,
  NANO_CFG_VOC_K,
  NANO_CFG_CHUNK,             /* frames per decoder + vocoder weight pass (32 = 0.5 s default, 64 = 1 s) */
  NANO_CFG_MAX_PHONEMES,      /* longest phoneme sequence per call (split sentences above it) */
  NANO_CFG_ENC_ACT,           /* enum nano_act */
  NANO_CFG_DEC_ACT,
  NANO_CFG_VOC_ACT,
  NANO_CFG_MAX_DUR,           /* frames per phoneme clamp */
  NANO_CFG_COND_ACT,          /* 0: none, 1: SiLU on the conditioning vector */
  NANO_CFG_FIRST_CHUNK,       /* frames of the first chunk (lower latency); 0 = chunk */
  NANO_CFG_COUNT = 48
};

/* header.fcfg[] indices (float32 values). */
enum nano_fcfg {
  NANO_FCFG_DUR_OFFSET = 0,   /* frames = exp(log_dur) - dur_offset (then * pace * word factor, rounded) */
  NANO_FCFG_MAG_MAX,          /* vocoder magnitude clip, 100 */
  NANO_FCFG_WS_PITCH_DIV,     /* word shift: pitch += shift / 3 */
  NANO_FCFG_WS_ENERGY_DIV,    /* word shift: energy += shift / 10 */
  NANO_FCFG_LN_EPS,           /* 1e-5 */
  NANO_FCFG_COUNT = 8
};

typedef struct {
  char magic[4];          /* "STNB" */
  uint32_t version;       /* NANO_BIN_VERSION */
  uint32_t header_bytes;  /* 256 */
  uint32_t n_tensors;
  uint32_t dir_offset;
  uint32_t data_offset;
  uint32_t file_bytes;
  uint32_t flags;         /* 0 */
  int32_t cfg[NANO_CFG_COUNT];
  float fcfg[NANO_FCFG_COUNT];
} nano_bin_header;

typedef struct {
  char name[NANO_BIN_NAME_LEN];   /* NUL-padded ASCII, e.g. "dec.2.pw1.w" */
  uint16_t dtype;                 /* enum nano_dtype */
  uint16_t ndim;                  /* 1..3 */
  uint32_t shape[3];              /* unused trailing dims = 1 */
  uint32_t offset;                /* from the start of the file, multiple of 16 */
  uint32_t nbytes;
} nano_bin_entry;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(nano_bin_header) == NANO_BIN_HEADER_BYTES, "nano_bin_header size");
_Static_assert(sizeof(nano_bin_entry) == NANO_BIN_ENTRY_BYTES, "nano_bin_entry size");
#endif

#ifdef __cplusplus
}
#endif
#endif
