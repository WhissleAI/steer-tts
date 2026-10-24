/* SteerTTS edge, C API (for C, Swift, Kotlin/JNI, Rust, Go ... bindings). Thread-compatible: one engine per thread,
 * or serialize calls on a shared engine.
 *
 *   char err[256];
 *   steertts_engine* e = steertts_create("steer-tts-edge", 1, err, sizeof err);
 *   steertts_options o; steertts_default_options(&o); o.tag = "happiness"; o.ctrl = "rate=1,f0_iqr=1.5";
 *   float* audio; size_t n;
 *   if (steertts_synthesize(e, "I can't believe it worked.", &o, &audio, &n, err, sizeof err) == 0) {
 *     ... 24 kHz mono float32 ...
 *     steertts_free_audio(audio);
 *   }
 *   steertts_destroy(e);
 */
#ifndef STEERTTS_C_H
#define STEERTTS_C_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct steertts_engine steertts_engine;

typedef struct {
  const char* voice;   /* NULL: package default */
  const char* tag;     /* NULL: "none" */
  const char* ctrl;    /* "name=z,name=z" or NULL */
  float pace;          /* 1.0 */
  float guidance;      /* < 0: engine default (1.0) */
  int markup;          /* parse inline markup */
} steertts_options;

/* Returns NULL on failure (message in err). threads: ONNX Runtime intra-op threads. */
steertts_engine* steertts_create(const char* package_dir, int threads, char* err, size_t err_len);
void steertts_destroy(steertts_engine* e);
void steertts_default_options(steertts_options* o);
int steertts_sample_rate(void);

/* 0 on success: *audio (free with steertts_free_audio) holds *n samples. Non-zero on error (message in err). */
int steertts_synthesize(steertts_engine* e, const char* text, const steertts_options* o, float** audio, size_t* n,
                        char* err, size_t err_len);
void steertts_free_audio(float* audio);

/* Sentence-by-sentence rendering (inline markup is always parsed). The callback gets each chunk as soon as it is
 * ready; the pointer is valid only during the call. Return non-zero from the callback to stop. */
typedef int (*steertts_chunk_fn)(const float* audio, size_t n, void* user);
int steertts_stream(steertts_engine* e, const char* text, const steertts_options* o, steertts_chunk_fn fn, void* user,
                    char* err, size_t err_len);

/* 16-bit PCM WAV, 24 kHz mono. 0 on success. */
int steertts_save_wav(const float* audio, size_t n, const char* path);

#ifdef __cplusplus
}
#endif
#endif
