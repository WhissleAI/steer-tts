/* SteerTTS, stand-alone C API (C11 library, no dependencies beyond libc, libm and threads).
 *
 * One library for both models:
 *   v1    (WhissleAI/steer-tts):      acoustic model + flow refiner + Vocos, free-text prompts (MiniLM) and voice
 *                                     cloning (WavLM x-vectors + the prompted flow), fp32 or int8 weights;
 *   edge  (WhissleAI/steer-tts-edge): the int4/int8 edge build (acoustic model + small vocoder).
 * steertts_create() reads the model directory and picks the model from its config.json.
 *
 *   char err[256];
 *   steertts_engine* e = steertts_create("steer-tts-edge", 1, err, sizeof err);
 *   steertts_options o;
 *   steertts_default_options(&o);
 *   o.tag = "happiness";
 *   o.ctrl = "rate=1,f0_iqr=1.5";
 *   float* audio;
 *   size_t n;
 *   if (steertts_synthesize(e, "I can't believe it worked.", &o, &audio, &n, err, sizeof err) == 0) {
 *     steertts_save_wav(audio, n, "out.wav");         // 24 kHz mono float32
 *     steertts_free(audio);
 *   }
 *   steertts_destroy(e);
 *
 * Every request field of the Python packages is available, either through steertts_options or, complete and
 * validated exactly like Python (request.schema.json), through the JSON entry points (steertts_synthesize_json).
 *
 * Errors: functions returning int give 0 on success and non-zero on failure, with a message in err (when
 * err != NULL; truncated to err_len); constructors return NULL. Invalid requests fail with the Python packages'
 * messages, checked in the same order. A NULL engine, request or output pointer is an error, not a crash; output
 * pointers are reset (NULL / 0) before anything can fail. Numbers are parsed and printed independently of the C
 * locale, so an application may use setlocale() freely.
 * Threads: an engine may be used by one thread at a time (it has its own worker pool, see steertts_config.threads);
 * use one engine per thread for concurrent synthesis, or serialize calls. Different engines are independent. */
#ifndef STEERTTS_H
#define STEERTTS_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(STEERTTS_SHARED)
#ifdef STEERTTS_BUILD
#define STEERTTS_API __declspec(dllexport)
#else
#define STEERTTS_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define STEERTTS_API __attribute__((visibility("default")))
#else
#define STEERTTS_API
#endif

#define STEERTTS_VERSION "1.2.0"
#define STEERTTS_SAMPLE_RATE 24000

typedef struct steertts_engine steertts_engine;

/* ------------------------------------------------------------------ engine */
typedef struct {
  int threads;              /* worker threads (default 1; 0 = number of CPUs) */
  const char* precision;    /* v1: "fp32" (default) or "int8" (packed int8 weights, ~4x smaller; same compute).
                               edge: ignored (always its int4/int8 weights) */
  int flow;                 /* v1: load the flow refiner (default 1). 0 skips it: ~12x less compute, lower detail */
  int flow_steps;           /* v1: ODE steps of the refiner (default 10, midpoint = 2 evaluations per step) */
  float flow_temp;          /* v1: refiner noise temperature (default 0.4) */
  float guidance;           /* default guidance scale for requests that do not set one (default 1.0) */
  const char* assets_dir;   /* directory with the converted helper models and text front-end assets;
                               NULL: <model_dir>/native */
} steertts_config;

STEERTTS_API void steertts_default_config(steertts_config* c);
/* NULL on failure (message in err). */
STEERTTS_API steertts_engine* steertts_create_ex(const char* model_dir, const steertts_config* c, char* err,
                                                 size_t err_len);
/* = steertts_create_ex with the default config and `threads` worker threads. */
STEERTTS_API steertts_engine* steertts_create(const char* model_dir, int threads, char* err, size_t err_len);
STEERTTS_API void steertts_destroy(steertts_engine* e);
/* Replaces the worker pool with one of `threads` threads (0 = number of CPUs), without reloading the model: e.g. when
 * the CPUs the process may use change (Android moves background apps to its slow cores). The new workers inherit the
 * calling thread's CPU affinity and priority. Not while the engine is synthesizing. 0 on success. */
STEERTTS_API int steertts_set_threads(steertts_engine* e, int threads);
STEERTTS_API int steertts_threads(const steertts_engine* e);   /* worker threads in use (1 = the calling thread) */
/* Linux / Android: kernel thread ids of the engine's worker threads (the calling thread is not among them), e.g. for
 * Android's APerformanceHint sessions. Returns how many were written (at most max); 0 on other systems. */
STEERTTS_API int steertts_worker_tids(const steertts_engine* e, int32_t* tids, int max);

STEERTTS_API const char* steertts_version(void);
STEERTTS_API int steertts_sample_rate(void);
STEERTTS_API const char* steertts_model_type(const steertts_engine* e);   /* "steer-tts" or "steer-tts-edge" */
STEERTTS_API int steertts_num_voices(const steertts_engine* e);
STEERTTS_API const char* steertts_voice(const steertts_engine* e, int i);   /* NULL if out of range */
STEERTTS_API const char* steertts_default_voice(const steertts_engine* e);
/* "female", "male" or NULL (unknown) for voice i, from the model's voices.json. */
STEERTTS_API const char* steertts_voice_gender(const steertts_engine* e, int i);
/* The default voice of a gender ("female" / "male"): config.json "default_voices", else the default voice if it has
 * that gender, else an evaluated pick; NULL if the model has no voice of that gender (or no gender metadata). */
STEERTTS_API const char* steertts_gender_voice(const steertts_engine* e, const char* gender);
STEERTTS_API int steertts_num_tags(const steertts_engine* e);
STEERTTS_API const char* steertts_tag(const steertts_engine* e, int i);
STEERTTS_API int steertts_num_controls(void);                                /* 12 */
STEERTTS_API const char* steertts_control(int i);                            /* "rate", "loud", ... */
/* Capabilities of the loaded model: "prompt" (free-text prompts), "clone" (ref_wav), "flow". 1 = available. */
STEERTTS_API int steertts_has(const steertts_engine* e, const char* feature);

/* ------------------------------------------------------------------ requests */
/* Always initialise with steertts_default_options() and then set fields (a zeroed struct also works: pace 0 means
 * the default 1.0, but guidance 0 then means "no steering" and flow 0 "refiner off"). */
typedef struct {
  const char* voice;        /* built-in voice, NULL = model default */
  const char* tag;          /* style tag, NULL = "none" (or the prompt's tag) */
  const char* ctrl;         /* controls as "name=z,name=z" (z in [-5, 5]), NULL = none */
  float pace;               /* duration scale, 1.0 = natural (0.25 .. 4; 0 = default) */
  float guidance;           /* < 0: engine default */
  int markup;               /* parse inline markup (*word*, <whisper>..</whisper>, [pause 300ms], ...) */
  /* v1 additions (edge rejects prompt / ref_wav with an error) */
  const char* prompt;       /* free-text direction ("excited, a bit faster"), mapped to tag + ctrl */
  const char* ref_wav;      /* clone the voice of this WAV file (instead of voice) */
  const float* spk_vec;     /* or: a 512-d speaker vector (steertts_speaker_vector); both models; wins over voice */
  int flow;                 /* v1: -1 = engine default, 0 = off, 1 = on */
  int64_t seed;             /* v1 refiner noise seed (default 0) */
  const char* word_shift;   /* per-word prosody "word:key=v,key=v;word:..." with keys pitch (semitones),
                               energy (dB), dur (factor); word = index into the words of the text. Not with markup. */
} steertts_options;

STEERTTS_API void steertts_default_options(steertts_options* o);

/* Whole text in one call: 0 on success, *audio (free with steertts_free) holds *n samples at 24 kHz. With markup the
 * text is rendered sentence by sentence and joined (10 ms fades at segment edges), as Python's synthesize. */
STEERTTS_API int steertts_synthesize(steertts_engine* e, const char* text, const steertts_options* o, float** audio,
                                     size_t* n, char* err, size_t err_len);

/* Sentence-by-sentence rendering (inline markup is always parsed): fn gets each chunk as soon as it is ready; the
 * audio pointer is valid during the call only. Return non-zero from fn to stop (steertts_stream then returns 0). */
typedef int (*steertts_chunk_fn)(const float* audio, size_t n, void* user);
STEERTTS_API int steertts_stream(steertts_engine* e, const char* text, const steertts_options* o, steertts_chunk_fn fn,
                                 void* user, char* err, size_t err_len);

typedef struct {
  const char* text;         /* the segment's text ("" for a pause) */
  const char* tag;          /* its style tag */
  const char* ctrl;         /* its controls as "name=z,..." ("" if none) */
  int pause_ms;             /* > 0 for a pause segment */
  int index;                /* 0-based segment index */
  double synth_ms;          /* time spent rendering this chunk */
} steertts_chunk_info;
typedef int (*steertts_chunk_ex_fn)(const float* audio, size_t n, const steertts_chunk_info* info, void* user);
STEERTTS_API int steertts_stream_ex(steertts_engine* e, const char* text, const steertts_options* o,
                                    steertts_chunk_ex_fn fn, void* user, char* err, size_t err_len);

/* JSON requests (request.schema.json: text, voice, ref_wav, prompt, tag, ctrl, pace, guidance, word_shift, markup,
 * flow, seed; plus spk_vec, a list of 512 numbers), validated like the Python packages (v1: normalize_request; edge:
 * EdgeTTS, plus range checks on pace and guidance). A repeated key: the last one wins, as in Python's json. */
STEERTTS_API int steertts_synthesize_json(steertts_engine* e, const char* request_json, float** audio, size_t* n,
                                          char* err, size_t err_len);
STEERTTS_API int steertts_stream_json(steertts_engine* e, const char* request_json, steertts_chunk_ex_fn fn,
                                      void* user, char* err, size_t err_len);
/* The fully resolved request (defaults filled in, prompt mapped to tag + ctrl, prompt_mapping recorded), as Python's
 * SteerTTS.resolve; *json_out is NUL-terminated, free with steertts_free. */
STEERTTS_API int steertts_resolve_json(steertts_engine* e, const char* request_json, char** json_out, char* err,
                                       size_t err_len);

/* ------------------------------------------------------------------ director: planning a performance
 * The director plans how each sentence of a longer text is performed: speaking rate, loudness, pitch level, pitch
 * range, how the sentence ends and the pause after it, from the sentence, the next one and its own plans for the
 * sentences before. A ~0.44M-parameter model in 8 bits (director.bin, ~0.5 MB, in the edge model directory), trained
 * on measured audiobook narration: it plans in the units of SteerTTS's controls, so a plan becomes a request directly.
 * It is independent of the engine (no steertts_engine needed) and takes ~20-60 us per sentence.
 *
 *   steertts_director* d = steertts_director_load("steer-tts-edge", err, sizeof err);   // a dir or director.bin
 *   steertts_sentence s[] = {{"\u201cWho are you?\u201d she asked.", 1, 1, 2, 0}, {"Nobody answered.", 0, 0, 2, 1}};
 *   float plan[2 * STEERTTS_PLAN_DIMS];
 *   steertts_director_plan(d, s, 2, "children", plan, err, sizeof err);
 *   char ctrl[160];
 *   steertts_plan_ctrl(plan, 1.0f, ctrl, sizeof ctrl);          // "rate=0.21,loud=..,f0_med=..,f0_iqr=..,f0_end=.."
 *   int pause = steertts_plan_pause_ms(plan, 1.0f);             // silence after the sentence
 *
 * plan[]: STEERTTS_PLAN_DIMS numbers per sentence: rate, loud, f0_med, f0_iqr (control z-scores), end (the final word's
 * pitch against the sentence's, z), gap (log(1 + seconds of silence after the sentence)). */
#define STEERTTS_PLAN_DIMS 6

typedef struct steertts_director steertts_director;
typedef struct steertts_reading steertts_reading;

/* One sentence as the director reads it: its text (UTF-8), whether it holds dialogue (quoted speech), whether a speaker
 * is attributed, and its place in its paragraph (para_len sentences, this one at para_index). */
typedef struct {
  const char* text;
  int dialogue;
  int has_speaker;
  int para_len;
  int para_index;
} steertts_sentence;

/* path: director.bin, or a directory holding it (the edge model's). NULL on failure (message in err). */
STEERTTS_API steertts_director* steertts_director_load(const char* path, char* err, size_t err_len);
/* From bytes in memory (e.g. an Android asset); they are copied. */
STEERTTS_API steertts_director* steertts_director_from_memory(const void* data, size_t n, char* err, size_t err_len);
STEERTTS_API void steertts_director_free(steertts_director* d);
/* The genres the director knows ("children", "adventure", "fantasy", "mystery", "scifi", "gothic", "literary",
 * "humour", "nonfiction", "poetry"); NULL past the end. A genre steadies the register; unknown is fine (NULL). */
STEERTTS_API const char* steertts_director_genre(int i);

/* A whole text at once: n sentences in order -> plans[n * STEERTTS_PLAN_DIMS]. genre may be NULL. 0 on success. */
STEERTTS_API int steertts_director_plan(const steertts_director* d, const steertts_sentence* sentences, size_t n,
                                        const char* genre, float* plans, char* err, size_t err_len);

/* Sentence by sentence (e.g. messages as they arrive): a reading keeps the context. next: the sentence after this one
 * when it is known (the director looks one ahead), else NULL. 0 on success. */
STEERTTS_API steertts_reading* steertts_reading_begin(const steertts_director* d, const char* genre);
STEERTTS_API int steertts_reading_next(steertts_reading* r, const steertts_sentence* s, const steertts_sentence* next,
                                       float plan[STEERTTS_PLAN_DIMS]);
STEERTTS_API void steertts_reading_free(steertts_reading* r);

/* A plan as request controls: strength scales it (1 = as narrators perform, 0 = off); the ending is mapped to f0_end.
 * out[5]: rate, loud, f0_med, f0_iqr, f0_end (z), clamped to +-3; add them to a voice's own style controls. */
STEERTTS_API void steertts_plan_controls(const float plan[STEERTTS_PLAN_DIMS], float strength, float out[5]);
/* The same as a ctrl string ("rate=0.21,loud=-0.10,..."): 0 on success (the string fits). */
STEERTTS_API int steertts_plan_ctrl(const float plan[STEERTTS_PLAN_DIMS], float strength, char* ctrl, size_t len);
/* The silence to leave after the sentence, in milliseconds (80 .. 1500), shorter at a faster pace. */
STEERTTS_API int steertts_plan_pause_ms(const float plan[STEERTTS_PLAN_DIMS], float pace);

/* ------------------------------------------------------------------ helpers */
/* v1: the 512-d speaker vector of a reference clip: a WAV file, or (path == NULL) n float samples at sr Hz. */
STEERTTS_API int steertts_speaker_vector(steertts_engine* e, const char* wav_path, const float* samples, size_t n,
                                         int sr, float out[512], char* err, size_t err_len);
/* v1: a free-text prompt -> {"tag": .., "ctrl": {..}, "top_tags": [[tag, p], ..]} as JSON (steertts_free it). */
STEERTTS_API int steertts_map_prompt(steertts_engine* e, const char* prompt, char** json_out, char* err,
                                     size_t err_len);
/* Phoneme ids of a text, as the model sees them (for debugging): *ids / *word_idx (steertts_free them). */
STEERTTS_API int steertts_phonemes(steertts_engine* e, const char* text, int** ids, int** word_idx, size_t* n,
                                   char* err, size_t err_len);

STEERTTS_API void steertts_free(void* p);
STEERTTS_API void steertts_free_audio(float* audio);   /* = steertts_free, for the native/ C API's name */
/* 16-bit PCM WAV, 24 kHz mono. 0 on success. */
STEERTTS_API int steertts_save_wav(const float* audio, size_t n, const char* path);
/* Read a WAV file (PCM 8/16/24/32-bit or float, any rate, channels averaged) -> float samples (steertts_free). */
STEERTTS_API int steertts_load_wav(const char* path, float** audio, size_t* n, int* sr, char* err, size_t err_len);

#ifdef __cplusplus
}
#endif
#endif
