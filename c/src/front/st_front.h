/* SteerTTS edge text front end (C11, libc + libm + pthreads).
 *
 * Bit-exact port of the Python edge front end (steertts_edge: text.normalize / text.to_phonemes /
 * markup.MarkupSynth.parse with its edge_front backends) via the C++ port in native/src/front; verified against
 * native/tests/front/golden*.jsonl by tests/test_front.c. Assets: the edge package's native/front directory
 * (unicode.bin, lexicon.bin, oov.bin, gru.bin, hetero.tsv).
 *
 * Symbol table (118 ids, == text.SYMBOLS): 0 <pad>, 1 <sp> (word boundary), 2..8 punctuation , . ? ! ; : -
 * (only , . ? ! - occur), 9..77 ARPAbet phones (AA0 = 9 ... ZH = 77), 78..117 sound-event tokens ([laugh] = 78 ...
 * [clock] = 104, [reserved0..12] = 105..117).
 * "Nothing to say": the synthesiser renders no audio when no id >= 9 is present (st_front_has_speech); event-only
 * input like "[laugh]" (ids 1 78 1) IS rendered.
 *
 * Errors: inputs that make Python raise return -1 / NULL with st_geterr() == "<PythonError>: <Python's message>",
 * e.g. "ValueError: could not convert string to float: '1.2.3'", "OverflowError: abs(...) must be less than ...",
 * "IndexError: list index out of range" (messages longer than st_base's 1023-byte error buffer are truncated).
 *
 * Thread safety: an st_front is immutable after loading except for a G2P prediction cache guarded by a lock, so
 * every function may be called concurrently on one st_front. Not affected by the C locale (LC_NUMERIC). */
#ifndef ST_FRONT_H
#define ST_FRONT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_front st_front;
st_front* st_front_load(const char* asset_dir);   /* NULL + st_seterr on failure */
void st_front_free(st_front* f);

/* == text.to_phonemes(...)[:2]: ids start with <sp> and end with <sp> after the last word (if any); widx[i] = index
 * of the word id i belongs to, -1 for <sp> / punctuation / events. */
typedef struct {
  int* ids;
  int* widx;
  int n;
} st_phonemes;
int st_front_phonemes(st_front* f, const char* utf8, st_phonemes* out);   /* 0, or -1 + st_seterr (out zeroed) */
void st_phonemes_free(st_phonemes* p);
char* st_front_normalize(st_front* f, const char* utf8);   /* == text.normalize; st_free() it; NULL on error */
int st_front_has_speech(const st_phonemes* p);              /* any id >= 9 (events count) */
int st_front_symbol_id(const st_front* f, const char* sym);  /* == SYM2ID[sym], -1 if absent */
const char* st_front_symbol(const st_front* f, int id);      /* == SYMBOLS[id], NULL if out of range */

/* Per-word prosody shift in markup.PROSODY user units. The acoustic code applies, for every token whose widx ==
 * word: pitch += float32(pitch / 3.0), energy += float32(energy / 10.0), dur *= dur (see native front.h). */
typedef struct {
  int word;
  float pitch, energy, dur;
} st_word_shift;

typedef struct {
  char* text;            /* whitespace-collapsed segment text (feed to st_front_phonemes); "" for pauses */
  char* tag;             /* style tag, "none" = no style */
  int n_ctrl;
  char** ctrl_name;      /* sorted by name (Python: dict(sorted(items))) */
  float* ctrl_val;
  int n_ws;
  st_word_shift* ws;     /* sorted by word; word = index in st_front_phonemes(text).widx */
  int pause_ms;          /* > 0: silence of that length instead of speech; -1 with skip = 1: the PAUSE quirk */
  int skip;              /* 1: Python quirk segment, render nothing (see below) */
  char* event;           /* non-NULL: a recorded sound event ([laugh] ...) from the event bank, text "" */
} st_segment;

/* Optional event-bank membership test for st_front_parse_markup_ev: 1 if the lower-case token name (e.g. "laugh") is
 * rendered from the bank (its own segment), 0 to keep it in the TTS text (MarkupSynth with tts.events). */
typedef int (*st_event_fn)(void* user, const char* name);

/* Optional resolver for <style prompt="...">, the v1 prompt adapter (steertts/markup.py _style_of_prompt): returns
 * the span's style tag (the adapter's first top tag the model knows and != "none", else "none") and its controls.
 * All outputs are st_malloc'd (tag, ctrl_names, every name, ctrl_vals) and freed by the front end; duplicate names:
 * the later wins (dict). Returns 0, or -1 with st_seterr (parse_markup then fails with that message). */
typedef int (*st_prompt_style_fn)(void* user, const char* prompt, char** tag, int* n_ctrl, char*** ctrl_names,
                                  float** ctrl_vals);

/* == MarkupSynth(tts).parse(text, prompt=None, tag=default_tag, ctrl=default_ctrl) where tts.tag2i has the keys
 * known_tags. default_tag NULL / "" == None. default ctrl: insertion-ordered pairs (later duplicates win, like a
 * dict). prompt_fn NULL: the edge behaviour (a non-empty <style prompt> raises "ValueError: free-text prompts are
 * resolved on the host ..."); non-NULL: the v1 behaviour (the span gets prompt_fn's tag + ctrl; an empty prompt
 * is ("none", {}) without a call). Segments are in playback order; render each non-pause, non-skip segment with its
 * tag / ctrl / ws and insert pause_ms of silence for pause segments.
 * Quirk replicated for parity: a text piece that is exactly "PAUSE" / "EVENT" (e.g. "Stop. PAUSE") becomes a
 * segment holding the style tuple in Python's pause_ms / event field; here skip = 1 (pause_ms = -1 for "PAUSE",
 * 0 for "EVENT"). Engines skip those segments.
 * Streaming rule (MarkupSynth.stream()): pause -> int(pause_ms * 24000 / 1000) zero samples; speech -> rendered
 * audio, if longer than 2*240 samples with 240-sample linear fades (k/239) at both ends; concatenated in order. */
int st_front_parse_markup(st_front* f, const char* text, const char* default_tag, const char* const* ctrl_names,
                          const float* ctrl_vals, int n_ctrl, const char* const* known_tags, int n_tags,
                          st_prompt_style_fn prompt_fn, void* prompt_user, st_segment** segs, int* n_segs);
/* st_front_parse_markup plus the event bank: tokens with is_event(user, name) == 1 become event segments. */
int st_front_parse_markup_ev(st_front* f, const char* text, const char* default_tag, const char* const* ctrl_names,
                             const float* ctrl_vals, int n_ctrl, const char* const* known_tags, int n_tags,
                             st_prompt_style_fn prompt_fn, void* prompt_user, st_event_fn is_event, void* event_user,
                             st_segment** segs, int* n_segs);
void st_segments_free(st_segment* s, int n);

#ifdef __cplusplus
}
#endif
#endif
