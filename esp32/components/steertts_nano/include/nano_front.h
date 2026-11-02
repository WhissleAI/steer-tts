/* SteerTTS Nano text front end for microcontrollers: text (with steering markup) -> phoneme ids.
 *
 * Same behaviour as the SteerTTS edge text front end (normalisation of numbers, dates, times, currencies, units,
 * abbreviations and acronyms; CMUdict lexicon; heteronym disambiguation; neural G2P for unknown words; the markup
 * of tags, prosody spans, pauses and sound events), rewritten for small devices:
 *   - all data is one read-only file, front.bin (tools/make_front_bin.py), read in place from memory-mapped flash;
 *   - the rules are plain C (no regular-expression engine, nothing parsed at run time);
 *   - no heap: all working memory is one nano_front_ctx (about 10 KB), reused for every sentence.
 *
 * Phoneme ids use the edge symbol table (see steertts_nano.h). Not thread-safe per context. */
#ifndef NANO_FRONT_H
#define NANO_FRONT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NF_MAX_CP 640        /* code points per normalised sentence (longer input: an error) */
#define NF_MAX_TOK 256       /* G2P tokens per sentence */
#define NF_MAX_PH 512        /* phoneme ids per sentence */
#define NF_MAX_STYLES 8
#define NF_MAX_CTRL 12
#define NF_MAX_WS 32
#define NF_GRU_MAX_H 256

typedef struct nano_front nano_front;   /* the opened front.bin (pointers into it) */

typedef struct {
  const uint8_t* base;
  size_t len;
  /* section pointers (filled by nano_front_open) */
  const uint8_t *word, *space, *digit, *g2p_ascii, *g2p, *fold, *lower, *lex, *oov, *gru, *het_w, *het_t, *het_h;
  int has_td;
} nano_front_data;

/* Word prosody shift of a markup span: pitch += pitch / 3, energy += energy / 10, frames *= dur. */
typedef struct {
  int word;
  float pitch, energy, dur;
} nf_word_shift;

/* One synthesis segment of a markup text, in playback order (MarkupSynth.parse). */
typedef struct {
  char text[2 * NF_MAX_CP];         /* UTF-8, whitespace-collapsed; "" for pauses / events */
  char tag[32];                     /* style tag, "none" = no style */
  int n_ctrl;
  char ctrl_name[NF_MAX_CTRL][16];  /* sorted by name */
  float ctrl_val[NF_MAX_CTRL];
  int n_ws;
  nf_word_shift ws[NF_MAX_WS];
  int pause_ms;                     /* > 0: silence instead of speech */
  int skip;                         /* the "PAUSE" / "EVENT" literal quirk: render nothing */
  char event[24];                   /* non-empty: a recorded sound event from the event bank */
} nf_segment;

typedef int (*nf_event_fn)(void* user, const char* name);   /* 1 if `name` is rendered from the event bank */

typedef struct {
  char tag[32];
  int n;
  char name[NF_MAX_CTRL][16];
  float val[NF_MAX_CTRL];
} nf_style;

/* markup parser state (inside nano_front_ctx) */
typedef struct {
  const char* src;
  int len, pos, emph_close;
  const char* const* tags;
  int ntags;
  nf_event_fn is_event;
  void* ev_user;
  nf_style st[NF_MAX_STYLES];
  int nst, sstack[32], nss, pstack[32], nps;
  /* current run (text between tokens) being split into sentences */
  int run_kind, run_a, run_b, run_style, run_ms, run_syn, run_split;
  unsigned run_pros;
  char run_buf[48];
  /* the segment being built and the special segment queued behind it */
  int cur_n, cur_style, cur_nonspace, queued, q_kind, q_ms, q_skip;
  int pend, pend_a, pend_b, pend_style;   /* a piece that starts the next segment (after a style change) */
  unsigned pend_pros;
  char q_event[24];
  int maxw;   /* words of the current segment's text (n_words memo) */
  uint32_t memo_hash;
  int memo_len, memo_n, done;
} nf_markup;

/* Working memory (static or heap; ~10 KB). */
typedef struct {
  nano_front_data d;
  char err[160];
  int fail;
  /* normalisation: two code point buffers */
  uint32_t a[NF_MAX_CP], b[NF_MAX_CP];
  int na, nb;
  /* G2P */
  char asc[NF_MAX_CP];
  uint16_t tok_s[NF_MAX_TOK];
  uint8_t tok_n[NF_MAX_TOK];
  int ntok;
  /* the G2P GRU state lives in `a` (idle while words are converted) */
  int count_only, part_maxw;   /* word counting mode of to_phonemes (markup) */
  nf_markup mk;
} nano_front_ctx;

/* Open front.bin (16-byte aligned, read in place). 0 or -1 (ctx->err). */
int nano_front_open(nano_front_ctx* c, const void* data, size_t len);

/* text.to_phonemes: ids start and end with <sp> (1); widx[i] = word index of id i or -1. Returns the id count
 * (<= cap) or -1 with c->err ("ValueError: ..." for inputs Python rejects, "too long" past NF_MAX_CP). */
int nano_front_phonemes(nano_front_ctx* c, const char* utf8, int* ids, int* widx, int cap);
/* text.normalize into out (UTF-8). Returns the length or -1. */
int nano_front_normalize(nano_front_ctx* c, const char* utf8, char* out, int cap);

/* The neural G2P alone (raw model phoneme indices, <= 20), for tests. */
int nano_front_gru_word(nano_front_ctx* c, const char* word, int* raw);

/* Markup, one segment at a time (MarkupSynth(tts).parse(text, tag=default_tag, ctrl=default ctrl)).
 * known_tags: the model's style tags; is_event: optional event-bank membership (NULL: events stay in the text).
 * begin returns 0 / -1; next returns 1 with a segment, 0 at the end, -1 on error. `text` must stay valid. */
int nano_front_markup_begin(nano_front_ctx* c, const char* text, const char* default_tag, const char* const* ctrl_names,
                            const float* ctrl_vals, int n_ctrl, const char* const* known_tags, int n_tags,
                            nf_event_fn is_event, void* event_user);
int nano_front_markup_next(nano_front_ctx* c, nf_segment* seg);

#ifdef __cplusplus
}
#endif
#endif
