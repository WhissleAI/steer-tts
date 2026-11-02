/* SteerTTS Nano front end internals. The rules mirror the edge front end's regular expressions one by one; every
 * matcher documents the expression it implements and follows Python `re` semantics (leftmost match, greedy
 * quantifiers with backtracking where it can change the result, Unicode \w \d \s). */
#ifndef NF_INT_H
#define NF_INT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "nano_front.h"

typedef uint32_t u32;

/* ------------------------------------------------------------------ errors */
void nf_fail(nano_front_ctx* c, const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

/* ------------------------------------------------------------------ code point buffers (fixed capacity) */
typedef struct {
  u32* p;
  int n, cap;
  nano_front_ctx* c;
} nfs;

static inline void nfs_push(nfs* s, u32 ch) {
  if (s->n >= s->cap) {
    if (!s->c->fail) nf_fail(s->c, "ValueError: text too long for the device front end");
    return;
  }
  s->p[s->n++] = ch;
}
static inline void nfs_put(nfs* s, const u32* q, int n) {
  for (int i = 0; i < n; i++) nfs_push(s, q[i]);
}
static inline void nfs_ascii(nfs* s, const char* a) {
  while (*a) nfs_push(s, (unsigned char)*a++);
}

/* char buffers for number words */
typedef struct {
  char* p;
  int n, cap;
  nano_front_ctx* c;
} nfa;
static inline void nfa_ch(nfa* s, char ch) {
  if (s->n + 1 >= s->cap) {
    if (!s->c->fail) nf_fail(s->c, "ValueError: text too long for the device front end");
    return;
  }
  s->p[s->n++] = ch;
  s->p[s->n] = 0;
}
static inline void nfa_cat(nfa* s, const char* a) {
  while (*a) nfa_ch(s, *a++);
}
static inline void nfa_put(nfa* s, const char* a, int n) {
  for (int i = 0; i < n; i++) nfa_ch(s, a[i]);
}

/* ------------------------------------------------------------------ data tables (front.bin) */
static inline u32 rd32(const uint8_t* p) {
  return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}
int nf_is_word(const nano_front_ctx* c, u32 ch);    /* re \w */
int nf_is_space(const nano_front_ctx* c, u32 ch);   /* re \s, str.isspace() */
int nf_digit(const nano_front_ctx* c, u32 ch);      /* decimal digit value or -1 (re \d) */
u32 nf_fold(const nano_front_ctx* c, u32 ch);       /* re.IGNORECASE folding */
int nf_lower(const nano_front_ctx* c, u32 ch, u32* out);   /* str.lower() of one code point -> count */
const char* nf_g2p_clean(const nano_front_ctx* c, u32 ch, int* len);

/* lexicon (prefix-compressed, block indexed): 1 if found; phones copied (up to 64) */
int nf_lex_get(const uint8_t* sec, const char* w, uint8_t* ph, int* nph);

/* ------------------------------------------------------------------ pieces */
int nf_utf8_decode(const char* s, size_t n, nfs* out);
int nf_utf8_encode(const u32* s, int n, char* out, int cap);
void nf_normalize(nano_front_ctx* c, const u32* s, int n, nfs* out);   /* out may not alias s */
void nf_collapse_strip(nano_front_ctx* c, nfs* t);                     /* re.sub(\s+, " ").strip(), in place */
void nf_normalize_numbers2(nano_front_ctx* c, nfs* x, nfs* y);   /* g2p_en normalize_numbers: x -> x, y scratch */
int nf_to_phonemes(nano_front_ctx* c, const u32* s, int n, int* ids, int* widx, int cap);

/* number words (ASCII); digits: decimal digit string */
void nf_cardinal(nano_front_ctx* c, const char* digits, nfa* out);
void nf_ordinal(nano_front_ctx* c, const char* digits, nfa* out);
void nf_cardinal_float(nano_front_ctx* c, double v, nfa* out);
void nf_py_int(nano_front_ctx* c, const u32* s, int n, nfa* out);   /* str(int(s)) with Python's ValueError */
double nf_strtod(const char* a);

/* G2P of normalised text: token stream of symbol ids, -1 between words */
int nf_g2p(nano_front_ctx* c, const u32* text, int n, int* out, int cap);
int nf_in_cmu(const nano_front_ctx* c, const char* w);
int nf_gru_predict(nano_front_ctx* c, const char* word, int* out);   /* raw GRU phoneme indices (<= 20) */
int nf_gru_symbol(const nano_front_ctx* c, int raw);                 /* GRU index -> symbol id or -1 */

int nf_event_symbol(const char* lower_name, int n);
int nf_count_words(nano_front_ctx* c, const char* utf8, size_t len);   /* "[name]" symbol id or -1 */

#define NF_SP 1
#define NF_FIRST_PHONE 9

#endif
