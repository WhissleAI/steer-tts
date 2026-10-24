/* BERT WordPiece tokenizer of all-MiniLM-L6-v2 (uncased), equal to the HF tokenizer the model runs with:
 * special tokens [PAD] [UNK] [CLS] [SEP] [MASK] written literally in the text are kept as such; every other span is
 * normalized (clean text, CJK spacing, NFD + drop Mn, lowercase; src/st_unicode.h), split on whitespace and BERT
 * punctuation and cut greedily longest-match-first into vocabulary pieces ("##" continuation, a word over 100 code
 * points or with no cover -> [UNK]); the result is wrapped in [CLS] ... [SEP] and truncated to max_len ids. */
#ifndef ST_WORDPIECE_H
#define ST_WORDPIECE_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_wordpiece st_wordpiece;
/* vocab.txt: one token per line, id = line number. NULL + st_seterr on failure. */
st_wordpiece* st_wordpiece_load(const char* vocab_path);
void st_wordpiece_free(st_wordpiece* wp);
int st_wordpiece_vocab_size(const st_wordpiece* wp);
/* id of a token (exact bytes), -1 if absent */
int st_wordpiece_id(const st_wordpiece* wp, const char* tok, size_t len);
/* UTF-8 text -> ids [CLS] ... [SEP], at most max_len (>= 2) of them (the word pieces are truncated to max_len - 2,
 * as truncation=True, max_length=max_len does). Invalid UTF-8 bytes decode to U+FFFD, which the normalizer drops.
 * Returns the number of ids, -1 (+ st_seterr) on allocation failure. Thread-safe on a shared st_wordpiece. */
int st_wordpiece_encode(const st_wordpiece* wp, const char* text, int32_t* ids, int max_len);

#ifdef __cplusplus
}
#endif
#endif
