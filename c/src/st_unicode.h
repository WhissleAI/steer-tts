/* Unicode tables of the MiniLM WordPiece tokenizer (generated: src/st_unicode_tab.c by tools/gen_unicode.py, probed
 * from the HF `tokenizers` BertNormalizer / BertPreTokenizer the model runs with). */
#ifndef ST_UNICODE_H
#define ST_UNICODE_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ST_UNI_MAXOUT 4
/* BertNormalizer (clean text, CJK spacing, NFD + drop Mn, lowercase) of one code point: writes the result to out and
 * returns its length (0 = removed). NFD's canonical reordering across code points is st_uni_reorder_class's job. */
int st_uni_norm(uint32_t c, uint32_t out[ST_UNI_MAXOUT]);
/* 0: blocks reordering; 1..254: combining class of a kept mark NFD reorders; 255: removed, transparent. */
int st_uni_reorder_class(uint32_t c);
int st_uni_is_punct(uint32_t c);   /* BERT punctuation (applied to normalized text) */
int st_uni_is_space(uint32_t c);   /* pre-tokenizer whitespace */

#ifdef __cplusplus
}
#endif
#endif
