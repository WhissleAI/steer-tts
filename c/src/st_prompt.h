/* Free-text prompt mapper of SteerTTS v1 (steertts/prompt.py): all-MiniLM-L6-v2 sentence embedding (WordPiece,
 * 6-layer BERT, mean pooling, L2 normalisation) + the prompt Adapter -> style-tag distribution and, per control,
 * a "specified" probability and a z value; rounded exactly as PromptMapper.__call__ does. */
#ifndef ST_PROMPT_H
#define ST_PROMPT_H
#include "st_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ST_MINILM_DIM 384
#define ST_PROMPT_NCTRL 12   /* len(CONTROL_DIMS) */

typedef struct st_prompt st_prompt;
/* minilm / adapter: the two weight files (fp32 or the packed int8 layout; not owned, must outlive the st_prompt);
 * vocab: minilm_vocab.txt; tags: the adapter's tag list (config.json prompt_adapter.tags, copied). NULL + st_seterr. */
st_prompt* st_prompt_load(st_weights* minilm, const char* vocab_path, st_weights* adapter, const char* const* tags,
                          int n_tags);
void st_prompt_free(st_prompt* p);
int st_prompt_n_tags(const st_prompt* p);
const char* st_prompt_tag(const st_prompt* p, int i);   /* adapter tag name, NULL if out of range */

/* UTF-8 text -> L2-normalised sentence embedding (inputs truncated at 256 word pieces incl. [CLS]/[SEP]).
 * pool may be NULL. Thread-safe on a shared st_prompt. 0 / -1. */
int st_minilm_embed(st_prompt* p, st_pool* pool, const char* text, float out[ST_MINILM_DIM]);

typedef struct {
  int n_top; int top_tag[3]; float top_p[3];     /* PromptMapper top_tags: tag index into the adapter's list, p rounded
                                                    to 3 decimals, by decreasing probability */
  int n_ctrl; int ctrl_dim[12]; float ctrl_z[12];  /* dims with sigmoid(spec) > 0.5, z rounded to 2 decimals, in
                                                      CONTROL_DIMS order */
  int top_p_milli[3];                              /* the same rounded values as exact integers: p = milli / 1000, */
  int ctrl_z_centi[12];                            /* z = centi / 100 (top_p / ctrl_z print exactly with %.3f / %.2f) */
} st_prompt_result;
/* Python: SteerTTS.prompt_mapper(prompt) (no result cache: the tokenizer is uncased and whitespace-insensitive, so
 * the cache only changes results for exotic Unicode lowercasing). Thread-safe. 0 / -1. */
int st_prompt_map(st_prompt* p, st_pool* pool, const char* prompt, st_prompt_result* out);

/* round(x, ndigits) of Python for ndigits >= 0 (correctly rounded on the exact binary value, ties to even), as an
 * integer: returns round(x * 10^ndigits) exactly. |x| * 10^ndigits must be < 2^52. */
long long st_pyround_scaled(double x, int ndigits);

#ifdef __cplusplus
}
#endif
#endif
