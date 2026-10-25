/* SteerTTS text front end, internal declarations (C11 port of native/src/front, itself a bit-exact port of the
 * Python edge front end: steertts_edge/text.py, markup.py, edge_front/ modules).
 *
 * Error handling: every call that can raise in Python takes a per-call context `fx`. A Python exception is recorded
 * once with st_seterr("ValueError: ...") and x->err = 1; code checks x->err after calls that can fail and unwinds.
 * Allocation failures are reported the same way. Strings are UTF-32 code point arrays (st_u32s) inside the front
 * end, UTF-8 at the API. */
#ifndef ST_FRONT_INT_H
#define ST_FRONT_INT_H
#include <stddef.h>
#include <stdint.h>

#include "st_base.h"
#include "st_front.h"

typedef uint32_t u32;

typedef struct {
  int err;
} fx;
void fx_fail(fx* x, const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

/* ------------------------------------------------------------------ growable buffers (malloc-backed) */
typedef struct {
  u32* p;
  int n, cap;
} st_u32s;
typedef struct {
  char* p;  /* NUL-terminated once anything was added (use sb_cstr for a possibly empty buffer) */
  int n, cap;
} st_sb;

int u32s_reserve(fx* x, st_u32s* b, int n);
void u32s_put(fx* x, st_u32s* b, const u32* s, int n);
void u32s_push(fx* x, st_u32s* b, u32 c);
void u32s_ascii(fx* x, st_u32s* b, const char* s);   /* append an ascii string */
void u32s_free(st_u32s* b);
void u32s_move(st_u32s* dst, st_u32s* src);           /* free dst, take src's storage, reset src */
int u32s_eq(const u32* a, int na, const u32* b, int nb);
int u32_find(const u32* s, int n, u32 c);              /* index or -1 */

void sb_put(fx* x, st_sb* b, const char* s, int n);
void sb_cat(fx* x, st_sb* b, const char* s);
void sb_ch(fx* x, st_sb* b, char c);
void sb_fmt(fx* x, st_sb* b, const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;
const char* sb_cstr(const st_sb* b);
void sb_free(st_sb* b);
void sb_from_u32(fx* x, st_sb* b, const u32* s, int n);     /* ascii code points -> bytes (truncating) */
void sb_utf8(fx* x, st_sb* b, const u32* s, int n);         /* UTF-8 encode */
void u32s_utf8(fx* x, st_u32s* b, const char* s, size_t n);  /* lenient UTF-8 decode (invalid -> U+FFFD) */

/* ------------------------------------------------------------------ Unicode tables (unicode.bin) */
typedef struct {
  u32 c;
  char* s;
  int n, idx;   /* idx: file order (load-time sort key) */
} ucd_str;
typedef struct {
  u32 c;
  u32* s;
  int n, idx;
} ucd_u32;
typedef struct {
  u32 (*word)[2];
  int nword;
  u32* space;
  int nspace;
  u32 (*digit)[3];  /* lo, hi, value of lo */
  int ndigit;
  char* ascii_g2p[128];
  int ascii_g2p_n[128];
  ucd_str* g2p;     /* sorted by code point */
  int ng2p;
  u32 (*fold)[2];
  int nfold;
  ucd_u32* lower;   /* sorted by code point */
  int nlower;
} st_ucd;

int ucd_load(st_ucd* u, const char* path);
void ucd_free(st_ucd* u);
int ucd_is_word(const st_ucd* u, u32 c);    /* re \w */
int ucd_is_space(const st_ucd* u, u32 c);   /* re \s, str.isspace() */
int ucd_digit(const st_ucd* u, u32 c);      /* decimal digit value, -1 */
u32 ucd_fold(const st_ucd* u, u32 c);       /* re.IGNORECASE folding */
void ucd_lower_small(fx* x, const st_ucd* u, const u32* s, int n, st_u32s* out);
const char* ucd_g2p_clean(const st_ucd* u, u32 c, int* len);

/* ------------------------------------------------------------------ regex (Python sre semantics, see st_front_re.c) */
#define ST_RX_MAXG 9
typedef struct st_rx st_rx;
typedef struct {
  int start, end;
  int caps[2 * (ST_RX_MAXG + 1)];  /* 2 per group, group 0 = whole match; -1 = did not participate */
} st_rxm;
typedef struct {
  int a, b;
} st_span;
typedef struct {
  st_span* p;
  int n, cap;
} st_spans;

st_rx* st_rx_compile(const st_ucd* u, const char* utf8_pattern, int icase);   /* NULL + st_seterr */
void st_rx_free(st_rx* r);
int st_rx_groups(const st_rx* r);
int st_rx_match_at(fx* x, const st_rx* r, const u32* s, int n, int pos, st_rxm* m);   /* re.match at pos */
int st_rx_search(fx* x, const st_rx* r, const u32* s, int n, int pos, st_rxm* m);     /* re.search from pos */
/* finditer step: *pos starts at 0; returns 1 with the next match (and advances *pos), 0 at the end / on error. */
int st_rx_next(fx* x, const st_rx* r, const u32* s, int n, int* pos, st_rxm* m);
/* re.sub(pattern, f, s): f appends the replacement for match m of s to out (it may call fx_fail). */
typedef void (*st_rx_repl)(fx* x, void* ud, const st_rxm* m, const u32* s, st_u32s* out);
void st_rx_sub(fx* x, const st_rx* r, const u32* s, int n, st_rx_repl f, void* ud, st_u32s* out);
/* re.split with groups included (None -> empty span), as spans into s */
void st_rx_split(fx* x, const st_rx* r, const u32* s, int n, st_spans* out);
static inline int rxm_has(const st_rxm* m, int g) { return m->caps[2 * g] >= 0; }
static inline int rxm_len(const st_rxm* m, int g) { return rxm_has(m, g) ? m->caps[2 * g + 1] - m->caps[2 * g] : 0; }
static inline const u32* rxm_ptr(const st_rxm* m, const u32* s, int g) { return s + (rxm_has(m, g) ? m->caps[2 * g] : 0); }
void rxm_put(fx* x, st_u32s* out, const st_rxm* m, const u32* s, int g);   /* append group g */

/* ------------------------------------------------------------------ numwords (num2words / inflect / normnum) */
typedef struct {
  const st_ucd* u;
  st_rx *ord_suffix, *pair, *last_digit, *cl1, *cl2, *cl3, *ws, *final_one, *last_word;
  st_rx *comma_number, *pounds, *dollars, *decimal, *ordinal_re, *number;
} st_numwords;
int numwords_init(st_numwords* nw, const st_ucd* u);
void numwords_free(st_numwords* nw);
/* Python int(s) for (Unicode) decimal digits -> ascii digits without leading zeros; ValueError like int() */
void py_int_digits(fx* x, const st_ucd* u, const u32* s, int n, st_sb* out);
void nw_cardinal(fx* x, const char* digits, st_sb* out);   /* num2words(n), appends */
void nw_ordinal(fx* x, const char* digits, st_sb* out);    /* num2words(n, to="ordinal") */
void nw_cardinal_float(fx* x, double v, st_sb* out);        /* num2words(<float>) */
void nw_normalize_numbers(fx* x, const st_numwords* nw, const u32* s, int n, st_u32s* out);
double st_front_strtod(const char* ascii);                  /* locale-independent strtod */

/* ------------------------------------------------------------------ G2P (edge_front) */
#define TOK_SPACE (-1)
typedef struct {
  char* buf;          /* the whole file */
  size_t len;
  uint32_t n;
  const char* wo;     /* u32 le [n+1] word offsets */
  const char* po;     /* u32 le [n+1] phone offsets */
  const char* words;
  const uint8_t* phones;
} st_lexicon;

typedef struct {
  int H, NG, NP;
  float *enc_x, *dec_x, *enc_s, *enc_b, *dec_s, *dec_b, *fc_s, *fc_b;
  int8_t *enc_q, *dec_q, *fc_q;
  int* phon2sym;
} st_gru;

typedef struct {
  char* w;
  int* p1;
  int n1;
  int* p2;
  int n2;
  char* pos;
} st_hom;
typedef struct {
  char* k;
  int v;
  char* s;
  int idx;   /* file order (load-time sort key) */
} st_kv;
typedef struct {
  st_hom* hom;
  int nhom;
  st_kv* w;
  int nw;
  st_kv* td;
  int ntd;
  int has_td;
} st_hetero;

typedef struct st_g2p_cache st_g2p_cache;
typedef struct {
  const st_ucd* u;
  st_numwords nw;
  st_lexicon cmu, oov;
  st_gru gru;
  st_hetero het;
  st_rx *hang, *word;
  int punct_id[128];
  st_g2p_cache* cache;
} st_g2p;

int g2p_init(st_g2p* g, const char* dir, const st_ucd* u, const st_front* f);
void g2p_free(st_g2p* g);
/* g2p_en.G2p.__call__ token stream: symbol ids and TOK_SPACE between words */
void g2p_call(fx* x, const st_g2p* g, const u32* text, int n, int** out, int* nout);
int g2p_in_cmu(const st_g2p* g, const char* w);
/* GRU greedy decode, raw phoneme indices (out has room for 20); returns the count. margin may be NULL. */
int gru_predict_raw(const st_gru* g, const char* word, int* out, double* margin);

/* ------------------------------------------------------------------ the front end object */
#define ST_N_SYMBOLS 118
#define ST_SYM_SP 1
#define ST_FIRST_PHONE 9
struct st_front {
  st_ucd ucd;
  char* symbols[ST_N_SYMBOLS];
  int sym_sorted[ST_N_SYMBOLS];   /* symbol ids sorted by name (bsearch) */
  st_g2p g2p;
  st_rx *quotes, *range, *by, *compound, *currency, *percent, *ordinal, *comma_num, *decimal, *long_digits, *camel_acr, *acronym,
      *ws, *event;
  st_rx* units[15];
  const char* unit_words[15];
  st_rx *emph, *token, *sent_end;   /* markup */
};

void front_normalize(fx* x, const st_front* f, const u32* s, int n, st_u32s* out);
int front_to_phonemes(fx* x, const st_front* f, const u32* s, int n, st_phonemes* out);   /* 0 / -1 */
void front_strip(const st_front* f, const u32* s, int n, int* a, int* b);                  /* str.strip() */
void front_collapse_ws(fx* x, const st_front* f, const u32* s, int n, st_u32s* out);        /* re.sub(\s+, " ") */

#endif
