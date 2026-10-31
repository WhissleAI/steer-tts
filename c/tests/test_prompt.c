/* Prompt mapper test against tests/golden/prompt (tests/make_golden_prompt.py): WordPiece ids on a varied corpus,
 * MiniLM embeddings and the full PromptMapper results for ~200 prompts in fp32 and int8, plus speed.
 *
 *   test_prompt [golden_dir [extra_tokens.txt]]     weights: $STEERTTS_V1_MODEL/native (77 = skipped if missing) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/st_base.h"
#include "../src/st_prompt.h"
#include "../src/st_wordpiece.h"

#define DEFAULT_V1 "../models/steer-tts"

static int file_exists(const char* p) {
  FILE* f = fopen(p, "rb");
  if (f) fclose(f);
  return f != NULL;
}

/* hex -> bytes in place-free buffer (NUL bytes dropped: a C string cannot carry them and the tokenizer removes them
 * transparently). Returns 1 if a NUL was dropped. */
static int unhex(const char* h, size_t n, char* out) {
  int nul = 0;
  size_t m = 0;
  for (size_t i = 0; i + 1 < n; i += 2) {
    unsigned v;
    sscanf(h + i, "%2x", &v);
    if (v) out[m++] = (char)v;
    else nul = 1;
  }
  out[m] = 0;
  return nul;
}

static char** read_lines(const char* path, int* n) {
  char* data;
  size_t len;
  *n = 0;
  if (st_read_file(path, &data, &len) != 0) return NULL;
  int cap = 1;
  for (size_t i = 0; i < len; ++i) cap += data[i] == '\n';
  char** lines = (char**)calloc((size_t)cap + 1, sizeof(char*));
  char* s = data;
  while (*s) {
    char* e = strchr(s, '\n');
    size_t L = e ? (size_t)(e - s) : strlen(s);
    lines[*n] = (char*)malloc(L + 1);
    memcpy(lines[*n], s, L);
    lines[(*n)++][L] = 0;
    s += L + (e ? 1 : 0);
  }
  st_free(data);
  return lines;
}

static void free_lines(char** l, int n) {
  for (int i = 0; i < n; ++i) free(l[i]);
  free(l);
}

static int test_tokens(const st_wordpiece* wp, const char* path, int verbose_fails) {
  int n, bad = 0, nul = 0;
  char** lines = read_lines(path, &n);
  if (!lines) {
    printf("tokens: cannot read %s: %s\n", path, st_geterr());
    return -1;
  }
  double t0 = st_time_ms();
  long ntok = 0;
  for (int i = 0; i < n; ++i) {
    char* tab = strchr(lines[i], '\t');
    if (!tab) continue;
    char* text = (char*)malloc((size_t)(tab - lines[i]) / 2 + 1);
    nul += unhex(lines[i], (size_t)(tab - lines[i]), text);
    int32_t want[256], got[256];
    int nw = 0;
    for (char* s = tab + 1; *s && nw < 256;) {
      want[nw++] = (int32_t)strtol(s, &s, 10);
      if (*s == ',') ++s;
    }
    int ng = st_wordpiece_encode(wp, text, got, 256);
    ntok += ng;
    if (ng != nw || memcmp(got, want, sizeof(int32_t) * (size_t)nw) != 0) {
      if (bad++ < verbose_fails) {
        printf("  token mismatch line %d (n %d vs %d) text hex %.*s\n   got ", i + 1, ng, nw, (int)(tab - lines[i]) > 200 ? 200 : (int)(tab - lines[i]), lines[i]);
        for (int j = 0; j < ng && j < 40; ++j) printf("%d ", got[j]);
        printf("\n  want ");
        for (int j = 0; j < nw && j < 40; ++j) printf("%d ", want[j]);
        printf("\n");
      }
    }
    free(text);
  }
  double ms = st_time_ms() - t0;
  printf("tokens %s: %d / %d texts identical (%d with NUL bytes dropped), %ld ids, %.1f us/text\n", path, n - bad, n, nul,
         ntok, 1000.0 * ms / (n ? n : 1));
  free_lines(lines, n);
  return bad;
}

/* distance of x * 10^nd to the nearest rounding boundary (k + 0.5), in units of 10^-nd */
static double tie_dist(double x, int nd) {
  double y = x * pow(10, nd);
  return fabs(y - floor(y) - 0.5);
}

typedef struct {
  int ntop, top[3], pm[3], nctrl, dim[12], zc[12];
  double raw_p[3], raw_sp[12], raw_v[12];
} gold_map;

static void parse_map(const char* line, gold_map* g) {
  memset(g, 0, sizeof *g);
  const char* s = line;
  while (*s && *s != '\t') {
    g->top[g->ntop] = (int)strtol(s, (char**)&s, 10);
    ++s;
    g->pm[g->ntop++] = (int)strtol(s, (char**)&s, 10);
    if (*s == ';') ++s;
  }
  ++s;
  while (*s && *s != '\t') {
    g->dim[g->nctrl] = (int)strtol(s, (char**)&s, 10);
    ++s;
    g->zc[g->nctrl++] = (int)strtol(s, (char**)&s, 10);
    if (*s == ';') ++s;
  }
  ++s;
  for (int i = 0; i < 3; ++i) g->raw_p[i] = strtod(s, (char**)&s);
  s = strchr(s, '|') + 1;
  for (int i = 0; i < 12; ++i) g->raw_sp[i] = strtod(s, (char**)&s);
  s = strchr(s, '|') + 1;
  for (int i = 0; i < 12; ++i) g->raw_v[i] = strtod(s, (char**)&s);
}

static int same_map(const st_prompt_result* r, const gold_map* g) {
  if (r->n_top != g->ntop || r->n_ctrl != g->nctrl) return 0;
  for (int i = 0; i < g->ntop; ++i)
    if (r->top_tag[i] != g->top[i] || r->top_p_milli[i] != g->pm[i]) return 0;
  for (int i = 0; i < g->nctrl; ++i)
    if (r->ctrl_dim[i] != g->dim[i] || r->ctrl_z_centi[i] != g->zc[i]) return 0;
  return 1;
}

static void print_map_diff(int i, const st_prompt_result* r, const gold_map* g) {
  printf("  map mismatch prompt %d:\n    C   top", i);
  for (int j = 0; j < r->n_top; ++j) printf(" %d:%d", r->top_tag[j], r->top_p_milli[j]);
  printf("  ctrl");
  for (int j = 0; j < r->n_ctrl; ++j) printf(" %d:%d", r->ctrl_dim[j], r->ctrl_z_centi[j]);
  printf("\n    py  top");
  for (int j = 0; j < g->ntop; ++j) printf(" %d:%d(raw %.9g, tie dist %.2g)", g->top[j], g->pm[j], g->raw_p[j], tie_dist(g->raw_p[j], 3));
  printf("  ctrl");
  for (int j = 0; j < g->nctrl; ++j)
    printf(" %d:%d(raw %.9g, tie dist %.2g)", g->dim[j], g->zc[j], g->raw_v[g->dim[j]], tie_dist(g->raw_v[g->dim[j]], 2));
  printf("\n    py spec probs:");
  for (int j = 0; j < 12; ++j) printf(" %.6f", g->raw_sp[j]);
  printf("\n");
}

static const char* const TAGS[] = {"none", "adoration", "amazement", "amusement", "anger", "anxiety", "apologetic",
  "bored", "confidence", "confusion", "contempt", "contentment", "cuteness", "desire", "determined", "disappointment",
  "disgust", "distress", "ecstasy", "embarrassment", "emphasis", "enunciated", "excited", "fast", "fear", "frustration",
  "guilt", "happiness", "hesitant", "high_pitch", "interest", "irritation", "laughing", "loud", "low_pitch",
  "mysterious", "neutral", "nostalgia", "pain", "panic", "pride", "realization", "relief", "sadness", "sarcasm",
  "serenity", "skeptical", "slow", "sympathy", "tired", "whisper"};

/* the adapter tags from config.json (falls back to the v1 list above) */
static int load_tags(const char* v1, char*** out) {
  char path[4096], *data;
  size_t len;
  snprintf(path, sizeof path, "%s/config.json", v1);
  int n = (int)(sizeof TAGS / sizeof TAGS[0]);
  *out = NULL;
  if (st_read_file(path, &data, &len) != 0) return n;
  st_json* j = st_json_parse(data, len);
  st_free(data);
  const st_json* t = st_json_get(st_json_get(j, "prompt_adapter"), "tags");
  if (t && t->type == ST_JARR) {
    n = t->n;
    *out = (char**)calloc((size_t)n, sizeof(char*));
    for (int i = 0; i < n; ++i) (*out)[i] = st_strdup(st_json_str(st_json_at(t, i), ""));
  }
  st_json_free(j);
  return n;
}

int main(int argc, char** argv) {
  const char* gold = argc > 1 ? argv[1] : "golden/prompt";
  char path[4096];
  snprintf(path, sizeof path, "%s/prompts.txt", gold);
  if (argc <= 1 && !file_exists(path)) {
    gold = "tests/golden/prompt";
    snprintf(path, sizeof path, "%s/prompts.txt", gold);
  }
  const char* v1 = getenv("STEERTTS_V1_MODEL") ? getenv("STEERTTS_V1_MODEL") : DEFAULT_V1;
  char nat[2048];
  snprintf(nat, sizeof nat, "%s/native", v1);
  snprintf(path, sizeof path, "%s/minilm.safetensors", nat);
  if (!file_exists(path)) {
    printf("skip: %s missing (run tools/export_minilm.py)\n", path);
    return 77;
  }
  int fails = 0;
  snprintf(path, sizeof path, "%s/minilm_vocab.txt", nat);
  double t0 = st_time_ms();
  st_wordpiece* wp = st_wordpiece_load(path);
  if (!wp) {
    printf("vocab: %s\n", st_geterr());
    return 1;
  }
  printf("vocab: %d tokens, loaded in %.1f ms\n", st_wordpiece_vocab_size(wp), st_time_ms() - t0);
  char tp[4096];
  snprintf(tp, sizeof tp, "%s/tokens.txt", gold);
  int tb = test_tokens(wp, tp, 10);
  fails += tb != 0;
  if (argc > 2) fails += test_tokens(wp, argv[2], 10) != 0;
  { /* robustness: random bytes (invalid UTF-8 included) must encode to valid ids within max_len */
    unsigned seed = 7;
    int bad = 0;
    for (int it = 0; it < 20000; ++it) {
      char buf[600];
      int n = (int)((seed = seed * 1664525u + 1013904223u) >> 23) % 599;
      for (int i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        buf[i] = (char)(1 + (seed >> 24) % 255);
      }
      buf[n] = 0;
      int32_t ids[64];
      int max_len = 2 + it % 62, ng = st_wordpiece_encode(wp, buf, ids, max_len);
      if (ng < 2 || ng > max_len || ids[0] != st_wordpiece_id(wp, "[CLS]", 5) || ids[ng - 1] != st_wordpiece_id(wp, "[SEP]", 5)) ++bad;
      for (int j = 0; j < ng; ++j) bad += ids[j] < 0 || ids[j] >= st_wordpiece_vocab_size(wp);
    }
    printf("tokens random bytes: %s\n", bad ? "FAIL" : "ok");
    fails += bad != 0;
  }
  st_wordpiece_free(wp);

  /* rounding helper sanity (Python: round(0.125, 2) == 0.12, round(0.135, 2) == 0.14 (binary above), round(2.675, 2) == 2.67) */
  if (st_pyround_scaled(0.125, 2) != 12 || st_pyround_scaled(0.135, 2) != 14 || st_pyround_scaled(2.675, 2) != 267 ||
      st_pyround_scaled(-0.5, 0) != 0 || st_pyround_scaled(1.5, 0) != 2 || st_pyround_scaled(-2.5, 0) != -2 ||
      st_pyround_scaled(0.0005, 3) != 1 || st_pyround_scaled(-1.005, 2) != -100) {
    printf("FAIL: st_pyround_scaled\n");
    ++fails;
  }

  int np;
  snprintf(path, sizeof path, "%s/prompts.txt", gold);
  char** ph = read_lines(path, &np);
  if (!ph) {
    printf("prompts: %s\n", st_geterr());
    return 1;
  }
  char** prompts = (char**)calloc((size_t)np, sizeof(char*));
  for (int i = 0; i < np; ++i) {
    prompts[i] = (char*)malloc(strlen(ph[i]) / 2 + 1);
    unhex(ph[i], strlen(ph[i]), prompts[i]);
  }
  char** tags;
  int n_tags = load_tags(v1, &tags);
  const char* const* tagp = tags ? (const char* const*)tags : TAGS;
  st_pool* pool4 = st_pool_create(4);

  for (int prec = 0; prec < 2; ++prec) {
    const char* pn = prec ? "int8" : "fp32";
    char mp[4096], ap[4096], vp[4096];
    snprintf(mp, sizeof mp, prec ? "%s/int8/minilm.safetensors" : "%s/minilm.safetensors", nat);
    snprintf(ap, sizeof ap, prec ? "%s/int8/prompt_adapter.safetensors" : "%s/prompt_adapter.safetensors", v1);
    snprintf(vp, sizeof vp, "%s/minilm_vocab.txt", nat);
    t0 = st_time_ms();
    st_weights* mw = st_weights_open(mp);
    st_weights* aw = mw ? st_weights_open(ap) : NULL;
    st_prompt* P = aw ? st_prompt_load(mw, vp, aw, tagp, n_tags) : NULL;
    if (!P) {
      printf("%s: load failed: %s\n", pn, st_geterr());
      return 1;
    }
    double load_ms = st_time_ms() - t0;
    /* embeddings */
    snprintf(path, sizeof path, "%s/emb_%s.f32", gold, pn);
    char* eg;
    size_t elen;
    if (st_read_file(path, &eg, &elen) != 0 || elen != sizeof(float) * 384 * (size_t)np) {
      printf("%s: bad %s\n", pn, path);
      return 1;
    }
    const float* E = (const float*)eg;
    double min_cos = 1, max_abs = 0;
    float e[384];
    for (int i = 0; i < np; ++i) {
      if (st_minilm_embed(P, pool4, prompts[i], e) != 0) {
        printf("embed: %s\n", st_geterr());
        return 1;
      }
      double d = 0, na = 0, nb = 0;
      for (int c = 0; c < 384; ++c) {
        double a = e[c], b = E[(size_t)i * 384 + c];
        d += a * b;
        na += a * a;
        nb += b * b;
        max_abs = fabs(a - b) > max_abs ? fabs(a - b) : max_abs;
      }
      double cs = d / sqrt(na * nb);
      min_cos = cs < min_cos ? cs : min_cos;
    }
    st_free(eg);
    int emb_ok = min_cos >= 0.99999;
    printf("%s: embeddings %d prompts: min cos %.8f, max abs diff %.3g -> %s\n", pn, np, min_cos, max_abs,
           emb_ok ? "ok" : "FAIL");
    fails += !emb_ok;
    /* mapping */
    snprintf(path, sizeof path, "%s/map_%s.txt", gold, pn);
    int nm;
    char** ml = read_lines(path, &nm);
    if (!ml || nm != np) {
      printf("%s: bad %s\n", pn, path);
      return 1;
    }
    int exact = 0, top1 = 0, near = 0;
    for (int i = 0; i < np; ++i) {
      st_prompt_result r;
      gold_map g;
      parse_map(ml[i], &g);
      if (st_prompt_map(P, pool4, prompts[i], &r) != 0) {
        printf("map: %s\n", st_geterr());
        return 1;
      }
      top1 += r.top_tag[0] == g.top[0];
      if (same_map(&r, &g)) {
        ++exact;
        continue;
      }
      /* a near tie: a probability / z within 1e-3 rounding units of a boundary, or a spec prob within 1e-4 of 0.5 */
      int is_near = 0;
      for (int j = 0; j < 3; ++j) is_near |= tie_dist(g.raw_p[j], 3) < 1e-3;
      for (int j = 0; j < 12; ++j) is_near |= fabs(g.raw_sp[j] - 0.5) < 1e-4 || tie_dist(g.raw_v[j], 2) < 1e-3;
      near += is_near;
      print_map_diff(i, &r, &g);
    }
    printf("%s: mapping %d / %d exact after rounding (top-1 tag %d / %d), %d mismatches, %d of them near ties\n", pn, exact,
           np, top1, np, np - exact, near);
    fails += exact != np && (np - exact) != near;   /* only near ties are tolerated */
    free_lines(ml, nm);
    /* speed */
    for (int th = 0; th < 2; ++th) {
      st_pool* pl = th ? pool4 : NULL;
      st_prompt_result r;
      double s = st_time_ms();
      int reps = 0;
      for (int i = 0; i < np; ++i) {
        if (strlen(prompts[i]) > 200) continue;   /* typical prompts only */
        st_prompt_map(P, pl, prompts[i], &r);
        ++reps;
      }
      printf("%s: %.2f ms / prompt (%d threads, %d short prompts)\n", pn, (st_time_ms() - s) / reps, th ? 4 : 1, reps);
    }
    {
      const char* longp = NULL;
      for (int i = 0; i < np; ++i)
        if (strlen(prompts[i]) > 1000) longp = prompts[i];
      if (longp) {
        st_prompt_result r;
        for (int th = 0; th < 2; ++th) {
          double s = st_time_ms();
          for (int k = 0; k < 5; ++k) st_prompt_map(P, th ? pool4 : NULL, longp, &r);
          printf("%s: %.2f ms / 256-token prompt (%d threads)\n", pn, (st_time_ms() - s) / 5, th ? 4 : 1);
        }
      }
    }
    printf("%s: load %.1f ms, peak RSS %zu MB\n", pn, load_ms, st_peak_rss_kb() / 1024);
    st_prompt_free(P);
    st_weights_close(aw);
    st_weights_close(mw);
  }
  st_pool_destroy(pool4);
  if (tags) {
    for (int i = 0; i < n_tags; ++i) st_free(tags[i]);
    free(tags);
  }
  for (int i = 0; i < np; ++i) free(prompts[i]);
  free(prompts);
  free_lines(ph, np);
  printf(fails ? "FAILED (%d)\n" : "all ok\n", fails);
  return fails ? 1 : 0;
}
