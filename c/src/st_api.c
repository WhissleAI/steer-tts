/* The public API (include/steertts.h): model loading, requests, rendering, streaming.
 *
 * Mirrors the Python packages: steertts.SteerTTS (v1: resolve / _render / markup stream) and steertts_edge.EdgeTTS
 * (edge: the same path without flow, prompts and cloning; text with nothing to say renders no audio). The request
 * checks and their messages follow steertts/request.py (normalize_request) and EdgeTTS._check. */
#ifndef STEERTTS_BUILD
#define STEERTTS_BUILD
#endif
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/steertts.h"
#include "front/st_front.h"
#include "st_acoustic.h"
#include "st_audio.h"
#include "st_base.h"
#include "st_dsp.h"
#include "st_flow.h"
#include "st_mel.h"
#include "st_prompt.h"
#include "st_vocoder.h"
#include "st_wavlm.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define N_CTRL 12
#define PITCH_SCALE 3.0
#define SR 24000

static const char* const k_controls[N_CTRL] = {"rate", "loud", "f0_med", "f0_iqr", "f0_end", "pause",
                                               "tilt", "voiced", "flat", "centroid", "cpps", "h1h2"};

/* ------------------------------------------------------------------ small helpers */
typedef struct {
  char* s;
  size_t n, cap;
  int oom;
} sbuf;

static void sb_put(sbuf* b, const char* s, size_t n) {
  if (b->oom) return;
  if (b->n + n + 1 > b->cap) {
    size_t c = b->cap ? b->cap * 2 : 256;
    while (c < b->n + n + 1) c *= 2;
    char* t = (char*)realloc(b->s, c);
    if (!t) {
      b->oom = 1;
      return;
    }
    b->s = t;
    b->cap = c;
  }
  memcpy(b->s + b->n, s, n);
  b->n += n;
  b->s[b->n] = 0;
}

static void sb_str(sbuf* b, const char* s) { sb_put(b, s, strlen(s)); }

static void sb_fmt(sbuf* b, const char* fmt, ...) {
  char tmp[512];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
  va_end(ap);
  if (n < 0) return;
  if ((size_t)n < sizeof tmp) {
    sb_put(b, tmp, (size_t)n);
    return;
  }
  char* big = (char*)malloc((size_t)n + 1);
  if (!big) {
    b->oom = 1;
    return;
  }
  va_start(ap, fmt);
  vsnprintf(big, (size_t)n + 1, fmt, ap);
  va_end(ap);
  sb_put(b, big, (size_t)n);
  free(big);
}

static void sb_json_str(sbuf* b, const char* s) {
  sb_put(b, "\"", 1);
  for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
    if (*p == '"' || *p == '\\') {
      char e[2] = {'\\', (char)*p};
      sb_put(b, e, 2);
    } else if (*p < 0x20) {
      sb_fmt(b, "\\u%04x", *p);
    } else
      sb_put(b, (const char*)p, 1);
  }
  sb_put(b, "\"", 1);
}

/* a copy that survives the buffer (st_malloc'd) */
static char* sb_take(sbuf* b) {
  if (b->oom) {
    free(b->s);
    st_seterr("out of memory");
    return NULL;
  }
  char* r = st_strdup(b->s ? b->s : "");
  free(b->s);
  memset(b, 0, sizeof *b);
  return r;
}

/* Python's repr() of a float (shortest round-trip digits, scientific iff exp < -4 or exp >= 16) */
static void py_float(char* out, size_t cap, double v, int is_int) {
  if (is_int && fabs(v) < 9.2e18) {
    snprintf(out, cap, "%lld", (long long)v);
    return;
  }
  if (isnan(v)) {
    snprintf(out, cap, "nan");
    return;
  }
  if (isinf(v)) {
    snprintf(out, cap, v > 0 ? "inf" : "-inf");
    return;
  }
  char e[64];
  int p;
  for (p = 0; p < 17; ++p) {
    snprintf(e, sizeof e, "%.*e", p, v);
    if (strtod(e, NULL) == v) break;
  }
  /* e = [-]d.ddde[+-]XX */
  char* ep = strchr(e, 'e');
  int ex = atoi(ep + 1);
  *ep = 0;
  char digits[40];
  int nd = 0, neg = e[0] == '-';
  for (char* q = e; *q; ++q)
    if (isdigit((unsigned char)*q)) digits[nd++] = *q;
  digits[nd] = 0;
  while (nd > 1 && digits[nd - 1] == '0') digits[--nd] = 0;
  char buf[96];
  size_t k = 0;
  if (neg) buf[k++] = '-';
  if (ex < -4 || ex >= 16) {
    buf[k++] = digits[0];
    if (nd > 1) {
      buf[k++] = '.';
      for (int i = 1; i < nd; ++i) buf[k++] = digits[i];
    }
    k += (size_t)snprintf(buf + k, sizeof buf - k, "e%c%02d", ex < 0 ? '-' : '+', ex < 0 ? -ex : ex);
  } else if (ex < 0) {
    buf[k++] = '0';
    buf[k++] = '.';
    for (int i = 0; i < -ex - 1; ++i) buf[k++] = '0';
    for (int i = 0; i < nd; ++i) buf[k++] = digits[i];
  } else {
    for (int i = 0; i <= ex; ++i) buf[k++] = i < nd ? digits[i] : '0';
    buf[k++] = '.';
    if (nd > ex + 1)
      for (int i = ex + 1; i < nd; ++i) buf[k++] = digits[i];
    else
      buf[k++] = '0';
  }
  buf[k] = 0;
  snprintf(out, cap, "%s", buf);
}

/* the shortest decimal that reads back as the same float32 (0.1f -> "0.1"), Python float formatting */
static void float_short(char* out, size_t cap, float v) {
  for (int p = 1; p <= 9; ++p) {
    char e[48];
    snprintf(e, sizeof e, "%.*e", p - 1, (double)v);
    double d = strtod(e, NULL);   /* same locale as the snprintf above */
    if (!isfinite(v) || (float)d == v) {
      py_float(out, cap, d, 0);
      return;
    }
  }
  py_float(out, cap, (double)v, 0);
}

static size_t utf8_len(const char* s) {
  size_t n = 0;
  for (; *s; ++s)
    if (((unsigned char)*s & 0xC0) != 0x80) ++n;
  return n;
}

/* Python's str.strip() whitespace (str.isspace), on UTF-8 */
static int is_blank(const char* s) {
  const unsigned char* p = (const unsigned char*)s;
  while (*p) {
    unsigned c = *p, n = 1;
    if (c >= 0xC0 && c < 0xE0 && (p[1] & 0xC0) == 0x80) c = ((c & 0x1F) << 6) | (p[1] & 0x3F), n = 2;
    else if (c >= 0xE0 && c < 0xF0 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80)
      c = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F), n = 3;
    int ws = (c >= 9 && c <= 13) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
             (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
    if (!ws) return 0;
    p += n;
  }
  return 1;
}

static int ctrl_index(const char* name) {
  for (int i = 0; i < N_CTRL; ++i)
    if (!strcmp(name, k_controls[i])) return i;
  return -1;
}

static void join_controls(char* out, size_t cap) {
  size_t k = 0;
  out[0] = 0;
  for (int i = 0; i < N_CTRL; ++i) k += (size_t)snprintf(out + k, cap - k, "%s%s", i ? ", " : "", k_controls[i]);
}

/* ------------------------------------------------------------------ engine */
typedef struct {
  uint64_t key;    /* FNV-1a of the 24 kHz samples (0 = empty): a file rewritten at the same path is recomputed */
  float vec[512];
} xv_cache;

struct steertts_engine {
  int edge;
  char *dir, *assets, *precision;
  steertts_config cfg;
  st_pool* pool;
  /* voices / tags */
  int n_voices;
  char** voices;
  const float* voice_rows;   /* v1: [n, 512] speaker vectors; edge: [n, 64] conditioning slots */
  int voice_dim;
  st_weights* w_voices;
  char* default_voice;
  int n_tags;
  char** tags;
  /* models */
  st_front* front;
  st_weights *w_ac, *w_voc;
  st_acoustic* ac;
  st_vocoder* voc;
  st_weights *w_flow, *w_pflow, *w_minilm, *w_adapter, *w_wavlm;
  st_flow *flow, *pflow;
  int pflow_tried;
  st_prompt* prompt;
  char** ptags;
  int n_ptags;
  st_wavlm* wavlm;
  st_json* cfg_json;
  xv_cache xv;
};

static char* path_join(const char* a, const char* b) {
  size_t n = strlen(a) + strlen(b) + 2;
  char* p = (char*)st_malloc(n);
  if (p) snprintf(p, n, "%s/%s", a, b);
  return p;
}

static int file_exists(const char* p) {
  FILE* f = fopen(p, "rb");
  if (!f) return 0;
  fclose(f);
  return 1;
}

static st_json* load_json_file(const char* path) {
  char* d;
  size_t n;
  if (st_read_file(path, &d, &n)) return NULL;
  st_json* j = st_json_parse(d, n);
  if (!j) st_seterr("%s: %s", path, st_geterr());
  st_free(d);
  return j;
}

static st_weights* open_weights(const char* dir, const char* sub, const char* name) {
  char rel[512];
  snprintf(rel, sizeof rel, "%s%s%s", sub ? sub : "", sub ? "/" : "", name);
  char* p = path_join(dir, rel);
  if (!p) return NULL;
  st_weights* w = st_weights_open(p);
  st_free(p);
  return w;
}

static void free_list(char** l, int n);

static char** str_list(const st_json* arr, int* n) {
  if (!arr || arr->type != ST_JARR) {
    st_seterr("expected a list");
    return NULL;
  }
  char** out = (char**)st_calloc(sizeof(char*) * (size_t)(arr->n + 1));
  if (!out) return NULL;
  for (int i = 0; i < arr->n; ++i)
    if (!(out[i] = st_strdup(st_json_str(&arr->items[i], "")))) {
      free_list(out, i);
      return NULL;
    }
  *n = arr->n;
  return out;
}

static void free_list(char** l, int n) {
  if (!l) return;
  for (int i = 0; i < n; ++i) st_free(l[i]);
  st_free(l);
}

static int find_str(char* const* l, int n, const char* s) {
  for (int i = 0; i < n; ++i)
    if (!strcmp(l[i], s)) return i;
  return -1;
}

void steertts_default_config(steertts_config* c) {
  memset(c, 0, sizeof *c);
  c->threads = 1;
  c->precision = "fp32";
  c->flow = 1;
  c->flow_steps = 10;
  c->flow_temp = 0.4f;
  c->guidance = 1.0f;
}

static int n_cpus(void) {
#ifdef _WIN32
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (int)si.dwNumberOfProcessors;
#else
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (int)n : 1;
#endif
}

static int load_v1(steertts_engine* e) {
  const st_json* c = e->cfg_json;
  int int8 = !strcmp(e->precision, "int8");
  const char* sub = int8 ? st_json_str(st_json_get(st_json_get(st_json_get(c, "precisions"), "int8"), "dir"), "int8")
                         : NULL;
  /* tags from control_stats.json (the acoustic model's tag table) */
  char* p = path_join(e->dir, "control_stats.json");
  st_json* stats = p ? load_json_file(p) : NULL;
  st_free(p);
  if (!stats) return -1;
  e->tags = str_list(st_json_get(stats, "tags"), &e->n_tags);
  st_json_free(stats);
  if (!e->tags) return -1;
  e->voices = str_list(st_json_get(c, "voices_order"), &e->n_voices);
  if (!e->voices) return -1;
  e->default_voice = st_strdup(st_json_str(st_json_get(c, "default_voice"), "voice_1688"));
  const st_json* files = st_json_get(c, "files");
  e->w_voices = open_weights(e->dir, sub, st_json_str(st_json_get(files, "speaker_vectors"), "speaker_vectors.safetensors"));
  if (!e->w_voices) return -1;
  e->voice_rows = st_weights_f32(e->w_voices, "vec", (int64_t)e->n_voices * 512);
  if (!e->voice_rows) {
    st_seterr("speaker_vectors / voices_order mismatch (%s)", st_geterr());
    return -1;
  }
  e->voice_dim = 512;
  const st_json* ac = st_json_get(c, "acoustic");
  e->w_ac = open_weights(e->dir, sub, st_json_str(st_json_get(files, "acoustic"), "model.safetensors"));
  if (!e->w_ac || !(e->ac = st_acoustic_load(e->w_ac, st_json_get(ac, "arch")))) return -1;
  /* Vocos (converted by tools/export_vocos.py): assets/vocos.safetensors or assets/int8/vocos.safetensors */
  e->w_voc = open_weights(e->assets, int8 ? "int8" : NULL, "vocos.safetensors");
  if (!e->w_voc) {
    st_seterr("v1 needs the converted Vocos weights: %s (%s); see c/README.md", e->assets, st_geterr());
    return -1;
  }
  if (!(e->voc = st_vocoder_load(e->w_voc, 512, 1536, 8, 1024, 256))) return -1;
  if (e->cfg.flow) {
    const st_json* r = st_json_get(c, "refiner");
    const char* f = st_json_str(st_json_get(r, "file"), NULL);
    if (f) {
      e->w_flow = open_weights(e->dir, sub, f);
      if (!e->w_flow || !(e->flow = st_flow_load(e->w_flow, st_json_get(r, "cfg")))) return -1;
    }
  }
  return 0;
}

static int load_edge(steertts_engine* e) {
  const st_json* c = e->cfg_json;
  e->tags = str_list(st_json_get(c, "tags"), &e->n_tags);
  if (!e->tags) return -1;
  char* p = path_join(e->dir, "voices.json");
  st_json* vj = p ? load_json_file(p) : NULL;
  st_free(p);
  if (!vj) return -1;
  const st_json* vl = st_json_get(vj, "voices");
  e->n_voices = vl ? vl->n : 0;
  e->voices = (char**)st_calloc(sizeof(char*) * (size_t)(e->n_voices + 1));
  for (int i = 0; e->voices && i < e->n_voices; ++i)
    if (!(e->voices[i] = st_strdup(st_json_str(st_json_get(&vl->items[i], "name"), "")))) e->n_voices = i;
  if (!e->voices) {
    st_json_free(vj);
    return -1;
  }
  st_json_free(vj);
  e->default_voice = st_strdup(st_json_str(st_json_get(c, "default_voice"), e->n_voices ? e->voices[0] : ""));
  e->w_voices = open_weights(e->dir, NULL, "voices.safetensors");
  if (!e->w_voices) return -1;
  int64_t sh[8];
  if (st_weights_shape(e->w_voices, "s", sh) != 2 || sh[0] != e->n_voices) {
    st_seterr("voices.safetensors / voices.json mismatch");
    return -1;
  }
  e->voice_dim = (int)sh[1];
  e->voice_rows = st_weights_f32(e->w_voices, "s", 0);
  if (!e->voice_rows) return -1;
  const st_json* ac = st_json_get(c, "acoustic");
  e->w_ac = open_weights(e->dir, NULL, st_json_str(st_json_get(ac, "file"), "acoustic.safetensors"));
  if (!e->w_ac || !(e->ac = st_acoustic_load(e->w_ac, st_json_get(ac, "arch")))) return -1;
  const st_json* v = st_json_get(c, "vocoder");
  e->w_voc = open_weights(e->dir, NULL, st_json_str(st_json_get(v, "file"), "vocoder.safetensors"));
  if (!e->w_voc) return -1;
  e->voc = st_vocoder_load(e->w_voc, (int)st_json_num(st_json_get(v, "dim"), 384),
                           (int)st_json_num(st_json_get(v, "inter"), 896), (int)st_json_num(st_json_get(v, "layers"), 8),
                           1024, 256);
  return e->voc ? 0 : -1;
}

steertts_engine* steertts_create_ex(const char* model_dir, const steertts_config* cfg, char* err, size_t err_len);

static void copy_err(char* err, size_t len) {
  if (err && len) snprintf(err, len, "%s", st_geterr());
}

steertts_engine* steertts_create_ex(const char* model_dir, const steertts_config* cfg, char* err, size_t err_len) {
  steertts_config dflt;
  if (!cfg) {
    steertts_default_config(&dflt);
    cfg = &dflt;
  }
  steertts_engine* e = (steertts_engine*)st_calloc(sizeof *e);
  if (!e) {
    copy_err(err, err_len);
    return NULL;
  }
  e->cfg = *cfg;
  if (e->cfg.flow_steps <= 0) e->cfg.flow_steps = 10;
  if (!(e->cfg.flow_temp >= 0)) e->cfg.flow_temp = 0.4f;
  if (!(e->cfg.guidance >= 0 && e->cfg.guidance <= 4)) {
    st_seterr("guidance=%g is outside [0.0, 4.0]", (double)e->cfg.guidance);
    goto fail;
  }
  e->precision = st_strdup(cfg->precision && *cfg->precision ? cfg->precision : "fp32");
  if (strcmp(e->precision, "fp32") && strcmp(e->precision, "int8")) {
    st_seterr("precision must be one of ('fp32', 'int8')");
    goto fail;
  }
  if (!model_dir || !*model_dir) model_dir = getenv("STEERTTS_MODEL");
  if (!model_dir || !*model_dir) {
    st_seterr("no model given: pass a local model directory or set STEERTTS_MODEL");
    goto fail;
  }
  e->dir = st_strdup(model_dir);
  e->assets = cfg->assets_dir ? st_strdup(cfg->assets_dir) : path_join(model_dir, "native");
  e->cfg.precision = e->precision;
  e->cfg.assets_dir = e->assets;
  char* p = path_join(model_dir, "config.json");
  e->cfg_json = p ? load_json_file(p) : NULL;
  st_free(p);
  if (!e->cfg_json) goto fail;
  const char* mt = st_json_str(st_json_get(e->cfg_json, "model_type"), "");
  if (!strcmp(mt, "steer-tts-edge")) e->edge = 1;
  else if (strcmp(mt, "steer-tts")) {
    st_seterr("%s: unknown model_type '%s' (expected steer-tts or steer-tts-edge)", model_dir, mt);
    goto fail;
  }
  int th = cfg->threads == 0 ? n_cpus() : cfg->threads;
  e->pool = st_pool_create(th < 1 ? 1 : th);
  if (!e->pool) goto fail;
  p = path_join(e->assets, "front");
  e->front = p ? st_front_load(p) : NULL;
  if (!e->front) {
    st_seterr("text front-end assets: %s", st_geterr());
    st_free(p);
    goto fail;
  }
  st_free(p);
  if ((e->edge ? load_edge(e) : load_v1(e)) != 0) goto fail;
  /* the engine passes 512-d speaker vectors and keeps g on the stack; a model with other sizes is not supported */
  if (st_acoustic_g_dim(e->ac) > 512 || st_acoustic_spk_enc_dim(e->ac) != 512) {
    st_seterr("%s: unsupported acoustic model (g_dim %d, speaker vector %d; expected <= 512 and 512)", model_dir,
              st_acoustic_g_dim(e->ac), st_acoustic_spk_enc_dim(e->ac));
    goto fail;
  }
  if (e->edge) {
    int spk_dim = (int)st_json_num(st_json_get(st_json_get(st_json_get(e->cfg_json, "acoustic"), "arch"), "spk_dim"), 64);
    if (e->voice_dim != spk_dim) {
      st_seterr("voices.safetensors rows have %d values, the acoustic model expects %d", e->voice_dim, spk_dim);
      goto fail;
    }
  }
  if (e->n_voices < 1 || find_str(e->voices, e->n_voices, e->default_voice) < 0) {
    st_seterr("default voice %s is not in the voice table", e->default_voice);
    goto fail;
  }
  return e;
fail:
  copy_err(err, err_len);
  steertts_destroy(e);
  return NULL;
}

steertts_engine* steertts_create(const char* model_dir, int threads, char* err, size_t err_len) {
  steertts_config c;
  steertts_default_config(&c);
  c.threads = threads;
  return steertts_create_ex(model_dir, &c, err, err_len);
}

void steertts_destroy(steertts_engine* e) {
  if (!e) return;
  st_prompt_free(e->prompt);
  st_wavlm_free(e->wavlm);
  st_flow_free(e->flow);
  st_flow_free(e->pflow);
  st_vocoder_free(e->voc);
  st_acoustic_free(e->ac);
  st_weights_close(e->w_minilm);
  st_weights_close(e->w_adapter);
  st_weights_close(e->w_wavlm);
  st_weights_close(e->w_flow);
  st_weights_close(e->w_pflow);
  st_weights_close(e->w_voc);
  st_weights_close(e->w_ac);
  st_weights_close(e->w_voices);
  st_front_free(e->front);
  st_pool_destroy(e->pool);
  st_json_free(e->cfg_json);
  free_list(e->voices, e->n_voices);
  free_list(e->tags, e->n_tags);
  free_list(e->ptags, e->n_ptags);
  st_free(e->default_voice);
  st_free(e->dir);
  st_free(e->assets);
  st_free(e->precision);
  st_free(e);
}

const char* steertts_version(void) { return STEERTTS_VERSION; }
int steertts_sample_rate(void) { return SR; }
const char* steertts_model_type(const steertts_engine* e) { return !e ? NULL : e->edge ? "steer-tts-edge" : "steer-tts"; }
int steertts_num_voices(const steertts_engine* e) { return e ? e->n_voices : 0; }
const char* steertts_voice(const steertts_engine* e, int i) { return e && i >= 0 && i < e->n_voices ? e->voices[i] : NULL; }
const char* steertts_default_voice(const steertts_engine* e) { return e ? e->default_voice : NULL; }
int steertts_num_tags(const steertts_engine* e) { return e ? e->n_tags : 0; }
const char* steertts_tag(const steertts_engine* e, int i) { return e && i >= 0 && i < e->n_tags ? e->tags[i] : NULL; }
int steertts_num_controls(void) { return N_CTRL; }
const char* steertts_control(int i) { return i >= 0 && i < N_CTRL ? k_controls[i] : NULL; }

static int asset_exists(const steertts_engine* e, const char* name) {
  char rel[256];
  snprintf(rel, sizeof rel, "%s%s", strcmp(e->precision, "int8") ? "" : "int8/", name);
  char* p = path_join(e->assets, rel);
  int ok = p && file_exists(p);
  st_free(p);
  return ok;
}

int steertts_has(const steertts_engine* e, const char* f) {
  if (!e || !f) return 0;
  if (!strcmp(f, "flow")) return e->flow != NULL;
  if (e->edge) return 0;
  if (!strcmp(f, "prompt")) {
    char* v = path_join(e->assets, "minilm_vocab.txt");
    int ok = v && file_exists(v) && asset_exists(e, "minilm.safetensors");
    st_free(v);
    return ok;
  }
  if (!strcmp(f, "clone")) return asset_exists(e, "wavlm_sv.safetensors");
  return 0;
}

/* ------------------------------------------------------------------ lazy v1 parts */
static int need_prompt(steertts_engine* e) {
  if (e->prompt) return 0;
  if (e->edge) {
    st_seterr("prompt= is not available in the edge build (free-text prompts and voice cloning run on a host with "
              "the full model, WhissleAI/steer-tts)");
    return -1;
  }
  int int8 = !strcmp(e->precision, "int8");
  const st_json* pa = st_json_get(e->cfg_json, "prompt_adapter");
  const char* sub = int8 ? "int8" : NULL;
  if (!e->ptags && !(e->ptags = str_list(st_json_get(pa, "tags"), &e->n_ptags))) return -1;
  if (!e->w_adapter && !(e->w_adapter = open_weights(e->dir, sub, st_json_str(st_json_get(pa, "file"), "prompt_adapter.safetensors"))))
    return -1;
  if (!e->w_minilm && !(e->w_minilm = open_weights(e->assets, sub, "minilm.safetensors"))) {
    st_seterr("free-text prompts need the converted MiniLM encoder in %s (%s); see c/README.md", e->assets, st_geterr());
    return -1;
  }
  char* vocab = path_join(e->assets, "minilm_vocab.txt");
  if (!vocab) return -1;
  e->prompt = st_prompt_load(e->w_minilm, vocab, e->w_adapter, (const char* const*)e->ptags, e->n_ptags);
  st_free(vocab);
  return e->prompt ? 0 : -1;
}

static int need_wavlm(steertts_engine* e) {
  if (e->wavlm) return 0;
  if (e->edge) {
    st_seterr("ref_wav= is not available in the edge build (free-text prompts and voice cloning run on a host with "
              "the full model, WhissleAI/steer-tts)");
    return -1;
  }
  if (!e->w_wavlm && !(e->w_wavlm = open_weights(e->assets, strcmp(e->precision, "int8") ? NULL : "int8", "wavlm_sv.safetensors"))) {
    st_seterr("voice cloning needs the converted WavLM speaker encoder in %s (%s); see c/README.md", e->assets,
              st_geterr());
    return -1;
  }
  e->wavlm = st_wavlm_load(e->w_wavlm);
  return e->wavlm ? 0 : -1;
}

/* the prompted flow (lazy, as SteerTTS.pflow): *out = NULL without error when the release has none (no config entry or
 * no file, as _load_flow); a file that is there but cannot be loaded is an error */
static int get_pflow(steertts_engine* e, st_flow** out) {
  *out = e->pflow;
  if (e->pflow || e->pflow_tried || e->edge || !e->cfg.flow) return 0;
  const st_json* r = st_json_get(e->cfg_json, "prompted_flow");
  const char* f = st_json_str(st_json_get(r, "file"), NULL);
  char rel[512];
  snprintf(rel, sizeof rel, "%s%s", strcmp(e->precision, "int8") ? "" : "int8/", f ? f : "");
  char* path = f ? path_join(e->dir, rel) : NULL;
  int present = path && file_exists(path);
  st_free(path);
  if (!present) {
    e->pflow_tried = 1;
    return 0;
  }
  if (!e->w_pflow && !(e->w_pflow = open_weights(e->dir, strcmp(e->precision, "int8") ? NULL : "int8", f))) return -1;
  if (!(e->pflow = st_flow_load(e->w_pflow, st_json_get(r, "cfg")))) return -1;   /* not cached: retried next time */
  *out = e->pflow;
  return 0;
}

/* ------------------------------------------------------------------ requests */
typedef struct {
  int word;
  float pitch, energy, dur;   /* user units: semitones, dB, factor */
  int has;                    /* which keys the request gave (1 pitch, 2 energy, 4 dur), for resolve output */
} wshift;

typedef struct {
  char* text;
  char* voice;      /* NULL when cloning */
  char* ref_wav;
  float* spk_vec;   /* owned copy or NULL */
  char* prompt;
  char* tag;        /* NULL = unset ("none" after resolve) */
  int n_ctrl;
  int ctrl_dim[N_CTRL];
  double ctrl_val[N_CTRL];
  double pace, guidance;
  int markup, flow;
  int64_t seed;
  int n_ws;
  wshift* ws;
  /* prompt mapping (after resolve) */
  int mapped;
  char* map_tag;
  int map_n_ctrl;
  int map_dim[N_CTRL];
  double map_val[N_CTRL];
  int map_n_top;
  char* map_top_tag[3];
  double map_top_p[3];
} req_t;

static void req_free(req_t* r) {
  st_free(r->text);
  st_free(r->voice);
  st_free(r->ref_wav);
  st_free(r->spk_vec);
  st_free(r->prompt);
  st_free(r->tag);
  st_free(r->ws);
  st_free(r->map_tag);
  for (int i = 0; i < 3; ++i) st_free(r->map_top_tag[i]);
  memset(r, 0, sizeof *r);
}

static void ctrl_set(req_t* r, int dim, double v) {
  for (int i = 0; i < r->n_ctrl; ++i)
    if (r->ctrl_dim[i] == dim) {
      r->ctrl_val[i] = v;
      return;
    }
  r->ctrl_dim[r->n_ctrl] = dim;
  r->ctrl_val[r->n_ctrl++] = v;
}

/* Python repr() of a JSON value, for error messages */
static void py_repr(const st_json* v, char* out, size_t cap) {
  if (!v || v->type == ST_JNULL) snprintf(out, cap, "None");
  else if (v->type == ST_JBOOL) snprintf(out, cap, v->num ? "True" : "False");
  else if (v->type == ST_JNUM) py_float(out, cap, v->num, v->is_int);
  else if (v->type == ST_JSTR) snprintf(out, cap, "'%.200s'", v->str);
  else if (v->type == ST_JARR) snprintf(out, cap, v->n ? "[...]" : "[]");
  else snprintf(out, cap, v->n ? "{...}" : "{}");
}

/* Python truthiness of a JSON value (absent = None) */
static int py_truthy(const st_json* v) {
  if (!v) return 0;
  switch (v->type) {
    case ST_JNULL: return 0;
    case ST_JBOOL:
    case ST_JNUM: return v->num != 0;
    case ST_JSTR: return v->str[0] != 0;
    default: return v->n > 0;
  }
}

static int is_null(const st_json* v) { return !v || v->type == ST_JNULL; }

/* normalize_request's _num: a number (not a bool) inside [lo, hi] (edge's ctrl check prints the range as a tuple) */
static int json_num(const char* name, const st_json* v, double lo, double hi, const char* lo_s, const char* hi_s,
                    int tuple, double* out) {
  char rep[256];
  py_repr(v, rep, sizeof rep);
  if (!v || v->type != ST_JNUM) {
    st_seterr("%s must be a number, got %s", name, rep);
    return -1;
  }
  if (!(v->num >= lo && v->num <= hi)) {
    if (tuple) st_seterr("%s=%s is outside (%s, %s)", name, rep, lo_s, hi_s);
    else st_seterr("%s=%s is outside [%s, %s]", name, rep, lo_s, hi_s);
    return -1;
  }
  *out = v->num;
  return 0;
}

static const char* const k_fields[] = {"text", "voice", "ref_wav", "prompt", "tag", "ctrl", "pace", "guidance",
                                       "word_shift", "markup", "flow", "seed", "prompt_mapping", "$schema", "spk_vec"};

static int cmp_str(const void* a, const void* b) { return strcmp(*(const char* const*)a, *(const char* const*)b); }

static const char* const k_edge_off = " is not available in the edge build (free-text prompts and voice cloning run on a "
                                      "host with the full model, WhissleAI/steer-tts)";

static void tag_list(const steertts_engine* e, sbuf* b) {
  for (int i = 0; i < e->n_tags; ++i) sb_fmt(b, "%s%s", i ? ", " : "", e->tags[i]);
}

/* word_shift = {word_index: {pitch, energy, dur}} (normalize_request) */
static int parse_word_shift(const st_json* ws, req_t* r) {
  for (int i = 0; i < ws->n; ++i) {
    const char* k = ws->keys[i];
    const char* s = k;
    while (isspace((unsigned char)*s)) ++s;
    int neg = *s == '-';
    if (*s == '+' || *s == '-') ++s;
    const char* d0 = s;
    long long w = 0;
    while (*s >= '0' && *s <= '9') {
      if (w < INT_MAX) w = w * 10 + (*s - '0');
      ++s;
    }
    const char* t = s;
    while (isspace((unsigned char)*t)) ++t;
    if (s == d0 || *t) {
      st_seterr("word_shift key '%s' is not a word index", k);
      return -1;
    }
    const st_json* sh = &ws->items[i];
    int bad = neg && w > 0;
    if (!bad && sh->type == ST_JOBJ)
      for (int j = 0; j < sh->n; ++j) bad |= strcmp(sh->keys[j], "pitch") && strcmp(sh->keys[j], "energy") &&
                                            strcmp(sh->keys[j], "dur");
    if (bad || sh->type != ST_JOBJ) {
      st_seterr("word_shift['%s'] must be a non-negative index -> {pitch, energy, dur}", k);
      return -1;
    }
    int word = w > INT_MAX ? INT_MAX : (int)w;   /* an index past the text simply never applies */
    int j = 0;
    for (; j < r->n_ws; ++j)
      if (r->ws[j].word == word) break;
    if (j == r->n_ws) {
      wshift* nw = (wshift*)realloc(r->ws, sizeof(wshift) * (size_t)(r->n_ws + 1));
      if (!nw) {
        st_seterr("out of memory");
        return -1;
      }
      r->ws = nw;
      r->ws[j].word = word, r->ws[j].pitch = 0, r->ws[j].energy = 0, r->ws[j].dur = 1;
      r->ws[j].has = 0;
      r->n_ws++;
    }
    for (int q = 0; q < sh->n; ++q) {
      static const struct { const char* k; double lo, hi; const char *los, *his; } lim[3] = {
          {"pitch", -12, 12, "-12.0", "12.0"}, {"energy", -20, 20, "-20.0", "20.0"}, {"dur", 0.25, 4, "0.25", "4.0"}};
      int ki = !strcmp(sh->keys[q], "pitch") ? 0 : !strcmp(sh->keys[q], "energy") ? 1 : 2;
      char nm[128];
      double v;
      snprintf(nm, sizeof nm, "word_shift.%.40s.%s", k, sh->keys[q]);
      if (json_num(nm, &sh->items[q], lim[ki].lo, lim[ki].hi, lim[ki].los, lim[ki].his, 0, &v)) return -1;
      if (ki == 0) r->ws[j].pitch = (float)v;
      else if (ki == 1) r->ws[j].energy = (float)v;
      else r->ws[j].dur = (float)v;
      r->ws[j].has |= 1 << ki;
    }
  }
  return 0;
}

/* a request object -> req_t, validated in the order and with the messages of the model's Python package
 * (steertts.request.normalize_request for v1, EdgeTTS.__call__ / _check / control_tensors for edge) */
static int validate_request(steertts_engine* e, const st_json* j, req_t* r, int streaming) {
  memset(r, 0, sizeof *r);
  if (!j || j->type != ST_JOBJ) {
    st_seterr("a request is a JSON object");
    return -1;
  }
  {
    const char* unk[64];
    int nu = 0;
    for (int i = 0; i < j->n && nu < 64; ++i) {
      int ok = 0;
      for (size_t f = 0; f < sizeof k_fields / sizeof *k_fields; ++f) ok |= !strcmp(j->keys[i], k_fields[f]);
      int dup = 0;
      for (int q = 0; q < nu; ++q) dup |= !strcmp(unk[q], j->keys[i]);
      if (!ok && !dup) unk[nu++] = j->keys[i];
    }
    if (nu) {
      qsort(unk, (size_t)nu, sizeof *unk, cmp_str);   /* Python: sorted(unknown) */
      sbuf b = {0};
      for (int q = 0; q < nu; ++q) sb_fmt(&b, "%s'%s'", q ? ", " : "", unk[q]);
      st_seterr("unknown request field(s) [%s]; see request.schema.json", b.s ? b.s : "");
      free(b.s);
      return -1;
    }
  }
  const st_json *text = st_json_get(j, "text"), *voice = st_json_get(j, "voice"), *ref = st_json_get(j, "ref_wav"),
                *prompt = st_json_get(j, "prompt"), *tag = st_json_get(j, "tag"), *ctrl = st_json_get(j, "ctrl"),
                *ws = st_json_get(j, "word_shift"), *markup = st_json_get(j, "markup"), *flow = st_json_get(j, "flow"),
                *seed = st_json_get(j, "seed"), *sv = st_json_get(j, "spk_vec");
  if (!text || text->type != ST_JSTR || (!e->edge && is_blank(text->str))) {
    st_seterr("text must be a non-empty string");
    return -1;
  }
  if (utf8_len(text->str) > 5000) {
    st_seterr("text is longer than 5000 characters; split it (or stream it sentence by sentence)");
    return -1;
  }
  if (!(r->text = st_strdup(text->str))) return -1;
  if (!is_null(sv)) {
    if (sv->type != ST_JARR || sv->n != 512) {
      st_seterr("spk_vec must be a list of 512 numbers");
      return -1;
    }
    for (int i = 0; i < 512; ++i)
      if (sv->items[i].type != ST_JNUM) {
        st_seterr("spk_vec must be a list of 512 numbers");
        return -1;
      }
    if (!(r->spk_vec = (float*)st_malloc(sizeof(float) * 512))) return -1;
    for (int i = 0; i < 512; ++i) r->spk_vec[i] = (float)sv->items[i].num;
  }
  if (!is_null(markup) && markup->type != ST_JBOOL) {
    st_seterr("markup must be true or false");
    return -1;
  }
  r->markup = !is_null(markup) && markup->num != 0;
  char known[256], rep[256];
  join_controls(known, sizeof known);

  if (e->edge) {
    /* EdgeTTS: tag or "none"; _check: word_shift with markup / stream, prompt, ref_wav, tag, voice */
    if (py_truthy(tag) && tag->type != ST_JSTR) {
      st_seterr("tag must be a string");
      return -1;
    }
    if (py_truthy(ws) && (r->markup || streaming)) {
      st_seterr("word_shift does not combine with markup / stream(); use inline markup (*word*, <slow>, <high> ...) "
                "there");
      return -1;
    }
    if (!is_null(prompt)) {
      st_seterr("prompt=%s", k_edge_off);
      return -1;
    }
    if (!is_null(ref)) {
      st_seterr("ref_wav=%s", k_edge_off);
      return -1;
    }
    const char* t = py_truthy(tag) ? tag->str : "none";
    if (find_str(e->tags, e->n_tags, t) < 0) {
      sbuf b = {0};
      tag_list(e, &b);
      st_seterr("unknown tag '%s'; known tags: %s", t, b.s ? b.s : "");
      free(b.s);
      return -1;
    }
    r->tag = st_strdup(t);
    if (!r->spk_vec && !is_null(voice)) {
      if (voice->type != ST_JSTR || find_str(e->voices, e->n_voices, voice->str) < 0) {
        py_repr(voice, rep, sizeof rep);
        st_seterr("unknown voice %s; this edge build ships %d voices (see tts.voices), e.g. %s, %s, %s", rep,
                  e->n_voices, e->voices[0], e->voices[1 % e->n_voices], e->voices[2 % e->n_voices]);
        return -1;
      }
    }
    if (!r->spk_vec) r->voice = st_strdup(!is_null(voice) && *voice->str ? voice->str : e->default_voice);
    if (py_truthy(ctrl)) {
      if (ctrl->type != ST_JOBJ) {
        st_seterr("ctrl must be an object {control: z}");
        return -1;
      }
      for (int i = 0; i < ctrl->n; ++i) {   /* control_tensors: name, then range, item by item */
        int d = ctrl_index(ctrl->keys[i]);
        if (d < 0) {
          st_seterr("unknown control '%s'; known: %s", ctrl->keys[i], known);
          return -1;
        }
        char nm[64];
        double x;
        snprintf(nm, sizeof nm, "ctrl.%s", ctrl->keys[i]);
        if (json_num(nm, &ctrl->items[i], -5, 5, "-5.0", "5.0", 1, &x)) return -1;
        ctrl_set(r, d, x);
      }
    }
  } else {
    /* normalize_request */
    if (!is_null(ref)) {
      if (ref->type != ST_JSTR) {
        st_seterr("ref_wav must be a path string");
        return -1;
      }
      if (py_truthy(voice)) {
        st_seterr("give either voice (a built-in voice) or ref_wav (clone a reference clip), not both");
        return -1;
      }
      r->ref_wav = st_strdup(ref->str);
    } else if (!r->spk_vec) {
      const char* v = py_truthy(voice) && voice->type == ST_JSTR ? voice->str : NULL;
      if (py_truthy(voice) && !v) {
        py_repr(voice, rep, sizeof rep);
        st_seterr("unknown voice %s; list them with tts.voices / `steertts voices`", rep);
        return -1;
      }
      if (!v) v = e->default_voice;
      if (find_str(e->voices, e->n_voices, v) < 0) {
        st_seterr("unknown voice '%s'; list them with tts.voices / `steertts voices`", v);
        return -1;
      }
      r->voice = st_strdup(v);
    }
    if (!is_null(prompt)) {
      if (prompt->type != ST_JSTR) {
        st_seterr("prompt must be a string");
        return -1;
      }
      r->prompt = st_strdup(prompt->str);
    }
    if (!is_null(tag)) {
      if (tag->type != ST_JSTR) {
        st_seterr("tag must be a string");
        return -1;
      }
      if (find_str(e->tags, e->n_tags, tag->str) < 0) {
        sbuf b = {0};
        tag_list(e, &b);
        st_seterr("unknown tag '%s'; known: %s", tag->str, b.s ? b.s : "");
        free(b.s);
        return -1;
      }
      r->tag = st_strdup(tag->str);
    }
    if (py_truthy(ctrl)) {
      if (ctrl->type != ST_JOBJ) {
        st_seterr("ctrl must be an object {control: z}");
        return -1;
      }
      for (int i = 0; i < ctrl->n; ++i)
        if (ctrl_index(ctrl->keys[i]) < 0) {
          st_seterr("unknown control '%s'; known: %s", ctrl->keys[i], known);
          return -1;
        }
      for (int i = 0; i < ctrl->n; ++i) {
        char nm[64];
        double x;
        snprintf(nm, sizeof nm, "ctrl.%s", ctrl->keys[i]);
        if (json_num(nm, &ctrl->items[i], -5, 5, "-5.0", "5.0", 0, &x)) return -1;
        ctrl_set(r, ctrl_index(ctrl->keys[i]), x);
      }
    }
  }
  const st_json* pace = st_json_get(j, "pace");
  const st_json* guid = st_json_get(j, "guidance");
  r->pace = 1.0;
  r->guidance = e->cfg.guidance;
  if (!is_null(pace) && json_num("pace", pace, 0.25, 4.0, "0.25", "4.0", 0, &r->pace)) return -1;
  if (!is_null(guid) && json_num("guidance", guid, 0.0, 4.0, "0.0", "4.0", 0, &r->guidance)) return -1;
  r->flow = e->cfg.flow != 0;
  if (!e->edge) {   /* edge accepts flow / seed and ignores them, as EdgeTTS does */
    if (!is_null(flow)) {
      if (flow->type != ST_JBOOL) {
        st_seterr("flow must be true or false");
        return -1;
      }
      r->flow = flow->num != 0;
    }
    if (!is_null(seed)) {
      if (seed->type != ST_JNUM || !seed->is_int || seed->num < 0 || seed->num > 9.2e18) {
        st_seterr("seed must be a non-negative integer");
        return -1;
      }
      r->seed = (int64_t)seed->num;
    }
  }
  if (py_truthy(ws)) {
    if (ws->type != ST_JOBJ) {
      st_seterr("word_shift must be an object {word_index: {pitch, energy, dur}}");
      return -1;
    }
    if (!e->edge && r->markup) {
      st_seterr("word_shift and markup cannot be combined (markup re-indexes words per segment)");
      return -1;
    }
    if (parse_word_shift(ws, r)) return -1;
  }
  if (!r->text || (r->tag == NULL && e->edge) || (!r->voice && !r->ref_wav && !r->spk_vec)) {
    st_seterr("out of memory");
    return -1;
  }
  return 0;
}

static int req_from_json(steertts_engine* e, const char* json, req_t* r, int streaming) {
  memset(r, 0, sizeof *r);
  st_json* j = st_json_parse(json, strlen(json));
  if (!j) return -1;
  int rc = validate_request(e, j, r, streaming);
  st_json_free(j);
  return rc;
}

/* a JSON number (Python float formatting, NaN / Infinity spelled the JSON way) */
static void sb_num(sbuf* b, double v) {
  if (isnan(v)) sb_str(b, "NaN");
  else if (isinf(v)) sb_str(b, v > 0 ? "Infinity" : "-Infinity");
  else {
    char rep[48];
    py_float(rep, sizeof rep, v, 0);
    sb_str(b, rep);
  }
}

/* "name=z,name=z" -> {"name": z, ...} (names are checked by the validator, values must be numbers here) */
static int ctrl_str_json(sbuf* b, const char* s) {
  sb_str(b, "{");
  int first = 1;
  const char* p = s;
  while (*p) {
    while (*p == ' ' || *p == ',') ++p;
    if (!*p) break;
    const char* end = strchr(p, ',');
    if (!end) end = p + strlen(p);
    const char* eq = memchr(p, '=', (size_t)(end - p));
    if (!eq) {
      st_seterr("ctrl must look like name=z,name=z (got '%.*s')", (int)(end - p), p);
      return -1;
    }
    const char* ne = eq;
    while (ne > p && ne[-1] == ' ') --ne;
    char* name = (char*)malloc((size_t)(ne - p) + 1);
    char* num = (char*)malloc((size_t)(end - eq));
    if (!name || !num) {
      free(name);
      free(num);
      st_seterr("out of memory");
      return -1;
    }
    memcpy(name, p, (size_t)(ne - p));
    name[ne - p] = 0;
    memcpy(num, eq + 1, (size_t)(end - eq - 1));
    num[end - eq - 1] = 0;
    char* q;
    double v = st_strtod(num, &q);
    while (*q == ' ') ++q;
    if (q == num || *q) {
      st_seterr("ctrl.%s must be a number, got '%s'", name, num);
      free(name);
      free(num);
      return -1;
    }
    sb_str(b, first ? "" : ", ");
    sb_json_str(b, name);
    sb_str(b, ": ");
    sb_num(b, v);
    first = 0;
    free(name);
    free(num);
    p = end;
  }
  sb_str(b, "}");
  return 0;
}

/* "word:key=v,key=v;word:..." -> {"word": {"key": v}, ...} */
static int ws_str_json(sbuf* b, const char* s) {
  sb_str(b, "{");
  int first = 1;
  const char* p = s;
  while (*p) {
    while (*p == ' ' || *p == ';') ++p;
    if (!*p) break;
    const char* c = p;
    while (*c >= '0' && *c <= '9') ++c;
    if (c == p || *c != ':') {
      st_seterr("word_shift must look like word:key=v,key=v;word:... (got '%s')", p);
      return -1;
    }
    sb_fmt(b, "%s\"%.*s\": {", first ? "" : ", ", (int)(c - p), p);
    first = 0;
    p = c + 1;
    int f2 = 1;
    while (*p && *p != ';') {
      while (*p == ' ' || *p == ',') ++p;
      if (!*p || *p == ';') break;
      const char* eq = p;
      while (*eq && *eq != '=' && *eq != ',' && *eq != ';') ++eq;
      if (*eq != '=') {
        st_seterr("word_shift must look like word:key=v,key=v;word:... (got '%s')", p);
        return -1;
      }
      char* q;
      double v = st_strtod(eq + 1, &q);
      if (q == eq + 1 || (*q && *q != ',' && *q != ';' && *q != ' ')) {
        st_seterr("word_shift value for '%.*s' must be a number (got '%s')", (int)(eq - p), p, eq + 1);
        return -1;
      }
      sb_fmt(b, "%s\"%.*s\": ", f2 ? "" : ", ", (int)(eq - p), p);
      sb_num(b, v);
      f2 = 0;
      p = q;
    }
    sb_str(b, "}");
  }
  sb_str(b, "}");
  return 0;
}

void steertts_default_options(steertts_options* o) {
  if (!o) return;
  memset(o, 0, sizeof *o);
  o->pace = 1.0f;
  o->guidance = -1.0f;
  o->flow = -1;
}

/* steertts_options -> the equivalent JSON request, validated by the same code as the JSON entry points */
static int req_from_options(steertts_engine* e, const char* text, const steertts_options* o, req_t* r, int streaming) {
  steertts_options d;
  if (!o) {
    steertts_default_options(&d);
    o = &d;
  }
  memset(r, 0, sizeof *r);
  sbuf b = {0};
  sb_str(&b, "{\"text\": ");
  sb_json_str(&b, text ? text : "");
  if (o->voice) sb_str(&b, ", \"voice\": "), sb_json_str(&b, o->voice);
  if (o->tag) sb_str(&b, ", \"tag\": "), sb_json_str(&b, o->tag);
  if (o->prompt) sb_str(&b, ", \"prompt\": "), sb_json_str(&b, o->prompt);
  if (o->ref_wav) sb_str(&b, ", \"ref_wav\": "), sb_json_str(&b, o->ref_wav);
  if (o->ctrl && *o->ctrl) {
    sb_str(&b, ", \"ctrl\": ");
    if (ctrl_str_json(&b, o->ctrl)) goto fail;
  }
  if (o->word_shift && *o->word_shift) {
    sb_str(&b, ", \"word_shift\": ");
    if (ws_str_json(&b, o->word_shift)) goto fail;
  }
  if (o->pace != 0) sb_str(&b, ", \"pace\": "), sb_num(&b, o->pace);   /* 0 (a zeroed struct) = default */
  if (o->guidance >= 0 || isnan(o->guidance)) sb_str(&b, ", \"guidance\": "), sb_num(&b, o->guidance);
  if (o->markup) sb_str(&b, ", \"markup\": true");
  if (o->flow >= 0) sb_fmt(&b, ", \"flow\": %s", o->flow ? "true" : "false");
  if (o->seed < 0) {
    st_seterr("seed must be a non-negative integer");
    goto fail;
  }
  if (o->seed) sb_fmt(&b, ", \"seed\": %lld", (long long)o->seed);
  if (o->spk_vec) {
    sb_str(&b, ", \"spk_vec\": [");
    for (int i = 0; i < 512; ++i) {
      if (i) sb_str(&b, ", ");
      sb_num(&b, o->spk_vec[i]);
    }
    sb_str(&b, "]");
  }
  sb_str(&b, "}");
  if (b.oom) {
    st_seterr("out of memory");
    goto fail;
  }
  int rc = req_from_json(e, b.s, r, streaming);
  free(b.s);
  return rc;
fail:
  free(b.s);
  return -1;
}

/* SteerTTS.resolve: map the prompt (explicit tag wins, explicit ctrl dims override the prompt's) */
static int resolve_prompt(steertts_engine* e, req_t* r) {
  if (!r->prompt || !*r->prompt) {
    st_free(r->prompt);
    r->prompt = NULL;
    if (!r->tag) r->tag = st_strdup("none");
    return 0;
  }
  if (need_prompt(e)) return -1;
  st_prompt_result m;
  if (st_prompt_map(e->prompt, e->pool, r->prompt, &m)) return -1;
  r->mapped = 1;
  const char* tag = "none";
  for (int i = 0; i < m.n_top; ++i) {
    const char* t = e->ptags[m.top_tag[i]];
    if (find_str(e->tags, e->n_tags, t) >= 0) {
      tag = t;
      break;
    }
  }
  r->map_tag = st_strdup(tag);
  r->map_n_top = m.n_top;
  for (int i = 0; i < m.n_top; ++i) {
    r->map_top_tag[i] = st_strdup(e->ptags[m.top_tag[i]]);
    r->map_top_p[i] = m.top_p_milli[i] / 1000.0;   /* exactly Python's round(p, 3) */
  }
  r->map_n_ctrl = m.n_ctrl;
  for (int i = 0; i < m.n_ctrl; ++i) r->map_dim[i] = m.ctrl_dim[i], r->map_val[i] = m.ctrl_z_centi[i] / 100.0;
  /* ctrl = {**prompt ctrl, **request ctrl} */
  int nd = r->n_ctrl, dims[N_CTRL];
  double vals[N_CTRL];
  memcpy(dims, r->ctrl_dim, sizeof dims);
  memcpy(vals, r->ctrl_val, sizeof vals);
  r->n_ctrl = 0;
  for (int i = 0; i < m.n_ctrl; ++i) ctrl_set(r, m.ctrl_dim[i], r->map_val[i]);
  for (int i = 0; i < nd; ++i) ctrl_set(r, dims[i], vals[i]);
  if (!r->tag) r->tag = st_strdup(tag);
  st_free(r->prompt);
  r->prompt = NULL;
  return 0;
}

/* ------------------------------------------------------------------ rendering */
typedef struct {
  float* y;
  size_t n, cap;
  int chunks;
} fbuf;

static int fb_add(fbuf* b, const float* x, size_t n) {
  if (n == 0) return 0;
  if (b->n + n > b->cap) {
    size_t c = b->cap ? b->cap * 2 : 48000;
    while (c < b->n + n) c *= 2;
    float* t = (float*)realloc(b->y, sizeof(float) * c);
    if (!t) {
      st_seterr("out of memory");
      return -1;
    }
    b->y = t;
    b->cap = c;
  }
  if (x) memcpy(b->y + b->n, x, sizeof(float) * n);
  else memset(b->y + b->n, 0, sizeof(float) * n);
  b->n += n;
  return 0;
}

/* the speaker side of a request: a 512-d vector (clone / spk_vec / v1 voice) or a 64-d slot (edge voice) */
typedef struct {
  const float* vec;     /* 512-d or NULL */
  const float* slot;    /* 64-d or NULL */
  const float* ref24;   /* the reference clip at 24 kHz (cloning) */
  size_t ref_n;
  int ref_sr;           /* its original rate */
  float* owned_ref;
} spk_t;

/* a reference clip: plausible rate, finite samples, only the part that is used (the first 8 s + filter reach) */
static int ref_prepare(float* y, size_t* n, int sr) {
  if (sr < 1000 || sr > 768000) {
    st_seterr("reference audio: sample rate %d Hz is outside [1000, 768000]", sr);
    return -1;
  }
  size_t keep = (size_t)(9.0 * sr) + 256;
  if (*n > keep) *n = keep;
  for (size_t i = 0; i < *n; ++i)
    if (!isfinite(y[i])) {
      st_seterr("reference audio contains NaN or infinite samples");
      return -1;
    }
  return 0;
}

static int ref_vector(steertts_engine* e, const char* path, spk_t* s) {
  float* y;
  size_t n;
  int sr;
  if (st_wav_read(path, &y, &n, &sr)) return -1;
  if (ref_prepare(y, &n, sr)) {
    st_free(y);
    return -1;
  }
  if (sr != SR) {
    float* z;
    size_t nz;
    if (st_resample(y, n, sr, SR, &z, &nz)) {
      st_free(y);
      return -1;
    }
    st_free(y);
    y = z;
    n = nz;
  }
  s->owned_ref = y;
  s->ref24 = y;
  s->ref_n = n;
  s->ref_sr = sr;
  uint64_t h = 1469598103934665603ULL ^ (uint64_t)n;
  const unsigned char* bytes = (const unsigned char*)y;
  for (size_t i = 0; i < n * sizeof(float); ++i) h = (h ^ bytes[i]) * 1099511628211ULL;
  h = h ? h : 1;
  if (e->xv.key != h) {
    e->xv.key = 0;
    if (need_wavlm(e) || st_wavlm_xvector(e->wavlm, e->pool, y, n, e->xv.vec)) return -1;
    e->xv.key = h;
  }
  s->vec = e->xv.vec;
  return 0;
}

static int speaker_of(steertts_engine* e, const req_t* r, spk_t* s) {
  memset(s, 0, sizeof *s);
  if (r->ref_wav) return ref_vector(e, r->ref_wav, s);
  if (r->spk_vec) {
    s->vec = r->spk_vec;
    return 0;
  }
  int vi = find_str(e->voices, e->n_voices, r->voice ? r->voice : e->default_voice);
  if (vi < 0) {
    st_seterr("unknown voice '%s'", r->voice);
    return -1;
  }
  if (e->edge) s->slot = e->voice_rows + (size_t)vi * e->voice_dim;
  else s->vec = e->voice_rows + (size_t)vi * 512;
  return 0;
}

/* one forward pass (SteerTTS._render / EdgeTTS._render) -> *y (st_malloc'd, may be n = 0) */
static int render(steertts_engine* e, const char* text, const spk_t* spk, const char* tag, int n_ctrl, const int* dims,
                  const double* vals, double pace, const wshift* ws, int n_ws, double guidance, int flow, int64_t seed,
                  float** y, size_t* n) {
  *y = NULL;
  *n = 0;
  st_phonemes ph;
  if (st_front_phonemes(e->front, text, &ph)) return -1;
  int rc = -1;
  float* mel = NULL;
  float g[512], g0[512];
  float* shifts = NULL;
  if (e->edge && !st_front_has_speech(&ph)) {
    st_phonemes_free(&ph);
    *y = (float*)st_malloc(sizeof(float));
    return *y ? 0 : -1;
  }
  int nsym = st_acoustic_n_symbols(e->ac), N = 0;
  for (int i = 0; i < ph.n; ++i)
    if (ph.ids[i] < nsym) ph.ids[N] = ph.ids[i], ph.widx[N] = ph.widx[i], ++N;
  if (N == 0) {   /* cannot happen (a leading <sp> is always there); keep Python's empty result */
    st_phonemes_free(&ph);
    *y = (float*)st_malloc(sizeof(float));
    return *y ? 0 : -1;
  }
  float c[N_CTRL] = {0}, m[N_CTRL] = {0}, z[N_CTRL] = {0};
  for (int i = 0; i < n_ctrl; ++i) c[dims[i]] = (float)vals[i], m[dims[i]] = 1.f;
  int ti = tag ? find_str(e->tags, e->n_tags, tag) : 0;
  if (ti < 0) ti = 0;
  if (st_acoustic_condition(e->ac, spk->slot, spk->vec, ti, c, m, g)) goto done;
  int guided = guidance != 1.0 && (n_ctrl > 0 || (tag && strcmp(tag, "none")));
  if (guided && st_acoustic_condition(e->ac, spk->slot, spk->vec, 0, z, z, g0)) goto done;
  /* per-word shifts in model units (pipeline._render): pitch / PITCH_SCALE, energy / 10 */
  int n_words = 0;
  for (int i = 0; i < N; ++i) n_words = ph.widx[i] + 1 > n_words ? ph.widx[i] + 1 : n_words;
  st_shift* sh = NULL;
  if (n_ws && n_words) {
    sh = (st_shift*)st_calloc(sizeof(st_shift) * (size_t)n_words);
    if (!sh) goto done;
    for (int w = 0; w < n_words; ++w) sh[w].dur = 1.f;
    for (int i = 0; i < n_ws; ++i)
      if (ws[i].word >= 0 && ws[i].word < n_words) {
        sh[ws[i].word].pitch = (float)(ws[i].pitch / PITCH_SCALE);
        sh[ws[i].word].energy = (float)(ws[i].energy / 10.0);
        sh[ws[i].word].dur = ws[i].dur;
      }
    shifts = (float*)sh;
  }
  int T;
  if (st_acoustic_infer(e->ac, e->pool, ph.ids, ph.widx, N, g, guided ? g0 : NULL, (float)guidance, (float)pace, sh,
                        sh ? n_words : 0, &mel, &T, NULL))
    goto done;
  if (!e->edge && flow) {
    st_flow* fl = NULL;
    st_flow* pf = NULL;
    if (spk->ref24 && get_pflow(e, &pf)) goto done;
    if (e->flow) fl = pf ? pf : e->flow;
    else if (spk->ref24) fl = pf;
    if (fl) {
      float* pm = NULL;
      int P = 0;
      if (spk->ref24 && st_flow_is_prompted(fl) &&
          st_prompt_mel(spk->ref24, spk->ref_n, spk->ref_sr, mel, T, &pm, &P))
        goto done;
      float* out = (float*)st_malloc(sizeof(float) * (size_t)T * 100);
      if (!out) {
        st_free(pm);
        goto done;
      }
      int frc = st_flow_sample(fl, e->pool, mel, T, g, e->cfg.flow_steps, e->cfg.flow_temp, (uint64_t)seed, pm, P, 0.0f,
                               out);
      st_free(pm);
      st_free(mel);
      mel = out;
      if (frc) goto done;
    }
  }
  if (st_vocoder_decode(e->voc, e->pool, mel, T, y, n)) goto done;
  rc = 0;
done:
  st_free(shifts);
  st_free(mel);
  st_phonemes_free(&ph);
  return rc;
}

static void fade_ends(float* y, size_t n) {
  const int fade = SR / 100;
  if (n <= (size_t)(2 * fade)) return;
  const double step = 1.0 / (fade - 1);
  for (int i = 0; i < fade; ++i) {
    float r = i == fade - 1 ? 1.0f : (float)(i * step);
    y[i] *= r;
    y[n - 1 - (size_t)i] *= r;
  }
}

typedef struct {
  steertts_engine* e;
} prompt_cb_ctx;

/* <style prompt="..."> inside markup (MarkupSynth._style_of_prompt): the adapter's best known tag != "none" */
static int style_of_prompt(void* user, const char* prompt, char** tag, int* n_ctrl, char*** names, float** vals) {
  steertts_engine* e = ((prompt_cb_ctx*)user)->e;
  if (need_prompt(e)) return -1;
  st_prompt_result m;
  if (st_prompt_map(e->prompt, e->pool, prompt, &m)) return -1;
  const char* t = "none";
  for (int i = 0; i < m.n_top; ++i) {
    const char* c = e->ptags[m.top_tag[i]];
    if (find_str(e->tags, e->n_tags, c) >= 0 && strcmp(c, "none")) {
      t = c;
      break;
    }
  }
  *tag = st_strdup(t);
  *n_ctrl = m.n_ctrl;
  *names = (char**)st_calloc(sizeof(char*) * (size_t)(m.n_ctrl + 1));
  *vals = (float*)st_calloc(sizeof(float) * (size_t)(m.n_ctrl + 1));
  if (!*tag || !*names || !*vals) return -1;
  for (int i = 0; i < m.n_ctrl; ++i) {
    (*names)[i] = st_strdup(k_controls[m.ctrl_dim[i]]);
    (*vals)[i] = (float)(m.ctrl_z_centi[i] / 100.0);
  }
  return 0;
}

typedef int (*chunk_sink)(void* user, const float* y, size_t n, const steertts_chunk_info* info);

/* markup segments in order (MarkupSynth.stream): pauses as zeros, speech with 10 ms fades */
static int run_segments(steertts_engine* e, const req_t* r, const spk_t* spk, chunk_sink sink, void* user) {
  const char* names[N_CTRL];
  float vals[N_CTRL];
  for (int i = 0; i < r->n_ctrl; ++i) names[i] = k_controls[r->ctrl_dim[i]], vals[i] = (float)r->ctrl_val[i];
  st_segment* segs = NULL;
  int ns = 0;
  prompt_cb_ctx pc = {e};
  if (st_front_parse_markup(e->front, r->text, r->tag, names, vals, r->n_ctrl, (const char* const*)e->tags, e->n_tags,
                            e->edge ? NULL : style_of_prompt, &pc, &segs, &ns))
    return -1;
  int rc = 0;
  for (int i = 0; i < ns && rc == 0; ++i) {
    st_segment* s = &segs[i];
    if (s->skip) continue;
    double t0 = st_time_ms();
    float* y = NULL;
    size_t n = 0;
    if (s->pause_ms > 0) {
      n = (size_t)((long long)s->pause_ms * SR / 1000);
      y = (float*)st_calloc(sizeof(float) * (n ? n : 1));
      if (!y) {
        rc = -1;
        break;
      }
    } else {
      int dims[N_CTRL];
      double dv[N_CTRL];
      int nc = 0;
      for (int k = 0; k < s->n_ctrl && nc < N_CTRL; ++k) {
        int d = ctrl_index(s->ctrl_name[k]);
        if (d >= 0) dims[nc] = d, dv[nc++] = s->ctrl_val[k];
      }
      wshift* ws = NULL;
      if (s->n_ws) {
        ws = (wshift*)st_malloc(sizeof(wshift) * (size_t)s->n_ws);
        if (!ws) {
          rc = -1;
          break;
        }
        for (int k = 0; k < s->n_ws; ++k)
          ws[k].word = s->ws[k].word, ws[k].pitch = s->ws[k].pitch, ws[k].energy = s->ws[k].energy,
          ws[k].dur = s->ws[k].dur;
      }
      rc = render(e, s->text, spk, s->tag, nc, dims, dv, r->pace, ws, s->n_ws, r->guidance, r->flow, r->seed, &y, &n);
      st_free(ws);
      if (rc) break;
      fade_ends(y, n);
    }
    char ctrl_s[512] = "";
    size_t k = 0;
    for (int c = 0; c < s->n_ctrl && k < sizeof ctrl_s; ++c) {
      char rep[48];
      float_short(rep, sizeof rep, s->ctrl_val[c]);
      k += (size_t)snprintf(ctrl_s + k, sizeof ctrl_s - k, "%s%s=%s", c ? "," : "", s->ctrl_name[c], rep);
    }
    steertts_chunk_info info = {s->text ? s->text : "", s->tag ? s->tag : "none", ctrl_s, s->pause_ms, i,
                                st_time_ms() - t0};
    int st = sink(user, y, n, &info);
    st_free(y);
    if (st < 0) {
      rc = -1;
      break;
    }
    if (st) break;
  }
  st_segments_free(segs, ns);
  return rc;
}

static int collect_sink(void* user, const float* y, size_t n, const steertts_chunk_info* info) {
  (void)info;
  fbuf* b = (fbuf*)user;
  b->chunks++;
  return fb_add(b, y, n) ? -1 : 0;
}

static int synth_req(steertts_engine* e, req_t* r, float** audio, size_t* n) {
  *audio = NULL;
  *n = 0;
  if (resolve_prompt(e, r)) return -1;
  spk_t spk;
  if (speaker_of(e, r, &spk)) {
    st_free(spk.owned_ref);
    return -1;
  }
  int rc;
  if (r->markup) {
    fbuf b = {0};
    rc = run_segments(e, r, &spk, collect_sink, &b);
    if (rc == 0 && b.chunks == 0) rc = fb_add(&b, NULL, 1);   /* Python: np.zeros(1) when there is no segment */
    if (rc == 0) {
      *audio = (float*)st_malloc(sizeof(float) * (b.n ? b.n : 1));
      if (!*audio) rc = -1;
      else {
        if (b.n) memcpy(*audio, b.y, sizeof(float) * b.n);
        *n = b.n;
      }
    }
    free(b.y);
  } else {
    rc = render(e, r->text, &spk, r->tag, r->n_ctrl, r->ctrl_dim, r->ctrl_val, r->pace, r->ws, r->n_ws, r->guidance,
                r->flow, r->seed, audio, n);
  }
  st_free(spk.owned_ref);
  return rc;
}

typedef struct {
  steertts_chunk_ex_fn fn;
  steertts_chunk_fn plain;
  void* user;
} stream_ctx;

static int stream_sink(void* user, const float* y, size_t n, const steertts_chunk_info* info) {
  stream_ctx* s = (stream_ctx*)user;
  return (s->fn ? s->fn(y, n, info, s->user) : s->plain(y, n, s->user)) != 0;   /* > 0: stop */
}

static int stream_req(steertts_engine* e, req_t* r, stream_ctx* sc) {
  if (resolve_prompt(e, r)) return -1;   /* SteerTTS.stream: resolve, then the word_shift check */
  if (r->n_ws) {
    st_seterr("stream() does not support word_shift (segments re-index words); use inline markup such as "
              "<high>word</high> or *word*, or synthesize() for a whole-utterance word_shift");
    return -1;
  }
  spk_t spk;
  if (speaker_of(e, r, &spk)) {
    st_free(spk.owned_ref);
    return -1;
  }
  int rc = run_segments(e, r, &spk, stream_sink, sc);
  st_free(spk.owned_ref);
  return rc;
}

/* ------------------------------------------------------------------ public entry points */
#define NEED(cond, msg)            \
  do {                             \
    if (!(cond)) {                 \
      st_seterr("%s", msg);        \
      copy_err(err, err_len);      \
      return -1;                   \
    }                              \
  } while (0)

int steertts_synthesize(steertts_engine* e, const char* text, const steertts_options* o, float** audio, size_t* n,
                        char* err, size_t err_len) {
  NEED(audio && n, "steertts_synthesize: audio and n must not be NULL");
  *audio = NULL;
  *n = 0;
  NEED(e, "steertts_synthesize: engine is NULL");
  NEED(text, "text must be a non-empty string");
  req_t r;
  int rc = req_from_options(e, text, o, &r, 0);
  if (rc == 0) rc = synth_req(e, &r, audio, n);
  req_free(&r);
  if (rc) copy_err(err, err_len);
  return rc;
}

static int stream_entry(steertts_engine* e, const char* text, const char* json, const steertts_options* o,
                        stream_ctx* sc, char* err, size_t err_len) {
  NEED(e, "steertts_stream: engine is NULL");
  NEED(sc->fn || sc->plain, "steertts_stream: the chunk callback is NULL");
  NEED(text || json, json ? "" : "text must be a non-empty string");
  req_t r;
  int rc = json ? req_from_json(e, json, &r, 1) : req_from_options(e, text, o, &r, 1);
  if (rc == 0) rc = stream_req(e, &r, sc);
  req_free(&r);
  if (rc) copy_err(err, err_len);
  return rc;
}

int steertts_stream(steertts_engine* e, const char* text, const steertts_options* o, steertts_chunk_fn fn, void* user,
                    char* err, size_t err_len) {
  stream_ctx sc = {NULL, fn, user};
  return stream_entry(e, text ? text : NULL, NULL, o, &sc, err, err_len);
}

int steertts_stream_ex(steertts_engine* e, const char* text, const steertts_options* o, steertts_chunk_ex_fn fn,
                       void* user, char* err, size_t err_len) {
  stream_ctx sc = {fn, NULL, user};
  return stream_entry(e, text, NULL, o, &sc, err, err_len);
}

int steertts_synthesize_json(steertts_engine* e, const char* json, float** audio, size_t* n, char* err,
                             size_t err_len) {
  NEED(audio && n, "steertts_synthesize_json: audio and n must not be NULL");
  *audio = NULL;
  *n = 0;
  NEED(e, "steertts_synthesize_json: engine is NULL");
  NEED(json, "a request is a JSON object");
  req_t r;
  int rc = req_from_json(e, json, &r, 0);
  if (rc == 0) rc = synth_req(e, &r, audio, n);
  req_free(&r);
  if (rc) copy_err(err, err_len);
  return rc;
}

int steertts_stream_json(steertts_engine* e, const char* json, steertts_chunk_ex_fn fn, void* user, char* err,
                         size_t err_len) {
  NEED(json, "a request is a JSON object");
  stream_ctx sc = {fn, NULL, user};
  return stream_entry(e, NULL, json, NULL, &sc, err, err_len);
}

static void sb_ctrl(sbuf* b, int n, const int* dims, const double* vals) {
  sb_str(b, "{");
  for (int i = 0; i < n; ++i) {
    char rep[48];
    py_float(rep, sizeof rep, vals[i], 0);
    sb_fmt(b, "%s\"%s\": %s", i ? ", " : "", k_controls[dims[i]], rep);
  }
  sb_str(b, "}");
}

int steertts_resolve_json(steertts_engine* e, const char* json, char** out, char* err, size_t err_len) {
  NEED(out, "steertts_resolve_json: out must not be NULL");
  *out = NULL;
  NEED(e, "steertts_resolve_json: engine is NULL");
  NEED(json, "a request is a JSON object");
  req_t r;
  int rc = req_from_json(e, json, &r, 0);
  if (rc == 0) rc = resolve_prompt(e, &r);
  if (rc == 0) {
    sbuf b = {0};
    char rep[48];
    sb_str(&b, "{\"text\": ");
    sb_json_str(&b, r.text);
    if (r.voice) sb_str(&b, ", \"voice\": "), sb_json_str(&b, r.voice);
    if (r.ref_wav) sb_str(&b, ", \"ref_wav\": "), sb_json_str(&b, r.ref_wav);
    if (r.spk_vec) {
      sb_str(&b, ", \"spk_vec\": [");
      for (int i = 0; i < 512; ++i) {
        if (i) sb_str(&b, ", ");
        sb_num(&b, r.spk_vec[i]);
      }
      sb_str(&b, "]");
    }
    sb_str(&b, ", \"tag\": ");
    sb_json_str(&b, r.tag ? r.tag : "none");
    sb_str(&b, ", \"ctrl\": ");
    sb_ctrl(&b, r.n_ctrl, r.ctrl_dim, r.ctrl_val);
    py_float(rep, sizeof rep, r.pace, 0);
    sb_fmt(&b, ", \"pace\": %s", rep);
    py_float(rep, sizeof rep, r.guidance, 0);
    sb_fmt(&b, ", \"guidance\": %s", rep);
    if (!e->edge) sb_fmt(&b, ", \"flow\": %s, \"seed\": %lld", r.flow ? "true" : "false", (long long)r.seed);
    sb_fmt(&b, ", \"markup\": %s", r.markup ? "true" : "false");
    if (r.n_ws) {
      sb_str(&b, ", \"word_shift\": {");
      for (int i = 0; i < r.n_ws; ++i) {   /* only the keys the request gave, as normalize_request */
        const char* key[3] = {"pitch", "energy", "dur"};
        float val[3] = {r.ws[i].pitch, r.ws[i].energy, r.ws[i].dur};
        sb_fmt(&b, "%s\"%d\": {", i ? ", " : "", r.ws[i].word);
        int first = 1;
        for (int k = 0; k < 3; ++k)
          if (r.ws[i].has & (1 << k)) {
            char rep[48];
            float_short(rep, sizeof rep, val[k]);
            sb_fmt(&b, "%s\"%s\": %s", first ? "" : ", ", key[k], rep);
            first = 0;
          }
        sb_str(&b, "}");
      }
      sb_str(&b, "}");
    }
    if (r.mapped) {
      sb_str(&b, ", \"prompt_mapping\": {\"tag\": ");
      sb_json_str(&b, r.map_tag);
      sb_str(&b, ", \"ctrl\": ");
      sb_ctrl(&b, r.map_n_ctrl, r.map_dim, r.map_val);
      sb_str(&b, ", \"top_tags\": [");
      for (int i = 0; i < r.map_n_top; ++i) {
        py_float(rep, sizeof rep, r.map_top_p[i], 0);
        sb_fmt(&b, "%s[", i ? ", " : "");
        sb_json_str(&b, r.map_top_tag[i]);
        sb_fmt(&b, ", %s]", rep);
      }
      sb_str(&b, "]}");
    }
    sb_str(&b, "}");
    *out = sb_take(&b);
    if (!*out) rc = -1;
  }
  req_free(&r);
  if (rc) copy_err(err, err_len);
  return rc;
}

int steertts_map_prompt(steertts_engine* e, const char* prompt, char** out, char* err, size_t err_len) {
  NEED(out, "steertts_map_prompt: out must not be NULL");
  *out = NULL;
  NEED(e, "steertts_map_prompt: engine is NULL");
  req_t r;
  memset(&r, 0, sizeof r);
  r.prompt = st_strdup(prompt ? prompt : "");
  int rc = 0;
  if (!prompt || !*prompt) {
    st_seterr("prompt must be a non-empty string");
    rc = -1;
  }
  if (rc == 0) rc = resolve_prompt(e, &r);
  if (rc == 0) {
    sbuf b = {0};
    char rep[48];
    sb_str(&b, "{\"tag\": ");
    sb_json_str(&b, r.map_tag);
    sb_str(&b, ", \"ctrl\": ");
    sb_ctrl(&b, r.map_n_ctrl, r.map_dim, r.map_val);
    sb_str(&b, ", \"top_tags\": [");
    for (int i = 0; i < r.map_n_top; ++i) {
      py_float(rep, sizeof rep, r.map_top_p[i], 0);
      sb_fmt(&b, "%s[", i ? ", " : "");
      sb_json_str(&b, r.map_top_tag[i]);
      sb_fmt(&b, ", %s]", rep);
    }
    sb_str(&b, "]}");
    *out = sb_take(&b);
    if (!*out) rc = -1;
  }
  req_free(&r);
  if (rc) copy_err(err, err_len);
  return rc;
}

int steertts_speaker_vector(steertts_engine* e, const char* path, const float* samples, size_t n, int sr,
                            float out[512], char* err, size_t err_len) {
  int rc;
  if (path) {
    spk_t s;
    memset(&s, 0, sizeof s);
    rc = ref_vector(e, path, &s);
    if (rc == 0) memcpy(out, s.vec, sizeof(float) * 512);
    st_free(s.owned_ref);
  } else {
    rc = need_wavlm(e);
    float* y = NULL;
    float* cp = NULL;
    size_t ny = n;
    if (rc == 0 && (!samples || n == 0)) {
      st_seterr("steertts_speaker_vector: no samples");
      rc = -1;
    }
    if (rc == 0 && !(cp = (float*)st_malloc(sizeof(float) * n))) rc = -1;
    if (rc == 0) {
      memcpy(cp, samples, sizeof(float) * n);
      rc = ref_prepare(cp, &ny, sr);
      samples = cp;
    }
    if (rc == 0 && sr != SR) rc = st_resample(samples, ny, sr, SR, &y, &ny);
    if (rc == 0) rc = st_wavlm_xvector(e->wavlm, e->pool, y ? y : samples, ny, out);
    st_free(y);
    st_free(cp);
  }
  if (rc) copy_err(err, err_len);
  return rc;
}

int steertts_phonemes(steertts_engine* e, const char* text, int** ids, int** widx, size_t* n, char* err,
                      size_t err_len) {
  st_phonemes ph;
  if (st_front_phonemes(e->front, text, &ph)) {
    copy_err(err, err_len);
    return -1;
  }
  *ids = (int*)st_malloc(sizeof(int) * (size_t)(ph.n + 1));
  *widx = (int*)st_malloc(sizeof(int) * (size_t)(ph.n + 1));
  if (!*ids || !*widx) {
    st_phonemes_free(&ph);
    copy_err(err, err_len);
    return -1;
  }
  memcpy(*ids, ph.ids, sizeof(int) * (size_t)ph.n);
  memcpy(*widx, ph.widx, sizeof(int) * (size_t)ph.n);
  *n = (size_t)ph.n;
  st_phonemes_free(&ph);
  return 0;
}

void steertts_free(void* p) { st_free(p); }
void steertts_free_audio(float* audio) { st_free(audio); }

int steertts_save_wav(const float* audio, size_t n, const char* path) { return st_wav_write(path, audio, n, SR); }

int steertts_load_wav(const char* path, float** audio, size_t* n, int* sr, char* err, size_t err_len) {
  if (st_wav_read(path, audio, n, sr)) {
    copy_err(err, err_len);
    return -1;
  }
  return 0;
}
