/* End-to-end parity: full requests through the public API against the Python packages' renders
 * (tests/make_golden_engine.py). Model dirs: $STEERTTS_V1_MODEL (v1, with its native/ assets) and
 * $STEERTTS_EDGE_MODEL; a model that is not available is skipped. Cases that need optional assets (prompt: MiniLM,
 * ref_wav: WavLM) are skipped when the assets are missing. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "st_base.h"
#include "steertts.h"

static void dump(const st_json* j, char** buf, size_t* n, size_t* cap, const char* ref_dir);

static void put(char** buf, size_t* n, size_t* cap, const char* s) {
  size_t l = strlen(s);
  if (*n + l + 1 > *cap) {
    while (*n + l + 1 > *cap) *cap = *cap ? *cap * 2 : 1024;
    *buf = (char*)realloc(*buf, *cap);
  }
  memcpy(*buf + *n, s, l + 1);
  *n += l;
}

static void put_str(char** buf, size_t* n, size_t* cap, const char* s) {
  put(buf, n, cap, "\"");
  for (; *s; ++s) {
    char c[8] = {*s, 0};
    if (*s == '"' || *s == '\\') c[0] = '\\', c[1] = *s, c[2] = 0;
    put(buf, n, cap, c);
  }
  put(buf, n, cap, "\"");
}

static void dump(const st_json* j, char** buf, size_t* n, size_t* cap, const char* ref_dir) {
  char tmp[64];
  switch (j->type) {
    case ST_JNULL: put(buf, n, cap, "null"); break;
    case ST_JBOOL: put(buf, n, cap, j->num ? "true" : "false"); break;
    case ST_JNUM:
      if (j->is_int) snprintf(tmp, sizeof tmp, "%lld", (long long)j->num);
      else snprintf(tmp, sizeof tmp, "%.17g%s", j->num, j->num == floor(j->num) ? ".0" : "");
      put(buf, n, cap, tmp);
      break;
    case ST_JSTR: put_str(buf, n, cap, j->str); break;
    case ST_JARR:
      put(buf, n, cap, "[");
      for (int i = 0; i < j->n; ++i) {
        if (i) put(buf, n, cap, ", ");
        dump(&j->items[i], buf, n, cap, ref_dir);
      }
      put(buf, n, cap, "]");
      break;
    case ST_JOBJ:
      put(buf, n, cap, "{");
      for (int i = 0; i < j->n; ++i) {
        if (i) put(buf, n, cap, ", ");
        put_str(buf, n, cap, j->keys[i]);
        put(buf, n, cap, ": ");
        if (!strcmp(j->keys[i], "ref_wav") && j->items[i].type == ST_JSTR) {
          char p[1024];
          snprintf(p, sizeof p, "%s/%s", ref_dir, j->items[i].str);
          put_str(buf, n, cap, p);
        } else
          dump(&j->items[i], buf, n, cap, ref_dir);
      }
      put(buf, n, cap, "}");
      break;
  }
}

static double snr_db(const float* ref, const float* y, size_t n) {
  double s = 0, e = 0;
  for (size_t i = 0; i < n; ++i) {
    s += (double)ref[i] * ref[i];
    e += ((double)ref[i] - y[i]) * ((double)ref[i] - y[i]);
  }
  return e == 0 ? 200 : 10 * log10(s / e);
}

/* returns failures; *skipped counts skipped cases */
static int run(const char* model, const char* precision, const char* golden, int threads, double min_snr,
               double min_snr_noflow, double min_snr_clone, int* checked, int* skipped) {
  st_file* f = st_file_open(golden);
  if (!f) {
    printf("FAIL %s: %s\n", golden, st_geterr());
    return 1;
  }
  const char* meta = st_file_metadata(f, "requests");
  st_json* reqs = meta ? st_json_parse(meta, strlen(meta)) : NULL;
  steertts_config c;
  steertts_default_config(&c);
  c.threads = threads;
  c.precision = precision;
  char err[1024];
  double t0 = st_time_ms();
  steertts_engine* e = steertts_create_ex(model, &c, err, sizeof err);
  if (!e || !reqs) {
    printf("FAIL %s: %s\n", golden, e ? "bad metadata" : err);
    st_file_close(f);
    st_json_free(reqs);
    steertts_destroy(e);
    return 1;
  }
  printf("%s [%s, %s]: loaded in %.0f ms\n", steertts_model_type(e), precision, golden, st_time_ms() - t0);
  int fails = 0;
  char dir[1024];
  snprintf(dir, sizeof dir, "%s", golden);
  char* slash = strrchr(dir, '/');
  if (slash) *slash = 0;
  for (int i = 0; i < reqs->n; ++i) {
    const st_json* r = &reqs->items[i];
    int clone = st_json_get(r, "ref_wav") != NULL;
    if ((st_json_get(r, "prompt") && !steertts_has(e, "prompt")) || (clone && !steertts_has(e, "clone"))) {
      printf("skip case %d (assets for %s missing)\n", i, clone ? "cloning" : "prompts");
      ++*skipped;
      continue;
    }
    char* js = NULL;
    size_t n = 0, cap = 0;
    dump(r, &js, &n, &cap, dir);
    const st_tensor* ref = st_file_find(f, st_name("case%d", i));
    float* y = NULL;
    size_t ny = 0;
    double t1 = st_time_ms();
    int rc = steertts_synthesize_json(e, js, &y, &ny, err, sizeof err);
    double ms = st_time_ms() - t1;
    size_t nr = ref ? (size_t)st_numel(ref) : 0;
    float* rf = (float*)malloc(sizeof(float) * (nr + 1));
    if (ref) memcpy(rf, ref->data, sizeof(float) * nr);
    double snr = rc == 0 && ny == nr ? snr_db(rf, y, nr) : -1;
    /* v1 renders with the flow refiner: PyTorch itself agrees only to 50-60 dB across its own thread counts on guided
     * requests (the ODE amplifies summation-order differences), so those get a lower bar than flow=false renders */
    const st_json* fl = st_json_get(r, "flow");
    int flow = !strcmp(steertts_model_type(e), "steer-tts") && !(fl && fl->type == ST_JBOOL && !fl->num);
    double need = clone ? min_snr_clone : flow ? min_snr : min_snr_noflow;
    int ok = rc == 0 && ny == nr && (nr == 0 || snr >= need);
    printf("%s case %d: %zu vs %zu samples, SNR %.1f dB (need %.0f), %.0f ms  %s%s\n", ok ? "ok  " : "FAIL", i, ny, nr,
           snr, need, ms, js, rc ? err : "");
    fails += !ok;
    ++*checked;
    free(rf);
    free(js);
    steertts_free(y);
  }
  steertts_destroy(e);
  st_json_free(reqs);
  st_file_close(f);
  return fails;
}

/* renders must not depend on the thread count (every reduction keeps its order however work is split) */
static int determinism(const char* model, const char* json) {
  float* y[2] = {NULL, NULL};
  size_t n[2] = {0, 0};
  char err[512];
  int threads[2] = {1, 3};
  for (int k = 0; k < 2; ++k) {
    steertts_engine* e = steertts_create(model, threads[k], err, sizeof err);
    if (!e || steertts_synthesize_json(e, json, &y[k], &n[k], err, sizeof err)) {
      printf("FAIL determinism %s: %s\n", model, err);
      steertts_destroy(e);
      return 1;
    }
    steertts_destroy(e);
  }
  int same = n[0] == n[1] && !memcmp(y[0], y[1], sizeof(float) * n[0]);
  printf("%s determinism: 1 vs 3 threads %s (%s)\n", same ? "ok  " : "FAIL", same ? "bit-identical" : "differ", model);
  steertts_free(y[0]);
  steertts_free(y[1]);
  return !same;
}

int main(void) {
  const char* v1 = getenv("STEERTTS_V1_MODEL");
  const char* edge = getenv("STEERTTS_EDGE_MODEL");
  if (!v1) v1 = "../models/steer-tts";
  if (!edge) edge = "../models/steer-tts-edge";
  int fails = 0, checked = 0, skipped = 0;
  char probe[1200];
  snprintf(probe, sizeof probe, "%s/config.json", v1);
  FILE* p = fopen(probe, "rb");
  if (p) {
    fclose(p);
    /* the refiner's ODE and the cloning path (resampler, x-vector) amplify float differences: thresholds per path */
    fails += run(v1, "fp32", "golden/engine/v1_fp32.safetensors", 4, 30, 60, 25, &checked, &skipped);
    fails += run(v1, "int8", "golden/engine/v1_int8.safetensors", 4, 30, 60, 25, &checked, &skipped);
    fails += determinism(v1, "{\"text\": \"It is over.\", \"tag\": \"sadness\", \"guidance\": 1.5}");
  } else
    printf("skip v1 (no model at %s)\n", v1);
  snprintf(probe, sizeof probe, "%s/config.json", edge);
  p = fopen(probe, "rb");
  if (p) {
    fclose(p);
    fails += run(edge, "fp32", "golden/engine/edge.safetensors", 1, 60, 60, 60, &checked, &skipped);
    fails += determinism(edge, "{\"text\": \"Well, *that* was fast. [pause 300ms] <whisper>Quiet.</whisper>\", "
                               "\"markup\": true}");
  } else
    printf("skip edge (no model at %s)\n", edge);
  if (!checked) {
    printf("nothing to test\n");
    return 77;
  }
  printf("%d checked, %d skipped, %d failed\n", checked, skipped, fails);
  return fails != 0;
}
