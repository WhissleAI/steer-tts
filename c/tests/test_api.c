/* Public-API robustness: comma-decimal locale, NULL arguments, output reset on failure, error messages that name the
 * cause, streaming stop / errors. Uses the edge model ($STEERTTS_EDGE_MODEL, default ../models/steer-tts-edge). */
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "steertts.h"

static int fails;
#define CHECK(cond, what)                                    \
  do {                                                       \
    int ok_ = (cond);                                        \
    printf("%s %s\n", ok_ ? "ok  " : "FAIL", what);          \
    fails += !ok_;                                           \
  } while (0)

static int count_cb(const float* a, size_t n, void* u) {
  (void)a;
  (void)n;
  return ++*(int*)u >= 2;   /* stop after two chunks */
}

int main(void) {
  const char* model = getenv("STEERTTS_EDGE_MODEL");
  if (!model) model = "../models/steer-tts-edge";
  char probe[1024];
  snprintf(probe, sizeof probe, "%s/config.json", model);
  FILE* f = fopen(probe, "rb");
  if (!f) {
    printf("skip: no edge model at %s\n", model);
    return 77;
  }
  fclose(f);
  char err[512];

  /* a comma-decimal locale must not change parsing (config.json, JSON requests, ctrl strings) */
  const char* loc = setlocale(LC_ALL, "de_DE.UTF-8");
  if (!loc) loc = setlocale(LC_ALL, "en_DK.utf8");
  if (!loc) loc = setlocale(LC_ALL, "fr_FR.UTF-8");
  printf("locale: %s\n", loc ? loc : "(none available, C locale)");
  steertts_engine* e = steertts_create(model, 2, err, sizeof err);
  CHECK(e != NULL, "create under the locale");
  if (!e) return 1;
  float* y = (float*)1;
  size_t n = 12345;
  CHECK(steertts_synthesize_json(e, "{\"text\": \"Hello there.\", \"pace\": 1.25, \"ctrl\": {\"rate\": 0.5}}", &y, &n,
                                 err, sizeof err) == 0 && n > 0,
        "JSON request with fractions under the locale");
  steertts_free(y);
  steertts_options o;
  steertts_default_options(&o);
  o.ctrl = "rate=1.5,f0_iqr=-0.25";
  CHECK(steertts_synthesize(e, "Hello there.", &o, &y, &n, err, sizeof err) == 0, "ctrl string with fractions");
  steertts_free(y);
  char* r = NULL;
  CHECK(steertts_resolve_json(e, "{\"text\": \"x\", \"pace\": 1.25}", &r, err, sizeof err) == 0 && r &&
            strstr(r, "\"pace\": 1.25"),
        "resolve prints '.' decimals under the locale");
  steertts_free(r);
  setlocale(LC_ALL, "C");

  /* outputs are reset before validation can fail */
  y = (float*)1;
  n = 12345;
  CHECK(steertts_synthesize_json(e, "{\"text\": \"x\", \"pace\": 9}", &y, &n, err, sizeof err) != 0 && y == NULL &&
            n == 0,
        "outputs reset on a failed request");
  CHECK(!strcmp(err, "pace=9 is outside [0.25, 4.0]"), "Python's message for an out-of-range pace");

  /* NULL arguments are errors, not crashes */
  CHECK(steertts_synthesize(NULL, "x", NULL, &y, &n, err, sizeof err) != 0, "NULL engine");
  CHECK(steertts_synthesize(e, "x", NULL, NULL, NULL, err, sizeof err) != 0, "NULL outputs");
  CHECK(steertts_synthesize(e, NULL, NULL, &y, &n, err, sizeof err) != 0, "NULL text");
  CHECK(steertts_synthesize_json(e, NULL, &y, &n, err, sizeof err) != 0, "NULL JSON");
  CHECK(steertts_stream(e, "x", NULL, NULL, NULL, err, sizeof err) != 0, "NULL callback");
  CHECK(steertts_resolve_json(e, "{\"text\": \"x\"}", NULL, err, sizeof err) != 0, "NULL resolve output");
  CHECK(steertts_has(e, NULL) == 0 && steertts_model_type(NULL) == NULL && steertts_num_voices(NULL) == 0,
        "NULL-tolerant getters");
  /* per-gender defaults: each names a voice of that gender; the default voice is the female one when it is female */
  for (int gi = 0; gi < 2; ++gi) {
    const char* g = gi ? "male" : "female";
    const char* v = steertts_gender_voice(e, g);
    int k = -1;
    for (int i = 0; v && i < steertts_num_voices(e); ++i)
      if (!strcmp(steertts_voice(e, i), v)) k = i;
    CHECK(k >= 0 && !strcmp(steertts_voice_gender(e, k), g), "gender default voice has that gender");
  }
  for (int i = 0; i < steertts_num_voices(e); ++i)
    if (!strcmp(steertts_voice(e, i), steertts_default_voice(e)) && steertts_voice_gender(e, i) &&
        !strcmp(steertts_voice_gender(e, i), "female"))
      CHECK(!strcmp(steertts_gender_voice(e, "female"), steertts_default_voice(e)), "female default = default voice");
  CHECK(steertts_gender_voice(e, "other") == NULL && steertts_gender_voice(NULL, "male") == NULL &&
            steertts_voice_gender(e, -1) == NULL && steertts_voice_gender(e, steertts_num_voices(e)) == NULL,
        "gender getters reject bad input");
  /* resizing the pool keeps the model and renders the same audio */
  {
    float *y1 = NULL, *y3 = NULL;
    size_t n1 = 0, n3 = 0;
    int ok = steertts_set_threads(e, 1) == 0 && steertts_threads(e) == 1 &&
             steertts_synthesize(e, "Hello there.", NULL, &y1, &n1, err, sizeof err) == 0 &&
             steertts_set_threads(e, 3) == 0 && steertts_threads(e) == 3 &&
             steertts_synthesize(e, "Hello there.", NULL, &y3, &n3, err, sizeof err) == 0;
    CHECK(ok && n1 == n3 && n1 > 0 && !memcmp(y1, y3, n1 * sizeof(float)), "set_threads: same audio at 1 and 3 threads");
    steertts_free(y1);
    steertts_free(y3);
    CHECK(steertts_set_threads(NULL, 2) != 0 && steertts_threads(NULL) == 0, "set_threads: NULL engine");
  }
  CHECK(steertts_synthesize(e, "x", NULL, &y, &n, NULL, 0) == 0, "err == NULL is allowed");
  steertts_free(y);

  /* streaming: the callback can stop it; nothing-to-say markup gives no audio (Python EdgeTTS: 0 samples) */
  int chunks = 0;
  CHECK(steertts_stream(e, "One. Two. Three. Four.", NULL, count_cb, &chunks, err, sizeof err) == 0 && chunks == 2,
        "stream stops when the callback returns non-zero");
  CHECK(steertts_synthesize_json(e, "{\"text\": \"...\", \"markup\": true}", &y, &n, err, sizeof err) == 0 && n == 0,
        "markup with nothing to say renders 0 samples (edge)");
  steertts_free(y);

  /* errors name their cause */
  steertts_engine* bad = steertts_create("/nonexistent/model", 1, err, sizeof err);
  CHECK(!bad && strstr(err, "/nonexistent/model") != NULL, "a missing model dir is named in the error");
  steertts_config c;
  steertts_default_config(&c);
  c.assets_dir = "/nonexistent/assets";
  bad = steertts_create_ex(model, &c, err, sizeof err);
  CHECK(!bad && strstr(err, "/nonexistent/assets") != NULL && !strstr(err, "assets: text front-end assets"),
        "a missing assets dir is named once in the error");
  CHECK(steertts_synthesize_json(e, "{\"text\": \"x\", \"spk_vec\": [1, 2]}", &y, &n, err, sizeof err) != 0 &&
            !strcmp(err, "spk_vec must be a list of 512 numbers"),
        "spk_vec is validated");
  steertts_destroy(e);
  printf(fails ? "%d FAILED\n" : "all ok\n", fails);
  return fails != 0;
}
