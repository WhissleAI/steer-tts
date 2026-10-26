/* The director: plans against the trained model's (golden/director/plans.txt, from the 8-bit PyTorch model on
 * held-out chapters), sentence-by-sentence readings equal to whole-text planning, the ctrl string in a comma-decimal
 * locale, pause bounds, and NULL arguments. Needs director.bin in the edge model directory
 * ($STEERTTS_EDGE_MODEL, default ../models/steer-tts-edge); skipped without it. */
#include <locale.h>
#include <math.h>
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

#define MAXS 4096

int main(void) {
  const char* model = getenv("STEERTTS_EDGE_MODEL");
  if (!model) model = "../models/steer-tts-edge";
  char err[512];
  steertts_director* d = steertts_director_load(model, err, sizeof err);
  if (!d) {
    printf("skip: %s\n", err);
    return 77;
  }
  FILE* f = fopen("golden/director/plans.txt", "r");
  CHECK(f != NULL, "golden plans present");
  if (!f) return 1;

  static steertts_sentence s[MAXS];
  static char* texts[MAXS];
  static float want[MAXS * STEERTTS_PLAN_DIMS], got[MAXS * STEERTTS_PLAN_DIMS];
  char line[8192];
  int n = 0, genre = -1, chapters = 0, sentences = 0;
  double worst = 0;
  for (;;) {
    char* l = fgets(line, sizeof line, f);
    if (!l || strncmp(line, "chapter ", 8) == 0) {
      if (n) {   /* plan the chapter just read, and compare */
        if (steertts_director_plan(d, s, (size_t)n, steertts_director_genre(genre), got, err, sizeof err) != 0) {
          printf("FAIL plan: %s\n", err);
          fails++;
        }
        for (int i = 0; i < n * STEERTTS_PLAN_DIMS; i++) {
          double e = fabs((double)got[i] - (double)want[i]);
          if (e > worst) worst = e;
        }
        /* a reading fed one sentence at a time gives the same plans */
        steertts_reading* r = steertts_reading_begin(d, steertts_director_genre(genre));
        float p[STEERTTS_PLAN_DIMS];
        double diff = 0;
        for (int i = 0; i < n; i++) {
          steertts_reading_next(r, &s[i], i + 1 < n ? &s[i + 1] : NULL, p);
          for (int k = 0; k < STEERTTS_PLAN_DIMS; k++) diff = fmax(diff, fabs(p[k] - got[i * STEERTTS_PLAN_DIMS + k]));
        }
        steertts_reading_free(r);
        if (diff > 1e-6) {
          printf("FAIL reading differs from plan by %g\n", diff);
          fails++;
        }
        for (int i = 0; i < n; i++) free(texts[i]);
        sentences += n;
        chapters++;
        n = 0;
      }
      if (!l) break;
      genre = atoi(line + 8);
      continue;
    }
    if (n >= MAXS) continue;
    char* tab = strchr(line, '\t');
    if (!tab) continue;
    *tab = 0;
    char* text = tab + 1;
    text[strcspn(text, "\n")] = 0;
    float* w = want + n * STEERTTS_PLAN_DIMS;
    sscanf(line, "%d %d %d %d %f %f %f %f %f %f", &s[n].dialogue, &s[n].has_speaker, &s[n].para_len, &s[n].para_index,
           &w[0], &w[1], &w[2], &w[3], &w[4], &w[5]);
    texts[n] = strdup(text);
    s[n].text = texts[n];
    n++;
  }
  fclose(f);
  printf("     %d chapters, %d sentences, largest difference %.2e\n", chapters, sentences, worst);
  CHECK(sentences > 100 && worst < 1e-3, "plans match the trained model");

  /* the ctrl string ignores a comma-decimal locale */
  const char* loc = setlocale(LC_ALL, "de_DE.UTF-8");
  if (!loc) loc = setlocale(LC_ALL, "fr_FR.UTF-8");
  float plan[STEERTTS_PLAN_DIMS] = {0.5f, -1.25f, 4.0f, 0, 1, 0.3f};
  char ctrl[160];
  CHECK(steertts_plan_ctrl(NULL, plan, 1.0f, ctrl, sizeof ctrl) == 0, "ctrl string fits");
  CHECK(strcmp(ctrl, "rate=0.50,loud=-1.25,f0_med=3.00,f0_iqr=0.00,f0_end=0.80") == 0, ctrl);
  CHECK(steertts_plan_ctrl(NULL, plan, 1.0f, ctrl, 10) != 0 && ctrl[0] == 0, "a short buffer is an error, left empty");
  setlocale(LC_ALL, "C");
  float z[5], zc[5];
  float small[STEERTTS_PLAN_DIMS] = {0, 0, 0.5f, 0, 0, 0};
  steertts_plan_controls(NULL, small, 1.0f, z);
  steertts_plan_controls(d, small, 1.0f, zc);
  CHECK(z[2] == 0.5f && zc[2] >= 0.5f, "the voice's calibration asks for at least the planned pitch change");
  steertts_plan_controls(d, plan, 0.0f, z);
  CHECK(z[0] == 0 && z[2] == 0 && z[4] == 0, "strength 0 turns the plan off");
  int p1 = steertts_plan_pause_ms(plan, 1.0f), p2 = steertts_plan_pause_ms(plan, 2.0f);
  CHECK(p1 > p2 && p2 >= 80, "pauses shorten with pace, never under 80 ms");
  float huge[STEERTTS_PLAN_DIMS] = {0, 0, 0, 0, 0, 9};
  CHECK(steertts_plan_pause_ms(huge, 1.0f) == 1500, "pauses top out at 1.5 s");

  /* NULLs are errors, not crashes */
  CHECK(steertts_director_load(NULL, err, sizeof err) == NULL, "NULL path");
  CHECK(steertts_director_from_memory("nope", 4, err, sizeof err) == NULL && strstr(err, "STDR"), err);
  CHECK(steertts_director_plan(NULL, s, 1, NULL, got, err, sizeof err) != 0, "NULL director");
  CHECK(steertts_reading_next(NULL, NULL, NULL, plan) != 0, "NULL reading");
  CHECK(steertts_director_genre(0) && !strcmp(steertts_director_genre(9), "poetry") && !steertts_director_genre(10),
        "genre names");
  steertts_director_free(NULL);
  steertts_reading_free(NULL);
  steertts_director_free(d);
  printf("%s\n", fails ? "FAILED" : "all passed");
  return fails != 0;
}
