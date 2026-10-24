/* Golden parity test + benchmark for the C front end (src/front).
 *
 *   test_front [golden_dir]
 * golden_dir defaults to ../../native/tests/front (run from c/tests); assets from $STEERTTS_FRONT_ASSETS or the
 * local edge release. Checks normalize / to_phonemes (ids, word_idx, errors) on golden.jsonl "ph" lines, parse_markup
 * on "mk" lines (default known tags = steertts-v1-edge config.json "tags"), the GRU on golden_g2p.jsonl, the v1
 * <style prompt> callback path, and concurrent use of one st_front. Exit 0 = all exact, 77 = skipped. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <pthread.h>
#endif

#include "st_base.h"
#include "st_front.h"
#include "st_front_int.h"

static const char* DEFAULT_ASSETS =
    "../models/steer-tts-edge/native/front";
static const char* TAGS[] = {
    "none", "adoration", "amazement", "amusement", "anger", "anxiety", "apologetic", "bored", "confidence",
    "confusion", "contempt", "contentment", "cuteness", "desire", "determined", "disappointment", "disgust",
    "distress", "ecstasy", "embarrassment", "emphasis", "enunciated", "excited", "fast", "fear", "frustration",
    "guilt", "happiness", "hesitant", "high_pitch", "interest", "irritation", "laughing", "loud", "low_pitch",
    "mysterious", "neutral", "nostalgia", "pain", "panic", "pride", "realization", "relief", "sadness", "sarcasm",
    "serenity", "skeptical", "slow", "sympathy", "tired", "whisper"};
#define N_TAGS ((int)(sizeof TAGS / sizeof TAGS[0]))

static int shown = 0;
static void report(const char* what, const char* text, const char* detail) {
  if (shown++ >= 12) return;
  printf("MISMATCH %-8s text=\"%.160s\"\n    %.600s\n", what, text, detail);
}

typedef struct {
  long n, ok;
} stat;

static int file_exists(const char* p) {
  FILE* f = fopen(p, "rb");
  if (!f) return 0;
  fclose(f);
  return 1;
}

/* iterate the lines of a whole file buffer */
static char* next_line(char** cur) {
  char* s = *cur;
  if (!*s) return NULL;
  char* e = strchr(s, '\n');
  if (e) {
    *e = 0;
    *cur = e + 1;
  } else {
    *cur = s + strlen(s);
  }
  return s;
}

static int same_ints(const int* a, int n, const st_json* b) {
  if (!b || b->type != ST_JARR || b->n != n) return 0;
  for (int k = 0; k < n; ++k)
    if ((double)a[k] != b->items[k].num) return 0;
  return 1;
}

/* compare one segment list with the golden "segs"; returns NULL or a reason */
static const char* cmp_segs(const st_segment* s, int n, const st_json* py, char* why, size_t wn) {
  if (py->n != n) {
    snprintf(why, wn, "segment count py=%d c=%d", py->n, n);
    return why;
  }
  for (int k = 0; k < n; ++k) {
    const st_json* ps = st_json_at(py, k);
    const st_segment* cs = &s[k];
    int pause = (int)st_json_num(st_json_get(ps, "pause_ms"), 0);
    const char* event = st_json_str(st_json_get(ps, "event"), "");
    int want_skip = pause == -1 || *event;
    if (strcmp(cs->text, st_json_str(st_json_get(ps, "text"), "")))
      snprintf(why, wn, "seg %d: text py=\"%s\" c=\"%s\"", k, st_json_str(st_json_get(ps, "text"), ""), cs->text);
    else if (strcmp(cs->tag, st_json_str(st_json_get(ps, "tag"), "")))
      snprintf(why, wn, "seg %d: tag py=%s c=%s", k, st_json_str(st_json_get(ps, "tag"), ""), cs->tag);
    else if (cs->pause_ms != pause || cs->skip != want_skip || (*event && strcmp(event, "<style-tuple>")))
      snprintf(why, wn, "seg %d: pause_ms/skip py=%d/%s c=%d/%d", k, pause, event, cs->pause_ms, cs->skip);
    else {
      const st_json* ctrl = st_json_get(ps, "ctrl");
      const st_json* ws = st_json_get(ps, "ws");
      if (ctrl->n != cs->n_ctrl) {
        snprintf(why, wn, "seg %d: ctrl size", k);
        return why;
      }
      for (int c = 0; c < cs->n_ctrl; ++c) {
        const st_json* kv = st_json_at(ctrl, c);
        if (strcmp(cs->ctrl_name[c], st_json_str(st_json_at(kv, 0), "")) ||
            cs->ctrl_val[c] != (float)st_json_num(st_json_at(kv, 1), 0)) {
          snprintf(why, wn, "seg %d: ctrl %d", k, c);
          return why;
        }
      }
      if (ws->n != cs->n_ws) {
        snprintf(why, wn, "seg %d: word_shift keys py=%d c=%d", k, ws->n, cs->n_ws);
        return why;
      }
      for (int c = 0; c < cs->n_ws; ++c) {
        const st_json* w = st_json_at(ws, c);
        const st_word_shift* x = &cs->ws[c];
        if (x->word != (int)st_json_num(st_json_at(w, 0), -9) ||
            x->pitch != (float)st_json_num(st_json_at(w, 1), 0) ||
            x->energy != (float)st_json_num(st_json_at(w, 2), 0) ||
            x->dur != (float)st_json_num(st_json_at(w, 3), 0)) {
          snprintf(why, wn, "seg %d: word_shift[%d]", k, x->word);
          return why;
        }
      }
      continue;
    }
    return why;
  }
  return NULL;
}

/* ------------------------------------------------------------------ v1 <style prompt> callback (stub mapper) */
static int g_calls;
static int stub_mapper(void* user, const char* prompt, char** tag, int* n, char*** names, float** vals) {
  ++g_calls;
  if (!strcmp(prompt, "fail")) {
    st_seterr("stub mapper failed");
    return -1;
  }
  *tag = st_strdup(!strcmp(prompt, "angry and fast") ? "anger" : "none");
  *n = 3;
  *names = st_malloc(3 * sizeof(char*));
  *vals = st_malloc(3 * sizeof(float));
  (*names)[0] = st_strdup("speed");
  (*names)[1] = st_strdup("arousal");
  (*names)[2] = st_strdup("speed");  /* duplicate: the later value wins */
  (*vals)[0] = 1.0f;
  (*vals)[1] = 0.5f;
  (*vals)[2] = 1.25f;
  return 0;
}

static int test_prompt_callback(st_front* f) {
  int bad = 0, n;
  st_segment* s;
  const char* d_names[] = {"pitch"};
  float d_vals[] = {2.0f};
  /* expectations checked against steertts/markup.py (v1) with the same stub as tts.prompt_mapper */
  g_calls = 0;
  int rc = st_front_parse_markup(f, "Hello there. <style prompt=\"angry and fast\">Get out now!</style> Bye.",
                                 "whisper", d_names, d_vals, 1, TAGS, N_TAGS, stub_mapper, NULL, &s, &n);
  if (rc || n != 3 || g_calls != 1 || strcmp(s[0].text, "Hello there.") || strcmp(s[0].tag, "whisper") ||
      s[0].n_ctrl != 1 || strcmp(s[1].text, "Get out now!") || strcmp(s[1].tag, "anger") || s[1].n_ctrl != 2 ||
      strcmp(s[1].ctrl_name[0], "arousal") || s[1].ctrl_val[0] != 0.5f || strcmp(s[1].ctrl_name[1], "speed") ||
      s[1].ctrl_val[1] != 1.25f || strcmp(s[2].text, "Bye.") || strcmp(s[2].tag, "whisper")) {
    printf("MISMATCH prompt callback: basic span (rc=%d n=%d calls=%d)\n", rc, n, g_calls);
    bad = 1;
  }
  if (!rc) st_segments_free(s, n);
  /* empty prompt: ("none", {}) without a call; nested named tag inside a prompt span */
  g_calls = 0;
  rc = st_front_parse_markup(f, "<style prompt=\"\">A b.</style><style prompt=\"calm\">C <sadness>d</sadness></style>",
                             NULL, NULL, NULL, 0, TAGS, N_TAGS, stub_mapper, NULL, &s, &n);
  if (rc || g_calls != 1 || n != 3 || strcmp(s[0].tag, "none") || s[0].n_ctrl || strcmp(s[1].text, "C") ||
      strcmp(s[1].tag, "none") || s[1].n_ctrl != 2 || strcmp(s[2].text, "d") || strcmp(s[2].tag, "sadness") ||
      s[2].n_ctrl) {
    printf("MISMATCH prompt callback: empty prompt / nesting (rc=%d n=%d calls=%d)\n", rc, n, g_calls);
    bad = 1;
  }
  if (!rc) st_segments_free(s, n);
  /* callback error propagates; NULL callback = edge ValueError */
  rc = st_front_parse_markup(f, "<style prompt=\"fail\">x</style>", NULL, NULL, NULL, 0, TAGS, N_TAGS, stub_mapper,
                             NULL, &s, &n);
  if (rc != -1 || strcmp(st_geterr(), "stub mapper failed") || s || n) {
    printf("MISMATCH prompt callback: error path (rc=%d err=%s)\n", rc, st_geterr());
    bad = 1;
  }
  rc = st_front_parse_markup(f, "<style prompt=\"x\">x</style>", NULL, NULL, NULL, 0, TAGS, N_TAGS, NULL, NULL, &s,
                             &n);
  if (rc != -1 || strncmp(st_geterr(), "ValueError: free-text prompts", 29)) {
    printf("MISMATCH prompt callback: NULL callback (rc=%d err=%s)\n", rc, st_geterr());
    bad = 1;
  }
  printf("%-12s %8d %8d\n", "prompt_cb", 4, bad ? 0 : 4);
  return bad;
}

/* ------------------------------------------------------------------ concurrency */
typedef struct {
  st_front* f;
  const char** texts;
  const st_phonemes* ref;
  int n, start, bad;
} thr_arg;
static void* thr_main(void* p) {
  thr_arg* a = (thr_arg*)p;
  for (int k = 0; k < a->n; ++k) {
    int i = (k + a->start) % a->n;
    st_phonemes ph;
    if (st_front_phonemes(a->f, a->texts[i], &ph)) {
      a->bad++;
      continue;
    }
    if (ph.n != a->ref[i].n || memcmp(ph.ids, a->ref[i].ids, (size_t)ph.n * sizeof(int)) ||
        memcmp(ph.widx, a->ref[i].widx, (size_t)ph.n * sizeof(int)))
      a->bad++;
    st_phonemes_free(&ph);
  }
  return NULL;
}

int main(int argc, char** argv) {
  const char* gdir = argc > 1 ? argv[1] : "../../native/tests/front";
  const char* assets = getenv("STEERTTS_FRONT_ASSETS");
  char probe[1024];
  if (!assets || !*assets) assets = DEFAULT_ASSETS;
  snprintf(probe, sizeof probe, "%s/unicode.bin", assets);
  char golden[1024], golden_g2p[1024];
  snprintf(golden, sizeof golden, "%s/golden.jsonl", gdir);
  snprintf(golden_g2p, sizeof golden_g2p, "%s/golden_g2p.jsonl", gdir);
  if (!file_exists(probe) || !file_exists(golden)) {
    printf("test_front: SKIP (no front assets at %s or no golden file at %s)\n", assets, golden);
    return 77;
  }
  double t0 = st_time_ms();
  st_front* f = st_front_load(assets);
  if (!f) {
    printf("test_front: load failed: %s\n", st_geterr());
    return 1;
  }
  printf("front end loaded in %.1f ms\n", st_time_ms() - t0);

  char* buf;
  size_t len;
  if (st_read_file(golden, &buf, &len)) {
    printf("cannot read %s\n", golden);
    return 1;
  }
  stat s_norm = {0, 0}, s_ids = {0, 0}, s_widx = {0, 0}, s_err = {0, 0}, s_mk = {0, 0}, s_mkerr = {0, 0};
  double t_ph = 0, t_mk = 0;
  long n_ph = 0;
  const char** texts = NULL;   /* non-error ph texts (benchmark / thread test) */
  st_phonemes* refs = NULL;
  int ntexts = 0, captexts = 0;
  st_json** keep = NULL;       /* parsed lines owning texts[] */
  int nkeep = 0;
  char why[1024], detail[4096];
  char* cur = buf;
  for (char* line; (line = next_line(&cur)) != NULL;) {
    if (!*line) continue;
    st_json* j = st_json_parse(line, strlen(line));
    if (!j) {
      printf("bad JSON line: %s\n", st_geterr());
      return 1;
    }
    const char* k = st_json_str(st_json_get(j, "k"), "");
    const char* text = st_json_str(st_json_get(j, "text"), "");
    int keep_j = 0;
    if (!strcmp(k, "ph")) {
      /* normalize */
      char* got = st_front_normalize(f, text);
      const st_json* ne = st_json_get(j, "norm_error");
      int ok = ne ? (!got && !strcmp(st_geterr(), ne->str))
                  : (got && !strcmp(got, st_json_str(st_json_get(j, "norm"), "")));
      s_norm.n++;
      s_norm.ok += ok;
      if (!ok) {
        snprintf(detail, sizeof detail, "py=\"%s\"\n    c =\"%s\"",
                 ne ? ne->str : st_json_str(st_json_get(j, "norm"), ""), got ? got : st_geterr());
        report("norm", text, detail);
      }
      st_free(got);
      /* to_phonemes */
      st_phonemes p;
      double a = st_time_ms();
      int rc = st_front_phonemes(f, text, &p);
      t_ph += st_time_ms() - a;
      ++n_ph;
      const st_json* pe = st_json_get(j, "error");
      if (pe) {
        ok = rc == -1 && !strcmp(st_geterr(), pe->str);
        s_err.n++;
        s_err.ok += ok;
        if (!ok) {
          snprintf(detail, sizeof detail, "py=%s\n    c =%s", pe->str, rc ? st_geterr() : "<no error>");
          report("error", text, detail);
        }
        st_phonemes_free(&p);
        st_json_free(j);
        continue;
      }
      int ok_ids = !rc && same_ints(p.ids, p.n, st_json_get(j, "ids"));
      int ok_w = !rc && same_ints(p.widx, p.n, st_json_get(j, "widx"));
      s_ids.n++;
      s_ids.ok += ok_ids;
      s_widx.n++;
      s_widx.ok += ok_w;
      if (!ok_ids || !ok_w) {
        int o = snprintf(detail, sizeof detail, "py =");
        const st_json* ids = st_json_get(j, "ids");
        for (int i = 0; i < ids->n && o < 2000; ++i)
          o += snprintf(detail + o, sizeof detail - o, "%s ", st_front_symbol(f, (int)ids->items[i].num));
        o += snprintf(detail + o, sizeof detail - o, "\n    c  =");
        if (rc) o += snprintf(detail + o, sizeof detail - o, "%s", st_geterr());
        for (int i = 0; i < p.n && o < 4000; ++i)
          o += snprintf(detail + o, sizeof detail - o, "%s ", st_front_symbol(f, p.ids[i]));
        report(ok_ids ? "word_idx" : "ids", text, detail);
      }
      if (ntexts == captexts) {
        captexts = captexts ? 2 * captexts : 1024;
        texts = realloc((void*)texts, (size_t)captexts * sizeof *texts);
        refs = realloc(refs, (size_t)captexts * sizeof *refs);
        keep = realloc(keep, (size_t)captexts * sizeof *keep);
      }
      texts[ntexts] = text;
      refs[ntexts++] = p;
      keep[nkeep++] = j;
      keep_j = 1;
    } else if (!strcmp(k, "mk")) {
      const st_json* tg = st_json_get(j, "tag");
      const char* tag = tg && tg->type == ST_JSTR ? tg->str : NULL;
      const st_json* ctrl = st_json_get(j, "ctrl");
      int nc = ctrl && ctrl->type == ST_JARR ? ctrl->n : 0;
      const char* names[64];
      float vals[64];
      for (int c = 0; c < nc && c < 64; ++c) {
        names[c] = st_json_str(st_json_at(st_json_at(ctrl, c), 0), "");
        vals[c] = (float)st_json_num(st_json_at(st_json_at(ctrl, c), 1), 0);
      }
      st_segment* segs;
      int nseg;
      double a = st_time_ms();
      int rc = st_front_parse_markup(f, text, tag, names, vals, nc, TAGS, N_TAGS, NULL, NULL, &segs, &nseg);
      t_mk += st_time_ms() - a;
      s_mk.n++;
      const st_json* pe = st_json_get(j, "error");
      if (pe) {
        int ok = rc == -1 && !strcmp(st_geterr(), pe->str);
        s_mk.ok += ok;
        s_mkerr.n++;
        s_mkerr.ok += ok;
        if (!ok) {
          snprintf(detail, sizeof detail, "py=%s\n    c =%s", pe->str, rc ? st_geterr() : "<no error>");
          report("mk-error", text, detail);
        }
      } else {
        const char* r = rc ? (snprintf(why, sizeof why, "c raised %s", st_geterr()), why)
                           : cmp_segs(segs, nseg, st_json_get(j, "segs"), why, sizeof why);
        s_mk.ok += r == NULL;
        if (r) report("markup", text, r);
      }
      if (!rc) st_segments_free(segs, nseg);
    }
    if (!keep_j) st_json_free(j);
  }

  /* GRU G2P words */
  stat s_g2p = {0, 0};
  double t_g2p = 0;
  char* gbuf;
  size_t glen;
  if (!st_read_file(golden_g2p, &gbuf, &glen)) {
    cur = gbuf;
    for (char* line; (line = next_line(&cur)) != NULL;) {
      if (!*line) continue;
      st_json* j = st_json_parse(line, strlen(line));
      if (!j) return 1;
      int out[20];
      double gap, a = st_time_ms();
      int n = gru_predict_raw(&f->g2p.gru, st_json_str(st_json_get(j, "w"), ""), out, &gap);
      t_g2p += st_time_ms() - a;
      int ok = same_ints(out, n, st_json_get(j, "p"));
      s_g2p.n++;
      s_g2p.ok += ok;
      if (!ok) {
        snprintf(detail, sizeof detail, "py gap=%g c gap=%g", st_json_num(st_json_get(j, "gap"), 0), gap);
        report("g2p", st_json_str(st_json_get(j, "w"), ""), detail);
      }
      st_json_free(j);
    }
    st_free(gbuf);
  }

  printf("\n%-12s %8s %8s %9s\n", "field", "cases", "exact", "rate");
  struct {
    const char* name;
    stat* s;
  } rows[] = {{"normalize", &s_norm}, {"ids", &s_ids},     {"word_idx", &s_widx}, {"errors(ph)", &s_err},
              {"markup", &s_mk},      {"errors(mk)", &s_mkerr}, {"gru_g2p", &s_g2p}};
  int all = 1;
  for (size_t r = 0; r < sizeof rows / sizeof rows[0]; ++r) {
    stat* s = rows[r].s;
    printf("%-12s %8ld %8ld %8.3f%%\n", rows[r].name, s->n, s->ok, 100.0 * s->ok / (s->n ? s->n : 1));
    all = all && s->ok == s->n && s->n > 0;
  }
  if (test_prompt_callback(f)) all = 0;

  /* benchmark: a fresh front end (cold G2P cache), then the same texts again (warm cache) */
  st_front* fb = st_front_load(assets);
  double cold = 0, warm = 0;
  size_t chars = 0;
  for (int pass = 0; fb && pass < 2; ++pass) {
    double a = st_time_ms();
    for (int i = 0; i < ntexts; ++i) {
      st_phonemes p;
      if (!st_front_phonemes(fb, texts[i], &p)) st_phonemes_free(&p);
    }
    *(pass ? &warm : &cold) = st_time_ms() - a;
  }
  for (int i = 0; i < ntexts; ++i) chars += strlen(texts[i]);
  st_front_free(fb);
  printf("\ntiming: to_phonemes %.1f us/case in the test loop, parse_markup %.1f us/case, GRU %.1f us/word\n",
         1e3 * t_ph / (n_ph ? n_ph : 1), 1e3 * t_mk / (s_mk.n ? s_mk.n : 1), 1e3 * t_g2p / (s_g2p.n ? s_g2p.n : 1));
  printf("bench: %d texts (avg %.1f bytes): cold cache %.1f us/text (%.0f texts/s), warm cache %.1f us/text "
         "(%.0f texts/s)\n",
         ntexts, (double)chars / (ntexts ? ntexts : 1), 1e3 * cold / ntexts, ntexts / (cold / 1e3),
         1e3 * warm / ntexts, ntexts / (warm / 1e3));

#ifndef _WIN32
  /* one st_front from 4 threads at once (shared G2P cache) */
  {
    pthread_t th[4];
    thr_arg ta[4];
    st_front* ft = st_front_load(assets);
    for (int t = 0; ft && t < 4; ++t) {
      ta[t] = (thr_arg){ft, texts, refs, ntexts, t * ntexts / 4, 0};
      pthread_create(&th[t], NULL, thr_main, &ta[t]);
    }
    int tbad = 0;
    for (int t = 0; ft && t < 4; ++t) {
      pthread_join(th[t], NULL);
      tbad += ta[t].bad;
    }
    st_front_free(ft);
    printf("%-12s %8d %8d\n", "threads(4)", 4 * ntexts, 4 * ntexts - tbad);
    if (tbad || !ft) all = 0;
  }
#endif
  printf("peak RSS %zu kB\n", st_peak_rss_kb());

  for (int i = 0; i < ntexts; ++i) st_phonemes_free(&refs[i]);
  for (int i = 0; i < nkeep; ++i) st_json_free(keep[i]);
  free((void*)texts);
  free(refs);
  free(keep);
  st_free(buf);
  st_front_free(f);
  printf(all ? "ALL EXACT\n" : "MISMATCHES PRESENT\n");
  return all ? 0 : 1;
}
