/* steertts: command line for the stand-alone C library (both models; same flags as the Python / native CLIs).
 *
 *   steertts "Hello there." -o hello.wav --model steer-tts-edge
 *   steertts "I can't believe it worked." --tag happiness --ctrl rate=1,f0_iqr=1.5 -o happy.wav
 *   steertts "Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow>" --markup -o markup.wav
 *   steertts "First sentence. Second one." --stream -o - | aplay -f S16_LE -r 24000 -c 1
 *   steertts "See you soon." --prompt "excited, a bit faster" -o s.wav           (v1)
 *   steertts "Now in your voice." --ref-wav me.wav -o clone.wav                   (v1)
 *   steertts --json '{"text": "Hi.", "ctrl": {"f0_end": 2}}' -o hi.wav
 *   steertts --list voices | tags | controls
 *   steertts --bench [--bench-json]
 *
 * The model directory is --model, else $STEERTTS_MODEL, else $STEERTTS_EDGE_MODEL, else the current directory. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "steertts.h"

static const char* usage =
    "usage: steertts [TEXT | -] [-o OUT.wav | -o -] [--model DIR] [--voice NAME] [--tag TAG] [--ctrl name=z,...]\n"
    "                [--rate Z] [--pace X] [--guidance S] [--markup] [--stream] [--threads N]\n"
    "                [--prompt TEXT] [--ref-wav WAV] [--word-shift W:key=v,...;...] [--seed N] [--no-flow]\n"
    "                [--flow-steps N] [--precision fp32|int8] [--assets DIR] [--json REQUEST] [--resolve]\n"
    "                [--list voices|tags|controls] [--bench] [--bench-json] [--version]\n"
    "\n"
    "  TEXT           text to speak ('-' reads stdin)\n"
    "  -o OUT         output WAV (16-bit, 24 kHz; default out.wav). '-' writes raw 16-bit PCM to stdout\n"
    "  --markup       parse inline markup (*emphasis*, [pause 300ms], <slow>, <whisper> ...)\n"
    "  --stream       render sentence by sentence (implies markup), writing each chunk as soon as it is ready\n"
    "  --guidance S   style guidance scale (1.0 = off)\n"
    "  --pace X       global duration divisor (1.2 = 20% faster)\n"
    "  --threads N    worker threads (default 1, 0 = all CPUs)\n"
    "  --prompt T     v1: free-text direction, mapped to a style tag + controls\n"
    "  --ref-wav W    v1: clone the voice of a WAV file\n"
    "  --no-flow      v1: skip the flow refiner (~12x less compute)\n"
    "  --precision P  v1: fp32 (default) or int8 weights\n"
    "  --json R       a full request as JSON (request.schema.json); TEXT and the request flags are then ignored\n"
    "  --resolve      print the resolved request (prompt mapped, defaults filled in) instead of rendering\n"
    "  --bench        render 20 fixed sentences, print real-time factor, latency and peak memory\n"
    "  --bench-json   the same, as one JSON line\n";

/* strict number arguments: the whole string must parse (C locale, as the library) */
double st_strtod(const char* s, char** end);
static double num_arg(const char* flag, const char* v) {
  char* end;
  double x = st_strtod(v, &end);
  if (end == v || *end) {
    fprintf(stderr, "steertts: %s needs a number, got '%s'\n", flag, v);
    exit(2);
  }
  return x;
}

static void write_pcm(const float* y, size_t n) {
  short buf[4096];
  for (size_t i = 0; i < n;) {
    size_t k = 0;
    for (; k < 4096 && i < n; ++k, ++i) {
      float v = y[i] < -1.f ? -1.f : y[i] > 1.f ? 1.f : y[i];
      buf[k] = (short)lrintf(v * 32767.0f);
    }
    fwrite(buf, 2, k, stdout);
  }
  fflush(stdout);
}

/* Same sentences and probes as edge/tools/bench_speed.py and the native CLI, so numbers compare across runtimes. */
static const char* bench_texts[] = {
    "The weather was calm when we left the harbour this morning.",
    "Please remember to bring the documents to the meeting on Thursday.",
    "I never thought a small garden could grow so many tomatoes.",
    "She opened the letter slowly and read it twice before speaking.",
    "Our train was delayed by forty minutes because of the snow.",
    "He laughed, shook his head, and walked back into the kitchen.",
    "Thanks for calling. How can I help you today?",
    "The museum opens at nine and closes at half past five.",
    "If you turn left at the bakery, you will see the station.",
    "Nobody expected the quiet student to win the debate.",
    "Could you send me the report before lunch?",
    "The river had risen overnight, and the old bridge was closed.",
    "We should probably leave now if we want to catch the last bus.",
    "Her grandmother kept every letter in a wooden box under the bed.",
    "That was the best soup I have had in years.",
    "The engineers tested the new design for three weeks before approving it.",
    "Wait, did you hear that noise outside?",
    "A small dog followed us all the way home from the park.",
    "The price of coffee went up again this month.",
    "When the lights came back on, everyone started clapping."};
static const char* bench_short = "Hello, how are you?";

typedef struct {
  double t0, first;
  FILE* raw;
  float* y;
  size_t n, cap;
} sink;

double now_ms(void);

static int first_only(const float* a, size_t n, void* u) {
  (void)a;
  (void)n;
  sink* s = (sink*)u;
  if (s->first < 0) s->first = now_ms() - s->t0;
  return 1;   /* stop after the first chunk */
}

static int collect(const float* a, size_t n, void* u) {
  sink* s = (sink*)u;
  if (s->first < 0) s->first = now_ms() - s->t0;
  if (s->raw) {
    write_pcm(a, n);
    return 0;
  }
  if (s->n + n > s->cap) {
    size_t c = s->cap ? s->cap * 2 : 48000;
    while (c < s->n + n) c *= 2;
    float* t = (float*)realloc(s->y, sizeof(float) * c);
    if (!t) {
      fprintf(stderr, "steertts: out of memory collecting the stream\n");
      exit(2);
    }
    s->y = t;
    s->cap = c;
  }
  memcpy(s->y + s->n, a, sizeof(float) * n);
  s->n += n;
  return 0;
}

static int cmp_d(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return x < y ? -1 : x > y;
}

/* peak RSS in MB from the library's helper (via a tiny re-declaration: the CLI links the static library) */
size_t st_peak_rss_kb(void);
double st_time_ms(void);
double now_ms(void) { return st_time_ms(); }

static int bench(steertts_engine* e, const steertts_options* base, double load_s, int threads, int as_json) {
  char err[512];
  float* y;
  size_t n;
  steertts_options o = *base;
  for (int i = 0; i < 2; ++i) {
    if (steertts_synthesize(e, bench_texts[i], &o, &y, &n, err, sizeof err)) {
      fprintf(stderr, "steertts: %s\n", err);
      return 2;
    }
    steertts_free(y);
  }
  double ttfa[5], first[5], per[20], synth = 0, audio = 0;
  for (int i = 0; i < 5; ++i) {
    double t0 = now_ms();
    if (steertts_synthesize(e, bench_short, &o, &y, &n, err, sizeof err)) {
      fprintf(stderr, "steertts: %s\n", err);
      return 2;
    }
    ttfa[i] = (now_ms() - t0) / 1000;
    steertts_free(y);
  }
  for (int i = 0; i < 5; ++i) {
    sink s = {now_ms(), -1, NULL, NULL, 0, 0};
    if (steertts_stream(e, "Hello, how are you? And a second sentence follows.", &o, first_only, &s, err, sizeof err)) {
      fprintf(stderr, "steertts: %s\n", err);
      return 2;
    }
    first[i] = s.first / 1000;
  }
  for (int i = 0; i < 20; ++i) {
    double t0 = now_ms();
    if (steertts_synthesize(e, bench_texts[i], &o, &y, &n, err, sizeof err)) {
      fprintf(stderr, "steertts: %s\n", err);
      return 2;
    }
    double d = (now_ms() - t0) / 1000, a = (double)n / STEERTTS_SAMPLE_RATE;
    synth += d;
    audio += a;
    per[i] = d / a;
    steertts_free(y);
  }
  qsort(ttfa, 5, sizeof(double), cmp_d);
  qsort(first, 5, sizeof(double), cmp_d);
  qsort(per, 20, sizeof(double), cmp_d);
  double rss = st_peak_rss_kb() / 1024.0;
  if (as_json)
    printf("{\"threads\": %d, \"load_s\": %.3f, \"rtf\": %.4f, \"rtf_median\": %.4f, \"audio_s\": %.2f, "
           "\"synth_s\": %.3f, \"ttfa_short_s\": %.3f, \"ttfa_stream_first_chunk_s\": %.3f, \"peak_rss_mb\": %.1f, "
           "\"runtime\": \"c\", \"model\": \"%s\", \"version\": \"%s\"}\n",
           threads, load_s, synth / audio, per[10], audio, synth, ttfa[2], first[2], rss, steertts_model_type(e),
           steertts_version());
  else
    printf("%s, threads %d: load %.2f s, RTF %.4f (median %.4f) over %.1f s of audio, short sentence %.0f ms, "
           "first streamed chunk %.0f ms, peak RSS %.0f MB\n",
           steertts_model_type(e), threads, load_s, synth / audio, per[10], audio, ttfa[2] * 1000, first[2] * 1000, rss);
  return 0;
}

static char* read_stdin(void) {
  size_t cap = 4096, n = 0;
  char* s = (char*)malloc(cap);
  int c;
  while (s && (c = getchar()) != EOF) {
    if (n + 1 >= cap) {
      char* t = (char*)realloc(s, cap *= 2);
      if (!t) {
        free(s);
        return NULL;
      }
      s = t;
    }
    s[n++] = (char)c;
  }
  if (s) s[n] = 0;
  return s;
}

int main(int argc, char** argv) {
  const char *text = NULL, *out = "out.wav", *model = NULL, *list = NULL, *json = NULL, *ctrl = NULL;
  char ctrl_buf[512];
  double rate = 0;
  int have_rate = 0, stream = 0, do_bench = 0, bench_json = 0, resolve = 0;
  steertts_config cfg;
  steertts_options o;
  steertts_default_config(&cfg);
  steertts_default_options(&o);
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (fprintf(stderr, "steertts: %s needs a value\n", a), exit(2), ""))
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      fputs(usage, stdout);
      return 0;
    } else if (!strcmp(a, "--version")) {
      printf("steertts %s\n", steertts_version());
      return 0;
    } else if (!strcmp(a, "-o") || !strcmp(a, "--out")) out = NEXT();
    else if (!strcmp(a, "--model")) model = NEXT();
    else if (!strcmp(a, "--voice")) o.voice = NEXT();
    else if (!strcmp(a, "--tag")) o.tag = NEXT();
    else if (!strcmp(a, "--ctrl")) ctrl = NEXT();
    else if (!strcmp(a, "--rate")) rate = num_arg(a, NEXT()), have_rate = 1;
    else if (!strcmp(a, "--pace")) o.pace = (float)num_arg(a, NEXT());
    else if (!strcmp(a, "--guidance")) {
      o.guidance = (float)num_arg(a, NEXT());
      if (o.guidance < 0) {
        fprintf(stderr, "steertts: --guidance must be in [0, 4]\n");
        return 2;
      }
    }
    else if (!strcmp(a, "--markup")) o.markup = 1;
    else if (!strcmp(a, "--stream")) stream = 1;
    else if (!strcmp(a, "--threads")) cfg.threads = (int)num_arg(a, NEXT());
    else if (!strcmp(a, "--prompt")) o.prompt = NEXT();
    else if (!strcmp(a, "--ref-wav")) o.ref_wav = NEXT();
    else if (!strcmp(a, "--word-shift")) o.word_shift = NEXT();
    else if (!strcmp(a, "--seed")) o.seed = (int64_t)num_arg(a, NEXT());
    else if (!strcmp(a, "--no-flow")) cfg.flow = 0;
    else if (!strcmp(a, "--flow-steps")) cfg.flow_steps = (int)num_arg(a, NEXT());
    else if (!strcmp(a, "--precision")) cfg.precision = NEXT();
    else if (!strcmp(a, "--assets")) cfg.assets_dir = NEXT();
    else if (!strcmp(a, "--json")) json = NEXT();
    else if (!strcmp(a, "--resolve")) resolve = 1;
    else if (!strcmp(a, "--list")) list = NEXT();
    else if (!strcmp(a, "--bench")) do_bench = 1;
    else if (!strcmp(a, "--bench-json")) do_bench = bench_json = 1;
    else if (a[0] == '-' && a[1]) {
      fprintf(stderr, "steertts: unknown option %s\n", a);
      return 2;
    } else
      text = a;
  }
  if (have_rate) {
    char rb[40];
    snprintf(rb, sizeof rb, "%.17g", rate);
    for (char* q = rb; *q; ++q)
      if (*q == ',') *q = '.';   /* a comma-decimal locale */
    snprintf(ctrl_buf, sizeof ctrl_buf, "%s%srate=%s", ctrl ? ctrl : "", ctrl ? "," : "", rb);
    ctrl = ctrl_buf;
  }
  o.ctrl = ctrl;
  if (!model) model = getenv("STEERTTS_MODEL");
  if (!model) model = getenv("STEERTTS_EDGE_MODEL");
  if (!model) model = ".";
  char err[1024];
  double t0 = now_ms();
  steertts_engine* e = steertts_create_ex(model, &cfg, err, sizeof err);
  if (!e) {
    fprintf(stderr, "steertts: %s\n", err);
    return 2;
  }
  double load = (now_ms() - t0) / 1000;
  int rc = 0;
  if (list) {
    if (!strcmp(list, "voices"))
      for (int i = 0; i < steertts_num_voices(e); ++i) printf("%s\n", steertts_voice(e, i));
    else if (!strcmp(list, "tags"))
      for (int i = 0; i < steertts_num_tags(e); ++i) printf("%s\n", steertts_tag(e, i));
    else if (!strcmp(list, "controls"))
      for (int i = 0; i < steertts_num_controls(); ++i) printf("%s\n", steertts_control(i));
    else {
      fprintf(stderr, "steertts: --list takes voices, tags or controls\n");
      rc = 2;
    }
    steertts_destroy(e);
    return rc;
  }
  if (do_bench) {
    rc = bench(e, &o, load, cfg.threads, bench_json);
    steertts_destroy(e);
    return rc;
  }
  char* owned = NULL;
  if (!json && !text) {
    fputs(usage, stderr);
    steertts_destroy(e);
    return 2;
  }
  if (text && !strcmp(text, "-")) text = owned = read_stdin();
  if (resolve) {
    char* r = NULL;
    char tmp[64];
    if (!json) {
      fprintf(stderr, "steertts: --resolve needs --json\n");
      rc = 2;
    } else if (steertts_resolve_json(e, json, &r, err, sizeof err)) {
      fprintf(stderr, "steertts: %s\n", err);
      rc = 2;
    } else {
      printf("%s\n", r);
      steertts_free(r);
    }
    (void)tmp;
    steertts_destroy(e);
    free(owned);
    return rc;
  }
  sink s = {now_ms(), -1, !strcmp(out, "-") ? stdout : NULL, NULL, 0, 0};
  float* y = NULL;
  size_t n = 0;
  if (stream) {
    if (json) {
      fprintf(stderr, "steertts: use --json without --stream, or the stream flags without --json\n");
      rc = 2;
    } else if (steertts_stream(e, text, &o, collect, &s, err, sizeof err)) {
      fprintf(stderr, "steertts: %s\n", err);
      rc = 2;
    }
    y = s.y;
    n = s.n;
  } else {
    int r = json ? steertts_synthesize_json(e, json, &y, &n, err, sizeof err)
                 : steertts_synthesize(e, text, &o, &y, &n, err, sizeof err);
    if (r) {
      fprintf(stderr, "steertts: %s\n", err);
      rc = 2;
    } else if (s.raw)
      write_pcm(y, n);
  }
  double synth = (now_ms() - s.t0) / 1000;
  if (rc == 0 && !s.raw) {
    if (steertts_save_wav(y, n, out)) {
      fprintf(stderr, "steertts: cannot write %s\n", out);
      rc = 2;
    } else {
      double dur = (double)n / STEERTTS_SAMPLE_RATE;
      fprintf(stderr, "%s: %.2f s audio, load %.2f s, synth %.2f s", out, dur, load, synth);
      if (dur > 0) fprintf(stderr, " (RTF %.3f)", synth / dur);
      if (stream && s.first >= 0) fprintf(stderr, ", first chunk %.0f ms", s.first);
      fprintf(stderr, ", peak RSS %.0f MB\n", st_peak_rss_kb() / 1024.0);
    }
  }
  if (stream) free(y);
  else steertts_free(y);
  free(owned);
  steertts_destroy(e);
  return rc;
}
