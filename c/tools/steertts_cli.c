/* steertts: command line for the stand-alone C library (both models; same flags as the Python / native CLIs).
 *
 *   steertts "Hello there." -o hello.wav --model steer-tts-edge
 *   steertts "I can't believe it worked." --tag happiness --ctrl rate=1,f0_iqr=1.5 -o happy.wav
 *   steertts "Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow>" --markup -o markup.wav
 *   steertts "First sentence. Second one." --stream -o - | aplay -f S16_LE -r 24000 -c 1
 *   steertts "See you soon." --prompt "excited, a bit faster" -o s.wav           (v1)
 *   steertts "Now in your voice." --ref-wav me.wav -o clone.wav                   (v1)
 *   steertts --json '{"text": "Hi.", "ctrl": {"f0_end": 2}}' -o hi.wav
 *   steertts "Hello there. How are you today?" --play                                 (to the speakers)
 *   steertts -i --voice NAME            (interactive: the model stays loaded, one line = one utterance, played)
 *   steertts "Good morning." --voice male --play                     (--voice female / male: that gender's default)
 *   steertts --list voices [--gender female|male] | tags | controls
 *   steertts --bench [--bench-json]
 *
 * The model directory is --model, else $STEERTTS_MODEL, else $STEERTTS_EDGE_MODEL, else the current directory.
 * Playback (--play, -i) pipes raw 16-bit PCM into $STEERTTS_PLAYER, else the first of play (sox), aplay, paplay,
 * ffplay found on PATH; on macOS without any of them each utterance is played with afplay from a temporary WAV. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L   /* popen, mkstemp, access */
#endif
#include <math.h>
#include <signal.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#define popen _popen
#define pclose _pclose
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

#include "steertts.h"

static const char* usage =
    "usage: steertts [TEXT | -] [-o OUT.wav | -o -] [--model DIR] [--voice NAME] [--tag TAG] [--ctrl name=z,...]\n"
    "                [--rate Z] [--pace X] [--guidance S] [--markup] [--stream] [--threads N]\n"
    "                [--prompt TEXT] [--ref-wav WAV] [--word-shift W:key=v,...;...] [--seed N] [--no-flow]\n"
    "                [--flow-steps N] [--precision fp32|int8] [--assets DIR] [--json REQUEST] [--resolve]\n"
    "                [--play] [-i] [--list voices|tags|controls] [--gender female|male] [--bench] [--bench-json]\n"
    "                [--perform FILE [--plan] [--genre G] [--strength S] [--director PATH]] [--version]\n"
    "\n"
    "  TEXT           text to speak ('-' reads stdin)\n"
    "  -o OUT         output WAV (16-bit, 24 kHz; default out.wav). '-' writes raw 16-bit PCM to stdout\n"
    "  --voice V      a voice name, or female / male for that gender's default voice\n"
    "  --gender G     with --list voices: only the female or male voices\n"
    "  --markup       parse inline markup (*emphasis*, [pause 300ms], <slow>, <whisper> ...)\n"
    "  --stream       render sentence by sentence (implies markup), writing each chunk as soon as it is ready\n"
    "  --play         play through the speakers instead of writing out.wav (with -o FILE: both); implies --stream\n"
    "  -i             interactive: load once, then speak each line read from stdin (implies --stream; plays unless\n"
    "                 -o -). Lines starting with / are commands: /voice NAME, /tag TAG, /ctrl k=z,..., /voices, /help\n"
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
    "  --bench-json   the same, as one JSON line\n"
    "  --perform F    read a whole text file ('-': stdin) as a performance: paragraphs at blank lines, each sentence\n"
    "                 planned by the director (rate, loudness, pitch, ending, pause) and rendered into one WAV\n"
    "  --plan         with --perform: print each sentence's plan (tab-separated) instead of rendering\n"
    "  --genre G      with --perform: children, adventure, fantasy, mystery, scifi, gothic, literary, humour,\n"
    "                 nonfiction or poetry (steadies the register; default unknown)\n"
    "  --strength S   with --perform: how strongly the plan is applied (default 1; 0 = no director)\n"
    "  --director P   director.bin, or its directory (default: the model directory)\n";

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

/* 0 on success, -1 when the reader went away (a closed pipe or player) */
static int write_pcm(FILE* f, const float* y, size_t n) {
  short buf[4096];
  for (size_t i = 0; i < n;) {
    size_t k = 0;
    for (; k < 4096 && i < n; ++k, ++i) {
      float v = y[i] < -1.f ? -1.f : y[i] > 1.f ? 1.f : y[i];
      buf[k] = (short)lrintf(v * 32767.0f);
    }
    if (fwrite(buf, 2, k, f) != k) return -1;
  }
  return fflush(f) ? -1 : 0;
}

/* ------------------------------------------------------------------ playback */
typedef struct {
  FILE* pipe;        /* a player reading raw s16le 24 kHz mono from stdin, or */
  const char* file;  /* a player taking a WAV path (macOS afplay): the audio is collected and played per utterance */
} player;

#ifndef _WIN32
static int on_path(const char* exe) {
  const char* p = getenv("PATH");
  char buf[1024];
  while (p && *p) {
    size_t k = strcspn(p, ":");
    if (k && k + strlen(exe) + 2 < sizeof buf) {
      snprintf(buf, sizeof buf, "%.*s/%s", (int)k, p, exe);
      if (access(buf, X_OK) == 0) return 1;
    }
    p += k + (p[k] == ':');
  }
  return 0;
}
#endif

static int player_open(player* pl) {
  static const char* const cmds[][2] = {
      {"play", "play -q --buffer 1024 -t raw -r 24000 -e signed -b 16 -c 1 -"},   /* small buffer: low latency */
      {"aplay", "aplay -q -t raw -f S16_LE -r 24000 -c 1"},
      {"paplay", "paplay --raw --format=s16le --rate=24000 --channels=1"},
      {"ffplay", "ffplay -loglevel error -nodisp -autoexit -f s16le -ar 24000 -i -"}};
  const char* cmd = getenv("STEERTTS_PLAYER");
  pl->pipe = NULL;
  pl->file = NULL;
#ifndef _WIN32
  signal(SIGPIPE, SIG_IGN);   /* a player that quits is a write error, not a crash */
  for (size_t i = 0; !(cmd && *cmd) && i < sizeof cmds / sizeof cmds[0]; ++i)
    if (on_path(cmds[i][0])) cmd = cmds[i][1];
  if (!(cmd && *cmd) && on_path("afplay")) {
    pl->file = "afplay";
    return 0;
  }
#endif
  if (!(cmd && *cmd)) {
    fprintf(stderr, "steertts: no audio player found: install sox (play) or alsa-utils (aplay), or set "
                    "STEERTTS_PLAYER to a command that reads raw 16-bit 24 kHz mono PCM from stdin\n");
    return -1;
  }
#ifdef _WIN32
  pl->pipe = popen(cmd, "wb");
#else
  pl->pipe = popen(cmd, "w");
#endif
  if (!pl->pipe) {
    fprintf(stderr, "steertts: cannot start the player: %s\n", cmd);
    return -1;
  }
  return 0;
}

/* Players hold back their last partial buffer until more input arrives (sox's default buffer is 340 ms): without
 * this the end of each line would only play together with the next one. 400 ms of silence pushes it out. */
static int player_flush_tail(FILE* f) {
  static const float silence[STEERTTS_SAMPLE_RATE * 2 / 5];
  return write_pcm(f, silence, sizeof silence / sizeof silence[0]);
}

/* file players: play one utterance's collected audio (returns once it has played) */
static int player_play_file(const player* pl, const float* y, size_t n) {
#ifdef _WIN32
  (void)pl, (void)y, (void)n;
  return -1;
#else
  const char* dir = getenv("TMPDIR");
  char path[512], cmd[600];
  snprintf(path, sizeof path, "%s/steertts-play-XXXXXX", dir && *dir ? dir : "/tmp");
  int fd = mkstemp(path);
  if (fd < 0) return -1;
  close(fd);
  int rc = steertts_save_wav(y, n, path);
  if (!rc) {
    snprintf(cmd, sizeof cmd, "%s '%s'", pl->file, path);
    rc = system(cmd) ? -1 : 0;
  }
  remove(path);
  return rc;
#endif
}

static void player_close(player* pl) {
  if (pl->pipe) pclose(pl->pipe);   /* returns once the queued audio has played */
  pl->pipe = NULL;
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
  int closed;   /* the raw output went away */
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
    if (write_pcm(s->raw, a, n)) {
      fprintf(stderr, "steertts: the audio output was closed\n");
      s->closed = 1;
      return 1;
    }
    s->n += n;   /* counted only */
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
    sink s = {now_ms(), -1, NULL, NULL, 0, 0, 0};
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

/* one line of stdin without its line ending (malloc'd), NULL at end of input */
static char* read_line(void) {
  size_t cap = 256, n = 0;
  char* s = (char*)malloc(cap);
  int c;
  while (s && (c = getchar()) != EOF && c != '\n') {
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
  if (!s || (c == EOF && n == 0)) {
    free(s);
    return NULL;
  }
  while (n && (s[n - 1] == '\r' || s[n - 1] == ' ' || s[n - 1] == '\t')) --n;
  s[n] = 0;
  return s;
}

/* <whisper>, </slow>, [pause 300ms], *word*: text that is read aloud literally unless markup is on */
static int looks_like_markup(const char* t) {
  if (strstr(t, "[pause")) return 1;
  for (const char* p = t; (p = strchr(p, '<')) != NULL; ++p) {
    const char* q = p + 1 + (p[1] == '/');
    if ((*q >= 'a' && *q <= 'z') && strchr(q, '>')) return 1;
  }
  const char* a = strchr(t, '*');
  return a && a[1] && a[1] != ' ' && strchr(a + 1, '*');
}

/* --voice / /voice: a voice name, or "female" / "male" for that gender's default; NULL (message printed) if unknown */
static const char* resolve_voice(const steertts_engine* e, const char* v) {
  for (int i = 0; i < steertts_num_voices(e); ++i)
    if (!strcmp(steertts_voice(e, i), v)) return v;
  if (!strcmp(v, "female") || !strcmp(v, "male")) {
    const char* g = steertts_gender_voice(e, v);
    if (!g) fprintf(stderr, "steertts: this model has no %s voice (or no gender metadata)\n", v);
    return g;
  }
  fprintf(stderr, "steertts: unknown voice '%s' (--list voices lists them; female / male pick a default)\n", v);
  return NULL;
}

static void list_voices(const steertts_engine* e, const char* gender) {
  for (int i = 0; i < steertts_num_voices(e); ++i) {
    const char* g = steertts_voice_gender(e, i);
    if (!gender || (g && !strcmp(g, gender))) printf("%s\n", steertts_voice(e, i));
  }
}

static int has_name(const char* x, int count, const char* (*get)(const steertts_engine*, int), const steertts_engine* e) {
  for (int i = 0; i < count; ++i)
    if (!strcmp(get(e, i), x)) return 1;
  return 0;
}

static const char* interactive_help =
    "  TEXT           speak it (inline markup: *emphasis*, [pause 300ms], <slow>..</slow>, <whisper>..</whisper>)\n"
    "  /voice NAME    switch voice (or female / male)   /voices [female|male]  list them\n"
    "  /tag TAG       style tag (none)      /tags      list them\n"
    "  /ctrl k=z,...  controls (none)       /controls  list them\n"
    "  /quit          or Ctrl-D\n";

/* -i: the engine stays loaded; each stdin line is streamed to raw (a player pipe or stdout) or, with a file player,
 * collected and played. A failed line is reported and the loop goes on; a closed output ends it. */
static int interactive(steertts_engine* e, steertts_options o, FILE* raw, const player* pl) {
  char voice[128], tag[128], ctrl[512], err[1024];
  int tty = isatty(fileno(stdin)), rc = 0;
  if (tty)
    fprintf(stderr, "%s, voice %s. Type a line to speak it; /voice male, /voices and /help, Ctrl-D quits.\n",
            steertts_model_type(e), o.voice ? o.voice : steertts_default_voice(e));
  for (;;) {
    if (tty) fputs("> ", stderr);
    char* line = read_line();
    if (!line) break;
    char* t = line;
    while (*t == ' ' || *t == '\t') ++t;
    if (!*t) {
      free(line);
      continue;
    }
    if (*t == '/') {
      char* arg = t + strcspn(t, " \t");
      if (*arg) *arg++ = 0;
      while (*arg == ' ' || *arg == '\t') ++arg;
      if (!strcmp(t, "/quit") || !strcmp(t, "/exit") || !strcmp(t, "/q")) {
        free(line);
        break;
      } else if (!strcmp(t, "/voices")) {
        const char* cur = o.voice ? o.voice : steertts_default_voice(e);
        int col = 0;
        for (int i = 0; i < steertts_num_voices(e); ++i) {
          const char *v = steertts_voice(e, i), *g = steertts_voice_gender(e, i);
          if (*arg && (!g || strcmp(g, arg))) continue;
          if (col + (int)strlen(v) + 4 > 100) fputc('\n', stderr), col = 0;
          col += fprintf(stderr, "%s%s %c", strcmp(v, cur) ? "  " : " *", v, g ? g[0] : ' ');
        }
        fprintf(stderr, "\n  (f = female, m = male; defaults: female %s, male %s)\n",
                steertts_gender_voice(e, "female") ? steertts_gender_voice(e, "female") : "-",
                steertts_gender_voice(e, "male") ? steertts_gender_voice(e, "male") : "-");
      } else if (!strcmp(t, "/tags")) {
        for (int i = 0; i < steertts_num_tags(e); ++i) fprintf(stderr, "%s%s", i ? ", " : "", steertts_tag(e, i));
        fputc('\n', stderr);
      } else if (!strcmp(t, "/controls")) {
        for (int i = 0; i < steertts_num_controls(); ++i) fprintf(stderr, "%s%s", i ? ", " : "", steertts_control(i));
        fputc('\n', stderr);
      } else if (!strcmp(t, "/voice") && *arg) {
        const char* v = resolve_voice(e, arg);
        if (v) {
          snprintf(voice, sizeof voice, "%s", v), o.voice = voice;
          if (strcmp(v, arg)) fprintf(stderr, "  %s\n", v);
        }
      } else if (!strcmp(t, "/tag") && *arg) {
        if (!strcmp(arg, "none"))
          o.tag = NULL;
        else if (!has_name(arg, steertts_num_tags(e), steertts_tag, e))
          fprintf(stderr, "unknown tag '%s' (/tags lists them)\n", arg);
        else
          snprintf(tag, sizeof tag, "%s", arg), o.tag = tag;
      } else if (!strcmp(t, "/ctrl")) {
        if (!*arg || !strcmp(arg, "none"))
          o.ctrl = NULL;
        else
          snprintf(ctrl, sizeof ctrl, "%s", arg), o.ctrl = ctrl;   /* checked when the next line renders */
      } else
        fputs(interactive_help, stderr);
      free(line);
      continue;
    }
    sink s = {now_ms(), -1, raw, NULL, 0, 0, 0};
    int r = steertts_stream(e, t, &o, collect, &s, err, sizeof err);
    free(line);
    if (raw && raw != stdout && !s.closed && player_flush_tail(raw)) s.closed = 1;
    if (r)
      fprintf(stderr, "steertts: %s\n", err);
    else if (!raw && player_play_file(pl, s.y, s.n))
      fprintf(stderr, "steertts: %s could not play the audio\n", pl->file);
    else if (tty && !s.closed && s.first >= 0)
      fprintf(stderr, "  (%.2f s of audio, first chunk after %.0f ms)\n", (double)s.n / STEERTTS_SAMPLE_RATE, s.first);
    free(s.y);
    if (s.closed) {
      rc = 2;
      break;
    }
  }
  return rc;
}

/* ------------------------------------------------------------------ --perform: a text read as a performance */

/* a whole file ('-': stdin), NUL-terminated */
static int read_text(const char* path, char** text, size_t* len) {
  FILE* f = strcmp(path, "-") ? fopen(path, "rb") : stdin;
  if (!f) {
    fprintf(stderr, "steertts: cannot open %s\n", path);
    return 1;
  }
  size_t cap = 1 << 16, n = 0;
  char* b = (char*)malloc(cap);
  for (size_t got; b && (got = fread(b + n, 1, cap - n - 1, f)) > 0;) {
    n += got;
    if (n + 1 >= cap) b = (char*)realloc(b, cap *= 2);
  }
  if (f != stdin) fclose(f);
  if (!b) return 1;
  b[n] = 0;
  *text = b;
  *len = n;
  return 0;
}

typedef struct {
  char** text;
  steertts_sentence* s;
  size_t n, cap;
} sentences;

static void add_sentence(sentences* v, const char* a, size_t len) {
  while (len && isspace((unsigned char)a[0])) a++, len--;
  while (len && isspace((unsigned char)a[len - 1])) len--;
  if (!len) return;
  if (v->n == v->cap) {
    v->cap = v->cap ? 2 * v->cap : 256;
    v->text = (char**)realloc(v->text, v->cap * sizeof *v->text);
    v->s = (steertts_sentence*)realloc(v->s, v->cap * sizeof *v->s);
  }
  char* t = (char*)malloc(len + 1);
  memcpy(t, a, len);
  t[len] = 0;
  for (char* q = t; *q; q++)
    if (*q == '\n' || *q == '\r' || *q == '\t') *q = ' ';
  v->text[v->n] = t;
  memset(&v->s[v->n], 0, sizeof v->s[v->n]);
  v->s[v->n].text = t;
  v->n++;
}

static int is_abbrev(const char* p, const char* start) {   /* p at the '.' */
  static const char* const ab[] = {"Mr", "Mrs", "Ms", "Dr", "St", "Mt", "Jr", "Sr", "vs", "etc", "e.g", "i.e", "No", 0};
  for (int k = 0; ab[k]; k++) {
    size_t n = strlen(ab[k]);
    if ((size_t)(p - start) >= n && strncmp(p - n, ab[k], n) == 0 && (p - n == start || !isalpha((unsigned char)p[-(long)n - 1])))
      return 1;
  }
  return p - start >= 1 && isupper((unsigned char)p[-1]) && (p - start == 1 || !isalpha((unsigned char)p[-2]));  /* "J." */
}

/* Paragraphs at blank lines; sentences at . ! ? (and closing quotes) before a space and a capital or a quote. */
static void split_text(const char* text, sentences* v) {
  const char* p = text;
  while (*p) {
    const char* end = strstr(p, "\n\n");
    while (end && end[2] == '\n') end++;
    size_t plen = end ? (size_t)(end - p) : strlen(p);
    size_t first = v->n;
    const char* a = p;
    for (const char* q = p; q < p + plen; q++) {
      if (*q != '.' && *q != '!' && *q != '?') continue;
      if (*q == '.' && is_abbrev(q, p)) continue;
      const char* e = q + 1;
      while (e < p + plen && (*e == '.' || *e == '!' || *e == '?' || *e == '"' || *e == '\'' || *e == ')' ||
                              ((unsigned char)*e == 0xE2 && (unsigned char)e[1] == 0x80 &&
                               ((unsigned char)e[2] == 0x9D || (unsigned char)e[2] == 0x99))))
        e += (unsigned char)*e == 0xE2 ? 3 : 1;
      if (e >= p + plen) break;
      if (!isspace((unsigned char)*e)) continue;
      const char* nx = e;
      while (nx < p + plen && isspace((unsigned char)*nx)) nx++;
      if (nx < p + plen && (isupper((unsigned char)*nx) || *nx == '"' || (unsigned char)*nx == 0xE2)) {
        add_sentence(v, a, (size_t)(e - a));
        a = e;
      }
    }
    add_sentence(v, a, (size_t)(p + plen - a));
    for (size_t i = first; i < v->n; i++) {
      const char* t = v->s[i].text;
      int quoted = strchr(t, '"') != NULL || strstr(t, "\xE2\x80\x9C") != NULL || strstr(t, "\xE2\x80\x9D") != NULL;
      v->s[i].dialogue = quoted;
      v->s[i].has_speaker = quoted && (strstr(t, " said") || strstr(t, " asked") || strstr(t, " cried") ||
                                       strstr(t, " replied") || strstr(t, " whispered") || strstr(t, " shouted"));
      v->s[i].para_len = (int)(v->n - first);
      v->s[i].para_index = (int)(i - first);
    }
    p += plen;
    while (*p == '\n') p++;
  }
}

static int perform(const char* model, const steertts_config* cfg, const steertts_options* base, const char* user_ctrl,
                   const char* file, const char* director_path, const char* genre, float strength, int plan_only,
                   const char* out) {
  char err[512];
  size_t len = 0;
  char* text = NULL;
  if (read_text(file, &text, &len) != 0) return 1;
  sentences v = {0};
  split_text(text, &v);
  free(text);
  if (!v.n) {
    fprintf(stderr, "steertts: no sentences in %s\n", file);
    return 1;
  }
  steertts_director* d = steertts_director_load(director_path ? director_path : model, err, sizeof err);
  if (!d) {
    fprintf(stderr, "steertts: %s\n", err);
    return 1;
  }
  float* plans = (float*)malloc(v.n * STEERTTS_PLAN_DIMS * sizeof(float));
  if (!plans || steertts_director_plan(d, v.s, v.n, genre, plans, err, sizeof err) != 0) {
    fprintf(stderr, "steertts: %s\n", err);
    return 1;
  }
  int rc = 0;
  if (plan_only) {
    printf("sentence\tdialogue\tpara\trate\tloud\tf0_med\tf0_iqr\tend\tpause_ms\ttext\n");
    for (size_t i = 0; i < v.n; i++) {
      const float* p = plans + i * STEERTTS_PLAN_DIMS;
      printf("%zu\t%d\t%d/%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d\t%s\n", i, v.s[i].dialogue, v.s[i].para_index + 1,
             v.s[i].para_len, p[0], p[1], p[2], p[3], p[4], steertts_plan_pause_ms(p, base->pace > 0 ? base->pace : 1),
             v.s[i].text);
    }
  } else {
    steertts_engine* e = steertts_create_ex(model, cfg, err, sizeof err);
    if (!e) {
      fprintf(stderr, "steertts: %s\n", err);
      return 1;
    }
    float* all = NULL;
    size_t total = 0;
    for (size_t i = 0; i < v.n && !rc; i++) {
      const float* p = plans + i * STEERTTS_PLAN_DIMS;
      char ctrl[640], pc[160];
      steertts_plan_ctrl(d, p, strength, pc, sizeof pc);
      snprintf(ctrl, sizeof ctrl, "%s%s%s", strength > 0 ? pc : "", user_ctrl && strength > 0 ? "," : "",
               user_ctrl ? user_ctrl : "");   /* the user's own controls last: they win */
      steertts_options o = *base;
      o.ctrl = ctrl[0] ? ctrl : NULL;
      float* a = NULL;
      size_t n = 0;
      if (steertts_synthesize(e, v.s[i].text, &o, &a, &n, err, sizeof err) != 0) {
        fprintf(stderr, "steertts: sentence %zu: %s\n", i, err);
        rc = 1;
        break;
      }
      size_t gap = strength > 0 ? (size_t)steertts_plan_pause_ms(p, o.pace > 0 ? o.pace : 1) * STEERTTS_SAMPLE_RATE / 1000
                                : STEERTTS_SAMPLE_RATE * 3 / 10;
      if (v.s[i].para_index == v.s[i].para_len - 1 && strength <= 0) gap = STEERTTS_SAMPLE_RATE * 6 / 10;
      all = (float*)realloc(all, (total + n + gap) * sizeof(float));
      memcpy(all + total, a, n * sizeof(float));
      memset(all + total + n, 0, gap * sizeof(float));
      total += n + gap;
      steertts_free(a);
    }
    if (!rc) {
      rc = steertts_save_wav(all, total, out);
      if (!rc) fprintf(stderr, "%s: %zu sentences, %.1f s\n", out, v.n, (double)total / STEERTTS_SAMPLE_RATE);
    }
    free(all);
    steertts_destroy(e);
  }
  for (size_t i = 0; i < v.n; i++) free(v.text[i]);
  free(v.text);
  free(v.s);
  free(plans);
  steertts_director_free(d);
  return rc;
}

int main(int argc, char** argv) {
  const char *text = NULL, *out = "out.wav", *model = NULL, *list = NULL, *json = NULL, *ctrl = NULL, *gender = NULL;
  char ctrl_buf[512];
  double rate = 0;
  int have_rate = 0, stream = 0, do_bench = 0, bench_json = 0, resolve = 0, play = 0, inter = 0, out_set = 0;
  const char *perform_file = NULL, *director_path = NULL, *genre = NULL;
  int plan_only = 0;
  float strength = 1.0f;
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
    } else if (!strcmp(a, "-o") || !strcmp(a, "--out")) out = NEXT(), out_set = 1;
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
    else if (!strcmp(a, "--play")) play = 1;
    else if (!strcmp(a, "-i") || !strcmp(a, "--interactive")) inter = 1;
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
    else if (!strcmp(a, "--gender")) gender = NEXT();
    else if (!strcmp(a, "--bench")) do_bench = 1;
    else if (!strcmp(a, "--bench-json")) do_bench = bench_json = 1;
    else if (!strcmp(a, "--perform")) perform_file = NEXT();
    else if (!strcmp(a, "--plan")) plan_only = 1;
    else if (!strcmp(a, "--genre")) genre = NEXT();
    else if (!strcmp(a, "--strength")) strength = (float)num_arg(a, NEXT());
    else if (!strcmp(a, "--director")) director_path = NEXT();
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
  int to_stdout = !strcmp(out, "-");
  if (inter && (text || json)) {
    fprintf(stderr, "steertts: -i reads its text from stdin: drop TEXT / --json\n");
    return 2;
  }
  if (inter && out_set && !to_stdout) {
    fprintf(stderr, "steertts: -i plays, or with -o - writes raw PCM to stdout; it does not write WAV files\n");
    return 2;
  }
  if (play && to_stdout) {
    fprintf(stderr, "steertts: --play and -o - both claim the audio; use one\n");
    return 2;
  }
  if (!model) model = getenv("STEERTTS_MODEL");
  if (!model) model = getenv("STEERTTS_EDGE_MODEL");
  if (!model) model = ".";
  char err[1024];
  double t0 = now_ms();
  if (perform_file)
    return perform(model, &cfg, &o, ctrl, perform_file, director_path, genre, strength, plan_only, out);
  steertts_engine* e = steertts_create_ex(model, &cfg, err, sizeof err);
  if (!e) {
    fprintf(stderr, "steertts: %s\n", err);
    return 2;
  }
  double load = (now_ms() - t0) / 1000;
  int rc = 0;
  if (list) {
    if (!strcmp(list, "voices") && gender && strcmp(gender, "female") && strcmp(gender, "male")) {
      fprintf(stderr, "steertts: --gender takes female or male\n");
      rc = 2;
    } else if (!strcmp(list, "voices"))
      list_voices(e, gender);
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
  if (o.voice && !(o.voice = resolve_voice(e, o.voice))) {
    steertts_destroy(e);
    return 2;
  }
  player pl = {NULL, NULL};
  if (inter) {
    if (!to_stdout && player_open(&pl)) rc = 2;
    else rc = interactive(e, o, to_stdout ? stdout : pl.pipe, &pl);
    player_close(&pl);
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
  int save = !play || out_set;   /* --play alone does not write out.wav */
  if (play && !json) stream = 1;  /* play as it renders (and parse markup, as -i does) */
  if (text && !json && !stream && !o.markup && looks_like_markup(text))
    fprintf(stderr, "steertts: warning: the text looks like markup but is read as plain text; add --markup\n");
  if (play && player_open(&pl)) {
    steertts_destroy(e);
    free(owned);
    return 2;
  }
  /* --play without a file streams straight into a pipe player; otherwise the audio is collected first */
  sink s = {now_ms(), -1, to_stdout ? stdout : (!save && pl.pipe) ? pl.pipe : NULL, NULL, 0, 0, 0};
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
    } else if (s.raw && write_pcm(s.raw, y, n)) {
      fprintf(stderr, "steertts: the audio output was closed\n");
      rc = 2;
    }
  }
  double synth = (now_ms() - s.t0) / 1000;
  if (s.closed) rc = 2;
  if (rc == 0 && !s.raw && save) {
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
  if (rc == 0 && play && !s.raw) {
    if (pl.pipe ? write_pcm(pl.pipe, y, n) : player_play_file(&pl, y, n)) {
      fprintf(stderr, "steertts: the player could not play the audio\n");
      rc = 2;
    }
  }
  player_close(&pl);
  if (stream) free(y);
  else steertts_free(y);
  free(owned);
  steertts_destroy(e);
  return rc;
}
