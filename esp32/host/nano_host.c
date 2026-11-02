/* nano_host: run SteerTTS Nano on a host (benchmark, WAV output, streaming self-check).
 *
 *   nano_host nano.bin -s 10 -o out.wav          10 s of audio from random phonemes (5 frames each)
 *   nano_host nano.bin --ids "1 45 9 60 1" -o out.wav
 *   nano_host nano.bin --front front.bin --text "Hello <slow>there</slow>! It's 4:15 pm." -o out.wav
 *   nano_host nano.bin -s 5 --check-chunks       chunked output must equal a single-chunk run, sample for sample */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nano_front.h"
#include "steertts_nano.h"

typedef struct {
  int16_t* pcm;
  long n, cap;
} sink;

static int on_pcm(void* user, const int16_t* pcm, int n) {
  sink* s = (sink*)user;
  if (s->n + n > s->cap) {
    s->cap = (s->n + n) * 2 + 4096;
    s->pcm = (int16_t*)realloc(s->pcm, (size_t)s->cap * 2);
    if (!s->pcm) return 1;
  }
  memcpy(s->pcm + s->n, pcm, (size_t)n * 2);
  s->n += n;
  return 0;
}

static void put32(FILE* f, uint32_t v) {
  uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
  fwrite(b, 1, 4, f);
}
static void put16(FILE* f, uint16_t v) {
  uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)};
  fwrite(b, 1, 2, f);
}

static int write_wav(const char* path, const int16_t* pcm, long n, int sr) {
  FILE* f = fopen(path, "wb");
  if (!f) return -1;
  fwrite("RIFF", 1, 4, f);
  put32(f, (uint32_t)(36 + n * 2));
  fwrite("WAVEfmt ", 1, 8, f);
  put32(f, 16);
  put16(f, 1);
  put16(f, 1);
  put32(f, (uint32_t)sr);
  put32(f, (uint32_t)sr * 2);
  put16(f, 2);
  put16(f, 16);
  fwrite("data", 1, 4, f);
  put32(f, (uint32_t)(n * 2));
  for (long i = 0; i < n; i++) put16(f, (uint16_t)pcm[i]);
  fclose(f);
  return 0;
}

static uint32_t rng = 1;

/* utterances of random phonemes (a word boundary every 4) with fixed durations */
static long synth_seconds(nano_ctx* c, const nano_model* m, double seconds, int dur, int chunk, int first,
                          sink* out) {
  int maxph = m->cfg[NANO_CFG_MAX_PHONEMES];
  double fps = (double)m->cfg[NANO_CFG_SAMPLE_RATE] / m->cfg[NANO_CFG_HOP];
  long ph_total = (long)(seconds * fps / dur + 0.5);
  int* ids = (int*)malloc(sizeof(int) * (size_t)maxph);
  long done = 0, samples = 0;
  while (done < ph_total) {
    int n = ph_total - done < maxph ? (int)(ph_total - done) : maxph;
    nano_bench_ids(ids, n, &rng);
    nano_request r;
    nano_request_default(&r);
    r.ids = ids;
    r.n_ids = n;
    r.fixed_dur = dur;
    r.chunk = chunk;
    r.first_chunk = first;
    long k = nano_synthesize(c, &r, on_pcm, out);
    if (k < 0) {
      free(ids);
      return -1;
    }
    samples += k;
    done += n;
  }
  free(ids);
  return samples;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: %s nano.bin [-s seconds] [--ids \"1 45 9 ...\"] [--dur frames] [--chunk N] [--first N]\n"
            "       [--voice V] [--tag T] [--seed N] [-o out.wav] [--check-chunks] [--front front.bin --text TEXT]\n"
            "       [--ctrl rate=1.5,f0_med=-1] [--pace P (> 1 faster)]\n",
            argv[0]);
    return 2;
  }
  const char *model = argv[1], *wav = NULL, *ids_s = NULL, *front_path = NULL, *text = NULL, *ctrl_s = NULL;
  float pace = 1.0f;
  double seconds = 10;
  int dur = 5, chunk = 0, first = 0, voice = 0, tag = 0, check = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "-s") && i + 1 < argc) seconds = atof(argv[++i]);
    else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_s = argv[++i];
    else if (!strcmp(argv[i], "--dur") && i + 1 < argc) dur = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) chunk = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--first") && i + 1 < argc) first = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--voice") && i + 1 < argc) voice = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--tag") && i + 1 < argc) tag = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) rng = (uint32_t)atoi(argv[++i]) | 1u;
    else if (!strcmp(argv[i], "-o") && i + 1 < argc) wav = argv[++i];
    else if (!strcmp(argv[i], "--check-chunks")) check = 1;
    else if (!strcmp(argv[i], "--front") && i + 1 < argc) front_path = argv[++i];
    else if (!strcmp(argv[i], "--text") && i + 1 < argc) text = argv[++i];
    else if (!strcmp(argv[i], "--ctrl") && i + 1 < argc) ctrl_s = argv[++i];
    else if (!strcmp(argv[i], "--pace") && i + 1 < argc) pace = (float)atof(argv[++i]);
    else {
      fprintf(stderr, "unknown argument %s\n", argv[i]);
      return 2;
    }
  }
  static const char* const ctrl_dims[12] = {"rate", "loud", "f0_med", "f0_iqr", "f0_end", "pause",
                                            "tilt", "voiced", "flat", "centroid", "cpps", "h1h2"};
  float uctrl[12] = {0};
  uint8_t uset[12] = {0};
  if (ctrl_s) {   /* name=value,... */
    char buf[512];
    snprintf(buf, sizeof buf, "%s", ctrl_s);
    for (char* tk = strtok(buf, ","); tk; tk = strtok(NULL, ",")) {
      char* eq = strchr(tk, '=');
      int found = 0;
      if (eq) {
        *eq = 0;
        for (int j = 0; j < 12; j++)
          if (!strcmp(tk, ctrl_dims[j])) {
            uctrl[j] = (float)atof(eq + 1);
            uset[j] = 1;
            found = 1;
          }
      }
      if (!found) {
        fprintf(stderr, "bad --ctrl entry %s (controls: rate loud f0_med f0_iqr f0_end pause tilt voiced flat "
                        "centroid cpps h1h2)\n", tk);
        return 2;
      }
    }
  }
  FILE* f = fopen(model, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", model);
    return 1;
  }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  void* blob = NULL;
  if (posix_memalign(&blob, 64, (size_t)len) || fread(blob, 1, (size_t)len, f) != (size_t)len) {
    fprintf(stderr, "cannot read %s\n", model);
    return 1;
  }
  fclose(f);
  nano_model m;
  if (nano_model_open(&m, blob, (size_t)len)) {
    fprintf(stderr, "%s\n", nano_last_error());
    return 1;
  }
  int cmax = chunk > 0 ? chunk : m.cfg[NANO_CFG_CHUNK];
  size_t need = nano_arena_bytes(&m, cmax);
  printf("INFO|model %s: %u bytes, %.3f M int8 params, chunk %d (first %d), arena %u bytes, kernels %s\n", model,
         (unsigned)len, m.n_params_i8 / 1e6, cmax, first > 0 ? first : m.cfg[NANO_CFG_FIRST_CHUNK], (unsigned)need,
         nano_kernel_impl());
  void* arena = malloc(need);
  nano_ctx* c = nano_ctx_create(&m, arena, need);
  if (!c) {
    fprintf(stderr, "%s\n", nano_last_error());
    return 1;
  }
  sink out = {0};
  long n;
  uint32_t seed0 = rng;
  if (text) {
    if (!front_path) {
      fprintf(stderr, "--text needs --front front.bin\n");
      return 2;
    }
    static nano_front_ctx fc;
    static nf_segment seg;
    static int ids[NF_MAX_PH], widx[NF_MAX_PH];
    FILE* ff = fopen(front_path, "rb");
    if (!ff) {
      fprintf(stderr, "cannot open %s\n", front_path);
      return 1;
    }
    fseek(ff, 0, SEEK_END);
    long fl = ftell(ff);
    fseek(ff, 0, SEEK_SET);
    void* fb = NULL;
    if (posix_memalign(&fb, 64, (size_t)fl) || fread(fb, 1, (size_t)fl, ff) != (size_t)fl) return 1;
    fclose(ff);
    if (nano_front_open(&fc, fb, (size_t)fl)) {
      fprintf(stderr, "%s\n", fc.err);
      return 1;
    }
    if (nano_front_markup_begin(&fc, text, NULL, NULL, NULL, 0, NULL, 0, NULL, NULL)) {
      fprintf(stderr, "%s\n", fc.err);
      return 1;
    }
    int r;
    while ((r = nano_front_markup_next(&fc, &seg)) > 0) {
      if (seg.pause_ms > 0 || seg.skip || seg.event[0] || !seg.text[0]) {
        static int16_t z[256];
        for (long k = (long)seg.pause_ms * m.cfg[NANO_CFG_SAMPLE_RATE] / 1000; k > 0; k -= 256)
          on_pcm(&out, z, k < 256 ? (int)k : 256);
        continue;
      }
      int k = nano_front_phonemes(&fc, seg.text, ids, widx, NF_MAX_PH);
      if (k < 0) {
        fprintf(stderr, "%s\n", fc.err);
        return 1;
      }
      printf("INFO|segment \"%s\" tag %s: %d phoneme ids\n", seg.text, seg.tag, k);
      nano_word_shift ws[NF_MAX_WS];
      for (int i = 0; i < seg.n_ws; i++) {
        ws[i].word = seg.ws[i].word;
        ws[i].pitch = seg.ws[i].pitch;
        ws[i].energy = seg.ws[i].energy;
        ws[i].dur = seg.ws[i].dur;
      }
      float ctrl[12];
      uint8_t cset[12];
      memcpy(ctrl, uctrl, sizeof ctrl);   /* --ctrl defaults, overridden by the markup's controls */
      memcpy(cset, uset, sizeof cset);
      for (int i = 0; i < seg.n_ctrl; i++)
        for (int j = 0; j < 12; j++)
          if (!strcmp(seg.ctrl_name[i], ctrl_dims[j])) {
            ctrl[j] = seg.ctrl_val[i];
            cset[j] = 1;
          }
      nano_request rq;
      nano_request_default(&rq);
      rq.ids = ids;
      rq.n_ids = k < m.cfg[NANO_CFG_MAX_PHONEMES] ? k : m.cfg[NANO_CFG_MAX_PHONEMES];
      rq.widx = widx;
      rq.ws = ws;
      rq.n_ws = seg.n_ws;
      rq.voice = voice;
      rq.tag = tag;
      rq.ctrl = ctrl;
      rq.ctrl_set = cset;
      rq.pace = pace;
      rq.chunk = chunk;
      rq.first_chunk = first;
      if (nano_synthesize(c, &rq, on_pcm, &out) < 0) {
        fprintf(stderr, "%s\n", nano_last_error());
        return 1;
      }
    }
    if (r < 0) {
      fprintf(stderr, "%s\n", fc.err);
      return 1;
    }
    n = out.n;
  } else if (ids_s) {
    int ids[1024], k = 0;
    for (char* p = (char*)ids_s; *p && k < 1024;) {
      char* e;
      long v = strtol(p, &e, 10);
      if (e == p) break;
      ids[k++] = (int)v;
      p = e;
    }
    nano_request r;
    nano_request_default(&r);
    r.ids = ids;
    r.n_ids = k;
    r.voice = voice;
    r.tag = tag;
    r.ctrl = uctrl;
    r.ctrl_set = uset;
    r.pace = pace;
    r.chunk = chunk;
    r.first_chunk = first;
    n = nano_synthesize(c, &r, on_pcm, &out);
  } else {
    n = synth_seconds(c, &m, seconds, dur, chunk, first, &out);
  }
  if (n < 0) {
    fprintf(stderr, "%s\n", nano_last_error());
    return 1;
  }
  const nano_profile* p = nano_ctx_profile(c);
  nano_print_report(p, 1000.0, "host");
  int32_t peak = 0;
  double rms = 0;
  for (long i = 0; i < out.n; i++) {
    int a = out.pcm[i] < 0 ? -out.pcm[i] : out.pcm[i];
    if (a > peak) peak = a;
    rms += (double)out.pcm[i] * out.pcm[i];
  }
  printf("RESULT|host arena peak|%u bytes|of %u|\n", (unsigned)p->arena_peak, (unsigned)need);
  printf("RESULT|host pcm|%ld samples|hash %08x|peak %d, rms %.1f\n", out.n,
         (unsigned)nano_hash_pcm(2166136261u, out.pcm, (int)out.n), (int)peak, out.n ? __builtin_sqrt(rms / out.n) : 0.0);
  if (wav && write_wav(wav, out.pcm, out.n, m.cfg[NANO_CFG_SAMPLE_RATE])) fprintf(stderr, "cannot write %s\n", wav);
  if (check) {
    /* the same request in one big chunk and in small odd chunks must give identical PCM */
    int sizes[3] = {1, 7, 1000};
    int bad = 0;
    for (int t = 0; t < 3; t++) {
      size_t need2 = nano_arena_bytes(&m, sizes[t]);
      void* a2 = malloc(need2);
      nano_ctx* c2 = nano_ctx_create(&m, a2, need2);
      sink o2 = {0};
      rng = seed0;
      long n2;
      if (ids_s) {
        n2 = -1;
        fprintf(stderr, "--check-chunks needs -s\n");
      } else {
        n2 = c2 ? synth_seconds(c2, &m, seconds, dur, sizes[t], sizes[t], &o2) : -1;
      }
      long diff = 0;
      if (n2 != out.n) diff = -1;
      else
        for (long i = 0; i < n2; i++) diff += o2.pcm[i] != out.pcm[i];
      printf("CHECK|chunk %d vs %d: %ld samples, %ld differ -> %s\n", sizes[t], cmax, n2, diff,
             diff == 0 ? "OK" : "FAIL");
      bad |= diff != 0;
      free(o2.pcm);
      free(a2);
    }
    if (bad) return 3;
  }
  free(out.pcm);
  free(arena);
  free(blob);
  return 0;
}
