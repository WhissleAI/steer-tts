/* SteerTTS Nano benchmark for the ESP32-S3 (ESP-IDF 4.4 / 5.x app in main/, PlatformIO + Arduino example).
 *
 * Prints RESULT|name|time|rate|extra lines:
 *   1. kernel checks (SIMD vs portable C, must be bit-identical) and kernel micro-benchmarks (SRAM / flash weights);
 *   2. a synthesis run of NANO_BENCH_SECONDS of audio from random phonemes with the nano.bin in the `steertts`
 *      partition (random weights are fine: the cost does not depend on the values), with per-stage cycles,
 *      real-time factor, first-audio latency, arena / heap / stack high-water marks and a PCM hash that must match
 *      the host build (`nano_host nano.bin -s <seconds>`) for the same nano.bin. */
#if defined(ESP_PLATFORM)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
#include "esp_rom_sys.h"
#define NANO_CPU_MHZ() ((double)esp_rom_get_cpu_ticks_per_us())
#else
#include "esp32s3/rom/ets_sys.h"
#define NANO_CPU_MHZ() ((double)ets_get_cpu_frequency())
#endif
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nano_bench.h"
#include "nano_front.h"
#include "nano_partition.h"
#include "steertts_nano.h"

#ifndef CONFIG_NANO_BENCH_SECONDS
#define CONFIG_NANO_BENCH_SECONDS 10
#endif
#ifndef CONFIG_NANO_BENCH_TASK_STACK
#define CONFIG_NANO_BENCH_TASK_STACK 12288
#endif

static double g_mhz = 240.0;

static uint32_t rnd_state = 0x1234567u;
static int8_t rnd8(void) {
  rnd_state = rnd_state * 1664525u + 1013904223u;
  return (int8_t)(rnd_state >> 24);
}

static void kernel_checks(void) {
  enum { K = 512, O = 64, R = 16, C = 512, KW = 7 };
  int8_t* in = (int8_t*)heap_caps_aligned_alloc(16, (R + KW) * K, MALLOC_CAP_INTERNAL);
  int8_t* w = (int8_t*)heap_caps_aligned_alloc(16, O * K, MALLOC_CAP_INTERNAL);
  int8_t* o1 = (int8_t*)heap_caps_aligned_alloc(16, R * C, MALLOC_CAP_INTERNAL);
  int8_t* o2 = (int8_t*)heap_caps_aligned_alloc(16, R * C, MALLOC_CAP_INTERNAL);
  int32_t* b = (int32_t*)malloc(C * 4);
  int32_t* m = (int32_t*)malloc(C * 4);
  int32_t* s = (int32_t*)malloc(C * 4);
  for (int i = 0; i < (R + KW) * K; i++) in[i] = rnd8();
  for (int i = 0; i < O * K; i++) w[i] = rnd8();
  for (int i = 0; i < C; i++) {
    b[i] = rnd8() * 40;
    m[i] = (1 << 30) + rnd8() * 1000000;
    s[i] = -(9 + i % 4);
  }
  int32_t d_ref = nano_dot_s8_ref(in, w, K), d_simd = nano_dot_s8(in, w, K);
  nano_gemm_s8_ref(in, K, R, w, K, b, m, s, o1, O, O, -128, 127);
  nano_gemm_s8(in, K, R, w, K, b, m, s, o2, O, O, -128, 127);
  int dg = 0;
  for (int i = 0; i < R * O; i++) dg += o1[i] != o2[i];
  nano_dw_s8_ref(in, R, C, w, KW, b, m, s, o1, -128, 127);
  nano_dw_s8(in, R, C, w, KW, b, m, s, o2, -128, 127);
  int dd = 0;
  for (int i = 0; i < R * C; i++) dd += o1[i] != o2[i];
  printf("CHECK|int8 dot ref=%ld %s=%ld, gemm %d/%d differ, depthwise %d/%d differ -> %s\n", (long)d_ref,
         nano_kernel_impl(), (long)d_simd, dg, R * O, dd, R * C, (d_ref == d_simd && !dg && !dd) ? "OK" : "FAIL");

  /* micro-benchmarks, comparable with the board timing log */
  uint64_t t0 = nano_cycles();
  for (int it = 0; it < 4; it++) nano_gemm_s8(in, K, R, w, K, b, m, s, o2, O, O, -128, 127);
  double us = (double)(nano_cycles() - t0) / g_mhz / 4;
  double macs = (double)O * K * R;
  printf("RESULT|int8 GEMM %dx%d x B=%d SRAM, nano %s|%.1f us|%.3f GMAC/s|%.2f MAC/cycle\n", O, K, R,
         nano_kernel_impl(), us, macs / us / 1e3, macs / (us * g_mhz));
  t0 = nano_cycles();
  for (int it = 0; it < 4; it++) nano_gemm_s8_ref(in, K, R, w, K, b, m, s, o2, O, O, -128, 127);
  us = (double)(nano_cycles() - t0) / g_mhz / 4;
  printf("RESULT|int8 GEMM %dx%d x B=%d SRAM, plain C|%.1f us|%.3f GMAC/s|%.2f MAC/cycle\n", O, K, R, us,
         macs / us / 1e3, macs / (us * g_mhz));
  t0 = nano_cycles();
  for (int it = 0; it < 4; it++) nano_dw_s8(in, R, C, w, KW, b, m, s, o2, -128, 127);
  us = (double)(nano_cycles() - t0) / g_mhz / 4;
  macs = (double)R * C * KW;
  printf("RESULT|int8 depthwise conv1d k=%d C=%d T=%d, nano %s|%.1f us|%.3f GMAC/s|%.2f MAC/cycle\n", KW, C, R,
         nano_kernel_impl(), us, macs / us / 1e3, macs / (us * g_mhz));
  t0 = nano_cycles();
  for (int it = 0; it < 4; it++) nano_dw_s8_ref(in, R, C, w, KW, b, m, s, o2, -128, 127);
  us = (double)(nano_cycles() - t0) / g_mhz / 4;
  printf("RESULT|int8 depthwise conv1d k=%d C=%d T=%d, plain C|%.1f us|%.3f GMAC/s|%.2f MAC/cycle\n", KW, C, R, us,
         macs / us / 1e3, macs / (us * g_mhz));
  heap_caps_free(in);
  heap_caps_free(w);
  heap_caps_free(o1);
  heap_caps_free(o2);
  free(b);
  free(m);
  free(s);
}

/* weights streamed from flash: a 2048 x 512 int8 GEMM over the first 1 MB of the mapped nano.bin, B rows */
static void flash_gemm_bench(const nano_model* m) {
  enum { K = 512, O = 2048 };
  if (m->len < (size_t)O * K + 4096) return;
  const int8_t* w = (const int8_t*)m->base + ((m->hdr->data_offset + 15) & ~15u);
  int Bs[3] = {1, 16, 64};
  int8_t* in = (int8_t*)heap_caps_aligned_alloc(16, 64 * K, MALLOC_CAP_INTERNAL);
  int8_t* out = (int8_t*)heap_caps_aligned_alloc(16, 64 * O / 8, MALLOC_CAP_INTERNAL);
  int32_t* q = (int32_t*)calloc(O, 4);
  int32_t* sh = (int32_t*)calloc(O, 4);
  if (!in || !out || !q || !sh) return;
  for (int i = 0; i < 64 * K; i++) in[i] = rnd8();
  for (int i = 0; i < O; i++) q[i] = 1 << 30, sh[i] = -10;
  for (int bi = 0; bi < 3; bi++) {
    int B = Bs[bi];
    uint64_t t0 = nano_cycles();
    /* 8 slices of 256 output channels, each written to the same small output buffer */
    for (int sl = 0; sl < 8; sl++)
      nano_gemm_s8(in, K, B, w + (size_t)sl * 256 * K, K, NULL, q, sh, out, 256, 256, -128, 127);
    double us = (double)(nano_cycles() - t0) / g_mhz;
    double macs = (double)O * K * B;
    printf("RESULT|int8 GEMM %dx%d FLASH (1MB) x B=%d, nano %s|%.1f us|%.3f GMAC/s|%.2f MAC/cycle\n", O, K, B,
           nano_kernel_impl(), us, macs / us / 1e3, macs / (us * g_mhz));
  }
  heap_caps_free(in);
  heap_caps_free(out);
  free(q);
  free(sh);
}

typedef struct {
  uint32_t hash;
  long n;
  int peak;
  uint64_t t_start, t_first;
} pcm_sink;

static int on_pcm(void* user, const int16_t* pcm, int n) {
  pcm_sink* s = (pcm_sink*)user;
  if (!s->t_first) s->t_first = nano_cycles();
  s->hash = nano_hash_pcm(s->hash, pcm, n);
  for (int i = 0; i < n; i++) {
    int a = pcm[i] < 0 ? -pcm[i] : pcm[i];
    if (a > s->peak) s->peak = a;
  }
  s->n += n;
  return 0;   /* an application would write pcm to I2S here */
}

/* ------------------------------------------------------------------ text front end */
static nano_front_ctx g_front;
static nf_segment g_seg;
static int g_ids[NF_MAX_PH], g_widx[NF_MAX_PH];

static const char* const TAGS[] = {
    "none", "adoration", "amazement", "amusement", "anger", "anxiety", "apologetic", "bored", "confidence",
    "confusion", "contempt", "contentment", "cuteness", "desire", "determined", "disappointment", "disgust",
    "distress", "ecstasy", "embarrassment", "emphasis", "enunciated", "excited", "fast", "fear", "frustration",
    "guilt", "happiness", "hesitant", "high_pitch", "interest", "irritation", "laughing", "loud", "low_pitch",
    "mysterious", "neutral", "nostalgia", "pain", "panic", "pride", "realization", "relief", "sadness", "sarcasm",
    "serenity", "skeptical", "slow", "sympathy", "tired", "whisper"};
static const char* const CTRL[] = {"rate", "loud", "f0_med", "f0_iqr", "f0_end", "pause",
                                   "tilt", "voiced", "flat", "centroid", "cpps", "h1h2"};

static int front_checks(void) {
  static const char* cases[][2] = {
      {"Q4 results", "Q four results"},
      {"H2O and CO2", "H two O and C O two"},
      {"the \xe2\x80\x99" "90s", "the nineties"},
      {"the 2010s", "the twenty tens"},
      {"pages 10-12", "pages ten to twelve"},
      {"5 p.m.", "five P M"},
      {"Nov 3", "November three"},
      {"Jan was here.", "Jan was here."},
      {"-5\xc2\xb0" "C", "minus five degrees Celsius"},
      {"7-2", "seven to two"},
      {"31-03-2018", "thirty one to three to twenty eighteen"},
  };
  int ok = 0, n = (int)(sizeof cases / sizeof cases[0]);
  char out[256];
  for (int i = 0; i < n; i++) {
    int r = nano_front_normalize(&g_front, cases[i][0], out, (int)sizeof out);
    if (r >= 0 && !strcmp(out, cases[i][1]))
      ok++;
    else
      printf("INFO|front mismatch: \"%s\" -> \"%s\" (want \"%s\")\n", cases[i][0], r >= 0 ? out : g_front.err,
             cases[i][1]);
  }
  printf("CHECK|front normalisation %d/%d shared cases -> %s\n", ok, n, ok == n ? "OK" : "FAIL");
  static const char* texts[][2] = {
      {"plain sentence", "The quick brown fox jumps over the lazy dog."},
      {"numbers, date, time, money",
       "On Nov 3, 2024 at 4:15 pm, Dr. Smith paid $42.50 for 3 widgets."},
      {"unknown words (neural G2P)", "Zorblax quintessentially flummoxed the Krazwhip."},
  };
  for (int i = 0; i < 3; i++) {
    uint64_t t0 = nano_cycles();
    int k = nano_front_phonemes(&g_front, texts[i][1], g_ids, g_widx, NF_MAX_PH);
    double us = (double)(nano_cycles() - t0) / g_mhz;
    int words = 0;
    for (int j = 0; j < k; j++)
      if (g_widx[j] + 1 > words) words = g_widx[j] + 1;
    printf("RESULT|front to_phonemes, %s|%.1f us|%d ids, %d words|%s\n", texts[i][0], us, k, words,
           k < 0 ? g_front.err : "");
  }
  return ok == n;
}

static int tag_index(const char* t) {
  for (int k = 0; k < 51; k++)
    if (!strcmp(TAGS[k], t)) return k;
  return 0;
}

/* text with markup -> speech, segment by segment */
static void text_to_speech(nano_ctx* c, const nano_model* m, pcm_sink* sink) {
  const char* text =
      "Hello there! <slow>This is SteerTTS Nano</slow>, speaking from an ESP32-S3 with *no* PSRAM. "
      "[pause 300ms] It costs $4.99, and it reads numbers like 1,234 or 3:45 pm.";
  int nseg = 0;
  uint64_t t_front = 0, t_all = nano_cycles();
  long samples0 = sink->n;
  sink->t_first = 0;
  sink->t_start = nano_cycles();
  if (nano_front_markup_begin(&g_front, text, NULL, NULL, NULL, 0, TAGS, 51, NULL, NULL)) {
    printf("ERROR|markup: %s\n", g_front.err);
    return;
  }
  for (;;) {
    uint64_t t0 = nano_cycles();
    int r = nano_front_markup_next(&g_front, &g_seg);
    if (r <= 0) {
      if (r < 0) printf("ERROR|markup: %s\n", g_front.err);
      t_front += nano_cycles() - t0;
      break;
    }
    nseg++;
    if (g_seg.pause_ms > 0 || g_seg.skip || g_seg.event[0] || !g_seg.text[0]) {
      static int16_t zeros[256];
      long n = (long)g_seg.pause_ms * m->cfg[NANO_CFG_SAMPLE_RATE] / 1000;
      t_front += nano_cycles() - t0;
      for (; n > 0; n -= 256) on_pcm(sink, zeros, n < 256 ? (int)n : 256);
      continue;
    }
    int k = nano_front_phonemes(&g_front, g_seg.text, g_ids, g_widx, NF_MAX_PH);
    t_front += nano_cycles() - t0;
    if (k < 0) {
      printf("ERROR|front: %s\n", g_front.err);
      return;
    }
    static nano_word_shift ws[NF_MAX_WS];
    for (int i = 0; i < g_seg.n_ws; i++) {
      ws[i].word = g_seg.ws[i].word;
      ws[i].pitch = g_seg.ws[i].pitch;
      ws[i].energy = g_seg.ws[i].energy;
      ws[i].dur = g_seg.ws[i].dur;
    }
    float ctrl[12] = {0};
    uint8_t cset[12] = {0};
    for (int i = 0; i < g_seg.n_ctrl; i++)
      for (int j = 0; j < 12; j++)
        if (!strcmp(g_seg.ctrl_name[i], CTRL[j])) {
          ctrl[j] = g_seg.ctrl_val[i];
          cset[j] = 1;
        }
    nano_request r2;
    nano_request_default(&r2);
    r2.ids = g_ids;
    r2.n_ids = k > m->cfg[NANO_CFG_MAX_PHONEMES] ? m->cfg[NANO_CFG_MAX_PHONEMES] : k;
    r2.widx = g_widx;
    r2.ws = ws;
    r2.n_ws = g_seg.n_ws;
    r2.tag = tag_index(g_seg.tag) < m->cfg[NANO_CFG_N_TAGS] ? tag_index(g_seg.tag) : 0;
    r2.ctrl = ctrl;
    r2.ctrl_set = cset;
    if (nano_synthesize(c, &r2, on_pcm, sink) < 0) {
      printf("ERROR|synthesis: %s\n", nano_last_error());
      return;
    }
  }
  double all = (double)(nano_cycles() - t_all) / g_mhz, fr = (double)t_front / g_mhz;
  double audio = (double)(sink->n - samples0) / m->cfg[NANO_CFG_SAMPLE_RATE];
  printf("RESULT|text to speech, %d segments, %.2f s audio|%.1f us|RTF %.3f|front end %.1f us (%.1f%%), "
         "first audio after %.1f us\n",
         nseg, audio, all, audio > 0 ? all / 1e6 / audio : 0.0, fr, all > 0 ? 100.0 * fr / all : 0.0,
         sink->t_first ? (double)(sink->t_first - sink->t_start) / g_mhz : 0.0);
}

static void bench_task(void* arg) {
  (void)arg;
  g_mhz = NANO_CPU_MHZ();
  printf("\n=== STEERTTS NANO BENCH START ===\n");
  printf("INFO|ESP-IDF %s, cpu %.0f MHz, free internal heap %u KB, largest block %u KB, kernels %s\n",
         esp_get_idf_version(), g_mhz, (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
         (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024), nano_kernel_impl());
  kernel_checks();

  static nano_model m;   /* ~6 KB of tensor pointers: keep it off the task stack */
  static nano_partition_map map;
  int rc = nano_model_open_partition(&m, "steertts", &map);
  if (rc) {
    printf("ERROR|cannot open nano.bin from partition 'steertts' (%d): %s\n", rc, nano_last_error());
    printf("=== BENCH DONE ===\n");
    vTaskDelete(NULL);
    return;
  }
  printf("INFO|nano.bin in partition at 0x%x (%u KB partition): %u bytes, %.3f M int8 params, chunk %d (first %d)\n",
         (unsigned)map.partition_offset, (unsigned)(map.partition_size / 1024), (unsigned)m.hdr->file_bytes,
         m.n_params_i8 / 1e6, m.cfg[NANO_CFG_CHUNK], m.cfg[NANO_CFG_FIRST_CHUNK]);
  flash_gemm_bench(&m);

  size_t need = nano_arena_bytes(&m, 0);
  size_t free0 = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  void* arena = heap_caps_malloc(need, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!arena) {
    printf("ERROR|cannot allocate the %u byte arena (largest block %u)\n", (unsigned)need,
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    vTaskDelete(NULL);
    return;
  }
  nano_ctx* c = nano_ctx_create(&m, arena, need);
  int maxph = m.cfg[NANO_CFG_MAX_PHONEMES];
  int* ids = (int*)malloc(sizeof(int) * maxph);
  double fps = (double)m.cfg[NANO_CFG_SAMPLE_RATE] / m.cfg[NANO_CFG_HOP];
  const int dur = 5;
  long ph_total = (long)(CONFIG_NANO_BENCH_SECONDS * fps / dur + 0.5), done = 0;
  uint32_t st = 1;
  pcm_sink sink = {2166136261u, 0, 0, 0, 0};
  uint64_t t_first_utt = 0;
  sink.t_start = nano_cycles();
  while (done < ph_total) {
    int n = ph_total - done < maxph ? (int)(ph_total - done) : maxph;
    nano_bench_ids(ids, n, &st);
    nano_request r;
    nano_request_default(&r);
    r.ids = ids;
    r.n_ids = n;
    r.fixed_dur = dur;
    if (nano_synthesize(c, &r, on_pcm, &sink) < 0) {
      printf("ERROR|synthesis failed: %s\n", nano_last_error());
      break;
    }
    if (!t_first_utt) t_first_utt = sink.t_first;
    done += n;
  }
  uint64_t t_end = nano_cycles();
  nano_print_report(nano_ctx_profile(c), g_mhz, "board");
  double wall_us = (double)(t_end - sink.t_start) / g_mhz;
  double audio_s = (double)sink.n / m.cfg[NANO_CFG_SAMPLE_RATE];
  printf("RESULT|board wall clock %.2f s audio|%.1f us|RTF %.3f|incl. callbacks\n", audio_s, wall_us,
         audio_s > 0 ? wall_us / 1e6 / audio_s : 0.0);
  printf("RESULT|board first audio latency|%.1f us|first chunk %d frames|\n",
         (double)(t_first_utt - sink.t_start) / g_mhz, m.cfg[NANO_CFG_FIRST_CHUNK]);
  printf("RESULT|board arena|%u bytes|peak %u|of %u KB free internal before\n", (unsigned)need,
         (unsigned)nano_ctx_profile(c)->arena_peak, (unsigned)(free0 / 1024));
  printf("RESULT|board heap|%u KB free now|min ever %u KB|largest block %u KB\n",
         (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
         (unsigned)(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL) / 1024),
         (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024));
  UBaseType_t hw = uxTaskGetStackHighWaterMark(NULL);
  printf("RESULT|board stack|%u bytes used|of %u|task high-water mark %u bytes free\n",
         (unsigned)(CONFIG_NANO_BENCH_TASK_STACK - hw), (unsigned)CONFIG_NANO_BENCH_TASK_STACK, (unsigned)hw);
  printf("RESULT|board pcm|%ld samples|hash %08x|peak %d\n", sink.n, (unsigned)sink.hash, sink.peak);

  static nano_partition_map fmap;
  if (nano_front_open_partition(&g_front, "steertts", &fmap) == 0) {
    printf("INFO|front.bin at flash 0x%x: %u bytes mapped; context %u bytes, segment %u bytes\n",
           (unsigned)fmap.partition_offset, (unsigned)fmap.len, (unsigned)sizeof g_front, (unsigned)sizeof g_seg);
    UBaseType_t hw0 = uxTaskGetStackHighWaterMark(NULL);
    front_checks();
    text_to_speech(c, &m, &sink);
    UBaseType_t hw1 = uxTaskGetStackHighWaterMark(NULL);
    printf("RESULT|board stack after front end|%u bytes used|of %u|was %u before the front end\n",
           (unsigned)(CONFIG_NANO_BENCH_TASK_STACK - hw1), (unsigned)CONFIG_NANO_BENCH_TASK_STACK,
           (unsigned)(CONFIG_NANO_BENCH_TASK_STACK - hw0));
    printf("RESULT|board heap after front end|%u KB free now|min ever %u KB|\n",
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
           (unsigned)(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL) / 1024));
  } else {
    printf("INFO|no front.bin in the steertts partition (at +0x%x): %s\n", NANO_FRONT_PARTITION_OFFSET, g_front.err);
  }
  printf("=== BENCH DONE ===\n");
  free(ids);
  heap_caps_free(arena);
  vTaskDelete(NULL);
}

void nano_bench_start(int core) {
  /* one core for TTS; the other stays free for audio output / the application */
  xTaskCreatePinnedToCore(bench_task, "nano_bench", CONFIG_NANO_BENCH_TASK_STACK, NULL, 5, NULL, core);
}
#endif
