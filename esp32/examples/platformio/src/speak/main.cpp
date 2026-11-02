// SteerTTS Nano: speak a sentence through an I2S amplifier (PlatformIO + Arduino-ESP32 2.0.x, ESP32-S3, no PSRAM).
//
// Wiring (change the pins below): an I2S DAC / class-D amplifier such as a MAX98357A on
//   BCLK -> GPIO 4, LRC / WS -> GPIO 5, DIN -> GPIO 6.
// nano.bin and front.bin must be in the "steertts" data partition (see ../../README.md).
//
// Synthesis runs in its own task on core 0. It is faster than real time, so the I2S DMA queue fills and i2s_write()
// blocks, but the task still yields (vTaskDelay(1)) once per chunk of audio and after every sentence: a task that
// never lets core 0's idle task run trips the idle task watchdog after 5 s.
#include <Arduino.h>
#include <driver/i2s.h>
#include <esp_heap_caps.h>
#include <string.h>

#include "nano_front.h"
#include "nano_partition.h"
#include "steertts_nano.h"

#ifndef SPEAK_PIN_BCLK
#define SPEAK_PIN_BCLK 4
#endif
#ifndef SPEAK_PIN_WS
#define SPEAK_PIN_WS 5
#endif
#ifndef SPEAK_PIN_DOUT
#define SPEAK_PIN_DOUT 6
#endif
#define SPEAK_I2S I2S_NUM_0
#define SPEAK_VOLUME 1.0f   // PCM gain (1.0 = as synthesized)
#define YIELD_EVERY 32      // PCM blocks (256 samples = 16 ms each) between yields: about once per 0.5 s chunk

static const char* const TAGS[] = {
    "none", "adoration", "amazement", "amusement", "anger", "anxiety", "apologetic", "bored", "confidence",
    "confusion", "contempt", "contentment", "cuteness", "desire", "determined", "disappointment", "disgust",
    "distress", "ecstasy", "embarrassment", "emphasis", "enunciated", "excited", "fast", "fear", "frustration",
    "guilt", "happiness", "hesitant", "high_pitch", "interest", "irritation", "laughing", "loud", "low_pitch",
    "mysterious", "neutral", "nostalgia", "pain", "panic", "pride", "realization", "relief", "sadness", "sarcasm",
    "serenity", "skeptical", "slow", "sympathy", "tired", "whisper"};
static const int N_TAGS = sizeof TAGS / sizeof TAGS[0];
static const char* const CTRL[NANO_N_CTRL] = {"rate", "loud",  "f0_med", "f0_iqr",   "f0_end", "pause",
                                              "tilt", "voiced", "flat",  "centroid", "cpps",   "h1h2"};

// Large state is static: a nano_model is ~6 KB, the front end context 9.5 KB, a segment 2.1 KB.
static nano_model g_model;
static nano_front_ctx g_front;
static nf_segment g_seg;
static int g_ids[NF_MAX_PH], g_widx[NF_MAX_PH];
static nano_word_shift g_ws[NF_MAX_WS];
static int16_t g_stereo[2 * 256];
static int g_blocks;

// PCM callback: n mono samples (one hop, or the tail of the utterance) -> stereo frames to I2S.
static int to_i2s(void* user, const int16_t* pcm, int n) {
  (void)user;
  while (n > 0) {
    int k = n < 256 ? n : 256;
    for (int i = 0; i < k; i++) {
      int v = (int)(pcm[i] * SPEAK_VOLUME);
      v = v > 32767 ? 32767 : v < -32768 ? -32768 : v;
      g_stereo[2 * i] = g_stereo[2 * i + 1] = (int16_t)v;
    }
    size_t written;
    i2s_write(SPEAK_I2S, g_stereo, sizeof(int16_t) * 2 * k, &written, portMAX_DELAY);
    pcm += k;
    n -= k;
  }
  if (++g_blocks % YIELD_EVERY == 0) vTaskDelay(1);   // let the idle task run (task watchdog)
  return 0;                                             // nonzero would stop synthesis
}

static void silence_ms(int ms, int sample_rate) {
  static const int16_t zeros[256] = {0};
  for (long n = (long)ms * sample_rate / 1000; n > 0; n -= 256) to_i2s(NULL, zeros, n < 256 ? (int)n : 256);
}

static int tag_index(const char* t, int n_tags) {
  for (int k = 0; k < N_TAGS && k < n_tags; k++)
    if (!strcmp(TAGS[k], t)) return k;
  return 0;
}

// Speak text with steering markup: one sentence / style run per nano_synthesize call.
static int speak(nano_ctx* ctx, const char* text, int voice) {
  const int sr = g_model.cfg[NANO_CFG_SAMPLE_RATE];
  if (nano_front_markup_begin(&g_front, text, NULL, NULL, NULL, 0, TAGS, N_TAGS, NULL, NULL)) {
    Serial.printf("markup: %s\n", g_front.err);
    return -1;
  }
  int r;
  while ((r = nano_front_markup_next(&g_front, &g_seg)) > 0) {
    if (g_seg.pause_ms > 0) {   // [pause 300ms]
      silence_ms(g_seg.pause_ms, sr);
      continue;
    }
    if (g_seg.skip || g_seg.event[0] || !g_seg.text[0]) continue;   // no event bank on the device
    int n = nano_front_phonemes(&g_front, g_seg.text, g_ids, g_widx, NF_MAX_PH);
    if (n < 0) {
      Serial.printf("front end: %s\n", g_front.err);
      return -1;
    }
    nano_request req;
    nano_request_default(&req);
    req.ids = g_ids;
    req.n_ids = n > g_model.cfg[NANO_CFG_MAX_PHONEMES] ? g_model.cfg[NANO_CFG_MAX_PHONEMES] : n;
    req.widx = g_widx;
    for (int i = 0; i < g_seg.n_ws; i++) {   // *emphasis* and prosody spans: per-word shifts
      g_ws[i].word = g_seg.ws[i].word;
      g_ws[i].pitch = g_seg.ws[i].pitch;
      g_ws[i].energy = g_seg.ws[i].energy;
      g_ws[i].dur = g_seg.ws[i].dur;
    }
    req.ws = g_ws;
    req.n_ws = g_seg.n_ws;
    req.voice = voice;
    req.tag = tag_index(g_seg.tag, g_model.cfg[NANO_CFG_N_TAGS]);
    float ctrl[NANO_N_CTRL] = {0};   // controls from the markup (<slow>, <high> ...); unset ones stay unset
    uint8_t set[NANO_N_CTRL] = {0};
    for (int i = 0; i < g_seg.n_ctrl; i++)
      for (int j = 0; j < NANO_N_CTRL; j++)
        if (!strcmp(g_seg.ctrl_name[i], CTRL[j])) {
          ctrl[j] = g_seg.ctrl_val[i];
          set[j] = 1;
        }
    req.ctrl = ctrl;
    req.ctrl_set = set;
    req.pace = 1.0f;
    if (nano_synthesize(ctx, &req, to_i2s, NULL) < 0) {
      Serial.printf("synthesis: %s\n", nano_last_error());
      return -1;
    }
    vTaskDelay(1);   // yield between sentences too
  }
  if (r < 0) Serial.printf("markup: %s\n", g_front.err);
  return r;
}

static void i2s_start(int sample_rate) {
  i2s_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
  cfg.sample_rate = sample_rate;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 256;   // 8 x 16 ms of audio queued
  cfg.tx_desc_auto_clear = true;   // silence instead of repeating the last buffer on underrun
  i2s_driver_install(SPEAK_I2S, &cfg, 0, NULL);
  i2s_pin_config_t pins;
  memset(&pins, 0, sizeof pins);
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = SPEAK_PIN_BCLK;
  pins.ws_io_num = SPEAK_PIN_WS;
  pins.data_out_num = SPEAK_PIN_DOUT;
  pins.data_in_num = I2S_PIN_NO_CHANGE;
  i2s_set_pin(SPEAK_I2S, &pins);
  i2s_zero_dma_buffer(SPEAK_I2S);
}

static void tts_task(void* arg) {
  (void)arg;
  static nano_partition_map mmap_model, mmap_front;
  if (nano_model_open_partition(&g_model, "steertts", &mmap_model)) {
    Serial.printf("cannot open nano.bin in the steertts partition: %s\n", nano_last_error());
    vTaskDelete(NULL);
  }
  if (nano_front_open_partition(&g_front, "steertts", &mmap_front)) {
    Serial.printf("cannot open front.bin: %s\n", g_front.err);
    vTaskDelete(NULL);
  }
  size_t need = nano_arena_bytes(&g_model, 0);   // ~122 KB of internal RAM
  void* arena = heap_caps_aligned_alloc(16, need, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  nano_ctx* ctx = arena ? nano_ctx_create(&g_model, arena, need) : NULL;
  if (!ctx) {
    Serial.printf("cannot allocate the %u byte arena\n", (unsigned)need);
    vTaskDelete(NULL);
  }
  i2s_start(g_model.cfg[NANO_CFG_SAMPLE_RATE]);

  // voice 0 = nano_01, 1 = nano_02; markup: <tag>...</tag>, <slow>, <high>, *emphasis*, [pause 300ms]
  speak(ctx, "Hello! I am SteerTTS Nano, running on an ESP32-S3. [pause 300ms] "
             "<happiness>It is a *lovely* day.</happiness> <slow>Thanks for listening.</slow>", 0);

  silence_ms(200, g_model.cfg[NANO_CFG_SAMPLE_RATE]);   // flush the DMA queue
  Serial.println("done");
  vTaskDelete(NULL);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  // TTS on core 0 with a 12 KB stack; loop() and the rest of the sketch keep core 1.
  xTaskCreatePinnedToCore(tts_task, "tts", 12288, NULL, 5, NULL, 0);
}

void loop() { delay(1000); }
