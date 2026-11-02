# SteerTTS Nano on the ESP32-S3

**SteerTTS Nano** is the smallest SteerTTS model: a steerable English text-to-speech model with 2.97 MB of int8
weights, 16 kHz output, and the same steering as the larger builds (voice, 51 style tags, 12 numeric controls,
per-word prosody, inline markup). This folder is its runtime for microcontrollers: a dependency-free C library that
runs on an **ESP32-S3 without PSRAM**, text in, audio out.

* Weights ([`WhissleAI/steer-tts-nano`](https://huggingface.co/WhissleAI/steer-tts-nano)) and the text front end data
  stay in flash (memory-mapped, never copied to RAM).
* All working memory is internal SRAM: a 122 KB synthesis arena plus a 12 KB front-end context.
* Audio streams out in 16 ms blocks while the rest of the sentence is still being computed.
* The same C code builds on any host (Linux, macOS) for testing.

## Measured on an ESP32-S3

ESP32-S3 at 240 MHz, one core, no PSRAM, 8 MB flash, Arduino-ESP32 2.0.17:

| | |
|---|---|
| Real-time factor, synthesis | **0.66** (10 s of speech in 6.6 s) |
| Real-time factor, from text (front end + synthesis) | **0.67** |
| First audio | **0.35 s** after the text arrives |
| Working RAM | **122 KB** arena (+ 9.5 KB front-end context, 2.1 KB segment, ~5 KB stack) |
| Flash | **2.97 MB** `nano.bin` + **1.8 MB** `front.bin` (text front end) |
| Accuracy | the int8 network is bit-exact with the reference runtime (`steertts_nano` on Hugging Face) |
| Unknown words | words outside the pronunciation dictionary go through a neural G2P on the device: about **0.27 s** each |

The second core stays free for I2S output and your application.

## Supported boards and toolchains

| | |
|---|---|
| Boards | ESP32-S3 with 8 MB flash or more; PSRAM is not needed (and not used). Tested on an ESP32-S3-DevKitC-1-N8. |
| Toolchains | PlatformIO + Arduino-ESP32 2.0.x (ESP-IDF 4.4); ESP-IDF 4.4 / 5.x; any C11 compiler for the host build |
| Input | text with steering markup (`<slow>`, `*emphasis*`, `[pause 300ms]`, style tags), or phoneme ids |
| Output | 16 kHz mono int16, streamed in 256-sample blocks |
| Voices | `nano_01` (voice 0, female), `nano_02` (voice 1, male) |

## Quick start

All commands run from this folder (`esp32/`). You need Python 3 and the Hugging Face CLI (`pip install
huggingface_hub`). The model repositories ask you to accept the licence once; run `hf auth login` first.

**1. Weights and text front end.** Download `nano.bin` and the device front end image `front.bin`:

```bash
hf download WhissleAI/steer-tts-nano nano.bin front.bin --local-dir models/steer-tts-nano
```

**2. (Optional) Rebuild `front.bin`** from the edge model's front end assets with
`python tools/make_front_bin.py <steer-tts-edge>/native/front -o models/steer-tts-nano/front.bin`.

**3. Partition image.** Pack both into one image of the `steertts` flash partition:

```bash
python tools/pack_partition.py models/steer-tts-nano/nano.bin models/steer-tts-nano/front.bin -o models/steertts_partition.bin
```

**4a. Flash with PlatformIO** (Arduino-ESP32). The project in `examples/platformio` has two programs: `speak` says a
sentence through an I2S amplifier, `bench` prints the benchmark.

```bash
cd examples/platformio
pio run -e speak -t upload --upload-port /dev/ttyACM0
python ~/.platformio/packages/tool-esptoolpy/esptool.py --chip esp32s3 -p /dev/ttyACM0 -b 921600 \
    write_flash 0x110000 ../../models/steertts_partition.bin       # once: it survives app re-flashes
pio device monitor -p /dev/ttyACM0 -b 115200                       # press RESET to run again
```

**4b. Or flash with ESP-IDF** (4.4 or 5.x; builds the benchmark app in `main/`):

```bash
. $IDF_PATH/export.sh
idf.py set-target esp32s3
idf.py -p /dev/ttyACM0 flash
python $IDF_PATH/components/partition_table/parttool.py -p /dev/ttyACM0 \
    write_partition --partition-name steertts --input models/steertts_partition.bin
idf.py -p /dev/ttyACM0 monitor                                     # Ctrl-] to quit
```

The console is UART0 (the board's USB-UART port). On the native USB port, add `-DARDUINO_USB_CDC_ON_BOOT=1` to
`build_flags` in `platformio.ini`.

Just measuring speed? `python tools/make_random_nano.py -o models/nano_random.bin` writes random weights in the same
format: the audio is noise, the cost is the same.

## Arduino example: speak a sentence

[`examples/platformio/src/speak/main.cpp`](examples/platformio/src/speak/main.cpp) reads text with markup, runs the
front end sentence by sentence and streams the PCM to an I2S DAC or class-D amplifier (MAX98357A or similar;
BCLK GPIO 4, WS GPIO 5, DIN GPIO 6 by default). The core of it:

```cpp
static int to_i2s(void* user, const int16_t* pcm, int n) {
  /* mono -> stereo frames */
  i2s_write(I2S_NUM_0, stereo, 4 * n, &written, portMAX_DELAY);
  if (++blocks % 32 == 0) vTaskDelay(1);   // yield about once per 0.5 s chunk
  return 0;
}

static void tts_task(void*) {          // pinned to core 0
  nano_model_open_partition(&model, "steertts", &map1);
  nano_front_open_partition(&front, "steertts", &map2);
  nano_ctx* ctx = nano_ctx_create(&model, arena, nano_arena_bytes(&model, 0));
  speak(ctx, "Hello! [pause 300ms] <happiness>It is a *lovely* day.</happiness>", /*voice*/ 0);
}
```

**Yield between chunks.** Synthesis is compute-bound. A task on core 0 that never blocks starves core 0's idle task,
and the idle task watchdog resets the board after 5 s. The example calls `vTaskDelay(1)` once per chunk of audio and
after every sentence; a blocking `i2s_write` also yields once the DMA queue is full. Do the same in your own
firmware (or `yield()` / `delay(1)` in Arduino code). The benchmark (`src/bench`) instead disables the core-0
watchdog, so that its timings contain no yields.

## C API

Headers: [`steertts_nano.h`](components/steertts_nano/include/steertts_nano.h) (synthesis),
[`nano_front.h`](components/steertts_nano/include/nano_front.h) (text front end),
[`nano_partition.h`](components/steertts_nano/include/nano_partition.h) (flash mapping). The runtime never calls
`malloc`: you provide the arena.

```c
#include "nano_front.h"
#include "nano_partition.h"

static nano_model model;                 /* ~6 KB of tensor pointers: static, not on a small stack */
static nano_front_ctx front;             /* 9.5 KB */
static nf_segment seg;                   /* 2.1 KB */
static int ids[NF_MAX_PH], widx[NF_MAX_PH];

/* PCM callback: n samples (one 256-sample hop, or the tail); return nonzero to stop */
static int on_pcm(void* user, const int16_t* pcm, int n) { /* write to I2S, yield now and then */ return 0; }

nano_partition_map m1, m2;
nano_model_open_partition(&model, "steertts", &m1);           /* or nano_model_open(&model, ptr, len) */
nano_front_open_partition(&front, "steertts", &m2);           /* or nano_front_open(&front, ptr, len) */
size_t need = nano_arena_bytes(&model, 0);                    /* ~122 KB at the default 0.5 s chunk */
void* arena = heap_caps_aligned_alloc(16, need, MALLOC_CAP_INTERNAL);
nano_ctx* ctx = nano_ctx_create(&model, arena, need);

nano_request r;
nano_request_default(&r);
r.n_ids = nano_front_phonemes(&front, "The meeting moved to Thursday at 3 PM.", ids, widx, NF_MAX_PH);
r.ids = ids;
r.widx = widx;
r.voice = 1;                                                  /* nano_02 */
r.tag = 27;                                                   /* "happiness" (index in the tag list) */
float ctrl[12] = {[0] = 1.0f, [3] = 1.5f};                    /* rate +1, f0_iqr +1.5 */
uint8_t set[12] = {[0] = 1, [3] = 1};                         /* only these two are set */
r.ctrl = ctrl;
r.ctrl_set = set;
r.pace = 1.0f;
long samples = nano_synthesize(ctx, &r, on_pcm, NULL);        /* -1 on error: nano_last_error() */
```

### The request (`nano_request`)

| Field | Meaning |
|---|---|
| `ids`, `n_ids` | phoneme ids from the front end (edge symbol table, at most 200 per call: one sentence) |
| `voice` | built-in voice: 0 = `nano_01`, 1 = `nano_02` |
| `tag` | style tag index, 0 = none; 51 tags (`happiness`, `whisper`, `sarcasm`, `slow` ...; the list is in the examples) |
| `ctrl`, `ctrl_set` | the 12 controls as speaker-relative z-scores (about -2 .. +2 is the useful range), in the order rate, loud, f0_med, f0_iqr, f0_end, pause, tilt, voiced, flat, centroid, cpps, h1h2; `ctrl_set[i]` = 1 marks a control as set. An unset control is left to the model, which is not the same as setting it to 0 (the speaker's average). `ctrl = NULL`: none set. |
| `pace` | speed factor, > 1 faster (default 1) |
| `widx`, `ws`, `n_ws` | per-word prosody shifts: for word `ws[i].word`, `pitch` and `energy` offsets and a `dur` factor (from `*emphasis*` and spans like `<high>`, or your own); `widx` maps each phoneme id to its word |
| `chunk`, `first_chunk` | frames per weight pass (0 = model default: 32, and 16 for the first chunk) |

### The text front end

`nano_front_phonemes()` turns one sentence into phoneme ids and word indices: normalisation of numbers, dates,
times, currencies, units, abbreviations and acronyms, the CMUdict lexicon, heteronym choice, and a neural G2P for
unknown words. It behaves like the SteerTTS edge front end.

For text with markup, `nano_front_markup_begin()` / `nano_front_markup_next()` return one segment at a time (a
sentence or a style run): its text, style tag, control values, word shifts, or a pause. Map each segment to a
request as `examples/platformio/src/speak/main.cpp` does. Markup: `<tag>...</tag>` styles, `<slow>` `<fast>`
`<high>` `<low>` `<loud>` `<soft>` `<emph>` prosody spans, `*emphasis*`, `[pause 300ms]`. Sound events
(`[laugh]`) are skipped on the device (there is no event bank). See [docs/FRONT_END.md](docs/FRONT_END.md).

Limits of the no-PSRAM configuration:

* one sentence of at most 200 phonemes per synthesis call (the markup parser already cuts text into sentences);
* 640 code points per normalised sentence in the front end;
* one synthesis at a time per arena (contexts can share a model).

## How it runs

The pipeline: **text front end** -> phoneme ids -> **encoder** (conv blocks with AdaLN steering) -> duration /
pitch / energy **predictors** -> length regulator -> **decoder** -> 80-band log-mel -> **vocoder** (Vocos-style
ConvNeXt blocks, magnitude + phase head) -> iSTFT -> PCM. It is conv-only, with no attention, so it streams.

Weights are read straight from flash through the cache, so flash time adds to compute time:

* **Chunked weight passes:** the decoder and vocoder run on chunks of 32 frames (0.5 s), with one pass over their
  weights per chunk (weight-stationary int8 GEMM over all frames of the chunk). The first chunk is shorter (16
  frames) for a fast start.
* **Exact streaming:** every conv layer keeps its last k-1 input rows between chunks, so chunked audio is identical,
  sample for sample, to an unchunked run.
* **Integer arithmetic:** int8 weights and activations with TFLite-style per-channel requantization, integer-sum
  LayerNorms, lookup tables instead of transcendental functions per frame.
* **S3 SIMD:** the int8 dot products use the 128-bit PIE instructions (`EE.VMULAS.S8.ACCX` for the GEMMs and the G2P
  GRU, `EE.VMULAS.S8.QACC` for the depthwise convs). They are bit-exact with the portable C kernels (checked at boot
  by the benchmark). The FFT uses esp-dsp when available.

`nano.bin` format: [docs/NANO_BIN.md](docs/NANO_BIN.md).

## Partition layout (8 MB flash)

| Name | Type | Offset | Size | Content |
|---|---|---|---|---|
| nvs | data/nvs | 0x9000 | 24 KB | |
| phy_init | data/phy | 0xF000 | 4 KB | |
| factory | app | 0x10000 | 1 MB | the firmware (~300 KB ESP-IDF, ~370 KB Arduino) |
| steertts | data/0x40 | 0x110000 | 6.9 MB | `nano.bin` at +0 (up to 4 MB), `front.bin` at +0x400000 (flash 0x510000) |

## Benchmark output

`bench` (PlatformIO) and the ESP-IDF app print:

```
CHECK|int8 dot ref=59976 esp32s3-simd=59976, gemm 0/1024 differ, depthwise 0/8192 differ -> OK
RESULT|board synth 10.00 s audio (125 phonemes, 625 frames)|... us|RTF ...|...
RESULT|board first audio latency|... us|first chunk 16 frames|
RESULT|board arena|125600 bytes|peak ...|...
RESULT|board int8 hashes|enc ...|dec-in ..., mel ..., head ...
CHECK|front normalisation 11/11 shared cases -> OK
RESULT|front to_phonemes, unknown words (neural G2P)|... us|45 ids, 5 words|
RESULT|text to speech, 4 segments, ... s audio|... us|RTF ...|front end ... us (...%), first audio after ... us
```

The int8 hashes match `nano_host` for the same `nano.bin`: every int8 tensor is bit-identical across platforms. The
PCM hash matches too when the firmware is built with the portable FFT (`idf.py -DNANO_PORTABLE_FFT=ON build`).
`idf.py menuconfig` -> SteerTTS Nano sets the benchmark length.

## Host build and tests

```bash
cmake -S host -B build-host && cmake --build build-host -j && ctest --test-dir build-host
./build-host/nano_host models/steer-tts-nano/nano.bin --front models/steer-tts-nano/front.bin \
    --text "Hello <slow>there</slow>! It's 4:15 pm." -o hello.wav
./build-host/nano_host models/steer-tts-nano/nano.bin -s 10 -o /dev/null --check-chunks   # streaming self-check
```

`ctest` runs the kernel tests; the front end conformance test (against the golden files in `../native/tests/front`)
is in [docs/FRONT_END.md](docs/FRONT_END.md).

## Layout

| Path | What |
|---|---|
| `components/steertts_nano/` | the runtime: ESP-IDF component, PlatformIO library (`library.json`) and host library |
| `  include/` | `steertts_nano.h` (synthesis), `nano_front.h` (text front end), `nano_partition.h`, `nano_bin.h` |
| `  src/` | portable C kernels and pipeline; `nano_simd_s3.c`: ESP32-S3 SIMD int8 kernels |
| `  front/` | the text front end (rules in plain C, data read from `front.bin`) |
| `  port/` | maps the `steertts` flash partition (`esp_partition_mmap`, IDF 4.4 and 5.x) |
| `  bench/` | the on-device benchmark |
| `main/` | ESP-IDF benchmark app |
| `examples/platformio/` | PlatformIO + Arduino: `speak` (I2S) and `bench` |
| `host/` | host build: `nano_host` CLI (text or ids to WAV, benchmark, streaming check), kernel and front end tests |
| `tools/` | `make_front_bin.py`, `pack_partition.py`, `nanobin.py` (nano.bin reader / writer), `make_random_nano.py`, `gen_tables.py` |
| `partitions.csv` | flash layout |

## Licence

Code: CC BY-NC 4.0 + Whissle terms, the same as the rest of this repository ([LICENSE](../LICENSE)); non-commercial
use only. Copyright (c) 2026 Whissle Inc. Commercial use (including in a product or device) needs a separate written
licence: ksingla@whissle.ai. The weights are under the same terms (see the
[model card](https://huggingface.co/WhissleAI/steer-tts-nano)). Third-party notices: [NOTICE](NOTICE).
