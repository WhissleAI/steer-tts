# Stand-alone C library

[`c/`](../c) is SteerTTS as one portable C11 library: `libsteertts` (static or shared), one header
([`steertts.h`](../c/include/steertts.h)) and a `steertts` command-line tool. It depends on nothing but libc, libm and
threads: no Python, no PyTorch, no ONNX Runtime, no C++ runtime. It runs **both models** from their Hugging Face
directories:

| Model | What runs in C | On disk |
|---|---|---|
| [v1](https://huggingface.co/WhissleAI/steer-tts) | acoustic model, flow refiner, Vocos, free-text prompts (MiniLM + adapter), voice cloning (WavLM x-vectors + the prompted flow), fp32 or int8 | 215 MB (fp32) / 52 MB (int8) model + helper assets (below) |
| [edge](https://huggingface.co/WhissleAI/steer-tts-edge) | acoustic model + vocoder, int4 / int8 weights kept packed in memory | 12 MB + 2.6 MB front-end assets |

Every request field of the Python packages is supported (text, voice, tag, ctrl, pace, guidance, word_shift, markup,
prompt, ref_wav, flow, seed), with the same validation and error messages, plus sentence streaming.

## Build

```bash
cd c
cmake -B build && cmake --build build -j       # or just: make
# -> build/libsteertts.a, build/libsteertts.so, build/steertts
```

Any C11 compiler: tested with GCC 11 and Clang (Zig 0.14), on x86-64 (AVX2 kernels, picked at run time) and ARM64
(NEON). Cross-compiling needs nothing special: `make CC=aarch64-linux-gnu-gcc`, or
`make CC="zig cc -target aarch64-linux-gnu"`. A fully static program (`cc -static app.c -lsteertts -lm -lpthread`) is
about 1.4 MB.

### Use it from your project

```bash
cmake --install build --prefix /usr/local        # lib, header, CMake package, pkg-config file, CLI
```

```cmake
find_package(steertts 1.2 REQUIRED)
target_link_libraries(app steertts::steertts)          # shared; steertts::steertts_static for static
```

or `cc app.c $(pkg-config --cflags --libs steertts)`, or vendor the directory with `add_subdirectory(c)`
(tests are then off by default). CMake options: `STEERTTS_BUILD_SHARED`, `STEERTTS_BUILD_CLI`,
`STEERTTS_BUILD_EXAMPLES`, `STEERTTS_BUILD_TESTS`. The shared library exports only the `steertts_*` API.

### Examples

[`c/examples/`](../c/examples): `simple.c` (one sentence to a WAV), `stream.c` (sentence streaming to raw PCM, play
it with `| aplay -f S16_LE -r 24000 -c 1`), `json_request.c` (full JSON requests, voice / tag listing, prompt
resolution, error handling), `speak.c` (a long-running app: the model loaded once, every typed line played through
the speakers as it renders, voice switching, errors reported without leaving the loop; POSIX). They build with the
library (`build/example_*`).

### Python binding

[`c/python/steertts_c.py`](../c/python/steertts_c.py) wraps the shared library with ctypes (no compiler needed,
numpy optional):

```python
from steertts_c import Engine
tts = Engine("steer-tts-edge", threads=2)          # Engine("steer-tts", precision="int8") for v1
wav = tts("Are you coming with us tonight?", ctrl={"f0_end": 2, "f0_iqr": 1.5})   # float32 numpy, 24 kHz
for audio, info in tts.stream("First sentence. <whisper>Second.</whisper>"):
    print(info["text"], info["tag"], len(audio))
v1 = Engine("steer-tts")
vec = v1.speaker_vector("consenting_speaker.wav")   # 512-d x-vector ...
wav = tts("Cloned on the edge model.", spk_vec=vec)  # ... drives the edge model too
tts.save(wav, "out.wav")
```

The same pattern works from any language with a C FFI (Swift, Kotlin/JNI, Rust, Go, C#): one opaque handle, plain C
types, JSON for the full request.

## Models

**Edge** runs from the Hugging Face directory as is:

```bash
hf download WhissleAI/steer-tts-edge --local-dir steer-tts-edge \
    --include config.json --include "*.safetensors" --include voices.json --include "native/*"
./build/steertts --model steer-tts-edge "Hello from C." -o hello.wav
```

**v1** needs, next to the model, a `native/` directory with the helper models converted to safetensors (Vocos, and
optionally MiniLM for prompts and WavLM for cloning) and the text front-end assets. Build it once with Python
(torch, vocos, transformers):

```bash
hf download WhissleAI/steer-tts --local-dir steer-tts
python c/tools/export_assets.py steer-tts/native --front steer-tts-edge/native/front
./build/steertts --model steer-tts "Hello from C." -o hello.wav
./build/steertts --model steer-tts --precision int8 "Hello from C." -o hello8.wav
```

| `native/` file | Needed for | fp32 | int8 |
|---|---|---|---|
| `front/` | always | 2.6 MB | (shared) |
| `vocos.safetensors` | always | 54 MB | 14 MB |
| `minilm.safetensors` + `minilm_vocab.txt` | `prompt` | 91 MB | 23 MB |
| `wavlm_sv.safetensors` | `ref_wav` | 401 MB | 115 MB |

Prompts and cloning load on first use; without their files the rest works and those requests return an error that
names the missing file. `steertts_has(engine, "prompt")` / `"clone"` tell you what is available.

## Command line

```bash
export STEERTTS_MODEL=$PWD/steer-tts            # or --model DIR
steertts "I can't believe it worked." --tag happiness --ctrl rate=1,f0_iqr=1.5 -o happy.wav
steertts "Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow>" --markup -o markup.wav
steertts "First sentence. Second one." --stream -o - | aplay -f S16_LE -r 24000 -c 1   # plays as it renders
steertts "First sentence. Second one." --stream --play                                # the same, player found for you
steertts -i --voice voice_0372      # interactive: loads once, speaks each line you type (/help for commands)
steertts "See you soon." --prompt "excited, a bit faster" -o prompt.wav               # v1
steertts "Now in your voice." --ref-wav consenting_speaker.wav -o clone.wav            # v1
steertts --json '{"text": "Are you serious?", "ctrl": {"f0_end": 2}, "guidance": 1.5}' -o q.wav
steertts --json '{"text": "Hi.", "prompt": "soft and sad"}' --resolve                 # the request it maps to
steertts --list voices | tags | controls
steertts --list voices --gender male            # 96 male / 104 female voices in edge (voices.json "gender")
steertts "Good morning." --voice male --play     # female / male: that gender's default voice
steertts --bench --threads 1                    # real-time factor, latency, peak memory on this machine
```

Other flags: `--voice`, `--pace`, `--guidance`, `--word-shift 3:pitch=2,dur=1.3`, `--seed`, `--no-flow`,
`--flow-steps`, `--precision fp32|int8`, `--threads N` (0 = all cores), `--assets DIR`.

Voices: `--voice` takes a voice name or `female` / `male`. The female default is the model's default voice
(`voice_1688`); the male one is `config.json` `"default_voices": {"male": ...}` when the model names one, else
`voice_2500`: of the mid-pitch, average-pace male voices it scored best (every word right under Whisper on hard
sentences with numbers and names, heard as male with p = 0.999 by a gender classifier). In C: `steertts_voice_gender(e, i)` and
`steertts_gender_voice(e, "male")`.

Playback: `--play` sends the audio to the speakers instead of `out.wav` (with `-o FILE` it does both); it implies
`--stream`, so markup is parsed and playback starts after the first sentence. `-i` keeps the
model loaded and speaks every line read from stdin, with inline markup and sentence streaming, until Ctrl-D or
`/quit`; `/voice NAME`, `/tag TAG`, `/ctrl k=z,...` change the request for the following lines, and `/voices`, `/tags`,
`/controls` list what the model has. With `-o -` it writes the raw PCM to stdout instead of playing. The player is
`$STEERTTS_PLAYER` (any command that reads raw 16-bit 24 kHz mono PCM on stdin), else the first of `play` (sox),
`aplay`, `paplay`, `ffplay` on `PATH`; on a stock macOS without them each utterance is played with `afplay`.
Players hold back a partial buffer until more input arrives (sox: 340 ms by default), so `-i` follows every line
with 400 ms of silence; otherwise a line's last word would play only with the next line. On
Windows set `STEERTTS_PLAYER`, for example to the `ffplay` line: `ffplay -loglevel error -nodisp -autoexit -f s16le
-ar 24000 -i -`.

## C API

```c
#include "steertts.h"

char err[512];
steertts_engine* e = steertts_create("steer-tts-edge", /*threads=*/1, err, sizeof err);
if (!e) { fprintf(stderr, "%s\n", err); return 1; }

steertts_options o;
steertts_default_options(&o);
o.tag = "happiness";
o.ctrl = "rate=1,f0_iqr=1.5";
float* audio;
size_t n;
if (steertts_synthesize(e, "I can't believe it worked.", &o, &audio, &n, err, sizeof err) == 0) {
  steertts_save_wav(audio, n, "out.wav");            /* 24 kHz mono float32 */
  steertts_free(audio);
}

/* sentence streaming: the callback gets each chunk as soon as it is rendered */
static int on_chunk(const float* audio, size_t n, void* user) { play(audio, n); return 0; /* non-zero stops */ }
steertts_stream(e, "First sentence. <whisper>Second, quietly.</whisper>", &o, on_chunk, NULL, err, sizeof err);

/* the complete request as JSON (request.schema.json), validated like Python */
steertts_synthesize_json(e, "{\"text\": \"See you soon.\", \"prompt\": \"excited\"}", &audio, &n, err, sizeof err);
steertts_destroy(e);
```

- `steertts_create_ex(dir, &config, ...)` takes a `steertts_config`: `threads`, `precision` (`"fp32"` / `"int8"`, v1),
  `flow`, `flow_steps`, `flow_temp`, default `guidance`, `assets_dir`.
- `steertts_options` covers voice, tag, ctrl, pace, guidance, markup, prompt, ref_wav, spk_vec, flow, seed and
  word_shift; `steertts_default_options` fills in the defaults.
- `steertts_stream_ex` / `steertts_stream_json` also pass each chunk's segment (text, tag, controls, pause) and its
  render time.
- `steertts_resolve_json`, `steertts_map_prompt`, `steertts_speaker_vector` (512-d x-vector of a clip, usable as
  `spk_vec` with either model), `steertts_phonemes`, `steertts_load_wav` / `steertts_save_wav`, voice / tag / control
  listing.
- Errors: 0 / non-NULL on success; otherwise the message is in `err`. Request errors use the Python messages.
- Threads: an engine runs one request at a time on its own worker pool. Use one engine per thread to synthesize in
  parallel; engines share nothing.
- Determinism: a request renders the same samples, bit for bit, at any thread count and on every run (PyTorch's
  results change with its thread count). Different CPUs (AVX2 vs NEON vs portable) differ at the float-rounding
  level.

The names match the [native C++ runtime's C API](native.md#c-api) (`steertts_create`, `steertts_options`,
`steertts_synthesize`, `steertts_stream`, `steertts_free_audio`, `steertts_save_wav`), so code written against it
compiles against this library unchanged.

## Parity with Python

`c/tests` holds a test per component plus an end-to-end test, each against references from the Python packages.
Run them with `cmake -B build && cmake --build build && (cd build && ctest)`; model directories come from
`STEERTTS_V1_MODEL` and `STEERTTS_EDGE_MODEL`, and tests without their model are skipped.

| Component | Reference | Result |
|---|---|---|
| Text front end | steertts_edge.text / markup (6,343 texts, 610 markup inputs, 33,265 G2P words) | exact |
| Acoustic model (v1 fp32, v1 int8, edge int4) | model.infer | identical durations / frame counts, mel SNR 121–126 dB |
| Vocos / edge vocoder | vocos / steertts_edge | 96–109 dB |
| Flow refiner, prompted flow (fp32, int8) | Flow.sample, same seed | 124–133 dB; the noise is torch.randn bit for bit |
| Mel front end, prompt mel | vocos MelSpectrogramFeatures, SteerTTS.prompt_mel | 115–117 dB |
| MiniLM tokenizer | HF tokenizer (287k texts incl. every code point) | identical ids |
| Prompt mapping (fp32, int8; 200 prompts) | PromptMapper | identical tags, probabilities and controls |
| WavLM x-vector (fp32, int8) | WavLMForXVector | 1 − cos < 3e-12 (same 16 kHz input) |
| End to end (18 requests: v1 fp32 / int8, edge) | SteerTTS / EdgeTTS | identical lengths; 75–103 dB, 53 dB with guidance, 35 dB when cloning |

Two parts differ by design:

- **Text front end.** The C library uses the edge front end (a C port of `steertts_edge.text`) for both models. The
  v1 Python package uses g2p_en with NLTK. They give the same phonemes for 95.0 % of the 6,343 test texts and 99.6 %
  of words; the rest are unusual names, a few homographs (read, content, used) and some number words. The end-to-end
  references above are rendered with the edge front end patched into Python v1 so that they compare the models.
- **Resampling for cloning.** Python resamples the reference clip with soxr; C uses its own windowed-sinc resampler
  (75–85 dB SNR against soxr on speech). That shifts the speaker vector by about 1e-7 in cosine and is why the cloning
  case agrees at 35 dB rather than 80+.

## Evaluation

The C library and the Python packages rendered the same 218 sentences (every 5th sentence of the Seed-TTS English
test set, voices rotating over the 200 edge voices), scored with the same judges: Whisper-large-v3-turbo with
Whisper's English normaliser for WER, and UTMOSv2 for naturalness. Scripts: [`c/eval/`](../c/eval)
(`render.py`, then `score.py wer|utmos|report`).

| Model | Runtime | WER % | UTMOSv2 | Same length as Python | Waveform SNR vs Python, median / min | Log-mel difference, median |
|---|---|---|---|---|---|---|
| v1 fp32 (flow) | **C** | 1.04 | 2.89 | 218 / 218 | 88.1 / 51.4 dB | 0.00006 dB |
| v1 fp32 (flow) | Python | 1.04 | 2.87 | | | |
| v1 fp32 (flow) | Python with g2p_en (its own front end) | 1.04 | 2.87 | | | |
| v1 int8 (flow) | **C** | 1.09 | 2.85 | 218 / 218 | 87.6 / 42.0 dB | 0.00006 dB |
| v1 int8 (flow) | Python | 1.09 | 2.86 | | | |
| edge | **C** | 1.13 | 2.14 | 218 / 218 | 88.5 / 56.9 dB | 0.00005 dB |
| edge | Python | 1.13 | 2.13 | | | |

- The C and Python renders are the same audio: identical lengths everywhere, identical WER, and spectra that differ by
  about 0.0001 dB. The UTMOSv2 differences (±0.02) are the judge's own noise: UTMOSv2 scores random crops, and the same
  file scored twice differs by 0.23 on average (up to 0.6), so a 218-clip mean moves by about ±0.02 between runs. The lowest SNRs are the flow refiner's sensitivity to summation order (PyTorch alone agrees with
  itself at 50–60 dB across its own thread counts on guided renders).
- The front-end choice does not cost accuracy: Python v1 with its original g2p_en front end gives the same WER as with
  the edge front end the C library uses.
- These sets were rendered at 8 threads on a busy machine: C RTF 0.48 (v1 fp32), 0.40 (v1 int8), 0.049 (edge) against
  Python 1.54, 2.26 and 0.228 under the same load.

## Quality assurance

- **Parity tests** per component and end to end (`ctest`, table above), including bit-exact torch.randn, exact text
  front end and identical tokenizer ids on 287k texts.
- **Determinism.** Renders are bit-identical at 1, 2, 3 and 8 threads on every path (flow, int8, prompts, cloning,
  markup); `test_engine` checks it.
- **Request validation** matches Python message for message: `c/tests/check_messages.py` compares 36 invalid requests
  (v1 and edge) against `normalize_request` / `EdgeTTS`, all identical, in the same order of checks.
- **Independent audit** of the API layer, the foundation (kernels, thread pool, file and JSON parsing) and the model
  components, with reproductions for every finding. The fixes include: error messages that lost their cause, number
  parsing under comma-decimal locales (an app calling `setlocale`), hostile or corrupted model files (shape and offset
  overflow, inconsistent quantization metadata, JSON read past the buffer), thread-pool robustness (failed thread
  creation, nested calls), NULL arguments, and clang's FMA contraction (now off on every compiler, so renders round the
  same everywhere).
- **Sanitizers and fuzzing.** The whole test suite runs clean under AddressSanitizer + UndefinedBehaviorSanitizer
  (also with `STEERTTS_NO_SIMD=1`), the thread pool under ThreadSanitizer; random and malformed requests (1,620 runs on
  both models), mutated safetensors files and JSON run clean. `c/tests/fuzz_requests.c` reproduces the request fuzz.
- **Compilers and targets.** GCC 11 and Clang (Zig 0.14), warning-free with `-Wall -Wextra -pedantic`, also at -O3;
  native x86-64 (AVX2) and ARM64 (NEON) tested, the latter under QEMU as Raspberry Pi 4 / Pi 5 CPUs with the full
  end-to-end test. The library and CLI also compile and link warning-free for macOS (ARM64 and x86-64), Windows (MinGW)
  and static musl Linux (a fully static musl binary renders correctly, 19 MB RAM). `.github/workflows/c.yml` builds and
  tests Linux x86-64, Linux ARM64 and macOS on every push.
- **API robustness** (`test_api`): comma-decimal locale, NULL arguments, outputs reset on failure, streaming stop, error
  messages that name the missing file.

## Speed and memory

The 20 benchmark sentences of `--bench` (65.6 s of audio), on a 2018 desktop CPU (Threadripper 2970WX, AVX2), load
average ~4. RTF = render time / audio duration (lower is faster).

| Build | Threads | RTF | Short sentence | Peak RSS | Load |
|---|---|---|---|---|---|
| **edge, C** | 1 | **0.061** | 88 ms | **27 MB** | 0.02 s |
| edge, native C++ + ONNX Runtime | 1 | 0.079 | 149 ms | 97 MB | 0.50 s |
| **edge, C** | 4 | **0.028** | 41 ms | 29 MB | 0.02 s |
| edge, native C++ + ONNX Runtime | 4 | 0.064 | 152 ms | 97 MB | 0.53 s |
| **v1 fp32, C** (with flow) | 1 | **0.88** | 1.36 s | 173 MB | 0.03 s |
| v1 fp32, Python + PyTorch (with flow) | 1 | 1.06 | 1.63 s | 930 MB | 0.77 s |
| **v1 int8, C** (with flow) | 1 | 0.88 | 1.27 s | **65 MB** | 0.04 s |
| **v1 fp32, C** (with flow) | 4 | **0.33** | 0.50 s | 173 MB | 0.03 s |
| v1 fp32, Python + PyTorch (with flow) | 4 | 0.51 | 0.97 s | 942 MB | 0.77 s |
| **v1 fp32, C, no flow** | 1 | **0.095** | 139 ms | 105 MB | 0.02 s |
| v1 fp32, Python + PyTorch, no flow | 1 | 0.104 | 162 ms | 848 MB | 0.58 s |

The same comparison repeated back to back on the release night, with the machine shared by other jobs (load average
about 30; all runtimes slower, the ordering unchanged):

| Build | Threads | C library | Python + PyTorch | native C++ + ONNX Runtime | Python + ONNX Runtime |
|---|---|---|---|---|---|
| edge | 1 | **0.083**, 27 MB | 0.156, 361 MB | 0.147, 96 MB | 0.147, 129 MB |
| edge | 4 | **0.063**, 27 MB | 0.144, 368 MB | 0.137, 95 MB | 0.119, 128 MB |
| v1 fp32, flow | 1 | **1.59**, 173 MB | 2.18, 927 MB | | |
| v1 fp32, flow | 4 | **0.79**, 172 MB | 1.35, 937 MB | | |
| v1 int8, flow | 4 | **0.49**, 63 MB | | | |
| v1 fp32, no flow | 1 | **0.147**, 105 MB | 0.231, 848 MB | | |
| v1 fp32, no flow | 4 | **0.051**, 103 MB | 0.147, 852 MB | | |

Memory is mostly the weights, which stay memory-mapped (int8 / int4 weights are dequantized panel by panel inside
the matrix multiply, never expanded). The flow refiner (20 network evaluations per sentence) is 90 % of v1's time;
`--no-flow` or the edge build are the fast options. Cloning adds the WavLM pass (~0.9 s for an 8 s clip at 4
threads, once per reference clip) and the prompted flow.

## Layout

| Path | What |
|---|---|
| `include/steertts.h` | the public API |
| `src/st_base.h`, `st_base.c`, `st_weights.c`, `st_kernels.c` | errors, memory, thread pool, JSON, safetensors and the packed int8 / int4 layouts, GEMM (AVX2 / NEON / portable), conv, attention, FFT |
| `src/front/` | text front end (normalisation, numbers, lexicon, int8 GRU G2P, heteronyms, markup) |
| `src/st_acoustic.c` | acoustic model |
| `src/st_flow.c`, `st_rng.c`, `st_mel.c` | flow refiner and prompted flow, torch-compatible randn, mel front end |
| `src/st_vocoder.c`, `st_dsp.c` | Vocos-style vocoders, iSTFT, WAV writing |
| `src/st_prompt.c`, `st_wordpiece.c`, `st_unicode_tab.c` | MiniLM + prompt adapter, WordPiece tokenizer and its Unicode tables |
| `src/st_wavlm.c`, `st_audio.c` | WavLM x-vectors, WAV reading, resampling |
| `src/st_api.c` | requests, rendering, streaming |
| `tools/steertts_cli.c` | the command line |
| `examples/` | example programs |
| `python/steertts_c.py` | ctypes binding |
| `eval/` | the evaluation below: render with C and Python, score WER / UTMOSv2 / parity |
| `tools/export_*.py`, `st_pack.py` | helper-model converters |
| `tests/` | per-component and end-to-end parity tests, reference generators |
