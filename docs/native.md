# Native C++ runtime (ONNX Runtime)

> For new projects, prefer the [stand-alone C library](c.md): it runs both models (this runtime is edge only), needs no
> ONNX Runtime, is faster (RTF 0.061 vs 0.079 on one thread) and uses about a quarter of the memory (27 vs 97 MB). Its
> C API keeps the names of the C API below.

`native/` is SteerTTS edge without Python: a C++17 library, a C API and a command-line tool on top of
[ONNX Runtime](https://onnxruntime.ai). It reads the same package as the Python runtimes
([WhissleAI/steer-tts-edge](https://huggingface.co/WhissleAI/steer-tts-edge)) and renders the same audio.

What runs where:

| Stage | Implementation |
|---|---|
| Text normalisation, tokenizer, lexicon, int8 GRU G2P, heteronyms | C++ port of the Python front end (`src/front/`), exact output parity on a golden test set |
| Inline markup, sentence segmentation | C++ (`src/front/`), parity on the same test set |
| Encoder + variance predictors, decoder, vocoder network | ONNX Runtime, three graphs with int4 / int8 `MatMulNBits` weights |
| Duration rounding, per-word shifts, length regulation, guidance | C++ (`src/engine.cpp`) |
| iSTFT, fades, WAV | C++ (`src/dsp.cpp`), no FFT library |

## Build

Requirements: CMake ≥ 3.16, a C++17 compiler, Linux (x86-64 or ARM64). ONNX Runtime is a prebuilt release; CMake
downloads it unless you point it at one.

```bash
git clone https://github.com/WhissleAI/steer-tts && cd steer-tts/native
cmake -B build -DCMAKE_BUILD_TYPE=Release          # downloads onnxruntime-linux-<arch>-1.31.0 into build/
cmake --build build -j8
# or with an unpacked ONNX Runtime release:  cmake -B build -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.31.0
```

Get the model (13.9 MB: config, voices, `onnx/` and the native front-end assets in `native/front/`; if the
Hugging Face repo asks you to accept its licence, run `hf auth login` and accept on the model page first):

```bash
pip install huggingface_hub
huggingface-cli download WhissleAI/steer-tts-edge --local-dir steer-tts-edge \
    --include config.json --include "voices.*" --include "onnx/*" --include "native/*"
```

## Command line

```bash
export STEERTTS_EDGE_MODEL=$PWD/steer-tts-edge
./build/steertts-edge "Hello there, nice to meet you." -o hello.wav
./build/steertts-edge "I can't believe it worked." --voice voice_0047 --tag happiness --ctrl rate=1,f0_iqr=1.5 -o happy.wav
./build/steertts-edge "Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow>" --markup -o markup.wav
./build/steertts-edge "First sentence. Second one." --stream -o - | aplay -f S16_LE -r 24000 -c 1   # plays as it renders
./build/steertts-edge --list voices          # or: tags, controls
./build/steertts-edge --bench                # real-time factor, latency, peak memory on this machine
```

Same flags as the Python `steertts-edge`: `--voice`, `--tag`, `--ctrl name=z,...`, `--rate`, `--pace`,
`--guidance`, `--markup`, `--threads`. `--stream` renders sentence by sentence and, with `-o -`, writes raw 16-bit
PCM to stdout as each sentence is ready.

## C++ API

```cpp
#include "steertts/steertts.h"

steertts::Engine tts("steer-tts-edge", {/*threads=*/1});
steertts::Request r;
r.text = "Are you coming with us tonight?";
r.ctrl = {{"f0_end", 2.0f}, {"f0_iqr", 1.5f}};
std::vector<float> wav = tts.synthesize(r);            // 24 kHz mono float32
steertts::Engine::save_wav(wav, "out.wav");

r.text = "First sentence. <whisper>Second, quietly.</whisper>";
tts.stream(r, [](const steertts::Chunk& c) {            // one chunk per sentence / style span
  play(c.audio);                                        // c.segment: text, tag, ctrl; c.synth_ms
  return true;                                          // false stops
});
```

`Request` has every edge field of the [request](request.md): `text`, `voice`, `tag`, `ctrl`, `pace`, `guidance`,
`word_shift`, `markup`, and `spk_vec` (a host-computed 512-d speaker vector). Errors (unknown voice / tag / control,
out-of-range values) throw `std::invalid_argument` with the same messages as Python. Link against the `steertts`
CMake target.

## C API

`include/steertts/steertts_c.h` wraps the engine for C, Swift, Kotlin/JNI, Rust, Go and so on:

```c
char err[256];
steertts_engine* e = steertts_create("steer-tts-edge", 1, err, sizeof err);
steertts_options o; steertts_default_options(&o);
o.tag = "happiness"; o.ctrl = "rate=1,f0_iqr=1.5"; o.markup = 1;
float* audio; size_t n;
if (steertts_synthesize(e, "I can't believe it worked.", &o, &audio, &n, err, sizeof err) == 0) {
  steertts_save_wav(audio, n, "out.wav");
  steertts_free_audio(audio);
}
steertts_stream(e, "One. Two. Three.", &o, on_chunk, user_data, err, sizeof err);   /* callback per sentence */
steertts_destroy(e);
```

See `examples/c_example.c`. One engine per thread, or serialize calls to a shared engine.

## Memory and speed

Same machine and sentences as [models.md](models.md#memory-and-speed-edge) (a shared, loaded 2018 desktop CPU,
one thread): load 1.0 s, RTF 0.16, 0.39 s for a short sentence, peak RSS 98 MB (107 MB streaming a 53 s paragraph,
69 MB after load). The binary is 0.46 MB (stripped); the ONNX Runtime shared library is 25 MB (aarch64) / 28 MB
(x86-64). Run `steertts-edge --bench` for your machine.

Memory follows the longest segment, not the whole text: `stream()` / `--stream` render one sentence (or style span)
at a time. A single very long sentence still costs more, since attention memory grows with the square of its
length.

## Raspberry Pi and other ARM64 boards

The same source builds for 64-bit ARM Linux (Raspberry Pi OS 64-bit, Ubuntu, Debian), where CMake downloads the
`aarch64` ONNX Runtime release. On a Pi:

```bash
sudo apt install cmake g++
cmake -B build && cmake --build build -j4
./build/steertts-edge --bench --threads 1
```

Cross-compiling from an x86-64 machine works with any aarch64 toolchain; the repo's emulation test uses Zig as the
compiler and a Debian bookworm arm64 root filesystem (the base of Raspberry Pi OS) under QEMU user mode. See
[raspberry-pi.md](raspberry-pi.md) for what was measured and what was estimated.

## Tests

```bash
cmake -B build -DSTEERTTS_TEST_MODEL=$PWD/steer-tts-edge && cmake --build build -j8 && (cd build && ctest)
# front_golden: front end vs Python on 6,343 texts, 610 markup inputs and 33,265 G2P words (exact match required)
# engine_parity: audio vs the Python ONNX runtime on 13 requests (same length, SNR >= 40 dB)
```

`tools/make_front_golden.py` and `tools/make_engine_golden.py` regenerate the references from the Python package.
