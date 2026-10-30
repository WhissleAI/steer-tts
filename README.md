# SteerTTS

A tiny English text-to-speech model you steer with numbers: 12 speaker-relative controls (rate, loudness, pitch,
pitch range, final rise/fall, tilt, breathiness …), 51 style tags, plain-text prompts and inline markup, all compiled
into one inspectable request. The full model has 9.6M parameters; the edge build fits in 12 MB and runs on a
Raspberry Pi-class CPU without Python. Both run from one dependency-free C library.

**[Listen to samples](https://whissleai.github.io/steer-tts/)** (no login) ·
[v1 weights](https://huggingface.co/WhissleAI/steer-tts) · [edge weights](https://huggingface.co/WhissleAI/steer-tts-edge) ·
[Issues](../../issues)

## Pick a build

| Build | Size | Runtime | Use it for |
|---|---|---|---|
| [v1](docs/models.md) (fp32 / int8) | 215 / 52 MB | Python + PyTorch, GPU or CPU | Best quality, free-text prompts, voice cloning, 2,825 voices |
| [edge, Python](docs/python.md#steertts_edge-edge) | 12.6 MB | Python + ONNX Runtime (or PyTorch) | Small and fast CPU synthesis from Python, 200 voices |
| [**C library**](docs/c.md) (v1 or edge) | v1 or edge files | C11, no dependencies | Apps, devices, Raspberry Pi; every feature of both models, no Python, ~27 MB RAM for edge |
| [edge, native](docs/native.md) | 13.9 MB | C++17 / C API + ONNX Runtime | The edge build on ONNX Runtime |

Every build takes the same [request](docs/request.md) (text, voice, tag, controls, guidance, markup) and renders
sentence by sentence for streaming. The edge build (2.0.0) carries the v2 acoustic model in 12 MB: Seed-TTS / CV3-Eval
WER 0.93 % / 2.44 % and UTMOS 4.17 / 4.15, more intelligible than v1 without its flow refiner (0.96 % / 2.62 %) and
0.03 UTMOS below it ([numbers](docs/models.md#quality)). Inline sound events (`[laugh]`, `[cough]` …) play recorded
sounds matched to the voice.

## Quick start

**v1** (the Hugging Face repo asks you to accept the licence once; `huggingface-cli login` first):

```bash
pip install "steertts[all] @ git+https://github.com/WhissleAI/steer-tts"
steertts say "I can't believe it worked." --tag happiness --ctrl rate=1,f0_iqr=1.5 -o happy.wav
steertts play "Hi there. <whisper>This plays as it is rendered.</whisper>"
```

**Edge from Python** (ONNX Runtime, no torch):

```bash
pip install "steertts-edge[hub] @ git+https://github.com/WhissleAI/steer-tts#subdirectory=edge"
steertts-edge "Are you coming with us tonight?" --ctrl f0_end=2,f0_iqr=1.5 -o question.wav
```

```python
from steertts_edge import OnnxTTS
tts = OnnxTTS.from_pretrained("WhissleAI/steer-tts-edge")
wav = tts("Well, *that* was fast. [pause 400ms] <whisper>Don't tell anyone.</whisper>", markup=True)
wav = tts("Oh no, I did it again. [laugh] Okay, let me start over.")   # a recorded laugh, matched to the voice
tts.save(wav, "out.wav")
```

**C library** (both models; any C11 compiler, x86-64 or ARM64):

```bash
cd c && cmake -B build && cmake --build build -j                 # or: make
hf download WhissleAI/steer-tts-edge --local-dir models/steer-tts-edge \
    --include config.json --include "*.safetensors" --include voices.json --include "native/*"
./build/steertts --model models/steer-tts-edge "Hello from C." -o hello.wav
```

## Documentation

- [Models and builds](docs/models.md): what each build includes, quality, memory and speed, how the edge graphs are made
- [The request](docs/request.md): controls, tags, guidance, per-word shifts, markup
- [Python](docs/python.md): `steertts` and `steertts_edge` APIs and CLIs
- [C library](docs/c.md): the dependency-free C11 library and CLI for both models, API, parity, speed
- [The director](docs/director.md): reading longer text as a performance (a 0.5 MB model that plans each sentence's controls and pauses)
- [Native C++ + ONNX Runtime](docs/native.md): the edge build on ONNX Runtime
- [Raspberry Pi](docs/raspberry-pi.md): what runs on a Pi 4 / Pi 5, measured and estimated
- [FAQ](docs/faq.md): memory, speed, licensing, limitations

## Repository layout

| Path | What |
|---|---|
| `steertts/` | v1 inference package (Python) |
| `edge/` | `steertts_edge`: edge runtimes (ONNX Runtime, PyTorch), tests, ONNX export and packing tools |
| `c/` | stand-alone C11 library + CLI for v1 and edge, helper-model converters, parity tests |
| `native/` | C++17 runtime on ONNX Runtime (edge): text front end port, engine, CLI, tests |
| `docs/` | Documentation and the [samples page](https://whissleai.github.io/steer-tts/) |
| `leaderboard/` | Open TTS Leaderboard evaluation container |
| `tests/` | v1 tests |

## Licence

Code (this repo) and weights (Hugging Face): CC BY-NC 4.0 + Whissle terms, non-commercial use only. Copyright (c) 2026
Whissle Inc. See [LICENSE](LICENSE); third-party components keep their own licences (NOTICE files).

Versions up to steertts v1.2.0 and steertts-edge v1.3.0 were released under MIT; from 2026-10-29 on, the code is
licensed under CC BY-NC 4.0 + Whissle terms.

- **Commercial use of the code, weights and voices** (including commercial use of generated audio) needs a
  separate written licence from Whissle Inc. Contact ksingla@whissle.ai.
- Whissle Inc. keeps all rights not expressly granted. The public licences do not limit its own use of SteerTTS, or
  its right to license SteerTTS (including later versions and derived models) on other terms. No patent rights are
  granted.
- "SteerTTS" and "Whissle" are trademarks of Whissle Inc. You may say that your work is based on SteerTTS. Do not
  name or market a product or model as "SteerTTS".
- Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md).
- The recorded event bank (`events.safetensors` in the model packages, used for `[laugh]` `[cough]` `[dog]` ...) is
  not covered by the licence above: its in-voice clips come from VocalSound (Gong et al., 2022; CC BY-SA 4.0) and
  its environmental clips are FSD50K clips released under CC0 1.0. See the packages' NOTICE files.
