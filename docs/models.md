# Models and builds

SteerTTS v1 comes in two models and five runnable forms. They share the request format ([request.md](request.md)),
the voices' names, the style tags and the 12 controls.

| Build | Weights | Download | Runtime | Where it runs | Install |
|---|---|---|---|---|---|
| **v1, fp32** | [WhissleAI/steer-tts](https://huggingface.co/WhissleAI/steer-tts) | 215 MB (+ Vocos, WavLM, MiniLM) | Python, PyTorch | GPU or CPU | `pip install "steertts[all] @ git+https://github.com/WhissleAI/steer-tts"` |
| **v1, int8** | same repo, `int8/` | 52 MB (whole install 193 MB) | Python, PyTorch | GPU or CPU | same, `precision="int8"` |
| **edge, ONNX Runtime** | [WhissleAI/steer-tts-edge](https://huggingface.co/WhissleAI/steer-tts-edge) | 12.6 MB | Python, ONNX Runtime | CPU (x86-64, ARM64) | `pip install "steertts-edge[hub] @ git+https://github.com/WhissleAI/steer-tts#subdirectory=edge"` |
| **edge, PyTorch** | same repo | 11.8 MB | Python, PyTorch | CPU | same, plus the `[torch]` extra |
| **edge, native C/C++** | same repo | 13.9 MB | C++17 + ONNX Runtime, C API | CPU (Linux x86-64 / ARM64) | [native.md](native.md) |

## Which one?

- **Best quality, cloning, free-text prompts:** v1 fp32 (int8 sounds the same and downloads a quarter of the size).
  The flow refiner adds mel detail but is ~12× the compute; turn it off (`flow=False`) for real-time CPU use.
- **Small and fast on CPU, from Python:** edge with ONNX Runtime. Same intelligibility as v1 (without the refiner),
  about 0.1 lower UTMOS, every control at ≥ 95 % of the v1 effect.
- **Inside an app, a device, or anything that isn't Python:** the native C/C++ runtime. Same graphs and numerics as
  the Python ONNX runtime, no Python, a C API for bindings.
- **Reference / research:** edge with PyTorch is the evaluated reference the ONNX graphs are checked against.

## What each build includes

| | v1 | edge |
|---|:-:|:-:|
| 12 controls, 51 tags, guidance, per-word shifts, markup, sentence streaming | ✓ | ✓ |
| Built-in voices | 2,825 | 200 |
| Free-text prompts (`prompt=`) | ✓ | resolve on a host, send tag + ctrl |
| Voice cloning (`ref_wav=`) | ✓ (low similarity) | pass a host-computed speaker vector (`spk_vec`) |
| Flow-matching refiner | ✓ (optional) | — |
| Vocoder | Vocos (13.5M, fp32) | Vocos-style (6.2M, int4) |

## Quality

Open TTS Leaderboard protocol, run locally (not an official submission): Qwen3-ASR-1.7B WER, UTMOS22-strong, voice
`voice_1361`, Seed-TTS-eval English (1,088 sentences) and CV3-Eval English (500 sentences).

| Build | Size | Seed-TTS WER | Seed-TTS UTMOS | CV3-Eval WER | CV3-Eval UTMOS |
|---|---|---|---|---|---|
| v1 fp32, refiner on | 215 MB | 0.94 % | 4.27 | 2.62 % | 4.24 |
| v1 int8, refiner on | 52 MB | 1.03 % | 4.27 | 2.66 % | 4.25 |
| v1 fp32, refiner off | 215 MB | 0.96 % | 4.20 | 2.62 % | 4.17 |
| edge 1.x (v1 int4 acoustic + int4 vocoder) | 12 MB | 0.96 % | 4.11 | 2.67 % | 4.08 |
| **edge 2.0.0** (v2 int4 acoustic + int4 vocoder) | 12 MB | **0.93 %** | **4.17** | **2.44 %** | **4.15** |

The edge build's ONNX and native runtimes render the same audio as its PyTorch runtime up to float rounding (same
durations; mel spectrograms agree to ~3·10⁻⁵, waveforms at 52–90 dB SNR), so the edge rows hold for all three. Edge 2.0.0 carries the v2 acoustic model (same architecture and files as
1.x).
See the model cards for steering readback, per-tag numbers and limitations.

## Memory and speed (edge)

Measured on one machine, one thread, the same 20 sentences (65.6 s of audio) for every runtime, run back to back
(`edge/tools/bench_speed.py --runtime native onnx torch`). The machine is a shared AMD Threadripper 2970WX (2018,
3.0 GHz, no boost) with a load average of ~23 from other jobs; the same PyTorch runtime measured RTF 0.079 at a load
of 14, so read the absolute speeds as pessimistic and compare the rows with each other.

| Runtime | Device payload | Load | RTF (1 thread) | Short sentence, warm | Peak RSS |
|---|---|---|---|---|---|
| native C++ (`steertts-edge` binary) | 13.9 MB | 1.0 s | 0.16 | 0.39 s | **98 MB** |
| Python + ONNX Runtime (`OnnxTTS`) | 12.6 MB | 1.6 s | 0.21 | 0.37 s | 127 MB |
| Python + PyTorch (`EdgeTTS`) | 11.8 MB | 3.8 s | 0.20 | 0.42 s | 369 MB |

- **Speed is about the same in all three**; memory is not. PyTorch's runtime alone is ~200 MB before any model loads,
  and it unpacks the int4 weights to fp32 (59 MB). ONNX Runtime keeps them packed (`MatMulNBits`).
- **Streaming caps memory.** Native peak RSS streaming a 53 s paragraph sentence by sentence: 107 MB (69 MB after load,
  before the first sentence). Rendering a long paragraph in one call costs more, because attention memory grows with
  the square of the segment length.
- **Threads.** At this model size, 4 threads were no faster than 1 on the loaded machine; measure on yours.
- The device payload excludes the runtime itself: the ONNX Runtime shared library is 25–28 MB on disk (x86-64 /
  aarch64), Python + numpy + ONNX Runtime ~43 MB of RSS before the model loads.

## How the edge graphs are made

`edge/tools/export_onnx.py` exports the PyTorch edge model to three ONNX graphs (encoder + variance predictors,
decoder, vocoder network). Sequences are zero-padded to a multiple of the scan chunk (64) and the real length is
an input, with masks that make padded positions inert. The scan is written as plain batched matmuls (ONNX
Runtime's generic multi-operand Einsum was 47 % of the decoder time), and attention takes its key-padding mask as a
broadcast bias instead of an expanded T×T tensor.

`edge/tools/pack_onnx.py` then replaces every Linear, Conv1d (via im2col) and AdaLN projection with ONNX Runtime's
`MatMulNBits`, holding the package's own int4 / int8 codes and scales, so nothing is re-quantized. Weights stay
packed in memory and are unpacked block by block inside the matmul kernel. Duration rounding, length regulation,
guidance mixing and the iSTFT run outside the graphs (numpy in Python, plain C++ natively).
