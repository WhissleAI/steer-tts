# Raspberry Pi

**Short version.** Both native builds run, and render the same audio as on x86, under an emulated Raspberry Pi 4
(Cortex-A72) and an emulated Pi 5-class CPU (ARMv8.2). The [C library](c.md) is the lighter of the two: the edge model
needs about 30–45 MB of memory and loads in a few hundredths of a second. Speed on real Pi hardware has **not been
measured yet**; instruction counts put the edge model at **about 2.2–4.1× faster than real time on one Pi 4 core and
5.5–9× on a Pi 5**. If you run it on a Pi, please post `steertts --bench --threads 1` in an issue.

| Build on a Pi (estimates, see below) | Pi 4, 1 core | Pi 5, 1 core | Pi 5, 4 cores | Memory |
|---|---|---|---|---|
| edge, C library | 2.2–4.1× real time | 5.5–9× | n/a (1 core is enough) | ~30–45 MB |
| edge, native C++ + ONNX Runtime | 1.8–3.5× | 4.5–8× | n/a | ~100 MB |
| v1 without flow (`--no-flow`), C library | 1.5–2.8× | 3.7–6.2× | n/a | ~120 MB |
| v1 with flow, C library | 0.13–0.23× (not real time) | 0.3–0.5× | ~0.8–1.4× (borderline) | 85–175 MB |

## The C library on ARM

- **Correctness.** The ARM64 build (Zig as the cross compiler, glibc 2.36, the Raspberry Pi OS bookworm baseline)
  passes the end-to-end test under QEMU user mode with `-cpu cortex-a72` (Pi 4) and `-cpu max,sve=off` (Pi 5 class):
  all 18 requests (v1 fp32 and int8 with flow, prompts and cloning; edge with controls, tags, guidance, markup and
  per-word shifts) give exactly the same number of samples as Python, at 77–104 dB SNR. The guided v1 request agrees at
  36 dB, which is inside the range PyTorch itself shows on that request across its own thread counts (50–60 dB on
  x86). The C library needs no emulated `/proc/cpuinfo`.
- **Instruction counts** (QEMU plugin, guest instructions, 5 sentences / 17.6 s of audio, streamed, 1 thread):

| | Load | Per second of audio |
|---|---|---|
| edge, C library | 0.07 G | 0.66 G |
| edge, native C++ + ONNX Runtime | 3.7 G | 0.79 G |
| v1 without flow, C library | 0.11 G | 0.97 G |
| v1 with flow, C library (int8 weights; fp32 is the same compute) | 0.17 G | 11.6 G |

  With the same IPC × clock ranges as below (Cortex-A72: 0.8–1.5 × 1.8 GHz; Cortex-A76: 1.5–2.5 × 2.4 GHz) that gives
  the table above. The four-core figure scales the one-core estimate by the 2.7× four-thread speed-up measured for v1
  on x86. The flow refiner is 92 % of v1's compute; on a Pi use the edge model or `--no-flow` for real-time speech.
- **Build on the Pi:** `sudo apt install cmake gcc && cd steer-tts/c && cmake -B build && cmake --build build -j4`, or
  just `make`. The models: `c/tools/get_models.sh edge` (curl only).

The rest of this page describes the ONNX Runtime build's emulation, which used the same rootfs and method.

## What was tested, and how

There was no physical Pi for this release, so the test splits into what emulation can show exactly and what it
can't.

**Emulation (exact): it runs, and it's correct on ARM.**

- Build: `native/` cross-compiled for `aarch64-linux-gnu` (glibc 2.36, the Raspberry Pi OS bookworm baseline),
  linked against the official `onnxruntime-linux-aarch64-1.31.0` release.
- Userland: a Debian bookworm arm64 root filesystem (the base of Raspberry Pi OS 64-bit), run with QEMU user mode.
- CPU: `-cpu cortex-a72` (the Pi 4's core: ARMv8.0 NEON, no dot-product or fp16 arithmetic), and `-cpu max,sve=off`
  for a Pi 5-class ARMv8.2 core (dot-product and fp16 arithmetic, no SVE). The guest sees a 4-core Pi
  `/proc/cpuinfo` and CPU topology, so ONNX Runtime picks the kernels it would pick on the real board.
- Result: on both CPUs, all 13 requests of `tests/engine_golden` (plain text, controls, tags, guidance, per-word shifts,
  markup, numbers, empty text) render with **exactly the same number of samples** as the x86 reference and at
  51–86 dB SNR to it, the same level at which the x86 PyTorch and ONNX runtimes agree with each other. ONNX
  Runtime's ARM NEON kernels produce the same audio.
- Without the emulated `/proc/cpuinfo`, ONNX Runtime 1.31 crashes at load under QEMU user mode (its CPU detection
  parses the host's x86 topology); on a real Pi this doesn't apply.

**Estimate 1: instruction counts.** A QEMU plugin counts the guest instructions the real ARM code executes
(ONNX Runtime's NEON kernels, the front end, the iSTFT). That count doesn't depend on how busy the build machine is.
Dividing it by a Cortex-A72 / Cortex-A76 instructions-per-cycle range and the Pi's clock gives a time:

| | Cortex-A72 (Pi 4) | ARMv8.2 (Pi 5 class) |
|---|---|---|
| Instructions to load the model | 3.7 G | 3.7 G |
| Instructions to render 17.6 s of audio (5 sentences, streamed) | 13.95 G | 13.95 G |
| Per second of audio | 0.79 G | 0.79 G |
| Assumed IPC × clock | 0.8–1.5 × 1.8 GHz | 1.5–2.5 × 2.4 GHz (Cortex-A76) |
| **Estimated RTF, 1 thread** | **0.29–0.55** | **0.13–0.22** |
| Estimated load time | 1.4–2.6 s | 0.6–1.0 s |

Both CPUs execute the same instruction count: on these graphs ONNX Runtime uses the same fp32 NEON kernels on ARMv8.0
and ARMv8.2. The IPC ranges are assumptions (typical sustained IPC of each core on NEON-heavy code), and they are
the main uncertainty; memory bandwidth on the Pi (LPDDR4 / LPDDR4X) could push the real number towards the slow end.

**Estimate 2: single-core benchmark ratio.** Geekbench 6 single-core: Raspberry Pi 5 ≈ 764, Raspberry Pi 4 ≈ 340
([Raspberry Pi Foundation](https://www.raspberrypi.com/news/benchmarking-raspberry-pi-5/)), against the x86 machine the speed table was measured on ([2970WX score](https://www.cpu-monkey.com/en/compare_cpu-amd_ryzen_threadripper_2970wx-vs-intel_core_i5_3340s)):

| | x86 reference | Pi 5 | Pi 4 |
|---|---|---|---|
| Geekbench 6 single-core | ~850 (estimated: 1,193 at the 2970WX's 4.2 GHz boost, scaled to the 3.0 GHz this machine runs at) | 764 | 340 |
| Native RTF, 1 thread | 0.15–0.16 (measured, loaded machine) | **≈ 0.17** | **≈ 0.38** |

The x86 RTF was measured on a busy machine (load average ~25), so this estimate leans pessimistic. The two estimates
agree: a Pi 5 renders roughly 4.5–8× faster than real time and a Pi 4 roughly 1.8–3.5× faster, on one core.

## Memory

Peak RSS of the native binary (measured on x86; the ARM build runs the same code with the same ONNX Runtime
release):

| | RSS |
|---|---|
| After loading the model | 69 MB |
| 20-sentence benchmark | 98 MB |
| Streaming a 53 s paragraph | 107 MB |

On disk: 13.9 MB of model files, the 0.46 MB binary and the 25 MB `libonnxruntime.so` (aarch64). The Python ONNX
runtime also works on a Pi (ONNX Runtime has aarch64 wheels) at about 30 MB more RSS.

## Building on the Pi

```bash
sudo apt install cmake g++ git
git clone https://github.com/WhissleAI/steer-tts && cd steer-tts/native
cmake -B build && cmake --build build -j4                 # downloads onnxruntime-linux-aarch64
pip install huggingface_hub && hf auth login       # if the repo asks you to accept its licence
huggingface-cli download WhissleAI/steer-tts-edge --local-dir model --include config.json --include "voices.*" --include "onnx/*" --include "native/*" --include events.safetensors
./build/steertts-edge --model model --bench --threads 1
./build/steertts-edge --model model "Hello from a Raspberry Pi." --stream -o - | aplay -f S16_LE -r 24000 -c 1
```

Needs 64-bit Raspberry Pi OS (or another aarch64 Linux). A Pi 3 / Zero 2 W (Cortex-A53, 512 MB–1 GB RAM) is not a
target.

## Reproducing the emulation

`native/tools/pi_emulation.sh` sets everything up without root: it fetches `qemu-user-static` from the distro
archive (unpacked, not installed), exports a Debian arm64 root filesystem with Docker (no ARM container is run),
cross-compiles with Zig, writes the emulated Pi `/proc/cpuinfo` and topology, then runs the parity test and the
instruction counts.
