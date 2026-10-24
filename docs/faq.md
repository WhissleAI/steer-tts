# FAQ

**I can't hear the samples on Hugging Face.**
The Hugging Face repos ask visitors to log in and accept the licence terms (CC BY-NC 4.0), which also hides their
audio. All samples are on the [samples page](https://whissleai.github.io/steer-tts/), no login needed. To download
the weights, run `hf auth login` (or `huggingface-cli login`) and accept the terms on the model page once.

**The edge model is 12 MB. Why did the Python package use 370 MB of RAM?**
12 MB is the download. In the 1.0.0 PyTorch runtime most of the memory was PyTorch itself (~200 MB before any model
loads), plus the int4 weights unpacked to fp32 (59 MB), plus working memory that grew with the length of the text.
Since 1.1.0 the default runtime is ONNX Runtime: weights stay packed in memory (int4 / int8, unpacked block by block
inside the matmul), and `stream()` renders one sentence at a time. Measured peak RSS:

| Runtime | Peak RSS |
|---|---|
| Python + PyTorch (1.0.0 default) | 369 MB |
| Python + ONNX Runtime (1.1.0 default) | 127 MB |
| native C++ + ONNX Runtime | 98 MB (69 MB after load) |
| [C library](c.md) | 27 MB |

(20-sentence benchmark, one thread; streaming a 53 s paragraph: 136 MB in Python + ONNX Runtime, 107 MB native. The
C library keeps the packed weights memory-mapped and needs no runtime of its own.)

**Is it real-time on a Raspberry Pi?**
See [raspberry-pi.md](raspberry-pi.md) for what was measured (the ARM64 build, under emulation, renders the same
audio as x86) and what was estimated (speed on Pi 4 / Pi 5 hardware).

**Is it fast?**
On a desktop CPU core the edge build renders many times faster than real time; time to first audio is the time to
render the first sentence. See [models.md](models.md#memory-and-speed-edge) for numbers and how they were measured.

**Why are renders slightly different between the ONNX and PyTorch runtimes, or across thread counts?**
Different kernels sum in a different order. Durations (and so sample counts) are identical; the mel spectrograms
agree to ~3·10⁻⁵ and the waveforms at 50–120 dB SNR, far below audibility. For a fixed runtime, device and thread
count, renders are deterministic.

**Can it clone a voice / follow a free-text prompt on device?**
Yes, with v1 in the [C library](c.md), which runs MiniLM and WavLM itself (fp32 or int8). The Python edge build and the
native ONNX runtime do not; with those, resolve prompts to `tag` + `ctrl` on a host and pass a host-computed speaker
vector as `spk_vec` for cloning. Cloning similarity is low in every build (see the v1 model card).

**Do the emotion tags work?**
Partly. They act as style switches; voice-quality and delivery tags (`whisper`, `loud`, `slow`, `high_pitch`) work
strongly, but an independent emotion classifier recognises only `sadness` among the emotions. Combine tags with
explicit controls for a dependable effect ([request.md](request.md#style-tags)).

**Other languages?** English only (ARPAbet front end).

**Can I use it commercially?**
The code is MIT. The weights are CC BY-NC 4.0 (non-commercial use only).

**How do I report a bad render?**
Open an [issue](../../issues) with the exact request (text, voice, tag, ctrl, build and runtime) and the audio.
