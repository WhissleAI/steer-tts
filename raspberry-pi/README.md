# Run on Raspberry Pi

SteerTTS's edge model on a Raspberry Pi 4 or 5 (64-bit Raspberry Pi OS or Ubuntu), through the dependency-free
[C library](../docs/c.md): one header, a static or shared library, no Python.

## Install

```bash
git clone https://github.com/WhissleAI/steer-tts && cd steer-tts
raspberry-pi/install.sh              # tuned build for this board, tests, install to ~/.local, download the model
raspberry-pi/install.sh --service    # ... and run the steertts-pi server at boot (http://127.0.0.1:8080)
```

By hand: `cd c && cmake --preset pi5 && cmake --build --preset pi5` (`pi4` for a Pi 4), then
`sh tools/get_models.sh edge`. Cross-compiling from x86-64 with Docker: `raspberry-pi/cross/cross-build.sh pi5`.
If the model repo asks you to accept its licence, accept it on [its page](https://huggingface.co/WhissleAI/steer-tts-edge)
and `export HF_TOKEN=<read token>` first.

## C API

```c
#include "steertts.h"

char err[512];
steertts_engine* e = steertts_create("steer-tts-edge", 0, err, sizeof err);   /* 0 threads = every core */
steertts_options o;
steertts_default_options(&o);
o.voice = steertts_gender_voice(e, "female");
o.tag = "happiness";
o.ctrl = "f0_end=2,f0_iqr=1.5";
float* audio;
size_t n;
if (steertts_synthesize(e, "Are you coming with us tonight?", &o, &audio, &n, err, sizeof err) == 0) {
  steertts_save_wav(audio, n, "question.wav");
  steertts_free(audio);
}
steertts_destroy(e);
```

```bash
cc hello.c $(pkg-config --cflags --libs steertts) -o hello      # ~/.local install: PKG_CONFIG_PATH=~/.local/lib/pkgconfig
```

[`examples/hello.c`](examples/hello.c) is this program; [`examples/streaming.c`](examples/streaming.c) plays while it
renders (`steertts_stream_ex` + `steertts_set_first_chunk_ms`). API reference: [docs/c.md](../docs/c.md#c-api).

## steertts-pi

An HTTP / stdin server on the same API ([`server/`](server)):

```bash
steertts-pi --model ~/.local/share/steertts/models/steer-tts-edge
curl -s 'localhost:8080/synthesize?text=Hello+there.&ctrl=f0_end=2' -o hello.wav
curl -sN localhost:8080/stream -d '{"text": "First sentence. <whisper>Second.</whisper>"}' | aplay
```

`POST /synthesize` and `/stream` take the usual JSON [request](../docs/request.md); `GET /voices`, `/tags`, `/controls`,
`/health`; `--stdin` reads one request per line. `steertts-pi --help` lists the options.

## Measured on a Raspberry Pi 5

Pi 5 (4 GB), Raspberry Pi OS trixie, GCC 14, wall supply, no cooler. RTF = render time / audio length (lower is
faster); `bench.sh` reproduces it.

| | 1 thread | 4 threads |
|---|---|---|
| Real-time factor, peak | 0.092 (11× real time) | 0.029 (34×) |
| Real-time factor, 10 min sustained | 0.118 (8.5×, throttled) | 0.050 (20×, throttled at 85–90 °C) |
| First audio, streaming (short / long first sentence) | 0.10 / 0.53 s | 0.03 / 0.16 s |
| Memory (peak RSS) | 26 MB | 26 MB |

**A Pi 5 needs a cooler** for sustained use (without one it throttles even at 1 thread) (and its 27 W supply: on a laptop USB port it under-volts;
`bench.sh` and `install.sh` warn about it).
