# SteerTTS in C

A stand-alone C11 library (`libsteertts`, one header, no dependencies beyond libc, libm and threads) and a `steertts`
command line that run both SteerTTS models: v1 (with flow refiner, free-text prompts and voice cloning; fp32 or
int8) and the int4/int8 edge build. Full documentation: [docs/c.md](../docs/c.md).

```bash
cmake -B build && cmake --build build -j          # or: make
hf download WhissleAI/steer-tts-edge --local-dir models/steer-tts-edge \
    --include config.json --include "*.safetensors" --include voices.json --include "native/*"
./build/steertts --model models/steer-tts-edge "Hello from C." -o hello.wav
./build/steertts --model models/steer-tts-edge "Hello from C." --play     # to the speakers (sox, aplay, ...)
./build/steertts --model models/steer-tts-edge -i                         # keep it loaded, type lines to speak
```

```c
#include "steertts.h"
char err[512];
steertts_engine* e = steertts_create("models/steer-tts-edge", 1, err, sizeof err);
steertts_options o;
steertts_default_options(&o);
o.ctrl = "f0_end=2,f0_iqr=1.5";
float* audio;
size_t n;
if (steertts_synthesize(e, "Are you coming with us tonight?", &o, &audio, &n, err, sizeof err) == 0) {
  steertts_save_wav(audio, n, "question.wav");
  steertts_free(audio);
}
steertts_destroy(e);
```

v1 additionally needs its helper models converted once (`python tools/export_assets.py <v1 dir>/native --front
<edge dir>/native/front`); see [docs/c.md](../docs/c.md#models).

Tests: `cmake -B build && cmake --build build && (cd build && ctest)`, with the models in `models/steer-tts` and
`models/steer-tts-edge` (or `STEERTTS_V1_MODEL` / `STEERTTS_EDGE_MODEL`); tests whose model is missing are skipped.
