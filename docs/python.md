# Python

Two packages live in this repo: `steertts` (the full v1 model, repo root) and `steertts_edge` (the 12 MB edge build,
[`edge/`](../edge)). Both return 24 kHz mono float32 numpy arrays and take the same [request](request.md) fields.

## steertts (v1)

```bash
pip install "steertts[all] @ git+https://github.com/WhissleAI/steer-tts"   # core: torch, vocos, g2p_en ...
# [all] adds transformers (prompts, cloning), parselmouth (measuring), sounddevice (playback)
```

The weights are on Hugging Face at [WhissleAI/steer-tts](https://huggingface.co/WhissleAI/steer-tts). The repo asks
you to accept its licence terms once (CC BY-NC 4.0): log in with `huggingface-cli login` and accept on the model
page before the first download.

```python
from steertts import SteerTTS

tts = SteerTTS.from_pretrained("WhissleAI/steer-tts")              # cuda if available, else cpu
tts8 = SteerTTS.from_pretrained("WhissleAI/steer-tts", precision="int8")   # 52 MB download instead of 215 MB

wav = tts("The meeting moved to Thursday.")                          # default voice voice_1688
wav = tts("Are you serious?", voice="voice_0042", ctrl={"f0_end": 2.0, "f0_iqr": 1.5})
wav = tts("It is over.", tag="sadness", ctrl={"loud": -1}, guidance=1.5)
wav = tts("Now in your voice.", ref_wav="consenting_speaker.wav")    # cloning (low similarity, see the model card)
wav = tts("Quick reply.", flow=False)                                # skip the flow refiner: ~12x less compute

req = tts.resolve({"text": "See you soon.", "prompt": "excited, a bit faster"})
print(req)                       # {'text': ..., 'tag': 'excited', 'ctrl': {'rate': 0.29}, 'prompt_mapping': {...}}
wav = tts.synthesize(req)        # a resolved request renders exactly as the prompt would

for chunk in tts.stream("First sentence. Second sentence."):        # sentence-level streaming
    play(chunk.audio)
tts.save(wav, "out.wav")
```

CLI: `steertts say "Hello." -o hello.wav`, `steertts play "..."` (streams to the speakers), `steertts voices`,
`steertts measure out.wav --text "..."` (reads the 12 controls back from audio), `steertts schema`.

## steertts_edge (edge)

```bash
pip install "steertts-edge[hub] @ git+https://github.com/WhissleAI/steer-tts#subdirectory=edge"    # ONNX Runtime
pip install "steertts-edge[hub,torch] @ git+https://github.com/WhissleAI/steer-tts#subdirectory=edge"  # + PyTorch runtime
```

The weights are at [WhissleAI/steer-tts-edge](https://huggingface.co/WhissleAI/steer-tts-edge); if the repo asks you
to accept its licence, log in once (`hf auth login`) and accept on the model page. `from_pretrained` downloads only
what the chosen runtime reads.

```python
from steertts_edge import OnnxTTS      # ONNX Runtime, no torch import (recommended)
# from steertts_edge import EdgeTTS    # PyTorch reference runtime, same API, same audio up to float rounding

tts = OnnxTTS.from_pretrained("WhissleAI/steer-tts-edge", threads=1)     # or OnnxTTS("path/to/package")
wav = tts("I can't believe it worked.", tag="happiness", ctrl={"rate": 1.0})
wav = tts("Well, *that* was fast. [pause 400ms] <whisper>Don't tell anyone.</whisper>", markup=True)
for chunk in tts.stream("First sentence. Second sentence. <slow>Third.</slow>"):
    play(chunk.audio)                  # chunk.segment: text, tag, ctrl; chunk.synth_ms
tts.save(wav, "out.wav")
print(tts.voices[:5], tts.tags[:5], tts.controls)
```

CLI: `steertts-edge "Hello there." -o hello.wav [--runtime onnx|torch] [--voice ...] [--tag ...] [--ctrl rate=1]
[--markup] [--threads N]`, `steertts-edge --list voices`.

**Determinism.** A request renders the same samples every time on the same device and thread count. Different
thread counts (ONNX Runtime) or devices differ at the 1e-7 level.

**Memory.** Use `stream()` for long text: it renders one sentence at a time, so memory follows the longest sentence,
not the whole text. A single call on a long paragraph renders it in one pass (attention memory grows with the square
of its length).

## Tests

```bash
STEERTTS_MODEL=<local v1 dir> pytest tests                                   # v1
cd edge && STEERTTS_EDGE_MODEL=<local edge package> pytest tests             # edge: every installed runtime
```
