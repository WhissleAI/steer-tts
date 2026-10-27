# steertts-edge

Python runtimes for SteerTTS edge, the 12 MB build of [SteerTTS v1](../README.md). The weights, voices and text
front-end assets are on Hugging Face: [WhissleAI/steer-tts-edge](https://huggingface.co/WhissleAI/steer-tts-edge).

```bash
pip install "steertts-edge[hub] @ git+https://github.com/WhissleAI/steer-tts#subdirectory=edge"
steertts-edge "Are you coming with us tonight?" --ctrl f0_end=2,f0_iqr=1.5 -o question.wav
```

```python
from steertts_edge import OnnxTTS          # ONNX Runtime (default); EdgeTTS = the PyTorch runtime ([torch] extra)
tts = OnnxTTS.from_pretrained("WhissleAI/steer-tts-edge")
wav = tts("I can't believe it worked.", tag="happiness", ctrl={"rate": 1.0})
```

Longer text as a performance, planned sentence by sentence by the director (`director.bin`, numpy only):

```python
from steertts_edge import Director
d = Director("path/to/steer-tts-edge")     # the package directory, or director.bin
texts = ["“Who are you?” she asked.", "Nobody answered."]
for text, plan in zip(texts, d.plan(texts, genre="children")):
    wav = tts(text, ctrl=d.ctrl(plan)); gap_ms = Director.pause_ms(plan)
```

- [Python API and CLI](../docs/python.md#steertts_edge-edge)
- [The request](../docs/request.md): controls, tags, guidance, markup
- [Models, quality, memory and speed](../docs/models.md)

| Path | What |
|---|---|
| `steertts_edge/` | The package: `OnnxTTS` (`ort.py`), `EdgeTTS` (`pipeline.py`, PyTorch), `Director` (`director.py`), text front end (`text.py`, `edge_front/`), markup, CLI |
| `tests/` | pytest suite; runs every functional test on each installed runtime. `STEERTTS_EDGE_MODEL=<package dir> pytest tests` |
| `tools/export_onnx.py` | Export the PyTorch edge model to the three ONNX graphs (fp32) |
| `tools/pack_onnx.py` | Swap every Linear / Conv1d / AdaLN weight for `MatMulNBits` with the package's own int4 / int8 codes, compact the graphs |
| `tools/bench_speed.py` | Speed and memory: `--runtime onnx torch native` |

Rebuilding the ONNX graphs from a package directory:

```bash
python tools/export_onnx.py <package_dir> /tmp/fp32
python tools/pack_onnx.py <package_dir> /tmp/fp32 <package_dir>/onnx
```

Code MIT; weights CC BY-NC 4.0 (see the model card).
