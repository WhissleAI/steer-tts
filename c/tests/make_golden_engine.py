"""End-to-end references for tests/test_engine.c: full requests rendered by the Python packages.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<repo>:<repo>/edge python make_golden_engine.py \
        --v1 <v1 model dir> --edge <edge package> --out golden/engine

The C library has one text front end (the edge front end, a port of steertts_edge.text); the v1 Python package uses
g2p_en. To compare the models and not the G2P backends, the v1 references are rendered with steertts_edge.text's
to_phonemes patched into the v1 pipeline (the acoustic model, flow, vocoder, prompt mapper and speaker encoder are
v1's own). Each output file holds one float32 waveform per case and the requests as JSON metadata.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

V1_CASES = [
    {"text": "The meeting moved to Thursday."},
    {"text": "Are you serious?", "voice": "voice_0042", "ctrl": {"f0_end": 2.0, "f0_iqr": 1.5}},
    {"text": "It is over.", "tag": "sadness", "ctrl": {"loud": -1}, "guidance": 1.5},
    {"text": "Quick reply without the refiner.", "flow": False},
    {"text": "Well, *that* was fast. [pause 300ms] <whisper>Don't tell anyone.</whisper>", "markup": True},
    {"text": "I said the blue one.", "word_shift": {"3": {"pitch": 3, "dur": 1.3}}},
    {"text": "Faster, with another seed.", "pace": 1.3, "seed": 7},
    {"text": "See you soon.", "prompt": "excited, a bit faster"},
    {"text": "Now in a cloned voice.", "ref_wav": "REF"},
]
V1_INT8_CASES = [V1_CASES[0], V1_CASES[2], V1_CASES[7]]
EDGE_CASES = [
    {"text": "The meeting moved to Thursday."},
    {"text": "Are you serious?", "voice": "voice_0010", "ctrl": {"f0_end": 2.0, "f0_iqr": 1.5}},
    {"text": "It is over.", "tag": "sadness", "ctrl": {"loud": -1}, "guidance": 1.5},
    {"text": "Well, *that* was fast. [pause 300ms] <whisper>Don't tell anyone.</whisper>", "markup": True},
    {"text": "I said the blue one.", "word_shift": {"3": {"pitch": 3, "dur": 1.3}}},
    {"text": "...", "pace": 1.0},
]


def write(path, cases, wavs):
    save_file({f"case{i}": np.ascontiguousarray(w, np.float32) for i, w in enumerate(wavs)}, str(path),
              metadata={"requests": json.dumps(cases)})
    print(path, [len(w) for w in wavs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--edge", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "golden" / "engine"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    import steertts_edge.text as et
    et.set_assets(Path(a.edge) / "front")
    import steertts.markup as M
    import steertts.pipeline as P
    P.to_phonemes = M.to_phonemes = et.to_phonemes   # see the module docstring

    from steertts import SteerTTS
    ref = out / "ref_voice.wav"
    v1 = SteerTTS(a.v1, device="cpu")
    if not ref.exists():   # a reference clip for cloning: a render of another built-in voice, 16-bit PCM
        import soundfile as sf
        y = v1.synthesize({"text": "This is the reference speaker, talking for a few seconds about nothing in "
                                   "particular.", "voice": "voice_0100"})
        sf.write(str(ref), y, 24000, subtype="PCM_16")
    cases = [dict(c, ref_wav=str(ref)) if c.get("ref_wav") == "REF" else c for c in V1_CASES]
    write(out / "v1_fp32.safetensors", [dict(c, ref_wav="ref_voice.wav") if "ref_wav" in c else c for c in cases],
          [v1.synthesize(c) for c in cases])
    v8 = SteerTTS(a.v1, device="cpu", precision="int8")
    write(out / "v1_int8.safetensors", V1_INT8_CASES, [v8.synthesize(c) for c in V1_INT8_CASES])

    from steertts_edge import EdgeTTS
    ed = EdgeTTS(a.edge)
    wavs = []
    for c in EDGE_CASES:
        kw = {k: v for k, v in c.items() if k != "text"}
        wavs.append(ed(c["text"], **kw))
    write(out / "edge.safetensors", EDGE_CASES, wavs)


if __name__ == "__main__":
    main()
