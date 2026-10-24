"""Reference renders for the native engine parity test, from the Python ONNX runtime (steertts_edge.OnnxTTS).

    python make_engine_golden.py <package_dir> <out_dir>

Writes <out_dir>/cases.jsonl (one request per line, plus "out": the raw float32 file and "mel_T" when known) and
<out_dir>/case_XXX.f32. The C++ test (tests/engine_parity.cpp) renders the same requests and compares.
"""
import json
import sys
from pathlib import Path

import numpy as np

PKG, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(PKG))
from steertts_edge.ort import OnnxTTS  # noqa: E402

CASES = [
    dict(text="Hi, thanks for calling. I can help you reschedule your appointment today."),
    dict(text="Are you coming with us tonight?", ctrl={"f0_end": 2, "f0_iqr": 1.5}),
    dict(text="The weather was cold, but the view from the top was worth every step.", ctrl={"rate": -1.5}),
    dict(text="The weather was cold, but the view from the top was worth every step.", ctrl={"rate": 1.5}),
    dict(text="Please keep this between the two of us.", tag="whisper"),
    dict(text="I really thought it would work this time.", tag="sadness", guidance=1.5),
    dict(text="Well, *that* was fast. [pause 400ms] <slow>Let me think about it.</slow> <whisper>Don't tell anyone.</whisper>",
         markup=True),
    dict(text="On 3/14/2025 the price rose 12.5% to $1,299.99.", voice="voice_0010",
         word_shift={"1": {"pitch": 3, "dur": 1.4}}),
    dict(text="It's 7:45 pm, call 555-0198 [laugh] now!", voice="voice_0005", tag="excited", pace=1.2),
    dict(text="x"),
    dict(text="   "),
    dict(text="First sentence here. Second one follows! And a third, with a question?", markup=True, tag="happiness",
         ctrl={"loud": 0.5}, guidance=2.0),
    dict(text="<high>Up here</high> and <low>down there</low>, <loud>loud</loud> then <soft>soft</soft>. [pause 1.2s] Done.",
         markup=True),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tts = OnnxTTS(PKG, threads=1)
    with open(OUT / "cases.jsonl", "w") as f:
        for i, c in enumerate(CASES):
            kw = dict(c)
            if "word_shift" in kw:
                kw["word_shift"] = {int(k): v for k, v in kw["word_shift"].items()}
            y = np.asarray(tts(**kw), np.float32)
            name = f"case_{i:03d}.f32"
            y.tofile(OUT / name)
            f.write(json.dumps(dict(c, out=name, n=int(len(y)))) + "\n")
            print(name, len(y), c["text"][:50])


if __name__ == "__main__":
    main()
