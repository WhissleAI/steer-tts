"""SteerTTS examples. Run after `pip install ".[all]"`; uses the model next to this file, else $STEERTTS_MODEL, else WhissleAI/steer-tts.

python inference.py                  # renders every example below into ./samples/
python inference.py --device cpu
"""
import argparse
import json
import os
from pathlib import Path

from steertts import SteerTTS

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    local = str(HERE) if (HERE / "config.json").exists() else "WhissleAI/steer-tts"
    ap.add_argument("--model", default=os.environ.get("STEERTTS_MODEL", local))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--ref-wav", help="a consenting speaker's recording, for the cloning example")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    tts = SteerTTS.from_pretrained(a.model, device=a.device)
    print(tts, f"{len(tts.voices)} voices, default {tts.default_voice}")

    # 1. plain text, default voice
    tts.save(tts("The meeting has been moved to Thursday afternoon at three."), out / "01_default.wav")
    # 2. explicit controls (speaker-relative z): faster and higher, rising at the end
    tts.save(tts("Are you coming with us tonight?", ctrl={"rate": 1.0, "f0_med": 1.0, "f0_end": 2.0}), out / "02_ctrl.wav")
    # 3. a style tag plus a control, stronger steering via guidance
    tts.save(tts("I really thought it would work this time.", tag="sadness", ctrl={"loud": -1.0}, guidance=1.5),
             out / "03_tag_guidance.wav")
    # 4. another voice (see voices.json / `steertts voices`)
    v = tts.voices[42]
    print(v, tts.voice_info[v]["description"])
    tts.save(tts("Good evening, and welcome to the program.", voice=v), out / "04_other_voice.wav")
    # 5. a free-text direction -> the request it maps to (needs the `prompt` extra)
    req = tts.resolve({"text": "Please keep this between the two of us.", "prompt": "whispering, slow and secretive"})
    print(json.dumps(req, indent=1))
    tts.save(tts.synthesize(req), out / "05_prompt.wav")
    # 6. inline markup, streamed sentence by sentence
    text = "Well, *that* was fast. [pause 400ms] <slow>Let me think about it.</slow> <whisper>Don't tell anyone.</whisper>"
    chunks = []
    for c in tts.stream(text):
        print(f"  chunk {c.segment.tag:8s} {len(c.audio) / tts.sample_rate:.2f}s rendered in {c.synth_ms:.0f} ms")
        chunks.append(c.audio)
    import numpy as np
    tts.save(np.concatenate(chunks), out / "06_markup_stream.wav")
    # 7. per-word prosody without markup: word 1 ("never", 0-based) higher, louder and longer
    tts.save(tts("I never said she stole it.", word_shift={1: {"pitch": 3, "energy": 4, "dur": 1.3}}), out / "07_word_shift.wav")
    # 8. zero-shot cloning (needs the `clone` extra and a CONSENTING speaker's clip)
    if a.ref_wav:
        tts.save(tts("This is a cloned voice reading a new sentence.", ref_wav=a.ref_wav), out / "08_clone.wav")
    print("wrote", sorted(p.name for p in out.glob("*.wav")))


if __name__ == "__main__":
    main()
