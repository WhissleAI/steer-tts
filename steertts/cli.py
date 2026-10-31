"""steertts command line.

  steertts say "Hello there." -o hello.wav                                    # default voice, default style
  steertts say "I can't believe it." --voice voice_1688 --tag anger --ctrl loud=1.5,f0_med=1 -o angry.wav
  steertts say "Keep it down." --prompt "whispering and slow" --show-request -o soft.wav
  steertts say "Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow>" --markup -o m.wav
  steertts say "Now in your voice." --ref-wav me.wav -o clone.wav
  steertts say --request req.json -o out.wav                                  # a JSON request (request.schema.json)
  steertts stream "First sentence. Second sentence." -o s.wav                 # prints per-chunk timing
  steertts play "First sentence. Second sentence." --voice voice_1688         # speak through the speakers, streaming
  steertts say "Hello there." --play                                         # play instead of (or with -o, as well as) saving
  steertts voices [--gender female] | steertts tags | steertts schema | steertts prompt "soft and sad"
  steertts measure out.wav --text "..." --voice voice_1688                    # read the 12 controls back (z)

The model directory comes from --model, else $STEERTTS_MODEL, else the current directory if it holds config.json,
else WhissleAI/steer-tts from Hugging Face. Playback (--play / play) uses sounddevice, ffplay or aplay.
"""
import argparse
import json
import os
import sys
import time


def _model_dir(a):
    if a.model:
        return a.model
    if os.environ.get("STEERTTS_MODEL"):
        return os.environ["STEERTTS_MODEL"]
    if os.path.exists("config.json") and os.path.exists("voices.json"):
        return "."
    print("steertts: no local model; using WhissleAI/steer-tts from Hugging Face (--model DIR to change)", file=sys.stderr)
    return "WhissleAI/steer-tts"


def _ctrl(s):
    out = {}
    for kv in filter(None, (s or "").split(",")):
        k, _, v = kv.partition("=")
        out[k.strip()] = float(v)
    return out


def _load(a, **kw):
    from . import SteerTTS
    t0 = time.time()
    tts = SteerTTS.from_pretrained(_model_dir(a), device=a.device, flow=not a.no_flow, precision=a.precision, **kw)
    print(f"loaded {tts} in {time.time() - t0:.1f}s", file=sys.stderr)
    return tts


def _request(a):
    req = json.load(open(a.request)) if a.request else {}
    for k, v in dict(text=a.text, voice=a.voice, ref_wav=a.ref_wav, prompt=a.prompt, tag=a.tag,
                     pace=a.pace, guidance=a.guidance, seed=a.seed).items():
        if v is not None:
            req[k] = v
    if a.ctrl:
        req["ctrl"] = {**req.get("ctrl", {}), **_ctrl(a.ctrl)}
    if getattr(a, "markup", False):
        req["markup"] = True
    if a.no_flow:
        req["flow"] = False
    return req


def main(argv=None):
    ap = argparse.ArgumentParser(prog="steertts", description="SteerTTS: steerable English TTS")
    ap.add_argument("--model", help="model directory or HF repo id (default: $STEERTTS_MODEL or .)")
    ap.add_argument("--device", help="cpu | cuda (default: cuda if available)")
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32",
                    help="int8: the 4x smaller weight files in int8/ (see the model card for the measured differences)")
    ap.add_argument("--no-flow", action="store_true", help="skip the flow-matching refiner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("say", "stream", "play"):
        p = sub.add_parser(name)
        p.add_argument("text", nargs="?")
        p.add_argument("--request", help="JSON request file (fields as in request.schema.json)")
        p.add_argument("--voice")
        p.add_argument("--ref-wav", dest="ref_wav", help="clone this reference clip (consent required)")
        p.add_argument("--prompt", help="free-text direction, e.g. 'soft and sad'")
        p.add_argument("--tag")
        p.add_argument("--ctrl", help="comma list dim=z, e.g. rate=1.5,f0_med=-1")
        p.add_argument("--pace", type=float)
        p.add_argument("--guidance", type=float)
        p.add_argument("--seed", type=int)
        p.add_argument("--markup", action="store_true")
        p.add_argument("--show-request", action="store_true", help="print the resolved request JSON")
        p.add_argument("--play", action="store_true", help="play through the speakers as chunks are rendered")
        p.add_argument("-o", "--out", help="output wav (default out.wav unless playing)")
    p = sub.add_parser("voices")
    p.add_argument("--gender", choices=["female", "male"])
    p.add_argument("--json", action="store_true")
    sub.add_parser("tags")
    sub.add_parser("schema")
    p = sub.add_parser("prompt")
    p.add_argument("prompt")
    p = sub.add_parser("measure")
    p.add_argument("wav")
    p.add_argument("--text")
    p.add_argument("--voice")
    a = ap.parse_args(argv)

    if a.cmd == "schema":
        from .request import schema
        print(json.dumps(schema(), indent=1))
        return
    if a.cmd in ("voices", "tags"):
        from .pipeline import resolve_model_dir
        d = resolve_model_dir(_model_dir(a))
        if a.cmd == "tags":
            print(" ".join(json.load(open(d / "control_stats.json"))["tags"]))
            return
        vs = json.load(open(d / "voices.json"))["voices"]
        vs = [v for v in vs if not a.gender or v.get("gender") == a.gender]
        if a.json:
            print(json.dumps(vs, indent=1))
        else:
            for v in vs:
                print(f"{v['name']}  {v['description']}")
        return
    if a.cmd == "measure":
        from .audio import load_wav
        from .measure import measure, measure_z, n_phones
        from .pipeline import resolve_model_dir
        y, _ = load_wav(a.wav)
        if a.voice:
            st = json.load(open(resolve_model_dir(_model_dir(a)) / "control_stats.json"))
            z = measure_z(y, a.text, st["spk_mean"][a.voice], st["pooled_sd"])
            print(json.dumps({k: round(v, 2) for k, v in z.items()}))
        else:
            from .config import CONTROL_DIMS
            raw = measure(y, n_phones=n_phones(a.text) if a.text else None, want_mel=False)["controls"]
            print(json.dumps({k: round(float(v), 3) for k, v in zip(CONTROL_DIMS, raw)}))
        return
    tts = _load(a)
    if a.cmd == "prompt":
        print(json.dumps(tts.map_prompt(a.prompt), indent=1))
        return
    req = _request(a)
    if not req.get("text"):
        sys.exit("steertts: no text (positional argument or --request)")
    req = tts.resolve(req)
    if getattr(a, "show_request", False) or "prompt_mapping" in req:
        print(json.dumps(req, indent=1), file=sys.stderr)
    import numpy as np
    play = a.play or a.cmd == "play"
    out = a.out or (None if play else "out.wav")
    player = None
    if play:
        from .player import Player
        player = Player(tts.sample_rate)
        print(f"playing via {player.name}", file=sys.stderr)
    if a.cmd == "say":
        t0 = time.time()
        wav = tts.synthesize(req)
        dt = time.time() - t0
        if player:
            player.play(wav)
    else:
        if req.get("word_shift"):
            sys.exit("steertts: word_shift is not supported when streaming; use inline markup (*word*, <high>word</high>)")
        t0, chunks = time.time(), []
        try:
            for i, c in enumerate(tts.stream(request=req)):
                chunks.append(c.audio)
                if player:
                    player.play(c.audio)
                print(f"chunk {i}: {len(c.audio) / tts.sample_rate:.2f}s audio at +{(time.time() - t0) * 1000:.0f} ms "
                      f"({c.segment.text or ('pause %d ms' % c.segment.pause_ms)!r})", file=sys.stderr)
        except KeyboardInterrupt:
            print("stopped", file=sys.stderr)
        wav = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
        dt = time.time() - t0
    if out:
        tts.save(wav, out)
    print(f"{out or 'played'}: {len(wav) / tts.sample_rate:.2f}s audio rendered in {dt:.2f}s", file=sys.stderr)
    if player:
        player.close()
        if player.err:
            sys.exit(f"steertts: playback failed: {player.err}")


if __name__ == "__main__":
    main()
