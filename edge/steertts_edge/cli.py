"""steertts-edge command line (RELEASE, new in 1.0.0).

    steertts-edge "Hello there." -o hello.wav
    steertts-edge "I can't believe it worked." --voice voice_0042 --tag happiness --ctrl rate=1,f0_iqr=1.5 -o happy.wav
    steertts-edge "Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow>" --markup -o markup.wav
    steertts-edge --list voices | tags | controls

The package is looked up in --model, else $STEERTTS_EDGE_MODEL, else the current directory if it holds config.json,
else downloaded from the Hugging Face repo WhissleAI/steer-tts-edge (needs huggingface_hub).
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO_ID = "WhissleAI/steer-tts-edge"


def parse_ctrl(s):
    out = {}
    for kv in filter(None, (s or "").split(",")):
        k, sep, v = kv.partition("=")
        if not sep:
            raise SystemExit(f"--ctrl expects name=value pairs, got {kv!r}")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise SystemExit(f"--ctrl {k.strip()}: {v!r} is not a number")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="steertts-edge", description="SteerTTS v1 edge build: text to 24 kHz speech.")
    ap.add_argument("text", nargs="?", help="text to speak ('-' reads stdin)")
    ap.add_argument("-o", "--out", default="out.wav", help="output WAV (16-bit, 24 kHz)")
    ap.add_argument("--model", default=None, help="package directory or Hugging Face repo id")
    ap.add_argument("--voice", default=None, help="built-in voice name (default: the package default)")
    ap.add_argument("--tag", default="none", help="style tag (see --list tags)")
    ap.add_argument("--ctrl", default="", help="controls as name=z pairs, e.g. rate=1,loud=-0.5 (see --list controls)")
    ap.add_argument("--rate", type=float, default=None, help="shortcut for --ctrl rate=Z")
    ap.add_argument("--pace", type=float, default=1.0, help="global duration divisor (1.2 = 20%% faster)")
    ap.add_argument("--guidance", type=float, default=None, help="style guidance scale (1.0 = off)")
    ap.add_argument("--markup", action="store_true", help="parse inline markup (*emphasis*, [pause 300ms], <whisper>..)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads")
    ap.add_argument("--runtime", choices=["auto", "onnx", "torch"], default="auto",
                    help="onnx: ONNX Runtime (default when onnxruntime is installed); torch: PyTorch")
    ap.add_argument("--list", choices=["voices", "tags", "controls"], help="print names and exit")
    a = ap.parse_args(argv)

    runtime = a.runtime
    if runtime == "auto":
        try:
            import onnxruntime  # noqa: F401
            runtime = "onnx"
        except ImportError:
            runtime = "torch"
    if runtime == "onnx":
        from . import OnnxTTS as Runtime
    else:
        from . import EdgeTTS as Runtime
    model = a.model or os.environ.get("STEERTTS_EDGE_MODEL") or ("." if Path("config.json").exists() else REPO_ID)
    t0 = time.time()
    tts = Runtime.from_pretrained(model, threads=a.threads)
    if a.list:
        print("\n".join({"voices": tts.voices, "tags": tts.tags, "controls": tts.controls}[a.list]))
        return 0
    if a.text is None:
        ap.error("text is required (or --list)")
    text = sys.stdin.read() if a.text == "-" else a.text
    ctrl = parse_ctrl(a.ctrl)
    if a.rate is not None:
        ctrl["rate"] = a.rate
    load = time.time() - t0
    t1 = time.time()
    try:
        wav = tts(text, voice=a.voice, tag=a.tag, ctrl=ctrl or None, pace=a.pace, guidance=a.guidance,
                  markup=a.markup)
    except ValueError as e:
        print(f"steertts-edge: {e}", file=sys.stderr)
        return 2
    synth = time.time() - t1
    tts.save(wav, a.out)
    dur = len(wav) / tts.sample_rate
    print(f"{a.out}: {dur:.2f} s audio, load {load:.2f} s, synth {synth:.2f} s"
          + (f" (RTF {synth / dur:.3f})" if dur else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
