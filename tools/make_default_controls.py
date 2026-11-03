#!/usr/bin/env python3
"""Write the per-voice default profile ("default_controls") into a package config.json.

    python tools/make_default_controls.py PKG_DIR [--out config.json] [--extrapolate]

Measured entries (2026-11-02; CPU sweep on 30 held-out assistant-style sentences, UTMOS22 +
whisper-small.en): the default voice and the nano voices. With --extrapolate every other voice whose neutral
articulation rate (control_stats.json spk_mean.rate, or the edge build's voices.safetensors spk_mean[:, 0]) is
below TARGET_RATE gets pace = TARGET_RATE / rate, capped at PACE_CAP (unmeasured: listen before shipping).
The block lives under its own key so it merges with other config keys (e.g. "output").
"""
import argparse, json
from pathlib import Path

MEASURED = {                     # voice -> pace multiplier (no ctrl offsets: a requested rate overrides the rate a tag picks)
    "voice_1688": 1.10,          # full + edge default voice: 4.54 -> 4.99 syl/s, dUTMOS -0.03 +- 0.05 (full)
    "nano_01": 1.12,             # nano: 4.48 -> ~5.0 syl/s (1.10: 4.93, 1.20: 5.39 measured)
    "nano_02": 1.14,             # nano: 4.42 -> 5.03 syl/s, dUTMOS -0.02 +- 0.06
}
TARGET_RATE = 5.769              # phones/s (control_stats units) = 1.10 x voice_1688's neutral rate (29th percentile)
PACE_CAP = 1.15


def rates(pkg):
    st = pkg / "control_stats.json"
    if st.exists():
        return {v: s["rate"] for v, s in json.load(open(st))["spk_mean"].items()}
    vs = pkg / "voices.safetensors"
    if vs.exists():
        from safetensors.numpy import load_file
        names = [v["name"] for v in json.load(open(pkg / "voices.json"))["voices"]]
        return dict(zip(names, load_file(str(vs))["spk_mean"][:, 0].tolist()))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg"); ap.add_argument("--out"); ap.add_argument("--extrapolate", action="store_true")
    a = ap.parse_args()
    pkg = Path(a.pkg)
    raw = (pkg / "config.json").read_text()
    cfg = json.loads(raw)
    known = set(cfg.get("voices_order") or cfg.get("voices") or [])
    if not known and (pkg / "voices.json").exists():
        known = {v["name"] for v in json.load(open(pkg / "voices.json"))["voices"]}
    voices = {}
    if a.extrapolate:
        for v, r in rates(pkg).items():
            if v in known and r < TARGET_RATE:
                voices[v] = {"pace": round(min(TARGET_RATE / r, PACE_CAP), 3)}
    for v, p in MEASURED.items():
        if v in known:
            voices[v] = {"pace": p}
    cfg["default_controls"] = {
        "version": 1,
        "voices": dict(sorted(voices.items())),
    }
    out = Path(a.out) if a.out else pkg / "config.json"
    if "\n" in raw.strip():  # keep the file's own layout: indented, or one compact line
        text = json.dumps(cfg, indent=1, ensure_ascii=False) + ("\n" if raw.endswith("\n") else "")
    else:
        text = json.dumps(cfg, separators=(",", ":"), ensure_ascii=False)
    out.write_text(text)
    print(f"{out}: default_controls for {len(voices)} voices")


if __name__ == "__main__":
    main()
