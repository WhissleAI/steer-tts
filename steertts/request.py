"""The synthesis request: one JSON object, documented by request.schema.json (shipped inside this package and at the
model repo root). normalize_request() validates a request against a loaded model and fills in defaults; it raises
ValueError with a precise message instead of silently ignoring a misspelt field, tag or control.
"""
import json
from importlib import resources

from .config import CONTROL_DIMS, CONTROL_RANGE

DEFAULTS = dict(pace=1.0, guidance=1.0, flow=True, seed=0, markup=False)
FIELDS = {"text", "voice", "ref_wav", "prompt", "tag", "ctrl", "pace", "guidance", "word_shift", "markup", "flow",
          "seed", "prompt_mapping", "$schema"}
WORD_SHIFT_KEYS = {"pitch": (-12.0, 12.0), "energy": (-20.0, 20.0), "dur": (0.25, 4.0)}


def schema():
    """The JSON Schema (draft 2020-12) of a request, as a dict."""
    return json.loads(resources.files(__package__).joinpath("request.schema.json").read_text())


def _num(name, v, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a number, got {v!r}")
    if not lo <= float(v) <= hi:
        raise ValueError(f"{name}={v} is outside [{lo}, {hi}]")
    return float(v)


def normalize_request(req, tts=None):
    """-> a new dict with defaults filled in. tts (a SteerTTS) enables voice / tag checks."""
    if not isinstance(req, dict):
        raise ValueError("a request is a JSON object")
    unknown = set(req) - FIELDS
    if unknown:
        raise ValueError(f"unknown request field(s) {sorted(unknown)}; see request.schema.json")
    out = {k: v for k, v in req.items() if k != "$schema"}
    text = out.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > 5000:
        raise ValueError("text is longer than 5000 characters; split it (or stream it sentence by sentence)")
    if out.get("ref_wav") is not None:
        if out.get("voice"):
            raise ValueError("give either voice (a built-in voice) or ref_wav (clone a reference clip), not both")
    else:
        out.pop("ref_wav", None)
        out["voice"] = out.get("voice") or (tts.default_voice if tts is not None else None)
        if tts is not None and out["voice"] not in tts.spk_ref:
            raise ValueError(f"unknown voice {out['voice']!r}; list them with tts.voices / `steertts voices`")
    if out.get("prompt") is not None and not isinstance(out["prompt"], str):
        raise ValueError("prompt must be a string")
    if out.get("tag") is not None:
        if not isinstance(out["tag"], str):
            raise ValueError("tag must be a string")
        if tts is not None and out["tag"] not in tts.tag2i:
            raise ValueError(f"unknown tag {out['tag']!r}; known: {', '.join(tts.tags)}")
    ctrl = out.get("ctrl") or {}
    if not isinstance(ctrl, dict):
        raise ValueError("ctrl must be an object {control: z}")
    for k in ctrl:
        if k not in CONTROL_DIMS:
            raise ValueError(f"unknown control {k!r}; known: {', '.join(CONTROL_DIMS)}")
    out["ctrl"] = {k: _num(f"ctrl.{k}", v, *CONTROL_RANGE) for k, v in ctrl.items()}
    defaults = dict(DEFAULTS)
    if tts is not None:  # the loaded model's own defaults (SteerTTS(guidance=..., flow=...))
        defaults.update(guidance=tts.guidance, flow=bool(tts.use_flow))
    for k, v in defaults.items():
        if out.get(k) is None:
            out[k] = v
    out["pace"] = _num("pace", out["pace"], 0.25, 4.0)
    out["guidance"] = _num("guidance", out["guidance"], 0.0, 4.0)
    for k in ("flow", "markup"):
        if not isinstance(out[k], bool):
            raise ValueError(f"{k} must be true or false")
    if isinstance(out["seed"], bool) or not isinstance(out["seed"], int) or out["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    ws = out.get("word_shift")
    if ws:
        if not isinstance(ws, dict):
            raise ValueError("word_shift must be an object {word_index: {pitch, energy, dur}}")
        if out["markup"]:
            raise ValueError("word_shift and markup cannot be combined (markup re-indexes words per segment)")
        norm = {}
        for w, sh in ws.items():
            try:
                wi = int(w)
            except (TypeError, ValueError):
                raise ValueError(f"word_shift key {w!r} is not a word index") from None
            if wi < 0 or not isinstance(sh, dict) or set(sh) - set(WORD_SHIFT_KEYS):
                raise ValueError(f"word_shift[{w!r}] must be a non-negative index -> {{pitch, energy, dur}}")
            norm[str(wi)] = {k: _num(f"word_shift.{w}.{k}", v, *WORD_SHIFT_KEYS[k]) for k, v in sh.items()}
        out["word_shift"] = norm
    else:
        out.pop("word_shift", None)
    return out
