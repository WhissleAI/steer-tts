"""Per-voice default profile ("default_controls" in config.json), applied underneath every request.

The controls are z-scores relative to each voice's own neutral speech, so a voice that is a slow speaker in the
training data stays slow at z = 0. A profile corrects that per voice without changing what a request means:

    "default_controls": {
      "version": 1,
      "voices": {"voice_1688": {"pace": 1.10}},     # per voice
      "clone":  {"pace": 1.0},                      # ref_wav / spk_vec renders (optional)
      "*":      {"pace": 1.0}                       # every other voice (optional)
    }

    pace  multiplies the request's pace           (effective = request pace * profile pace, clipped)
    ctrl  {dim: offset} is added to a control the request SETS (effective = value + offset, clipped);
          a control the request leaves unset stays unset (mask 0), so tags keep choosing it.

Missing key / voice -> identity (pace 1, no offsets): old packages render exactly as before. A request opts out with
"default_controls": false (research, the leaderboard protocol, A/B against the raw model). The same table and the same
two rules are implemented by every runtime (python full + edge, C, C++ native, nano.bin).
"""
IDENTITY = {"pace": 1.0, "ctrl": {}}


def voice_profile(cfg, voice=None, clone=False):
    """-> {"pace": float, "ctrl": {dim: offset}} for a voice name (or the clone profile)."""
    dc = (cfg or {}).get("default_controls") or {}
    p = dc.get("clone") if clone else (dc.get("voices") or {}).get(voice)
    if p is None:
        p = dc.get("*") if not clone else None
    p = p or {}
    return {"pace": float(p.get("pace", 1.0)), "ctrl": {k: float(v) for k, v in (p.get("ctrl") or {}).items()}}


def apply_profile(profile, ctrl, pace, ctrl_range=(-5.0, 5.0), pace_range=(0.25, 4.0)):
    """(ctrl, pace) of a validated request -> the values the model sees."""
    lo, hi = ctrl_range
    off = profile.get("ctrl") or {}
    ctrl = {k: min(max(float(v) + off.get(k, 0.0), lo), hi) for k, v in (ctrl or {}).items()}
    pace = min(max(float(pace) * profile.get("pace", 1.0), pace_range[0]), pace_range[1])
    return ctrl, pace


def check_table(cfg, voices, dims):
    """Validate a config's default_controls against the voice and control lists (raises ValueError)."""
    dc = (cfg or {}).get("default_controls")
    if not dc:
        return
    entries = dict((dc.get("voices") or {}))
    for k in ("clone", "*"):
        if k in dc:
            entries[k] = dc[k]
    for v, p in entries.items():
        if v not in ("clone", "*") and v not in voices:
            raise ValueError(f"default_controls: unknown voice {v!r}")
        if not 0.5 <= float(p.get("pace", 1.0)) <= 2.0:
            raise ValueError(f"default_controls[{v!r}].pace outside [0.5, 2.0]")
        for d, o in (p.get("ctrl") or {}).items():
            if d not in dims or not -2.0 <= float(o) <= 2.0:
                raise ValueError(f"default_controls[{v!r}].ctrl.{d}={o} (unknown dim or |offset| > 2)")
