"""Output level: the last stage between the vocoder and the caller (the same file ships in steertts, steertts_edge and
steertts_nano; the C library (st_output.c), the native engine (native/src/output.cpp) and the ESP32 firmware implement
the same stage).

The raw model output sits ~9-10 dB under what phones, browsers and other TTS engines play at (~-25 LUFS integrated,
peaks ~-8 dBFS). The package config.json "output" block holds one calibration that every runtime reads:

    "output": {"target_lufs": -17.0, "true_peak_db": -1.0, "knee_db": -3.0, "gain_db": 8.1,
               "voice_gain_db": {"voice_1688": 7.9, ...}}

voice_gain_db[v] is the gain that puts voice v's neutral speech at target_lufs (measured on neutral renders of
held-out sentences; gain_db is the fallback for a cloned / custom speaker vector). Levels (`level=`):

  "auto"  (default) the voice's calibrated gain, then a soft-knee limiter (tanh above knee_db, ceiling
          true_peak_db, the ESP32 firmware's curve), and for a whole clip a final true-peak guard (4x oversampled):
          neutral speech lands at target_lufs and steering keeps its relative level (loud / soft / whisper are
          still louder / quieter than neutral). Streaming applies the same fixed gain + limiter per chunk (no
          look-ahead, no added latency); a streamed utterance equals the whole clip up to the true-peak guard.
  "clip"  whole clip: BS.1770 integrated loudness of this clip -> target_lufs exactly, then limiter + true-peak
          guard (every clip at the same loudness: flattens loudness steering). Streaming: as "auto".
  "off"   the raw model output, bit for bit.

gain_db= overrides the calibrated gain (dB, -6 .. 18, as the firmware's gain_db).
"""
import math

import numpy as np

LEVELS = ("auto", "clip", "off")
DEFAULT = {"target_lufs": -17.0, "true_peak_db": -1.0, "knee_db": -3.0, "gain_db": 0.0, "voice_gain_db": {}}
GAIN_RANGE = (-6.0, 18.0)


def db2lin(db):
    return 10.0 ** (db / 20.0)


def soft_limit(x, knee_db=-3.0, ceiling_db=-1.0):
    """Memoryless soft-knee limiter: |x| <= knee unchanged, above it knee + (ceil - knee) * tanh((|x| - knee) /
    (ceil - knee)), so the output never exceeds the ceiling and the curve is continuous with slope 1 at the knee."""
    x = np.asarray(x, np.float32)
    t, c = np.float32(db2lin(knee_db)), np.float32(db2lin(ceiling_db))
    a = np.abs(x)
    over = a > t
    if not over.any():
        return x
    y = x.copy()
    y[over] = np.copysign(t + (c - t) * np.tanh((a[over] - t) / (c - t)), x[over]).astype(np.float32)
    return y


# ------------------------------------------------------------------ true peak (BS.1770-4 annex 2: 4x oversampling)
_OS, _TAPS = 4, 48


def _interp_fir():
    """48-tap windowed-sinc (Hann) 4x interpolator, as 4 phases of 12 taps. The C runtimes build the same table."""
    n = np.arange(_TAPS, dtype=np.float64)
    m = n - (_TAPS - 1) / 2.0
    h = np.sinc(m / _OS) * (0.5 - 0.5 * np.cos(2 * np.pi * (n + 0.5) / _TAPS))
    return h.reshape(-1, _OS).T.copy()  # [phase, tap]


_FIR = _interp_fir()


def true_peak(x):
    """Linear true peak (max of |x| and of its 4x oversampled version)."""
    x = np.asarray(x, np.float64)
    if not len(x):
        return 0.0
    p = float(np.abs(x).max())
    for ph in _FIR:
        p = max(p, float(np.abs(np.convolve(x, ph, mode="full")).max()))
    return p


def true_peak_db(x):
    return 20.0 * math.log10(true_peak(x) + 1e-12)


# ------------------------------------------------------------------ BS.1770-4 integrated loudness (mono)
def _biquads(sr):
    def shelf(fc=1500.0, q=1 / math.sqrt(2), g=4.0):
        A, w = 10 ** (g / 40), 2 * math.pi * fc / sr
        al, c = math.sin(w) / (2 * q), math.cos(w)
        b = [A * ((A + 1) + (A - 1) * c + 2 * math.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c),
             A * ((A + 1) + (A - 1) * c - 2 * math.sqrt(A) * al)]
        a = [(A + 1) - (A - 1) * c + 2 * math.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * c),
             (A + 1) - (A - 1) * c - 2 * math.sqrt(A) * al]
        return [v / a[0] for v in b], [v / a[0] for v in a]

    def hp(fc=38.0, q=0.5):
        w = 2 * math.pi * fc / sr
        al, c = math.sin(w) / (2 * q), math.cos(w)
        b, a = [(1 + c) / 2, -(1 + c), (1 + c) / 2], [1 + al, -2 * c, 1 - al]
        return [v / a[0] for v in b], [v / a[0] for v in a]

    return shelf(), hp()


def _lfilter(b, a, x):
    try:
        from scipy.signal import lfilter
        return lfilter(b, a, x)
    except ImportError:  # numpy-only installs (steertts-edge ONNX runtime, nano): direct form II transposed
        y = np.empty_like(x)
        z1 = z2 = 0.0
        b0, b1, b2 = b
        _, a1, a2 = a
        for i, v in enumerate(x.tolist()):
            o = b0 * v + z1
            z1 = b1 * v - a1 * o + z2
            z2 = b2 * v - a2 * o
            y[i] = o
        return y


def lufs(x, sr):
    """Integrated loudness (LUFS) of a mono clip: K-weighting, 400 ms blocks / 100 ms hop, -70 LUFS absolute and
    -10 LU relative gates. Clips shorter than one block use their whole length. -70.0 for silence."""
    x = np.asarray(x, np.float64)
    if not len(x):
        return -70.0
    (b1, a1), (b2, a2) = _biquads(sr)
    y = _lfilter(b2, a2, _lfilter(b1, a1, x))
    blk, hop = int(0.4 * sr), int(0.1 * sr)
    if len(y) < blk:
        z = np.array([np.mean(y ** 2)])
    else:
        c = np.concatenate([[0.0], np.cumsum(y ** 2)])
        st = np.arange(0, len(y) - blk + 1, hop)
        z = (c[st + blk] - c[st]) / blk
    lv = -0.691 + 10 * np.log10(z + 1e-12)
    z = z[lv > -70]
    if not len(z):
        return -70.0
    rel = -0.691 + 10 * np.log10(z.mean()) - 10
    z = z[(-0.691 + 10 * np.log10(z)) > rel]
    return float(-0.691 + 10 * np.log10(z.mean()))


# ------------------------------------------------------------------ the stage
def check_level(level, gain_db=None):
    if level not in LEVELS:
        raise ValueError(f"level must be one of {', '.join(LEVELS)}, got {level!r}")
    if gain_db is not None:
        if isinstance(gain_db, bool) or not isinstance(gain_db, (int, float)):
            raise ValueError(f"gain_db must be a number, got {gain_db!r}")
        if not GAIN_RANGE[0] <= float(gain_db) <= GAIN_RANGE[1]:
            raise ValueError(f"gain_db={gain_db} is outside [{GAIN_RANGE[0]}, {GAIN_RANGE[1]}]")


class Output:
    """Reads the package config's "output" block (absent: level 'auto' behaves as 'off' would at 0 dB gain, i.e.
    only the limiter, so an old package is never made louder by guesswork)."""

    def __init__(self, cfg, sr):
        o = dict(DEFAULT, **((cfg or {}).get("output") or {}))
        self.sr = sr
        self.target_lufs, self.true_peak_db = float(o["target_lufs"]), float(o["true_peak_db"])
        self.knee_db, self.gain_db = float(o["knee_db"]), float(o["gain_db"])
        self.voice_gain_db = {k: float(v) for k, v in (o.get("voice_gain_db") or {}).items()}

    def gain(self, voice=None, gain_db=None):
        """dB gain of the 'auto' level for a voice (None / unknown: the fallback gain_db)."""
        if gain_db is not None:
            return float(gain_db)
        return self.voice_gain_db.get(voice, self.gain_db) if voice is not None else self.gain_db

    def _apply(self, x, g_db):
        return soft_limit(np.asarray(x, np.float32) * np.float32(db2lin(g_db)), self.knee_db, self.true_peak_db)

    def chunk(self, x, voice=None, level="auto", gain_db=None):
        """A streamed chunk: fixed gain + limiter (memoryless: chunking never changes the result)."""
        check_level(level, gain_db)
        if level == "off" or not len(x):
            return x
        return self._apply(x, self.gain(voice, gain_db))

    def clip(self, x, voice=None, level="auto", gain_db=None):
        """A whole clip: gain ('auto': the voice's, 'clip': to target_lufs), limiter, then a true-peak guard."""
        check_level(level, gain_db)
        if level == "off" or not len(x):
            return x
        if level == "clip" and gain_db is None:
            g = self.target_lufs - lufs(x, self.sr) if np.any(x) else 0.0
            g = min(max(g, GAIN_RANGE[0]), 30.0)  # silence-like clips: never boost by more than 30 dB
        else:
            g = self.gain(voice, gain_db)
        y = self._apply(x, g)
        tp, cap = true_peak(y), db2lin(self.true_peak_db)
        if tp > cap:
            y = (y * np.float32(cap / tp)).astype(np.float32)
        return y


def gate(x, sr, gain=1.0, frame_ms=8.0, open_db=-48.0, hold_frames=6, closed=0.1):
    """The ESP32 firmware's optional pause gate (off by default everywhere): per frame, RMS after `gain` above open_db
    opens the gate (held hold_frames), otherwise it closes to `closed` (-20 dB), ramped linearly across the frame."""
    x = np.asarray(x, np.float32)
    n = int(round(sr * frame_ms / 1000.0))
    y = np.empty_like(x)
    g, hold, thr = 1.0, 0, db2lin(open_db)
    for s in range(0, len(x), n):
        f = x[s:s + n]
        rms = gain * float(np.sqrt(np.mean(f.astype(np.float64) ** 2))) if len(f) else 0.0
        if rms > thr:
            hold = hold_frames
        elif hold > 0:
            hold -= 1
        tgt = 1.0 if hold > 0 else closed
        r = g + (tgt - g) * np.arange(1, len(f) + 1, dtype=np.float32) / len(f)
        y[s:s + n] = f * r
        g = tgt
    return y


def leveled(chunks, out, voice=None, level="auto", gain_db=None):
    """Applies Output.chunk to a stream of markup Chunks (objects with .audio), lazily: no added latency."""
    check_level(level, gain_db)
    for c in chunks:
        c.audio = out.chunk(c.audio, voice, level, gain_db)
        yield c


# ------------------------------------------------------------------ calibration (release step, re-run for new weights)
CAL_LINES = ("The library opens at nine on weekdays and stays open late on Thursdays.",
             "Did you remember to bring the charger for the camera?",
             "After the storm passed, we walked down to the harbour and counted the boats that had come loose overnight.")


def calibrate(tts, voices=None, lines=CAL_LINES, target_lufs=-17.0, true_peak_db=-1.0, knee_db=-3.0, offset_db=0.0,
              log=None):
    """The config.json "output" block for a loaded model: per voice, the gain that brings the integrated loudness of
    its neutral renders of `lines` (raw output, level="off", joined with 250 ms gaps) to target_lufs. offset_db is
    added to every gain (e.g. the measured level difference of the flow refiner when calibrating with flow=False).
    gain_db (the fallback for a cloned / custom speaker) is the median voice gain. Gains are rounded to 0.1 dB."""
    sr = tts.sample_rate
    voices = list(voices or tts.voices)
    gains = {}
    for i, v in enumerate(voices):
        ys = [np.asarray(tts(t, voice=v, level="off"), np.float32) for t in lines]
        y = np.concatenate([np.concatenate([a, np.zeros(sr // 4, np.float32)]) for a in ys])
        gains[v] = round(target_lufs - lufs(y, sr) + offset_db, 1)
        if log and i % 50 == 0:
            log(f"{i}/{len(voices)} {v} {gains[v]:+.1f} dB")
    med = float(np.median(list(gains.values()))) if gains else 0.0
    return {"target_lufs": target_lufs, "true_peak_db": true_peak_db, "knee_db": knee_db, "gain_db": round(med, 1),
            "voice_gain_db": gains}
