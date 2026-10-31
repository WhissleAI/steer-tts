"""Recorded sound events for inline [laugh] [cough] [dog] ... tokens (events.safetensors, numpy only).

The acoustic model renders speech; a laugh or a cough is a noisy, bursty sound that its mel decoder blurs into a
voiced hum, so event tokens are rendered from a small bank of real recordings instead. Each clip is stored as a
Vocos log-mel and goes through the same vocoder as the speech, so it shares the voice's channel:

  in-voice nonverbals  laugh cough sigh sneeze throat   a clip of the voice's gender with the nearest pitch,
                                                         re-pitched to the voice (at most +-4 semitones) and set
                                                         2 dB below the surrounding speech
  environmental        dog knock door phone bell ...     a clip of that class, 3 dB below the speech

    bank = EventBank("events.safetensors")
    mel = bank.mel("laugh", gender="female", f0=210.0, ref_loud=EventBank.loudness(speech_mel), seed=0)  # [100, T]

The bank's licence is separate from the model's: see LICENSE.md / NOTICE in the package.
"""
import json
from pathlib import Path

import numpy as np

SR, HOP, N_MELS = 24000, 256, 100
NONVERBAL = ("laugh", "cough", "sigh", "sneeze", "throat")
GAIN = dict(nonverbal=-2.0, env=-3.0)  # dB relative to the speech level
MAX_SHIFT = 4.0  # semitones
GAP_S = 0.08  # silence each side of an event


def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, np.float64) / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, np.float64) / 2595.0) - 1.0)


def pitch_shift(mel, ratio):
    """Shift a log-mel [100, T] by a frequency ratio (formants move with the pitch, like resampling)."""
    if abs(np.log(ratio)) < 1e-3:
        return mel
    pts = np.linspace(0.0, _hz2mel(SR / 2), N_MELS + 2)  # torchaudio htk mel scale, f_min 0, f_max sr/2
    step = pts[1] - pts[0]
    src = (_hz2mel(_mel2hz(pts[1:-1]) / ratio) - pts[1]) / step  # fractional source band of each output band
    src = np.clip(src, 0, N_MELS - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, N_MELS - 1)
    w = (src - i0)[:, None]
    return ((1 - w) * mel[i0] + w * mel[i1]).astype(np.float32)


_FB = None


def wav_mel(y, n_fft=1024):
    """Vocos log-mel [100, T] of a 24 kHz waveform in numpy (torchaudio MelSpectrogram: hann, center, power 1, htk,
    no filter norm; log clamped at 1e-7 (vocos safe_log)): used only to read the level of the speech around an event."""
    global _FB
    if _FB is None:
        f = np.linspace(0, SR / 2, n_fft // 2 + 1)
        pts = _mel2hz(np.linspace(0.0, _hz2mel(SR / 2), N_MELS + 2))
        lo, c, hi = pts[:-2, None], pts[1:-1, None], pts[2:, None]
        _FB = np.maximum(0, np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c))).astype(np.float32)  # [100, F]
    y = np.asarray(y, np.float32).reshape(-1)
    if len(y) < n_fft:
        return np.full((N_MELS, 1), np.log(1e-7), np.float32)
    y = np.pad(y, (n_fft // 2, n_fft // 2), mode="reflect")
    n = 1 + (len(y) - n_fft) // HOP
    idx = np.arange(n_fft)[None] + HOP * np.arange(n)[:, None]
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    S = np.abs(np.fft.rfft(y[idx] * win, axis=1)).astype(np.float32)  # [T, F]
    return np.log(np.maximum(_FB @ S.T, 1e-7))


class EventBank:
    def __init__(self, path):
        from safetensors.numpy import load_file
        from safetensors import safe_open
        t = load_file(str(path))
        with safe_open(str(path), "np") as f:
            meta = f.metadata() or {}
        if meta.get("format") != "steertts-events-1":
            raise ValueError(f"{path}: not a SteerTTS event bank")
        self.path = Path(path)
        self._q, self._lo, self._st = t["mel_q"], t["lo"], t["step"]
        self.index = json.loads(meta["index"])
        self.types = sorted({c["type"] for c in self.index})

    @classmethod
    def find(cls, model_dir, cfg=None):
        """The package's bank (config 'events' file, else events.safetensors), or None when it has none."""
        name = ((cfg or {}).get("events") or {}).get("file", "events.safetensors")
        p = Path(model_dir) / name
        return cls(p) if p.exists() else None

    def __contains__(self, typ):
        return typ in self.types

    def _clip(self, i):
        c = self.index[i]
        q = self._q[c["start"]:c["start"] + c["n"]].astype(np.float32)
        return (self._lo[i] + q * self._st[i]).T  # [100, T]

    @staticmethod
    def loudness(mel):
        """Level of a log-mel [.., 100, T] (numpy or torch): mean band level over its louder half of frames."""
        m = np.asarray(mel.detach().float().cpu() if hasattr(mel, "detach") else mel, np.float32)
        m = m.reshape(-1, m.shape[-1]) if m.ndim > 2 else m
        e = np.sort(m.mean(0))
        return float(e[len(e) // 2:].mean()) if len(e) else -6.0

    def pick(self, typ, gender=None, f0=None, seed=0):
        cand = [i for i, c in enumerate(self.index) if c["type"] == typ]
        if not cand:
            raise KeyError(typ)
        same = [i for i in cand if gender and self.index[i]["gender"] == gender]
        cand = same or cand
        if f0 and typ in NONVERBAL:
            voiced = [i for i in cand if self.index[i]["f0"] > 0]
            if voiced:  # the two nearest in pitch; the seed alternates between them (variety across calls)
                voiced.sort(key=lambda i: (abs(np.log(self.index[i]["f0"] / f0)), i))
                cand = voiced[:2]
        return cand[int(seed) % len(cand)]  # plain modulo: the C library picks the same clip

    def mel(self, typ, gender=None, f0=None, ref_loud=None, seed=0, gain_db=None):
        """-> float32 log-mel [100, T] of one event, with GAP_S of silence each side, ready for the vocoder."""
        i = self.pick(typ, gender, f0, seed)
        m = self._clip(i)
        c = self.index[i]
        if typ in NONVERBAL and f0 and c["f0"] > 0:
            st = float(np.clip(12 * np.log2(f0 / c["f0"]), -MAX_SHIFT, MAX_SHIFT))
            m = pitch_shift(m, 2 ** (st / 12))
        if gain_db is None:
            gain_db = GAIN["nonverbal" if typ in NONVERBAL else "env"]
        if ref_loud is not None:  # log-mel is natural-log magnitude: dB / 20 * ln(10)
            m = m + (ref_loud + gain_db / 20 * np.log(10) - self.loudness(m))
        g = np.full((N_MELS, int(GAP_S * SR / HOP)), np.log(1e-7), np.float32)  # the vocos silence floor
        return np.concatenate([g, m, g], 1).astype(np.float32)
