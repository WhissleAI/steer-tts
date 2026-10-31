"""Waveform I/O and the Vocos mel front end (24 kHz, 100 bands, hop 256)."""
from pathlib import Path

import numpy as np
import torch

from .config import HOP, N_FFT, N_MELS, SR

_mel_fe = None


def mel_extractor():
    global _mel_fe
    if _mel_fe is None:
        from vocos.feature_extractors import MelSpectrogramFeatures
        _mel_fe = MelSpectrogramFeatures(sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, padding="center")
    return _mel_fe


def load_wav(x, sr=None):
    """path or float array (at `sr`, default 24 kHz) -> (mono float32 at 24 kHz, original sample rate)."""
    if isinstance(x, (str, Path)):
        import soundfile as sf
        y, sr = sf.read(str(x), dtype="float32")
    else:
        y, sr = np.asarray(x, np.float32), (sr or SR)
    y = y.mean(1) if y.ndim > 1 else y
    if sr != SR:
        import soxr
        y = soxr.resample(y, sr, SR)
    return np.asarray(y, np.float32), sr


def save_wav(wav, path):
    import soundfile as sf
    sf.write(str(path), np.asarray(wav, np.float32), SR)
