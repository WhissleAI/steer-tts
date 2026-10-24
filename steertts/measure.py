"""Read the 12 controls back from audio, in the units the model's controls are defined in.
Needs `praat-parselmouth` (the `measure` extra).

    from steertts.measure import measure_z
    z = measure_z(wav, text, tts.voice_stats("voice_1688"), tts.pooled_sd)   # {dim: speaker-relative z}

Raw measurements are converted to speaker-relative z with the voice's neutral mean (control_stats.json:
spk_mean) and the pooled within-speaker SD. `rate` needs the text (phone count); a few dims can be undefined
(NaN) on very short or unvoiced clips (f0_* need >= 10 voiced frames, f0_end >= 8 in the last 300 ms).
"""
import numpy as np
import parselmouth
import torch

from .audio import mel_extractor
from .config import CONTROL_DIMS, HOP, N_FFT, SR
from .text import SP, SYMBOLS, to_phonemes

_FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)
_PUNCT = {i for i, s in enumerate(SYMBOLS) if len(s) == 1}


def stft_mag(y):
    x = torch.from_numpy(y)
    X = torch.stft(x, N_FFT, HOP, N_FFT, torch.hann_window(N_FFT), center=True, return_complex=True)
    return X.abs().numpy()  # [513, T]


def pitch_track(y, n_frames, lo=60.0, hi=700.0):
    """Two-pass utterance-adaptive autocorrelation pitch (De Looze & Hirst) sampled at mel frame centres."""
    snd = parselmouth.Sound(y.astype(np.float64), SR)

    def run(lo, hi):
        p = snd.to_pitch_ac(time_step=HOP / SR, pitch_floor=lo, pitch_ceiling=hi, very_accurate=False)
        f = p.selected_array["frequency"]
        return p.xs(), f

    xs, f = run(lo, hi)
    v = f[f > 0]
    if len(v) > 20:
        q25, q75 = np.percentile(v, [25, 75])
        lo2, hi2 = max(50.0, 0.75 * q25), min(900.0, 1.75 * q75)
        if hi2 > lo2 + 40:
            xs, f = run(lo2, hi2)
    t = np.arange(n_frames) * HOP / SR
    idx = np.clip(np.searchsorted(xs, t), 1, len(xs) - 1) if len(xs) > 1 else np.zeros(n_frames, int)
    if len(xs) > 1:
        left = idx - 1
        idx = np.where(np.abs(xs[left] - t) < np.abs(xs[idx] - t), left, idx)
        near = np.abs(xs[idx] - t) <= HOP / SR
        f0 = np.where(near, f[idx], 0.0)
    else:
        f0 = np.zeros(n_frames)
    return f0.astype(np.float32), snd


def cpps(P, f_lo=60.0, f_hi=330.0):
    """Smoothed cepstral peak prominence (dB) over the given power-spectrum frames [513, n]:
    real cepstrum of the dB spectrum, peak in the 60-330 Hz quefrency band above a linear
    regression trend, cepstra smoothed over 3 frames x 3 quefrency bins first (Hillenbrand-style)."""
    if P.shape[1] < 3:
        return np.nan
    L = 10 * np.log10(P + 1e-10)
    C = np.fft.irfft(L, axis=0)[: N_FFT // 2]  # quefrency in samples
    C = 20 * np.log10(np.abs(C) + 1e-10)
    k = np.ones(3) / 3
    C = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 1, C)  # time smoothing
    C = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, C)  # quefrency smoothing
    q = np.arange(C.shape[0]) / SR
    band = (q >= 1 / f_hi) & (q <= 1 / f_lo)
    fit = (q >= 0.001) & (q <= 0.02)
    A = np.vstack([q[fit], np.ones(fit.sum())]).T
    coef, *_ = np.linalg.lstsq(A, C[fit], rcond=None)  # [2, n]
    i = np.argmax(C[band], 0)
    qb = q[band][i]
    peak = C[band][i, np.arange(C.shape[1])]
    trend = coef[0] * qb + coef[1]
    return float(np.mean(peak - trend))


def measure(y, n_phones=None, want_mel=True):
    """Returns dict(mel, f0, energy, controls(np.array len 12), speech_mask)."""
    y = y.astype(np.float32)
    out = {}
    if want_mel:
        mel = mel_extractor()(torch.from_numpy(y)[None])[0].numpy()  # [100, T]
        out["mel"] = mel
        T = mel.shape[1]
    S = stft_mag(y)
    T = S.shape[1] if not want_mel else min(T, S.shape[1])
    S = S[:, :T]
    P = S ** 2
    energy = 10 * np.log10(P.mean(0) + 1e-10)
    f0, snd = pitch_track(y, T)
    voiced = f0 > 0
    # adaptive VAD: 40% of the way from the noise floor (5th pct) to the speech peak (95th pct)
    lo_e, hi_e = np.percentile(energy, [5, 95])
    speech = energy > lo_e + 0.4 * (hi_e - lo_e)
    out.update(f0=f0, energy=energy.astype(np.float32), speech=speech)

    c = dict.fromkeys(CONTROL_DIMS, np.nan)
    sp = np.where(speech)[0]
    if len(sp) >= 5:
        a, b = sp[0], sp[-1] + 1
        span = (b - a) * HOP / SR
        inside = np.zeros(T, bool); inside[a:b] = True
        low = inside & ~speech
        # silent runs >= 100 ms inside the speech span count as pauses
        pause_frames, run = 0, 0
        for x in np.r_[low, False]:
            if x:
                run += 1
            else:
                if run * HOP / SR >= 0.1:
                    pause_frames += run
                run = 0
        pause_s = pause_frames * HOP / SR
        c["pause"] = pause_s / span
        if n_phones:
            c["rate"] = n_phones / max(span - pause_s, 0.2)
        sf_ = speech & inside
        c["loud"] = float(energy[sf_].mean())
        c["voiced"] = float(voiced[sf_].mean())
        Ps = P[:, sf_]
        lo_band = Ps[(_FREQS >= 50) & (_FREQS < 1000)].sum(0)
        hi_band = Ps[(_FREQS >= 1000) & (_FREQS < 5000)].sum(0)
        c["tilt"] = float(np.mean(10 * np.log10((hi_band + 1e-10) / (lo_band + 1e-10))))
        Pb = Ps[(_FREQS >= 50) & (_FREQS < 8000)] + 1e-10
        c["flat"] = float(np.mean(np.exp(np.log(Pb).mean(0)) / Pb.mean(0)))
        c["flat"] = float(np.log(c["flat"]))
        c["centroid"] = float(np.log((Ps * _FREQS[:, None]).sum(0).sum() / (Ps.sum() + 1e-10) + 1))
        fv = f0[voiced & inside]
        if len(fv) >= 10:
            st = 12 * np.log2(fv / 100.0)
            c["f0_med"] = float(np.median(st))
            q25, q75 = np.percentile(st, [25, 75])
            c["f0_iqr"] = float(q75 - q25)
            tv = np.where(voiced & inside)[0] * HOP / SR
            last = tv > tv[-1] - 0.3
            if last.sum() >= 8:
                c["f0_end"] = float(np.polyfit(tv[last], st[last], 1)[0])
            # H1-H2 from harmonic peaks (uncorrected), voiced speech frames
            Sdb = 10 * np.log10(P + 1e-10)
            vals = []
            for i in np.where(voiced & inside)[0][::2]:
                f = f0[i]
                m1 = (_FREQS > 0.85 * f) & (_FREQS < 1.15 * f)
                m2 = (_FREQS > 1.85 * f) & (_FREQS < 2.15 * f)
                if m1.any() and m2.any():
                    vals.append(Sdb[m1, i].max() - Sdb[m2, i].max())
            if vals:
                c["h1h2"] = float(np.median(vals))
        c["cpps"] = cpps(P[:, speech & inside])
    out["controls"] = np.array([c[k] for k in CONTROL_DIMS], np.float32)
    return out


def n_phones(text):
    ids, _, _ = to_phonemes(text)
    return sum(1 for i in ids if i != SP and i not in _PUNCT and not SYMBOLS[i].startswith("["))


def measure_z(wav, text, spk_mean, pooled_sd):
    """-> {dim: z} relative to the voice's neutral speech (spk_mean: {dim: raw}, pooled_sd: {dim: raw SD})."""
    raw = measure(np.asarray(wav, np.float32), n_phones=n_phones(text) if text else None, want_mel=False)["controls"]
    return {k: float((r - spk_mean[k]) / pooled_sd[k]) for k, r in zip(CONTROL_DIMS, raw)}
