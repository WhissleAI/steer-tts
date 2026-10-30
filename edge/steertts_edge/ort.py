"""SteerTTS edge on ONNX Runtime: the same rendering path as pipeline.EdgeTTS (model.infer -> vocoder.decode), with
the networks in three ONNX graphs and the glue (duration rounding, word shifts, length regulation, guidance mixing,
iSTFT) in numpy. No torch import.

    from steertts_edge.ort import OnnxTTS
    tts = OnnxTTS("path/to/package")            # needs onnx/ (enc.onnx, dec.onnx, voc.onnx) in the package
    wav = tts("I can't believe it worked.", tag="happiness", ctrl={"rate": 1.0})
"""
import json
from pathlib import Path

import numpy as np

from .config import CONTROL_DIMS, CONTROL_RANGE, HOP, N_FFT, PITCH_SCALE, SR
from .text import SYM2ID, set_assets, to_phonemes


CHUNK = 64  # the graphs take sequences zero-padded to a multiple of the sequence-mixing chunk, plus the real length


def _pad(a, axis=1):
    n = a.shape[axis]
    p = (-n) % CHUNK
    if not p:
        return a
    w = [(0, 0)] * a.ndim
    w[axis] = (0, p)
    return np.pad(a, w)


def _hann(n):
    """torch.hann_window(n) (periodic) in float32."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def istft(re, im, n_fft=N_FFT, hop=HOP):
    """torch.istft(re + i*im, n_fft, hop, n_fft, hann, center=True) for one spectrogram [n_fft/2+1, T]."""
    win = _hann(n_fft)
    T = re.shape[1]
    frames = np.fft.irfft(re.T + 1j * im.T, n_fft, axis=1).astype(np.float32) * win  # [T, n_fft]
    size = n_fft + hop * (T - 1)
    y = np.zeros(size, np.float32)
    env = np.zeros(size, np.float32)
    w2 = win * win
    for t in range(T):
        y[t * hop:t * hop + n_fft] += frames[t]
        env[t * hop:t * hop + n_fft] += w2
    a, b = n_fft // 2, size - n_fft // 2
    return y[a:b] / env[a:b]


class OnnxTTS:
    sample_rate = SR
    controls = list(CONTROL_DIMS)

    def __init__(self, pkg, guidance=1.0, threads=None, onnx_dir="onnx"):
        import onnxruntime as ort
        from safetensors.numpy import load_file
        d = Path(pkg)
        self.dir, self.guidance = d, guidance
        self.cfg = cfg = json.load(open(d / "config.json"))
        set_assets(d / cfg["front"]["dir"])
        self.tags = list(cfg["tags"])
        self.tag2i = {t: i for i, t in enumerate(self.tags)}
        self.voice_info = {v["name"]: v for v in json.load(open(d / "voices.json"))["voices"]}
        self.voices = list(self.voice_info)
        vt = load_file(str(d / "voices.safetensors"))
        self._s = vt["s"].astype(np.float32)
        self._mean = vt["spk_mean"].astype(np.float32)
        self._vi = {v: i for i, v in enumerate(self.voices)}
        self.default_voice = cfg.get("default_voice", self.voices[0])
        from .events import EventBank
        self.events = EventBank.find(d, cfg)  # recorded [laugh] [cough] ... (None: tokens stay silent)
        self.spk_ref = self._vi
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False      # return activation memory after each call (RSS stays near weights)
        so.add_session_config_entry("session.disable_prepacking", "1")   # no second, pre-packed weight copy
        so.enable_mem_pattern = False        # no pre-planned block for all intermediates (+24 MB peak with it)
        od = d / onnx_dir
        mk = lambda f: ort.InferenceSession(str(od / f), so, providers=["CPUExecutionProvider"])
        self.enc, self.dec, self.voc = mk("enc.onnx"), mk("dec.onnx"), mk("voc.onnx")
        sp = load_file(str(od / "spkproj.safetensors"))
        self._spw, self._spb = sp["w"], sp["b"]
        self.n_sym = int(cfg["acoustic"]["arch"]["n_symbols"])

    @classmethod
    def from_pretrained(cls, repo_id="WhissleAI/steer-tts-edge", revision=None, **kw):
        """OnnxTTS from a local package directory or a Hugging Face model repo (needs `huggingface_hub`, the optional
        `[hub]` extra). Only what this runtime reads is downloaded: config, voices, front end and onnx/ (~13 MB)."""
        if Path(repo_id).is_dir():
            return cls(repo_id, **kw)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError("OnnxTTS.from_pretrained(<repo id>) needs huggingface_hub: "
                              "pip install 'steertts-edge[hub]' (or pass a local directory)") from e
        d = snapshot_download(repo_id, revision=revision,
                              allow_patterns=["config.json", "voices.safetensors", "voices.json", "front/*", "onnx/*"])
        return cls(d, **kw)

    # ------------------------------------------------------------------ requests (as EdgeTTS)
    def control_arrays(self, ctrl):
        c = np.zeros((1, len(CONTROL_DIMS)), np.float32)
        m = np.zeros((1, len(CONTROL_DIMS)), np.float32)
        for k, v in (ctrl or {}).items():
            if k not in CONTROL_DIMS:
                raise ValueError(f"unknown control {k!r}; known: {', '.join(CONTROL_DIMS)}")
            if not CONTROL_RANGE[0] <= float(v) <= CONTROL_RANGE[1]:
                raise ValueError(f"ctrl.{k}={v} is outside {CONTROL_RANGE}")
            i = CONTROL_DIMS.index(k)
            c[0, i], m[0, i] = float(v), 1.0
        return c, m

    def voice_stats(self, voice):
        return dict(zip(CONTROL_DIMS, self._mean[self._vi[voice]].tolist()))

    def _check(self, voice, tag, ignored):
        if ignored.get("word_shift"):
            raise ValueError("word_shift does not combine with markup / stream(); use inline markup "
                             "(*word*, <slow>, <high> ...) there")
        for k in ("prompt", "ref_wav"):
            if ignored.get(k) is not None:
                raise ValueError(f"{k}= is not available in the edge build (free-text prompts and voice cloning "
                                 "run on a host with the full model, WhissleAI/steer-tts)")
        if tag not in self.tag2i:
            raise ValueError(f"unknown tag {tag!r}; known tags: {', '.join(self.tags)}")
        if voice is not None and voice not in self._vi:
            raise ValueError(f"unknown voice {voice!r}; this edge build ships {len(self.voices)} voices "
                             f"(see tts.voices), e.g. {', '.join(self.voices[:3])}")

    def __call__(self, text, voice=None, tag="none", ctrl=None, pace=1.0, word_shift=None, guidance=None,
                 spk_vec=None, markup=False, **_ignored):
        tag = tag or "none"
        self._check(voice if spk_vec is None else None, tag, dict(_ignored, word_shift=word_shift if markup else None))
        if markup or (self.events and not word_shift and self._has_event(text)):
            from .markup import MarkupSynth
            return MarkupSynth(self).render(text, voice=voice, tag=tag, ctrl=ctrl, pace=pace, guidance=guidance,
                                            spk_vec=spk_vec)
        return self._render(text, voice, tag, ctrl, pace, word_shift, guidance=guidance, spk_vec=spk_vec)

    synthesize = __call__

    def stream(self, text, voice=None, tag="none", ctrl=None, pace=1.0, guidance=None, spk_vec=None, **_ignored):
        from .markup import MarkupSynth
        tag = tag or "none"
        self._check(voice if spk_vec is None else None, tag, _ignored)
        return MarkupSynth(self).stream(text, voice=voice, tag=tag, ctrl=ctrl, pace=pace, guidance=guidance,
                                        spk_vec=spk_vec)

    # ------------------------------------------------------------------ rendering (== model.infer, batch 1)
    def _enc(self, ids, spk, tag, c, m):
        """-> g, and h / logdur / pitch / energy cut to the real phone count."""
        N = ids.shape[1]
        g, h, logdur, pitch, energy = self.enc.run(None, dict(ids=_pad(ids), n=np.asarray([N], np.int64), spk=spk,
                                                              tag=tag, ctrl=c, cmask=m))
        return g, h[:, :N], logdur[:, :N], pitch[:, :N], energy[:, :N]

    def _dec(self, h, pitch, energy, tok, g):
        T = tok.shape[1]
        return self.dec.run(None, dict(h=_pad(h), pitch=_pad(pitch), energy=_pad(energy), tok=_pad(tok),
                                       nf=np.asarray([T], np.int64), g=g))[0][:, :, :T]

    def mel(self, text, voice=None, tag="none", ctrl=None, pace=1.0, word_shift=None, guidance=None, spk_vec=None):
        ids, widx, _ = to_phonemes(text)
        if not any(i >= SYM2ID["AA0"] for i in ids):
            return None
        if max(ids) >= self.n_sym:
            keep = [k for k, i in enumerate(ids) if i < self.n_sym]
            ids, widx = [ids[k] for k in keep], [widx[k] for k in keep]
        ids_a = np.asarray([ids], np.int64)
        widx_a = np.asarray(widx, np.int64)
        c, m = self.control_arrays(ctrl)
        c = np.nan_to_num(c)
        voice = voice or self.default_voice
        if spk_vec is not None:
            sv = np.asarray(spk_vec, np.float32).reshape(1, -1)
            spk = (sv @ self._spw.T + self._spb).astype(np.float32)
        else:
            if voice not in self._vi:
                raise ValueError(f"unknown voice {voice!r}; this edge build ships {len(self.voices)} voices")
            spk = self._s[self._vi[voice]][None]
        tg = np.asarray([self.tag2i.get(tag, 0)], np.int64)
        s = self.guidance if guidance is None else guidance
        guided = s != 1.0 and (ctrl or tag not in (None, "none"))
        g, h, logdur, pitch, energy = self._enc(ids_a, spk, tg, c, m)
        if guided:
            z = np.zeros_like(c)
            g0, h0, logdur0, pitch0, energy0 = self._enc(ids_a, spk, np.zeros_like(tg), z, z)
            s = np.float32(s)
            logdur = logdur0 + s * (logdur - logdur0)
            pitch = pitch0 + s * (pitch - pitch0)
            energy = energy0 + s * (energy - energy0)
        dur = np.maximum(np.exp(logdur) - np.float32(1), 0) / np.float32(pace)
        if word_shift:
            for w, sh in word_shift.items():
                sel = widx_a == int(w)
                pitch[0, sel] += np.float32(sh.get("pitch", 0) / PITCH_SCALE)
                energy[0, sel] += np.float32(sh.get("energy", 0) / 10.0)
                dur[0, sel] *= np.float32(sh.get("dur", 1.0))
        dur = np.maximum(np.round(dur).astype(np.int64), 0)
        dur[:, 0] = np.maximum(dur[:, 0], 1)
        tok = np.repeat(np.arange(dur.shape[1]), dur[0])[None].astype(np.int64)
        mel = self._dec(h, pitch, energy, tok, g)
        if guided:
            mel0 = self._dec(h0, pitch, energy, tok, g0)
            mel = mel0 + s * (mel - mel0)
        return mel

    def _render(self, text, voice=None, tag="none", ctrl=None, pace=1.0, word_shift=None, ref_wav=None,
                guidance=None, flow=False, seed=0, spk_vec=None):
        mel = self.mel(text, voice, tag, ctrl, pace, word_shift, guidance, spk_vec)
        if mel is None:
            return np.zeros(0, np.float32)
        re, im = self.voc.run(None, dict(mel=mel.astype(np.float32)))
        return istft(re[0], im[0]).astype(np.float32)

    def _vocode(self, mel):
        """log-mel [100, T] (numpy) -> waveform, through the same vocoder as the speech."""
        re, im = self.voc.run(None, dict(mel=np.asarray(mel, np.float32)[None]))
        return istft(re[0], im[0]).astype(np.float32)

    def _has_event(self, text):
        from .text import _EVENT_RE
        return any(m.group(1).lower() in self.events for m in _EVENT_RE.finditer(text or ""))

    def save(self, wav, path):
        import wave
        y = np.clip(np.asarray(wav, np.float32).reshape(-1), -1.0, 1.0)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((y * 32767.0).round().astype("<i2").tobytes())
