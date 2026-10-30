"""SteerTTS v1 edge pipeline: text + built-in voice + style tag / controls -> 24 kHz waveform, from a package
directory (config.json, acoustic.safetensors (int4/int8 packed), vocoder.safetensors, voices.safetensors,
voices.json, front/).

    from steertts_edge import EdgeTTS
    tts = EdgeTTS("path/to/package")                                # cpu
    wav = tts("I can't believe it worked.", voice="voice_1688", tag="happiness", ctrl={"rate": 1.0})
    wav = tts("Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow>", markup=True)

Same rendering path as the full release with flow=False (steertts.SteerTTS._render): model.infer -> vocoder.
Not on device (host-side in the full release): free-text prompts (MiniLM adapter; resolve them on a host into
tag + ctrl), the flow refiners, and the WavLM speaker encoder (a host-computed 512-d speaker vector can be passed as
spk_vec= to clone a voice).
"""
import json
from pathlib import Path

import numpy as np
import torch

from .config import CONTROL_DIMS, CONTROL_RANGE, PITCH_SCALE, SR
from .model import SteerTTSModel
from .packed import load_packed
from .text import SYM2ID, set_assets, to_phonemes
from .vocoder import Vocoder


class EdgeTTS:
    sample_rate = SR
    controls = list(CONTROL_DIMS)

    def __init__(self, pkg, device="cpu", guidance=1.0, threads=None):
        """pkg: a local directory holding the edge package (see from_pretrained for a Hugging Face repo id)."""
        from safetensors.torch import load_file
        if threads:
            torch.set_num_threads(threads)
        d = Path(pkg)
        self.dir, self.dev, self.guidance = d, device, guidance
        self.cfg = cfg = json.load(open(d / "config.json"))
        set_assets(d / cfg["front"]["dir"])
        self.tags = list(cfg["tags"])
        self.tag2i = {t: i for i, t in enumerate(self.tags)}
        self.pooled_sd = cfg["pooled_sd"]
        self.voice_info = {v["name"]: v for v in json.load(open(d / "voices.json"))["voices"]}
        self.voices = list(self.voice_info)
        vt = load_file(str(d / "voices.safetensors"))
        self._s = vt["s"].float().to(device)                       # [V, 64] conditioning slot per voice
        self._mean = vt["spk_mean"].float().numpy()                 # [V, 12] neutral raw control means
        self._vi = {v: i for i, v in enumerate(self.voices)}
        a = cfg["acoustic"]
        self.model = SteerTTSModel(0, len(self.tags), **a["arch"]).to(device).eval()
        sd, _ = load_packed(d / a["file"], device)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("spk.") for k in missing), (missing, unexpected)
        v = cfg["vocoder"]
        self.vocoder = Vocoder(v["dim"], v["inter"], v["layers"], padding=v.get("padding", "center")).to(device).eval()
        vsd, _ = load_packed(d / v["file"], device)
        vsd = {k: t for k, t in vsd.items() if not k.startswith(("feature_extractor.", "head.istft."))}
        self.vocoder.load_state_dict(vsd, strict=True)
        self.default_voice = cfg.get("default_voice", self.voices[0])
        from .events import EventBank
        self.events = EventBank.find(d, cfg)  # recorded [laugh] [cough] ... (None: tokens stay silent)
        self.spk_ref = self._vi   # name -> row (markup / request checks use `voice in tts.spk_ref`)

    # RELEASE (steer-tts-edge 1.0.0): load from a Hugging Face repo id or a local directory.
    @classmethod
    def from_pretrained(cls, repo_id="WhissleAI/steer-tts-edge", revision=None, **kw):
        """EdgeTTS from a local package directory or a Hugging Face model repo (needs `huggingface_hub`, the
        optional `[hub]` extra). Only the on-device files are downloaded (about 12 MB)."""
        if Path(repo_id).is_dir():
            return cls(repo_id, **kw)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError("EdgeTTS.from_pretrained(<repo id>) needs huggingface_hub: "
                              "pip install 'steertts-edge[hub]' (or pass a local directory)") from e
        d = snapshot_download(repo_id, revision=revision,
                              allow_patterns=["config.json", "*.safetensors", "voices.json", "front/*"])
        return cls(d, **kw)

    # ------------------------------------------------------------------ requests
    def control_tensors(self, ctrl):
        c = torch.zeros(1, len(CONTROL_DIMS), device=self.dev)
        m = torch.zeros(1, len(CONTROL_DIMS), device=self.dev)
        for k, v in (ctrl or {}).items():
            if k not in CONTROL_DIMS:
                raise ValueError(f"unknown control {k!r}; known: {', '.join(CONTROL_DIMS)}")
            if not CONTROL_RANGE[0] <= float(v) <= CONTROL_RANGE[1]:
                raise ValueError(f"ctrl.{k}={v} is outside {CONTROL_RANGE}")
            i = CONTROL_DIMS.index(k)
            c[0, i], m[0, i] = float(v), 1.0
        return c, m

    def voice_stats(self, voice):
        """The voice's neutral raw control means (for measure_z readback)."""
        return dict(zip(CONTROL_DIMS, self._mean[self._vi[voice]].tolist()))

    def _check(self, voice, tag, ignored):
        # RELEASE: reject full-model-only arguments instead of silently ignoring them (flow / seed stay accepted
        # and ignored, for call compatibility with steertts.SteerTTS), and list the valid names in the errors.
        if ignored.get("word_shift"):  # RELEASE: as steertts v1.0.1, instead of dropping it silently
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
        tag = tag or "none"  # RELEASE: tag=None means no style tag (was: ValueError "unknown tag None")
        self._check(voice if spk_vec is None else None, tag, dict(_ignored, word_shift=word_shift if markup else None))
        if markup or (self.events and not word_shift and self._has_event(text)):
            from .markup import MarkupSynth
            return MarkupSynth(self).render(text, voice=voice, tag=tag, ctrl=ctrl, pace=pace, guidance=guidance,
                                            spk_vec=spk_vec)
        return self._render(text, voice, tag, ctrl, pace, word_shift, guidance=guidance, spk_vec=spk_vec)

    synthesize = __call__

    def stream(self, text, voice=None, tag="none", ctrl=None, pace=1.0, guidance=None, spk_vec=None, **_ignored):
        """Sentence-level streaming with inline markup: yields markup.Chunk (.audio float32, .segment, .synth_ms)
        per sentence / pause / style span, in order. RELEASE: was only reachable as MarkupSynth(tts).stream."""
        from .markup import MarkupSynth
        tag = tag or "none"
        self._check(voice if spk_vec is None else None, tag, _ignored)
        return MarkupSynth(self).stream(text, voice=voice, tag=tag, ctrl=ctrl, pace=pace, guidance=guidance,
                                        spk_vec=spk_vec)

    # ------------------------------------------------------------------ rendering (== SteerTTS._render, flow off)
    @torch.no_grad()
    def mel(self, text, voice=None, tag="none", ctrl=None, pace=1.0, word_shift=None, guidance=None, spk_vec=None):
        ids, widx, _ = to_phonemes(text)
        # RELEASE: to_phonemes always returns at least the leading <sp>, so the original `if not ids` never fired and
        # empty / whitespace / punctuation-only / non-Latin-script text rendered ~10-200 ms of near-silence. Text
        # with nothing to say (no phoneme or sound-event token) now returns an empty waveform, as documented.
        if not any(i >= SYM2ID["AA0"] for i in ids):
            return None
        n_sym = self.model.emb.num_embeddings
        if max(ids) >= n_sym:
            keep = [k for k, i in enumerate(ids) if i < n_sym]
            ids, widx = [ids[k] for k in keep], [widx[k] for k in keep]
        ids_t = torch.tensor([ids], device=self.dev)
        widx_t = torch.tensor([widx], device=self.dev)
        c, m = self.control_tensors(ctrl)
        voice = voice or self.default_voice
        if spk_vec is not None:
            sv = torch.as_tensor(np.asarray(spk_vec, np.float32), device=self.dev).reshape(1, -1)
            spk = None
        else:
            if voice not in self._vi:
                raise ValueError(f"unknown voice {voice!r}; this edge build ships {len(self.voices)} voices")
            spk, sv = self._s[self._vi[voice]][None], None
        tg = torch.tensor([self.tag2i.get(tag, 0)], device=self.dev)
        if word_shift:
            word_shift = {int(w): dict(pitch=s.get("pitch", 0) / PITCH_SCALE, energy=s.get("energy", 0) / 10.0,
                                       dur=s.get("dur", 1.0)) for w, s in word_shift.items()}
        kw = {}
        s = self.guidance if guidance is None else guidance
        if s != 1.0 and (ctrl or tag not in (None, "none")):
            kw.update(guidance=s, base=(torch.zeros_like(tg), torch.zeros_like(c), torch.zeros_like(m)))
        mel, _ = self.model.infer(ids_t, spk, tg, c, m, pace=pace, word_idx=widx_t, word_shift=word_shift,
                                  spk_vec=sv, **kw)
        return mel

    @torch.no_grad()
    def _render(self, text, voice=None, tag="none", ctrl=None, pace=1.0, word_shift=None, ref_wav=None,
                guidance=None, flow=False, seed=0, spk_vec=None):
        mel = self.mel(text, voice, tag, ctrl, pace, word_shift, guidance, spk_vec)
        if mel is None:
            return np.zeros(0, np.float32)
        return self.vocoder.decode(mel.float()).squeeze().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _vocode(self, mel):
        """log-mel [100, T] (numpy) -> waveform, through the same vocoder as the speech."""
        m = torch.from_numpy(np.asarray(mel, np.float32))[None].to(self.dev)
        return self.vocoder.decode(m).squeeze().cpu().numpy().astype(np.float32)

    def _has_event(self, text):
        from .text import _EVENT_RE
        return any(m.group(1).lower() in self.events for m in _EVENT_RE.finditer(text or ""))

    def save(self, wav, path):
        """16-bit PCM WAV at 24 kHz (clipped to [-1, 1]). RELEASE: stdlib `wave` instead of soundfile, one
        dependency fewer; same format as soundfile's default WAV subtype (PCM_16)."""
        import wave
        y = np.clip(np.asarray(wav, np.float32).reshape(-1), -1.0, 1.0)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((y * 32767.0).round().astype("<i2").tobytes())
