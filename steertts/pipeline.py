"""SteerTTS inference: a request (text + voice + tag / controls / free-text direction) -> 24 kHz waveform.

    from steertts import SteerTTS
    tts = SteerTTS.from_pretrained("WhissleAI/steer-tts")              # or a local directory; cuda if available
    wav = tts("I can't believe it worked.", voice="voice_1688", tag="happiness", ctrl={"rate": 1.0})
    req = tts.resolve({"text": "Hello there.", "prompt": "soft and sad"})  # the request the prompt maps to
    wav = tts.synthesize(req)
    for chunk in tts.stream("First sentence. <whisper>Second one.</whisper>"):   # sentence streaming + markup
        play(chunk.audio)
    wav = tts("Now in your voice.", ref_wav="me.wav")                 # zero-shot cloning

The rendering path is deterministic for a given seed (text front end, model, flow sampler, Vocos), so outputs
are reproducible bit for bit on the same device.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from .audio import load_wav, mel_extractor, save_wav
from .config import CONTROL_DIMS, DEFAULT_VOICE, PITCH_SCALE, SR
from .model import SteerTTSModel
from .request import normalize_request
from .text import to_phonemes


PRECISIONS = ("fp32", "int8")


def resolve_model_dir(repo_or_path=None, revision=None, token=None, precision="fp32"):
    """Local directory, else a Hugging Face repo id (downloaded once into the HF cache). None -> $STEERTTS_MODEL.
    Downloads only what this package reads: the top-level json files plus the fp32 weight files, or for
    precision="int8" the int8/ weight files instead (the repo also holds other runtimes' files, e.g. native/ for the
    C library, and the model card's samples)."""
    repo_or_path = repo_or_path or os.environ.get("STEERTTS_MODEL")
    if not repo_or_path:
        raise ValueError("no model given: pass a local model directory or an HF repo id, or set STEERTTS_MODEL")
    if os.path.isdir(repo_or_path):
        return Path(repo_or_path)
    from huggingface_hub import list_repo_files, snapshot_download
    files = list_repo_files(repo_or_path, revision=revision, token=token)
    top = [f for f in files if "/" not in f]
    keep = [f for f in top if f.endswith(".json")]
    if precision == "int8":
        keep += [f for f in files if f.startswith("int8/") and f.endswith(".safetensors")]
        keep += [f for f in top if f == "events.safetensors"]  # the event bank has no int8 copy
    else:
        keep += [f for f in top if f.endswith(".safetensors")]
    return Path(snapshot_download(repo_or_path, revision=revision, token=token, allow_patterns=keep))


class SteerTTS:
    sample_rate = SR
    controls = list(CONTROL_DIMS)

    def __init__(self, model_dir, device=None, flow=True, flow_steps=10, flow_temp=0.4, guidance=1.0, precision="fp32"):
        """flow: use the flow-matching refiner (and, for cloning, the prompted flow) when the weights are present.
        guidance: default steering scale (1 = plain conditional render; see the model card for the trade-off).
        precision: "fp32" (default) or "int8" (packed int8 weights in int8/, external models rounded to int8 at load;
        compute stays fp32; see steertts.quant)."""
        from safetensors.torch import load_file
        d = Path(model_dir)
        self.model_dir = d
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        self.precision = precision
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = json.load(open(d / "config.json"))
        stats = json.load(open(d / "control_stats.json"))
        self.pooled_sd, self.spk_mean, self.tags = stats["pooled_sd"], stats["spk_mean"], list(stats["tags"])
        self.tag2i = {t: i for i, t in enumerate(self.tags)}
        self.voice_info = {v["name"]: v for v in json.load(open(d / "voices.json"))["voices"]}
        ref = self._weights(self.cfg["files"]["speaker_vectors"], "cpu")
        self.voices = list(self.cfg["voices_order"]) if "voices_order" in self.cfg else list(self.voice_info)
        assert ref["vec"].shape[0] == len(self.voices), "speaker_vectors / voices.json mismatch"
        self.spk_ref = {v: ref["vec"][i].numpy() for i, v in enumerate(self.voices)}
        b = self.cfg["acoustic"]
        self.n_speakers = b["n_speakers"]
        self.model = SteerTTSModel(b["n_speakers"], b["n_tags"], **b["arch"]).to(self.dev).eval()
        self.model.load_state_dict(self._weights(self.cfg["files"]["acoustic"], self.dev))
        from vocos import Vocos
        self.vocos = self._external(Vocos.from_pretrained(self.cfg["vocoder"]).to(self.dev).eval())
        self.use_flow, self.flow_steps, self.flow_temp, self.guidance = flow, flow_steps, flow_temp, guidance
        self.flow = self._load_flow("refiner") if flow else None
        self._pflow, self._xv, self._pm, self._vec_cache = None, None, None, {}
        self.default_voice = self.cfg.get("default_voice", DEFAULT_VOICE)
        from .events import EventBank
        self.events = EventBank.find(d, self.cfg)  # recorded [laugh] [cough] [dog] ... (None: tokens stay silent)

    @classmethod
    def from_pretrained(cls, repo_or_path=None, revision=None, token=None, precision="fp32", **kw):
        return cls(resolve_model_dir(repo_or_path, revision, token, precision), precision=precision, **kw)

    def _weights(self, name, device):
        """A weight file of this release at self.precision -> fp32 state dict."""
        from safetensors.torch import load_file
        if self.precision == "fp32":
            return load_file(str(self.model_dir / name), device=device)
        from .quant import load_packed
        return load_packed(self.model_dir / self.cfg["precisions"][self.precision]["dir"] / name, device=device)

    def _has(self, name):
        sub = self.cfg["precisions"][self.precision]["dir"] if self.precision != "fp32" else "."
        return (self.model_dir / sub / name).exists()

    def _external(self, module):
        """Vocos / WavLM / MiniLM come from their own repos; at int8 they are rounded in place (steertts.quant)."""
        if self.precision == "int8":
            from .quant import round_int8_
            round_int8_(module)
        return module

    # ------------------------------------------------------------------ lazy parts
    def _load_flow(self, which):
        from safetensors.torch import load_file
        from .flow import Flow
        spec = self.cfg.get(which)
        if not spec or not self._has(spec["file"]):
            return None
        return Flow.from_state(self._weights(spec["file"], "cpu"), spec["cfg"], self.dev)

    @property
    def pflow(self):
        if self._pflow is None and self.use_flow:
            self._pflow = self._load_flow("prompted_flow") or False
        return self._pflow or None

    @property
    def prompt_mapper(self):
        if self._pm is None:
            from safetensors.torch import load_file
            from .prompt import PromptMapper
            p = self.cfg["prompt_adapter"]
            self._pm = PromptMapper(self._weights(p["file"], "cpu"), p["tags"], self.dev)
            self._external(self._pm.enc.m)
        return self._pm

    # ------------------------------------------------------------------ requests
    def map_prompt(self, prompt):
        """Free-text direction -> dict(tag, ctrl, top_tags): the tag used is the adapter's highest-ranked tag that
        this model knows."""
        r = self.prompt_mapper(prompt)
        tag = next((t for t, _ in r["top_tags"] if t in self.tag2i), "none")
        return dict(tag=tag, ctrl=dict(r["ctrl"]), top_tags=[list(t) for t in r["top_tags"]])

    def resolve(self, request=None, **kw):
        """Validate a request (dict or keyword arguments; see request.schema.json) and return it fully resolved:
        defaults filled in, the free-text `prompt` mapped to `tag` + `ctrl` (explicit `tag` wins, explicit `ctrl`
        dims override the prompt's), and the mapping recorded under `prompt_mapping`. The result is itself a valid
        request and renders identically."""
        req = normalize_request({**(request or {}), **kw}, self)
        if req.get("prompt"):
            m = self.map_prompt(req["prompt"])
            req["prompt_mapping"] = m
            if req.get("tag") is None:
                req["tag"] = m["tag"]
            req["ctrl"] = {**m["ctrl"], **req.get("ctrl", {})}
            del req["prompt"]
        req.setdefault("tag", "none")
        if req["tag"] is None:
            req["tag"] = "none"
        return req

    def synthesize(self, request=None, **kw):
        """Render a request -> float32 waveform at 24 kHz. Markup requests are rendered segment by segment."""
        req = self.resolve(request, **kw)
        if req["markup"] or (self.events and not req.get("word_shift") and self._has_event(req["text"])):
            return self.markup().render(**self._stream_args(req))
        return self._render(req["text"], self._voice(req), req["tag"], req["ctrl"], req["pace"], req.get("word_shift"),
                            req.get("ref_wav"), req["guidance"], req["flow"], req["seed"])

    def __call__(self, text, voice=None, **kw):
        return self.synthesize(dict(kw, text=text, voice=voice) if voice else dict(kw, text=text))

    say = __call__

    def stream(self, text=None, request=None, **kw):
        """Sentence-level streaming with inline markup: yields Chunk(audio, segment, synth_ms) in order, the first
        as soon as the first sentence / span is rendered."""
        if text is not None:
            kw["text"] = text
        req = self.resolve(request, **kw)
        if req.get("word_shift"):
            raise ValueError("stream() does not support word_shift (segments re-index words); use inline markup "
                             "such as <high>word</high> or *word*, or synthesize() for a whole-utterance word_shift")
        return self.markup().stream(**self._stream_args(req))

    def markup(self):
        from .markup import MarkupSynth
        return MarkupSynth(self)

    def _voice(self, req):
        return None if req.get("ref_wav") else req["voice"]

    def _stream_args(self, req):
        return dict(text=req["text"], voice=self._voice(req), tag=req["tag"], ctrl=req["ctrl"], pace=req["pace"],
                    ref_wav=req.get("ref_wav"), guidance=req["guidance"], flow=req["flow"], seed=req["seed"])

    # ------------------------------------------------------------------ rendering
    @torch.no_grad()
    def speaker_vector(self, voice=None, ref_wav=None):
        """[1,512]: a reference clip's x-vector (path or 24 kHz float array), else the stored vector of a voice."""
        if ref_wav is not None:
            key = str(ref_wav) if isinstance(ref_wav, (str, Path)) else None
            if key is None or key not in self._vec_cache:
                if self._xv is None:
                    from .speaker import XVec
                    self._xv = XVec(self.dev)
                    self._external(self._xv.m)
                y, _ = load_wav(ref_wav)
                v = self._xv([y])[0]
                if key is not None:
                    self._vec_cache[key] = v
            v = self._vec_cache[key] if key is not None else v
        else:
            if voice not in self.spk_ref:
                raise ValueError(f"unknown voice {voice!r}; see .voices")
            v = self.spk_ref[voice]
        return torch.from_numpy(np.asarray(v, np.float32))[None].to(self.dev)

    @torch.no_grad()
    def prompt_mel(self, ref_wav, mel, max_s=8.0):
        """Log-mel of the reference clip (first max_s s), level-matched to this render; bands above a band-limited
        clip's bandwidth are filled from the render's own spectral shape."""
        y, sr = load_wav(ref_wav)
        if not isinstance(ref_wav, (str, Path)):
            sr = SR
        y = y[: int(max_s * SR)]
        if len(y) < SR // 2:
            return None
        pm = mel_extractor()(torch.from_numpy(y)[None]).to(self.dev).float()
        mel = mel.float()
        nb = pm.shape[1]
        if sr < SR:
            hz = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + (SR / 2) / 700), nb + 2) / 2595) - 1)
            kb = int(np.searchsorted(hz[2:], 0.95 * sr / 2))
            nb = max(8, min(kb, pm.shape[1]))
        loud = lambda m: m[:, :nb].mean(1)[0].sort().values[m.shape[-1] // 2:].mean()
        pm = pm + (loud(mel) - loud(pm))
        if nb < pm.shape[1]:
            shape = mel.mean(-1, keepdim=True)
            pm[:, nb:] = pm[:, nb - 1:nb] + (shape[:, nb:] - shape[:, nb - 1:nb])
        return pm

    def control_tensors(self, ctrl):
        """Mask semantics: a dim named in ctrl is requested (value + mask 1); every other dim is unspecified (mask 0)
        and the model chooses it from voice, tag and text. ctrl={'rate': 0} is NOT the same as omitting rate: it
        asks for the voice's neutral rate."""
        c = torch.zeros(1, len(CONTROL_DIMS), device=self.dev)
        m = torch.zeros(1, len(CONTROL_DIMS), device=self.dev)
        for k, v in (ctrl or {}).items():
            i = CONTROL_DIMS.index(k)
            c[0, i], m[0, i] = float(v), 1.0
        return c, m

    @torch.no_grad()
    def _render(self, text, voice=None, tag="none", ctrl=None, pace=1.0, word_shift=None, ref_wav=None,
                guidance=None, flow=True, seed=0, return_mel=False):
        if ref_wav is None and voice is None:
            voice = self.default_voice
        ids, widx, _ = to_phonemes(text)
        if not ids:
            return np.zeros(0, np.float32)
        n_sym = self.model.emb.num_embeddings
        if max(ids) >= n_sym:
            keep = [k for k, i in enumerate(ids) if i < n_sym]
            ids, widx = [ids[k] for k in keep], [widx[k] for k in keep]
        ids_t = torch.tensor([ids], device=self.dev)
        widx_t = torch.tensor([widx], device=self.dev)
        c, m = self.control_tensors(ctrl)
        spk = torch.tensor([self.n_speakers], device=self.dev)  # table row: unused, the speaker vector conditions
        spk_vec = self.speaker_vector(voice, ref_wav)
        tg = torch.tensor([self.tag2i.get(tag, 0)], device=self.dev)
        if word_shift:  # user units: pitch in semitones, energy in dB, dur factor
            word_shift = {int(w): dict(pitch=s.get("pitch", 0) / PITCH_SCALE, energy=s.get("energy", 0) / 10.0,
                                       dur=s.get("dur", 1.0)) for w, s in word_shift.items()}
        kw = {}
        s = self.guidance if guidance is None else guidance
        if s != 1.0 and (ctrl or tag not in (None, "none")):
            kw.update(guidance=s, base=(torch.zeros_like(tg), torch.zeros_like(c), torch.zeros_like(m)))
        mel, aux = self.model.infer(ids_t, spk, tg, c, m, pace=pace, word_idx=widx_t, word_shift=word_shift,
                                    spk_vec=spk_vec, **kw)
        fl = None
        if flow and self.flow is not None:
            fl = self.pflow if (ref_wav is not None and self.pflow is not None) else self.flow
        elif flow and ref_wav is not None:
            fl = self.pflow
        if fl is not None:
            pm = self.prompt_mel(ref_wav, mel) if (ref_wav is not None and getattr(fl.net, "prompt", False)) else None
            if pm is None:
                mel = fl.sample(mel.float(), aux["g"].float(), steps=self.flow_steps, temp=self.flow_temp, seed=seed)
            else:
                P = pm.shape[-1]
                full = torch.cat([pm, mel.float()], -1)
                cm = torch.zeros_like(full[:, :1]); cm[..., :P] = 1
                mel = fl.sample(full, aux["g"].float(), steps=self.flow_steps, temp=self.flow_temp, seed=seed,
                                ctx=full, cm=cm, cfg=0.0)[..., P:]
        wav = self.vocos.decode(mel.float()).squeeze().cpu().numpy().astype(np.float32)
        return (wav, mel) if return_mel else wav

    @torch.no_grad()
    def _vocode(self, mel):
        """log-mel [100, T] (numpy) -> waveform, through the same vocoder as the speech."""
        m = torch.from_numpy(np.asarray(mel, np.float32))[None].to(self.dev)
        return self.vocos.decode(m).squeeze().cpu().numpy().astype(np.float32)

    def _has_event(self, text):
        from .text import _EVENT_RE
        return any(m.group(1).lower() in self.events for m in _EVENT_RE.finditer(text or ""))

    # ------------------------------------------------------------------ helpers
    def voice_stats(self, voice):
        """The voice's neutral raw control means (for steertts.measure.measure_z)."""
        return self.spk_mean[voice]

    def save(self, wav, path):
        save_wav(wav, path)

    def __repr__(self):
        prec = "" if self.precision == "fp32" else f", precision={self.precision!r}"
        return f"SteerTTS({str(self.model_dir)!r}, device={self.dev!r}, flow={self.flow is not None}{prec})"


SteerTTSPipeline = SteerTTS  # name used by whissle-sdk <= 0.7
