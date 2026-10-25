"""Inline markup + sentence-level streaming.

Markup (nestable; unknown tags are dropped and their text spoken normally):
  *word*  or  <emph>words</emph>          emphasis: pitch +2.5 st, energy +4 dB, 20% longer       (per word)
  <slow> <fast> <high> <low> <loud> <soft>   per-word duration / pitch / energy changes inside the
                                              sentence - prosody stays continuous                     (per word)
  [pause] [pause 300ms] [pause 1.5s]          silence (default 400 ms, max 5 s)
  <whisper>..</whisper>, <sadness>..</sadness>, any style tag the model knows
                                              the span is rendered as its own segment with that tag   (span)
  <style prompt="angry and fast">..</style>   the span gets the prompt adapter's tag + controls       (span)
  [laugh] [cough] [dog] [knock] ...           event tokens: the model reserves time for them, but renders no
                                              sound (a recorded-sound mixer
                                              is not part of this release)

Rendering: the text is cut into segments at sentence ends (. ! ?), pauses and style-span boundaries; each segment
is one forward pass with its tag/controls and per-word shifts, and segments are yielded in order, so a caller can
play the first sentence while the rest renders (stream()). A style switch mid-sentence joins two separately
rendered segments (a short audible boundary): there is no token-level style conditioning.
"""
import re
from dataclasses import dataclass, field

import numpy as np

from .config import SR
from .text import to_phonemes

PROSODY = {  # Synth.word_shift user units: pitch semitones, energy dB, dur factor
    "emph": dict(pitch=2.5, energy=4.0, dur=1.2), "slow": dict(dur=1.4), "fast": dict(dur=0.75),
    "high": dict(pitch=3.0), "low": dict(pitch=-3.0), "loud": dict(energy=6.0), "soft": dict(energy=-6.0),
}
_TOKEN = re.compile(r'(<\s*(/?)\s*([A-Za-z_]+)(?:\s+prompt\s*=\s*"([^"]*)")?\s*>)'
                    r'|(\[\s*([A-Za-z_]+)(?:\s+([\d.]+)\s*(ms|s)?)?\s*\])', re.I)
# a sentence ends at . ! ? and a space, but not after a title, a common abbreviation or an initial ("Dr. Smith",
# "e.g. this", "J. K. Rowling"): those would split one sentence into two segments, with a pause between
_SENT_ABBREV = ["Dr", "Mr", "Mrs", "Ms", "Prof", "St", "Jr", "Sr", "vs", "No", "Fig", "Vol", "pp", "approx", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec"]
_SENT_END = re.compile(r"(?<=[.!?])" + "".join(rf"(?<!\b{a}\.)" for a in _SENT_ABBREV)
                       + r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\b[ap]\.m\.)(?<!\b[A-Z]\.)\s+")


@dataclass
class Segment:
    text: str
    tag: str
    ctrl: dict
    word_shift: dict = field(default_factory=dict)
    pause_ms: int = 0  # silence instead of speech when > 0 (text empty)
    event: str = ""  # environmental sound event rendered by the EnvMixer (text empty)


@dataclass
class Chunk:
    audio: np.ndarray
    segment: Segment
    synth_ms: float


def _n_words(text):
    if not text.strip():
        return 0
    _, widx, _ = to_phonemes(text)
    return max(widx) + 1 if max(widx) >= 0 else 0


class MarkupSynth:
    def __init__(self, tts):
        """tts: a steertts.SteerTTS (its prompt adapter is loaded on first use of a prompt)."""
        self.s = tts

    # ------------------------------------------------------------------ parsing
    def _style_of_prompt(self, prompt):
        if not prompt:
            return "none", {}
        m = self.s.prompt_mapper(prompt)
        # the adapter's best tag that this model knows (the adapter can rank first a tag the acoustic model
        # lacks, e.g. 'irritation')
        tag = next((t for t, _ in m["top_tags"] if t in self.s.tag2i and t != "none"), "none")
        return tag, dict(m["ctrl"])

    def parse(self, text, prompt=None, tag=None, ctrl=None):
        """-> list[Segment]. prompt/tag/ctrl give the default style for unmarked text."""
        d_tag, d_ctrl = self._style_of_prompt(prompt)
        if tag:
            d_tag = tag if tag in self.s.tag2i else d_tag
        d_ctrl = {**d_ctrl, **(ctrl or {})}
        text = re.sub(r"\*([^*\n]+)\*", r"<emph>\1</emph>", text)
        runs = []  # (text, (tag, ctrl_items), frozenset(prosody)) or ("PAUSE", ms)
        style_stack, pros_stack = [(d_tag, tuple(sorted(d_ctrl.items())))], []
        pos = 0
        for m in _TOKEN.finditer(text):
            if m.start() > pos:
                runs.append((text[pos:m.start()], style_stack[-1], frozenset(pros_stack)))
            pos = m.end()
            if m.group(5):  # [pause ...] / [event]
                name = m.group(6).lower()
                if name == "pause":
                    v, unit = m.group(7), (m.group(8) or "ms").lower()
                    ms = int(float(v) * (1000 if unit == "s" else 1)) if v else 400
                    runs.append(("PAUSE", min(ms, 5000)))
                else:  # event token (or unknown: dropped by to_phonemes) stays in the TTS text
                    runs.append((f" [{name}] ", style_stack[-1], frozenset(pros_stack)))
                continue
            closing, name, prm = bool(m.group(2)), m.group(3).lower(), m.group(4)
            if name in PROSODY:
                if closing:
                    if name in pros_stack:
                        pros_stack.remove(name)
                else:
                    pros_stack.append(name)
            elif name == "style" or name in self.s.tag2i:
                if closing:
                    if len(style_stack) > 1:
                        style_stack.pop()
                elif name == "style":
                    t, c = self._style_of_prompt(prm)
                    style_stack.append((t, tuple(sorted(c.items()))))
                else:
                    style_stack.append((name, ()))  # a named style tag resets the default controls
            # unknown tag: the marker is dropped, its inner text is spoken normally
        if pos < len(text):
            runs.append((text[pos:], style_stack[-1], frozenset(pros_stack)))

        # split runs at sentence ends so each sentence is its own (streamable) segment
        pieces = []
        for r in runs:
            if r[0] in ("PAUSE", "EVENT"):
                pieces.append(r); continue
            parts = _SENT_END.split(r[0])
            for i, p in enumerate(parts):
                pieces.append((p, r[1], r[2]))
                if i < len(parts) - 1:
                    pieces.append(("BREAK",))
        # group consecutive pieces with the same style into segments
        segs, cur, cur_style = [], [], None

        def flush():
            if cur and "".join(p[0] for p in cur).strip():
                segs.append(self._segment(cur, cur_style))

        for p in pieces:
            if p[0] in ("PAUSE", "BREAK", "EVENT"):
                flush(); cur, cur_style = [], None
                if p[0] == "PAUSE":
                    segs.append(Segment("", "none", {}, pause_ms=p[1]))
                elif p[0] == "EVENT":
                    segs.append(Segment("", "none", {}, event=p[1]))
                continue
            if cur_style is not None and p[1] != cur_style:
                flush(); cur = []
            cur_style = p[1]
            cur.append(p)
        flush()
        return segs

    def _segment(self, pieces, style):
        text, shift, offset = "", {}, 0
        for p_text, _, pros in pieces:
            before = _n_words(text)
            text += p_text
            if pros:
                after = _n_words(text)
                sh = {}
                for name in pros:
                    for k, v in PROSODY[name].items():
                        sh[k] = sh.get(k, 1.0 if k == "dur" else 0.0) * v if k == "dur" else sh.get(k, 0.0) + v
                for w in range(before, after):
                    shift[w] = sh
        tag, ctrl_items = style
        return Segment(re.sub(r"\s+", " ", text).strip(), tag, dict(ctrl_items), shift)

    # ------------------------------------------------------------------ rendering
    def stream(self, text, voice=None, prompt=None, tag=None, ctrl=None, pace=1.0, ref_wav=None, guidance=None,
               flow=True, seed=0):
        """Yields Chunk per segment, in order (first chunk = first sentence / span)."""
        import time
        fade = int(0.01 * SR)
        for seg in self.parse(text, prompt, tag, ctrl):
            t0 = time.time()
            if seg.pause_ms:
                y = np.zeros(int(seg.pause_ms * SR / 1000), np.float32)
            else:
                y = np.asarray(self.s._render(seg.text, voice, tag=seg.tag, ctrl=seg.ctrl or None, pace=pace,
                                              word_shift=seg.word_shift or None, ref_wav=ref_wav, guidance=guidance,
                                              flow=flow, seed=seed), np.float32)
                if len(y) > 2 * fade:  # 10 ms fades so concatenated segments never click
                    ramp = np.linspace(0, 1, fade, dtype=np.float32)
                    y[:fade] *= ramp; y[-fade:] *= ramp[::-1]
            yield Chunk(y, seg, (time.time() - t0) * 1000)

    def render(self, text, **kw):
        return np.concatenate([c.audio for c in self.stream(text, **kw)] or [np.zeros(1, np.float32)])
