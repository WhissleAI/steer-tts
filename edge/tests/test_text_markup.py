"""Text edge cases, inline markup, streaming."""
import numpy as np
import pytest

from conftest import SR, check_audio

LONG = ("The committee met on a grey Tuesday morning to review the proposal for the new library. "
        "Several members had read the full report, while others had only skimmed the summary on the train. "
        "After a short introduction, the chair asked the architect to explain why the budget had grown, and she "
        "answered calmly, pointing to the cost of steel, the delays caused by the winter storms, and the extra "
        "work needed to protect the old oak trees beside the entrance. A long discussion followed. Some argued "
        "that the town could not afford it; others said the library would serve children, students and retired "
        "readers for the next fifty years, and that cutting corners now would cost far more later. By noon they "
        "had agreed on a smaller reading room, a larger garden, and a promise to publish every invoice online so "
        "that anyone could check how the money was being spent. Everyone left tired but quietly satisfied. The next meeting is in May.")


@pytest.mark.parametrize("text", ["", "   ", "\n\t", "...", "!?", "—", "日本語", "😀"])
def test_nothing_to_say(tts, text):
    y = tts(text)
    assert y.dtype == np.float32 and len(y) == 0


def test_long_paragraph(tts):
    assert 900 < len(LONG) < 1200
    d = check_audio(tts(LONG), 30, 120)
    assert 40 < d < 100, d
    d2 = check_audio(tts(LONG, markup=True), 30, 120)   # sentence-split rendering
    assert abs(d2 - d) < 0.3 * d


@pytest.mark.parametrize("text", [
    "It costs $3,499.99, about £2,800 or €3,200.",
    "Call 555-0198 before 5:30 pm on March 3rd, 2024.",
    "Dr. Smith lives at 221B Baker St. in the U.K.",
    "Growth was 12.5% in 1999-2003, roughly 4x4 faster.",
    "The 21st century began in 2001; the 100,000th visitor arrived at 10:45.",
    "Mr. and Mrs. O'Brien met the CEO of NASA and the BBC.",
    "Speed: 120 km/h, weight 75 kg, file 3 GB.",
])
def test_numbers_currency_dates_abbreviations(tts, text):
    check_audio(tts(text), 1.0, 20.0)


@pytest.mark.parametrize("text", [
    "Zyxquorpian and Thraxwellington met Ngozi Okonkwo-Adeyemi in Szczebrzeszyn.",
    "“Smart quotes” — and ‘single ones’ – with an ellipsis…",
    "Café, naïve, résumé, façade, Zoë, São Paulo, Dvořák.",
    "Hello!!! Is this... really?! Yes; truly: (maybe) [sort of] {braces} <angle>",
    "MIXED case AND ALLCAPS words like IKEA, FBI and UNESCO.",
    "x",
])
def test_oov_unicode_punctuation(tts, text):
    check_audio(tts(text), 0.15, 20.0)


def test_markup_features(tts):
    plain = tts("Well, that was fast.")
    emph = tts("Well, *that* was fast.", markup=True)
    assert len(emph) > len(plain) and not np.array_equal(plain[: len(emph)], emph[: len(plain)])
    paused = tts("First part. [pause 400ms] Second part.", markup=True)
    nopause = tts("First part. Second part.", markup=True)
    assert abs((len(paused) - len(nopause)) - int(0.4 * SR)) <= 1
    base = len(tts("Let me think about it.", markup=True))
    assert len(tts("<slow>Let me think about it.</slow>", markup=True)) > 1.15 * base
    assert len(tts("<fast>Let me think about it.</fast>", markup=True)) < 0.9 * base
    rms = lambda y: float(np.sqrt(np.mean(y ** 2)))
    assert rms(tts("<loud>Let me think about it.</loud>", markup=True)) > rms(
        tts("<soft>Let me think about it.</soft>", markup=True))
    for t in ("<high>up here</high> and <low>down there</low>.", "<whisper>Don't tell anyone.</whisper>",
              "Normal, then <sadness>this part is sad.</sadness> And back.", "<emph>Really</emph> now.",
              "[pause 1.5s] Late start.", "Unknown <foo>tags</foo> are spoken normally.", "Bad [pause] default.",
              "<slow><loud>nested *spans*</loud></slow> work."):
        check_audio(tts(t, markup=True), 0.3, 20.0, min_peak=0.005)
    with pytest.raises(ValueError, match="host"):
        tts('<style prompt="angry">No prompts on device.</style>', markup=True)


def test_stream(tts):
    chunks = list(tts.stream("First sentence here. [pause 300ms] *Second* one. <whisper>Third.</whisper>"))
    kinds = [("pause" if c.segment.pause_ms else c.segment.tag) for c in chunks]
    assert kinds == ["none", "pause", "none", "whisper"], kinds
    assert len(chunks[1].audio) == int(0.3 * SR)
    assert chunks[2].segment.word_shift
    full = tts("First sentence here. [pause 300ms] *Second* one. <whisper>Third.</whisper>", markup=True)
    assert np.array_equal(np.concatenate([c.audio for c in chunks]), full)
    assert all(c.synth_ms >= 0 for c in chunks)
