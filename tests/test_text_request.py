import json
from pathlib import Path

import pytest

from steertts import CONTROL_DIMS, normalize_request, schema
from steertts.text import SYMBOLS, normalize, to_phonemes


def test_front_end_expands_numbers_units_acronyms():
    # numbers are spelled out by the front end itself (21-99 without g2p_en's hyphens, so they stay dictionary words)
    t = normalize("It costs $3,000, 20% more than 1990-1995, at 65 mph, a 4x4 via the IOC.")
    assert "three thousand dollars" in t and "twenty percent" in t and "nineteen ninety to nineteen ninety five" in t
    assert "sixty five miles per hour" in t and "four by four" in t and "I O C" in t and "-" not in t


# normalisation cases shared by every runtime's tests (Python, edge, C, C++): the expected spoken form of each text
FRONT_CASES = json.loads((Path(__file__).parent / "front_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", FRONT_CASES, ids=[c["text"] for c in FRONT_CASES])
def test_front_cases(case):
    assert normalize(case["text"]) == case["norm"]


def test_letters_times_and_apostrophes_read_as_spoken():
    def phones(t):
        return [SYMBOLS[i] for i in to_phonemes(t)[0]]
    assert phones("aren’t you") == phones("aren't you")             # curly apostrophe = ASCII apostrophe
    assert phones("5 a.m.") == phones("5 AM") == phones("5am")      # a.m. / AM / am after a number: the letters A M
    assert phones("4:50 am")[-5:] == ["EY1", "<sp>", "EH1", "M", "<sp>"]
    assert phones("5 p.m.")[-6:] == ["P", "IY1", "<sp>", "EH1", "M", "<sp>"]
    assert "<sp> AE1 M <sp>" in " ".join(phones("I am here"))       # the word "am" is untouched


def test_phonemes_and_word_index():
    ids, widx, words = to_phonemes("Hello there, world.")
    assert len(ids) == len(widx) and max(widx) == 2 and words == ["Hello", "there", "world"]
    ids, _, _ = to_phonemes("Ha [laugh] okay [notatoken] fine")
    assert SYMBOLS.index("[laugh]") in ids  # event tokens survive; unknown [..] markers are dropped


def test_schema_examples_validate():
    jsonschema = pytest.importorskip("jsonschema")
    s = schema()
    jsonschema.Draft202012Validator.check_schema(s)
    for ex in s["examples"]:
        jsonschema.validate(ex, s)
    assert set(s["properties"]["ctrl"]["properties"]) == set(CONTROL_DIMS)


def test_request_validation(tts):
    r = normalize_request({"text": "Hi."}, tts)
    assert r["voice"] == tts.default_voice and r["guidance"] == 1.0 and r["ctrl"] == {}
    for bad, msg in [({"text": "x", "ctrl": {"speed": 1}}, "unknown control"),
                     ({"text": "x", "ctrl": {"rate": 9}}, "outside"),
                     ({"text": "x", "tag": "angry"}, "unknown tag"),
                     ({"text": "x", "voice": "voice_9999x"}, "unknown voice"),
                     ({"text": "x", "voice": tts.default_voice, "ref_wav": "a.wav"}, "not both"),
                     ({"text": "x", "colour": 1}, "unknown request field"),
                     ({"text": ""}, "non-empty")]:
        with pytest.raises(ValueError, match=msg):
            normalize_request(bad, tts)


def test_resolved_request_is_valid_json(tts):
    jsonschema = pytest.importorskip("jsonschema")
    r = tts.resolve({"text": "Hello.", "tag": "whisper", "ctrl": {"rate": -1}})
    jsonschema.validate(json.loads(json.dumps(r)), schema())


def test_stream_rejects_word_shift(tts):
    with pytest.raises(ValueError, match="stream\\(\\) does not support word_shift"):
        tts.stream("I never said that.", word_shift={1: {"pitch": 3}})
