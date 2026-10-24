import json

import pytest

from steertts import CONTROL_DIMS, normalize_request, schema
from steertts.text import SYMBOLS, normalize, to_phonemes


def test_front_end_expands_numbers_units_acronyms():
    # what g2p_en gets wrong is rewritten; bare integers are left to g2p_en, which reads them correctly
    t = normalize("It costs $3,000, 20% more than 1990-1995, at 65 mph, a 4x4 via the IOC.")
    assert "three thousand dollars" in t and "20 percent" in t and "1990 to 1995" in t
    assert "65 miles per hour" in t and "4 by 4" in t and "I O C" in t


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
