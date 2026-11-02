"""Text normalisation: the cases shared by every runtime's tests (tests/front_cases.json in the repo: the full
package, this package, the C library and the native C++ runtime all check the same texts and spoken forms)."""
import json
from pathlib import Path

import pytest

CASES_FILE = Path(__file__).resolve().parents[2] / "tests" / "front_cases.json"
CASES = json.loads(CASES_FILE.read_text(encoding="utf-8")) if CASES_FILE.exists() else []


@pytest.fixture(scope="module")
def text(model_dir):
    from steertts_edge import text as T
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    T.set_assets(Path(model_dir) / cfg.get("front", {}).get("dir", "front"))
    return T


@pytest.mark.skipif(not CASES, reason="no tests/front_cases.json (run from the repo)")
@pytest.mark.parametrize("case", CASES, ids=[c["text"] for c in CASES])
def test_front_cases(text, case):
    assert text.normalize(case["text"]) == case["norm"]


def test_letters_times_and_apostrophes_read_as_spoken(text):
    def phones(t):
        return [text.SYMBOLS[i] for i in text.to_phonemes(t)[0]]
    assert phones("aren\u2019t you") == phones("aren't you")        # curly apostrophe = ASCII apostrophe
    assert phones("5 a.m.") == phones("5 AM") == phones("5am")      # a.m. / AM / am after a number: the letters A M
    assert phones("4:50 am")[-5:] == ["EY1", "<sp>", "EH1", "M", "<sp>"]
    assert phones("5 p.m.")[-6:] == ["P", "IY1", "<sp>", "EH1", "M", "<sp>"]
