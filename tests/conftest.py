import os
from pathlib import Path

import pytest

MODEL = os.environ.get("STEERTTS_MODEL") or str(Path(__file__).resolve().parents[1])
DEV = os.environ.get("STEERTTS_TEST_DEVICE", "cpu")


@pytest.fixture(scope="session")
def model_dir():
    if not (Path(MODEL) / "config.json").exists():
        pytest.skip(f"no model directory at {MODEL} (set STEERTTS_MODEL)")
    return MODEL


@pytest.fixture(scope="session")
def tts(model_dir):
    from steertts import SteerTTS
    return SteerTTS(model_dir, device=DEV, flow=False)


@pytest.fixture(scope="session")
def tts_flow(model_dir):
    from steertts import SteerTTS
    return SteerTTS(model_dir, device=DEV)
