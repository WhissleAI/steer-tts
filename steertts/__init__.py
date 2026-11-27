"""SteerTTS: a small, numerically steerable English TTS (hybrid attention + bidirectional sequence mixing, AdaLN
on a masked 12-dim control request), with optional flow-matching refiners, a free-text prompt adapter and zero-shot
cloning from a reference clip. Director (numpy only) plans how each sentence of a longer text is performed
(director.bin; docs/director.md)."""
from .config import CONTROL_DIMS, DEFAULT_VOICE, SR
from .director import Director
from .pipeline import SteerTTS, SteerTTSPipeline, resolve_model_dir
from .request import normalize_request, schema

__all__ = ["SteerTTS", "SteerTTSPipeline", "CONTROL_DIMS", "DEFAULT_VOICE", "SR", "normalize_request", "schema",
           "resolve_model_dir", "Director"]
__version__ = "1.6.0"
