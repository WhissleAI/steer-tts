"""SteerTTS v1 edge build: int4 acoustic model + tiny vocoder + compact text front end (no torch hub / NLTK /
g2p_en / Vocos / WavLM / MiniLM at runtime). Two runtimes over the same package: EdgeTTS (PyTorch) and OnnxTTS
(ONNX Runtime, no torch). Both are imported lazily, so importing the package loads neither. Director (numpy only)
plans how each sentence of a longer text is performed (director.bin; docs/director.md)."""
from .config import CONTROL_DIMS, SR

__version__ = "1.4.0"
__all__ = ["EdgeTTS", "OnnxTTS", "Director", "SR", "CONTROL_DIMS", "__version__"]


def __getattr__(name):
    if name == "EdgeTTS":
        from .pipeline import EdgeTTS
        return EdgeTTS
    if name == "Director":
        from .director import Director
        return Director
    if name == "OnnxTTS":
        from .ort import OnnxTTS
        return OnnxTTS
    raise AttributeError(name)
