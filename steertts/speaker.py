"""Speaker vectors for zero-shot cloning: frozen WavLM-base-plus-sv x-vectors (L2-normalised, 512-d), the same
encoder the acoustic model is conditioned on. Needs `transformers` (the `clone` extra)."""
import numpy as np
import torch

from .config import SR

MODEL = "microsoft/wavlm-base-plus-sv"
DIM = 512


class XVec:
    def __init__(self, dev="cpu"):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL)
        self.m = WavLMForXVector.from_pretrained(MODEL).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, wavs24, max_s=8.0):
        """list of float32 arrays at 24 kHz -> [n, 512] L2-normalised (first max_s seconds of each)."""
        import soxr
        ys = [soxr.resample(w[: int(max_s * SR)].astype(np.float32), SR, 16000) for w in wavs24]
        ys = [np.tile(y, int(np.ceil(8000 / max(len(y), 1)))) if len(y) < 8000 else y for y in ys]
        x = self.fe(ys, sampling_rate=16000, return_tensors="pt", padding=True).to(self.dev)
        e = self.m(**x).embeddings
        return torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()
