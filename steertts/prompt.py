"""Free-text voice direction -> request fields (style tag + which controls + their z values).

A frozen sentence encoder (sentence-transformers/all-MiniLM-L6-v2, 22.7M, mean-pooled and L2-normalised exactly
as the sentence-transformers wrapper does) feeds a ~0.5M-parameter adapter whose outputs are,
for each control a "specified" probability (> 0.5 -> the dim is set) and a z value, plus a distribution
over the style tags. Needs `transformers` (the `prompt` extra).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CONTROL_DIMS

ENC = "sentence-transformers/all-MiniLM-L6-v2"
N = len(CONTROL_DIMS)


class Adapter(nn.Module):
    def __init__(self, n_tags, d_in=384, h=512):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h), nn.GELU())
        self.tag = nn.Linear(h, n_tags)
        self.spec = nn.Linear(h, N)
        self.val = nn.Linear(h, N)

    def forward(self, e):
        h = self.body(e)
        return self.tag(h), self.spec(h), self.val(h)


class MiniLM:
    """all-MiniLM-L6-v2 sentence embedding: token embeddings mean-pooled over the attention mask, L2-normalised,
    inputs truncated at 256 word pieces (the model's sentence-transformers configuration)."""

    def __init__(self, device="cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ENC)
        self.m = AutoModel.from_pretrained(ENC).to(device).eval()
        self.dev = device

    @torch.no_grad()
    def __call__(self, texts):
        b = self.tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.dev)
        h = self.m(**b).last_hidden_state
        m = b["attention_mask"][..., None].to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return F.normalize(e, p=2, dim=1)


class PromptMapper:
    def __init__(self, state_dict, tags, device="cpu"):
        self.tags, self.dev = list(tags), device
        self.m = Adapter(len(self.tags)).to(device).eval()
        self.m.load_state_dict(state_dict)
        self.enc = MiniLM(device)
        self.cache = {}

    @torch.no_grad()
    def __call__(self, prompt, spec_threshold=0.5, top_tags=3):
        """-> dict(top_tags=[(tag, p), ...], ctrl={dim: z}) for the dims whose specified-probability > threshold."""
        key = " ".join(prompt.lower().split())
        if key not in self.cache:
            lt, ls, lv = self.m(self.enc([prompt]))
            self.cache[key] = (torch.softmax(lt, -1)[0], torch.sigmoid(ls)[0], lv[0])
        tp, sp, v = self.cache[key]
        ctrl = {CONTROL_DIMS[i]: round(float(v[i]), 2) for i in range(N) if sp[i] > spec_threshold}
        top = torch.topk(tp, top_tags)
        return dict(top_tags=[(self.tags[i], round(float(p), 3)) for p, i in zip(top.values.tolist(), top.indices.tolist())],
                    ctrl=ctrl)
