"""Flow-matching mel decoders (optional), inference path.

refiner (flow.safetensors): a conditional flow (OT-CFM, as the Matcha-TTS decoder) that samples a detailed mel
    given the acoustic model's mel mu and its conditioning g. Durations, pitch, energy, the coarse spectrum and
    every control still come from the acoustic model; the flow adds texture (sharper harmonics / noise).
prompted flow (pflow.safetensors): the same with an audio prompt - the reference clip's own mel is given as
    context before the target (Voicebox / F5-TTS-style infilling). Used only for zero-shot cloning.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N_MELS

SIGMA = 1e-4

class TimeEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))

    def forward(self, t):  # [B] in [0, 1]
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = 1000 * t[:, None] * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class Block(nn.Module):
    """ConvNeXt block (depthwise dilated conv -> LN -> MLP) with AdaLN-zero conditioning on (t, g)."""

    def __init__(self, d, c, k=7, dil=1, mult=3):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, padding=dil * (k // 2), dilation=dil, groups=d)
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.mod = nn.Linear(c, 3 * d)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
        self.pw1, self.pw2 = nn.Linear(d, mult * d), nn.Linear(mult * d, d)

    def forward(self, x, m, c):  # x [B,d,T], m [B,1,T], c [B,c]
        sc, sh, gate = self.mod(c)[:, None].chunk(3, -1)
        h = self.dw(x * m).transpose(1, 2)
        h = self.ln(h) * (1 + sc) + sh
        h = self.pw2(F.gelu(self.pw1(h))) * gate
        return x + h.transpose(1, 2)


class AttnBlock(nn.Module):
    """Full self-attention over the sequence with AdaLN-zero conditioning: lets the audio prompt (prompted flow)
    reach every target frame, beyond the conv receptive field (~170 frames each side)."""

    def __init__(self, d, c, heads=6):
        super().__init__()
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.mod = nn.Linear(c, 3 * d)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, x, m, c):
        sc, sh, gate = self.mod(c)[:, None].chunk(3, -1)
        h = self.ln(x.transpose(1, 2)) * (1 + sc) + sh
        h, _ = self.att(h, h, h, key_padding_mask=~m[:, 0].bool(), need_weights=False)
        return x + (h * gate).transpose(1, 2) * m


def rope(x):  # x [B,H,T,dh]: rotary position embedding (Su et al. 2021) over the frame axis
    T, dh = x.shape[-2], x.shape[-1]
    f = torch.exp(-math.log(10000) * torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh)
    a = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * f[None]
    cos, sin = a.cos().to(x.dtype), a.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class PromptPool(nn.Module):
    """Perceiver-resampler pooling of the prompt (XTTS; SupertonicTTS's learnable timbre queries): n learnable
    queries cross-attend to the encoded prompt frames -> n global voice tokens. A learned null key is always
    present, so an item without a prompt still gets defined (prompt-free) tokens."""

    def __init__(self, d, n, heads=6, layers=2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n, d) * 0.02)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.heads = heads
        self.layers = nn.ModuleList([nn.ModuleDict(dict(
            lq=nn.LayerNorm(d), lk=nn.LayerNorm(d), q=nn.Linear(d, d), kv=nn.Linear(d, 2 * d), o=nn.Linear(d, d),
            lf=nn.LayerNorm(d), f=nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))))
            for _ in range(layers)])

    def forward(self, e, cm):  # e [B,d,T] encoded prompt (zero off-prompt), cm [B,1,T]
        B, d, T = e.shape
        kv_in = torch.cat([self.null.expand(B, 1, d), e.transpose(1, 2)], 1)
        keep = torch.cat([torch.ones(B, 1, device=e.device, dtype=torch.bool), cm[:, 0].bool()], 1)[:, None, None]
        q = self.q[None].expand(B, -1, -1)
        H = self.heads
        for l in self.layers:
            k, v = l["kv"](l["lk"](kv_in)).chunk(2, -1)
            qq = l["q"](l["lq"](q))
            sp = lambda z: z.view(B, -1, H, d // H).transpose(1, 2)
            h = F.scaled_dot_product_attention(sp(qq), sp(k), sp(v), attn_mask=keep)
            q = q + l["o"](h.transpose(1, 2).reshape(B, -1, d))
            q = q + l["f"](l["lf"](q))
        return q  # [B,n,d]


class RAttnBlock(nn.Module):
    """Self-attention with RoPE positions (frames), plus optional cross-attention to the pooled prompt tokens; both
    zero-gated (AdaLN-zero gate / zero output projection), so each block is an identity
    at init."""

    def __init__(self, d, c, heads=6, xattn=False):
        super().__init__()
        self.heads = heads
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.mod = nn.Linear(c, 3 * d)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.xattn = xattn
        if xattn:
            self.lx = nn.LayerNorm(d)
            self.xq, self.xkv, self.xo = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
            nn.init.zeros_(self.xo.weight); nn.init.zeros_(self.xo.bias)

    def forward(self, x, m, c, tok=None):  # x [B,d,T], tok [B,n,d]
        B, d, T = x.shape
        H = self.heads
        sp = lambda z: z.view(B, -1, H, d // H).transpose(1, 2)
        sc, sh, gate = self.mod(c)[:, None].chunk(3, -1)
        h = self.ln(x.transpose(1, 2)) * (1 + sc) + sh
        q, k, v = (sp(z) for z in self.qkv(h).chunk(3, -1))
        h = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=m[:, 0].bool()[:, None, None])
        h = self.o(h.transpose(1, 2).reshape(B, T, d)) * gate
        if self.xattn and tok is not None:
            hx = self.lx(x.transpose(1, 2) + h)
            k2, v2 = self.xkv(tok).chunk(2, -1)
            h = h + self.xo(F.scaled_dot_product_attention(sp(self.xq(hx)), sp(k2), sp(v2)).transpose(1, 2).reshape(B, T, d))
        return x + h.transpose(1, 2) * m


class FlowNet(nn.Module):
    """prompt=True (prompted flow): also takes a context mel ctx with mask cm (1 = real audio given: the reference
    clip's frames), as Voicebox / F5-TTS infilling. The context enters the input stack, a masked mean of it is added
    to the conditioning (global voice), and attention layers at attn_at carry it across the whole sequence.

    Optional blocks (all zero-initialised):
      rope_at   layers that get an extra RoPE self-attention block (RAttnBlock)
      n_ptok    pooled prompt tokens (PromptPool) that the RoPE blocks cross-attend to
      skips     E2-TTS-style long skips, block i -> block n-1-i
      align_at  layers whose target-frame mean is mapped to a speaker-verification vector (TLA-SA-style feature
                alignment; not used at inference)"""

    def __init__(self, n_mel=N_MELS, d=384, n_layers=12, g_dim=128, dils=(1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4),
                 prompt=False, attn_at=(3, 7, 11), rope_at=(), n_ptok=0, skips=False, align_at=(), align_dim=512):
        super().__init__()
        self.cfg = dict(n_mel=n_mel, d=d, n_layers=n_layers, g_dim=g_dim, dils=tuple(dils), prompt=prompt,
                        attn_at=tuple(attn_at), rope_at=tuple(rope_at), n_ptok=n_ptok, skips=skips,
                        align_at=tuple(align_at), align_dim=align_dim)
        self.prompt = prompt
        self.inp = nn.Conv1d((3 * n_mel + 1) if prompt else 2 * n_mel, d, 3, padding=1)
        self.temb = TimeEmb(d)
        self.gemb = nn.Sequential(nn.Linear(g_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, d, dil=dils[i % len(dils)]) for i in range(n_layers)])
        if prompt:
            self.attn = nn.ModuleDict({str(i): AttnBlock(d, d) for i in attn_at})
            self.ctx_enc = nn.Sequential(nn.Conv1d(n_mel, d, 3, padding=1), nn.SiLU(), nn.Conv1d(d, d, 3, padding=1))
            self.ctx_out = nn.Linear(d, d)
            nn.init.zeros_(self.ctx_out.weight); nn.init.zeros_(self.ctx_out.bias)
            self.rattn = nn.ModuleDict({str(i): RAttnBlock(d, d, xattn=n_ptok > 0) for i in rope_at})
            self.pool = PromptPool(d, n_ptok) if n_ptok else None
        self.skips = skips
        if skips:
            self.skip = nn.ModuleDict({str(i): nn.Linear(d, d) for i in range(n_layers - n_layers // 2, n_layers)})
            for l in self.skip.values():
                nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
        self.align_at = tuple(align_at)
        if align_at:
            self.align = nn.ModuleDict({str(i): nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, align_dim))
                                        for i in align_at})
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.out = nn.Conv1d(d, n_mel, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t, mu, g, m, ctx=None, cm=None, feats=False):  # normalised mels [B,100,T]; m, cm [B,1,T]
        """feats=True also returns {layer: speaker vector predicted from the target-frame mean} for align_at."""
        c = self.temb(t) + self.gemb(g)
        tok = None
        if self.prompt:
            if cm is None:
                ctx, cm = torch.zeros_like(x), torch.zeros_like(m)
            ctx = ctx * cm
            e = self.ctx_enc(ctx) * cm
            c = c + self.ctx_out(e.sum(-1) / cm.sum(-1).clamp_min(1))
            if getattr(self, "pool", None) is not None:
                tok = self.pool(e, cm)
            h = self.inp(torch.cat([x, mu, ctx, cm], 1) * m)
        else:
            h = self.inp(torch.cat([x, mu], 1) * m)
        rattn = getattr(self, "rattn", {})
        stack, out = [], {}
        n = len(self.blocks)
        for i, b in enumerate(self.blocks):
            if self.skips and str(i) in self.skip:
                h = h + self.skip[str(i)](stack.pop().transpose(1, 2)).transpose(1, 2) * m
            h = b(h, m, c)
            if self.prompt and str(i) in self.attn:
                h = self.attn[str(i)](h, m, c)
            if str(i) in rattn:
                h = rattn[str(i)](h, m, c, tok)
            if self.skips and i < n // 2:
                stack.append(h)
            if feats and i in self.align_at:
                w = m * (1 - cm) if cm is not None else m
                out[i] = self.align[str(i)](((h * w).sum(-1) / w.sum(-1).clamp_min(1)).float())
        v = self.out(self.ln(h.transpose(1, 2)).transpose(1, 2)) * m
        return (v, out) if feats else v


class Flow(nn.Module):
    """FlowNet + mel normalisation; sample() maps a base mel to a refined mel (log-mel units, same frames)."""

    def __init__(self, mean=-5.0, std=2.5, mu_cmn=False, **kw):
        """mu_cmn (prompted flow): remove the per-band time-mean of the base mel (its static spectral envelope:
        the long-term voice colouring the base model guessed from a 512-d speaker vector) so the flow must take the
        voice from the audio prompt (or g); timing / pitch / energy / phonetic detail of mu are kept. Without it, mu
        made from a speaker vector already carries that voice's colouring and the prompt adds
        little."""
        super().__init__()
        self.net = FlowNet(**kw)
        self.mu_cmn = mu_cmn
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def norm(self, m):
        return (m - self.mean) / self.std

    def _cmn(self, mu_n, w):
        """mu_n minus its per-band mean over the frames w marks (the target frames)."""
        if not self.mu_cmn:
            return mu_n
        return mu_n - (mu_n * w).sum(-1, keepdim=True) / w.sum(-1, keepdim=True).clamp_min(1)

    @torch.no_grad()
    def sample(self, mu, g, m=None, steps=10, temp=0.8, method="midpoint", seed=None, ctx=None, cm=None, cfg=0.0):
        """ctx / cm (prompted flow): real mel on the context frames (log-mel units), mask 1 there; mu is replaced by
        the context on those frames. Returns every frame (callers slice off the context).
        cfg (prompted flow): classifier-free guidance on the prompt, v + cfg * (v - v_no_prompt); the no-prompt
        velocity is the same net with the context removed."""
        m = torch.ones_like(mu[:, :1]) if m is None else m
        gen = None if seed is None else torch.Generator(device=mu.device).manual_seed(seed)
        mu_n = self._cmn(self.norm(mu), m if cm is None else m * (1 - cm))
        if cm is not None:
            ctx = self.norm(ctx) * cm
            mu_n = torch.where(cm.bool(), ctx, mu_n)
        x = temp * torch.randn(mu_n.shape, device=mu.device, generator=gen) * m
        ts = torch.linspace(0, 1, steps + 1, device=mu.device)

        def vel(x, t):
            v = self.net(x, t.expand(x.shape[0]), mu_n, g, m, ctx, cm)
            if cfg and cm is not None:
                v = v + cfg * (v - self.net(x, t.expand(x.shape[0]), mu_n, g, m, None, None))
            return v

        for i in range(steps):
            t0, dt = ts[i], ts[i + 1] - ts[i]
            v = vel(x, t0)
            if method == "midpoint":
                v = vel(x + 0.5 * dt * v, t0 + 0.5 * dt)
            x = x + dt * v
        return x * self.std + self.mean

    @staticmethod
    def from_state(state, cfg, device="cpu"):
        f = Flow(**cfg)
        f.load_state_dict(state)
        return f.to(device).eval()
