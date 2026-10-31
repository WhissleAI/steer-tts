"""Sequence-mixing layer of the SteerTTS encoder / decoder blocks (inference path).

A gated, input-dependent linear recurrence with a rotating state, evaluated in chunks of `chunk` steps: within a
chunk as batched matmuls, across chunks by carrying the state. `mix_forward(layer, u)` is the layer's output for a
whole sequence; tests/test_seqmix.py checks it against the step-by-step recurrence at float64.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqMixLayer(nn.Module):
    """d_model in/out width, d_state state size per head (even), head_dim channels per head, rank write rank."""

    def __init__(self, d_model: int, d_state: int = 32, rank: int = 1, head_dim: int = 32):
        super().__init__()
        assert d_state % 2 == 0 and d_model % head_dim == 0
        self.N, self.R, self.P = d_state, rank, head_dim
        self.H = H = d_model // head_dim
        rms = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm
        self.norm = rms(d_model)
        self.in_proj = nn.Linear(d_model, d_model * rank)
        self.k_proj = nn.Linear(d_model, H * d_state * rank)
        self.q_proj = nn.Linear(d_model, H * d_state * rank)
        self.v_proj = nn.Linear(d_model, d_model * rank)
        self.step_proj = nn.Linear(d_model, H)
        self.rot_proj = nn.Linear(d_model, H * (d_state // 2))
        self.mix_proj = nn.Linear(d_model, H)
        self.out_proj = nn.Linear(d_model * rank, d_model)
        self.k_norm = rms(d_state)
        self.q_norm = rms(d_state)
        self.k_bias = nn.Parameter(torch.ones(H, d_state, rank))
        self.q_bias = nn.Parameter(torch.ones(H, d_state, rank))
        self.decay_log = nn.Parameter(torch.zeros(H))

    def coeffs(self, nu: torch.Tensor):
        """Per-step decay a (B,T,H), write weights b (previous step) and c (current step) (B,T,H), keys k
        (B,T,H,N,R), queries q (B,T,H,N,R), values v (B,T,H,P,R) and rotation angles (B,T,H,N/2)."""
        Bsz, T, _ = nu.shape
        H, N, R, P = self.H, self.N, self.R, self.P
        step = F.softplus(self.step_proj(nu))
        a = torch.exp(step * -torch.exp(self.decay_log))
        m = torch.sigmoid(self.mix_proj(nu))
        c = m * step
        b = (1.0 - m) * step * a
        k = self.k_norm(self.k_proj(nu).view(Bsz, T, H, R, N)).permute(0, 1, 2, 4, 3) + self.k_bias
        q = self.q_norm(self.q_proj(nu).view(Bsz, T, H, R, N)).permute(0, 1, 2, 4, 3) + self.q_bias
        v = self.v_proj(nu).view(Bsz, T, H, P, R)
        ang = self.rot_proj(nu).view(Bsz, T, H, N // 2) * step.unsqueeze(-1)
        return a, b, c, k, q, v, ang

    def gate_and_proj(self, y: torch.Tensor, nu: torch.Tensor):
        """y (..., d_model*R) -> out_proj(y * SiLU(in_proj(nu)))."""
        return self.out_proj(y * F.silu(self.in_proj(nu)))


def rotate_by(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Rotate (even, odd) pairs of the second-to-last axis. v (..., N, X), ang (..., N/2)."""
    a, b = v[..., 0::2, :], v[..., 1::2, :]
    cs, sn = torch.cos(ang).unsqueeze(-1), torch.sin(ang).unsqueeze(-1)
    return torch.stack((cs * a - sn * b, sn * a + cs * b), dim=-2).flatten(-3, -2)


def mix_inputs(layer: SeqMixLayer, u: torch.Tensor):
    """The layer's coefficients in the rotated frame: decay a (B,T,H), keys (B,T,H,N,2R) and values (B,T,H,P,2R)
    of the current and the previous step, queries (B,T,H,N,R)."""
    a, b, c, k, q, v, ang = layer.coeffs(layer.norm(u))
    phi = torch.cumsum(ang, dim=1)
    phi_prev = torch.cat([torch.zeros_like(phi[:, :1]), phi[:, :-1]], dim=1)
    k_prev = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)
    v_prev = torch.cat([torch.zeros_like(v[:, :1]), v[:, :-1]], dim=1)
    keys = torch.cat([c[..., None, None] * rotate_by(k, -phi), b[..., None, None] * rotate_by(k_prev, -phi_prev)], -1)
    return a, keys, torch.cat([v, v_prev], dim=-1), rotate_by(q, -phi)


def chunked_mix(a, keys, vals, qs, chunk: int = 64):
    """y_t = q_t^T sum_{s<=t} (prod_{s<r<=t} a_r) keys_s vals_s^T, chunk by chunk.
    a (B,T,H), keys (B,T,H,N,K), vals (B,T,H,P,K), qs (B,T,H,N,R) -> y (B,T,H,R,P)."""
    B, T, H, N, K = keys.shape
    P, R = vals.shape[-2], qs.shape[-1]
    pad = (-T) % chunk
    if pad:  # a = 1 and zero keys / values / queries: padded steps change nothing
        a = torch.cat([a, a.new_ones(B, pad, H)], 1)
        keys, vals, qs = (torch.cat([t, t.new_zeros(B, pad, *t.shape[2:])], 1) for t in (keys, vals, qs))
    C, L = (T + pad) // chunk, chunk
    la = torch.log(a.clamp_min(1e-30)).view(B, C, L, H)
    kc, vc, qc = keys.view(B, C, L, H, N, K), vals.view(B, C, L, H, P, K), qs.view(B, C, L, H, N, R)
    cl = torch.cumsum(la, 2)
    # exponents are masked to -inf before exp: above the diagonal they are positive and can overflow
    tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=keys.device)).view(1, 1, L, L, 1)
    dec = torch.exp((cl.unsqueeze(3) - cl.unsqueeze(2)).masked_fill(~tril, float("-inf")))
    G = torch.einsum('bclhnr,bcmhnk->bchlmrk', qc, kc)
    y = torch.einsum('bclmh,bchlmrk,bcmhpk->bclhrp', dec, G, vc)
    U = torch.einsum('bclh,bclhnk,bclhpk->bchnp', torch.exp(cl[:, :, -1:] - cl), kc, vc)
    tot = cl[:, :, -1]
    ct = torch.cumsum(tot, 1)
    stril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=keys.device), -1).view(1, C, C, 1)
    Dc = torch.exp(((ct - tot).unsqueeze(2) - ct.unsqueeze(1)).masked_fill(~stril, float("-inf")))
    s0 = torch.einsum('bcjh,bjhnp->bchnp', Dc, U)
    y = y + torch.einsum('bclh,bclhnr,bchnp->bclhrp', torch.exp(cl), qc, s0)
    return y.reshape(B, C * L, H, R, P)[:, :T]


def mix_forward(layer: SeqMixLayer, u: torch.Tensor, chunk: int = 64):
    """The layer's output for u (B,T,d_model)."""
    Bsz, T, _ = u.shape
    nu = layer.norm(u)
    y = chunked_mix(*mix_inputs(layer, u), chunk)
    return layer.gate_and_proj(y.reshape(Bsz, T, layer.H * layer.R * layer.P), nu)
