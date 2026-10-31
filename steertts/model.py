"""SteerTTS acoustic model (inference path).

text (ARPAbet) -> encoder -> per-phone duration / pitch / energy predictors -> length regulator -> decoder ->
100-band log-mel (Vocos-compatible). Every block is conditioned through adaptive LayerNorm on
g = MLP([speaker vector, style tag, control z-vector * mask, mask]). Encoder and decoder blocks are "hybrid":
multi-head self-attention followed by a bidirectional sequence-mixing branch.

This module holds the inference path only: the released weights load into it unchanged and give the published
mels. There is no alignment module; the released weights do not contain the aligner.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CONTROL_DIMS, N_MELS
from .seqmix import SeqMixLayer, mix_forward
from .text import N_BASE, PAD

N_CTRL = len(CONTROL_DIMS)


def seq_mask(lens, max_len=None):
    max_len = max_len or int(lens.max())
    return torch.arange(max_len, device=lens.device)[None] < lens[:, None]


class SinusoidalPos(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, T, device):
        pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.d, 2, device=device).float() * (-math.log(10000.0) / self.d))
        pe = torch.zeros(T, self.d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe


class AdaLN(nn.Module):
    """LayerNorm whose scale/shift are predicted from the conditioning vector g."""

    def __init__(self, d, g_dim):
        super().__init__()
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.proj = nn.Linear(g_dim, 2 * d)

    def forward(self, x, g):  # x [B,T,d], g [B,g]
        s, b = self.proj(g)[:, None].chunk(2, -1)
        return self.ln(x) * (1 + s) + b


class BiSeqMix(nn.Module):
    """Bidirectional sequence mixer: a forward and a backward SeqMixLayer, summed. The backward pass reverses
    each sequence within its own length, so padding never enters the state. Runs in fp32."""

    def __init__(self, d, d_state=32, head_dim=32, rank=1, chunk=64):
        super().__init__()
        self.fwd = SeqMixLayer(d, d_state=d_state, rank=rank, head_dim=head_dim)
        self.bwd = SeqMixLayer(d, d_state=d_state, rank=rank, head_dim=head_dim)
        self.chunk = chunk

    def forward(self, x, mask):  # x [B,T,d], mask [B,T] True=valid
        T = x.shape[1]
        t = torch.arange(T, device=x.device)[None]
        lens = mask.sum(1, keepdim=True)
        rev = torch.where(t < lens, lens - 1 - t, t)[..., None].expand(-1, -1, x.shape[2])  # an involution
        with torch.autocast(x.device.type, enabled=False):
            x = (x * mask[..., None]).float()
            y = mix_forward(self.fwd, x, self.chunk)
            y = y + torch.gather(mix_forward(self.bwd, torch.gather(x, 1, rev), self.chunk), 1, rev)
        return y * mask[..., None]


class FFTBlock(nn.Module):
    """Pre-norm block: token mixer + conv FFN, all norms adaptive on g.
    mixer: "attn" (self-attention) or "hybrid" (attention, then a bidirectional sequence-mixing branch)."""

    def __init__(self, d, heads, ffn, k, g_dim, p=0.1, mixer="attn", state_dim=32):
        super().__init__()
        assert mixer in ("attn", "hybrid"), mixer
        self.mixer = mixer
        self.n1 = AdaLN(d, g_dim)
        self.att = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        if mixer == "hybrid":
            self.n_seq = AdaLN(d, g_dim)
            self.seq = BiSeqMix(d, d_state=state_dim)
        self.n2 = AdaLN(d, g_dim)
        self.c1 = nn.Conv1d(d, ffn, k, padding=k // 2)
        self.c2 = nn.Conv1d(ffn, d, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x, mask, g):  # mask [B,T] True=valid
        h = self.n1(x, g)
        h, _ = self.att(h, h, h, key_padding_mask=~mask, need_weights=False)
        x = x + self.drop(h)
        if self.mixer == "hybrid":
            x = x + self.drop(self.seq(self.n_seq(x, g), mask))
        h = self.n2(x, g).transpose(1, 2)
        h = self.c2(F.gelu(self.c1(h))).transpose(1, 2)
        x = x + self.drop(h)
        return x * mask[..., None]


class Stack(nn.Module):
    def __init__(self, n, d, heads, ffn, k, g_dim, p, mixer="attn", state_dim=32):
        super().__init__()
        self.pos = SinusoidalPos(d)
        self.layers = nn.ModuleList([FFTBlock(d, heads, ffn, k, g_dim, p, mixer, state_dim) for _ in range(n)])
        self.out_norm = AdaLN(d, g_dim)

    def forward(self, x, mask, g):
        x = x + self.pos(x.shape[1], x.device)[None]
        for l in self.layers:
            x = l(x, mask, g)
        return self.out_norm(x, g) * mask[..., None]


class VariancePredictor(nn.Module):
    def __init__(self, d, g_dim, k=3, p=0.15):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, k, padding=k // 2)
        self.n1 = AdaLN(d, g_dim)
        self.c2 = nn.Conv1d(d, d, k, padding=k // 2)
        self.n2 = AdaLN(d, g_dim)
        self.out = nn.Linear(d, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x, mask, g):
        h = F.relu(self.c1(x.transpose(1, 2))).transpose(1, 2)
        h = self.drop(self.n1(h, g))
        h = F.relu(self.c2(h.transpose(1, 2))).transpose(1, 2)
        h = self.drop(self.n2(h, g))
        return self.out(h).squeeze(-1) * mask


def length_regulate(x, dur, max_len=None):
    """x [B,N,d], dur [B,N] int -> [B,T,d], mel_lens."""
    lens = dur.sum(1)
    T = int(max_len or lens.max())
    idx = torch.cumsum(dur, 1)
    frames = torch.arange(T, device=x.device)[None]
    tok = (frames[:, :, None] >= idx[:, None, :]).sum(-1).clamp(max=x.shape[1] - 1)
    out = torch.gather(x, 1, tok[..., None].expand(-1, -1, x.shape[2]))
    return out * seq_mask(lens, T)[..., None], lens


class SteerTTSModel(nn.Module):
    """`arch` in config.json holds the non-default arguments."""

    def __init__(self, n_speakers, n_tags, d=192, heads=2, ffn=384, k=3, enc_layers=4, dec_layers=4, g_dim=128,
                 spk_dim=64, tag_dim=32, p=0.1, enc_mixer="attn", dec_mixer="attn", state_dim=32, spk_enc_dim=0,
                 n_symbols=None, n_env=0, ctrl_dims=None, no_mask=False, ctrl_norm="spk", level_norm=True):
        super().__init__()
        assert ctrl_dims is None and not no_mask and ctrl_norm == "spk", "unsupported model configuration"
        n_symbols = n_symbols or N_BASE
        self.emb = nn.Embedding(n_symbols, d, padding_idx=PAD)
        self.spk = nn.Embedding(n_speakers + 1, spk_dim)  # speaker table (unused when a speaker vector is given)
        self.spk_proj = nn.Linear(spk_enc_dim, spk_dim) if spk_enc_dim else None  # speaker-verification vector -> slot
        self.tag = nn.Embedding(n_tags, tag_dim)  # index 0 = 'none'
        self.cond = nn.Sequential(nn.Linear(spk_dim + tag_dim + 2 * N_CTRL, 256), nn.SiLU(), nn.Linear(256, g_dim))
        # acoustic-environment projection; present in the weights but not functional
        # in this release, so the pipeline does not expose it
        self.env_proj = nn.Sequential(nn.Linear(n_env, 128), nn.SiLU(), nn.Linear(128, g_dim)) if n_env else None
        self.encoder = Stack(enc_layers, d, heads, ffn, k, g_dim, p, enc_mixer, state_dim)
        self.decoder = Stack(dec_layers, d, heads, ffn, k, g_dim, p, dec_mixer, state_dim)
        self.dur_pred = VariancePredictor(d, g_dim)
        self.pitch_pred = VariancePredictor(d, g_dim)
        self.energy_pred = VariancePredictor(d, g_dim)
        self.pitch_emb = nn.Conv1d(1, d, 3, padding=1)
        self.energy_emb = nn.Conv1d(1, d, 3, padding=1)
        self.to_mel = nn.Linear(d, N_MELS)

    def condition(self, spk, tag, ctrl, cmask, spk_vec=None, env=None):
        """ctrl [B,12] z-values, cmask [B,12] 1 = this dim is requested, 0 = unspecified (the model picks it)."""
        c = torch.nan_to_num(ctrl) * cmask
        s = self.spk_proj(spk_vec) if (self.spk_proj is not None and spk_vec is not None) else self.spk(spk)
        g = self.cond(torch.cat([s, self.tag(tag), c, cmask], -1))
        if self.env_proj is not None and env is not None:
            g = g + self.env_proj(env)
        return g

    def encode(self, ids, text_lens, g):
        tmask = seq_mask(text_lens, ids.shape[1])
        e = self.emb(ids)
        return e, self.encoder(e, tmask, g), tmask

    @torch.no_grad()
    def infer(self, ids, spk, tag, ctrl, cmask, pace=1.0, word_idx=None, word_shift=None, spk_vec=None, env=None,
              guidance=1.0, base=None):
        """ids [1,N]. word_shift: {word_index: dict(pitch=st/PITCH_SCALE, energy=dB/10, dur=factor)}.
        guidance s != 1 with base=(tag0, ctrl0, cmask0): out = out(base) + s * (out(request) - out(base)) on every
        stage (durations first, so both passes share one timeline)."""
        g = self.condition(spk, tag, ctrl, cmask, spk_vec, env)
        g0 = self.condition(spk, *base, spk_vec, env) if (base is not None and guidance != 1.0) else None
        text_lens = torch.tensor([ids.shape[1]], device=ids.device)
        _, h, tmask = self.encode(ids, text_lens, g)
        logdur, pitch, energy = self.dur_pred(h, tmask, g), self.pitch_pred(h, tmask, g), self.energy_pred(h, tmask, g)
        if g0 is not None:
            _, h0, _ = self.encode(ids, text_lens, g0)
            mix = lambda a, a0: a0 + guidance * (a - a0)
            logdur = mix(logdur, self.dur_pred(h0, tmask, g0))
            pitch = mix(pitch, self.pitch_pred(h0, tmask, g0))
            energy = mix(energy, self.energy_pred(h0, tmask, g0))
        dur = (torch.exp(logdur) - 1).clamp_min(0) / pace
        if word_shift and word_idx is not None:
            for w, sh in word_shift.items():
                sel = word_idx[0] == w
                pitch[0, sel] += sh.get("pitch", 0.0)
                energy[0, sel] += sh.get("energy", 0.0)
                dur[0, sel] *= sh.get("dur", 1.0)
        dur = torch.round(dur).long().clamp_min(0)
        dur[:, 0] = dur[:, 0].clamp_min(1)
        pe = self.pitch_emb(pitch[:, None]).transpose(1, 2) + self.energy_emb(energy[:, None]).transpose(1, 2)
        x, lens = length_regulate(h + pe, dur)
        mel = self.to_mel(self.decoder(x, seq_mask(lens, x.shape[1]), g)).transpose(1, 2)
        if g0 is not None:
            x0, _ = length_regulate(h0 + pe, dur)
            mel0 = self.to_mel(self.decoder(x0, seq_mask(lens, x0.shape[1]), g0)).transpose(1, 2)
            mel = mel0 + guidance * (mel - mel0)
        return mel, dict(dur=dur, pitch=pitch, energy=energy, g=g)
