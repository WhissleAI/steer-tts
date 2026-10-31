"""Export the steer-tts-edge acoustic model + vocoder to ONNX (fp32 graphs, weights dequantized from the package).

    python export_onnx.py <pkg_dir> <out_dir>

Graphs (batch 1, dynamic lengths):
  enc.onnx   ids[1,N] i64, spk[1,64], tag[1] i64, ctrl[1,12], cmask[1,12] -> g[1,128], h[1,N,192], logdur/pitch/energy[1,N]
  dec.onnx   h[1,N,192], pitch[1,N], energy[1,N], tok[1,T] i64 (frame -> phone index), g[1,128] -> mel[1,100,T]
  voc.onnx   mel[1,100,T] -> re[1,513,T], im[1,513,T]   (iSTFT runs in numpy)
  spkproj    spk_proj weights (512 -> 64) go to spkproj.safetensors (host-computed clone vectors)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PKG, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(PKG))
OUT.mkdir(parents=True, exist_ok=True)

from steertts_edge import EdgeTTS  # noqa: E402
from steertts_edge import seqmix as _seqmix  # noqa: E402
import steertts_edge.model as _model  # noqa: E402


from mix_mm import mix_matmul  # noqa: E402

_seqmix.chunked_mix = mix_matmul   # mix_forward looks the name up at call time



CHUNK = 64   # sequence-mixing chunk; every sequence entering a graph is zero-padded to a multiple of it


def _biseqmix(self, x, mask):
    """BiSeqMix.forward without autocast (length already a multiple of CHUNK, so the scan adds no padding)."""
    T = x.shape[1]
    t = torch.arange(T, device=x.device)[None]
    lens = mask.sum(1, keepdim=True)
    rev = torch.where(t < lens, lens - 1 - t, t)[..., None].expand(-1, -1, x.shape[2])
    x = x * mask[..., None]
    y = _seqmix.mix_forward(self.fwd, x, self.chunk)
    y = y + torch.gather(_seqmix.mix_forward(self.bwd, torch.gather(x, 1, rev), self.chunk), 1, rev)
    return y * mask[..., None]


def _attn(att, h, mask):
    """nn.MultiheadAttention (batch_first, self-attention, eval) with the key-padding mask as an additive
    [B,1,1,T] bias: the exported MHA expanded it to a [B,heads,T,T] float tensor in every layer."""
    B, T, d = h.shape
    H = att.num_heads
    q, k, v = F.linear(h, att.in_proj_weight, att.in_proj_bias).chunk(3, -1)
    q, k, v = (t.reshape(B, T, H, d // H).transpose(1, 2) for t in (q, k, v))
    bias = torch.zeros(B, 1, 1, T).masked_fill(~mask[:, None, None, :], float("-inf"))
    a = torch.softmax((q * (d // H) ** -0.5) @ k.transpose(-1, -2) + bias, -1)
    return att.out_proj((a @ v).transpose(1, 2).reshape(B, T, d))


def _block(self, x, mask, g):
    """FFTBlock.forward on a padded sequence. Equal to the unpadded computation on valid positions: attention masks
    padded keys, the sequence-mixing scans are causal in each direction, and the conv input is zeroed at padded positions
    (AdaLN of a zero vector is its shift, not zero), as zero padding is at the end of an unpadded sequence."""
    h = self.n1(x, g)
    x = x + _attn(self.att, h, mask)
    x = x + self.seq(self.n_seq(x, g), mask)
    h = (self.n2(x, g) * mask[..., None]).transpose(1, 2)
    h = self.c2(F.gelu(self.c1(h))).transpose(1, 2)
    return (x + h) * mask[..., None]


def _stack(self, x, mask, g):
    x = (x + self.pos(x.shape[1], x.device)[None]) * mask[..., None]
    for l in self.layers:
        x = l(x, mask, g)
    return self.out_norm(x, g) * mask[..., None]


def _varpred(self, x, mask, g):
    m = mask[..., None]
    h = F.relu(self.c1((x * m).transpose(1, 2))).transpose(1, 2)
    h = self.n1(h, g) * m
    h = F.relu(self.c2(h.transpose(1, 2))).transpose(1, 2)
    h = self.n2(h, g)
    return self.out(h).squeeze(-1) * mask


_model.BiSeqMix.forward = _biseqmix
_model.FFTBlock.forward = _block
_model.Stack.forward = _stack
_model.VariancePredictor.forward = _varpred


def _mask(T, n):
    return torch.arange(T)[None] < n[:, None]


class Enc(nn.Module):
    """ids [1,Np] (zero-padded to a multiple of CHUNK), n [1] = number of real phones."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, n, spk, tag, ctrl, cmask):
        m = self.m
        g = m.cond(torch.cat([spk, m.tag(tag), ctrl * cmask, cmask], -1))
        mask = _mask(ids.shape[1], n)
        h = m.encoder(m.emb(ids), mask, g)
        return g, h, m.dur_pred(h, mask, g), m.pitch_pred(h, mask, g), m.energy_pred(h, mask, g)


class Dec(nn.Module):
    """h/pitch/energy as enc returned them (padded, pitch/energy may be edited), tok [1,Tp] frame -> phone index
    (zero-padded to a multiple of CHUNK), nf [1] = number of real frames."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, h, pitch, energy, tok, nf, g):
        m = self.m
        pe = m.pitch_emb(pitch[:, None]).transpose(1, 2) + m.energy_emb(energy[:, None]).transpose(1, 2)
        x = torch.gather(h + pe, 1, tok[..., None].expand(-1, -1, h.shape[2]))
        mask = _mask(tok.shape[1], nf)
        return m.to_mel(m.decoder(x, mask, g)).transpose(1, 2)


class Voc(nn.Module):
    """Vocoder network + magnitude/phase -> real/imag spectrum; the iSTFT runs outside the graph."""

    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, mel):
        x = self.v.head.out(self.v.backbone(mel)).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        return mag * torch.cos(p), mag * torch.sin(p)


def main():
    tts = EdgeTTS(str(PKG))
    m = tts.model.eval()
    assert all(l.mixer == "hybrid" and l.seq.chunk == CHUNK for st in (m.encoder, m.decoder) for l in st.layers)
    Np, Tp = 3 * CHUNK, 9 * CHUNK
    ids = torch.zeros(1, Np, dtype=torch.long)
    ids[0, :150] = torch.randint(4, 100, (150,))
    n = torch.tensor([150])
    spk, tag = tts._s[:1], torch.tensor([3])
    ctrl, cmask = torch.zeros(1, 12), torch.zeros(1, 12)
    ctrl[0, 0], cmask[0, 0] = 1.0, 1.0
    cN, cT = torch.export.Dim("cN", min=1, max=64), torch.export.Dim("cT", min=1, max=1024)
    dN, dT = CHUNK * cN, CHUNK * cT
    kw = dict(dynamo=True, opset_version=18, external_data=False)
    with torch.no_grad():
        enc = Enc(m).eval()
        torch.onnx.export(enc, (ids, n, spk, tag, ctrl, cmask), OUT / "enc.onnx",
                          input_names=["ids", "n", "spk", "tag", "ctrl", "cmask"],
                          output_names=["g", "h", "logdur", "pitch", "energy"],
                          dynamic_shapes={"ids": {1: dN}, "n": None, "spk": None, "tag": None, "ctrl": None,
                                          "cmask": None}, **kw)
        g, h, _, p, e = enc(ids, n, spk, tag, ctrl, cmask)
        tok = torch.zeros(1, Tp, dtype=torch.long)
        tok[0, :540] = torch.sort(torch.randint(0, 150, (540,)))[0]
        nf = torch.tensor([540])
        torch.onnx.export(Dec(m).eval(), (h, p, e, tok, nf, g), OUT / "dec.onnx",
                          input_names=["h", "pitch", "energy", "tok", "nf", "g"], output_names=["mel"],
                          dynamic_shapes={"h": {1: dN}, "pitch": {1: dN}, "energy": {1: dN}, "tok": {1: dT}, "nf": None,
                                          "g": None}, **kw)
        mel = torch.randn(1, 100, 157)
        torch.onnx.export(Voc(tts.vocoder).eval(), (mel,), OUT / "voc.onnx", input_names=["mel"],
                          output_names=["re", "im"], dynamic_shapes={"mel": {2: torch.export.Dim("T", min=2, max=65536)}},
                          **kw)
    from safetensors.numpy import save_file
    save_file({"w": m.spk_proj.weight.detach().numpy().astype(np.float32), "b": m.spk_proj.bias.detach().numpy().astype(np.float32)},
              str(OUT / "spkproj.safetensors"))
    for f in sorted(OUT.glob("*")):
        print(f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
