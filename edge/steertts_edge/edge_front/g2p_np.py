"""Tiny OOV G2P: pre-norm transformer encoder-decoder, int8 weights, pure-numpy greedy decoding.

Weights file (g2p.npz): for every float tensor `name` either `name` (float16, small vectors) or `name.q` (int8, per-row)
+ `name.s` (float16 row scales). Architecture = torch nn.Transformer*Layer equivalent (
norm_first=True, relu, batch_first).
"""
import numpy as np

CHARS = ["<pad>", "<unk>", "<bos>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz'-.")
PHONES = ["<pad>", "<unk>", "<s>", "</s>"] + [
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0',
    'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH',
    'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1',
    'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
CFG = dict(d=80, ff=160, heads=4, enc=2, dec=2, max_in=32, max_out=32)
_C2I = {c: i for i, c in enumerate(CHARS)}


def load_weights(path):
    z = np.load(path)
    W = {}
    for k in z.files:
        if k.endswith(".s"):
            continue
        if k.endswith(".q"):
            n = k[:-2]
            W[n] = z[k].astype(np.float32) * z[n + ".s"].astype(np.float32)[:, None]
        else:
            W[k] = z[k].astype(np.float32)
    return W


def _ln(x, w, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b


def _mha(q_in, kv_in, W, pre, h, mask=None):
    d = q_in.shape[-1]
    wi, bi = W[pre + "in_proj_weight"], W[pre + "in_proj_bias"]
    q = q_in @ wi[:d].T + bi[:d]
    k = kv_in @ wi[d:2 * d].T + bi[d:2 * d]
    v = kv_in @ wi[2 * d:].T + bi[2 * d:]
    hd = d // h
    q = q.reshape(-1, h, hd).transpose(1, 0, 2)
    k = k.reshape(-1, h, hd).transpose(1, 0, 2)
    v = v.reshape(-1, h, hd).transpose(1, 0, 2)
    a = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
    if mask is not None:
        a = a + mask
    a = np.exp(a - a.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    o = (a @ v).transpose(1, 0, 2).reshape(-1, d)
    return o @ W[pre + "out_proj.weight"].T + W[pre + "out_proj.bias"]


def _ff(x, W, pre):
    x = np.maximum(x @ W[pre + "linear1.weight"].T + W[pre + "linear1.bias"], 0)
    return x @ W[pre + "linear2.weight"].T + W[pre + "linear2.bias"]


class G2P:
    def __init__(self, path):
        self.W = load_weights(path)

    def encode(self, word):
        W, h = self.W, CFG["heads"]
        ids = [_C2I.get(c, 1) for c in word[:CFG["max_in"] - 1]] + [3]
        x = W["cemb.weight"][ids] + W["epos.weight"][:len(ids)]
        for i in range(CFG["enc"]):
            p = f"enc.{i}."
            y = _ln(x, W[p + "norm1.weight"], W[p + "norm1.bias"])
            x = x + _mha(y, y, W, p + "self_attn.", h)
            x = x + _ff(_ln(x, W[p + "norm2.weight"], W[p + "norm2.bias"]), W, p)
        return _ln(x, W["en.weight"], W["en.bias"])

    def predict(self, word, margin=False):
        """-> list of phones (strings), as g2p_en.G2p.predict. Incremental decoding with cached keys / values.
        margin=True also returns the smallest top-1 minus top-2 logit gap over the steps (build-time robustness)."""
        W, h, d = self.W, CFG["heads"], CFG["d"]
        hd = d // h
        mem = self.encode(word)
        cross, selfkv = [], []
        for i in range(CFG["dec"]):
            p = f"dec.{i}.multihead_attn."
            wi, bi = W[p + "in_proj_weight"], W[p + "in_proj_bias"]
            k = (mem @ wi[d:2 * d].T + bi[d:2 * d]).reshape(-1, h, hd).transpose(1, 0, 2)
            v = (mem @ wi[2 * d:].T + bi[2 * d:]).reshape(-1, h, hd).transpose(1, 0, 2)
            cross.append((k, v))
            selfkv.append(([], []))
        ys, gap = [2], 1e9
        for t in range(CFG["max_out"] - 1):
            x = W["pemb.weight"][ys[-1]] + W["dpos.weight"][t]
            for i in range(CFG["dec"]):
                p = f"dec.{i}."
                y = _ln(x, W[p + "norm1.weight"], W[p + "norm1.bias"])
                wi, bi = W[p + "self_attn.in_proj_weight"], W[p + "self_attn.in_proj_bias"]
                qkv = wi @ y + bi
                ks, vs = selfkv[i]
                ks.append(qkv[d:2 * d].reshape(h, hd))
                vs.append(qkv[2 * d:].reshape(h, hd))
                x = x + _attn1(qkv[:d].reshape(h, hd), np.stack(ks, 1), np.stack(vs, 1), W, p + "self_attn.")
                y = _ln(x, W[p + "norm2.weight"], W[p + "norm2.bias"])
                wi, bi = W[p + "multihead_attn.in_proj_weight"], W[p + "multihead_attn.in_proj_bias"]
                q = (wi[:d] @ y + bi[:d]).reshape(h, hd)
                x = x + _attn1(q, cross[i][0], cross[i][1], W, p + "multihead_attn.")
                x = x + _ff(_ln(x, W[p + "norm3.weight"], W[p + "norm3.bias"]), W, p)
            lo = _ln(x, W["dn.weight"], W["dn.bias"]) @ W["out.weight"].T + W["out.bias"]
            tk = int(lo.argmax())
            if margin:
                top2 = np.partition(lo, -2)[-2:]
                gap = min(gap, float(top2[1] - top2[0]))
            if tk in (0, 3):
                break
            ys.append(tk)
        out = [PHONES[t] for t in ys[1:]]
        return (out, gap) if margin else out


def _attn1(q, k, v, W, pre):
    """one query position: q [h, hd], k / v [h, T, hd]."""
    a = np.einsum("hd,htd->ht", q, k) / np.sqrt(q.shape[1])
    a = np.exp(a - a.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    o = np.einsum("ht,htd->hd", a, v).reshape(-1)
    return W[pre + "out_proj.weight"] @ o + W[pre + "out_proj.bias"]
