"""g2p_en's neural G2P (GRU encoder-decoder, 256 hidden; Park & Kim, Apache-2.0) re-implemented in numpy on
low-bit quantised weights. Same greedy decoding as g2p_en.G2p.predict (20 steps max, </s> = 3).

Weight file (gru.npz): per matrix `name.q` (uint8 codebook indices, two 4-bit codes packed per byte when bits <= 4)
+ `name.cb` (float16 per-row codebook [rows, 2**bits]) + `name.shape`; vectors stored float16.
"""
import numpy as np

GRAPHEMES = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
PHONEMES = ["<pad>", "<unk>", "<s>", "</s>"] + [
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0',
    'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH',
    'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1',
    'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
_G2I = {g: i for i, g in enumerate(GRAPHEMES)}


def quantize(W, bits, iters=12):
    """Per-row k-means (1-D, quantile init) codebook quantisation -> (codes uint8 [r, c], codebook float32 [r, K])."""
    K = 2 ** bits
    W = np.asarray(W, np.float32)
    qs = (np.arange(K) + 0.5) / K
    cb = np.quantile(W, qs, axis=1).T.copy()  # [r, K]
    for _ in range(iters):
        codes = np.abs(W[:, :, None] - cb[:, None, :]).argmin(-1)
        for k in range(K):
            m = codes == k
            cnt = m.sum(1)
            s = (W * m).sum(1)
            upd = cnt > 0
            cb[upd, k] = s[upd] / cnt[upd]
    codes = np.abs(W[:, :, None] - cb[:, None, :]).argmin(-1).astype(np.uint8)
    return codes, cb


def pack(codes, bits):
    if bits > 4:
        return codes
    flat = codes.reshape(-1)
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, np.uint8)])
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)


def unpack(q, shape, bits):
    if bits > 4:
        return q.reshape(shape)
    lo, hi = q & 15, q >> 4
    flat = np.empty(lo.size * 2, np.uint8)
    flat[0::2], flat[1::2] = lo, hi
    return flat[:int(np.prod(shape))].reshape(shape)


def save(variables, path, bits, uniform=True):
    """bits=16: float16; uniform=True: symmetric per-row int (fp16 row scales); else per-row k-means codebook."""
    out = {}
    for k, v in variables.items():
        v = np.asarray(v, np.float32)
        if v.ndim == 2 and bits < 16 and uniform:
            lv = 2 ** (bits - 1) - 1
            s = np.abs(v).max(1, keepdims=True) / lv
            s[s == 0] = 1e-8
            out[k + ".u"] = np.round(v / s).astype(np.int8)
            out[k + ".s"] = s[:, 0].astype(np.float16)
        elif v.ndim == 2 and bits < 16:
            codes, cb = quantize(v, bits)
            out[k + ".q"] = pack(codes, bits)
            out[k + ".cb"] = cb.astype(np.float16)
            out[k + ".shape"] = np.array(v.shape, np.int32)
        else:
            out[k] = v.astype(np.float16)
    out["bits"] = np.array([bits], np.int32)
    np.savez_compressed(path, **out)


def load(path):
    z = np.load(path)
    bits = int(z["bits"][0])
    W = {}
    for k in z.files:
        if k.endswith(".u"):
            n = k[:-2]
            W[n] = z[k].astype(np.float32) * z[n + ".s"].astype(np.float32)[:, None]
        elif k.endswith(".q"):
            n = k[:-2]
            shape = tuple(z[n + ".shape"])
            codes = unpack(z[k], shape, bits)
            cb = z[n + ".cb"].astype(np.float32)
            W[n] = np.take_along_axis(cb, codes.astype(np.int64), 1)
        elif not (k.endswith(".cb") or k.endswith(".shape") or k.endswith(".s") or k == "bits"):
            W[k] = z[k].astype(np.float32)
    return W


def _sig(x):
    return 1 / (1 + np.exp(-x))


class GruG2P:
    def __init__(self, path_or_vars):
        W = load(path_or_vars) if isinstance(path_or_vars, (str, bytes)) or hasattr(path_or_vars, "__fspath__") \
            else {k: np.asarray(v, np.float32) for k, v in path_or_vars.items()}
        self.W = W
        # pre-transpose and fold the input-side projections of the embeddings (exact up to float rounding)
        self.enc_x = W["enc_emb"] @ W["enc_w_ih"].T + W["enc_b_ih"]   # [29, 3H]
        self.dec_x = W["dec_emb"] @ W["dec_w_ih"].T + W["dec_b_ih"]   # [74, 3H]
        self.enc_hh, self.dec_hh = W["enc_w_hh"].T.copy(), W["dec_w_hh"].T.copy()
        self.fc = W["fc_w"].T.copy()

    @staticmethod
    def _cell(xp, h, whh, bhh):
        hp = h @ whh + bhh
        H = h.shape[-1]
        r = _sig(xp[:H] + hp[:H])
        z = _sig(xp[H:2 * H] + hp[H:2 * H])
        n = np.tanh(xp[2 * H:] + r * hp[2 * H:])
        return (1 - z) * n + z * h

    def predict(self, word, margin=False):
        W = self.W
        ids = [_G2I.get(c, 1) for c in word] + [2]
        h = np.zeros(self.enc_hh.shape[0], np.float32)
        for i in ids:
            h = self._cell(self.enc_x[i], h, self.enc_hh, W["enc_b_hh"])
        prev, preds, gap = 2, [], 1e9
        for _ in range(20):
            h = self._cell(self.dec_x[prev], h, self.dec_hh, W["dec_b_hh"])
            lo = h @ self.fc + W["fc_b"]
            p = int(lo.argmax())
            if margin:
                t2 = np.partition(lo, -2)[-2:]
                gap = min(gap, float(t2[1] - t2[0]))
            if p == 3:
                break
            preds.append(p)
            prev = p
        out = [PHONEMES[i] if i < len(PHONEMES) else "<unk>" for i in preds]
        return (out, gap) if margin else out
