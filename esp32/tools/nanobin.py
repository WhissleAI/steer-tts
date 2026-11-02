"""nano.bin writer / reader for SteerTTS Nano (format version 1, see docs/NANO_BIN.md).

    from nanobin import V0_CONFIG, expected_tensors, quantize_multiplier, write_nano_bin, read_nano_bin

An exporter builds a dict {name: numpy array} with exactly the tensors listed by expected_tensors(cfg) (dtypes and
shapes are checked) and calls write_nano_bin(path, cfg, fcfg, tensors). quantize_multiplier() turns a real
requantization scale into the (multiplier, shift) pair the int8 kernels use, so an exporter only has to supply float
scales: requant_params() builds the .m / .s tensors of a quantized layer from its input, weight and output scales.
"""
from __future__ import annotations

import math
import struct

import numpy as np

MAGIC = b"STNB"
VERSION = 1
HEADER_BYTES = 256
ENTRY_BYTES = 64
NAME_LEN = 40
ALIGN = 16

I8, I32, F32 = 1, 2, 3
_DT = {I8: np.int8, I32: np.int32, F32: np.float32}
_DT_CODE = {np.dtype(np.int8): I8, np.dtype(np.int32): I32, np.dtype(np.float32): F32}

ACT_RELU, ACT_GELU = 0, 1

# header.cfg[] order (int32), == enum nano_cfg in components/steertts_nano/include/nano_bin.h
CFG_KEYS = [
    "sample_rate", "n_fft", "hop", "win", "n_mels", "n_symbols", "d_cond", "n_voices", "n_ctrl", "n_tags",
    "enc_layers", "enc_dim", "enc_ff", "enc_k",
    "pred_layers", "pred_dim", "pred_k", "pe_k",
    "dec_layers", "dec_dim", "dec_ff", "dec_k",
    "voc_embed_k", "voc_layers", "voc_dim", "voc_ff", "voc_k",
    "chunk", "max_phonemes", "enc_act", "dec_act", "voc_act", "max_dur", "cond_act", "first_chunk",
]
CFG_COUNT = 48
# header.fcfg[] order (float32), == enum nano_fcfg
FCFG_KEYS = ["dur_offset", "mag_max", "ws_pitch_div", "ws_energy_div", "ln_eps"]
FCFG_COUNT = 8

# Architecture v0 sizes.
V0_CONFIG = dict(
    sample_rate=16000, n_fft=1024, hop=256, win=1024, n_mels=80, n_symbols=118, d_cond=128, n_voices=16, n_ctrl=24,
    n_tags=51,
    enc_layers=4, enc_dim=192, enc_ff=384, enc_k=5,
    pred_layers=2, pred_dim=192, pred_k=3, pe_k=3,
    dec_layers=4, dec_dim=160, dec_ff=320, dec_k=5,
    voc_embed_k=7, voc_layers=4, voc_dim=128, voc_ff=384, voc_k=7,
    chunk=32, max_phonemes=200, enc_act=ACT_RELU, dec_act=ACT_RELU, voc_act=ACT_RELU, max_dur=255, cond_act=1,
    first_chunk=16,
)
V0_FCONFIG = dict(dur_offset=1.0, mag_max=100.0, ws_pitch_div=3.0, ws_energy_div=10.0, ln_eps=1e-5)

PREDICTORS = ("dur", "f0", "en")


def _qlayer(name, out, k, inp, n_sc=2, depthwise=False):
    w = (k, out) if depthwise else ((out, inp) if k == 1 else (out, k, inp))
    return [
        (name + ".w", I8, w),
        (name + ".b", I32, (out,)),
        (name + ".m", I32, (out,)),
        (name + ".s", I32, (out,)),
        (name + ".ws", F32, (out,)),
        (name + ".sc", F32, (n_sc,)),
    ]


def _block(p, d, ff, k, d_cond, adaptive):
    t = _qlayer(p + ".dw", d, k, d, depthwise=True)
    if adaptive:
        t += [(p + ".ada.w", I8, (2 * d, d_cond)), (p + ".ada.ws", F32, (2 * d,)), (p + ".ada.b", F32, (2 * d,))]
    else:
        t += [(p + ".ln.g", F32, (d,)), (p + ".ln.b", F32, (d,))]
    t += _qlayer(p + ".pw1", ff, 1, d) + _qlayer(p + ".pw2", d, 1, ff) + [(p + ".res", F32, (1,))]
    return t


def expected_tensors(cfg):
    """[(name, dtype code, shape)] in file order for a configuration."""
    c = cfg
    t = [
        ("cond.voice", F32, (c["n_voices"], c["d_cond"])),
        ("cond.tag", F32, (c["n_tags"], c["d_cond"])),
        ("cond.ctrl.w", F32, (c["d_cond"], c["n_ctrl"])),
        ("cond.ctrl.b", F32, (c["d_cond"],)),
        ("enc.emb", I8, (c["n_symbols"], c["enc_dim"])),
    ]
    for i in range(c["enc_layers"]):
        t += _block(f"enc.{i}", c["enc_dim"], c["enc_ff"], c["enc_k"], c["d_cond"], True)
    for p in PREDICTORS:
        for j in range(c["pred_layers"]):
            inp = c["enc_dim"] if j == 0 else c["pred_dim"]
            t += _qlayer(f"pred.{p}.c{j}", c["pred_dim"], c["pred_k"], inp)
            t += [(f"pred.{p}.c{j}.ln.g", F32, (c["pred_dim"],)), (f"pred.{p}.c{j}.ln.b", F32, (c["pred_dim"],))]
        t += [(f"pred.{p}.out.w", F32, (c["pred_dim"],)), (f"pred.{p}.out.b", F32, (1,))]
    for p in ("f0", "en"):
        t += [(f"pe.{p}.w", F32, (c["pe_k"], c["enc_dim"])), (f"pe.{p}.b", F32, (c["enc_dim"],))]
    t += _qlayer("dec.in", c["dec_dim"], 1, c["enc_dim"])
    for i in range(c["dec_layers"]):
        t += _block(f"dec.{i}", c["dec_dim"], c["dec_ff"], c["dec_k"], c["d_cond"], True)
    t += _qlayer("dec.mel", c["n_mels"], 1, c["dec_dim"])
    t += _qlayer("voc.embed", c["voc_dim"], c["voc_embed_k"], c["n_mels"])
    t += [("voc.ln0.g", F32, (c["voc_dim"],)), ("voc.ln0.b", F32, (c["voc_dim"],))]
    for i in range(c["voc_layers"]):
        t += _block(f"voc.{i}", c["voc_dim"], c["voc_ff"], c["voc_k"], c["d_cond"], False)
    t += [("voc.lnf.g", F32, (c["voc_dim"],)), ("voc.lnf.b", F32, (c["voc_dim"],))]
    t += _qlayer("voc.head", c["n_fft"] + 2, 1, c["voc_dim"], n_sc=3)
    return t


def quantize_multiplier(real):
    """TFLite QuantizeMultiplier: real ~= mult * 2^(shift - 31), mult in [2^30, 2^31). Returns (mult, shift)."""
    real = float(real)
    if real == 0.0:
        return 0, 0
    q, e = math.frexp(real)
    qf = int(round(q * (1 << 31)))
    if qf == (1 << 31):
        qf //= 2
        e += 1
    if e < -31:
        return 0, 0
    if e > 30:
        raise ValueError(f"requantization scale {real} too large")
    return qf, e


def requant_params(s_in, w_scale, s_out):
    """Per-output-channel (mult, shift) int32 arrays for real scale s_in * w_scale[c] / s_out[c]."""
    w_scale = np.asarray(w_scale, np.float64)
    s_out = np.broadcast_to(np.asarray(s_out, np.float64), w_scale.shape)
    ms = [quantize_multiplier(s_in * ws / so) for ws, so in zip(w_scale, s_out)]
    return np.array([m for m, _ in ms], np.int32), np.array([s for _, s in ms], np.int32)


def write_nano_bin(path, cfg, fcfg, tensors):
    exp = expected_tensors(cfg)
    names = [n for n, _, _ in exp]
    extra = set(tensors) - set(names)
    if extra:
        raise ValueError(f"unexpected tensors: {sorted(extra)[:5]}")
    n = len(exp)
    dir_off = HEADER_BYTES
    data_off = (dir_off + n * ENTRY_BYTES + ALIGN - 1) // ALIGN * ALIGN
    entries, blobs, off = [], [], data_off
    for name, dt, shape in exp:
        if name not in tensors:
            raise ValueError(f"missing tensor {name}")
        a = np.ascontiguousarray(tensors[name])
        if a.dtype != np.dtype(_DT[dt]):
            raise ValueError(f"{name}: dtype {a.dtype}, expected {np.dtype(_DT[dt])}")
        if tuple(a.shape) != tuple(shape):
            raise ValueError(f"{name}: shape {a.shape}, expected {shape}")
        if len(name) >= NAME_LEN:
            raise ValueError(f"{name}: name too long")
        b = a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()
        sh = list(shape) + [1] * (3 - len(shape))
        entries.append(struct.pack("<40sHH3III", name.encode(), dt, len(shape), *sh, off, len(b)))
        pad = (-len(b)) % ALIGN
        blobs.append(b + b"\0" * pad)
        off += len(b) + pad
    cfgv = [int(cfg[k]) for k in CFG_KEYS] + [0] * (CFG_COUNT - len(CFG_KEYS))
    fv = [float(fcfg[k]) for k in FCFG_KEYS] + [0.0] * (FCFG_COUNT - len(FCFG_KEYS))
    hdr = MAGIC + struct.pack("<7I", VERSION, HEADER_BYTES, n, dir_off, data_off, off, 0)
    hdr += struct.pack(f"<{CFG_COUNT}i", *cfgv) + struct.pack(f"<{FCFG_COUNT}f", *fv)
    assert len(hdr) == HEADER_BYTES
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(b"".join(entries))
        f.write(b"\0" * (data_off - dir_off - n * ENTRY_BYTES))
        f.write(b"".join(blobs))
    return off


def read_nano_bin(path_or_bytes):
    """-> (cfg dict, fcfg dict, {name: numpy array}) (arrays are views into the file bytes)."""
    buf = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
    if buf[:4] != MAGIC:
        raise ValueError("not a nano.bin file")
    ver, hb, n, dir_off, data_off, fb, _ = struct.unpack_from("<7I", buf, 4)
    if ver != VERSION or hb != HEADER_BYTES or fb > len(buf):
        raise ValueError("unsupported or truncated nano.bin")
    cfgv = struct.unpack_from(f"<{CFG_COUNT}i", buf, 32)
    fv = struct.unpack_from(f"<{FCFG_COUNT}f", buf, 32 + 4 * CFG_COUNT)
    cfg = dict(zip(CFG_KEYS, cfgv))
    fcfg = dict(zip(FCFG_KEYS, fv))
    tens = {}
    for i in range(n):
        name, dt, nd, s0, s1, s2, off, nb = struct.unpack_from("<40sHH3III", buf, dir_off + i * ENTRY_BYTES)
        name = name.rstrip(b"\0").decode()
        shape = (s0, s1, s2)[:nd]
        tens[name] = np.frombuffer(buf, dtype=np.dtype(_DT[dt]).newbyteorder("<"), count=nb // np.dtype(_DT[dt]).itemsize,
                                   offset=off).reshape(shape)
    return cfg, fcfg, tens


def param_count(cfg):
    n = {"int8": 0, "other": 0}
    for name, dt, shape in expected_tensors(cfg):
        k = "int8" if dt == I8 else "other"
        n[k] += int(np.prod(shape))
    return n
