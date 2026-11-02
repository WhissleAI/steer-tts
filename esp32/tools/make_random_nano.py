#!/usr/bin/env python3
"""Write a nano.bin with random int8 weights (for benchmarks and runtime tests without released weights).

    python tools/make_random_nano.py -o nano_random.bin [--seed 0] [--set dec_layers=6 --set chunk=48 ...]

Shapes follow the v0 architecture (tools/nanobin.py V0_CONFIG); --set overrides any config value. Scales are chosen
so that activations stay in range (unit-variance pre-activations), so the audio is noise, but every kernel runs on
realistic value distributions.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nanobin as nb  # noqa: E402

ACT = 4.0 / 127.0   # activation scale: int8 covers [-4, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = dict(nb.V0_CONFIG)
    fcfg = dict(nb.V0_FCONFIG)
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k in cfg:
            cfg[k] = int(v)
        elif k in fcfg:
            fcfg[k] = float(v)
        else:
            sys.exit(f"unknown config key {k}")
    rng = np.random.default_rng(a.seed)
    t = {}
    for name, dt, shape in nb.expected_tensors(cfg):
        base, _, field = name.rpartition(".")
        if dt == nb.I8 and field == "w":
            t[name] = rng.integers(-127, 128, size=shape, dtype=np.int16).astype(np.int8)
        elif name == "enc.emb":
            t[name] = rng.integers(-64, 65, size=shape, dtype=np.int16).astype(np.int8)
        elif dt == nb.F32:
            t[name] = None   # filled below
        else:
            t[name] = None
    # quantized layers: weight scale for unit gain, then bias / mult / shift
    for name, dt, shape in nb.expected_tensors(cfg):
        if not name.endswith(".sc"):
            continue
        base = name[:-3]
        w = t[base + ".w"]
        out = w.shape[1] if base.endswith(".dw") else w.shape[0]
        fan_in = w.shape[0] if base.endswith(".dw") else int(np.prod(w.shape[1:]))
        ws = np.full(out, np.sqrt(3.0) / (127.0 * np.sqrt(fan_in)), np.float32) * rng.uniform(0.8, 1.2, out).astype(np.float32)
        if base == "voc.head":
            half = cfg["n_fft"] // 2 + 1
            s_out = np.array([ACT] * half + [np.pi / 127.0] * half, np.float64)
            sc = np.array([ACT, ACT, np.pi / 127.0], np.float32)
            bias_real = np.concatenate([np.full(half, -3.0), np.zeros(half)])
        else:
            s_out = ACT
            sc = np.array([ACT, ACT], np.float32)
            bias_real = rng.normal(0, 0.1, out)
        t[base + ".ws"] = ws
        t[base + ".sc"] = sc
        t[base + ".b"] = np.round(bias_real / (ACT * ws.astype(np.float64))).astype(np.int32)
        t[base + ".m"], t[base + ".s"] = nb.requant_params(ACT, ws, s_out)
    for name, dt, shape in nb.expected_tensors(cfg):
        if t.get(name) is not None:
            continue
        field = name.rsplit(".", 1)[1]
        if name.endswith(".res"):
            t[name] = np.array([ACT], np.float32)
        elif name.endswith(".ada.w"):
            t[name] = rng.integers(-127, 128, size=shape, dtype=np.int16).astype(np.int8)
        elif name.endswith(".ada.ws"):
            t[name] = np.full(shape, 0.1 / (127.0 * np.sqrt(cfg["d_cond"])), np.float32)
        elif name.endswith(".ada.b"):
            t[name] = np.zeros(shape, np.float32)
        elif field == "g":
            t[name] = rng.uniform(0.8, 1.2, shape).astype(np.float32)
        elif field == "b" and ".ln" in name:
            t[name] = rng.normal(0, 0.05, shape).astype(np.float32)
        elif name.startswith("cond."):
            t[name] = rng.normal(0, 0.3, shape).astype(np.float32)
        elif name.startswith("pred.") and name.endswith(".out.w"):
            t[name] = rng.normal(0, 0.02, shape).astype(np.float32)
        elif name == "pred.dur.out.b":
            t[name] = np.array([np.log(6.0)], np.float32)    # ~5 frames per phoneme
        elif name.startswith("pred."):
            t[name] = np.zeros(shape, np.float32)
        elif name.startswith("pe."):
            t[name] = rng.normal(0, 0.05, shape).astype(np.float32)
        else:
            raise SystemExit(f"no rule for {name}")
    n = nb.write_nano_bin(a.out, cfg, fcfg, t)
    pc = nb.param_count(cfg)
    print(f"{a.out}: {n} bytes, {pc['int8'] / 1e6:.3f} M int8 params, {pc['other'] / 1e3:.1f} k other values")


if __name__ == "__main__":
    main()
