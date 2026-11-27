"""Convert SteerTTS weights (safetensors, plain or packed int8 / int4) to GGUF for ggml-based runtimes.

    python c/tools/to_gguf.py <model.safetensors> <out.gguf> [--config config.json] [--f16] [--f32]

The C runtime does not need this (with -DSTEERTTS_GGML=ON it uploads the shipped safetensors to the device itself);
the GGUF is for other ggml programs (audio.cpp / codec.cpp style). Tensor names and logical shapes stay those of
the PyTorch state dict; GGUF stores dims innermost first, so a torch [O, C, k] weight is ne = {k, C, O}.

Storage, all lossless with respect to the source file:
  fp32           -> F32 (or F16 with --f16, which is lossy)
  packed "fp16"  -> F16 (F32 with --f32)
  packed int8    -> Q8_0 when the flattened input size is a multiple of 32 (every 32-block gets its row's fp16
                    scale, the int8 values are copied), else F32 (the exact dequantized values)
  Quantized tensors are stored as the 2-D matrix [O, C*k] the runtimes multiply with (ggml quantizes along the
  innermost dim); their original shape is in steertts.shape.<tensor> (JSON list).
  packed int4    -> Q4_0 when the group size is a multiple of 32 and there is no padding (same nibbles, same fp16
                    scale, repacked to ggml's order), else Q8_0 / F32
Metadata: general.architecture = "steertts", steertts.file (the source name), steertts.config (config.json as
text, when given), plus steertts.quant.<tensor> describing the source storage. numpy only.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGN = 32
T_F32, T_F16, T_Q4_0, T_Q8_0 = 0, 1, 2, 8
V_U32, V_STRING = 4, 8
ST_DT = {"F32": np.float32, "F16": np.float16, "BF16": None, "I8": np.int8, "U8": np.uint8, "I32": np.int32,
         "I64": np.int64, "F64": np.float64}


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        data = f.read()
    meta = head.pop("__metadata__", {}) or {}
    out = {}
    for name, info in head.items():
        a, b = info["data_offsets"]
        if info["dtype"] == "BF16":
            raw = np.frombuffer(data[a:b], dtype=np.uint16).astype(np.uint32) << 16
            arr = raw.view(np.float32)
        else:
            arr = np.frombuffer(data[a:b], dtype=ST_DT[info["dtype"]])
        out[name] = arr.reshape(info["shape"]) if info["shape"] else arr.reshape(())
    return out, meta


def q8_0(q, s):
    """int8 rows q [O, K] with fp16 row scales s [O] -> Q8_0 blocks (K % 32 == 0)."""
    O, K = q.shape
    blk = np.zeros((O, K // 32), dtype=[("d", "<f2"), ("qs", "i1", 32)])
    blk["d"] = np.repeat(s.reshape(O, 1).astype(np.float16), K // 32, axis=1)
    blk["qs"] = q.reshape(O, K // 32, 32)
    return blk.tobytes()


def q4_0(nib, s):
    """nibbles nib [O, K] (0..15, value = (nib - 8) * scale) with fp16 scales s [O, K/32] -> Q4_0 blocks."""
    O, K = nib.shape
    n = nib.reshape(O, K // 32, 32).astype(np.uint8)
    blk = np.zeros((O, K // 32), dtype=[("d", "<f2"), ("qs", "u1", 16)])
    blk["d"] = s.astype(np.float16)
    blk["qs"] = n[:, :, :16] | (n[:, :, 16:] << 4)   # ggml: element j low nibble, element j + 16 high nibble
    return blk.tobytes()


def convert(tensors, meta, f16, f32):
    """-> list of (name, logical shape, ggml type, bytes, note)"""
    layout = json.loads(meta["steertts_quant"])["layout"] if "steertts_quant" in meta else None
    items = []
    if layout is None:
        for name, a in tensors.items():
            a = np.ascontiguousarray(a, dtype=np.float32)
            if f16 and a.ndim >= 2:
                items.append((name, a.shape, T_F16, a.astype(np.float16).tobytes(), "fp32->f16"))
            else:
                items.append((name, a.shape, T_F32, a.tobytes(), "fp32"))
        return items
    for name, L in layout.items():
        s = L["s"]
        if s in ("raw", "fp16"):
            a = tensors[name]
            if s == "fp16" and not f32:
                items.append((name, a.shape, T_F16, np.ascontiguousarray(a, dtype=np.float16).tobytes(), "fp16"))
            else:
                items.append((name, a.shape, T_F32, np.ascontiguousarray(a, dtype=np.float32).tobytes(), s))
            continue
        shape = tuple(L["shape"])
        O, K = shape[0], int(np.prod(shape[1:]))
        q, sc = tensors[name + "::q"], tensors[name + "::s"].astype(np.float32)
        if s == "int8":
            q = q.reshape(O, K)
            if K % 32 == 0:
                items.append((name, (O, K), T_Q8_0, q8_0(q, sc.reshape(O)), "int8->q8_0"))
            else:
                deq = (q.astype(np.float32) * sc.reshape(O, 1)).astype(np.float32)
                items.append((name, shape, T_F32, deq.tobytes(), "int8->f32"))
        elif s == "int4":
            g, Kp = int(L["g"]), int(L.get("Kp", K))
            b = q.reshape(O, Kp // 2)
            nib = np.empty((O, Kp), dtype=np.uint8)
            nib[:, 0::2], nib[:, 1::2] = b & 15, b >> 4   # low nibble first
            sc = sc.reshape(O, Kp // g)
            if g % 32 == 0 and Kp == K:
                items.append((name, (O, K), T_Q4_0, q4_0(nib, np.repeat(sc, g // 32, axis=1)), "int4->q4_0"))
            else:
                deq = ((nib.astype(np.float32) - 8) * np.repeat(sc, g, axis=1))[:, :K]
                items.append((name, shape, T_F32, np.ascontiguousarray(deq, dtype=np.float32).tobytes(), "int4->f32"))
        else:
            raise SystemExit(f"{name}: unknown storage {s!r}")
    return items


def source_shapes(tensors, meta):
    if "steertts_quant" not in meta:
        return {n: a.shape for n, a in tensors.items()}
    out = {}
    for name, L in json.loads(meta["steertts_quant"])["layout"].items():
        out[name] = tuple(L["shape"]) if "shape" in L else tensors[name].shape
    return out


def gguf_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, items, kv):
    head = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(items), len(kv))
    for k, v in kv:
        head += gguf_str(k)
        head += struct.pack("<I", V_STRING) + gguf_str(v) if isinstance(v, str) else struct.pack("<II", V_U32, v)
    offs, off = [], 0
    for _, _, _, data, _ in items:
        offs.append(off)
        off += (len(data) + ALIGN - 1) // ALIGN * ALIGN
    for (name, shape, ty, _, _), o in zip(items, offs):
        ne = list(reversed(shape)) or [1]
        head += gguf_str(name) + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        head += struct.pack("<IQ", ty, o)
    head += b"\0" * ((-len(head)) % ALIGN)
    with open(path, "wb") as f:
        f.write(head)
        for _, _, _, data, _ in items:
            f.write(data)
            f.write(b"\0" * ((-len(data)) % ALIGN))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--config", help="config.json to embed as steertts.config")
    ap.add_argument("--f16", action="store_true", help="fp32 matrices as F16 (lossy)")
    ap.add_argument("--f32", action="store_true", help="packed fp16 tensors as F32")
    a = ap.parse_args()
    tensors, meta = read_safetensors(a.src)
    items = convert(tensors, meta, a.f16, a.f32)
    kv = [("general.architecture", "steertts"), ("general.alignment", ALIGN), ("steertts.file", Path(a.src).name)]
    if a.config:
        kv.append(("steertts.config", Path(a.config).read_text()))
    kv += [(f"steertts.quant.{n}", note) for n, _, _, _, note in items]
    shapes = source_shapes(tensors, meta)
    kv += [(f"steertts.shape.{n}", json.dumps(list(shapes[n]))) for n, sh, _, _, _ in items if tuple(shapes[n]) != tuple(sh)]
    write_gguf(a.out, items, kv)
    counts = {}
    for *_, note in items:
        counts[note] = counts.get(note, 0) + 1
    print(f"{a.out}: {len(items)} tensors ({', '.join(f'{v} {k}' for k, v in sorted(counts.items()))})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
