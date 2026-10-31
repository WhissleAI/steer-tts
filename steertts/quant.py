"""Quantized weights: the packed int8 files in `int8/` and the matching in-place int8 rounding of the external models.

Weight-only, round-to-nearest, symmetric. >=2-D parameters with >= 1024 elements are stored as int8 with one fp16 scale
per output row; every other float parameter as fp16; buffers unchanged. Weights are dequantized at load and compute
stays fp32, so the saving is download and disk size, not speed. The layout is recorded in each file's safetensors
metadata (`steertts_quant`). Vocos, WavLM-base-plus-sv and all-MiniLM-L6-v2 are downloaded from their own repos and
rounded the same way in place, which reproduces the evaluated int8 model exactly.
"""
import json

import numpy as np
import torch

MIN_NUMEL = 1024


def unpack(tensors, layout):
    sd = {}
    for k, L in layout.items():
        if L["s"] == "raw":
            sd[k] = tensors[k]
        elif L["s"] == "fp16":
            sd[k] = tensors[k].float()
        elif L["s"] == "int8":
            sd[k] = (tensors[k + "::q"].float() * tensors[k + "::s"].float()).reshape(L["shape"])
        elif L["s"] == "int4":
            shape, g, Kp = L["shape"], L["g"], L["Kp"]
            O, K = shape[0], int(np.prod(shape[1:]))
            b = tensors[k + "::q"]
            q = torch.stack([b & 15, b >> 4], 1).reshape(-1)[: O * Kp].float() - 8
            w = q.reshape(O, Kp // g, g) * tensors[k + "::s"].float()[..., None]
            sd[k] = w.reshape(O, Kp)[:, :K].reshape(shape)
        else:
            raise ValueError(f"unknown storage {L['s']!r} for {k}")
    return sd


def load_packed(path, device="cpu"):
    """A packed safetensors file -> fp32 state dict."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), "pt") as f:
        meta = json.loads(f.metadata()["steertts_quant"])
    return {k: v.to(device) for k, v in unpack(load_file(str(path)), meta["layout"]).items()}


@torch.no_grad()
def round_int8_(module):
    """Round a module's weights in place exactly as the packed int8 files were made (int8 per output row, fp16 rest)."""
    for _, p in module.named_parameters():
        if not p.is_floating_point():
            continue
        if p.dim() >= 2 and p.numel() >= MIN_NUMEL:
            w = p.float().reshape(p.shape[0], -1)
            s = (w.abs().amax(1, keepdim=True) / 127).clamp(min=1e-12).half().float()
            q = torch.nan_to_num((w / s).round().clamp(-127, 127))  # all-zero rows (scale 0) store 0, as in the files
            p.copy_((q * s).reshape(p.shape))
        else:
            p.copy_(p.half().float())
    return module
