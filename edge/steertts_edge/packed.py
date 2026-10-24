"""Packed weights: `<name>::q` + `<name>::s` per quantized tensor, layout in the
safetensors metadata key `steertts_quant`. int8: one fp16 scale per output row; int4: groups of g along the
flattened input dim, two values per byte, one fp16 scale per group."""
import json

import numpy as np
import torch


def unpack(tensors, lay):
    sd = {}
    for k, L in lay.items():
        if L["s"] == "raw":
            sd[k] = tensors[k]
        elif L["s"] == "fp16":
            sd[k] = tensors[k].float()
        elif L["s"] == "int8":
            sd[k] = (tensors[k + "::q"].float() * tensors[k + "::s"].float()).reshape(L["shape"])
        else:
            shape, g, Kp = L["shape"], L["g"], L["Kp"]
            O = shape[0]
            K = int(np.prod(shape[1:]))
            b = tensors[k + "::q"]
            q = torch.stack([b & 15, b >> 4], 1).reshape(-1)[: O * Kp].float() - 8
            w = q.reshape(O, Kp // g, g) * tensors[k + "::s"].float()[..., None]
            sd[k] = w.reshape(O, Kp)[:, :K].reshape(shape)
    return sd


def load_packed(path, device="cpu"):
    """-> (state dict of fp32 tensors, metadata dict)."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), "pt") as f:
        meta = json.loads(f.metadata()["steertts_quant"])
    return {k: v.to(device) for k, v in unpack(load_file(str(path)), meta["layout"]).items()}, meta
