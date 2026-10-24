"""Write weights in the two safetensors layouts the C runtime reads (see src/st_base.h, "weights").

save_plain(tensors, path)    name -> float32 tensor, as is.
save_int8(module, path)      the packed int8 layout of the v1 release (steertts.quant): every floating >=2-D parameter
                             with >= 1024 elements as int8 with one fp16 scale per output row (name::q, name::s), every
                             other floating parameter as fp16, everything else raw. Dequantized, this is exactly what
                             steertts.quant.round_int8_ does to Vocos / WavLM / MiniLM in place, so the C int8 build
                             renders the same model as the Python int8 build.
"""
import json

import numpy as np
import torch
from safetensors.numpy import save_file

MIN_NUMEL = 1024


def save_plain(tensors, path, metadata=None):
    out = {k: np.ascontiguousarray(v.detach().cpu().float().numpy() if torch.is_floating_point(v) else v.detach().cpu().numpy())
           for k, v in tensors.items()}
    save_file(out, str(path), metadata=metadata)


@torch.no_grad()
def save_int8(params, path, metadata=None):
    """params: name -> tensor (the module's parameters, plus any extra tensors the runtime needs)."""
    out, layout = {}, {}
    for k, p in params.items():
        p = p.detach().cpu()
        if not torch.is_floating_point(p):
            out[k] = p.numpy()
            layout[k] = {"s": "raw", "shape": list(p.shape)}
        elif p.dim() >= 2 and p.numel() >= MIN_NUMEL:
            w = p.float().reshape(p.shape[0], -1)
            s = (w.abs().amax(1, keepdim=True) / 127).clamp(min=1e-12).half().float()
            q = torch.nan_to_num((w / s).round().clamp(-127, 127))
            out[k + "::q"] = q.to(torch.int8).numpy()
            out[k + "::s"] = s.half().numpy()   # [O, 1], as the release files
            layout[k] = {"s": "int8", "shape": list(p.shape)}
        else:
            out[k] = p.half().numpy()
            layout[k] = {"s": "fp16", "shape": list(p.shape)}
    md = dict(metadata or {})
    md["steertts_quant"] = json.dumps({"variant": "int8", "layout": layout})
    save_file(out, str(path), metadata=md)
