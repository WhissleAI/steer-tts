"""Export the zero-shot cloning speaker encoder (microsoft/wavlm-base-plus-sv, WavLMForXVector) for the C runtime.

    python c/tools/export_wavlm.py <out_dir>

writes <out_dir>/wavlm_sv.safetensors (fp32, original Hugging Face names, only what `.embeddings` needs: no
classifier / AM-softmax / masked_spec_embed; the weight-normed positional conv is folded into a plain
wavlm.encoder.pos_conv_embed.conv.weight) and <out_dir>/int8/wavlm_sv.safetensors (packed int8 layout, see
st_pack.save_int8). The int8 file reproduces the Python int8 build, steertts.quant.round_int8_(WavLMForXVector):
round_int8_ walks named_parameters, which for the weight-normed conv are the parametrization's g (original0, fp16) and
v (original1, int8); the conv weight torch then uses is g * v / ||v||, so that weight is recomputed from the ROUNDED g
and v and stored as a raw fp32 tensor. Metadata "steertts_wavlm" holds the architecture numbers the C code checks.
Uses the Hugging Face cache (no network once the model has been downloaded).
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st_pack import save_int8, save_plain  # noqa: E402

REPO = "microsoft/wavlm-base-plus-sv"
POS_W = "wavlm.encoder.pos_conv_embed.conv.weight"
SKIP = ("classifier.", "objective.", "wavlm.masked_spec_embed")


def forward_params(m):
    """name -> tensor of everything the x-vector forward reads, weight norm folded."""
    out = {}
    for k, p in m.named_parameters():
        if k.startswith(SKIP) or ".parametrizations." in k:
            continue
        out[k] = p.detach().clone()
    out[POS_W] = m.wavlm.encoder.pos_conv_embed.conv.weight.detach().clone()   # g * v / ||v|| (dim 2)
    return dict(sorted(out.items()))


def arch(m):
    c = m.config
    return {"source": REPO, "steertts_wavlm": json.dumps(dict(
        conv_dim=list(c.conv_dim), conv_kernel=list(c.conv_kernel), conv_stride=list(c.conv_stride),
        feat_extract_norm=c.feat_extract_norm, conv_bias=bool(c.conv_bias), hidden=c.hidden_size,
        heads=c.num_attention_heads, layers=c.num_hidden_layers, inter=c.intermediate_size,
        pos_k=c.num_conv_pos_embeddings, pos_groups=c.num_conv_pos_embedding_groups, num_buckets=c.num_buckets,
        max_distance=c.max_bucket_distance, eps=c.layer_norm_eps, stable_ln=bool(c.do_stable_layer_norm),
        weighted_layer_sum=bool(c.use_weighted_layer_sum), tdnn_dim=list(c.tdnn_dim), tdnn_kernel=list(c.tdnn_kernel),
        tdnn_dilation=list(c.tdnn_dilation), xvector_dim=c.xvector_output_dim))}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("out")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    from transformers import WavLMForXVector
    from steertts.quant import round_int8_
    m = WavLMForXVector.from_pretrained(a.repo).eval()
    c = m.config
    assert c.feat_extract_norm == "group" and not c.conv_bias and not c.do_stable_layer_norm and c.use_weighted_layer_sum
    assert c.hidden_act == "gelu" and c.feat_extract_activation == "gelu" and not c.add_adapter
    meta = arch(m)
    out = Path(a.out)
    (out / "int8").mkdir(parents=True, exist_ok=True)
    fp = forward_params(m)
    save_plain(fp, out / "wavlm_sv.safetensors", metadata=meta)

    mq = round_int8_(copy.deepcopy(m))
    q = forward_params(mq)
    pos_w = q.pop(POS_W)
    path8 = out / "int8" / "wavlm_sv.safetensors"
    # dequantized, the packed tensors equal the rounded module's parameters (save_int8 re-applies the same rounding,
    # which is idempotent on already-rounded values); check that, then add the folded conv weight as "raw" fp32
    save_int8({k: m.state_dict()[k] if k in m.state_dict() else v for k, v in q.items()}, path8, metadata=meta)
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file
    with safe_open(str(path8), "np") as f:
        md = dict(f.metadata())
    tensors = load_file(str(path8))
    quant = json.loads(md["steertts_quant"])
    for k, v in q.items():
        L = quant["layout"][k]
        if L["s"] == "int8":
            d = torch.from_numpy(tensors[k + "::q"]).float() * torch.from_numpy(tensors[k + "::s"]).float().reshape(-1, 1)
        else:
            d = torch.from_numpy(tensors[k]).float()
        assert torch.equal(d.reshape(v.shape), v.float()), k
    tensors[POS_W] = pos_w.float().contiguous().numpy()
    quant["layout"][POS_W] = {"s": "raw", "shape": list(pos_w.shape)}
    md["steertts_quant"] = json.dumps(quant)
    save_file(dict(sorted(tensors.items())), str(path8), metadata=md)
    print(f"wrote {out / 'wavlm_sv.safetensors'} and {path8} ({len(fp)} tensors)")


if __name__ == "__main__":
    main()
