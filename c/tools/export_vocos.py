"""Export the v1 vocoder (charactr/vocos-mel-24khz) for the C runtime.

    python c/tools/export_vocos.py <out_dir>

writes <out_dir>/vocos.safetensors (fp32, original state-dict names, backbone.* and head.out.* only) and
<out_dir>/int8/vocos.safetensors (packed int8: dequantized it equals steertts.quant.round_int8_(vocos), i.e. the
Python int8 build). Uses the Hugging Face cache (no network once the model has been downloaded).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st_pack import save_int8, save_plain  # noqa: E402

REPO = "charactr/vocos-mel-24khz"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("out")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    from vocos import Vocos
    v = Vocos.from_pretrained(a.repo).eval()
    # parameters only (round_int8_ walks named_parameters); the iSTFT window and the mel front end are not needed
    params = {k: p for k, p in v.named_parameters() if k.startswith(("backbone.", "head.out."))}
    bb, head = v.backbone, v.head
    meta = {"source": a.repo, "steertts_vocoder": json.dumps(dict(
        dim=bb.embed.out_channels, inter=bb.convnext[0].pwconv1.out_features, layers=len(bb.convnext),
        n_mels=bb.embed.in_channels, n_fft=head.istft.n_fft, hop=head.istft.hop_length, padding=head.istft.padding))}
    out = Path(a.out)
    (out / "int8").mkdir(parents=True, exist_ok=True)
    save_plain(params, out / "vocos.safetensors", metadata=meta)
    save_int8(params, out / "int8" / "vocos.safetensors", metadata=meta)
    print(f"wrote {out / 'vocos.safetensors'} and {out / 'int8' / 'vocos.safetensors'} ({len(params)} tensors)")


if __name__ == "__main__":
    main()
