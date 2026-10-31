"""Build the helper-asset directory the C library needs next to a v1 model (the edge package already ships its own):

    python c/tools/export_assets.py <v1 model dir>/native --front <edge package>/native/front

writes
    vocos.safetensors, int8/vocos.safetensors         Vocos vocoder (charactr/vocos-mel-24khz, MIT)
    minilm.safetensors, int8/minilm.safetensors,      prompt encoder (sentence-transformers/all-MiniLM-L6-v2,
    minilm_vocab.txt                                   Apache-2.0)
    wavlm_sv.safetensors, int8/wavlm_sv.safetensors   speaker encoder (microsoft/wavlm-base-plus-sv)
    front/                                            text front-end assets (copied from the edge package)
The int8 files reproduce the Python package's int8 build exactly (steertts.quant.round_int8_). Needs torch, vocos,
transformers and safetensors; the three checkpoints are downloaded from the Hugging Face Hub on first use.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--front", required=True, help="the edge package's native/front directory")
    ap.add_argument("--skip", nargs="*", default=[], choices=["vocos", "minilm", "wavlm"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("vocos", "minilm", "wavlm"):
        if name not in a.skip:
            subprocess.run([sys.executable, str(HERE / f"export_{name}.py"), str(out)], check=True)
    src = Path(a.front)
    if not (src / "lexicon.bin").exists():
        sys.exit(f"{src} does not look like the edge package's native/front directory")
    shutil.copytree(src, out / "front", dirs_exist_ok=True)
    print("assets in", out)


if __name__ == "__main__":
    main()
