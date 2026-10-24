"""Export sentence-transformers/all-MiniLM-L6-v2 for the C runtime (src/st_prompt.c).

    python tools/export_minilm.py <out_dir>

writes
    <out>/minilm.safetensors        fp32 BertModel state dict, original names (embeddings.*, encoder.layer.N.*, pooler.*)
    <out>/int8/minilm.safetensors   the packed int8 layout (tools/st_pack.save_int8 over named_parameters()), i.e.
                                    exactly what steertts.quant.round_int8_ does to the encoder in the Python int8 build
    <out>/minilm_vocab.txt          the WordPiece vocabulary (one token per line, line number = id)
Needs transformers + the model in the HF cache (HF_HUB_OFFLINE=1 works).
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from st_pack import save_int8, save_plain  # noqa: E402

ENC = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    from transformers import AutoModel, AutoTokenizer
    out = pathlib.Path(sys.argv[1])
    (out / "int8").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ENC)
    m = AutoModel.from_pretrained(ENC).eval()
    md = {"source": ENC}
    sd = {k: v for k, v in m.state_dict().items() if not k.endswith("position_ids")}
    save_plain(sd, out / "minilm.safetensors", md)
    save_int8(dict(m.named_parameters()), out / "int8" / "minilm.safetensors", md)
    vocab = tok.get_vocab()
    inv = sorted(vocab.items(), key=lambda kv: kv[1])
    assert [i for _, i in inv] == list(range(len(inv))), "vocab ids are not dense"
    with open(out / "minilm_vocab.txt", "w", encoding="utf-8", newline="\n") as f:
        for t, _ in inv:
            assert "\n" not in t
            f.write(t + "\n")
    print(f"wrote {out}/minilm.safetensors, {out}/int8/minilm.safetensors, {out}/minilm_vocab.txt ({len(inv)} tokens)")


if __name__ == "__main__":
    with torch.no_grad():
        main()
