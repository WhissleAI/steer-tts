"""Golden plans for test_director.c, from the parity file export_director.py writes next to director.bin (the
8-bit PyTorch model's outputs on held-out chapters).

  python make_golden_director.py <parity.json>    -> golden/director/plans.txt

One chapter per block: "chapter <genre index or -1>", then per sentence a line
"<dialogue> <has_speaker> <para_len> <para_index> <6 plan numbers>\\t<text>". Keep director.bin in the edge model
directory in step with it (both come from the same export)."""
import json, os, sys
P = json.load(open(sys.argv[1]))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "director", "plans.txt")
with open(out, "w", encoding="utf-8") as f:
    for c in P["cases"]:
        g = c["genre"].index(1.0) if 1.0 in c["genre"] else -1
        f.write(f"chapter {g}\n")
        for inp, plan in zip(c["inputs"], c["out"]):
            t = inp["text"].replace("\t", " ").replace("\n", " ")
            f.write(f"{int(inp['dialogue'])} {int(inp['speaker'])} {inp['para_len']} {inp['para_index']} "
                    + " ".join(f"{v:.7g}" for v in plan) + f"\t{t}\n")
print(out)
