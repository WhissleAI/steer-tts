"""Evaluation, step 2: score the renders of render.py and summarise.

    python c/eval/score.py wer <eval dir>       # Whisper ($ASR, default whisper-small.en) + Whisper's English
                                                #   normaliser; own edit distance; CPU is fine
    python c/eval/score.py utmos <eval dir>     # UTMOSv2 (sarulab-speech/UTMOSv2, fusion_stage3 fold0) on the first
                                                #   $UTMOS_N (default 60) sentences of each set
    python c/eval/score.py report <eval dir>    # summary.json + a markdown table on stdout

WER is corpus-level (word edits / reference words) against the sentence text; scores are cached per wav, so the steps
can be re-run incrementally.
"""
import json
import os
import sys
from pathlib import Path

SETS = ["v1_fp32/c", "v1_fp32/py", "v1_fp32/py_g2pen", "v1_int8/c", "v1_int8/py", "edge/c", "edge/py"]


def wavs(d, limit=None):
    w = sorted(str(p) for p in Path(d).glob("*.wav"))
    return w[:limit] if limit else w


DEV = os.environ.get("DEVICE", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" else "cpu")
ASR = os.environ.get("ASR", "openai/whisper-large-v3-turbo" if DEV == "cuda" else "openai/whisper-small.en")
UTMOS_N = int(os.environ.get("UTMOS_N", "0" if DEV == "cuda" else "60")) or None   # UTMOSv2 on the first N per set


def load_cache(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def wer(root):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    cache_p = root / f"asr_{ASR.split('/')[-1]}.json"
    res = load_cache(cache_p)
    todo = [w for s in SETS for w in wavs(root / s) if w not in res]
    print(f"asr: {len(todo)} to transcribe", flush=True)
    if todo:
        name = ASR
        proc = WhisperProcessor.from_pretrained(name)
        m = WhisperForConditionalGeneration.from_pretrained(name, dtype=torch.float16 if DEV == "cuda" else torch.float32)
        m = m.to(DEV).eval()
        for i in range(0, len(todo), 8):
            ch = todo[i:i + 8]
            aud = []
            for w in ch:
                y, sr = sf.read(w, dtype="float32")
                import torchaudio.functional as F
                aud.append(F.resample(torch.from_numpy(y), sr, 16000).numpy() if len(y) else np.zeros(1600, np.float32))
            f = proc(aud, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                kw = {} if name.endswith(".en") else dict(language="en", task="transcribe")
                ids = m.generate(f.input_features.to(DEV, m.dtype), **kw)
            for w, t in zip(ch, proc.batch_decode(ids, skip_special_tokens=True)):
                res[w] = t.strip()
            if i % 80 == 0:
                json.dump(res, open(cache_p, "w"))
                print(f"  {i + len(ch)}/{len(todo)}", flush=True)
        json.dump(res, open(cache_p, "w"))
    norm = EnglishTextNormalizer({})
    texts = json.load(open(root / "requests.json"))["texts"]

    def lev(a, b):
        d = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            prev, d[0] = d[0], i
            for j, y in enumerate(b, 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
        return d[len(b)]

    out = {}
    for s in SETS:
        ws = wavs(root / s)
        if not ws:
            continue
        e = n = 0
        for w in ws:
            ref = norm(texts[int(Path(w).stem)]).split()
            e += lev(ref, norm(res[w]).split())
            n += len(ref)
        out[s] = dict(wer=100.0 * e / n, n=len(ws), asr=ASR)
        print(f"{s:18s} WER {100.0 * e / n:.2f} %  ({len(ws)} clips)")
    json.dump(out, open(root / "wer.json", "w"), indent=1)


def utmos(root):
    import torch
    import utmosv2
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    cache_p = root / "utmos_cache.json"
    res = load_cache(cache_p)
    todo = [w for s in SETS for w in wavs(root / s, UTMOS_N) if w not in res]
    print(f"utmosv2: {len(todo)} to score", flush=True)
    if todo:
        m = utmosv2.create_model(pretrained=True, device=DEV)
        for i, w in enumerate(todo):
            res[w] = float(m.predict(input_path=w, device=DEV, num_workers=0, verbose=False))
            if i % 100 == 99:
                json.dump(res, open(cache_p, "w"))
                print(f"  {i + 1}/{len(todo)}", flush=True)
        json.dump(res, open(cache_p, "w"))
    out = {}
    for s in SETS:
        v = [res[w] for w in wavs(root / s, UTMOS_N) if w in res]
        if v:
            out[s] = dict(utmosv2=sum(v) / len(v), n=len(v))
            print(f"{s:18s} UTMOSv2 {sum(v) / len(v):.3f}  ({len(v)} clips)")
    json.dump(out, open(root / "utmos.json", "w"), indent=1)


def report(root):
    W, U = load_cache(root / "wer.json"), load_cache(root / "utmos.json")
    rows = {}
    for s in SETS:
        cfg, side = s.split("/")
        t = load_cache(root / s / "timing.json")
        p = load_cache(root / cfg / "parity.json").get("summary", {}) if side == "c" else {}
        rows[s] = dict(wer=W.get(s, {}).get("wer"), utmosv2=U.get(s, {}).get("utmosv2"), rtf=t.get("rtf"), parity=p)
    json.dump(rows, open(root / "summary.json", "w"), indent=1)
    f = lambda v, fmt: fmt.format(v) if v is not None else "-"
    print("| Set | WER % | UTMOSv2 | RTF | same length | waveform SNR median / min (dB) | log-mel diff median (dB) |")
    print("|---|---|---|---|---|---|---|")
    for s, r in rows.items():
        p = r["parity"]
        print(f"| {s} | {f(r['wer'], '{:.2f}')} | {f(r['utmosv2'], '{:.3f}')} | {f(r['rtf'], '{:.3f}')} | "
              f"{str(p['same_len']) + '/' + str(p['n']) if p else '-'} | "
              f"{f(p.get('snr_db', {}).get('median'), '{:.1f}')} / {f(p.get('snr_db', {}).get('min'), '{:.1f}')} | "
              f"{f(p.get('mel_l1_db', {}).get('median'), '{:.4f}')} |")


if __name__ == "__main__":
    {"wer": wer, "utmos": utmos, "report": report}[sys.argv[1]](Path(sys.argv[2]))
