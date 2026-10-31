"""Evaluation, step 1: render the same sentences with the C library and with the Python packages.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<repo>:<repo>/edge:<repo>/c/python python c/eval/render.py \
        --v1 <v1 dir> --edge <edge dir> --texts seedtts_testset/en/meta.lst --every 5 --out <eval dir>

Configurations (directory under --out):
  v1_fp32/{c,py}   v1, flow refiner on, fp32           v1_int8/{c,py}   v1, flow on, int8 weights
  edge/{c,py}      edge (int4)                         v1_fp32/py_g2pen  Python v1 with its own g2p_en front end
The py renders (except py_g2pen) use the edge front end patched into Python, as the C library does, so c vs py
compares the model code; py_g2pen measures what the front-end choice changes. Voices rotate over the edge voices
(all of them exist in v1). Writes float32 WAVs and parity.json (sample counts, waveform SNR, log-mel distance, render
times) per configuration.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def logmel(y):
    import torchaudio
    m = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=100, power=1.0)
    return torch.log(torch.clamp(m(torch.from_numpy(np.asarray(y, np.float32))), min=1e-5)).numpy()


def compare(a, b):
    if len(a) != len(b) or len(a) == 0:
        return dict(same_len=len(a) == len(b), snr_db=None, mel_l1_db=None)
    e = np.sum((a.astype(np.float64) - b) ** 2)
    snr = 200.0 if e == 0 else 10 * np.log10(np.sum(a.astype(np.float64) ** 2) / e)
    la, lb = logmel(a), logmel(b)
    mel = float(np.mean(np.abs(la - lb)) * 20 / np.log(10))   # mean |log-mel difference| in dB
    return dict(same_len=True, snr_db=float(snr), mel_l1_db=mel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--edge", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    lines = [l.rstrip("\n").split("|") for l in open(a.texts, encoding="utf-8") if l.strip()]
    texts = [l[3] for l in lines][:: a.every]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import steertts.markup as M
    import steertts.pipeline as P
    import steertts_edge.text as et
    from steertts import SteerTTS
    from steertts_c import Engine
    from steertts_edge import EdgeTTS
    et.set_assets(Path(a.edge) / "front")
    g2pen = P.to_phonemes
    edge_voices = EdgeTTS(a.edge).voices
    reqs = [dict(text=t, voice=edge_voices[i % len(edge_voices)]) for i, t in enumerate(texts)]
    json.dump(dict(texts=texts, requests=reqs), open(out / "requests.json", "w"), indent=1)

    def run(name, fn, n_threads_note):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        times, lens = [], []
        for i, r in enumerate(reqs):
            t0 = time.time()
            y = np.asarray(fn(r), np.float32)
            times.append(time.time() - t0)
            lens.append(len(y))
            sf.write(str(d / f"{i:04d}.wav"), y, 24000, subtype="FLOAT")
        audio = sum(lens) / 24000
        json.dump(dict(render_s=sum(times), audio_s=audio, rtf=sum(times) / audio, threads=n_threads_note),
                  open(d / "timing.json", "w"))
        print(f"{name}: {len(reqs)} renders, {audio:.0f} s audio, RTF {sum(times) / audio:.3f}", flush=True)

    cfgs = {
        "v1_fp32": (lambda: SteerTTS(a.v1, device="cpu"), lambda: Engine(a.v1, threads=a.threads)),
        "v1_int8": (lambda: SteerTTS(a.v1, device="cpu", precision="int8"),
                    lambda: Engine(a.v1, threads=a.threads, precision="int8")),
        "edge": (lambda: EdgeTTS(a.edge), lambda: Engine(a.edge, threads=a.threads)),
    }
    for name, (mk_py, mk_c) in cfgs.items():
        if a.only and name not in a.only:
            continue
        c = mk_c()
        run(f"{name}/c", lambda r: c.synthesize(r), a.threads)
        c.close()
        py = mk_py()
        if name == "edge":
            run(f"{name}/py", lambda r: py(r["text"], voice=r["voice"]), a.threads)
        else:
            P.to_phonemes = M.to_phonemes = et.to_phonemes
            run(f"{name}/py", lambda r: py.synthesize(r), a.threads)
            if name == "v1_fp32":
                P.to_phonemes = M.to_phonemes = g2pen
                run(f"{name}/py_g2pen", lambda r: py.synthesize(r), a.threads)
        rows = []
        for i in range(len(reqs)):
            ya, _ = sf.read(str(out / name / "py" / f"{i:04d}.wav"), dtype="float32")
            yb, _ = sf.read(str(out / name / "c" / f"{i:04d}.wav"), dtype="float32")
            rows.append(compare(ya, yb))
        snr = np.array([r["snr_db"] for r in rows if r["snr_db"] is not None])
        mel = np.array([r["mel_l1_db"] for r in rows if r["mel_l1_db"] is not None])
        summ = dict(n=len(rows), same_len=sum(r["same_len"] for r in rows),
                    snr_db=dict(min=float(snr.min()), p5=float(np.percentile(snr, 5)), median=float(np.median(snr))),
                    mel_l1_db=dict(median=float(np.median(mel)), max=float(mel.max())))
        json.dump(dict(summary=summ, rows=rows), open(out / name / "parity.json", "w"), indent=1)
        print(name, json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
