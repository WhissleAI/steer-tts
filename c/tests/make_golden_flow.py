"""Golden data for c/tests/test_flow.c: torch.randn, the flow refiner (fp32 / int8), the prompted flow (cloning),
the Vocos mel front end and SteerTTS.prompt_mel, produced with the Python package.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<repo>:<repo>/edge python c/tests/make_golden_flow.py [--out c/tests/golden/flow]
    ... --bench          time Flow.sample in PyTorch (1 and 4 threads) on the refiner golden input

Files: <name>.bin (raw little-endian float32 unless noted) + meta.json (shapes, seeds, settings, randn hashes).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
MODEL = os.environ.get("STEERTTS_V1_MODEL", "../models/steer-tts")
TEXT = "The quiet harbour town woke slowly, as fishing boats drifted back with the morning tide."
TEXT_P = "Now say this in the voice of the reference clip."
REF_TEXT = "This short sentence is the reference recording for cloning."


def fnv1a(a):
    h = 0xcbf29ce484222325
    for b in np.ascontiguousarray(a, np.float32).tobytes():
        h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
    return "%016x" % h


def mu_g(tts, text, voice):
    from steertts.text import to_phonemes
    ids, widx, _ = to_phonemes(text)
    c, m = tts.control_tensors(None)
    mel, aux = tts.model.infer(torch.tensor([ids]), torch.tensor([tts.n_speakers]), torch.tensor([0]), c, m,
                               word_idx=torch.tensor([widx]), spk_vec=tts.speaker_vector(voice))
    return mel.float(), aux["g"].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "golden" / "flow"))
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    from steertts import SteerTTS
    from steertts.audio import load_wav, mel_extractor
    tts = SteerTTS(MODEL, device="cpu")
    with torch.no_grad():
        mu, g = mu_g(tts, TEXT, "voice_1688")
    if a.bench:
        return bench(tts, mu, g)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    save = lambda name, x: np.ascontiguousarray(np.asarray(x, np.float32)).tofile(out / f"{name}.bin")

    # ---- randn: full values for small sizes, FNV-1a of the float32 bytes for big ones
    rn = {}
    for seed in (0, 1, 1234, 2**33 + 5):
        for n in (1, 2, 15, 16, 17, 31, 1000, 100 * 537):
            r = torch.randn(n, generator=torch.Generator().manual_seed(seed)).numpy()
            rn[f"{seed}_{n}"] = fnv1a(r)
    meta["randn_fnv"] = rn
    save("randn_1234_1000", torch.randn(1000, generator=torch.Generator().manual_seed(1234)))

    # ---- refiner fp32 / int8 on a real mu / g
    with torch.no_grad():
        T = mu.shape[-1]
        save("ref_mu", mu[0].T)            # time-major [T,100]
        save("ref_g", g[0])
        seed = 1234
        meta["refiner"] = dict(T=T, seed=seed, steps=10, temp=0.4)
        save("ref_out_fp32", tts.flow.sample(mu, g, steps=10, temp=0.4, seed=seed)[0].T)
        save("ref_out_fp32_s1", tts.flow.sample(mu, g, steps=1, temp=0.4, seed=seed)[0].T)
        t8 = SteerTTS(MODEL, device="cpu", precision="int8")
        save("ref_out_int8", t8.flow.sample(mu, g, steps=10, temp=0.4, seed=seed)[0].T)

    # ---- reference clip: a synthesized 24 kHz utterance (int16-quantized) and a 16 kHz band-limited copy
    import soundfile as sf
    import soxr
    wav = tts(REF_TEXT, voice="voice_0042", seed=7)
    wav = np.round(np.clip(wav, -1, 1) * 32767).astype(np.int16)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    sf.write(tmp / "ref24.wav", wav, 24000, subtype="PCM_16")
    y16 = soxr.resample(wav.astype(np.float32) / 32768, 24000, 16000)
    sf.write(tmp / "ref16.wav", y16.astype(np.float32), 16000, subtype="FLOAT")
    wav.tofile(out / "clip24_i16.bin")                     # int16; C uses x / 32768 (= soundfile's float read)
    y24, sr24 = load_wav(tmp / "ref24.wav")
    y16_24, sr16 = load_wav(tmp / "ref16.wav")              # resampled back to 24 kHz, as the pipeline does
    save("clip16_24k", y16_24)
    meta["clip"] = dict(n24=len(y24), n16_24=len(y16_24), sr16=sr16)
    with torch.no_grad():
        mel24 = mel_extractor()(torch.from_numpy(y24)[None])[0]
        save("clip24_mel", mel24.T)
        meta["clip"]["mel_T"] = mel24.shape[-1]
        mup, gp = mu_g(tts, TEXT_P, "voice_0042")
        save("p_mu", mup[0].T)
        save("p_g", gp[0])
        pm24 = tts.prompt_mel(tmp / "ref24.wav", mup)
        pm16 = tts.prompt_mel(tmp / "ref16.wav", mup)
        save("pm24", pm24[0].T)
        save("pm16", pm16[0].T)
        meta["prompted"] = dict(T=mup.shape[-1], P24=pm24.shape[-1], P16=pm16.shape[-1], seed=5, steps=10, temp=0.4)

        def psample(fl, pm, steps=10, cfg=0.0):
            P = pm.shape[-1]
            full = torch.cat([pm, mup], -1)
            cm = torch.zeros_like(full[:, :1]); cm[..., :P] = 1
            return fl.sample(full, gp, steps=steps, temp=0.4, seed=5, ctx=full, cm=cm, cfg=cfg)[..., P:][0].T

        save("p_out_fp32", psample(tts.pflow, pm16))
        save("p_out_fp32_s1", psample(tts.pflow, pm16, steps=1))
        save("p_out_fp32_cfg", psample(tts.pflow, pm24, steps=2, cfg=0.5))
        save("p_out_int8", psample(t8.pflow, pm16))
        save("p_out_noprompt", tts.pflow.sample(mup, gp, steps=2, temp=0.4, seed=5)[0].T)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "randn_fnv"}))


def bench(tts, mu, g):
    T = mu.shape[-1]
    for th in (1, 4):
        torch.set_num_threads(th)
        with torch.no_grad():
            tts.flow.sample(mu, g, steps=1, temp=0.4, seed=0)
            best = 1e9
            for _ in range(3):
                t0 = time.perf_counter()
                tts.flow.sample(mu, g, steps=10, temp=0.4, seed=0)
                best = min(best, time.perf_counter() - t0)
        sec = T * 256 / 24000
        print(f"torch {th} thr: T={T} ({sec:.2f} s audio) {best*1e3:.0f} ms total, {best*1e3/20:.1f} ms/eval, "
              f"{best*1e3/sec:.0f} ms per s of audio")


if __name__ == "__main__":
    main()
