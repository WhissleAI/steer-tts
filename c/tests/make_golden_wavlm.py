"""Golden data for c/tests/test_wavlm.c (speaker encoder, WAV reader, resampler).

    python c/tests/make_golden_wavlm.py [--model <v1 model dir>]

Renders four clips with the v1 Python pipeline (0.3 / 2 / 6 / 12 s, synthetic voices; no recordings of real people),
stores them as 16-bit WAVs at 24 kHz and, resampled by soxr, at 16 kHz, and writes golden.safetensors with
  x16/fp32, x16/int8 [4, 512]   WavLMForXVector embeddings (L2-normalised) of the 16 kHz files as read (first 8 s,
                                tiled below 8000 samples) - the fp32 model and the Python int8 build (round_int8_)
  x24/fp32, x24/int8 [4, 512]   steertts.speaker.XVec on the 24 kHz files (soxr inside)
  rs/<case>/in, rs/<case>/out   soxr.resample reference pairs
  wav/<name>                    soundfile.read(dtype="float32") (+ .mean(1)) of the generated wav_<name>.wav files
Everything lands in c/tests/golden/wavlm/.
"""
import argparse
import copy
import os
import struct
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

OUT = Path(__file__).resolve().parent / "golden" / "wavlm"
CLIPS = [("Yes.", "voice_0001", 0.3),
         ("The weather looks lovely this afternoon.", "voice_1361", 2.0),
         ("After the meeting, we walked down to the river and watched the boats drift slowly past the old stone "
          "bridge.", "voice_0002", 6.0),
         ("Every morning the baker opened the shop before sunrise, lit the ovens, and arranged fresh loaves on the "
          "wooden shelves while the street outside was still quiet. By eight o'clock a long line of neighbours had "
          "formed at the door, chatting about the news.", "voice_0010", 12.0)]


def render(model_dir):
    from steertts.pipeline import SteerTTS
    tts = SteerTTS(model_dir, device="cpu")
    torch.manual_seed(0)
    for i, (text, voice, dur) in enumerate(CLIPS):
        w = np.asarray(tts(text, voice=voice), np.float32).reshape(-1)
        n = int(round(dur * 24000))
        if len(w) > n:   # the loudest window of the target length
            e = np.convolve(w.astype(np.float64) ** 2, np.ones(n), "valid")
            s = int(np.argmax(e))
            w = w[s:s + n]
        else:
            w = np.concatenate([w, np.zeros(n - len(w), np.float32)])
        w = w / max(1e-6, float(np.abs(w).max())) * 0.7
        sf.write(OUT / f"speech24_{i}.wav", w, 24000, subtype="PCM_16")
        y24, _ = sf.read(OUT / f"speech24_{i}.wav", dtype="float32")
        sf.write(OUT / f"speech16_{i}.wav", soxr.resample(y24, 24000, 16000), 16000, subtype="PCM_16")


def wav_files():
    """generated WAVs in every supported format + the soundfile reading of each"""
    rng = np.random.RandomState(7)
    n = 777
    t = np.arange(n) / 16000
    base = np.stack([0.6 * np.sin(2 * np.pi * 440 * t), 0.5 * np.sin(2 * np.pi * 1234 * t + 1)], 1)
    base += 0.3 * rng.uniform(-1, 1, base.shape)
    base[:4] = [[1.0, -1.0], [-1.0, 1.0], [0.0, -0.0], [0.99999, -0.99999]]   # extremes (soundfile clips to range)
    cases = {
        "u8": (base[:, :1], "WAV", "PCM_U8"), "i16": (base[:, :1], "WAV", "PCM_16"),
        "i24": (base[:, :1], "WAV", "PCM_24"), "i32": (base[:, :1], "WAV", "PCM_32"),
        "f32": (base[:, :1], "WAV", "FLOAT"), "f64": (base[:, :1] * 1.37, "WAV", "DOUBLE"),
        "i16_stereo": (base, "WAV", "PCM_16"), "f32_stereo": (base, "WAV", "FLOAT"),
        "ext_i24_stereo": (base, "WAVEX", "PCM_24"), "ext_f32_3ch": (np.concatenate([base, base[:, :1] * 0.5], 1),
                                                                     "WAVEX", "FLOAT"),
        "ext_i16_mono": (base[:, :1], "WAVEX", "PCM_16"),
    }
    out = {}
    for name, (x, fmt, sub) in cases.items():
        p = OUT / f"wav_{name}.wav"
        sf.write(p, x, 16000 if "stereo" not in name else 22050, format=fmt, subtype=sub)
        y, _ = sf.read(p, dtype="float32", always_2d=True)
        out[name] = y.mean(1).astype(np.float32) if y.shape[1] > 1 else y[:, 0].copy()
    # hand-made: extra chunks before / after fmt, an odd-sized chunk (padded), data size left open (0xFFFFFFFF)
    pcm = (base[:100, :1] * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)
    body = b"WAVE" + b"LIST" + struct.pack("<I", 5) + b"abcde" + b"\0" + b"fmt " + struct.pack("<I", 16) + fmt
    body += b"junk" + struct.pack("<I", 3) + b"xyz" + b"\0" + b"data" + struct.pack("<I", 0xFFFFFFFF) + pcm
    (OUT / "wav_chunks.wav").write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    y, _ = sf.read(OUT / "wav_chunks.wav", dtype="float32")
    out["chunks"] = y.astype(np.float32)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("STEERTTS_V1_MODEL",
                                                      "../models/steer-tts"))
    ap.add_argument("--no-render", action="store_true", help="reuse the speech*_*.wav files already there")
    a = ap.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.no_render:
        render(a.model)
    from transformers import AutoFeatureExtractor
    from steertts.quant import round_int8_
    from steertts.speaker import XVec
    xv = XVec()
    xq = XVec()
    xq.m = round_int8_(copy.deepcopy(xv.m))

    def emb16(x, y16):   # XVec.__call__ minus the resample
        y = y16[:128000]
        y = np.tile(y, int(np.ceil(8000 / max(len(y), 1)))) if len(y) < 8000 else y
        inp = x.fe([y], sampling_rate=16000, return_tensors="pt", padding=True)
        return torch.nn.functional.normalize(x.m(**inp).embeddings, dim=-1).float().numpy()[0]

    g = {}
    for tag, x in (("fp32", xv), ("int8", xq)):
        e16, e24 = [], []
        for i in range(len(CLIPS)):
            y16, _ = sf.read(OUT / f"speech16_{i}.wav", dtype="float32")
            y24, _ = sf.read(OUT / f"speech24_{i}.wav", dtype="float32")
            e16.append(emb16(x, y16))
            e24.append(x([y24])[0])
        g[f"x16/{tag}"] = np.stack(e16).astype(np.float32)
        g[f"x24/{tag}"] = np.stack(e24).astype(np.float32)
    for k in ("x16", "x24"):
        c = (g[f"{k}/fp32"] * g[f"{k}/int8"]).sum(1)
        print(f"{k}: python int8 vs fp32 cosine {np.round(c, 6)}")
    # resampler references
    y24, _ = sf.read(OUT / "speech24_1.wav", dtype="float32")
    t = np.arange(11025) / 44100
    chirp = (0.5 * np.sin(2 * np.pi * (100 * t + 0.5 * (21000 - 100) / t[-1] * t * t))).astype(np.float32)
    for case, x, sr_in, sr_out in (("s24_16", y24, 24000, 16000), ("c44_24", chirp, 44100, 24000)):
        g[f"rs/{case}/in"] = np.asarray(x, np.float32)
        g[f"rs/{case}/out"] = soxr.resample(x, sr_in, sr_out).astype(np.float32)
    for name, y in wav_files().items():
        g[f"wav/{name}"] = y
    from safetensors.numpy import save_file
    save_file({k: np.ascontiguousarray(v) for k, v in g.items()}, str(OUT / "golden.safetensors"),
              metadata={"clips": ",".join(f"{d}s" for _, _, d in CLIPS)})
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"wrote {OUT} ({total / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
