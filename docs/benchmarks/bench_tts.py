"""One TTS engine, one thread count, one fresh process: load time, speed, time to first audio, peak RSS.

    python -I bench_tts.py --engine steer|phonon|kokoro --model DIR --threads N --out results.jsonl \
        [--reps 3] [--long] [--wav-dir DIR] [--label host] [--quality quality_texts.jsonl]

Speed protocol (the same as `steertts --bench-json`, so the C library's own numbers compare directly):
  load (imports + model load) -> 2 warm-up sentences -> 5x the short line "Hello, how are you?" (blocking; median =
  time to first audio without streaming) -> 5x the two-sentence stream probe (median time to the first streamed
  chunk) -> the 20 bench sentences, --reps times (RTF = synthesis time / audio time per pass; the median pass is
  reported) -> with --long, the three paragraphs of texts.json streamed once each (RTF and first chunk).
--quality renders every text of a JSONL file ({id, text}) once to --wav-dir after the same warm-up (for WER / MOS).

Each engine runs with its maintainers' documented defaults: Steer edge (C library, default voice), Phonon (ptts,
q8 weights, default voice, temperature 0.3), Kokoro (kokoro KPipeline, PyTorch, voice af_heart, speed 1). CPU only.
Thread count: the engine's own setting plus OMP/MKL/OpenBLAS env vars, set before the engine is imported.
"""
import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
T_START = time.perf_counter()


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def cpu_name():
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
        model = Path("/proc/device-tree/model")
        if model.exists():
            return model.read_text().strip("\x00\n")
    except Exception:
        pass
    return platform.processor()


class PiSampler(threading.Thread):
    """Samples Raspberry Pi temperature, ARM clock and throttling bits (vcgencmd) every 0.5 s, if available."""

    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.stop = [], threading.Event()
        self.ok = subprocess.call(["which", "vcgencmd"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

    @staticmethod
    def read():
        def q(*a):
            return subprocess.check_output(["vcgencmd", *a], text=True).strip().split("=")[1]
        return float(q("measure_temp").rstrip("'C")), int(q("measure_clock", "arm")) / 1e6, int(q("get_throttled"), 16)

    def run(self):
        while self.ok and not self.stop.is_set():
            try:
                self.rows.append(self.read())
            except Exception:
                pass
            self.stop.wait(0.5)

    def summary(self):
        if not self.rows:
            return None
        temps, mhz, thr = zip(*self.rows)
        return {"max_temp_c": max(temps), "min_mhz": min(mhz), "mean_mhz": round(statistics.mean(mhz)),
                "throttle_now_bits_seen": sorted({b for t in thr for b in range(4) if t >> b & 1}),
                "samples": len(self.rows)}


# ---------------------------------------------------------------- engines

class Steer:
    name = "steer-tts-edge"

    def __init__(self, model, threads):
        sys.path.insert(0, str(HERE.parents[1] / "c" / "python"))
        from steertts_c import Engine, _api
        self.e = Engine(model, threads=threads)
        self.sr = 24000
        self.version = "steertts C library " + _api().steertts_version().decode()
        self.settings = {"voice": self.e.default_voice, "runtime": "C11 library via ctypes"}

    def synth(self, text):
        return self.e(text)

    def stream_first(self, text, full=False):
        t0, first, n = time.perf_counter(), [None], [0]

        def cb(a, info):
            if first[0] is None and len(a):
                first[0] = time.perf_counter() - t0
            n[0] += len(a)
            return not full
        self.e.stream_cb(cb, text)
        return first[0], time.perf_counter() - t0, n[0]


class Phonon:
    name = "phonon-31m-en"

    def __init__(self, model, threads):
        import ptts
        ptts.set_num_threads(threads)
        assert ptts.get_num_threads() == threads
        self.tts = ptts.TTS(model=str(model), device="cpu", quant="q8", lang="en")
        self.sr = self.tts.sample_rate
        self.np = __import__("numpy")
        self.version = "ptts " + ptts.__version__
        self.settings = {"voice": "default", "quant": "q8", "temperature": 0.3, "runtime": "ptts (Rust, Python wheel)"}

    def synth(self, text):
        return self.np.asarray(self.tts.synth(text), dtype=self.np.float32)

    def stream_first(self, text, full=False):
        t0, first, n = time.perf_counter(), None, 0
        with self.tts.stream(text) as audio:
            for pcm in audio:
                k = len(pcm)
                if not k:
                    continue
                if first is None:
                    first = time.perf_counter() - t0
                n += k
                if not full:
                    break
        return first, time.perf_counter() - t0, n


class Kokoro:
    name = "kokoro-82m"

    def __init__(self, model, threads):
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        from kokoro import KModel, KPipeline
        import kokoro
        model = Path(model)
        km = KModel(repo_id="hexgrad/Kokoro-82M", config=str(model / "config.json"),
                    model=str(model / "kokoro-v1_0.pth")).to("cpu").eval()
        self.p = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=km, device="cpu")
        self.voice = str(model / "voices" / "af_heart.pt")
        self.np, self.torch, self.sr = __import__("numpy"), torch, 24000
        self.version = f"kokoro {kokoro.__version__}, torch {torch.__version__}"
        self.settings = {"voice": "af_heart", "speed": 1.0, "lang_code": "a", "runtime": "PyTorch (kokoro KPipeline)"}

    def _segments(self, text):
        with self.torch.inference_mode():
            for _gs, _ps, audio in self.p(text, voice=self.voice, speed=1.0):
                if audio is not None:
                    yield audio.numpy() if hasattr(audio, "numpy") else self.np.asarray(audio)

    def synth(self, text):
        segs = list(self._segments(text))
        return self.np.concatenate(segs).astype(self.np.float32) if segs else self.np.zeros(0, self.np.float32)

    def stream_first(self, text, full=False):
        t0, first, n = time.perf_counter(), None, 0
        for a in self._segments(text):
            if first is None and len(a):
                first = time.perf_counter() - t0
            n += len(a)
            if not full:
                break
        return first, time.perf_counter() - t0, n


class KokoroOnnx:
    """Kokoro-82M v1.0 on ONNX Runtime through the kokoro-onnx package (espeak-ng G2P, fp32 graph)."""
    name = "kokoro-82m-onnx"

    def __init__(self, model, threads):
        import asyncio
        import onnxruntime as ort
        import kokoro_onnx
        from importlib.metadata import version
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        model = Path(model)
        sess = ort.InferenceSession(str(model / "kokoro-v1.0.onnx"), so, providers=["CPUExecutionProvider"])
        self.k = kokoro_onnx.Kokoro.from_session(sess, str(model / "voices-v1.0.bin"))
        self.np, self.asyncio, self.sr = __import__("numpy"), asyncio, 24000
        self.version = f"kokoro-onnx {version('kokoro-onnx')}, onnxruntime {ort.__version__}"
        self.settings = {"voice": "af_heart", "speed": 1.0, "lang": "en-us", "graph": "kokoro-v1.0.onnx (fp32)",
                         "runtime": "ONNX Runtime (kokoro-onnx)"}

    def synth(self, text):
        y, _ = self.k.create(text, voice="af_heart", speed=1.0, lang="en-us")
        return self.np.asarray(y, dtype=self.np.float32)

    def stream_first(self, text, full=False):
        async def go():
            t0, first, n = time.perf_counter(), None, 0
            agen = self.k.create_stream(text, voice="af_heart", speed=1.0, lang="en-us")
            async for y, _sr in agen:
                if first is None and len(y):
                    first = time.perf_counter() - t0
                n += len(y)
                if not full:
                    break
            await agen.aclose()
            return first, time.perf_counter() - t0, n
        return self.asyncio.run(go())


ENGINES = {"steer": Steer, "phonon": Phonon, "kokoro": Kokoro, "kokoro-onnx": KokoroOnnx}


def save_wav(path, audio, sr):
    import numpy as np
    import wave
    pcm = (np.clip(audio, -1, 1) * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=ENGINES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--quality", help="JSONL of {id, text}: render each once to --wav-dir (no speed bench)")
    ap.add_argument("--wav-dir")
    ap.add_argument("--label", default=platform.node())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "RAYON_NUM_THREADS"):
        os.environ[k] = str(a.threads)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    texts = json.loads((HERE / "texts.json").read_text())
    sampler = PiSampler()
    pi_before = sampler.read() if sampler.ok else None
    sampler.start()
    load0 = time.perf_counter()
    eng = ENGINES[a.engine](a.model, a.threads)
    load_s = time.perf_counter() - load0
    rss_load = peak_rss_mb()
    wav_dir = Path(a.wav_dir) if a.wav_dir else None
    if wav_dir:
        wav_dir.mkdir(parents=True, exist_ok=True)
    for t in texts["bench20"][:2]:
        eng.synth(t)                                                      # warm-up
    rec = {"engine": eng.name, "version": eng.version, "settings": eng.settings, "threads": a.threads,
           "label": a.label, "cpu": cpu_name(), "machine": platform.machine(), "os": platform.platform(),
           "python": platform.python_version(), "load_s": round(load_s, 3), "rss_after_load_mb": round(rss_load, 1),
           "loadavg_before": os.getloadavg(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "interpreter_start_to_ready_s": round(time.perf_counter() - T_START, 3)}
    if a.quality:
        rows = [json.loads(l) for l in open(a.quality) if l.strip()]
        out = []
        for r in rows:
            t0 = time.perf_counter()
            y = eng.synth(r["text"])
            d = time.perf_counter() - t0
            save_wav(wav_dir / f"{r['id']}.wav", y, eng.sr)
            out.append({"id": r["id"], "synth_s": round(d, 4), "audio_s": round(len(y) / eng.sr, 3)})
        rec.update(mode="quality", items=out, n=len(out))
    else:
        short = []
        for _ in range(5):
            t0 = time.perf_counter()
            y = eng.synth(texts["short"])
            short.append(time.perf_counter() - t0)
        first = [eng.stream_first(texts["stream_probe"])[0] for _ in range(5)]
        passes = []
        for rep in range(a.reps):
            synth = audio = 0.0
            per = []
            for i, t in enumerate(texts["bench20"]):
                t0 = time.perf_counter()
                y = eng.synth(t)
                d = time.perf_counter() - t0
                synth += d
                audio += len(y) / eng.sr
                per.append(d / (len(y) / eng.sr))
                if rep == 0 and wav_dir:
                    save_wav(wav_dir / f"bench{i:02d}.wav", y, eng.sr)
            passes.append({"rtf": synth / audio, "synth_s": synth, "audio_s": audio,
                           "rtf_median_sentence": statistics.median(per)})
        mid = sorted(passes, key=lambda p: p["rtf"])[len(passes) // 2]
        rec.update(mode="speed", rtf=round(mid["rtf"], 4), xrt=round(1 / mid["rtf"], 2),
                   rtf_passes=[round(p["rtf"], 4) for p in passes], audio_s=round(mid["audio_s"], 2),
                   rtf_median_sentence=round(mid["rtf_median_sentence"], 4),
                   ttfa_short_s=round(statistics.median(short), 4), ttfa_short_runs=[round(x, 4) for x in short],
                   ttfa_stream_first_chunk_s=round(statistics.median(first), 4),
                   ttfa_stream_runs=[round(x, 4) for x in first])
        if a.long:
            longs = []
            for k, t in enumerate(texts["long"]):
                f, total, n = eng.stream_first(t, full=True)
                longs.append({"first_chunk_s": round(f, 4), "synth_s": round(total, 3), "audio_s": round(n / eng.sr, 2),
                              "rtf": round(total / (n / eng.sr), 4)})
            rec["long"] = longs
            rec["long_rtf"] = round(sum(x["synth_s"] for x in longs) / sum(x["audio_s"] for x in longs), 4)
    sampler.stop.set()
    sampler.join(2)
    rec["peak_rss_mb"] = round(peak_rss_mb(), 1)
    rec["loadavg_after"] = os.getloadavg()
    if sampler.ok:
        rec["pi"] = {"before": pi_before, "after": sampler.read(), "during": sampler.summary()}
    line = json.dumps(rec)
    with open(a.out, "a") as f:
        f.write(line + "\n")
    print(line if not a.quality else json.dumps({k: v for k, v in rec.items() if k != "items"}))


if __name__ == "__main__":
    main()
