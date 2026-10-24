"""CPU speed of the edge package: load time, RTF over 20 sentences, time to first audio, peak RSS.

    python tools/bench_speed.py [package_dir] --runtime onnx torch native c --threads 1 4 [--native-bin PATH] [--c-bin PATH] [--write]

Each (runtime, thread count) runs in its own fresh process. onnx = OnnxTTS (ONNX Runtime, no torch import), torch =
EdgeTTS, native = the C++ CLI (`steertts-edge --bench-json`, same sentences and probes, built from native/), c = the
stand-alone C library's CLI (`steertts --bench-json`, built from c/ in the GitHub repo).
--write stores the result under "speed" in tools/release_numbers.json (then: python tools/render_tables.py).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TEXTS = [
    "The weather was calm when we left the harbour this morning.",
    "Please remember to bring the documents to the meeting on Thursday.",
    "I never thought a small garden could grow so many tomatoes.",
    "She opened the letter slowly and read it twice before speaking.",
    "Our train was delayed by forty minutes because of the snow.",
    "He laughed, shook his head, and walked back into the kitchen.",
    "Thanks for calling. How can I help you today?",
    "The museum opens at nine and closes at half past five.",
    "If you turn left at the bakery, you will see the station.",
    "Nobody expected the quiet student to win the debate.",
    "Could you send me the report before lunch?",
    "The river had risen overnight, and the old bridge was closed.",
    "We should probably leave now if we want to catch the last bus.",
    "Her grandmother kept every letter in a wooden box under the bed.",
    "That was the best soup I have had in years.",
    "The engineers tested the new design for three weeks before approving it.",
    "Wait, did you hear that noise outside?",
    "A small dog followed us all the way home from the park.",
    "The price of coffee went up again this month.",
    "When the lights came back on, everyone started clapping.",
]
SHORT = "Hello, how are you?"

CHILD = r"""
import json, resource, sys, time
t0 = time.perf_counter()
threads, runtime = int(sys.argv[2]), sys.argv[5]
if runtime == "torch":
    import torch
    torch.set_num_threads(threads)
    from steertts_edge import EdgeTTS as R
    version = "torch " + torch.__version__
else:
    import onnxruntime
    from steertts_edge import OnnxTTS as R
    version = "onnxruntime " + onnxruntime.__version__
tts = R(sys.argv[1], threads=threads)
load = time.perf_counter() - t0
texts, short = json.loads(sys.argv[3]), sys.argv[4]
for t in texts[:2]:
    tts(t)                                            # warm-up
# time to first audio, short sentence: one-shot call and first streamed chunk
ttfa = []
for _ in range(5):
    t = time.perf_counter(); tts(short); ttfa.append(time.perf_counter() - t)
first = []
for _ in range(5):
    t = time.perf_counter(); next(iter(tts.stream(short + " And a second sentence follows."))); first.append(time.perf_counter() - t)
synth = audio = 0.0
per = []
for t in texts:
    s = time.perf_counter(); y = tts(t); d = time.perf_counter() - s
    synth += d; audio += len(y) / tts.sample_rate; per.append(d / (len(y) / tts.sample_rate))
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
per.sort(); ttfa.sort(); first.sort()
print(json.dumps(dict(threads=threads, load_s=round(load, 3), rtf=round(synth / audio, 4),
                      rtf_median=round(per[len(per) // 2], 4), audio_s=round(audio, 2), synth_s=round(synth, 3),
                      ttfa_short_s=round(ttfa[2], 3), ttfa_stream_first_chunk_s=round(first[2], 3),
                      peak_rss_mb=round(rss, 1), runtime=runtime, version=version)))
"""


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg", nargs="?", default=str(HERE))
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--runtime", nargs="+", default=["onnx"], choices=["onnx", "torch", "native", "c"])
    ap.add_argument("--native-bin", default="native/build/steertts-edge", help="the C++ CLI for --runtime native")
    ap.add_argument("--c-bin", default="c/build/steertts", help="the C library's CLI for --runtime c")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    runs = []
    for rt in a.runtime:
        for n in a.threads:
            env = dict(os.environ, OMP_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n), CUDA_VISIBLE_DEVICES="")
            if rt in ("native", "c"):
                cmd = [a.native_bin if rt == "native" else a.c_bin, "--model", a.pkg, "--threads", str(n), "--bench-json"]
            else:
                cmd = [sys.executable, "-c", CHILD, a.pkg, str(n), json.dumps(TEXTS), SHORT, rt]
            out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
            runs.append(json.loads(out.strip().splitlines()[-1]))
            print(runs[-1], flush=True)
    res = dict(cpu=cpu_model(), load_avg_1m=round(os.getloadavg()[0], 1), n_texts=len(TEXTS), runs=runs)
    print(json.dumps(res, indent=1))
    if a.write:
        p = HERE / "tools" / "release_numbers.json"
        numbers = json.load(open(p)) if p.exists() else {}
        numbers["speed"] = res
        json.dump(numbers, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
