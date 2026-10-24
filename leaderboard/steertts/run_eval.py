"""
TTS synthesis for the Open TTS Leaderboard (SteerTTS backend, stage 1).

SteerTTS is a 9.6M-parameter acoustic model + Vocos (23.1M total; +17.6M flow refiner by default, or the 24.7M
prompted flow when cloning), loaded in-process from the `steertts` package. English only, so only the `en` splits.
Default voice = the pre-registered `voice_1361` (pinned here so the published numbers reproduce; the package default is now voice_1688) (no test-set voice selection); `--voice_clone` clones each sample's
`prompt_audio` (WavLM speaker vector + prompted flow; `prompt_text` is not used). Rows are synthesized one at a
time (no batched API); generation is timed after `--warmup_steps` untimed samples. `--ttfa_probe` times the first
sentence chunk of `SteerTTS.stream()` (sentence-level streaming).

    PYTHONPATH=scripts python steertts/run_eval.py --model_id WhissleAI/steer-tts --dataset_path bezzam/seed_tts_eval \
        --dataset tts --split en --device cuda:0 [--voice_clone] [--model_path /local/steer-tts]
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from steertts import SteerTTS
from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

SAMPLING_RATE = 24_000


def main(args):
    set_seed(args.seed)
    tts = SteerTTS.from_pretrained(args.model_path or args.model_id, device=args.device, flow=not args.no_flow)
    voice = args.voice or "voice_1361"  # the voice the published leaderboard numbers used
    print(f"Loaded {tts} voice={voice} voice_clone={args.voice_clone}")
    print_model_size(tts.model, tts.vocos, tts.flow, tts.pflow if args.voice_clone else None)
    is_cuda = str(tts.dev).startswith("cuda")

    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir
    prompt_dir = os.path.join(output_dir, "prompts")
    if args.voice_clone:
        os.makedirs(prompt_dir, exist_ok=True)

    def write_prompt(sample_id, prompt_audio, root):
        prel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
        ppath = os.path.join(root, prel)
        os.makedirs(os.path.dirname(ppath), exist_ok=True)
        if not os.path.exists(ppath):
            sf.write(ppath, np.asarray(prompt_audio["array"], dtype=np.float32), prompt_audio["sampling_rate"])
        return prel, ppath

    def request(text, ref):
        return dict(text=text, ref_wav=ref) if ref else dict(text=text, voice=voice)

    def synth_one(text, ref=None):
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        audio = tts.synthesize(request(text, ref))
        if is_cuda:
            torch.cuda.synchronize()
        return audio, time.perf_counter() - start

    dataset = load_tts_dataset(args, extra_columns=("prompt_audio",) if args.voice_clone else ())

    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices
        import tempfile
        tmp = tempfile.mkdtemp(prefix="ttfa_prompts_")
        chunk_counts = []

        def _gen(job, early_stop):
            ref = write_prompt(job["id"][0], job["prompt_audio"][0], tmp)[1] if args.voice_clone else None
            first, chunks = None, []
            for c in tts.stream(request=request(job[args.text_column][0], ref)):
                if first is None:
                    first = time.perf_counter()
                chunks.append(c.audio)
                if early_stop:
                    break
            chunk_counts.append(len(chunks))
            return (np.concatenate(chunks) if chunks else np.zeros(1, np.float32)), SAMPLING_RATE, first

        columns = dataset.column_names
        run_probe(_gen, [{c: [dataset[i][c]] for c in columns} for i in sample_indices(args.ttfa_probe, len(dataset))],
                  model_id=args.model_id, dataset_path=args.dataset_path, dataset_config=args.dataset, split=args.split,
                  out_dir=model_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
                  extra={"device": args.device, "note": "batch 1, SteerTTS.stream() sentence chunks",
                         "chunks_per_utterance": chunk_counts})
        return

    done_entries = load_done_entries(paths.manifest_path, args.resume)
    manifest_file = open_manifest(paths.manifest_path, done_entries)
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)

    def gen_batch(batch):
        out = []
        for k, text in enumerate(batch[args.text_column]):
            ref = write_prompt(batch["id"][k], batch["prompt_audio"][k], model_dir)[1] if args.voice_clone else None
            out.append(synth_one(text, ref))
        return out

    warm_up(gen_batch, dataset, starts, 1, args.warmup_steps)
    set_seed(args.seed)
    for i in tqdm(starts, desc="Generating"):
        row = dataset[i]
        prel = None
        ref = None
        if args.voice_clone:
            prel, ref = write_prompt(row["id"], row["prompt_audio"], model_dir)  # I/O outside the timed region
        audio, elapsed = synth_one(row[args.text_column], ref)
        rel_path = wav_rel_path(dataset_dir_name, row["id"])
        sf.write(os.path.join(model_dir, rel_path), audio, SAMPLING_RATE)
        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, elapsed, row[args.text_column],
                                                  prel if args.voice_clone else None))
        manifest_file.flush()
    manifest_file.close()
    print("Results saved at path:", os.path.abspath(paths.manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="WhissleAI/steer-tts",
                        help="Name used for the results folder; also the HF repo loaded unless --model_path is set.")
    parser.add_argument("--model_path", type=str, default=None, help="Local model directory (overrides --model_id).")
    parser.add_argument("--voice", type=str, default=None, help="Built-in voice (default: the model's default_voice).")
    parser.add_argument("--no_flow", action="store_true", help="Skip the flow refiner (faster, slightly hazier).")
    add_common_args(parser, voice_clone=False)
    args = parser.parse_args()
    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)
    main(args)
