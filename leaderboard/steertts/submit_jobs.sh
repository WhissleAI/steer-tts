#!/bin/bash
# HF Jobs TTS eval — SteerTTS backend (generate → transcribe → SIM* → score; *voice-clone only).
# Usage: HF_TOKEN=hf_... [VOICE_CLONE=true] bash steertts/submit_jobs.sh ["WhissleAI/steer-tts" ...]
# English only: SteerTTS has an English (ARPAbet) front end, so only the `en` splits are configured.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

TTS_SPACE="${TTS_SPACE:-WhissleAI/evals-steertts}"   # public Docker Space built from steertts/Dockerfile (TO CREATE)
RUN_EVAL_INJECT="$(inject_run_eval)"
SUPPORTS_VOICE_CLONE=true
supports_language() { [[ "$1" == "en" ]]; }

generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/steertts && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
)

MODEL_CONFIGS=("WhissleAI/steer-tts")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
