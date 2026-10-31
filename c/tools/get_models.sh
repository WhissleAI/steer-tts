#!/bin/sh
# Download the model files the C library reads, with curl only (no Python):
#
#   c/tools/get_models.sh edge            -> models/steer-tts-edge   (12 MB + 2.6 MB front-end assets + 0.5 MB director)
#   c/tools/get_models.sh v1 [int8|fp32]  -> models/steer-tts        (int8: 52 MB + helpers; fp32: 215 MB + helpers)
#   c/tools/get_models.sh edge /opt/tts   (any destination directory; default ./models)
#
# The Hugging Face repos may ask visitors to accept their licence (CC BY-NC 4.0) once: accept it on the model page, then
# export HF_TOKEN=<a read token from huggingface.co/settings/tokens>. v1 voice cloning needs the WavLM speaker encoder
# (115 MB int8 / 401 MB fp32, licence CC BY-SA 3.0): add CLONE=1 to download it too.
set -eu
what=${1:-edge}
case "$what" in
  edge) prec=""; dest=${2:-models} ;;
  v1) case "${2:-int8}" in fp32|int8) prec=${2:-int8}; dest=${3:-models} ;; *) prec=int8; dest=$2 ;; esac ;;
  *) echo "usage: $0 edge|v1 [int8|fp32] [dest]" >&2; exit 2 ;;
esac
auth=""
[ -n "${HF_TOKEN:-}" ] && auth="Authorization: Bearer $HF_TOKEN"

get() {   # repo dir file...
  repo=$1; dir=$2; shift 2
  for f in "$@"; do
    mkdir -p "$dir/$(dirname "$f")"
    printf '  %s\n' "$f"
    if ! curl -fL --retry 3 -s ${auth:+-H "$auth"} -o "$dir/$f" "https://huggingface.co/$repo/resolve/main/$f"; then
      echo "download of $repo/$f failed: if the repo asks for licence acceptance, accept it on" >&2
      echo "https://huggingface.co/$repo and export HF_TOKEN=<read token>" >&2
      exit 1
    fi
  done
}

front="native/front/gru.bin native/front/hetero.tsv native/front/lexicon.bin native/front/oov.bin native/front/SYMBOLS.txt
       native/front/unicode.bin"
if [ "$what" = edge ]; then
  d=$dest/steer-tts-edge
  echo "WhissleAI/steer-tts-edge -> $d"
  get WhissleAI/steer-tts-edge "$d" config.json voices.json acoustic.safetensors vocoder.safetensors voices.safetensors $front
  # the director (docs/director.md, 0.5 MB): optional, so a missing file only warns
  printf '  %s\n' director.bin
  curl -fL --retry 3 -s ${auth:+-H "$auth"} -o "$d/director.bin" "https://huggingface.co/WhissleAI/steer-tts-edge/resolve/main/director.bin" \
    || { rm -f "$d/director.bin"; echo "  (director.bin not available: --perform and steertts_director_* need it)" >&2; }
else
  d=$dest/steer-tts
  echo "WhissleAI/steer-tts ($prec) -> $d"
  get WhissleAI/steer-tts "$d" config.json control_stats.json voices.json request.schema.json $front native/minilm_vocab.txt
  if [ "$prec" = int8 ]; then
    get WhissleAI/steer-tts "$d" int8/model.safetensors int8/flow.safetensors int8/pflow.safetensors \
        int8/prompt_adapter.safetensors int8/speaker_vectors.safetensors native/int8/vocos.safetensors \
        native/int8/minilm.safetensors native/NOTICE
    [ "${CLONE:-0}" = 1 ] && get WhissleAI/steer-tts "$d" native/int8/wavlm_sv.safetensors
  else
    get WhissleAI/steer-tts "$d" model.safetensors flow.safetensors pflow.safetensors prompt_adapter.safetensors \
        speaker_vectors.safetensors native/vocos.safetensors native/minilm.safetensors native/NOTICE
    [ "${CLONE:-0}" = 1 ] && get WhissleAI/steer-tts "$d" native/wavlm_sv.safetensors
  fi
  [ "$prec" = int8 ] && echo "use it with: steertts --model $d --precision int8 ..."
  [ "${CLONE:-0}" = 1 ] || echo "(voice cloning: re-run with CLONE=1 to add the WavLM speaker encoder)"
fi
echo "done: $d"
