# Stage-1 image for the Open TTS Leaderboard (a public Docker Space, e.g. WhissleAI/evals-steertts).
# The eval scripts are injected at runtime; this only installs the model's dependencies.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 git && rm -rf /var/lib/apt/lists/*
# torchaudio must match the image's torch (vocos imports it)
RUN pip install --no-cache-dir "torchaudio==2.7.1" datasets tqdm \
    "steertts[clone] @ git+https://huggingface.co/WhissleAI/steer-tts"
# warm the caches the first job would otherwise download (vocoder, speaker encoder, NLTK data)
RUN python -c "from vocos import Vocos; Vocos.from_pretrained('charactr/vocos-mel-24khz')" && \
    python -c "from steertts.text import to_phonemes; to_phonemes('warm up')" && \
    python -c "from transformers import WavLMForXVector; WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')"
WORKDIR /app
