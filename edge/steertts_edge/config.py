"""Audio / control constants. They must match the model: the same mel front end as charactr/vocos-mel-24khz."""
SR = 24000
N_FFT = 1024
HOP = 256
N_MELS = 100
FPS = SR / HOP  # 93.75 mel frames per second
PITCH_SCALE = 3.0  # predicted token pitch is semitones re the speaker median / 3; word_shift pitch is given in st

# Utterance-level controls. Each is a SPEAKER-RELATIVE z-score: 0 = this voice's own neutral speech, +-1 = one
# pooled within-speaker standard deviation (pooled_sd in control_stats.json gives the raw unit per dim).
CONTROL_DIMS = [
    "rate",      # articulation rate, phones per second of speech (pauses excluded)
    "loud",      # mean RMS level of speech frames, dB
    "f0_med",    # median F0, semitones
    "f0_iqr",    # F0 interquartile range, semitones (pitch range / liveliness)
    "f0_end",    # F0 slope over the final 300 ms of voicing, st/s (+ rising, question-like)
    "pause",     # share of the speech span that is silent pauses >= 100 ms
    "tilt",      # alpha ratio: energy 1-5 kHz vs 50 Hz-1 kHz, dB (+ brighter / pressed)
    "voiced",    # voiced share of speech frames (whisper -> low)
    "flat",      # spectral flatness of speech frames, log (breathy / whisper -> high)
    "centroid",  # spectral centroid of speech frames, log Hz (brightness)
    "cpps",      # smoothed cepstral peak prominence, dB (periodicity; breathy / tense -> low)
    "h1h2",      # H1-H2, dB (breathy -> high)
]
CONTROL_RANGE = (-5.0, 5.0)  # accepted request range; values beyond +-3 are outside the well-tested range

DEFAULT_VOICE = "voice_1688"  # v1.0.1 default (UTMOS sweep on non-test sentences); v1.0.0 and the leaderboard used voice_1361
