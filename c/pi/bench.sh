#!/bin/sh
# Speed, memory and thermals of the C library on a Raspberry Pi (or any Linux machine).
#
#   c/pi/bench.sh                       # thread sweep: steertts --bench at 1, 2, 3 and 4 threads
#   c/pi/bench.sh --sustained 10        # then 10 minutes of continuous rendering, logging the SoC temperature,
#                                       # clock and throttling flags every 5 s (vcgencmd), and the RTF over time
#
# Environment: STEERTTS_BIN (default: build-pi*/steertts or build/steertts), STEERTTS_MODEL (default
# models/steer-tts-edge), THREADS (sustained run, default 4), OUT (log directory, default bench-<date>).
# Nothing is played; WAVs are not written.
set -eu
cd "$(dirname "$0")/.."
minutes=0
[ "${1:-}" = "--sustained" ] && minutes=${2:-10}
bin=${STEERTTS_BIN:-}
if [ -z "$bin" ]; then
  for b in build-pi5/steertts build-pi4/steertts build/steertts; do [ -x "$b" ] && { bin=$b; break; }; done
fi
[ -n "$bin" ] && [ -x "$bin" ] || { echo "no steertts binary: build first (see docs/raspberry-pi.md)" >&2; exit 1; }
model=${STEERTTS_MODEL:-models/steer-tts-edge}
out=${OUT:-bench-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out"
have_vc=0
command -v vcgencmd >/dev/null 2>&1 && vcgencmd measure_temp >/dev/null 2>&1 && have_vc=1

soc() {   # temp_C arm_MHz throttled_hex
  if [ $have_vc = 1 ]; then
    t=$(vcgencmd measure_temp | sed 's/[^0-9.]//g')
    f=$(vcgencmd measure_clock arm | sed 's/.*=//')
    th=$(vcgencmd get_throttled | sed 's/.*=//')
    echo "$t $((f / 1000000)) $th"
  elif [ -r /sys/class/thermal/thermal_zone0/temp ]; then
    echo "$(($(cat /sys/class/thermal/thermal_zone0/temp) / 1000)) $(($(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 0) / 1000)) n/a"
  else
    echo "n/a n/a n/a"
  fi
}

{
  echo "# $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || uname -m), $(uname -r), $(nproc) CPUs"
  echo "# binary: $bin ($("$bin" --version)), model: $model"
  echo "# soc before: $(soc)"
} | tee "$out/info.txt"

echo "threads sweep (steertts --bench: 20 sentences, 66 s of audio):"
for t in 1 2 3 4; do
  [ "$t" -le "$(nproc)" ] || continue
  "$bin" --model "$model" --bench-json --threads "$t" --first-chunk-ms 300 | tee -a "$out/sweep.jsonl"
done

if [ "$minutes" -gt 0 ]; then
  th=${THREADS:-4}
  echo "sustained: $minutes min at $th threads -> $out/sustained.tsv, $out/soc.tsv"
  printf 'elapsed_s\ttemp_c\tarm_mhz\tthrottled\n' > "$out/soc.tsv"
  printf 'elapsed_s\trtf\n' > "$out/sustained.tsv"
  start=$(date +%s)
  end=$((start + minutes * 60))
  ( while [ "$(date +%s)" -lt "$end" ]; do
      printf '%s\t%s\n' $(($(date +%s) - start)) "$(soc | tr ' ' '\t')" >> "$out/soc.tsv"; sleep 5
    done ) &
  mon=$!
  while [ "$(date +%s)" -lt "$end" ]; do
    r=$("$bin" --model "$model" --bench-json --threads "$th" | sed 's/.*"rtf": \([0-9.]*\).*/\1/')
    printf '%s\t%s\n' $(($(date +%s) - start)) "$r" >> "$out/sustained.tsv"
  done
  wait $mon 2>/dev/null || true
  awk -F'\t' 'NR > 1 { n++; s += $2; if (n == 1) f = $2; l = $2 } END { printf "RTF first %.4f, last %.4f, mean %.4f over %d runs\n", f, l, s / n, n }' "$out/sustained.tsv"
  awk -F'\t' 'NR > 1 && $2 != "n/a" { if ($2 > m) m = $2; c = $3; t = $4 } END { printf "SoC: max %.1f C, last clock %s MHz, throttled %s\n", m, c, t }' "$out/soc.tsv"
  echo "# soc after: $(soc)" | tee -a "$out/info.txt"
fi
