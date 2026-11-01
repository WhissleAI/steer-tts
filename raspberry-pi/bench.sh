#!/bin/sh
# Speed, memory and thermals of the C library on a Raspberry Pi (or any Linux machine).
#
#   raspberry-pi/bench.sh               # thread sweep: steertts --bench at 1, 2, 3 and 4 threads
#   raspberry-pi/bench.sh --sustained 10   # then 10 minutes of continuous rendering, logging the SoC temperature,
#                                       # clock and throttling flags every 5 s (vcgencmd), and the RTF over time
#
# Environment: STEERTTS_BIN (default: build-pi*/steertts or build/steertts), STEERTTS_MODEL (default
# models/steer-tts-edge), THREADS (sustained run, default 4), SWEEP=0 (skip the sweep), OUT (log directory, default
# bench-<date>).
# Nothing is played; WAVs are not written. Temperature, clock and throttling come from vcgencmd when it works, else
# from /sys (thermal_zone0, cpufreq, the firmware's get_throttled).
set -eu
cd "$(dirname "$0")/../c"
minutes=0
[ "${1:-}" = "--sustained" ] && minutes=${2:-10}
bin=${STEERTTS_BIN:-}
if [ -z "$bin" ]; then
  for b in build-pi5/steertts build-pi4/steertts build/steertts; do [ -x "$b" ] && { bin=$b; break; }; done
fi
[ -n "$bin" ] && [ -x "$bin" ] || { echo "no steertts binary: build first (raspberry-pi/README.md)" >&2; exit 1; }
model=${STEERTTS_MODEL:-models/steer-tts-edge}
out=${OUT:-bench-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out"
have_vc=0
command -v vcgencmd >/dev/null 2>&1 && vcgencmd measure_temp >/dev/null 2>&1 && have_vc=1

# Pi 5: the PMIC measures every rail; the sum of volts x amps is the board's power draw (excluding USB devices)
power() {
  vcgencmd pmic_read_adc 2>/dev/null | awk '
    $2 ~ /^current/ { k = substr($1, 1, length($1) - 2); x = $2; sub(/.*=/, "", x); sub(/A$/, "", x); a[k] = x }
    $2 ~ /^volt/ { k = substr($1, 1, length($1) - 2); x = $2; sub(/.*=/, "", x); sub(/V$/, "", x); v[k] = x }
    END { for (k in a) if (k in v) w += a[k] * v[k]; if (w > 0) printf "%.2f", w; else printf "n/a" }'
}

soc() {   # temp_C arm_MHz throttled_hex watts
  if [ $have_vc = 1 ]; then
    t=$(vcgencmd measure_temp | sed 's/[^0-9.]//g')
    f=$(vcgencmd measure_clock arm | sed 's/.*=//')
    th=$(vcgencmd get_throttled | sed 's/.*=//')
    echo "$t $((f / 1000000)) $th $(power)"
  elif [ -r /sys/class/thermal/thermal_zone0/temp ]; then
    # no vcgencmd (e.g. Ubuntu without libraspberrypi-bin): sysfs temperature and clock; the Raspberry Pi kernels
    # (Raspberry Pi OS and Ubuntu's raspi kernel) also expose the firmware's throttled flags
    th=n/a
    for f in /sys/devices/platform/soc/soc:firmware/get_throttled /sys/devices/platform/soc:firmware/get_throttled; do
      [ -r "$f" ] && { th=0x$(cat "$f"); break; }
    done
    echo "$(awk '{printf "%.1f", $1 / 1000}' /sys/class/thermal/thermal_zone0/temp) $(($(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 0) / 1000)) $th n/a"
  else
    echo "n/a n/a n/a n/a"
  fi
}

# get_throttled bits: 0 under-voltage now, 1 ARM clock capped now, 2 throttled now, 3 soft temperature limit now;
# 16-19 the same "has occurred since boot". Numbers taken while any of 0-3 is set are power- or heat-limited.
throttled() {
  if [ $have_vc = 1 ]; then vcgencmd get_throttled | sed 's/.*=//'
  else
    for f in /sys/devices/platform/soc/soc:firmware/get_throttled /sys/devices/platform/soc:firmware/get_throttled; do
      [ -r "$f" ] && { echo "0x$(cat "$f")"; return; }
    done
    echo n/a
  fi
}
warn_power() {   # $1: label
  t=$(throttled)
  [ "$t" = n/a ] && return 0
  if [ $((t & 0x1)) -ne 0 ] || [ $((t & 0x10000)) -ne 0 ]; then
    echo "WARNING ($1): under-voltage detected (get_throttled=$t). Use the official 27 W (5 V 5 A) supply for a Pi 5 or" >&2
    echo "         15 W (5 V 3 A) for a Pi 4; a laptop's USB-C port is not enough under load. Speeds measured now are" >&2
    echo "         power-limited." >&2
  fi
  if [ $((t & 0x8)) -ne 0 ] || [ $((t & 0x6)) -ne 0 ]; then
    echo "WARNING ($1): the CPU is throttled or at its temperature limit now (get_throttled=$t): add a heatsink / fan" >&2
  fi
}

{
  echo "# $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || uname -m), $(uname -r), $(nproc) CPUs"
  echo "# binary: $bin ($("$bin" --version)), model: $model"
  echo "# soc before: $(soc)"
  echo "# throttled before: $(throttled)"
} | tee "$out/info.txt"
warn_power "before"

[ "${SWEEP:-1}" = 1 ] && echo "threads sweep (steertts --bench: 20 sentences, 66 s of audio):"
for t in 1 2 3 4; do
  [ "${SWEEP:-1}" = 1 ] || break
  [ "$t" -le "$(nproc)" ] || continue
  before=$(throttled)
  r=$("$bin" --model "$model" --bench-json --threads "$t" --first-chunk-ms 300)
  echo "$r" | sed "s/}\$/, \"throttled_before\": \"$before\", \"throttled_after\": \"$(throttled)\"}/" | tee -a "$out/sweep.jsonl"
done
[ "${SWEEP:-1}" = 1 ] && warn_power "after the sweep"

if [ "$minutes" -gt 0 ]; then
  th=${THREADS:-4}
  echo "sustained: $minutes min at $th threads -> $out/sustained.tsv, $out/soc.tsv"
  printf 'elapsed_s\ttemp_c\tarm_mhz\tthrottled\twatts\n' > "$out/soc.tsv"
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
  awk -F'\t' 'NR > 1 && $2 != "n/a" { if ($2 > m) m = $2; c = $3; t = $4; if ($5 != "n/a") { w += $5; nw++ } }
    END { printf "SoC: max %.1f C, last clock %s MHz, throttled %s, mean power %s W\n", m, c, t, nw ? sprintf("%.2f", w / nw) : "n/a" }' "$out/soc.tsv"
  echo "# soc after: $(soc), throttled $(throttled)" | tee -a "$out/info.txt"
  warn_power "after the sustained run"
fi
