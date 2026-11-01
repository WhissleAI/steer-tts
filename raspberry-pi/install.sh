#!/bin/sh
# Build and install SteerTTS (C library, steertts CLI, steertts-pi server) on a Raspberry Pi, and download the edge
# model. Run it from a clone of the repo, on the Pi (64-bit Raspberry Pi OS, Ubuntu for Raspberry Pi, or another aarch64
# Linux with apt):
#
#   git clone https://github.com/WhissleAI/steer-tts && cd steer-tts
#   raspberry-pi/install.sh            # build (tuned for this Pi), test, install to ~/.local, download the model
#   raspberry-pi/install.sh --service  # ... and run steertts-pi at boot (systemd, port 8080, localhost only)
#
# Options (environment): PREFIX (default ~/.local), MODELS (default $PREFIX/share/steertts/models), CPU (override the
# detected -mcpu: cortex-a72, cortex-a76, native or "" for portable), HF_TOKEN (if the model repo asks you to accept
# its licence: accept it at https://huggingface.co/WhissleAI/steer-tts-edge, then export a read token).
# Flags: --service, --no-test, --no-model.
set -eu
service=0 test=1 model=1
for a in "$@"; do
  case "$a" in
    --service) service=1 ;;
    --no-test) test=0 ;;
    --no-model) model=0 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
here=$(cd "$(dirname "$0")" && pwd)
c=$(dirname "$here")/c
PREFIX=${PREFIX:-$HOME/.local}
MODELS=${MODELS:-$PREFIX/share/steertts/models}

arch=$(uname -m)
board=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "unknown board")
if [ -z "${CPU+x}" ]; then
  case "$board" in
    *"Raspberry Pi 5"*|*"Raspberry Pi 500"*|*"Compute Module 5"*) CPU=cortex-a76 ;;
    *"Raspberry Pi 4"*|*"Raspberry Pi 400"*|*"Compute Module 4"*) CPU=cortex-a72 ;;
    *) CPU= ;;
  esac
fi
os=$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || uname -s)
echo "board: $board ($arch, $os); tuning: ${CPU:-portable}; install prefix: $PREFIX"
[ "$arch" = aarch64 ] || echo "note: not aarch64 ($arch): building a portable build (32-bit Raspberry Pi OS is not a target)"

# under-voltage makes a Pi throttle (and can corrupt an SD card): say so before building
if command -v vcgencmd >/dev/null 2>&1 && t=$(vcgencmd get_throttled 2>/dev/null | sed 's/.*=//') && [ -n "$t" ]; then
  if [ $((t & 0x10001)) -ne 0 ]; then
    echo "WARNING: under-voltage detected since boot (get_throttled=$t). Use the official supply (Pi 5: 27 W, 5 V 5 A;"
    echo "         Pi 4: 15 W, 5 V 3 A); from a laptop's USB port the Pi throttles under load."
  fi
fi

need=""
for t in cmake cc make curl; do command -v $t >/dev/null 2>&1 || need="$need $t"; done
if [ -n "$need" ]; then
  echo "installing build tools:$need"
  if command -v apt-get >/dev/null 2>&1; then   # Raspberry Pi OS, Ubuntu, Debian
    sudo apt-get update && sudo apt-get install -y cmake gcc make curl
  else
    echo "install$need with your package manager, then run this again" >&2
    exit 1
  fi
fi

build=$c/build-pi
cmake -S "$c" -B "$build" -DCMAKE_BUILD_TYPE=Release -DSTEERTTS_CPU="$CPU" >/dev/null
cmake --build "$build" -j"$(nproc)"

if [ $model = 1 ]; then
  if [ -f "$MODELS/steer-tts-edge/config.json" ]; then
    echo "model: $MODELS/steer-tts-edge (already there)"
  else
    sh "$c/tools/get_models.sh" edge "$MODELS"
  fi
fi

if [ $test = 1 ]; then
  if [ -f "$MODELS/steer-tts-edge/config.json" ]; then
    (cd "$build" && STEERTTS_EDGE_MODEL="$MODELS/steer-tts-edge" ctest --output-on-failure -E "test_front")
  else
    (cd "$build" && ctest --output-on-failure -E "test_front")
  fi
fi

cmake --install "$build" --prefix "$PREFIX" >/dev/null
echo "installed: $PREFIX/bin/steertts, $PREFIX/bin/steertts-pi, $PREFIX/lib/libsteertts.*, $PREFIX/include/steertts.h"

if [ $service = 1 ]; then
  unit=/etc/systemd/system/steertts-pi.service
  sed -e "s|@BIN@|$PREFIX/bin/steertts-pi|" -e "s|@MODEL@|$MODELS/steer-tts-edge|" -e "s|@USER@|$(id -un)|" \
      "$here/server/steertts-pi.service.in" | sudo tee "$unit" >/dev/null
  sudo systemctl daemon-reload && sudo systemctl enable --now steertts-pi
  echo "service: steertts-pi on http://127.0.0.1:8080 (edit $unit for --host 0.0.0.0 / threads)"
fi
cat <<EOT

Try it:
  $PREFIX/bin/steertts --model $MODELS/steer-tts-edge "Hello from a Raspberry Pi." -o hello.wav
  $PREFIX/bin/steertts-pi --model $MODELS/steer-tts-edge &
  curl -s 'localhost:8080/synthesize?text=Hello+from+a+Raspberry+Pi.' -o hello.wav
  $here/bench.sh   # speed and memory on this Pi (--sustained 10 for thermals)
  examples: $here/examples (cc hello.c \$(pkg-config --cflags --libs steertts) -o hello;
            pkg-config needs PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig for a ~/.local install)
EOT
