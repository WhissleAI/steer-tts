#!/bin/sh
# Cross-compile libsteertts, steertts and steertts-pi for a Raspberry Pi in Docker (no ARM machine, no QEMU needed):
#
#   c/pi/cross-build.sh pi4        -> c/build-pi4-cross/   (Cortex-A72: Raspberry Pi 4, 400, CM4)
#   c/pi/cross-build.sh pi5        -> c/build-pi5-cross/   (Cortex-A76: Raspberry Pi 5, 500, CM5; not for a Pi 4)
#
# Then copy the build directory's steertts, steertts-pi (and test_* / libsteertts.so if you want them) to the Pi.
# Without Docker: sudo apt install crossbuild-essential-arm64 cmake, then cmake --preset pi4-cross && cmake --build
# --preset pi4-cross.
set -eu
target=${1:-pi4}
case "$target" in pi4|pi5) ;; *) echo "usage: $0 pi4|pi5" >&2; exit 2 ;; esac
here=$(cd "$(dirname "$0")" && pwd)
c=$(dirname "$here")
image=steertts-pi-cross:bookworm
docker image inspect "$image" >/dev/null 2>&1 || docker build -t "$image" -f "$here/Dockerfile.cross" "$here"
docker run --rm -u "$(id -u):$(id -g)" -v "$c:/src/c" -w /src/c "$image" \
  sh -c "cmake --preset $target-cross >/dev/null && cmake --build --preset $target-cross -j\$(nproc)"
echo "built: $c/build-$target-cross/steertts-pi, steertts, libsteertts.a"
