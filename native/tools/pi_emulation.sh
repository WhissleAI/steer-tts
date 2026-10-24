#!/usr/bin/env bash
# Emulated Raspberry Pi test of the native runtime, without root:
#   tools/pi_emulation.sh <work_dir> <package_dir>
# 1. qemu-user (static, for running) and qemu-user (dynamic, for the instruction-count plugin) from the distro archive,
#    unpacked with dpkg -x (Debian / Ubuntu hosts); liburing for the dynamic build
# 2. a Debian bookworm arm64 root filesystem (Raspberry Pi OS 64-bit's base), exported with `docker create` + `docker
#    export` (no ARM container is executed, so no binfmt registration is needed)
# 3. Zig as the aarch64 cross compiler (glibc 2.36), the official onnxruntime-linux-aarch64 release
# 4. an emulated Pi 4 (4x Cortex-A72, 1.8 GHz) and Pi 5 (4x Cortex-A76, 2.4 GHz) /proc/cpuinfo + CPU topology inside the
#    root filesystem: QEMU's -L prefix redirects the guest's opens there, so ONNX Runtime's cpuinfo sees a Pi
#    instead of the host (without it, ONNX Runtime crashes at load parsing the host's x86 topology)
# 5. engine_parity on both CPUs, then guest instruction counts (load only, and load + 5 sentences) per CPU
set -euo pipefail
W=$(realpath -m "$1"); PKG=$(realpath "$2")
HERE=$(cd "$(dirname "$0")/.." && pwd)
ORT_VER=${ORT_VER:-1.31.0}
ZIG_VER=${ZIG_VER:-0.14.1}
mkdir -p "$W" && cd "$W"

if [ ! -x qemu/usr/bin/qemu-aarch64-static ]; then
  apt-get download qemu-user-static qemu-user liburing2
  dpkg -x qemu-user-static_*.deb qemu && dpkg -x qemu-user_*.deb qemu_dyn && dpkg -x liburing2_*.deb uring
fi
export LD_LIBRARY_PATH="$W/uring/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ ! -d rootfs/usr ]; then
  cid=$(docker create --platform linux/arm64 debian:bookworm-slim)
  mkdir -p rootfs && docker export "$cid" | tar -x -C rootfs && docker rm "$cid" > /dev/null
  cp -a rootfs rootfs_pi5
fi

fake_pi() {  # root part features max_khz model
  local R=$1 part=$2 feat=$3 khz=$4 model=$5 variant=0x0 rev=3
  [ "$part" = 0xd0b ] && { variant=0x4; rev=1; }
  rm -rf "$R/proc/cpuinfo" "$R/sys/devices/system/cpu"; mkdir -p "$R/proc" "$R/sys/devices/system/cpu"
  : > "$R/proc/cpuinfo"
  for i in 0 1 2 3; do
    printf 'processor\t: %s\nBogoMIPS\t: 108.00\nFeatures\t: %s\nCPU implementer\t: 0x41\nCPU architecture: 8\nCPU variant\t: %s\nCPU part\t: %s\nCPU revision\t: %s\n\n' \
      "$i" "$feat" "$variant" "$part" "$rev" >> "$R/proc/cpuinfo"
  done
  printf 'Hardware\t: BCM2835\nModel\t\t: %s\n' "$model" >> "$R/proc/cpuinfo"
  local D=$R/sys/devices/system/cpu
  echo 0-3 > "$D/possible"; echo 0-3 > "$D/present"; echo 0-3 > "$D/online"
  for i in 0 1 2 3; do
    mkdir -p "$D/cpu$i/cpufreq" "$D/cpu$i/topology"
    echo "$khz" > "$D/cpu$i/cpufreq/cpuinfo_max_freq"; echo "$khz" > "$D/cpu$i/cpufreq/cpuinfo_min_freq"
    echo 0 > "$D/cpu$i/topology/physical_package_id"; echo "$i" > "$D/cpu$i/topology/core_id"
    for f in core_siblings_list package_cpus_list; do echo 0-3 > "$D/cpu$i/topology/$f"; done
    for f in thread_siblings_list core_cpus_list; do echo "$i" > "$D/cpu$i/topology/$f"; done
  done
}
fake_pi rootfs 0xd08 "fp asimd evtstrm crc32 cpuid" 1800000 "Raspberry Pi 4 Model B (emulated)"
fake_pi rootfs_pi5 0xd0b "fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp" \
  2400000 "Raspberry Pi 5 Model B (emulated)"

[ -x "zig-x86_64-linux-$ZIG_VER/zig" ] || { curl -sL "https://ziglang.org/download/$ZIG_VER/zig-x86_64-linux-$ZIG_VER.tar.xz" | tar -xJ; }
ZIG=$W/zig-x86_64-linux-$ZIG_VER/zig
mkdir -p bin
for t in "acc:cc -target aarch64-linux-gnu.2.36" "acxx:c++ -target aarch64-linux-gnu.2.36" "ar:ar" "ranlib:ranlib"; do
  printf '#!/bin/sh\nexec %s %s "$@"\n' "$ZIG" "${t#*:}" > "bin/${t%%:*}"; chmod +x "bin/${t%%:*}"
done
ORT=$W/onnxruntime-linux-aarch64-$ORT_VER
[ -d "$ORT" ] || curl -sL "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VER/onnxruntime-linux-aarch64-$ORT_VER.tgz" | tar -xz
CC=$W/bin/acc CXX=$W/bin/acxx cmake -S "$HERE" -B build_arm -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_AR="$W/bin/ar" -DCMAKE_RANLIB="$W/bin/ranlib" -DONNXRUNTIME_ROOT="$ORT" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build_arm -j8 --target steertts-edge engine_parity > /dev/null

# instruction-count plugin (QEMU 6.2 plugin API)
if [ ! -f plugin/libicount.so ]; then
  mkdir -p plugin
  curl -sL -o plugin/qemu-plugin.h https://raw.githubusercontent.com/qemu/qemu/v6.2.0/include/qemu/qemu-plugin.h
  cat > plugin/icount.c <<'EOF'
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <glib.h>
#include "qemu-plugin.h"
QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;
static uint64_t insns;
static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
    qemu_plugin_register_vcpu_tb_exec_inline(tb, QEMU_PLUGIN_INLINE_ADD_U64, &insns, qemu_plugin_tb_n_insns(tb));
}
static void at_exit(qemu_plugin_id_t id, void *p) {
    FILE *f = fopen(getenv("ICOUNT_OUT") ? getenv("ICOUNT_OUT") : "/dev/stderr", "w");
    fprintf(f, "%" PRIu64 "\n", insns); fclose(f);
}
QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc, char **argv) {
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
EOF
  gcc -shared -fPIC -O2 $(pkg-config --cflags glib-2.0) -o plugin/libicount.so plugin/icount.c
fi

TEXT="Hi, thanks for calling. I can help you reschedule your appointment today. The weather was cold, but the view from the top was worth every single step. Are you coming with us tonight? I really thought it would work this time. Please keep this between the two of us, and don't tell anyone else about it until Friday."
for cpu in cortex-a72 max,sve=off; do
  R=rootfs; [ "$cpu" = cortex-a72 ] || R=rootfs_pi5
  Q=(qemu/usr/bin/qemu-aarch64-static -cpu "$cpu" -L "$W/$R" -E "LD_LIBRARY_PATH=$ORT/lib")
  echo "== $cpu: parity with the Python ONNX runtime"
  "${Q[@]}" build_arm/engine_parity "$PKG" "$HERE/tests/engine_golden" | tail -1
  P=(qemu_dyn/usr/bin/qemu-aarch64 -cpu "$cpu" -L "$W/$R" -E "LD_LIBRARY_PATH=$ORT/lib" -plugin plugin/libicount.so)
  ICOUNT_OUT=ic_load.txt "${P[@]}" build_arm/steertts-edge --model "$PKG" --list tags > /dev/null
  ICOUNT_OUT=ic_all.txt "${P[@]}" build_arm/steertts-edge --model "$PKG" "$TEXT" --stream --threads 1 -o "arm_$cpu.wav" 2> /dev/null
  samples=$(( ($(stat -c %s "arm_$cpu.wav") - 44) / 2 ))
  echo "$cpu: load $(cat ic_load.txt) insns, load+synth $(cat ic_all.txt) insns, audio $samples samples" \
       "-> $(python3 -c "print(f'{($(cat ic_all.txt)-$(cat ic_load.txt))/($samples/24000)/1e9:.2f}')") G insns per second of audio"
done
