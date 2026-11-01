# Raspberry Pi

The edge model runs on a Raspberry Pi 4 or 5 through the [C library](c.md). On a Pi 5 it is 11× faster than real time on
one core and 34× on four, and needs 26 MB of memory.

Install, the C API example, the `steertts-pi` server and the measured numbers:
**[raspberry-pi/README.md](../raspberry-pi/README.md)**.

- Build presets: `cmake --preset pi4 | pi5 | pi4-cross | pi5-cross` in [`c/`](../c) (or `make PI=4 | PI=5`).
- A Pi 3 / Zero 2 W (Cortex-A53) is not a target. A Pi 5 needs a cooler for sustained use.
- [`native/`](../native) (C++ on ONNX Runtime) also builds on a Pi ([native.md](native.md#raspberry-pi-and-other-arm64-boards));
  the C library is lighter and faster.
