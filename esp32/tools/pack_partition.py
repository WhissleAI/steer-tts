#!/usr/bin/env python3
"""Pack nano.bin and front.bin into one image of the `steertts` flash partition.

    python tools/pack_partition.py nano.bin front.bin -o steertts_partition.bin

nano.bin goes at offset 0 and front.bin at 0x400000 (NANO_FRONT_PARTITION_OFFSET), the layout the firmware maps.
Write the image with parttool (write_partition --partition-name steertts) or esptool at the partition offset."""
import argparse
import sys

FRONT_OFFSET = 0x400000
PARTITION_SIZE = 0x6F0000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nano_bin")
    ap.add_argument("front_bin")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    nano = open(a.nano_bin, "rb").read()
    front = open(a.front_bin, "rb").read()
    if nano[:4] != b"STNB" or front[:4] != b"STFB":
        sys.exit("expected a nano.bin and a front.bin")
    if len(nano) > FRONT_OFFSET:
        sys.exit(f"nano.bin is {len(nano)} bytes, more than the {FRONT_OFFSET} reserved for it")
    if FRONT_OFFSET + len(front) > PARTITION_SIZE:
        sys.exit("front.bin does not fit the partition")
    img = nano + b"\xff" * (FRONT_OFFSET - len(nano)) + front
    open(a.out, "wb").write(img)
    print(f"{a.out}: {len(img)} bytes (nano.bin {len(nano)}, front.bin {len(front)} at 0x{FRONT_OFFSET:x})")


if __name__ == "__main__":
    main()
