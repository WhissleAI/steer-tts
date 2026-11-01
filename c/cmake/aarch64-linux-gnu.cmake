# Cross-compiling for 64-bit Raspberry Pi OS (or any aarch64 Linux) with the GNU cross compiler:
#   Debian / Ubuntu: sudo apt install crossbuild-essential-arm64   (or use pi/Dockerfile.cross)
#   cmake --preset pi4-cross   (or: cmake -B build-pi -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu.cmake
#                                       -DSTEERTTS_CPU=cortex-a72)
# Building on Debian bookworm keeps the binaries on the glibc of Raspberry Pi OS bookworm.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(ST_CROSS_PREFIX aarch64-linux-gnu- CACHE STRING "cross compiler prefix")
set(CMAKE_C_COMPILER ${ST_CROSS_PREFIX}gcc)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
