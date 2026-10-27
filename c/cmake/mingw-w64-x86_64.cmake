# Cross-compiling for Windows x64 from macOS or Linux with MinGW-w64 (brew install mingw-w64):
#   cmake -B build-win64 -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-x86_64.cmake && cmake --build build-win64 -j
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_RC_COMPILER x86_64-w64-mingw32-windres)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# self-contained binaries: no libgcc_s / libwinpthread DLLs to ship next to them
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-static-libgcc -static")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static-libgcc -static")
