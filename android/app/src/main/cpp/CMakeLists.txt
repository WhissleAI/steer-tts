# The app's JNI library: steertts_jni.c + the SteerTTS C library built from source (steer-tts/c), linked statically.
cmake_minimum_required(VERSION 3.13)
project(steertts_android C)

set(STEERTTS_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(STEERTTS_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(STEERTTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(STEERTTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../c steertts)
# optimised in debug builds too: an -O0 engine is ~10x slower and misrepresents the phone
target_compile_options(steertts_obj PRIVATE -O2)

add_library(steertts_jni SHARED steertts_jni.c)
target_compile_options(steertts_jni PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(steertts_jni steertts_static log)
# 16 KB page alignment (required for apps targeting Android 15+ devices with 16 KB pages)
target_link_options(steertts_jni PRIVATE -Wl,-z,max-page-size=16384)
