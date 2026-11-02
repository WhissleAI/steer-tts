# Host build of the SteerTTS Nano runtime (portable C kernels): library, CLI and tests.
#   cmake -S host -B build-host && cmake --build build-host -j && ctest --test-dir build-host
cmake_minimum_required(VERSION 3.16)
project(steertts_nano_host C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(NANO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../components/steertts_nano)
file(GLOB NANO_SRC ${NANO_DIR}/src/*.c ${NANO_DIR}/front/*.c)
add_library(steertts_nano STATIC ${NANO_SRC})
target_include_directories(steertts_nano PUBLIC ${NANO_DIR}/include PRIVATE ${NANO_DIR}/src ${NANO_DIR}/front)
# results must not depend on the compiler: no FMA contraction, IEEE float32
target_compile_options(steertts_nano PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra -Wno-unused-parameter -Wno-format-truncation)
target_compile_definitions(steertts_nano PRIVATE _POSIX_C_SOURCE=200809L)
target_link_libraries(steertts_nano PUBLIC m)

add_executable(nano_host nano_host.c)
target_compile_options(nano_host PRIVATE -ffp-contract=off -Wall)
target_compile_definitions(nano_host PRIVATE _POSIX_C_SOURCE=200809L)
target_link_libraries(nano_host steertts_nano)

add_executable(test_kernels test_kernels.c)
target_include_directories(test_kernels PRIVATE ${NANO_DIR}/src)
target_compile_options(test_kernels PRIVATE -ffp-contract=off -Wall)
target_link_libraries(test_kernels steertts_nano)

add_executable(test_front front/test_front.c)
target_compile_options(test_front PRIVATE -ffp-contract=off -Wall -Wno-format-truncation)
target_link_libraries(test_front steertts_nano)

enable_testing()
add_test(NAME kernels COMMAND test_kernels)
# front end conformance: needs front.bin and the converted golden file (see docs/FRONT_END.md)
if(DEFINED NANO_FRONT_BIN AND DEFINED NANO_FRONT_GOLDEN)
  add_test(NAME front COMMAND test_front ${NANO_FRONT_BIN} ${NANO_FRONT_GOLDEN})
endif()
