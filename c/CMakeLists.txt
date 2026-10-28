# SteerTTS stand-alone C library: libsteertts (static + shared), the steertts CLI, examples and tests.
# C11 + libm + threads, nothing else.
#
#   cmake -B build && cmake --build build -j && cmake --install build --prefix /usr/local
#   then: find_package(steertts) + target_link_libraries(app steertts::steertts), or pkg-config steertts
cmake_minimum_required(VERSION 3.13)
project(steertts VERSION 1.2.0 LANGUAGES C)

option(STEERTTS_BUILD_SHARED "build libsteertts.so / .dylib" ON)
option(STEERTTS_BUILD_CLI "build the steertts command line" ON)
option(STEERTTS_BUILD_EXAMPLES "build examples/" ON)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(ST_TOP ON)
else()
  set(ST_TOP OFF)   # added with add_subdirectory(): no tests by default
endif()
option(STEERTTS_BUILD_TESTS "build tests/ (ctest)" ${ST_TOP})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)   # ISO C11: no FMA contraction, so durations round exactly as in Python
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # no FMA contraction (clang contracts by default even in ISO mode): renders must round as the reference does
  add_compile_options(-Wall -Wextra -Wno-unused-parameter -ffp-contract=off)
  # no -ffast-math: renders must stay deterministic and close to the reference
  set(CMAKE_C_FLAGS_RELEASE "-O2")
endif()

include(GNUInstallDirs)
find_package(Threads REQUIRED)
file(GLOB ST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c ${CMAKE_CURRENT_SOURCE_DIR}/src/front/*.c)

# one set of objects for both libraries; only the steertts_* API (STEERTTS_API) is exported from the shared library
add_library(steertts_obj OBJECT ${ST_SOURCES})
target_include_directories(steertts_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(steertts_obj PRIVATE STEERTTS_BUILD)
set_target_properties(steertts_obj PROPERTIES C_VISIBILITY_PRESET hidden)
if(WIN32 AND STEERTTS_BUILD_SHARED)
  target_compile_definitions(steertts_obj PRIVATE STEERTTS_SHARED)
endif()

add_library(steertts_static STATIC $<TARGET_OBJECTS:steertts_obj>)
set_target_properties(steertts_static PROPERTIES OUTPUT_NAME steertts EXPORT_NAME steertts_static)
if(WIN32)   # steertts.lib would collide with the DLL's import library
  set_target_properties(steertts_static PROPERTIES OUTPUT_NAME steertts_static)
endif()
target_include_directories(steertts_static PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(steertts_static PUBLIC Threads::Threads $<$<NOT:$<PLATFORM_ID:Windows>>:m>)
set(ST_INSTALL steertts_static)

if(STEERTTS_BUILD_SHARED)
  add_library(steertts SHARED $<TARGET_OBJECTS:steertts_obj>)
  set_target_properties(steertts PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
  target_include_directories(steertts PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(steertts PRIVATE Threads::Threads $<$<NOT:$<PLATFORM_ID:Windows>>:m>)
  if(WIN32)
    target_compile_definitions(steertts INTERFACE STEERTTS_SHARED)   # consumers import the DLL's symbols
  endif()
  list(APPEND ST_INSTALL steertts)
  add_library(steertts::steertts ALIAS steertts)
else()
  add_library(steertts::steertts ALIAS steertts_static)
endif()
add_library(steertts::steertts_static ALIAS steertts_static)

if(STEERTTS_BUILD_CLI)
  add_executable(steertts_cli tools/steertts_cli.c)
  set_target_properties(steertts_cli PROPERTIES OUTPUT_NAME steertts)
  # MinGW writes an import library for every executable; named after the CLI it overwrote the DLL's
  # (libsteertts.dll.a then pointed programs at steertts.exe)
  if(WIN32)
    set_target_properties(steertts_cli PROPERTIES ARCHIVE_OUTPUT_NAME steertts_cli)
  endif()
  target_link_libraries(steertts_cli steertts_static)
  install(TARGETS steertts_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(STEERTTS_BUILD_EXAMPLES)
  set(ST_EXAMPLES simple stream json_request)
  if(NOT WIN32)
    list(APPEND ST_EXAMPLES speak)   # popen
  endif()
  foreach(ex ${ST_EXAMPLES})
    add_executable(example_${ex} examples/${ex}.c)
    target_link_libraries(example_${ex} steertts::steertts $<$<NOT:$<PLATFORM_ID:Windows>>:m>)
  endforeach()
endif()

# install: library, header, CMake package (steertts::steertts, steertts::steertts_static), pkg-config file
install(TARGETS ${ST_INSTALL} EXPORT steerttsTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/steertts.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT steerttsTargets NAMESPACE steertts:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/steertts)
include(CMakePackageConfigHelpers)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/steerttsConfig.cmake
     "include(CMakeFindDependencyMacro)\nfind_dependency(Threads)\n"
     "include(\${CMAKE_CURRENT_LIST_DIR}/steerttsTargets.cmake)\n"
     "if(NOT TARGET steertts::steertts)\n  add_library(steertts::steertts INTERFACE IMPORTED)\n"
     "  set_target_properties(steertts::steertts PROPERTIES INTERFACE_LINK_LIBRARIES steertts::steertts_static)\nendif()\n")
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/steerttsConfigVersion.cmake
                                 COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/steerttsConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/steerttsConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/steertts)
# the .pc file locates the install relative to itself (<libdir>/pkgconfig), so any --prefix works
file(RELATIVE_PATH PC_UP /p/${CMAKE_INSTALL_LIBDIR}/pkgconfig /p)
if(STEERTTS_BUILD_SHARED)
  set(PC_LIBS_EXTRA "")
else()
  set(PC_LIBS_EXTRA " -lm -lpthread")   # static-only install: the system libraries belong in Libs
endif()
configure_file(steertts.pc.in ${CMAKE_CURRENT_BINARY_DIR}/steertts.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/steertts.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
install(DIRECTORY tools/ DESTINATION ${CMAKE_INSTALL_DATADIR}/steertts/tools FILES_MATCHING PATTERN "*.py")
install(PROGRAMS tools/get_models.sh DESTINATION ${CMAKE_INSTALL_DATADIR}/steertts/tools)

# tests/test_*.c: one executable each, linked statically, with access to the internal headers
if(STEERTTS_BUILD_TESTS)
  enable_testing()
  file(GLOB ST_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.c)
  foreach(t ${ST_TESTS})
    get_filename_component(name ${t} NAME_WE)
    add_executable(${name} ${t})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/front)
    target_link_libraries(${name} steertts_static)
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endif()
