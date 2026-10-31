# SteerTTS text front end (C++17, standard library only). Used via add_subdirectory(src/front) from native/.
cmake_minimum_required(VERSION 3.16)
project(steertts_front CXX)

add_library(steertts_front STATIC
  ucd.cpp
  pyre.cpp
  numwords.cpp
  g2p.cpp
  frontend.cpp
  markup.cpp)
target_compile_features(steertts_front PUBLIC cxx_std_17)
target_include_directories(steertts_front
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(MSVC)
  target_compile_options(steertts_front PRIVATE /utf-8 /fp:precise)
else()
  # no FMA contraction: keeps the GRU G2P arithmetic identical on every target (see g2p.cpp dot_q)
  target_compile_options(steertts_front PRIVATE -ffp-contract=off)
endif()
find_package(Threads REQUIRED)
target_link_libraries(steertts_front PUBLIC Threads::Threads)
