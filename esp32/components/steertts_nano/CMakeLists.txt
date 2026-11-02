# SteerTTS Nano runtime as an ESP-IDF component (ESP-IDF 4.4 and 5.x). For a plain host build see ../../host.
set(srcs "src/nano_model.c" "src/nano_kernels.c" "src/nano_simd_s3.c" "src/nano_dsp.c" "src/nano_synth.c"
         "src/nano_tables.c" "src/nano_port.c" "src/nano_report.c" "port/nano_partition.c"
         "bench/nano_bench.c"
         "front/nf_core.c" "front/nf_text.c" "front/nf_num.c" "front/nf_g2p.c" "front/nf_api.c" "front/nf_markup.c")
if(IDF_VERSION_MAJOR GREATER_EQUAL 5)
  set(reqs esp_partition spi_flash freertos heap esp_rom)
else()
  set(reqs spi_flash freertos heap esp_rom)
endif()
# esp-dsp (optional): its FFT kernel replaces the portable one when the component is in the build
set(priv_reqs "")
idf_build_get_property(build_components BUILD_COMPONENTS)
foreach(dsp espressif__esp-dsp esp-dsp)
  if(dsp IN_LIST build_components)
    list(APPEND priv_reqs ${dsp})
  endif()
endforeach()
idf_component_register(SRCS ${srcs}
                       INCLUDE_DIRS "include"
                       PRIV_INCLUDE_DIRS "src" "front"
                       REQUIRES ${reqs}
                       PRIV_REQUIRES ${priv_reqs})
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off -Wno-unused-function -Wno-format-truncation)
# -DNANO_PORTABLE_FFT=ON: portable FFT instead of esp-dsp (PCM then matches the host build bit for bit)
if(NANO_PORTABLE_FFT)
  target_compile_definitions(${COMPONENT_LIB} PRIVATE NANO_NO_ESP_DSP=1)
endif()
