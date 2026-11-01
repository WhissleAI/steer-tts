# Plain-make build (CMake works too: cmake -B build && cmake --build build).
#   make                         libsteertts.a, libsteertts.so, the steertts CLI and the examples in build/
#   make CC=clang CFLAGS=-O3     any C11 compiler and your own optimisation flags (the required flags stay on)
#   make CC=aarch64-linux-gnu-gcc   cross-compiling
#   make PI=4 / make PI=5        tuned for a Raspberry Pi 4 (Cortex-A72) / Pi 5 (Cortex-A76); a PI=5 build does not run
#                                on a Pi 4. Same samples as a portable build (only instruction selection changes).
CC ?= cc
CFLAGS ?= -O2
ifeq ($(PI),4)
  CFLAGS += -mcpu=cortex-a72
else ifeq ($(PI),5)
  CFLAGS += -mcpu=cortex-a76
endif
# required: ISO C11, no FMA contraction (renders must round as the reference does), hidden internals
ST_CFLAGS = -std=c11 -ffp-contract=off -fPIC -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter -Iinclude -Isrc
LDLIBS = -lm -lpthread
OUT = build
SRC = $(wildcard src/*.c src/front/*.c)
OBJ = $(SRC:%.c=$(OUT)/obj/%.o)
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  SHARED = $(OUT)/libsteertts.dylib
  SOFLAGS = -dynamiclib -install_name @rpath/libsteertts.1.dylib
else
  SHARED = $(OUT)/libsteertts.so
  SOFLAGS = -shared -Wl,-soname,libsteertts.so.1
endif

all: $(OUT)/libsteertts.a $(SHARED) $(OUT)/steertts $(OUT)/steertts-pi $(OUT)/example_simple $(OUT)/example_stream \
     $(OUT)/example_json_request $(OUT)/example_speak $(OUT)/pi_hello \
     $(OUT)/pi_streaming

$(OUT)/obj/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(ST_CFLAGS) -DSTEERTTS_BUILD $(CFLAGS) -c $< -o $@

$(OUT)/libsteertts.a: $(OBJ)
	$(AR) rcs $@ $^

$(SHARED): $(OBJ)
	$(CC) $(SOFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(OUT)/steertts: tools/steertts_cli.c $(OUT)/libsteertts.a
	$(CC) $(ST_CFLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(OUT)/steertts-pi: ../raspberry-pi/server/steertts_pi.c $(OUT)/libsteertts.a
	$(CC) $(ST_CFLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(OUT)/pi_%: ../raspberry-pi/examples/%.c $(OUT)/libsteertts.a
	$(CC) $(ST_CFLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(OUT)/example_%: examples/%.c $(OUT)/libsteertts.a
	$(CC) $(ST_CFLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf $(OUT)

.PHONY: all clean
