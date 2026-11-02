/* Platform hooks: cycle counter. */
#include "nano_internal.h"

#if defined(ESP_PLATFORM)
/* CCOUNT, extended to 64 bits (the 32-bit counter wraps every 17.9 s at 240 MHz); per core, call often enough */
uint64_t nano_cycles(void) {
  static uint32_t last[2], hi[2];
  uint32_t cc, core;
  __asm__ volatile("rsr %0, ccount" : "=a"(cc));
  __asm__ volatile("rsr.prid %0" : "=a"(core));
  core = (core >> 13) & 1;
  if (cc < last[core]) hi[core]++;
  last[core] = cc;
  return ((uint64_t)hi[core] << 32) | cc;
}
#else
#include <time.h>
uint64_t nano_cycles(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#endif
