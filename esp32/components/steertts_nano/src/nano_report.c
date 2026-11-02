/* Benchmark report in the RESULT|name|time|rate|extra line format. */
#include <stdio.h>

#include "nano_internal.h"

void nano_print_report(const nano_profile* p, double cycles_per_us, const char* tag) {
  double audio_s = 0, total_us = 0;
  for (int s = 0; s < NANO_ST_COUNT; s++) total_us += (double)p->cycles[s] / cycles_per_us;
  if (p->samples) audio_s = (double)p->samples / (p->sample_rate > 0 ? p->sample_rate : 16000);
  for (int s = 0; s < NANO_ST_COUNT; s++) {
    double us = (double)p->cycles[s] / cycles_per_us;
    double gmac = us > 0 ? (double)p->macs[s] / (us * 1e3) : 0;
    printf("RESULT|%s stage %s|%.1f us|%.3f GMAC/s|%.2f MMAC, %.1f KB weights read, %u passes, %.1f ms per audio s\n",
           tag, nano_stage_name(s), us, gmac, (double)p->macs[s] / 1e6, (double)p->weight_bytes[s] / 1024.0,
           (unsigned)p->passes[s], audio_s > 0 ? us / 1e3 / audio_s : 0.0);
  }
  for (int o = 0; o < NANO_OP_COUNT; o++) {
    double us = (double)p->op_cycles[o] / cycles_per_us;
    printf("RESULT|%s op %s|%.1f us|%.1f%%|\n", tag, nano_op_name(o), us, total_us > 0 ? 100.0 * us / total_us : 0.0);
  }
  uint64_t macs = 0, wb = 0;
  for (int s = 0; s < NANO_ST_COUNT; s++) {
    macs += p->macs[s];
    wb += p->weight_bytes[s];
  }
  printf("RESULT|%s int8 hashes|enc %08x|dec-in %08x, mel %08x, head %08x\n", tag, (unsigned)p->hash_enc,
         (unsigned)p->hash_p, (unsigned)p->hash_mel, (unsigned)p->hash_head);
  printf("RESULT|%s synth %.2f s audio (%ld phonemes, %ld frames)|%.1f us|RTF %.3f|%.3f GMAC/s avg, %.2f MB weights read\n",
         tag, audio_s, p->phonemes, p->frames, total_us, audio_s > 0 ? total_us / 1e6 / audio_s : 0.0,
         total_us > 0 ? (double)macs / (total_us * 1e3) : 0.0, (double)wb / 1048576.0);
}

/* benchmark input: random phonemes with a word boundary (<sp>) every 4 ids, xorshift32 */
void nano_bench_ids(int* ids, int n, uint32_t* state) {
  for (int i = 0; i < n; i++) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    ids[i] = (i % 4 == 0) ? 1 : 9 + (int)(x % 69);
  }
}

/* FNV-1a over the little-endian PCM bytes: equal hashes = bit-identical audio across platforms */
uint32_t nano_hash_pcm(uint32_t h, const int16_t* pcm, int n) {
  for (int i = 0; i < n; i++) {
    uint16_t v = (uint16_t)pcm[i];
    h = (h ^ (v & 0xFF)) * 16777619u;
    h = (h ^ (v >> 8)) * 16777619u;
  }
  return h;
}
