/* nano.bin from a flash data partition (ESP-IDF): the image is memory-mapped, weights are read through the cache. */
#ifndef NANO_PARTITION_H
#define NANO_PARTITION_H

#include "steertts_nano.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const void* ptr;
  size_t len;
  uint32_t handle;
  uint32_t partition_size, partition_offset;
} nano_partition_map;

/* Map the data partition `label` (e.g. "steertts") and open the model in it. 0 on success; -1 no such partition,
 * -2 no nano.bin in it, -3 mmap failed, or nano_model_open's error. */
int nano_model_open_partition(nano_model* m, const char* label, nano_partition_map* map);

/* The front end data (front.bin) sits in the same partition at this offset (nano.bin at 0). */
#define NANO_FRONT_PARTITION_OFFSET 0x400000

/* Map front.bin from the partition `label` at NANO_FRONT_PARTITION_OFFSET into ctx. 0 or < 0 (ctx->err). */
int nano_front_open_partition(void* front_ctx, const char* label, nano_partition_map* map);

#ifdef __cplusplus
}
#endif
#endif
