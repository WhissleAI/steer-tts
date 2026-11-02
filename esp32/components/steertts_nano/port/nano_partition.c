/* Open nano.bin in place from a memory-mapped flash data partition (ESP-IDF 4.4 and 5.x). */
#if defined(ESP_PLATFORM)
#include <stdio.h>
#include <string.h>

#include "esp_idf_version.h"
#include "esp_partition.h"
#include "nano_front.h"
#include "nano_partition.h"

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
#define NANO_MMAP_DATA ESP_PARTITION_MMAP_DATA
typedef esp_partition_mmap_handle_t nano_mmap_handle_t;
#else
#include "esp_spi_flash.h"
#define NANO_MMAP_DATA SPI_FLASH_MMAP_DATA
typedef spi_flash_mmap_handle_t nano_mmap_handle_t;
#endif

int nano_model_open_partition(nano_model* m, const char* label, nano_partition_map* map) {
  memset(map, 0, sizeof *map);
  const esp_partition_t* p = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, label);
  if (!p) {
    nano_model_open(m, NULL, 0);
    return -1;
  }
  nano_bin_header h;
  if (esp_partition_read(p, 0, &h, sizeof h) != ESP_OK || memcmp(h.magic, NANO_BIN_MAGIC, 4) ||
      h.file_bytes > p->size) {
    nano_model_open(m, NULL, 0);   /* sets "not a nano.bin image" */
    return -2;
  }
  size_t len = (h.file_bytes + 0xFFFF) & ~(size_t)0xFFFF;   /* whole 64 KB MMU pages */
  if (len > p->size) len = p->size;
  const void* ptr = NULL;
  nano_mmap_handle_t hd;
  if (esp_partition_mmap(p, 0, len, NANO_MMAP_DATA, &ptr, &hd) != ESP_OK) return -3;
  map->ptr = ptr;
  map->len = len;
  map->handle = (uint32_t)hd;
  map->partition_size = p->size;
  map->partition_offset = p->address;
  return nano_model_open(m, ptr, h.file_bytes);
}

int nano_front_open_partition(void* front_ctx, const char* label, nano_partition_map* map) {
  nano_front_ctx* c = (nano_front_ctx*)front_ctx;
  memset(map, 0, sizeof *map);
  const esp_partition_t* p = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, label);
  if (!p || p->size <= NANO_FRONT_PARTITION_OFFSET + 16) {
    snprintf(c->err, sizeof c->err, "no partition '%s' large enough for front.bin", label);
    return -1;
  }
  size_t avail = p->size - NANO_FRONT_PARTITION_OFFSET;
  const void* ptr = NULL;
  nano_mmap_handle_t hd;
  if (esp_partition_mmap(p, NANO_FRONT_PARTITION_OFFSET, avail, NANO_MMAP_DATA, &ptr, &hd) != ESP_OK) {
    snprintf(c->err, sizeof c->err, "cannot map front.bin");
    return -3;
  }
  map->ptr = ptr;
  map->len = avail;
  map->handle = (uint32_t)hd;
  map->partition_size = p->size;
  map->partition_offset = p->address + NANO_FRONT_PARTITION_OFFSET;
  return nano_front_open(c, ptr, avail);
}
#endif
