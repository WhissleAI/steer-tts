/* SteerTTS Nano on-device benchmark (ESP32-S3): prints CHECK| / RESULT| lines on the console. */
#ifndef NANO_BENCH_H
#define NANO_BENCH_H
#ifdef __cplusplus
extern "C" {
#endif
/* start the benchmark task on `core` (reads nano.bin from the "steertts" data partition) */
void nano_bench_start(int core);
#ifdef __cplusplus
}
#endif
#endif
