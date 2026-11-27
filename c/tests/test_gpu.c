/* Whole requests on the GPU path against the CPU path: the same number of samples (the durations are decided by
 * the same host code from the device's predictor outputs, so they must round the same) and the waveform SNR.
 *
 *   test_gpu [device]      device: $STEERTTS_TEST_DEVICE, else "gpu" (cuda, vulkan, metal, ggml-cpu, CUDA0 ...)
 *   GPU_BENCH=1 test_gpu   also prints per-request times on both paths
 *
 * Models: $STEERTTS_V1_MODEL, $STEERTTS_EDGE_MODEL (default ../models/...). Exit 77 (skip) when the library has no
 * GPU support, the device is missing, or no model is found. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/st_base.h"
#include "steertts.h"

static int g_fail;

static const char* envor(const char* k, const char* d) {
  const char* v = getenv(k);
  return v && *v ? v : d;
}

static int have_model(const char* dir) {
  char p[1024];
  snprintf(p, sizeof p, "%s/config.json", dir);
  FILE* f = fopen(p, "rb");
  if (f) fclose(f);
  return f != NULL;
}

static const char* reqs[] = {
    "{\"text\": \"The meeting moved to Thursday.\"}",
    "{\"text\": \"Are you serious?\", \"ctrl\": {\"f0_end\": 2.0, \"f0_iqr\": 1.5}}",
    "{\"text\": \"It is over.\", \"tag\": \"sadness\", \"ctrl\": {\"loud\": -1}, \"guidance\": 1.5}",
    "{\"text\": \"I said the blue one.\", \"word_shift\": {\"3\": {\"pitch\": 3, \"dur\": 1.3}}}",
    "{\"text\": \"Well, *that* was fast. [pause 300ms] <whisper>Don't tell anyone.</whisper>\", \"markup\": true}",
    "{\"text\": \"When the storm finally passed, the fishermen dragged their boats back down to the water, counted "
    "the nets that had survived the night, and argued for an hour about whether the old lighthouse keeper had "
    "really seen the lights out past the reef, or whether he had simply fallen asleep again with the lamp burning "
    "and dreamed the whole thing.\"}",
};

static void run(const char* label, const char* dir, const char* precision, const char* device, double min_snr) {
  char err[512];
  steertts_config c;
  steertts_default_config(&c);
  c.threads = 4;
  c.precision = precision;
  steertts_engine* cpu = steertts_create_device(dir, &c, "cpu", err, sizeof err);
  steertts_engine* gpu = cpu ? steertts_create_device(dir, &c, device, err, sizeof err) : NULL;
  if (!cpu || !gpu) {
    printf("%s: %s\n", label, err);
    g_fail = 1;
    steertts_destroy(cpu);
    return;
  }
  printf("%s on %s\n", label, steertts_device(gpu));
  int bench = getenv("GPU_BENCH") != NULL;
  for (size_t i = 0; i < sizeof reqs / sizeof *reqs; ++i) {
    float *a = NULL, *b = NULL;
    size_t na = 0, nb = 0;
    double t0 = st_time_ms();
    int ra = steertts_synthesize_json(cpu, reqs[i], &a, &na, err, sizeof err);
    double t1 = st_time_ms();
    int rb = ra ? -1 : steertts_synthesize_json(gpu, reqs[i], &b, &nb, err, sizeof err);
    double t2 = st_time_ms();
    if (ra || rb) {
      printf("  FAIL case %zu: %s\n", i, err);
      g_fail = 1;
    } else {
      double s = 0, e = 0, mx = 0;
      size_t n = na < nb ? na : nb;
      for (size_t k = 0; k < n; ++k) {
        double d = (double)a[k] - b[k];
        s += (double)a[k] * a[k], e += d * d;
        if (fabs(d) > mx) mx = fabs(d);
      }
      double snr = e > 0 ? 10 * log10(s / e) : 999;
      int ok = na == nb && snr >= min_snr;
      printf("  %s case %zu: %zu vs %zu samples, wav maxabs %.2e SNR %.1f dB", ok ? "ok  " : "FAIL", i, na, nb, mx, snr);
      if (bench) printf(" | cpu %.0f ms, gpu %.0f ms", t1 - t0, t2 - t1);
      printf("\n");
      if (!ok) g_fail = 1;
    }
    steertts_free(a), steertts_free(b);
  }
  steertts_destroy(gpu);
  steertts_destroy(cpu);
}

int main(int argc, char** argv) {
  const char* device = argc > 1 ? argv[1] : envor("STEERTTS_TEST_DEVICE", "gpu");
  if (!steertts_gpu_available()) {
    printf("skip: built without GPU support (-DSTEERTTS_GGML=ON)\n");
    return 77;
  }
  const char* v1 = envor("STEERTTS_V1_MODEL", "../models/steer-tts");
  const char* edge = envor("STEERTTS_EDGE_MODEL", "../models/steer-tts-edge");
  const char* probe = have_model(edge) ? edge : have_model(v1) ? v1 : NULL;
  if (!probe) {
    printf("skip: no model (STEERTTS_V1_MODEL=%s, STEERTTS_EDGE_MODEL=%s)\n", v1, edge);
    return 77;
  }
  char err[512];
  steertts_engine* e = steertts_create_device(probe, NULL, device, err, sizeof err);
  if (!e) {
    printf("skip: device %s: %s\n", device, err);
    return 77;
  }
  steertts_destroy(e);
  if (have_model(edge)) run("edge", edge, "fp32", device, 50);
  if (have_model(v1)) {
    run("v1 fp32 (flow)", v1, "fp32", device, 40);
    run("v1 int8 (flow)", v1, "int8", device, 40);
  }
  printf(g_fail ? "FAILED\n" : "all passed\n");
  return g_fail;
}
