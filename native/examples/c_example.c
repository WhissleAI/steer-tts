/* Minimal C client: c_example <package_dir> "text" out.wav */
#include <stdio.h>

#include "steertts/steertts_c.h"

static int on_chunk(const float* audio, size_t n, void* user) {
  (void)audio;
  ++*(int*)user;
  fprintf(stderr, "chunk %d: %zu samples\n", *(int*)user, n);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <package_dir> \"text\" out.wav\n", argv[0]);
    return 2;
  }
  char err[512];
  steertts_engine* e = steertts_create(argv[1], 1, err, sizeof err);
  if (!e) {
    fprintf(stderr, "create: %s\n", err);
    return 1;
  }
  steertts_options o;
  steertts_default_options(&o);
  o.markup = 1;
  float* audio;
  size_t n;
  if (steertts_synthesize(e, argv[2], &o, &audio, &n, err, sizeof err)) {
    fprintf(stderr, "synthesize: %s\n", err);
    steertts_destroy(e);
    return 1;
  }
  steertts_save_wav(audio, n, argv[3]);
  printf("%s: %.2f s\n", argv[3], (double)n / steertts_sample_rate());
  steertts_free_audio(audio);
  int chunks = 0;
  steertts_stream(e, argv[2], &o, on_chunk, &chunks, err, sizeof err);
  o.tag = "no_such_tag";
  if (steertts_synthesize(e, "x", &o, &audio, &n, err, sizeof err)) printf("expected error: %.60s...\n", err);
  steertts_destroy(e);
  return 0;
}
