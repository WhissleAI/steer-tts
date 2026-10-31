/* The smallest SteerTTS program: one sentence with a style tag and two controls -> out.wav.
 *
 *   cc simple.c -lsteertts -o simple        (or: pkg-config --cflags --libs steertts)
 *   ./simple path/to/steer-tts-edge "Are you coming with us tonight?"
 */
#include <stdio.h>

#include "steertts.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s MODEL_DIR TEXT\n", argv[0]);
    return 2;
  }
  char err[512];
  steertts_engine* e = steertts_create(argv[1], /*threads=*/1, err, sizeof err);
  if (!e) {
    fprintf(stderr, "load failed: %s\n", err);
    return 1;
  }
  steertts_options o;
  steertts_default_options(&o);
  o.tag = "happiness";
  o.ctrl = "f0_end=2,f0_iqr=1.5";   /* rising end (question-like), livelier pitch */

  float* audio;
  size_t n;
  if (steertts_synthesize(e, argv[2], &o, &audio, &n, err, sizeof err) != 0) {
    fprintf(stderr, "synthesis failed: %s\n", err);
    steertts_destroy(e);
    return 1;
  }
  steertts_save_wav(audio, n, "out.wav");
  printf("out.wav: %.2f s of audio (%s)\n", (double)n / steertts_sample_rate(), steertts_model_type(e));
  steertts_free(audio);
  steertts_destroy(e);
  return 0;
}
