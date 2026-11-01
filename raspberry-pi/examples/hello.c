/* The smallest SteerTTS program: load the edge model, render one sentence with a voice, a style tag and steering
 * controls, write a WAV file.
 *
 *   cc hello.c $(pkg-config --cflags --libs steertts) -o hello                      # shared libsteertts.so
 *   cc hello.c $(pkg-config --cflags steertts) -l:libsteertts.a -lm -lpthread -o hello   # static, nothing to ship
 *   ./hello ~/.local/share/steertts/models/steer-tts-edge "Are you coming with us tonight?" question.wav
 */
#include <stdio.h>

#include "steertts.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s MODEL_DIR TEXT OUT.wav\n", argv[0]);
    return 2;
  }
  char err[512];
  steertts_engine* e = steertts_create(argv[1], 0, err, sizeof err);   /* 0 threads = every CPU */
  if (!e) {
    fprintf(stderr, "load failed: %s\n", err);
    return 1;
  }
  steertts_options o;
  steertts_default_options(&o);
  o.voice = steertts_gender_voice(e, "female");   /* or a name: steertts_voice(e, i), i < steertts_num_voices(e) */
  o.tag = "none";                                 /* a style tag: steertts_tag(e, i), e.g. "happiness", "whisper" */
  o.ctrl = "f0_end=2,f0_iqr=1.5";                 /* controls (z-scores in [-5, 5]): steertts_control(i) */
  float* audio;
  size_t n;
  int rc = steertts_synthesize(e, argv[2], &o, &audio, &n, err, sizeof err);
  if (rc == 0) {
    rc = steertts_save_wav(audio, n, argv[3]);   /* 16-bit, 24 kHz, mono */
    printf("%s: %.2f s of audio\n", argv[3], (double)n / STEERTTS_SAMPLE_RATE);
    steertts_free(audio);
  } else
    fprintf(stderr, "failed: %s\n", err);
  steertts_destroy(e);
  return rc != 0;
}
