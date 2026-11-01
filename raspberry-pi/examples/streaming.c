/* Streaming: raw 16-bit PCM on stdout while the text renders, sentence by sentence, the first sentence in pieces so
 * playback starts after ~0.1 s on a Pi 5 (the samples are the same as rendering it whole).
 *
 *   cc streaming.c $(pkg-config --cflags --libs steertts) -o streaming
 *   ./streaming MODEL_DIR "First sentence. <whisper>Second, quietly.</whisper>" | aplay -f S16_LE -r 24000 -c 1
 */
#include <math.h>
#include <stdio.h>

#include "steertts.h"

typedef struct {
  double first_ms;   /* render time of the first piece = time to first audio */
} progress;

static int on_chunk(const float* audio, size_t n, const steertts_chunk_info* info, void* user) {
  progress* p = (progress*)user;
  if (p->first_ms < 0) p->first_ms = info->synth_ms;
  /* info: index (sentence / segment), text, tag, ctrl, pause_ms; more = 1 while pieces of this sentence follow */
  short pcm[2048];
  for (size_t i = 0; i < n;) {
    size_t k = 0;
    for (; k < 2048 && i < n; ++k, ++i) {
      float v = audio[i] < -1.f ? -1.f : audio[i] > 1.f ? 1.f : audio[i];
      pcm[k] = (short)lrintf(v * 32767.f);
    }
    if (fwrite(pcm, sizeof(short), k, stdout) != k) return 1;   /* the player went away: stop rendering */
  }
  fflush(stdout);
  return 0;   /* non-zero stops the stream */
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s MODEL_DIR TEXT | aplay -f S16_LE -r 24000 -c 1\n", argv[0]);
    return 2;
  }
  char err[512];
  steertts_engine* e = steertts_create(argv[1], 0, err, sizeof err);
  if (!e) {
    fprintf(stderr, "load failed: %s\n", err);
    return 1;
  }
  steertts_set_first_chunk_ms(e, 300);   /* 0 = one chunk per sentence */
  steertts_options o;
  steertts_default_options(&o);         /* streaming always reads inline markup: <whisper>, *word*, [pause 300ms] */
  progress p = {-1};
  int rc = steertts_stream_ex(e, argv[2], &o, on_chunk, &p, err, sizeof err);
  if (rc) fprintf(stderr, "failed: %s\n", err);
  else fprintf(stderr, "first audio after %.0f ms\n", p.first_ms);
  steertts_destroy(e);
  return rc != 0;
}
