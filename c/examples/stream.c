/* Sentence streaming: each chunk is handed over as soon as it is rendered, with its segment (text, style, pause).
 * This example writes raw 16-bit PCM to stdout, so it plays while the rest renders:
 *
 *   ./stream path/to/model "First sentence. <whisper>Second, quietly.</whisper> [pause 500ms] *Third*!" \
 *       | aplay -f S16_LE -r 24000 -c 1          (macOS: | play -t raw -r 24000 -e signed -b 16 -c 1 -)
 */
#include <math.h>
#include <stdio.h>

#include "steertts.h"

static int on_chunk(const float* audio, size_t n, const steertts_chunk_info* info, void* user) {
  (void)user;
  fprintf(stderr, "chunk %d: %s%s [%s%s%s] %.0f ms\n", info->index, info->pause_ms ? "(pause) " : "", info->text,
          info->tag, *info->ctrl ? " " : "", info->ctrl, info->synth_ms);
  short pcm[1024];
  for (size_t i = 0; i < n;) {
    size_t k = 0;
    for (; k < 1024 && i < n; ++k, ++i) {
      float v = audio[i] < -1.f ? -1.f : audio[i] > 1.f ? 1.f : audio[i];
      pcm[k] = (short)lrintf(v * 32767.f);
    }
    fwrite(pcm, sizeof(short), k, stdout);
  }
  fflush(stdout);
  return 0;   /* non-zero would stop the stream */
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s MODEL_DIR TEXT | aplay -f S16_LE -r 24000 -c 1\n", argv[0]);
    return 2;
  }
  char err[512];
  steertts_engine* e = steertts_create(argv[1], 2, err, sizeof err);
  if (!e) {
    fprintf(stderr, "load failed: %s\n", err);
    return 1;
  }
  steertts_options o;
  steertts_default_options(&o);
  int rc = steertts_stream_ex(e, argv[2], &o, on_chunk, NULL, err, sizeof err);
  if (rc) fprintf(stderr, "stream failed: %s\n", err);
  steertts_destroy(e);
  return rc != 0;
}
