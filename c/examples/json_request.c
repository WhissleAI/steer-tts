/* Full requests as JSON (request.schema.json), the voice / tag lists, the resolved form of a free-text prompt, and
 * error handling.
 *
 *   ./json_request path/to/steer-tts          (v1: also maps a prompt; prompts need native/minilm.safetensors)
 *   ./json_request path/to/steer-tts-edge
 */
#include <stdio.h>
#include <string.h>

#include "steertts.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
    return 2;
  }
  char err[512];
  steertts_config c;
  steertts_default_config(&c);
  c.threads = 0;   /* all cores */
  steertts_engine* e = steertts_create_ex(argv[1], &c, err, sizeof err);
  if (!e) {
    fprintf(stderr, "load failed: %s\n", err);
    return 1;
  }
  printf("%s %s: %d voices (default %s), %d tags, controls:", steertts_model_type(e), steertts_version(),
         steertts_num_voices(e), steertts_default_voice(e), steertts_num_tags(e));
  for (int i = 0; i < steertts_num_controls(); ++i) printf(" %s", steertts_control(i));
  printf("\nprompts: %s, cloning: %s, flow refiner: %s\n", steertts_has(e, "prompt") ? "yes" : "no",
         steertts_has(e, "clone") ? "yes" : "no", steertts_has(e, "flow") ? "yes" : "no");

  /* a complete request */
  const char* req = "{\"text\": \"It is over.\", \"tag\": \"sadness\", \"ctrl\": {\"loud\": -1, \"rate\": -0.5},"
                    " \"guidance\": 1.5, \"word_shift\": {\"2\": {\"pitch\": -2, \"dur\": 1.4}}}";
  float* audio;
  size_t n;
  if (steertts_synthesize_json(e, req, &audio, &n, err, sizeof err) == 0) {
    steertts_save_wav(audio, n, "sad.wav");
    printf("sad.wav: %.2f s\n", (double)n / STEERTTS_SAMPLE_RATE);
    steertts_free(audio);
  } else
    printf("error: %s\n", err);

  /* what a free-text prompt resolves to (v1) */
  if (steertts_has(e, "prompt")) {
    char* resolved;
    if (steertts_resolve_json(e, "{\"text\": \"See you soon.\", \"prompt\": \"excited, a bit faster\"}", &resolved, err,
                              sizeof err) == 0) {
      printf("resolved: %s\n", resolved);
      steertts_free(resolved);
    }
  }

  /* invalid requests fail with the Python packages' messages */
  if (steertts_synthesize_json(e, "{\"text\": \"x\", \"ctrl\": {\"speed\": 1}}", &audio, &n, err, sizeof err) != 0)
    printf("expected error: %s\n", err);
  steertts_destroy(e);
  return 0;
}
