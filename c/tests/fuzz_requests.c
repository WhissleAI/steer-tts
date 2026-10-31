/* Random-request robustness check (not part of ctest): malformed JSON, random field values and random text (invalid
 * UTF-8, markup fragments, numbers, symbols) through the public API. Build with sanitizers:
 *   cc -O1 -g -fsanitize=address,undefined -Iinclude -Isrc src/ *.c src/front/ *.c tests/fuzz_requests.c -lm -lpthread
 *   ./a.out MODEL_DIR [iterations] [seed]
 * Every call must return cleanly (0 with audio, or an error message); the sanitizers catch the rest. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "steertts.h"

static unsigned long long rs;
static unsigned rnd(void) {
  rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
  return (unsigned)(rs >> 33);
}

static const char* frags[] = {"Hello", " world", ".", "!", "?", ",", " *wow*", "<whisper>", "</whisper>", "<slow>", "</slow>",
                              "[pause 300ms]", "[pause 9s]", "[pause 1.2.3]", "[laugh]", "[dog]", "<style prompt=\"sad\">",
                              "</style>", "<sadness>", "$3,000", "1990-1995", "4x4", "km/h", "12:30", "#", "@", "\\",
                              "\"", "\xc3\xa9", "\xe2\x80\x94", "\xf0\x9f\x98\x80", "\xff\xfe", "\xc3", "NASA", "IOC",
                              "  ", "\n", "\t", "1st", "2,243.5", "100000000000000000000000", "-", "...", "&", "\x01"};

static void rand_text(char* out, size_t cap) {
  size_t n = 0;
  int parts = (int)(rnd() % 8);
  out[0] = 0;
  for (int i = 0; i < parts; ++i) {
    const char* f = frags[rnd() % (sizeof frags / sizeof *frags)];
    size_t l = strlen(f);
    if (n + l + 1 >= cap) break;
    memcpy(out + n, f, l + 1);
    n += l;
  }
  if (rnd() % 4 == 0)   /* raw random bytes */
    for (int i = 0; i < 6 && n + 2 < cap; ++i) out[n++] = (char)(1 + rnd() % 255), out[n] = 0;
}

static void json_escape(char* dst, size_t cap, const char* s) {
  size_t k = 0;
  for (; *s && k + 8 < cap; ++s) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') dst[k++] = '\\', dst[k++] = (char)c;
    else if (c < 0x20) k += (size_t)snprintf(dst + k, cap - k, "\\u%04x", c);
    else dst[k++] = (char)c;
  }
  dst[k] = 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s MODEL_DIR [iterations] [seed]\n", argv[0]);
    return 2;
  }
  int iters = argc > 2 ? atoi(argv[2]) : 500;
  rs = argc > 3 ? strtoull(argv[3], NULL, 10) : 1;
  char err[1024];
  steertts_engine* e = steertts_create(argv[1], 2, err, sizeof err);
  if (!e) {
    fprintf(stderr, "%s\n", err);
    return 1;
  }
  static const char* keys[] = {"voice", "tag", "ctrl", "pace", "guidance", "markup", "flow", "seed", "word_shift",
                               "prompt", "bogus"};
  static const char* vals[] = {"1", "-3", "0.5", "1e9", "true", "false", "null", "\"none\"", "\"sadness\"",
                               "\"voice_0010\"", "{}", "[]", "{\"rate\": 1}", "{\"rate\": 7}", "{\"zz\": 1}",
                               "{\"1\": {\"pitch\": 3}}", "{\"x\": {}}", "{\"0\": {\"dur\": 9}}", "\"angry\""};
  int ok = 0, errs = 0;
  for (int it = 0; it < iters; ++it) {
    char text[256], esc[1024], json[4096];
    rand_text(text, sizeof text);
    json_escape(esc, sizeof esc, text);
    size_t k = (size_t)snprintf(json, sizeof json, "{\"text\": \"%s\"", esc);
    int nf = (int)(rnd() % 4);
    for (int f = 0; f < nf; ++f)
      k += (size_t)snprintf(json + k, sizeof json - k, ", \"%s\": %s", keys[rnd() % 11], vals[rnd() % 19]);
    snprintf(json + k, sizeof json - k, "}");
    if (rnd() % 10 == 0) json[rnd() % (strlen(json) + 1)] = 0;   /* truncated JSON */
    float* y = NULL;
    size_t n = 0;
    if (steertts_synthesize_json(e, json, &y, &n, err, sizeof err) == 0) {
      ++ok;
      steertts_free(y);
    } else
      ++errs;
    char* r = NULL;
    if (steertts_resolve_json(e, json, &r, err, sizeof err) == 0) steertts_free(r);
    steertts_options o;
    steertts_default_options(&o);
    o.markup = (int)(rnd() % 2);
    if (steertts_synthesize(e, text, &o, &y, &n, err, sizeof err) == 0) steertts_free(y);
  }
  printf("%d requests: %d rendered, %d rejected with a message\n", iters, ok, errs);
  steertts_destroy(e);
  return 0;
}
