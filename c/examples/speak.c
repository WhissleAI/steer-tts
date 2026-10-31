/* A long-running app: load the model once, then speak every line typed, through the speakers, as it renders.
 *
 *   ./speak path/to/model [VOICE]
 *   > Hello there. How are you today?
 *   > <whisper>Quiet now.</whisper> [pause 400ms] *Loud* again!
 *   > /voice voice_0372          (/voices lists them, /quit or Ctrl-D ends)
 *
 * Audio goes to a player reading raw 16-bit 24 kHz mono PCM on stdin: $STEERTTS_PLAYER, else sox's play, else
 * aplay. After each line, 400 ms of silence pushes the line's last words out of the player's buffer (players hold a
 * partial buffer until more input arrives). A line that fails (bad markup, unknown control) is reported and the loop
 * goes on. POSIX (popen). */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "steertts.h"

static int on_chunk(const float* audio, size_t n, const steertts_chunk_info* info, void* user) {
  (void)info;
  FILE* player = (FILE*)user;
  short pcm[1024];
  for (size_t i = 0; i < n;) {
    size_t k = 0;
    for (; k < 1024 && i < n; ++k, ++i) {
      float v = audio[i] < -1.f ? -1.f : audio[i] > 1.f ? 1.f : audio[i];
      pcm[k] = (short)lrintf(v * 32767.f);
    }
    if (fwrite(pcm, sizeof(short), k, player) != k) return 1;   /* the player quit: stop this line */
  }
  return fflush(player) != 0;   /* flush per chunk, so it plays while the next sentence renders */
}

static const char* player_cmd(void) {
  const char* cmd = getenv("STEERTTS_PLAYER");
  if (cmd && *cmd) return cmd;
  if (system("command -v play >/dev/null 2>&1") == 0)
    return "play -q --buffer 1024 -t raw -r 24000 -e signed -b 16 -c 1 -";
  if (system("command -v aplay >/dev/null 2>&1") == 0) return "aplay -q -t raw -f S16_LE -r 24000 -c 1";
  return NULL;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s MODEL_DIR [VOICE]\n", argv[0]);
    return 2;
  }
  char err[512], voice[128] = "", line[4096];
  steertts_engine* e = steertts_create(argv[1], 0, err, sizeof err);   /* once; 0 = all CPUs */
  if (!e) {
    fprintf(stderr, "load failed: %s\n", err);
    return 1;
  }
  steertts_options o;
  steertts_default_options(&o);
  if (argc > 2) snprintf(voice, sizeof voice, "%s", argv[2]), o.voice = voice;

  const char* cmd = player_cmd();
  FILE* player = cmd ? popen(cmd, "w") : NULL;
  if (!player) {
    fprintf(stderr, "no player: install sox or alsa-utils, or set STEERTTS_PLAYER\n");
    steertts_destroy(e);
    return 1;
  }
  signal(SIGPIPE, SIG_IGN);   /* a player that exits makes fwrite fail instead of killing us */

  fprintf(stderr, "%s, %d voices. Type a line to speak it; /voices, /voice NAME, /quit.\n> ",
          steertts_model_type(e), steertts_num_voices(e));
  while (fgets(line, sizeof line, stdin)) {
    line[strcspn(line, "\r\n")] = 0;
    if (!strcmp(line, "/quit")) break;
    if (!strcmp(line, "/voices")) {
      for (int i = 0; i < steertts_num_voices(e); ++i) fprintf(stderr, "%s ", steertts_voice(e, i));
      fprintf(stderr, "\n(default %s)\n", steertts_default_voice(e));
    } else if (!strncmp(line, "/voice ", 7)) {
      snprintf(voice, sizeof voice, "%s", line + 7);
      o.voice = voice;   /* an unknown name is reported by the next line's render */
    } else if (*line) {
      if (steertts_stream_ex(e, line, &o, on_chunk, player, err, sizeof err)) fprintf(stderr, "error: %s\n", err);
      static const float silence[STEERTTS_SAMPLE_RATE * 2 / 5];
      on_chunk(silence, sizeof silence / sizeof silence[0], NULL, player);
    }
    fprintf(stderr, "> ");
  }
  pclose(player);   /* returns once the queued audio has played */
  steertts_destroy(e);
  return 0;
}
