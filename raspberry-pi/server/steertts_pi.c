/* steertts-pi: a small speech server around libsteertts for a Raspberry Pi (or any POSIX machine).
 *
 *   steertts-pi --model models/steer-tts-edge                       # HTTP on 127.0.0.1:8080
 *   curl -s 'localhost:8080/synthesize?text=Hello+there.&ctrl=f0_end=2' -o hello.wav
 *   curl -sN localhost:8080/stream -d '{"text": "First. Second.", "voice": "voice_2500"}' | aplay
 *   echo 'Hello from stdin.' | steertts-pi --model DIR --stdin | aplay -f S16_LE -r 24000 -c 1
 *
 * HTTP (HTTP/1.1, one request per connection, requests rendered one at a time on one loaded engine):
 *   GET  /health                   {"status": "ok", "model", "version", "sample_rate", "threads", "busy"}
 *   GET  /voices | /tags | /controls   JSON lists (voices with their gender)
 *   POST /synthesize [?format=pcm]  body: a JSON request (request.schema.json: text, voice, tag, ctrl, pace, guidance,
 *                                   markup, word_shift, ...) -> audio/wav (16-bit, 24 kHz, mono), or raw s16le PCM
 *   POST /stream [?format=pcm]      the same request, streamed: sentence by sentence as each is rendered (chunked
 *                                   transfer; a WAV header with an open length, then PCM), the first sentence in
 *                                   pieces (--first-chunk-ms)
 *   GET  /synthesize?text=..&voice=..&tag=..&ctrl=k=z,k=z&pace=..&guidance=..&markup=1[&level=off][&gain_db=..]
 *                                  [&format=pcm]   and
 *   GET  /stream?...                the same with query parameters
 * Output level: by default (level auto) the voice's neutral speech plays at -17 LUFS with peaks <= -1 dBTP (/stream:
 * the same fixed gain + soft limiter per chunk, no added latency); level=clip normalises every clip to -17 LUFS;
 * level=off (query, JSON "level", or --level off for the whole server) gives the raw model output.
 * Errors: a JSON body {"error": "..."} with status 400 (bad request), 404, 405, 413 (too large) or 503 (too many
 * connections). The engine's messages are the Python packages' messages.
 *
 * --stdin: one request per input line (plain text, rendered with markup, or a JSON request object); the audio goes
 * to stdout as raw 16-bit PCM while it renders (or, with --out-dir, one WAV file per line, its path and timings
 * printed as a JSON line). Errors are reported on stderr and the next line is read. */
#if defined(__linux__)
#define _GNU_SOURCE   /* sched_setaffinity, CPU_SET, strncasecmp, getline */
#else
#define _DARWIN_C_SOURCE
#endif
#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <sched.h>
#endif

#include "steertts.h"

static const char* usage =
    "usage: steertts-pi --model DIR [--host ADDR] [--port N] [--threads N] [--first-chunk-ms MS] [--cpus LIST]\n"
    "                   [--max-conn N] [--voice V] [--level auto|clip|off] [--quiet]\n"
    "       steertts-pi --model DIR --stdin [--out-dir DIR] [--threads N] [--first-chunk-ms MS] [--voice V]\n"
    "\n"
    "  --model DIR         model directory (default $STEERTTS_MODEL, else ./models/steer-tts-edge)\n"
    "  --host ADDR         address to listen on (default 127.0.0.1; 0.0.0.0 for the whole network)\n"
    "  --port N            TCP port (default 8080)\n"
    "  --threads N         worker threads (default 0 = all CPUs)\n"
    "  --first-chunk-ms MS streaming: the first sentence's first piece holds about MS of audio (default 300;\n"
    "                      0 = one chunk per sentence). Same samples either way.\n"
    "  --cpus LIST         Linux: run on these CPUs only, e.g. 1-3 or 0,2 (the engine's threads inherit it)\n"
    "  --max-conn N        connections handled at once (default 8; requests still render one at a time)\n"
    "  --voice V           default voice for requests that name none (a voice, or female / male)\n"
    "  --level L           default output level: auto (-17 LUFS, peaks <= -1 dBTP; the default), clip, off (raw)\n"
    "  --stdin             read requests from stdin, one per line, instead of serving HTTP\n"
    "  --out-dir DIR       with --stdin: write DIR/NNNN.wav per line instead of PCM to stdout\n"
    "  --quiet             no request log on stderr\n";

#define MAX_HEAD 16384
#define MAX_BODY (1 << 20)

static steertts_engine* g_engine;
static pthread_mutex_t g_engine_mu = PTHREAD_MUTEX_INITIALIZER;
static atomic_int g_conns, g_busy;
static int g_max_conn = 8, g_quiet = 0;
static const char* g_voice;

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void log_msg(const char* fmt, ...) {
  if (g_quiet) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

/* ------------------------------------------------------------------ small string buffer */
typedef struct {
  char* s;
  size_t n, cap;
  int oom;
} sbuf;

static void sb_add(sbuf* b, const char* s, size_t n) {
  if (b->oom) return;
  if (b->n + n + 1 > b->cap) {
    size_t c = b->cap ? b->cap * 2 : 256;
    while (c < b->n + n + 1) c *= 2;
    char* t = (char*)realloc(b->s, c);
    if (!t) {
      b->oom = 1;
      return;
    }
    b->s = t, b->cap = c;
  }
  memcpy(b->s + b->n, s, n);
  b->n += n;
  b->s[b->n] = 0;
}
static void sb_str(sbuf* b, const char* s) { sb_add(b, s, strlen(s)); }
static void sb_printf(sbuf* b, const char* fmt, ...) {
  char tmp[512];
  va_list ap;
  va_start(ap, fmt);
  int k = vsnprintf(tmp, sizeof tmp, fmt, ap);
  va_end(ap);
  if (k > 0) sb_add(b, tmp, (size_t)k < sizeof tmp ? (size_t)k : sizeof tmp - 1);
}
/* a JSON string literal */
static void sb_json(sbuf* b, const char* s) {
  sb_add(b, "\"", 1);
  for (const unsigned char* p = (const unsigned char*)(s ? s : ""); *p; ++p) {
    if (*p == '"' || *p == '\\') {
      char e[2] = {'\\', (char)*p};
      sb_add(b, e, 2);
    } else if (*p < 0x20) {
      char e[8];
      snprintf(e, sizeof e, "\\u%04x", *p);
      sb_add(b, e, 6);
    } else
      sb_add(b, (const char*)p, 1);
  }
  sb_add(b, "\"", 1);
}

/* ------------------------------------------------------------------ PCM / WAV */
static size_t to_s16(const float* y, size_t n, unsigned char* out) {
  for (size_t i = 0; i < n; ++i) {
    float v = y[i];
    v = v > 1.f ? 1.f : (v < -1.f ? -1.f : v);
    long s = v == v ? lrintf(v * 32767.0f) : 0;   /* round half to even, as steertts_save_wav */
    out[2 * i] = (unsigned char)(s & 0xFF);
    out[2 * i + 1] = (unsigned char)((s >> 8) & 0xFF);
  }
  return 2 * n;
}

static void put32(unsigned char* b, uint32_t v) {
  b[0] = (unsigned char)v, b[1] = (unsigned char)(v >> 8), b[2] = (unsigned char)(v >> 16), b[3] = (unsigned char)(v >> 24);
}
/* 44-byte header; data_bytes 0xFFFFFFFF = open length (streaming) */
static void wav_header(unsigned char h[44], uint32_t data_bytes) {
  int sr = STEERTTS_SAMPLE_RATE;
  memcpy(h, "RIFF", 4);
  put32(h + 4, data_bytes == 0xFFFFFFFFu ? 0xFFFFFFFFu : 36 + data_bytes);
  memcpy(h + 8, "WAVEfmt ", 8);
  put32(h + 16, 16);
  h[20] = 1, h[21] = 0, h[22] = 1, h[23] = 0;   /* PCM, mono */
  put32(h + 24, (uint32_t)sr);
  put32(h + 28, (uint32_t)sr * 2);
  h[32] = 2, h[33] = 0, h[34] = 16, h[35] = 0;
  memcpy(h + 36, "data", 4);
  put32(h + 40, data_bytes);
}

/* ------------------------------------------------------------------ sockets */
static int send_all(int fd, const void* data, size_t n) {
  const char* p = (const char*)data;
  while (n) {
#ifdef MSG_NOSIGNAL
    ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
#else
    ssize_t k = send(fd, p, n, 0);
#endif
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return -1;
    p += k, n -= (size_t)k;
  }
  return 0;
}

static const char* status_text(int code) {
  switch (code) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Error";
  }
}

static int respond(int fd, int code, const char* type, const void* body, size_t n, const char* extra) {
  char head[512];
  int k = snprintf(head, sizeof head,
                   "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n%s\r\n", code,
                   status_text(code), type, n, extra ? extra : "");
  if (send_all(fd, head, (size_t)k)) return -1;
  return n ? send_all(fd, body, n) : 0;
}

static int respond_json(int fd, int code, const sbuf* b) {
  return respond(fd, code, "application/json", b->s ? b->s : "", b->n, NULL);
}

static int respond_error(int fd, int code, const char* msg) {
  sbuf b = {0};
  sb_str(&b, "{\"error\": ");
  sb_json(&b, msg);
  sb_str(&b, "}\n");
  int rc = respond_json(fd, code, &b);
  free(b.s);
  return rc;
}

/* ------------------------------------------------------------------ request parsing */
typedef struct {
  char method[8];
  char path[256];
  char* query;    /* points into target, NULL if none */
  char target[4096];
  char* body;
  size_t body_n;
} http_req;

static int hexv(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/* percent-decoding in place ('+' = space, as in forms) */
static void url_decode(char* s) {
  char* o = s;
  for (char* p = s; *p; ++p) {
    if (*p == '+') *o++ = ' ';
    else if (*p == '%' && hexv(p[1]) >= 0 && hexv(p[2]) >= 0) {
      *o++ = (char)(hexv(p[1]) * 16 + hexv(p[2]));
      p += 2;
    } else
      *o++ = *p;
  }
  *o = 0;
}

/* the value of key in a query string (decoded into out), 1 if present */
static int query_get(const char* q, const char* key, char* out, size_t cap) {
  if (!q) return 0;
  size_t kl = strlen(key);
  for (const char* p = q; *p;) {
    const char* amp = strchr(p, '&');
    size_t len = amp ? (size_t)(amp - p) : strlen(p);
    if (len >= kl && !strncmp(p, key, kl) && (len == kl || p[kl] == '=')) {
      size_t vl = len > kl ? len - kl - 1 : 0;
      if (vl >= cap) vl = cap - 1;
      memcpy(out, p + kl + (len > kl), vl);
      out[vl] = 0;
      url_decode(out);
      return 1;
    }
    if (!amp) break;
    p = amp + 1;
  }
  return 0;
}

static const char* find_header(const char* head, const char* name) {
  size_t nl = strlen(name);
  for (const char* p = strstr(head, "\r\n"); p && p[2]; p = strstr(p + 2, "\r\n")) {
    const char* l = p + 2;
    if (!strncasecmp(l, name, nl) && l[nl] == ':') {
      l += nl + 1;
      while (*l == ' ' || *l == '\t') ++l;
      return l;
    }
  }
  return NULL;
}

/* 0 ok, else an HTTP status to answer with */
static int read_request(int fd, http_req* r, char** buf_out) {
  char* buf = (char*)malloc(MAX_HEAD + 1);
  if (!buf) return 500;
  *buf_out = buf;
  size_t n = 0;
  char* end = NULL;
  while (!end) {
    if (n >= MAX_HEAD) return 413;
    ssize_t k = recv(fd, buf + n, MAX_HEAD - n, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) ? 408 : 400;
    n += (size_t)k;
    buf[n] = 0;
    end = strstr(buf, "\r\n\r\n");
  }
  size_t head_n = (size_t)(end - buf) + 4;
  end[2] = 0;   /* the header block, NUL-terminated after its last \r\n */
  if (sscanf(buf, "%7s %4095s", r->method, r->target) != 2) return 400;
  r->query = strchr(r->target, '?');
  if (r->query) *r->query++ = 0;
  size_t pl = strlen(r->target);
  if (pl >= sizeof r->path) return 404;
  memcpy(r->path, r->target, pl + 1);
  const char* te = find_header(buf, "Transfer-Encoding");
  if (te && !strncasecmp(te, "chunked", 7)) return 400;   /* send a Content-Length */
  const char* cl = find_header(buf, "Content-Length");
  unsigned long long want = cl ? strtoull(cl, NULL, 10) : 0;
  if (want > MAX_BODY) return 413;
  r->body = (char*)malloc((size_t)want + 1);
  if (!r->body) return 500;
  size_t have = n - head_n < want ? n - head_n : (size_t)want;
  memcpy(r->body, buf + head_n, have);
  while (have < want) {
    ssize_t k = recv(fd, r->body + have, (size_t)want - have, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) ? 408 : 400;
    have += (size_t)k;
  }
  r->body[want] = 0;
  r->body_n = (size_t)want;
  return 0;
}

/* ------------------------------------------------------------------ requests -> engine */
typedef struct {
  /* GET query form: steertts_options + its strings */
  steertts_options o;
  char text[8192], voice[128], tag[64], ctrl[512], num[64], level[16];
  int use_json;
  const char* json;
  int pcm;   /* format=pcm */
} synth_req;

/* "female" / "male": that gender's default voice (as the CLI's --voice); other names pass through */
static const char* gender_voice(const char* v) {
  if (v && (!strcmp(v, "female") || !strcmp(v, "male"))) {
    const char* g = steertts_gender_voice(g_engine, v);
    return g ? g : v;
  }
  return v;
}

static int parse_synth(const http_req* r, synth_req* s, char* err, size_t err_len) {
  steertts_default_options(&s->o);
  char fmt[16];
  s->pcm = query_get(r->query, "format", fmt, sizeof fmt) && !strcmp(fmt, "pcm");
  if (!strcmp(r->method, "POST")) {
    s->use_json = 1;
    s->json = r->body;
    if (!r->body_n) {
      snprintf(err, err_len, "POST needs a JSON request body, e.g. {\"text\": \"Hello.\"}");
      return -1;
    }
    return 0;
  }
  if (!query_get(r->query, "text", s->text, sizeof s->text)) {
    snprintf(err, err_len, "missing text (GET /synthesize?text=Hello)");
    return -1;
  }
  s->o.voice = query_get(r->query, "voice", s->voice, sizeof s->voice) ? gender_voice(s->voice) : g_voice;
  if (query_get(r->query, "tag", s->tag, sizeof s->tag)) s->o.tag = s->tag;
  if (query_get(r->query, "ctrl", s->ctrl, sizeof s->ctrl)) s->o.ctrl = s->ctrl;
  char* end;
  if (query_get(r->query, "pace", s->num, sizeof s->num)) {
    s->o.pace = strtof(s->num, &end);
    if (end == s->num || *end) return snprintf(err, err_len, "pace: not a number"), -1;
  }
  if (query_get(r->query, "guidance", s->num, sizeof s->num)) {
    s->o.guidance = strtof(s->num, &end);
    if (end == s->num || *end) return snprintf(err, err_len, "guidance: not a number"), -1;
  }
  if (query_get(r->query, "markup", s->num, sizeof s->num)) s->o.markup = !strcmp(s->num, "1") || !strcmp(s->num, "true");
  if (query_get(r->query, "level", s->level, sizeof s->level)) s->o.level = s->level;
  if (query_get(r->query, "gain_db", s->num, sizeof s->num)) {
    s->o.gain_db = strtof(s->num, &end);
    if (end == s->num || *end) return snprintf(err, err_len, "gain_db: not a number"), -1;
    s->o.use_gain_db = 1;
  }
  return 0;
}

/* a JSON request without a "voice" gets the server's default voice: prepended ("last key wins" keeps the request's
 * own); NULL when there is no default voice or the body is not an object */
static char* json_with_voice(const char* json) {
  if (!g_voice) return NULL;
  const char* p = json;
  while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
  if (*p != '{') return NULL;
  sbuf b = {0};
  sb_str(&b, "{\"voice\": ");
  sb_json(&b, g_voice);
  ++p;
  while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
  sb_str(&b, *p == '}' ? "" : ", ");
  sb_str(&b, p);
  if (b.oom) {
    free(b.s);
    return NULL;
  }
  return b.s;
}

typedef struct {
  int fd;
  int pcm;
  int header_sent, failed;
  size_t samples;
  double t0, first_ms;
  unsigned char* tmp;
  size_t tmp_cap;
} stream_out;

static int send_chunk(int fd, const void* data, size_t n) {
  char h[24];
  int k = snprintf(h, sizeof h, "%zx\r\n", n);
  if (send_all(fd, h, (size_t)k) || send_all(fd, data, n) || send_all(fd, "\r\n", 2)) return -1;
  return 0;
}

static int stream_head(stream_out* s) {
  const char* head = "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n"
                     "X-Sample-Rate: 24000\r\nX-Sample-Format: s16le\r\nCache-Control: no-store\r\n\r\n";
  char h[512];
  int k = snprintf(h, sizeof h, head, s->pcm ? "application/octet-stream" : "audio/wav");
  if (send_all(s->fd, h, (size_t)k)) return -1;
  if (!s->pcm) {
    unsigned char wh[44];
    wav_header(wh, 0xFFFFFFFFu);
    if (send_chunk(s->fd, wh, 44)) return -1;
  }
  s->header_sent = 1;
  return 0;
}

static int on_stream_chunk(const float* audio, size_t n, const steertts_chunk_info* info, void* user) {
  (void)info;
  stream_out* s = (stream_out*)user;
  if (!s->header_sent && stream_head(s)) return s->failed = 1;
  if (!n) return 0;
  if (s->first_ms < 0) s->first_ms = now_ms() - s->t0;
  if (2 * n > s->tmp_cap) {
    unsigned char* t = (unsigned char*)realloc(s->tmp, 2 * n);
    if (!t) return s->failed = 1;
    s->tmp = t, s->tmp_cap = 2 * n;
  }
  size_t bytes = to_s16(audio, n, s->tmp);
  if (send_chunk(s->fd, s->tmp, bytes)) return s->failed = 1;   /* the client went away: stop rendering */
  s->samples += n;
  return 0;
}

static void handle_synth(int fd, const http_req* r, int stream) {
  char err[1024] = "";
  synth_req* s = (synth_req*)calloc(1, sizeof *s);
  if (!s) {
    respond_error(fd, 500, "out of memory");
    return;
  }
  if (parse_synth(r, s, err, sizeof err)) {
    respond_error(fd, 400, err);
    free(s);
    return;
  }
  char* jv = s->use_json ? json_with_voice(s->json) : NULL;
  const char* json = jv ? jv : s->json;
  atomic_fetch_add(&g_busy, 1);
  pthread_mutex_lock(&g_engine_mu);
  double t0 = now_ms();
  if (stream) {
    stream_out so = {fd, s->pcm, 0, 0, 0, t0, -1, NULL, 0};
    int rc = s->use_json ? steertts_stream_json(g_engine, json, on_stream_chunk, &so, err, sizeof err)
                         : steertts_stream_ex(g_engine, s->text, &s->o, on_stream_chunk, &so, err, sizeof err);
    double ms = now_ms() - t0;
    pthread_mutex_unlock(&g_engine_mu);
    if (rc && !so.header_sent) respond_error(fd, 400, err);
    else if (!so.failed) {
      if (!so.header_sent && stream_head(&so)) so.failed = 1;   /* nothing to say: an empty stream */
      if (!so.failed) send_all(fd, "0\r\n\r\n", 5);              /* an error after audio began just ends it */
    }
    double audio_ms = so.samples * 1000.0 / STEERTTS_SAMPLE_RATE;
    log_msg("stream %s: %.0f ms audio in %.0f ms (RTF %.3f), first audio %.0f ms%s%s%s\n", rc ? "error" : "ok",
            audio_ms, ms, audio_ms > 0 ? ms / audio_ms : 0.0, so.first_ms, so.failed ? ", client gone" : "",
            rc ? ": " : "", rc ? err : "");
    free(so.tmp);
  } else {
    float* audio = NULL;
    size_t n = 0;
    int rc = s->use_json ? steertts_synthesize_json(g_engine, json, &audio, &n, err, sizeof err)
                         : steertts_synthesize(g_engine, s->text, &s->o, &audio, &n, err, sizeof err);
    double ms = now_ms() - t0;
    pthread_mutex_unlock(&g_engine_mu);
    if (rc) {
      respond_error(fd, 400, err);
      log_msg("synthesize error: %s\n", err);
    } else {
      size_t off = s->pcm ? 0 : 44;
      unsigned char* body = (unsigned char*)malloc(off + 2 * n + 1);
      if (!body) respond_error(fd, 500, "out of memory");
      else {
        if (!s->pcm) wav_header(body, (uint32_t)(2 * n));
        to_s16(audio, n, body + off);
        char extra[160];
        double audio_ms = n * 1000.0 / STEERTTS_SAMPLE_RATE;
        snprintf(extra, sizeof extra, "X-Sample-Rate: 24000\r\nX-Synth-Ms: %.1f\r\nX-Audio-Ms: %.1f\r\n", ms, audio_ms);
        respond(fd, 200, s->pcm ? "application/octet-stream" : "audio/wav", body, off + 2 * n, extra);
        log_msg("synthesize ok: %.0f ms audio in %.0f ms (RTF %.3f)\n", audio_ms, ms, audio_ms > 0 ? ms / audio_ms : 0.0);
        free(body);
      }
      steertts_free(audio);
    }
  }
  atomic_fetch_sub(&g_busy, 1);
  free(jv);
  free(s);
}

static void handle_list(int fd, const char* what) {
  sbuf b = {0};
  sb_str(&b, "[");
  if (!strcmp(what, "voices"))
    for (int i = 0; i < steertts_num_voices(g_engine); ++i) {
      const char* g = steertts_voice_gender(g_engine, i);
      sb_str(&b, i ? ", {\"name\": " : "{\"name\": ");
      sb_json(&b, steertts_voice(g_engine, i));
      sb_str(&b, ", \"gender\": ");
      if (g) sb_json(&b, g);
      else sb_str(&b, "null");
      sb_str(&b, "}");
    }
  else if (!strcmp(what, "tags"))
    for (int i = 0; i < steertts_num_tags(g_engine); ++i) {
      if (i) sb_str(&b, ", ");
      sb_json(&b, steertts_tag(g_engine, i));
    }
  else
    for (int i = 0; i < steertts_num_controls(); ++i) {
      if (i) sb_str(&b, ", ");
      sb_json(&b, steertts_control(i));
    }
  sb_str(&b, "]\n");
  if (b.oom) respond_error(fd, 500, "out of memory");
  else respond_json(fd, 200, &b);
  free(b.s);
}

static void handle_health(int fd) {
  sbuf b = {0};
  sb_str(&b, "{\"status\": \"ok\", \"model\": ");
  sb_json(&b, steertts_model_type(g_engine));
  sb_str(&b, ", \"version\": ");
  sb_json(&b, steertts_version());
  sb_printf(&b, ", \"sample_rate\": %d, \"threads\": %d, \"busy\": %d, \"default_voice\": ", steertts_sample_rate(),
            steertts_threads(g_engine), atomic_load(&g_busy));
  sb_json(&b, g_voice ? g_voice : steertts_default_voice(g_engine));
  sb_str(&b, "}\n");
  respond_json(fd, 200, &b);
  free(b.s);
}

static const char* help_text =
    "steertts-pi: GET /health, /voices, /tags, /controls; POST /synthesize or /stream with a JSON request\n"
    "({\"text\": \"Hello.\", \"ctrl\": {\"f0_end\": 2}}), or GET /synthesize?text=Hello.&voice=male\n"
    "Add ?format=pcm for raw 16-bit 24 kHz mono PCM instead of WAV.\n";

static int known_path(const char* p) {
  return !strcmp(p, "/health") || !strcmp(p, "/voices") || !strcmp(p, "/tags") || !strcmp(p, "/controls") ||
         !strcmp(p, "/");
}

static void* conn_main(void* arg) {
  int fd = (int)(intptr_t)arg;
  struct timeval tv = {30, 0};   /* a client that sends nothing is dropped after 30 s */
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);   /* stream chunks go out as soon as written */
#ifdef SO_NOSIGPIPE
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
  http_req r;
  memset(&r, 0, sizeof r);
  char* buf = NULL;
  int code = read_request(fd, &r, &buf);
  if (code) respond_error(fd, code, status_text(code));
  else {
    int get = !strcmp(r.method, "GET"), post = !strcmp(r.method, "POST");
    const char* p = r.path;
    if (!strcmp(p, "/synthesize") || !strcmp(p, "/stream")) {
      if (get || post) handle_synth(fd, &r, !strcmp(p, "/stream"));
      else respond_error(fd, 405, "use GET or POST");
    } else if (!known_path(p))
      respond_error(fd, 404, "not found: try /health, /voices, /tags, /controls, /synthesize, /stream");
    else if (!get)
      respond_error(fd, 405, "use GET");
    else if (!strcmp(p, "/health"))
      handle_health(fd);
    else if (!strcmp(p, "/"))
      respond(fd, 200, "text/plain; charset=utf-8", help_text, strlen(help_text), NULL);
    else
      handle_list(fd, p + 1);
  }
  free(buf);
  free(r.body);
  shutdown(fd, SHUT_WR);
  close(fd);
  atomic_fetch_sub(&g_conns, 1);
  return NULL;
}

static int serve(const char* host, const char* port) {
  struct addrinfo hints, *ai = NULL;
  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  int g = getaddrinfo(host, port, &hints, &ai);
  if (g) {
    fprintf(stderr, "steertts-pi: %s:%s: %s\n", host, port, gai_strerror(g));
    return 1;
  }
  int ls = -1;
  for (struct addrinfo* a = ai; a; a = a->ai_next) {
    ls = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (ls < 0) continue;
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(ls, a->ai_addr, a->ai_addrlen) == 0 && listen(ls, 64) == 0) break;
    close(ls);
    ls = -1;
  }
  freeaddrinfo(ai);
  if (ls < 0) {
    fprintf(stderr, "steertts-pi: cannot listen on %s:%s: %s\n", host, port, strerror(errno));
    return 1;
  }
  fprintf(stderr, "steertts-pi: %s, %d threads, listening on http://%s:%s\n", steertts_model_type(g_engine),
          steertts_threads(g_engine), host, port);
  for (;;) {
    int fd = accept(ls, NULL, NULL);
    if (fd < 0) {
      if (errno == EINTR || errno == ECONNABORTED) continue;
      fprintf(stderr, "steertts-pi: accept: %s\n", strerror(errno));
      sleep(1);
      continue;
    }
    if (atomic_fetch_add(&g_conns, 1) >= g_max_conn) {
      atomic_fetch_sub(&g_conns, 1);
      respond_error(fd, 503, "too many connections; try again");
      close(fd);
      continue;
    }
    pthread_t th;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&at, 1 << 20);   /* the engine's calling thread runs GEMM tiles on its stack */
    if (pthread_create(&th, &at, conn_main, (void*)(intptr_t)fd)) {
      atomic_fetch_sub(&g_conns, 1);
      respond_error(fd, 503, "cannot start a connection thread");
      close(fd);
    }
    pthread_attr_destroy(&at);
  }
}

/* ------------------------------------------------------------------ --stdin */
typedef struct {
  FILE* out;
  double t0, first_ms;
  size_t samples;
  float* all;
  size_t n, cap;
  int failed;
} line_out;

static int on_line_chunk(const float* audio, size_t n, const steertts_chunk_info* info, void* user) {
  (void)info;
  line_out* l = (line_out*)user;
  if (n && l->first_ms < 0) l->first_ms = now_ms() - l->t0;
  l->samples += n;
  if (l->out) {
    unsigned char tmp[8192];
    for (size_t i = 0; i < n; i += 4096) {
      size_t m = n - i < 4096 ? n - i : 4096;
      size_t b = to_s16(audio + i, m, tmp);
      if (fwrite(tmp, 1, b, l->out) != b) return l->failed = 1;
    }
    if (fflush(l->out)) return l->failed = 1;
    return 0;
  }
  if (l->n + n > l->cap) {
    size_t c = l->cap ? l->cap * 2 : 48000;
    while (c < l->n + n) c *= 2;
    float* t = (float*)realloc(l->all, sizeof(float) * c);
    if (!t) return l->failed = 1;
    l->all = t, l->cap = c;
  }
  memcpy(l->all + l->n, audio, sizeof(float) * n);
  l->n += n;
  return 0;
}

static int run_stdin(const char* out_dir) {
  if (out_dir && mkdir(out_dir, 0755) && errno != EEXIST) {
    fprintf(stderr, "steertts-pi: cannot create %s: %s\n", out_dir, strerror(errno));
    return 1;
  }
  char* line = NULL;
  size_t cap = 0;
  ssize_t len;
  int k = 0, bad = 0;
  while ((len = getline(&line, &cap, stdin)) >= 0) {
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
    const char* p = line;
    while (*p == ' ' || *p == '\t') ++p;
    if (!*p) continue;
    ++k;
    char err[1024] = "";
    line_out lo = {out_dir ? NULL : stdout, now_ms(), -1, 0, NULL, 0, 0, 0};
    int rc;
    if (*p == '{') {
      char* jv = json_with_voice(p);
      rc = steertts_stream_json(g_engine, jv ? jv : p, on_line_chunk, &lo, err, sizeof err);
      free(jv);
    } else {
      steertts_options o;
      steertts_default_options(&o);
      o.voice = g_voice;
      o.markup = 1;
      rc = steertts_stream_ex(g_engine, p, &o, on_line_chunk, &lo, err, sizeof err);
    }
    double ms = now_ms() - lo.t0, audio_ms = lo.samples * 1000.0 / STEERTTS_SAMPLE_RATE;
    if (rc || lo.failed) {
      sbuf b = {0};
      sb_json(&b, rc ? err : "the output was closed");
      fprintf(stderr, "{\"line\": %d, \"error\": %s}\n", k, b.s ? b.s : "\"\"");
      free(b.s);
      ++bad;
      if (lo.failed && !out_dir) {   /* stdout went away */
        free(lo.all);
        break;
      }
    } else if (out_dir) {
      char path[4096];
      snprintf(path, sizeof path, "%s/%04d.wav", out_dir, k);
      float silence = 0.f;
      if (steertts_save_wav(lo.n ? lo.all : &silence, lo.n, path)) {
        fprintf(stderr, "{\"line\": %d, \"error\": \"cannot write %s\"}\n", k, path);
        ++bad;
      } else {
        sbuf b = {0};
        sb_json(&b, path);
        printf("{\"line\": %d, \"wav\": %s, \"audio_ms\": %.1f, \"synth_ms\": %.1f, \"first_audio_ms\": %.1f}\n", k,
               b.s ? b.s : "\"\"", audio_ms, ms, lo.first_ms);
        fflush(stdout);
        free(b.s);
      }
    } else
      log_msg("line %d: %.0f ms audio in %.0f ms, first audio %.0f ms\n", k, audio_ms, ms, lo.first_ms);
    free(lo.all);
  }
  free(line);
  return bad ? 1 : 0;
}

/* ------------------------------------------------------------------ main */
#if defined(__linux__)
static int set_cpus(const char* list) {
  cpu_set_t set;
  CPU_ZERO(&set);
  const char* p = list;
  errno = 0;
  while (*p) {
    char* e;
    long a = strtol(p, &e, 10), b = a;
    if (e == p || a < 0 || a >= CPU_SETSIZE) return -1;
    if (*e == '-') {
      p = e + 1;
      b = strtol(p, &e, 10);
      if (e == p || b < a || b >= CPU_SETSIZE) return -1;
    }
    for (long c = a; c <= b; ++c) CPU_SET((int)c, &set);
    if (*e == ',') ++e;
    else if (*e) return -1;
    p = e;
  }
  return sched_setaffinity(0, sizeof set, &set);
}
#endif

static int count_cpus(void) {
#if defined(__linux__)
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof set, &set) == 0) return CPU_COUNT(&set);
#endif
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (int)n : 1;
}

int main(int argc, char** argv) {
  const char *model = NULL, *host = "127.0.0.1", *port = "8080", *cpus = NULL, *out_dir = NULL, *level = NULL;
  int threads = 0, first_ms = 300, use_stdin = 0;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (fprintf(stderr, "steertts-pi: %s needs a value\n", a), exit(2), ""))
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      fputs(usage, stdout);
      return 0;
    } else if (!strcmp(a, "--version")) {
      printf("steertts-pi %s\n", steertts_version());
      return 0;
    } else if (!strcmp(a, "--model")) model = NEXT();
    else if (!strcmp(a, "--host")) host = NEXT();
    else if (!strcmp(a, "--port")) port = NEXT();
    else if (!strcmp(a, "--threads")) threads = atoi(NEXT());
    else if (!strcmp(a, "--first-chunk-ms")) first_ms = atoi(NEXT());
    else if (!strcmp(a, "--cpus")) cpus = NEXT();
    else if (!strcmp(a, "--max-conn")) g_max_conn = atoi(NEXT());
    else if (!strcmp(a, "--voice")) g_voice = NEXT();
    else if (!strcmp(a, "--level")) level = NEXT();
    else if (!strcmp(a, "--stdin")) use_stdin = 1;
    else if (!strcmp(a, "--out-dir")) out_dir = NEXT();
    else if (!strcmp(a, "--quiet")) g_quiet = 1;
    else {
      fprintf(stderr, "steertts-pi: unknown option %s\n%s", a, usage);
      return 2;
    }
  }
  if (threads < 0 || first_ms < 0 || g_max_conn < 1) {
    fprintf(stderr, "steertts-pi: --threads and --first-chunk-ms take numbers >= 0, --max-conn >= 1\n");
    return 2;
  }
  if (!model) model = getenv("STEERTTS_MODEL");
  if (!model) model = "models/steer-tts-edge";
  signal(SIGPIPE, SIG_IGN);
  if (cpus) {
#if defined(__linux__)
    if (set_cpus(cpus)) {
      fprintf(stderr, "steertts-pi: --cpus %s: %s\n", cpus, errno ? strerror(errno) : "not a CPU list");
      return 2;
    }
#else
    fprintf(stderr, "steertts-pi: --cpus is Linux only; ignored\n");
#endif
  }
  if (threads == 0) threads = count_cpus();   /* the CPUs this process may use (after --cpus) */
  char err[1024];
  double t0 = now_ms();
  steertts_config cfg;
  steertts_default_config(&cfg);
  cfg.threads = threads;
  cfg.level = level;
  g_engine = steertts_create_ex(model, &cfg, err, sizeof err);
  if (!g_engine) {
    fprintf(stderr, "steertts-pi: %s\n", err);
    return 1;
  }
  if (g_voice) {   /* fail at start, not on every request */
    g_voice = gender_voice(g_voice);
    int ok = 0;
    for (int i = 0; !ok && i < steertts_num_voices(g_engine); ++i) ok = !strcmp(steertts_voice(g_engine, i), g_voice);
    if (!ok) {
      fprintf(stderr, "steertts-pi: unknown voice '%s' (GET /voices lists them)\n", g_voice);
      steertts_destroy(g_engine);
      return 2;
    }
  }
  steertts_set_first_chunk_ms(g_engine, first_ms);
  log_msg("steertts-pi: loaded %s in %.0f ms\n", model, now_ms() - t0);
  int rc = use_stdin ? run_stdin(out_dir) : serve(host, port);
  steertts_destroy(g_engine);
  return rc;
}
