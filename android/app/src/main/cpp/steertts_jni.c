/* JNI bridge for ai.whissle.steertts.demo.SteerTTS: one engine per handle (use it from one thread at a time), sentence
 * streaming into a Java callback or one plain-text render per call, cancellation between chunks.
 *
 * Thread placement. Phone CPUs mix fast and slow cores, and the engine's worker threads meet at a barrier after every
 * layer, so one worker on a slow core holds the others up (Pixel 10, 1 + 5 fast cores and 2 slow ones: 8 unpinned
 * threads render 7x slower than 4). Android also changes which CPUs a process may use: the foreground app gets all of
 * them, a background one (screen off, a TTS service called from elsewhere) a few slow ones. Before every request
 * place() therefore asks the kernel which CPUs are allowed now, takes the fast ones among them, pins the engine thread
 * there and, if that set changed, rebuilds the worker pool (its threads inherit the mask) with one thread per allowed
 * fast core, up to 6, or the requested count if lower. A pool sized for 6 cores but squeezed onto 1 renders 20x slower
 * than real time would allow; resizing costs well under a millisecond.
 *
 * Clock ramp-up. Between requests the cores idle at low frequency and the governor takes ~100 ms to raise it, which is
 * longer than a sentence renders at full speed (first-chunk latency went from ~30 ms to 100-500 ms). An ADPF
 * performance hint session (APerformanceHint, Android 13+, looked up at run time: minSdk is lower) covers the engine
 * thread and the workers; every request starts with notifyWorkloadIncrease (Android 16+) and reports how long it took
 * against a target, so the governor boosts as the burst starts instead of after it. */
#define _GNU_SOURCE
#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>
#include <sched.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "steertts.h"

enum { MAX_CPUS = 64, AUTO_MAX = 6, MAX_HINT_TIDS = 32 };
#define HINT_TARGET_NS 30000000LL   /* per request: a sentence renders in ~30 ms on boosted fast cores */

/* ADPF, resolved from libandroid.so: NULL where the system does not have it */
typedef struct APerformanceHintManager APerformanceHintManager;
typedef struct APerformanceHintSession APerformanceHintSession;
static struct {
  APerformanceHintManager* (*get_manager)(void);
  APerformanceHintSession* (*create)(APerformanceHintManager*, const int32_t*, size_t, int64_t);
  int (*report)(APerformanceHintSession*, int64_t);
  void (*close)(APerformanceHintSession*);
  int (*set_threads)(APerformanceHintSession*, const pid_t*, size_t);                     /* API 34 */
  int (*workload_increase)(APerformanceHintSession*, bool, bool, const char*);            /* API 36 */
  int loaded;
} adpf;

typedef struct {
  steertts_engine* e;
  atomic_int cancel;
  int want;           /* requested threads, <= 0 = auto */
  cpu_set_t placed;   /* the CPU set the pool was built for */
  int placed_ok;
  APerformanceHintSession* hint;   /* NULL: no ADPF */
  int32_t hint_tids[MAX_HINT_TIDS];   /* the threads the session covers: engine thread first, then the workers */
  int n_hint_tids;
  int64_t t0;                         /* start of the current request, CLOCK_MONOTONIC ns */
} handle;

typedef struct {
  JNIEnv* env;
  jobject sink;
  jmethodID on_chunk;
  handle* h;
} stream_ctx;

/* capacity of each CPU (cpu_capacity, else the maximum frequency), read once */
static long g_cap[MAX_CPUS];
static int g_ncpu = -1;
static long g_top;

static void read_capacities(void) {
  if (g_ncpu >= 0) return;
  int n = 0;
  for (; n < MAX_CPUS; ++n) {
    char path[96];
    long v = -1;
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", n);
    FILE* f = fopen(path, "r");
    if (!f) {
      snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", n);
      f = fopen(path, "r");
    }
    if (!f) break;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    g_cap[n] = v;
    if (v > g_top) g_top = v;
  }
  g_ncpu = n;
}

/* Pins the calling (engine) thread to the fast CPUs it may use now and resizes the pool when that set changed. */
static void place(handle* h) {
  read_capacities();
  cpu_set_t all, allowed, fast;
  CPU_ZERO(&all);
  for (int i = 0; i < (g_ncpu > 0 ? g_ncpu : MAX_CPUS); ++i) CPU_SET(i, &all);
  /* widen first: the kernel intersects the request with the process's cpuset, so reading back gives what is allowed */
  sched_setaffinity(0, sizeof all, &all);
  if (sched_getaffinity(0, sizeof allowed, &allowed)) return;
  CPU_ZERO(&fast);
  for (int i = 0; i < g_ncpu && g_top > 0; ++i)
    if (CPU_ISSET(i, &allowed) && g_cap[i] * 2 >= g_top) CPU_SET(i, &fast);
  if (CPU_COUNT(&fast) == 0) fast = allowed;   /* only slow cores allowed (or no capacity information) */
  sched_setaffinity(0, sizeof fast, &fast);
  if (h->placed_ok && CPU_EQUAL(&fast, &h->placed)) return;
  int n = CPU_COUNT(&fast), cap = h->want > 0 ? h->want : AUTO_MAX;
  if (steertts_set_threads(h->e, n < 1 ? 1 : n < cap ? n : cap) == 0) {
    h->placed = fast;
    h->placed_ok = 1;
  }
}

static void load_adpf(void) {
  if (adpf.loaded) return;
  adpf.loaded = 1;
  void* lib = dlopen("libandroid.so", RTLD_NOW);
  if (!lib) return;
  adpf.get_manager = (APerformanceHintManager * (*)(void)) dlsym(lib, "APerformanceHint_getManager");
  adpf.create = (APerformanceHintSession * (*)(APerformanceHintManager*, const int32_t*, size_t, int64_t))
      dlsym(lib, "APerformanceHint_createSession");
  adpf.report = (int (*)(APerformanceHintSession*, int64_t))dlsym(lib, "APerformanceHint_reportActualWorkDuration");
  adpf.close = (void (*)(APerformanceHintSession*))dlsym(lib, "APerformanceHint_closeSession");
  adpf.set_threads = (int (*)(APerformanceHintSession*, const pid_t*, size_t))dlsym(lib, "APerformanceHint_setThreads");
  adpf.workload_increase = (int (*)(APerformanceHintSession*, bool, bool, const char*))dlsym(
      lib, "APerformanceHint_notifyWorkloadIncrease");
  if (!adpf.get_manager || !adpf.create || !adpf.report || !adpf.close) adpf.get_manager = NULL;
}

static int64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* Points the hint session at the calling thread and the current workers (creating it on first use). */
static void hint_threads(handle* h) {
  load_adpf();
  if (!adpf.get_manager) return;
  int32_t t[MAX_HINT_TIDS];
  t[0] = gettid();
  int n = 1 + steertts_worker_tids(h->e, t + 1, MAX_HINT_TIDS - 1);
  if (h->hint && n == h->n_hint_tids && !memcmp(t, h->hint_tids, sizeof(int32_t) * (size_t)n)) return;
  if (h->hint && adpf.set_threads && adpf.set_threads(h->hint, t, (size_t)n) == 0) {
    memcpy(h->hint_tids, t, sizeof t);
    h->n_hint_tids = n;
    return;
  }
  if (h->hint) adpf.close(h->hint);   /* no setThreads (Android 13) or it failed: a new session */
  APerformanceHintManager* m = adpf.get_manager();
  h->hint = m ? adpf.create(m, t, (size_t)n, HINT_TARGET_NS) : NULL;
  h->n_hint_tids = h->hint ? n : 0;
  if (h->hint) memcpy(h->hint_tids, t, sizeof t);
  __android_log_print(ANDROID_LOG_INFO, "SteerTTS", "ADPF session %s (%d threads, workload hints %s)",
                      h->hint ? "on" : "unavailable", n, adpf.workload_increase ? "yes" : "no");
}

/* Before a request: placement, then tell the governor a burst starts now. */
static void begin_request(handle* h) {
  place(h);
  hint_threads(h);
  if (h->hint && adpf.workload_increase) adpf.workload_increase(h->hint, true, false, "steertts");
  h->t0 = now_ns();
}

static void end_request(handle* h) {
  if (h->hint) adpf.report(h->hint, now_ns() - h->t0);
}

static void throw_rt(JNIEnv* env, const char* msg) {
  jclass c = (*env)->FindClass(env, "java/lang/RuntimeException");
  if (c) (*env)->ThrowNew(env, c, msg);
}

static int on_chunk(const float* audio, size_t n, const steertts_chunk_info* info, void* user) {
  stream_ctx* s = (stream_ctx*)user;
  if (atomic_load(&s->h->cancel)) return 1;
  JNIEnv* env = s->env;
  jfloatArray a = (*env)->NewFloatArray(env, (jsize)n);
  if (!a) return 1;   /* OutOfMemoryError pending */
  (*env)->SetFloatArrayRegion(env, a, 0, (jsize)n, audio);
  jstring text = (*env)->NewStringUTF(env, info->text);
  jboolean more = (*env)->CallBooleanMethod(env, s->sink, s->on_chunk, a, info->index, info->synth_ms, text);
  (*env)->DeleteLocalRef(env, a);
  (*env)->DeleteLocalRef(env, text);
  return (*env)->ExceptionCheck(env) || !more;
}

JNIEXPORT jlong JNICALL Java_ai_whissle_steertts_demo_SteerTTS_nativeCreate(JNIEnv* env, jclass cls, jstring dir,
                                                                             jint threads) {
  const char* d = (*env)->GetStringUTFChars(env, dir, NULL);
  char err[512];
  steertts_engine* e = steertts_create(d, 1, err, sizeof err);   /* place() sizes the pool */
  (*env)->ReleaseStringUTFChars(env, dir, d);
  if (!e) {
    throw_rt(env, err);
    return 0;
  }
  handle* h = (handle*)calloc(1, sizeof *h);
  if (!h) {
    steertts_destroy(e);
    throw_rt(env, "out of memory");
    return 0;
  }
  h->e = e;
  h->want = threads;
  place(h);
  return (jlong)(intptr_t)h;
}

JNIEXPORT jint JNICALL Java_ai_whissle_steertts_demo_SteerTTS_nativeThreads(JNIEnv* env, jclass cls, jlong p) {
  handle* h = (handle*)(intptr_t)p;
  return h ? steertts_threads(h->e) : 0;
}

JNIEXPORT void JNICALL Java_ai_whissle_steertts_demo_SteerTTS_nativeDestroy(JNIEnv* env, jclass cls, jlong p) {
  handle* h = (handle*)(intptr_t)p;
  if (!h) return;
  if (h->hint) adpf.close(h->hint);
  steertts_destroy(h->e);
  free(h);
}

JNIEXPORT jobjectArray JNICALL Java_ai_whissle_steertts_demo_SteerTTS_nativeVoices(JNIEnv* env, jclass cls, jlong p,
                                                                                   jstring gender) {
  handle* h = (handle*)(intptr_t)p;
  if (!h) return NULL;
  const char* g = gender ? (*env)->GetStringUTFChars(env, gender, NULL) : NULL;
  int n = 0, total = steertts_num_voices(h->e);
  for (int i = 0; i < total; ++i) {
    const char* vg = steertts_voice_gender(h->e, i);
    n += !g || (vg && !strcmp(vg, g));
  }
  jobjectArray out = (*env)->NewObjectArray(env, n, (*env)->FindClass(env, "java/lang/String"), NULL);
  for (int i = 0, k = 0; out && i < total; ++i) {
    const char* vg = steertts_voice_gender(h->e, i);
    if (g && !(vg && !strcmp(vg, g))) continue;
    jstring s = (*env)->NewStringUTF(env, steertts_voice(h->e, i));
    (*env)->SetObjectArrayElement(env, out, k++, s);
    (*env)->DeleteLocalRef(env, s);
  }
  if (g) (*env)->ReleaseStringUTFChars(env, gender, g);
  return out;
}

JNIEXPORT jobjectArray JNICALL Java_ai_whissle_steertts_demo_SteerTTS_nativeTags(JNIEnv* env, jclass cls, jlong p) {
  handle* h = (handle*)(intptr_t)p;
  if (!h) return NULL;
  int n = steertts_num_tags(h->e);
  jobjectArray out = (*env)->NewObjectArray(env, n, (*env)->FindClass(env, "java/lang/String"), NULL);
  for (int i = 0; out && i < n; ++i) {
    jstring s = (*env)->NewStringUTF(env, steertts_tag(h->e, i));
    (*env)->SetObjectArrayElement(env, out, i, s);
    (*env)->DeleteLocalRef(env, s);
  }
  return out;
}

JNIEXPORT jstring JNICALL Java_ai_whissle_steertts_demo_SteerTTS_nativeVoiceGender(JNIEnv* env, jclass cls, jlong p,
                                                                                  jstring voice) {
  handle* h = (handle*)(intptr_t)p;
  if (!h || !voice) return NULL;
  const char* v = (*env)->GetStringUTFChars(env, voice, NULL);
  const char* g = NULL;
  for (int i = 0; i < steertts_num_voices(h->e); ++i)
    if (!strcmp(steertts_voice(h->e, i), v)) g = steertts_voice_gender(h->e, i);
  (*env)->ReleaseStringUTFChars(env, voice, v);
  return g ? (*env)->NewStringUTF(env, g) : NULL;
}

JNIEXPORT jstring JNICALL Java_ai_whissle_steertts_demo_SteerTTS_nativeGenderVoice(JNIEnv* env, jclass cls, jlong p,
                                                                                  jstring gender) {
  handle* h = (handle*)(intptr_t)p;
  if (!h || !gender) return NULL;
  const char* g = (*env)->GetStringUTFChars(env, gender, NULL);
  const char* v = steertts_gender_voice(h->e, g);
  (*env)->ReleaseStringUTFChars(env, gender, g);
  return v ? (*env)->NewStringUTF(env, v) : NULL;
}

JNIEXPORT void JNICALL Java_ai_whissle_steertts_demo_SteerTTS_nativeCancel(JNIEnv* env, jclass cls, jlong p) {
  handle* h = (handle*)(intptr_t)p;
  if (h) atomic_store(&h->cancel, 1);
}

typedef struct {
  const char *text, *voice, *tag, *ctrl;
} jstrs;

static void get_strs(JNIEnv* env, jstring t, jstring v, jstring tg, jstring c, jstrs* s) {
  s->text = (*env)->GetStringUTFChars(env, t, NULL);
  s->voice = v ? (*env)->GetStringUTFChars(env, v, NULL) : NULL;
  s->tag = tg ? (*env)->GetStringUTFChars(env, tg, NULL) : NULL;
  s->ctrl = c ? (*env)->GetStringUTFChars(env, c, NULL) : NULL;
}

static void release_strs(JNIEnv* env, jstring t, jstring v, jstring tg, jstring c, jstrs* s) {
  (*env)->ReleaseStringUTFChars(env, t, s->text);
  if (s->voice) (*env)->ReleaseStringUTFChars(env, v, s->voice);
  if (s->tag) (*env)->ReleaseStringUTFChars(env, tg, s->tag);
  if (s->ctrl) (*env)->ReleaseStringUTFChars(env, c, s->ctrl);
}

static void options(const jstrs* s, jfloat pace, int markup, steertts_options* o) {
  steertts_default_options(o);
  o->voice = s->voice;
  o->tag = s->tag;
  o->ctrl = s->ctrl && *s->ctrl ? s->ctrl : NULL;
  o->pace = pace;
  o->markup = markup;
}

/* Clears a cancel() that arrived while nothing (or a finished render) was running: the next request starts clean. */
JNIEXPORT void JNICALL Java_ai_whissle_steertts_demo_SteerTTS_nativeResetCancel(JNIEnv* env, jclass cls, jlong p) {
  handle* h = (handle*)(intptr_t)p;
  if (h) atomic_store(&h->cancel, 0);
}

/* Streams `text` with inline markup parsed to sink.onChunk(float[] audio, int index, double synthMs, String text),
 * which returns false to stop. Throws RuntimeException with the library's message on an invalid request. */
JNIEXPORT void JNICALL Java_ai_whissle_steertts_demo_SteerTTS_nativeStream(JNIEnv* env, jclass cls, jlong p,
                                                                          jstring text, jstring voice, jstring tag,
                                                                          jstring ctrl, jfloat pace, jobject sink) {
  handle* h = (handle*)(intptr_t)p;
  if (!h || !text || !sink) return;
  atomic_store(&h->cancel, 0);
  stream_ctx s = {env, sink, NULL, h};
  jclass sc = (*env)->GetObjectClass(env, sink);
  s.on_chunk = (*env)->GetMethodID(env, sc, "onChunk", "([FIDLjava/lang/String;)Z");
  if (!s.on_chunk) return;   /* NoSuchMethodError pending */
  begin_request(h);
  jstrs js;
  get_strs(env, text, voice, tag, ctrl, &js);
  steertts_options o;
  options(&js, pace, 1, &o);
  char err[512];
  int rc = steertts_stream_ex(h->e, js.text, &o, on_chunk, &s, err, sizeof err);
  end_request(h);
  release_strs(env, text, voice, tag, ctrl, &js);
  if (rc && !(*env)->ExceptionCheck(env)) throw_rt(env, err);
}

/* One render with a style tag; markup != 0 parses inline markup (*word*, [pause 300ms]), otherwise the text is read as
 * written (system text such as "5*3 <b>"). null if cancelled. */
JNIEXPORT jfloatArray JNICALL Java_ai_whissle_steertts_demo_SteerTTS_nativeSynthesize(JNIEnv* env, jclass cls, jlong p,
                                                                                     jstring text, jstring voice,
                                                                                     jstring tag, jstring ctrl,
                                                                                     jfloat pace, jboolean markup) {
  handle* h = (handle*)(intptr_t)p;
  if (!h || !text) return NULL;
  if (atomic_exchange(&h->cancel, 0)) return NULL;
  begin_request(h);
  jstrs js;
  get_strs(env, text, voice, tag, ctrl, &js);
  steertts_options o;
  options(&js, pace, markup ? 1 : 0, &o);
  float* y = NULL;
  size_t n = 0;
  char err[512];
  int rc = steertts_synthesize(h->e, js.text, &o, &y, &n, err, sizeof err);
  end_request(h);
  release_strs(env, text, voice, tag, ctrl, &js);
  if (rc) {
    throw_rt(env, err);
    return NULL;
  }
  jfloatArray a = (*env)->NewFloatArray(env, (jsize)n);
  if (a) (*env)->SetFloatArrayRegion(env, a, 0, (jsize)n, y);
  steertts_free(y);
  return a;
}
