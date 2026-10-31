/* Parity test of the acoustic model and the vocoders against PyTorch goldens (tests/make_golden_acoustic.py):
 * exact durations / frame counts, conditioning vector, mel max-abs error and SNR, waveform SNR.
 *
 *   test_acoustic [all|v1|edge] [threads]       parity (default: all, 4 threads)
 *   test_acoustic bench [threads]               time acoustic + vocoder on a ~5 s utterance (v1 fp32, v1 int8, edge)
 *
 * Model dirs: $STEERTTS_V1_MODEL (c_native/v1 layout: config.json, model.safetensors, int8/, native/vocos.safetensors)
 * and $STEERTTS_EDGE_MODEL (edge package). Exit 77 (skip) when they are missing. Goldens: $STEERTTS_GOLDEN or
 * tests/golden/acoustic next to this file. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/st_acoustic.h"
#include "../src/st_base.h"
#include "../src/st_vocoder.h"

#define V1_DEFAULT "../models/steer-tts"
#define EDGE_DEFAULT "../models/steer-tts-edge"

static int g_fail = 0;
static char g_golden[1024];

static const char* envor(const char* k, const char* d) {
  const char* v = getenv(k);
  return v && *v ? v : d;
}

static int exists(const char* path) {
  FILE* f = fopen(path, "rb");
  if (f) fclose(f);
  return f != NULL;
}

/* golden tensor -> malloc'd float / int copy */
static float* gf(const st_file* f, const char* name, int64_t* n) {
  const st_tensor* t = st_file_find(f, name);
  if (!t) return NULL;
  int64_t k = st_numel(t);
  float* o = (float*)malloc(sizeof(float) * (size_t)(k ? k : 1));
  for (int64_t i = 0; i < k; ++i) {
    if (t->dtype == ST_F32) memcpy(&o[i], (const char*)t->data + 4 * i, 4);
    else {
      int32_t v;
      memcpy(&v, (const char*)t->data + 4 * i, 4);
      o[i] = (float)v;
    }
  }
  if (n) *n = k;
  return o;
}

static int* gi(const st_file* f, const char* name, int64_t* n) {
  const st_tensor* t = st_file_find(f, name);
  if (!t || t->dtype != ST_I32) return NULL;
  int64_t k = st_numel(t);
  int* o = (int*)malloc(sizeof(int) * (size_t)(k ? k : 1));
  memcpy(o, t->data, sizeof(int) * (size_t)k);
  if (n) *n = k;
  return o;
}

static double maxabs(const float* a, const float* b, size_t n) {
  double m = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = fabs((double)a[i] - b[i]);
    if (d > m || d != d) m = d;
  }
  return m;
}

static double snr_db(const float* ref, const float* x, size_t n) {
  double s = 0, e = 0;
  for (size_t i = 0; i < n; ++i) {
    s += (double)ref[i] * ref[i];
    e += ((double)ref[i] - x[i]) * ((double)ref[i] - x[i]);
  }
  return e > 0 ? 10 * log10(s / e) : 999.0;
}

static void check(int ok, const char* what) {
  if (!ok) {
    printf("  FAIL: %s\n", what);
    g_fail = 1;
  }
}

typedef struct {
  st_weights* w;
  st_acoustic* a;
  st_json* cfg;
} model;

static int load_model(model* m, const char* dir, const char* wfile) {
  char path[1200];
  char* txt;
  size_t len;
  memset(m, 0, sizeof *m);
  snprintf(path, sizeof path, "%s/config.json", dir);
  if (st_read_file(path, &txt, &len)) return -1;
  m->cfg = st_json_parse(txt, len);
  st_free(txt);
  if (!m->cfg) return -1;
  snprintf(path, sizeof path, "%s/%s", dir, wfile);
  if (!(m->w = st_weights_open(path))) return -1;
  m->a = st_acoustic_load(m->w, st_json_get(st_json_get(m->cfg, "acoustic"), "arch"));
  return m->a ? 0 : -1;
}

static void free_model(model* m) {
  st_acoustic_free(m->a);
  st_weights_close(m->w);
  st_json_free(m->cfg);
}

/* one golden case: condition + infer, compare */
static void run_case(model* m, st_pool* p, const char* name, double min_snr) {
  char path[1200];
  snprintf(path, sizeof path, "%s/%s.safetensors", g_golden, name);
  st_file* f = st_file_open(path);
  if (!f) {
    printf("%s: cannot open golden: %s\n", name, st_geterr());
    g_fail = 1;
    return;
  }
  int64_t N, nw, nmel, ng;
  int* ids = gi(f, "ids", &N);
  int* widx = gi(f, "widx", NULL);
  int* dur_ref = gi(f, "dur", NULL);
  int* tag = gi(f, "tag", NULL);
  int* base = gi(f, "base", NULL);
  float* ctrl = gf(f, "ctrl", NULL);
  float* cmask = gf(f, "cmask", NULL);
  float* gd = gf(f, "guidance", NULL);
  float* pace = gf(f, "pace", NULL);
  float* shifts = gf(f, "shifts", &nw);
  float* g_ref = gf(f, "g", &ng);
  float* g0_ref = gf(f, "g0", NULL);
  float* mel_ref = gf(f, "mel", &nmel);
  float* spk_vec = gf(f, "spk_vec", NULL);
  float* spk64 = gf(f, "spk64", NULL);
  int gdim = st_acoustic_g_dim(m->a);
  float g[512], g0[512], zero[ST_N_CTRL] = {0};
  int n_words = (int)(nw / 3);
  st_shift* sh = (st_shift*)malloc(sizeof(st_shift) * (size_t)(n_words ? n_words : 1));
  for (int i = 0; i < n_words; ++i) sh[i].pitch = shifts[3 * i], sh[i].energy = shifts[3 * i + 1], sh[i].dur = shifts[3 * i + 2];
  int T_ref = (int)(nmel / ST_N_MELS);
  float* mel = NULL;
  int T = 0, *dur = NULL;
  double t0 = st_time_ms();
  int rc = st_acoustic_condition(m->a, spk64, spk_vec, tag[0], ctrl, cmask, g);
  if (!rc && base[0]) rc = st_acoustic_condition(m->a, spk64, spk_vec, 0, zero, zero, g0);
  if (!rc)
    rc = st_acoustic_infer(m->a, p, ids, widx, (int)N, g, base[0] ? g0 : NULL, gd[0], pace[0], sh, n_words, &mel, &T,
                           &dur);
  double ms = st_time_ms() - t0;
  if (rc) {
    printf("%s: error: %s\n", name, st_geterr());
    g_fail = 1;
  } else {
    int dur_eq = 1;
    for (int i = 0; i < N; ++i) dur_eq &= dur[i] == dur_ref[i];
    double ge = maxabs(g, g_ref, (size_t)gdim), g0e = base[0] ? maxabs(g0, g0_ref, (size_t)gdim) : 0;
    printf("%-14s N=%3d T=%4d (ref %4d) dur %s | g maxabs %.2e g0 %.2e", name, (int)N, T, T_ref,
           dur_eq ? "identical" : "DIFFER", ge, g0e);
    if (T == T_ref) {
      double me = maxabs(mel, mel_ref, (size_t)T * ST_N_MELS), sn = snr_db(mel_ref, mel, (size_t)T * ST_N_MELS);
      printf(" | mel maxabs %.2e SNR %.1f dB | %.0f ms\n", me, sn, ms);
      check(sn >= min_snr, "mel SNR");
    } else
      printf("\n");
    check(dur_eq && T == T_ref, "durations / frame count");
    check(ng == gdim, "g size");
  }
  st_free(mel);
  st_free(dur);
  free(ids), free(widx), free(dur_ref), free(tag), free(base), free(ctrl), free(cmask), free(gd), free(pace);
  free(shifts), free(g_ref), free(g0_ref), free(mel_ref), free(spk_vec), free(spk64), free(sh);
  st_file_close(f);
}

static void run_vocoder(const char* label, const char* wpath, int dim, int inter, const char* ref_name, st_pool* p,
                        double min_snr) {
  char path[1200];
  snprintf(path, sizeof path, "%s/vocoder.safetensors", g_golden);
  st_file* f = st_file_open(path);
  st_weights* w = st_weights_open(wpath);
  st_vocoder* v = w ? st_vocoder_load(w, dim, inter, 8, 1024, 256) : NULL;
  if (!f || !v) {
    printf("%s: %s\n", label, st_geterr());
    g_fail = 1;
    st_vocoder_free(v);
    st_weights_close(w);
    st_file_close(f);
    return;
  }
  int64_t nm, nref;
  float* mel = gf(f, "mel", &nm);
  float* ref = gf(f, ref_name, &nref);
  float* wav = NULL;
  size_t n = 0;
  double t0 = st_time_ms();
  int rc = st_vocoder_decode(v, p, mel, (int)(nm / ST_N_MELS), &wav, &n);
  double ms = st_time_ms() - t0;
  if (rc || (int64_t)n != nref) {
    printf("%s: %s (n=%zu ref %lld)\n", label, rc ? st_geterr() : "length mismatch", n, (long long)nref);
    g_fail = 1;
  } else {
    double sn = snr_db(ref, wav, n);
    printf("%-14s T=%d samples=%zu | wav maxabs %.2e SNR %.1f dB | %.0f ms\n", label, (int)(nm / ST_N_MELS), n,
           maxabs(wav, ref, n), sn, ms);
    check(sn >= min_snr, "wav SNR");
  }
  st_free(wav);
  free(mel), free(ref);
  st_vocoder_free(v);
  st_weights_close(w);
  st_file_close(f);
}

/* bench: the first 80 tokens of the long case (~5 s), acoustic then vocoder, best of 3 */
static void bench(const char* label, model* m, const char* voc_path, int dim, int inter, const char* golden, st_pool* p) {
  char path[1200];
  snprintf(path, sizeof path, "%s/%s.safetensors", g_golden, golden);
  st_file* f = st_file_open(path);
  st_weights* vw = st_weights_open(voc_path);
  st_vocoder* v = vw ? st_vocoder_load(vw, dim, inter, 8, 1024, 256) : NULL;
  if (!f || !v) {
    printf("%s: %s\n", label, st_geterr());
    exit(1);
  }
  int64_t N;
  int* ids = gi(f, "ids", &N);
  int* widx = gi(f, "widx", NULL);
  float* spk_vec = gf(f, "spk_vec", NULL);
  float* spk64 = gf(f, "spk64", NULL);
  float* ctrl = gf(f, "ctrl", NULL);
  float* cmask = gf(f, "cmask", NULL);
  if (N > 80) N = 80;
  float g[512];
  double best_a = 1e30, best_v = 1e30;
  int T = 0;
  for (int it = 0; it < 4; ++it) {
    float *mel, *wav;
    size_t n;
    double t0 = st_time_ms();
    if (st_acoustic_condition(m->a, spk64, spk_vec, 0, ctrl, cmask, g) ||
        st_acoustic_infer(m->a, p, ids, widx, (int)N, g, NULL, 1.f, 1.f, NULL, 0, &mel, &T, NULL)) {
      printf("%s\n", st_geterr());
      exit(1);
    }
    double t1 = st_time_ms();
    if (st_vocoder_decode(v, p, mel, T, &wav, &n)) {
      printf("%s\n", st_geterr());
      exit(1);
    }
    double t2 = st_time_ms();
    if (it > 0) {   /* first run warms caches / page-ins */
      best_a = t1 - t0 < best_a ? t1 - t0 : best_a;
      best_v = t2 - t1 < best_v ? t2 - t1 : best_v;
    }
    st_free(mel), st_free(wav);
  }
  double audio_ms = (T - 1) * 256 / 24.0;
  printf("%-10s threads=%d N=%d T=%d audio %.2f s | acoustic %.1f ms, vocoder %.1f ms, total %.1f ms, RTF %.4f\n",
         label, st_pool_size(p), (int)N, T, audio_ms / 1000, best_a, best_v, best_a + best_v,
         (best_a + best_v) / audio_ms);
  free(ids), free(widx), free(spk_vec), free(spk64), free(ctrl), free(cmask);
  st_vocoder_free(v);
  st_weights_close(vw);
  st_file_close(f);
}

int main(int argc, char** argv) {
  const char* what = argc > 1 ? argv[1] : "all";
  int threads = argc > 2 ? atoi(argv[2]) : 4;
  const char* v1 = envor("STEERTTS_V1_MODEL", V1_DEFAULT);
  const char* edge = envor("STEERTTS_EDGE_MODEL", EDGE_DEFAULT);
  const char* gold = getenv("STEERTTS_GOLDEN");
  if (gold && *gold) snprintf(g_golden, sizeof g_golden, "%s", gold);
  else {
    /* tests/golden/acoustic next to this source file, or relative to the working directory */
    const char* src = __FILE__;
    const char* sl = strrchr(src, '/');
    if (sl) snprintf(g_golden, sizeof g_golden, "%.*s/golden/acoustic", (int)(sl - src), src);
    else snprintf(g_golden, sizeof g_golden, "golden/acoustic");
    char probe[1200];
    snprintf(probe, sizeof probe, "%s/vocoder.safetensors", g_golden);
    if (!exists(probe)) snprintf(g_golden, sizeof g_golden, "tests/golden/acoustic");
  }
  char p1[1200], p2[1200], p3[1200], p4[1200], p5[1200];
  snprintf(p1, sizeof p1, "%s/model.safetensors", v1);
  snprintf(p2, sizeof p2, "%s/native/vocos.safetensors", v1);
  snprintf(p3, sizeof p3, "%s/native/int8/vocos.safetensors", v1);
  snprintf(p4, sizeof p4, "%s/acoustic.safetensors", edge);
  snprintf(p5, sizeof p5, "%s/vocoder.safetensors", edge);
  int do_v1 = strcmp(what, "edge") != 0, do_edge = strcmp(what, "v1") != 0;
  if ((do_v1 && (!exists(p1) || !exists(p2) || !exists(p3))) || (do_edge && (!exists(p4) || !exists(p5)))) {
    printf("skip: model files missing (STEERTTS_V1_MODEL=%s, STEERTTS_EDGE_MODEL=%s; v1 needs native/vocos from "
           "tools/export_vocos.py)\n", v1, edge);
    return 77;
  }
  st_pool* p = st_pool_create(threads);
  model m;
  if (!strcmp(what, "bench")) {
    if (load_model(&m, v1, "model.safetensors")) { printf("%s\n", st_geterr()); return 1; }
    bench("v1 fp32", &m, p2, 512, 1536, "v1_long", p);
    free_model(&m);
    if (load_model(&m, v1, "int8/model.safetensors")) { printf("%s\n", st_geterr()); return 1; }
    bench("v1 int8", &m, p3, 512, 1536, "v1int8_steer", p);
    free_model(&m);
    if (load_model(&m, edge, "acoustic.safetensors")) { printf("%s\n", st_geterr()); return 1; }
    bench("edge", &m, p5, 384, 896, "edge_long", p);
    free_model(&m);
    st_pool_destroy(p);
    return 0;
  }
  printf("goldens: %s, %d threads\n", g_golden, st_pool_size(p));
  if (do_v1) {
    if (load_model(&m, v1, "model.safetensors")) { printf("v1: %s\n", st_geterr()); return 1; }
    run_case(&m, p, "v1_plain", 50);
    run_case(&m, p, "v1_steer", 50);
    run_case(&m, p, "v1_long", 50);
    free_model(&m);
    if (load_model(&m, v1, "int8/model.safetensors")) { printf("v1 int8: %s\n", st_geterr()); return 1; }
    run_case(&m, p, "v1int8_steer", 50);
    free_model(&m);
    run_vocoder("vocos fp32", p2, 512, 1536, "wav_v1", p, 30);
    run_vocoder("vocos int8", p3, 512, 1536, "wav_v1int8", p, 30);
  }
  if (do_edge) {
    if (load_model(&m, edge, "acoustic.safetensors")) { printf("edge: %s\n", st_geterr()); return 1; }
    run_case(&m, p, "edge_steer", 50);
    run_case(&m, p, "edge_long", 50);
    free_model(&m);
    run_vocoder("edge vocoder", p5, 384, 896, "wav_edge", p, 30);
  }
  st_pool_destroy(p);
  printf("peak RSS %zu MB\n%s\n", st_peak_rss_kb() / 1024, g_fail ? "FAILED" : "all passed");
  return g_fail;
}
