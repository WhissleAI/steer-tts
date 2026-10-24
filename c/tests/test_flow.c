/* Flow refiner / prompted flow / mel / prompt_mel / torch randn against golden data from make_golden_flow.py.
 *
 *   test_flow [golden_dir]      model dir: $STEERTTS_V1_MODEL (exit 77 if missing)
 *   FLOW_BENCH=1 test_flow      also times the refiner (fp32 / int8, 1 and 4 threads) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/st_base.h"
#include "../src/st_flow.h"
#include "../src/st_mel.h"
#include "../src/st_rng.h"

static const char* g_dir;
static int g_fails;

static void* load_bin(const char* name, size_t elem, size_t n) {
  char path[1024];
  snprintf(path, sizeof path, "%s/%s.bin", g_dir, name);
  char* data;
  size_t len;
  if (st_read_file(path, &data, &len) != 0) {
    fprintf(stderr, "cannot read %s: %s\n", path, st_geterr());
    exit(1);
  }
  if (n && len != elem * n) {
    fprintf(stderr, "%s: %zu bytes, expected %zu\n", path, len, elem * n);
    exit(1);
  }
  return data;
}

static void compare(const char* what, const float* got, const float* ref, size_t n, double min_snr, double max_abs) {
  double num = 0, den = 0, mx = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = (double)got[i] - ref[i];
    num += d * d, den += (double)ref[i] * ref[i];
    if (fabs(d) > mx || d != d) mx = d != d ? INFINITY : fabs(d);
  }
  double snr = num > 0 ? 10 * log10(den / num) : INFINITY;
  int ok = snr >= min_snr && mx <= max_abs;
  printf("%-34s max|d| %.3g  SNR %6.1f dB  %s\n", what, mx, snr, ok ? "ok" : "FAIL");
  if (!ok) ++g_fails;
}

static uint64_t fnv1a(const float* x, size_t n) {
  const unsigned char* b = (const unsigned char*)x;
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n * 4; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
  return h;
}

static int meta_int(const st_json* meta, const char* sec, const char* key) {
  return (int)st_json_num(st_json_get(st_json_get(meta, sec), key), -1);
}

static st_json* load_json(const char* path) {
  char* text;
  size_t len;
  if (st_read_file(path, &text, &len) != 0) return NULL;
  st_json* j = st_json_parse(text, len);
  st_free(text);
  return j;
}

static void test_randn(const st_json* meta) {
  const st_json* h = st_json_get(meta, "randn_fnv");
  int bad = 0, n_ok = 0;
  for (int i = 0; h && i < h->n; ++i) {
    unsigned long long seed;
    size_t n;
    if (sscanf(h->keys[i], "%llu_%zu", &seed, &n) != 2) continue;
    float* r = (float*)malloc(sizeof(float) * n);
    st_randn_torch(seed, r, n);
    char hex[32];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long)fnv1a(r, n));
    if (strcmp(hex, st_json_str(&h->items[i], ""))) {
      printf("randn seed %llu n %zu: hash mismatch\n", seed, n);
      ++bad;
    } else {
      ++n_ok;
    }
    free(r);
  }
  float* ref = (float*)load_bin("randn_1234_1000", 4, 1000);
  float got[1000];
  st_randn_torch(1234, got, 1000);
  int diff = memcmp(ref, got, sizeof got) != 0;
  st_free(ref);
  printf("%-34s %d/%d (seed, n) cases bit-exact%s\n", "torch.randn", n_ok, n_ok + bad, diff ? ", 1000-array differs" : "");
  if (bad || diff || !n_ok) ++g_fails;
}

static void test_mel(const st_json* meta) {
  int n24 = meta_int(meta, "clip", "n24"), n16 = meta_int(meta, "clip", "n16_24"), mT = meta_int(meta, "clip", "mel_T");
  int16_t* pcm = (int16_t*)load_bin("clip24_i16", 2, (size_t)n24);
  float* y24 = (float*)malloc(sizeof(float) * n24);
  for (int i = 0; i < n24; ++i) y24[i] = pcm[i] / 32768.0f;
  float* ref = (float*)load_bin("clip24_mel", 4, (size_t)mT * 100);
  float* mel;
  int T;
  if (st_mel_spectrogram(y24, (size_t)n24, &mel, &T) || T != mT) {
    printf("mel: failed (%s) T %d vs %d\n", st_geterr(), T, mT);
    ++g_fails;
  } else {
    compare("mel spectrogram (3.4 s clip)", mel, ref, (size_t)T * 100, 90, 2e-3);
    st_free(mel);
  }
  st_free(ref);
  /* prompt_mel: 24 kHz clip and the 16 kHz band-limited copy (resampled to 24 kHz) against p_mu */
  int pT = meta_int(meta, "prompted", "T");
  float* mu = (float*)load_bin("p_mu", 4, (size_t)pT * 100);
  float* y16 = (float*)load_bin("clip16_24k", 4, (size_t)n16);
  const char* names[2] = {"pm24", "pm16"};
  const char* keys[2] = {"P24", "P16"};
  for (int k = 0; k < 2; ++k) {
    int P0 = meta_int(meta, "prompted", keys[k]);
    float* r = (float*)load_bin(names[k], 4, (size_t)P0 * 100);
    float* pm;
    int P;
    int rc = k == 0 ? st_prompt_mel(y24, (size_t)n24, 24000, mu, pT, &pm, &P)
                    : st_prompt_mel(y16, (size_t)n16, 16000, mu, pT, &pm, &P);
    if (rc || !pm || P != P0) {
      printf("prompt_mel %s: failed (%s)\n", names[k], st_geterr());
      ++g_fails;
    } else {
      compare(k == 0 ? "prompt_mel (24 kHz clip)" : "prompt_mel (16 kHz clip, band fill)", pm, r, (size_t)P * 100, 90,
              2e-3);
      st_free(pm);
    }
    st_free(r);
  }
  float* pm;
  int P;
  if (st_prompt_mel(y24, 11999, 24000, mu, pT, &pm, &P) || pm || P) {
    printf("prompt_mel: a < 0.5 s clip must give no prompt\n");
    ++g_fails;
  }
  st_free(mu), st_free(y16), st_free(pcm);
  free(y24);
}

static st_flow* open_flow(const char* model, const st_json* config, const char* entry, const char* sub,
                          st_weights** w) {
  const st_json* e = st_json_get(config, entry);
  char path[1024];
  snprintf(path, sizeof path, "%s/%s%s", model, sub, st_json_str(st_json_get(e, "file"), ""));
  *w = st_weights_open(path);
  if (!*w) {
    fprintf(stderr, "%s\n", st_geterr());
    return NULL;
  }
  st_flow* f = st_flow_load(*w, st_json_get(e, "cfg"));
  if (!f) fprintf(stderr, "flow load %s: %s\n", path, st_geterr());
  return f;
}

int main(int argc, char** argv) {
  static char dir[1024];
  if (argc > 1) g_dir = argv[1];
  else {
    const char* cands[] = {"tests/golden/flow", "c/tests/golden/flow", "../tests/golden/flow"};
    for (int i = 0; i < 3 && !g_dir; ++i) {
      snprintf(dir, sizeof dir, "%s/meta.json", cands[i]);
      FILE* fp = fopen(dir, "rb");
      if (fp) fclose(fp), g_dir = cands[i];
    }
    if (!g_dir) {
      fprintf(stderr, "golden dir not found; pass it as the first argument\n");
      return 1;
    }
  }
  const char* model = getenv("STEERTTS_V1_MODEL");
  if (!model) model = "../models/steer-tts";
  snprintf(dir, sizeof dir, "%s/meta.json", g_dir);
  st_json* meta = load_json(dir);
  if (!meta) {
    fprintf(stderr, "cannot read %s\n", dir);
    return 1;
  }
  test_randn(meta);
  test_mel(meta);

  snprintf(dir, sizeof dir, "%s/config.json", model);
  st_json* config = load_json(dir);
  if (!config) {
    printf("model dir %s missing: skipping the flow tests\n", model);
    st_json_free(meta);
    return g_fails ? 1 : 77;
  }
  st_pool* pool = st_pool_create(4);
  int T = meta_int(meta, "refiner", "T"), seed = meta_int(meta, "refiner", "seed");
  float* mu = (float*)load_bin("ref_mu", 4, (size_t)T * 100);
  float* g = (float*)load_bin("ref_g", 4, 128);
  float* out = (float*)malloc(sizeof(float) * T * 100);
  for (int prec = 0; prec < 2; ++prec) {
    st_weights* w;
    st_flow* f = open_flow(model, config, "refiner", prec ? "int8/" : "", &w);
    if (!f) {
      ++g_fails;
      continue;
    }
    if (!prec) {
      float* r1 = (float*)load_bin("ref_out_fp32_s1", 4, (size_t)T * 100);
      if (st_flow_sample(f, pool, mu, T, g, 1, 0.4f, (uint64_t)seed, NULL, 0, 0.f, out)) ++g_fails;
      compare("refiner fp32, 1 step (2 evals)", out, r1, (size_t)T * 100, 90, 2e-4);
      st_free(r1);
    }
    float* r = (float*)load_bin(prec ? "ref_out_int8" : "ref_out_fp32", 4, (size_t)T * 100);
    double t0 = st_time_ms();
    if (st_flow_sample(f, pool, mu, T, g, 10, 0.4f, (uint64_t)seed, NULL, 0, 0.f, out)) ++g_fails;
    double ms = st_time_ms() - t0;
    char what[64];
    snprintf(what, sizeof what, "refiner %s, 10 steps (%.0f ms)", prec ? "int8" : "fp32", ms);
    compare(what, out, r, (size_t)T * 100, 90, 2e-4);
    st_free(r);
    if (getenv("FLOW_BENCH")) {
      double sec = T * 256.0 / 24000;
      for (int th = 1; th <= 4; th *= 4) {
        st_pool* bp = st_pool_create(th);
        double best = 1e30;
        for (int it = 0; it < 3; ++it) {
          t0 = st_time_ms();
          st_flow_sample(f, bp, mu, T, g, 10, 0.4f, 0, NULL, 0, 0.f, out);
          double e = st_time_ms() - t0;
          best = e < best ? e : best;
        }
        printf("  bench %s %d thr: T=%d (%.2f s audio) %.0f ms, %.1f ms/eval, %.0f ms per s of audio\n",
               prec ? "int8" : "fp32", th, T, sec, best, best / 20, best / sec);
        st_pool_destroy(bp);
      }
    }
    st_flow_free(f);
    st_weights_close(w);
  }
  free(out);
  st_free(mu), st_free(g);

  /* prompted flow on the band-limited clip's prompt_mel (golden pm16, so this checks the flow alone) */
  int pT = meta_int(meta, "prompted", "T"), P = meta_int(meta, "prompted", "P16"), P24 = meta_int(meta, "prompted", "P24");
  int pseed = meta_int(meta, "prompted", "seed");
  mu = (float*)load_bin("p_mu", 4, (size_t)pT * 100);
  g = (float*)load_bin("p_g", 4, 128);
  float* pm = (float*)load_bin("pm16", 4, (size_t)P * 100);
  float* pm24 = (float*)load_bin("pm24", 4, (size_t)P24 * 100);
  out = (float*)malloc(sizeof(float) * pT * 100);
  for (int prec = 0; prec < 2; ++prec) {
    st_weights* w;
    st_flow* f = open_flow(model, config, "prompted_flow", prec ? "int8/" : "", &w);
    if (!f || !st_flow_is_prompted(f)) {
      ++g_fails;
      continue;
    }
    if (!prec) {
      float* r = (float*)load_bin("p_out_fp32_s1", 4, (size_t)pT * 100);
      st_flow_sample(f, pool, mu, pT, g, 1, 0.4f, (uint64_t)pseed, pm, P, 0.f, out);
      compare("prompted fp32, 1 step", out, r, (size_t)pT * 100, 90, 2e-4);
      st_free(r);
      r = (float*)load_bin("p_out_fp32_cfg", 4, (size_t)pT * 100);
      st_flow_sample(f, pool, mu, pT, g, 2, 0.4f, (uint64_t)pseed, pm24, P24, 0.5f, out);
      compare("prompted fp32, 2 steps, cfg 0.5", out, r, (size_t)pT * 100, 90, 2e-4);
      st_free(r);
      r = (float*)load_bin("p_out_noprompt", 4, (size_t)pT * 100);
      st_flow_sample(f, pool, mu, pT, g, 2, 0.4f, (uint64_t)pseed, NULL, 0, 0.f, out);
      compare("prompted fp32, no prompt, 2 steps", out, r, (size_t)pT * 100, 90, 2e-4);
      st_free(r);
    }
    float* r = (float*)load_bin(prec ? "p_out_int8" : "p_out_fp32", 4, (size_t)pT * 100);
    double t0 = st_time_ms();
    if (st_flow_sample(f, pool, mu, pT, g, 10, 0.4f, (uint64_t)pseed, pm, P, 0.f, out)) ++g_fails;
    char what[64];
    snprintf(what, sizeof what, "prompted %s, 10 steps (%.0f ms)", prec ? "int8" : "fp32", st_time_ms() - t0);
    compare(what, out, r, (size_t)pT * 100, 90, 2e-4);
    st_free(r);
    st_flow_free(f);
    st_weights_close(w);
  }
  free(out);
  st_free(mu), st_free(g), st_free(pm), st_free(pm24);
  st_pool_destroy(pool);
  st_json_free(config);
  st_json_free(meta);
  printf(g_fails ? "test_flow: %d FAILED\n" : "test_flow: all passed\n", g_fails);
  return g_fails ? 1 : 0;
}
