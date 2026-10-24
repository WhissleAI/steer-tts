/* Speaker encoder / WAV reader / resampler test against the Python goldens (c/tests/make_golden_wavlm.py).
 *   (a) xvector16 on identical 16 kHz input vs transformers WavLMForXVector (fp32)   cosine >= 0.99999
 *   (b) xvector from 24 kHz input (st_resample inside) vs steertts.speaker.XVec      cosine >= 0.999
 *   (c) the int8 file vs the Python int8 build (round_int8_), 16 kHz and 24 kHz input
 *   (d) st_wav_read on generated 8/16/24/32-bit, float32/64, stereo, extensible, odd-chunk files vs soundfile (exact)
 * plus resampler lengths / SNR vs soxr and timings. Weights: $STEERTTS_V1_MODEL/native (default
 * ../models/steer-tts); exit 77 when missing. Goldens: $STEERTTS_GOLDEN_DIR or
 * golden/wavlm next to this file. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/st_audio.h"
#include "../src/st_base.h"
#include "../src/st_wavlm.h"

static char g_dir[1024];
static int g_fails;

static const float* gold(st_file* g, const char* name, size_t* n) {
  const st_tensor* t = st_file_find(g, name);
  if (!t || t->dtype != ST_F32) {
    fprintf(stderr, "golden tensor %s missing\n", name);
    exit(1);
  }
  *n = (size_t)st_numel(t);
  return (const float*)t->data;   /* safetensors data of an all-F32 file is 4-byte aligned in practice */
}

static void check(int ok, const char* fmt, double a, double b) {
  printf("%s ", ok ? "ok  " : "FAIL");
  printf(fmt, a, b);
  printf("\n");
  g_fails += !ok;
}

static double cosine(const float* a, const float* b, int n, double* maxabs) {
  double ab = 0, aa = 0, bb = 0, m = 0;
  for (int i = 0; i < n; ++i) {
    ab += (double)a[i] * b[i];
    aa += (double)a[i] * a[i];
    bb += (double)b[i] * b[i];
    double d = fabs((double)a[i] - b[i]);
    m = d > m ? d : m;
  }
  *maxabs = m;
  return ab / sqrt(aa * bb);
}

static double snr_db(const float* ref, const float* y, size_t n) {
  double s = 0, e = 0;
  for (size_t i = 0; i < n; ++i) {
    s += (double)ref[i] * ref[i];
    e += ((double)ref[i] - y[i]) * ((double)ref[i] - y[i]);
  }
  return 10 * log10(s / (e > 1e-30 ? e : 1e-30));
}

static float* read_wav(const char* name, size_t* n, int* sr) {
  float* y;
  if (st_wav_read(st_name("%s/%s", g_dir, name), &y, n, sr)) {
    fprintf(stderr, "%s\n", st_geterr());
    exit(1);
  }
  return y;
}

static void test_wav(st_file* g) {
  static const char* names[] = {"u8", "i16", "i24", "i32", "f32", "f64", "i16_stereo", "f32_stereo",
                                "ext_i24_stereo", "ext_f32_3ch", "ext_i16_mono", "chunks"};
  for (size_t c = 0; c < sizeof names / sizeof *names; ++c) {
    size_t n, ng;
    int sr;
    float* y = read_wav(st_name("wav_%s.wav", names[c]), &n, &sr);
    const float* r = gold(g, st_name("wav/%s", names[c]), &ng);
    size_t bad = n == ng ? 0 : 1;
    for (size_t i = 0; i < n && i < ng; ++i) bad += y[i] != r[i];
    printf("%s wav %-15s %zu frames @ %d Hz, %zu mismatches\n", bad ? "FAIL" : "ok  ", names[c], n, sr, bad);
    g_fails += bad != 0;
    st_free(y);
  }
  /* malformed input must fail cleanly */
  float* y;
  size_t n;
  int sr;
  const char junk[] = "RIFF\x10\0\0\0WAVEfmt \x02\0\0\0ab";
  int r = st_wav_read_mem(junk, sizeof junk - 1, &y, &n, &sr);
  printf("%s wav malformed -> error \"%s\"\n", r ? "ok  " : "FAIL", r ? st_geterr() : "");
  g_fails += r == 0;
}

static void test_resample(st_file* g) {
  /* speech: SNR vs soxr >= 60 dB; a full-band chirp (up to 0.95 x 44.1k Nyquist, i.e. through the transition band
   * where the two filters' roll-offs differ) >= 45 dB */
  static const struct { const char* c; int a, b; double min_snr; } cases[] = {{"s24_16", 24000, 16000, 60},
                                                                               {"c44_24", 44100, 24000, 45}};
  for (int c = 0; c < 2; ++c) {
    size_t n, nr, ny;
    const float* x = gold(g, st_name("rs/%s/in", cases[c].c), &n);
    const float* ref = gold(g, st_name("rs/%s/out", cases[c].c), &nr);
    float* y;
    if (st_resample(x, n, cases[c].a, cases[c].b, &y, &ny)) {
      fprintf(stderr, "%s\n", st_geterr());
      exit(1);
    }
    double s = ny == nr ? snr_db(ref, y, ny) : -1;
    printf("%s resample %5d -> %5d: len %zu (soxr %zu), SNR vs soxr %.1f dB\n", ny == nr && s >= cases[c].min_snr ? "ok  " : "FAIL",
           cases[c].a, cases[c].b, ny, nr, s);
    g_fails += !(ny == nr && s >= cases[c].min_snr);
    st_free(y);
  }
  /* output lengths, incl. the soxr rounding of exact halves (24000 -> 44100, n = 120: 220.5 -> 220) */
  static const size_t len_cases[][4] = {{120, 24000, 44100, 220}, {1, 16000, 24000, 2}, {3, 24000, 16000, 2},
                                        {192000, 24000, 16000, 128000}, {7, 48000, 24000, 4}};
  for (int c = 0; c < 5; ++c) {
    size_t l = st_resample_len(len_cases[c][0], (int)len_cases[c][1], (int)len_cases[c][2]);
    printf("%s resample len %zu %zu->%zu = %zu (soxr %zu)\n", l == len_cases[c][3] ? "ok  " : "FAIL", len_cases[c][0],
           len_cases[c][1], len_cases[c][2], l, len_cases[c][3]);
    g_fails += l != len_cases[c][3];
  }
}

static void test_model(st_file* g, const char* path, const char* tag, double min16, double min24, st_pool* p1,
                       st_pool* p4) {
  double t0 = st_time_ms();
  st_weights* w = st_weights_open(path);
  st_wavlm* m = w ? st_wavlm_load(w) : NULL;
  if (!m) {
    fprintf(stderr, "%s\n", st_geterr());
    exit(1);
  }
  printf("     %s: load %.0f ms\n", tag, st_time_ms() - t0);
  size_t n16g, n24g;
  const float* r16 = gold(g, st_name("x16/%s", tag), &n16g);
  const float* r24 = gold(g, st_name("x24/%s", tag), &n24g);
  for (int i = 0; i < 4; ++i) {
    size_t n16, n24;
    int sr16, sr24;
    float* y16 = read_wav(st_name("speech16_%d.wav", i), &n16, &sr16);
    float* y24 = read_wav(st_name("speech24_%d.wav", i), &n24, &sr24);
    float e16[512], e24[512];
    double ma16, ma24;
    double ta = st_time_ms();
    if (st_wavlm_xvector16(m, p4, y16, n16, e16)) {
      fprintf(stderr, "%s\n", st_geterr());
      exit(1);
    }
    double tb = st_time_ms();
    if (st_wavlm_xvector(m, p4, y24, n24, e24)) {
      fprintf(stderr, "%s\n", st_geterr());
      exit(1);
    }
    double tc = st_time_ms();
    double c16 = cosine(e16, r16 + 512 * i, 512, &ma16), c24 = cosine(e24, r24 + 512 * i, 512, &ma24);
    char fmt[256];
    snprintf(fmt, sizeof fmt, "%s %5.2f s  (a/c) 16k: 1-cos %%.1e max|d| %%.2e   [%.0f ms]", tag, n24 / 24000.0, tb - ta);
    check(c16 >= min16, fmt, 1 - c16, ma16);
    snprintf(fmt, sizeof fmt, "%s %5.2f s  (b/c) 24k: 1-cos %%.1e max|d| %%.2e   [%.0f ms]", tag, n24 / 24000.0, tc - tb);
    check(c24 >= min24, fmt, 1 - c24, ma24);
    st_free(y16);
    st_free(y24);
  }
  {
    float one = 0.1f, e[512];
    double nn = 0;
    int r = st_wavlm_xvector(m, p4, &one, 1, e);
    for (int i = 0; i < 512; ++i) nn += (double)e[i] * e[i];
    printf("%s %s 1-sample input -> |e| = %.6f\n", r == 0 && fabs(nn - 1) < 1e-4 ? "ok  " : "FAIL", tag, sqrt(nn));
    g_fails += !(r == 0 && fabs(nn - 1) < 1e-4);
  }
  /* speed: an 8 s clip (the 12 s one, cropped by the model) at 24 kHz, 1 and 4 threads, best of 3 */
  {
    size_t n24;
    int sr;
    float* y24 = read_wav("speech24_3.wav", &n24, &sr);
    float e[512];
    for (int th = 0; th < 2; ++th) {
      double best = 1e30;
      for (int r = 0; r < 3; ++r) {
        double ta = st_time_ms();
        if (st_wavlm_xvector(m, th ? p4 : p1, y24, n24, e)) exit(1);
        double d = st_time_ms() - ta;
        best = d < best ? d : best;
      }
      printf("     %s: 8 s clip, %d thread%s: %.0f ms\n", tag, th ? 4 : 1, th ? "s" : "", best);
    }
    st_free(y24);
  }
  st_wavlm_free(m);
  st_weights_close(w);
}

int main(void) {
  const char* root = getenv("STEERTTS_V1_MODEL");
  if (!root || !*root) root = "../models/steer-tts";
  char fp32[1024], int8[1024];
  snprintf(fp32, sizeof fp32, "%s/native/wavlm_sv.safetensors", root);
  snprintf(int8, sizeof int8, "%s/native/int8/wavlm_sv.safetensors", root);
  const char* gd = getenv("STEERTTS_GOLDEN_DIR");
  if (gd && *gd) snprintf(g_dir, sizeof g_dir, "%s", gd);
  else {
    const char* f = __FILE__;
    const char* s = strrchr(f, '/');
    snprintf(g_dir, sizeof g_dir, "%.*sgolden/wavlm", s ? (int)(s - f + 1) : 0, f);
  }
  st_file* g = st_file_open(st_name("%s/golden.safetensors", g_dir));
  if (!g) {
    fprintf(stderr, "no goldens (%s); set STEERTTS_GOLDEN_DIR\n", st_geterr());
    return 1;
  }
  test_wav(g);
  test_resample(g);
  FILE* f1 = fopen(fp32, "rb");
  FILE* f2 = fopen(int8, "rb");
  if (!f1 || !f2) {
    printf("skip: %s / %s missing (c/tools/export_wavlm.py)\n", fp32, int8);
    if (f1) fclose(f1);
    if (f2) fclose(f2);
    st_file_close(g);
    return g_fails ? 1 : 77;
  }
  fclose(f1);
  fclose(f2);
  st_pool* p1 = st_pool_create(1);
  st_pool* p4 = st_pool_create(4);
  test_model(g, fp32, "fp32", 0.99999, 0.999, p1, p4);
  test_model(g, int8, "int8", 0.99999, 0.999, p1, p4);
  printf("     peak RSS %.0f MB\n", st_peak_rss_kb() / 1024.0);
  st_pool_destroy(p1);
  st_pool_destroy(p4);
  st_file_close(g);
  printf(g_fails ? "%d FAILED\n" : "all passed\n", g_fails);
  return g_fails ? 1 : 0;
}
