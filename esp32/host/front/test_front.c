/* Conformance test of the device front end against the reference golden files (converted by golden_to_tsv.py).
 *
 *   test_front front.bin golden.tsv [-v]
 * P lines: normalize + to_phonemes (ids, word indices, errors); S: the shared normalisation cases; G: the GRU G2P;
 * M: markup segments. Prints exact-match rates and the first mismatches of each kind. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nano_front.h"

static int unhex(const char* h, char* out, int cap) {
  if (!strcmp(h, "-")) return -1;
  int n = 0;
  for (; h[0] && h[1] && n < cap - 1; h += 2) {
    int a = h[0] <= '9' ? h[0] - '0' : h[0] - 'a' + 10, b = h[1] <= '9' ? h[1] - '0' : h[1] - 'a' + 10;
    out[n++] = (char)(a * 16 + b);
  }
  out[n] = 0;
  return n;
}
static int ints(const char* s, int* v, int cap) {
  if (!strcmp(s, "-")) return -1;
  int n = 0;
  for (char* p = (char*)s; *p && n < cap;) {
    char* e;
    long x = strtol(p, &e, 10);
    if (e == p) break;
    v[n++] = (int)x;
    p = e;
  }
  return n;
}
static void f32hex(float v, char* o) {
  unsigned char b[4];
  memcpy(b, &v, 4);
  sprintf(o, "%02x%02x%02x%02x", b[0], b[1], b[2], b[3]);
}

typedef struct {
  long n, ok;
} stat;
static int verbose = 0, shown[8];
static void report(int kind, const char* what, const char* text, const char* detail) {
  if (shown[kind]++ >= (verbose ? 40 : 6)) return;
  printf("MISMATCH %-8s text=\"%.150s\"\n    %.700s\n", what, text, detail);
}

static const char* TAGS[] = {
    "none", "adoration", "amazement", "amusement", "anger", "anxiety", "apologetic", "bored", "confidence",
    "confusion", "contempt", "contentment", "cuteness", "desire", "determined", "disappointment", "disgust",
    "distress", "ecstasy", "embarrassment", "emphasis", "enunciated", "excited", "fast", "fear", "frustration",
    "guilt", "happiness", "hesitant", "high_pitch", "interest", "irritation", "laughing", "loud", "low_pitch",
    "mysterious", "neutral", "nostalgia", "pain", "panic", "pride", "realization", "relief", "sadness", "sarcasm",
    "serenity", "skeptical", "slow", "sympathy", "tired", "whisper"};

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s front.bin golden.tsv [-v]\n", argv[0]);
    return 2;
  }
  verbose = argc > 3;
  FILE* f = fopen(argv[1], "rb");
  if (!f) return 2;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  void* blob = NULL;
  if (posix_memalign(&blob, 64, (size_t)len) || fread(blob, 1, (size_t)len, f) != (size_t)len) return 2;
  fclose(f);
  static nano_front_ctx C;
  if (nano_front_open(&C, blob, (size_t)len)) {
    printf("%s\n", C.err);
    return 1;
  }
  FILE* g = fopen(argv[2], "r");
  if (!g) return 2;
  static char line[1 << 20], text[1 << 16], want[1 << 16], got[1 << 16], werr[4096], why[8192];
  static int wi[8192], ww[8192], gi[8192], gw[8192];
  stat norm = {0, 0}, ids = {0, 0}, widx = {0, 0}, errs = {0, 0}, shared = {0, 0}, gru = {0, 0}, mk = {0, 0},
       mkerr = {0, 0}, errmsg = {0, 0};
  double t_ph = 0, t_gru = 0, t_mk = 0;
  long n_ph = 0;
  while (fgets(line, sizeof line, g)) {
    line[strcspn(line, "\n")] = 0;
    char* fld[8] = {0};
    int nf = 0;
    for (char* p = line; nf < 8;) {
      fld[nf++] = p;
      char* t = strchr(p, '\t');
      if (!t) break;
      *t = 0;
      p = t + 1;
    }
    if (fld[0][0] == 'P' && nf >= 7) {
      unhex(fld[1], text, sizeof text);
      int have_norm = unhex(fld[2], want, sizeof want) >= 0;
      int nerr = unhex(fld[3], werr, sizeof werr) >= 0;
      norm.n++;
      int r = nano_front_normalize(&C, text, got, (int)sizeof got);
      if (nerr) {
        errs.n++;
        if (r < 0) {
          errs.ok++;
          norm.ok++;
          errmsg.n++;
          if (!strncmp(C.err, werr, 40)) errmsg.ok++;
        } else {
          snprintf(why, sizeof why, "expected error %s, got \"%s\"", werr, got);
          report(0, "norm-err", text, why);
        }
      } else if (r >= 0 && have_norm && !strcmp(got, want)) {
        norm.ok++;
      } else {
        snprintf(why, sizeof why, "py=\"%s\"\n    c =\"%s\" %s", want, r >= 0 ? got : "", r < 0 ? C.err : "");
        report(0, "norm", text, why);
      }
      int nwi = ints(fld[4], wi, 8192), nww = ints(fld[5], ww, 8192);
      if (nwi < 0) continue;   /* error case: counted above */
      ids.n++;
      widx.n++;
      clock_t t0 = clock();
      int n = nano_front_phonemes(&C, text, gi, gw, 8192);
      t_ph += (double)(clock() - t0) / CLOCKS_PER_SEC;
      n_ph++;
      int same = n == nwi && !memcmp(gi, wi, sizeof(int) * (size_t)(n > 0 ? n : 0));
      if (same) {
        ids.ok++;
        if (nww == n && !memcmp(gw, ww, sizeof(int) * (size_t)n)) widx.ok++;
        else report(1, "widx", text, "word indices differ");
      } else {
        int k = 0, o = 0;
        o += snprintf(why + o, sizeof why - o, "py:");
        for (k = 0; k < nwi && o < 3000; k++) o += snprintf(why + o, sizeof why - o, " %d", wi[k]);
        o += snprintf(why + o, sizeof why - o, "\n    c :");
        for (k = 0; k < n && o < 6000; k++) o += snprintf(why + o, sizeof why - o, " %d", gi[k]);
        if (n < 0) snprintf(why + o, sizeof why - o, " error %s", C.err);
        report(1, "ids", text, why);
      }
    } else if (fld[0][0] == 'S' && nf >= 3) {
      unhex(fld[1], text, sizeof text);
      unhex(fld[2], want, sizeof want);
      shared.n++;
      int r = nano_front_normalize(&C, text, got, (int)sizeof got);
      if (r >= 0 && !strcmp(got, want))
        shared.ok++;
      else {
        snprintf(why, sizeof why, "py=\"%s\" c=\"%s\"", want, r >= 0 ? got : C.err);
        report(2, "shared", text, why);
      }
    } else if (fld[0][0] == 'G' && nf >= 3) {
      unhex(fld[1], text, sizeof text);
      int np = ints(fld[2], wi, 64);
      int out[24];
      gru.n++;
      clock_t t0 = clock();
      int n = nano_front_gru_word(&C, text, out);
      t_gru += (double)(clock() - t0) / CLOCKS_PER_SEC;
      if (n == np && !memcmp(out, wi, sizeof(int) * (size_t)n))
        gru.ok++;
      else {
        int o = snprintf(why, sizeof why, "py:");
        for (int k = 0; k < np; k++) o += snprintf(why + o, sizeof why - o, " %d", wi[k]);
        o += snprintf(why + o, sizeof why - o, "  c:");
        for (int k = 0; k < n; k++) o += snprintf(why + o, sizeof why - o, " %d", out[k]);
        report(3, "gru", text, why);
      }
    } else if (fld[0][0] == 'M' && nf >= 6) {
      unhex(fld[1], text, sizeof text);
      char tag[64], cs[4096], segs[1 << 16];
      int have_tag = unhex(fld[2], tag, sizeof tag) >= 0, have_ctrl = unhex(fld[3], cs, sizeof cs) >= 0;
      int have_segs = unhex(fld[4], segs, sizeof segs) >= 0, have_err = unhex(fld[5], werr, sizeof werr) >= 0;
      const char* names[16];
      float vals[16];
      char nb[16][32];
      int nc = 0;
      if (have_ctrl && cs[0]) {
        for (char* p = cs; *p && nc < 16;) {
          char* e = strchr(p, ',');
          if (e) *e = 0;
          char* eq = strrchr(p, '=');
          *eq = 0;
          snprintf(nb[nc], sizeof nb[nc], "%s", p);
          unsigned b[4];
          sscanf(eq + 1, "%2x%2x%2x%2x", &b[0], &b[1], &b[2], &b[3]);
          unsigned char bb[4] = {(unsigned char)b[0], (unsigned char)b[1], (unsigned char)b[2], (unsigned char)b[3]};
          memcpy(&vals[nc], bb, 4);
          names[nc] = nb[nc];
          nc++;
          if (!e) break;
          p = e + 1;
        }
      }
      clock_t t0 = clock();
      int rc = nano_front_markup_begin(&C, text, have_tag ? tag : NULL, names, vals, nc, TAGS, 51, NULL, NULL);
      static nf_segment S;
      int o = 0;
      got[0] = 0;
      while (rc == 0) {
        int k = nano_front_markup_next(&C, &S);
        if (k <= 0) {
          rc = k;
          break;
        }
        if (o) o += snprintf(got + o, sizeof got - o, "|");
        for (const char* p = S.text; *p; p++) o += snprintf(got + o, sizeof got - o, "%02x", (unsigned char)*p);
        o += snprintf(got + o, sizeof got - o, "~%s~", S.tag);
        for (int i = 0; i < S.n_ctrl; i++) {
          char h[9];
          f32hex(S.ctrl_val[i], h);
          o += snprintf(got + o, sizeof got - o, "%s%s=%s", i ? "," : "", S.ctrl_name[i], h);
        }
        o += snprintf(got + o, sizeof got - o, "~");
        for (int i = 0; i < S.n_ws; i++) {
          char a[9], b[9], d[9];
          f32hex(S.ws[i].pitch, a);
          f32hex(S.ws[i].energy, b);
          f32hex(S.ws[i].dur, d);
          o += snprintf(got + o, sizeof got - o, "%s%d/%s/%s/%s", i ? "," : "", S.ws[i].word, a, b, d);
        }
        o += snprintf(got + o, sizeof got - o, "~%d~%d", S.pause_ms, S.skip);
      }
      t_mk += (double)(clock() - t0) / CLOCKS_PER_SEC;
      if (have_err) {
        mkerr.n++;
        if (rc < 0) mkerr.ok++;
        else report(4, "mk-err", text, werr);
      } else {
        mk.n++;
        if (rc == 0 && have_segs && !strcmp(got, segs))
          mk.ok++;
        else {
          snprintf(why, sizeof why, "py=%s\n    c =%s %s", segs, got, rc < 0 ? C.err : "");
          report(4, "markup", text, why);
        }
      }
    }
  }
#define PR(name, s) printf("%-12s %7ld %7ld  %8.3f%%\n", name, (s).ok, (s).n, (s).n ? 100.0 * (s).ok / (s).n : 100.0)
  printf("%-12s %7s %7s  %9s\n", "check", "exact", "cases", "rate");
  PR("normalize", norm);
  PR("ids", ids);
  PR("word_idx", widx);
  PR("errors(ph)", errs);
  PR("err message", errmsg);
  PR("markup", mk);
  PR("errors(mk)", mkerr);
  PR("gru_g2p", gru);
  PR("shared", shared);
  printf("timing (host): to_phonemes %.1f us/text, GRU %.1f us/word, markup %.1f us/text\n",
         n_ph ? 1e6 * t_ph / n_ph : 0, gru.n ? 1e6 * t_gru / gru.n : 0, (mk.n + mkerr.n) ? 1e6 * t_mk / (mk.n + mkerr.n) : 0);
  printf("context %zu bytes\n", sizeof(nano_front_ctx));
  return 0;
}
