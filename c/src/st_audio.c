/* WAV reading (RIFF WAVE, PCM / float / extensible) and polyphase Kaiser-windowed-sinc resampling.
 *
 * Resampler: output sample k sits at input time t = k * sr_in / sr_out (exact rational, no delay) and is
 * sum_n x[n] h(t - n) with h(tau) = 2 fc sinc(2 fc tau) kaiser(tau / W), fc = 0.5 * min(1, sr_out / sr_in) * 0.9565
 * cycles per input sample (the centre of the soxr HQ transition band 0.913 .. 1.0 of the lower Nyquist) and a window
 * long enough for ~130 dB of stopband rejection over that transition width. With L / M the reduced rate ratio there are
 * L distinct phases; their taps are tabulated once (or, for a huge L, interpolated from a finely sampled prototype). */
#include "st_audio.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "st_base.h"

/* ------------------------------------------------------------------ WAV */
static uint32_t rd_u32(const unsigned char* p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t rd_u16(const unsigned char* p) { return (uint16_t)(p[0] | p[1] << 8); }

enum { WAV_PCM = 1, WAV_FLOAT = 3, WAV_EXT = 0xFFFE };

/* one sample of `bytes` bytes at p, as soundfile's float32 read gives it */
static float wav_sample(const unsigned char* p, int fmt, int bytes) {
  if (fmt == WAV_FLOAT) {
    if (bytes == 4) {
      uint32_t u = rd_u32(p);
      float f;
      memcpy(&f, &u, 4);
      return f;
    }
    uint64_t u = (uint64_t)rd_u32(p) | (uint64_t)rd_u32(p + 4) << 32;
    double d;
    memcpy(&d, &u, 8);
    return (float)d;
  }
  switch (bytes) {
    case 1: return (float)((int)p[0] - 128) * (1.f / 128.f);
    case 2: return (float)(int16_t)rd_u16(p) * (1.f / 32768.f);
    case 3: {
      int32_t v = (int32_t)((uint32_t)p[0] << 8 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 24);
      return (float)(v >> 8) * (1.f / 8388608.f);
    }
    default: return (float)(int32_t)rd_u32(p) * (1.f / 2147483648.f);
  }
}

int st_wav_read_mem(const void* data, size_t len, float** y, size_t* n, int* sr) {
  const unsigned char* d = (const unsigned char*)data;
  *y = NULL;
  *n = 0;
  if (len < 12 || memcmp(d, "RIFF", 4) || memcmp(d + 8, "WAVE", 4)) {
    if (len >= 4 && (!memcmp(d, "RF64", 4) || !memcmp(d, "RIFX", 4)))
      st_seterr("wav: RF64 / big-endian RIFX files are not supported");
    else st_seterr("wav: not a RIFF WAVE file");
    return -1;
  }
  int fmt = 0, ch = 0, bits = 0, align = 0, have_fmt = 0;
  uint32_t rate = 0;
  const unsigned char* pcm = NULL;
  size_t pcm_len = 0;
  size_t pos = 12;
  while (pos + 8 <= len) {
    const unsigned char* c = d + pos;
    size_t sz = rd_u32(c + 4), avail = len - pos - 8;
    if (!memcmp(c, "fmt ", 4)) {
      if (sz < 16 || sz > avail) {
        st_seterr("wav: bad fmt chunk");
        return -1;
      }
      fmt = rd_u16(c + 8);
      ch = rd_u16(c + 10);
      rate = rd_u32(c + 12);
      align = rd_u16(c + 20);
      bits = rd_u16(c + 22);
      if (fmt == WAV_EXT) {
        if (sz < 40) {
          st_seterr("wav: bad WAVE_FORMAT_EXTENSIBLE fmt chunk");
          return -1;
        }
        fmt = rd_u16(c + 8 + 24);   /* first two bytes of the sub-format GUID */
      }
      have_fmt = 1;
    } else if (!memcmp(c, "data", 4)) {
      pcm = c + 8;
      pcm_len = sz > avail ? avail : sz;   /* streaming writers leave the size open */
      break;
    }
    if (sz > avail) break;
    pos += 8 + sz + (sz & 1);
  }
  if (!have_fmt || !pcm) {
    st_seterr("wav: missing %s chunk", have_fmt ? "data" : "fmt");
    return -1;
  }
  if (ch < 1 || rate < 1 || rate > 0x7fffffff || align < ch || align % ch) {
    st_seterr("wav: bad header (channels %d, rate %u, block align %d)", ch, rate, align);
    return -1;
  }
  int bytes = align / ch;   /* container size; libsndfile normalises by it, not by the "valid bits" */
  if (!((fmt == WAV_PCM && bytes >= 1 && bytes <= 4) || (fmt == WAV_FLOAT && (bytes == 4 || bytes == 8)))) {
    st_seterr("wav: unsupported sample format (tag %d, %d bits)", fmt, bits);
    return -1;
  }
  size_t frames = pcm_len / (size_t)align;
  float* out = (float*)st_malloc(sizeof(float) * (frames ? frames : 1));
  if (!out) return -1;
  for (size_t i = 0; i < frames; ++i) {
    const unsigned char* p = pcm + i * (size_t)align;
    if (ch == 1) out[i] = wav_sample(p, fmt, bytes);
    else {
      float s = 0.f;   /* numpy float32 mean over channels: sequential sum, then / ch in float32 */
      for (int c = 0; c < ch; ++c) s += wav_sample(p + (size_t)c * bytes, fmt, bytes);
      out[i] = s / (float)ch;
    }
  }
  *y = out;
  *n = frames;
  *sr = (int)rate;
  return 0;
}

int st_wav_read(const char* path, float** y, size_t* n, int* sr) {
  char* data;
  size_t len;
  *y = NULL;
  *n = 0;
  if (st_read_file(path, &data, &len)) return -1;
  int r = st_wav_read_mem(data, len, y, n, sr);
  if (r) st_seterr("%s: %s", path, st_geterr());
  st_free(data);
  return r;
}

/* ------------------------------------------------------------------ resampling */
#define RS_PASS 0.913
#define RS_STOP 1.0
#define RS_ATTEN 130.0
#define RS_OS 4096           /* prototype oversampling for the interpolated (huge L) path */
#define RS_MAX_TABLE (1 << 21)

static double bessel_i0(double x) {
  double s = 1, t = 1, q = x * x / 4;
  for (int k = 1; k < 200; ++k) {
    t *= q / ((double)k * k);
    s += t;
    if (t < s * 1e-17) break;
  }
  return s;
}

typedef struct {
  double fc, W, beta, i0b;
} rs_proto;

static double proto_h(const rs_proto* r, double tau) {
  double u = tau / r->W;
  if (u <= -1 || u >= 1) return 0;
  const double pi = 3.14159265358979323846;
  double x = 2 * r->fc * tau, s = x == 0 ? 1 : sin(pi * x) / (pi * x);
  return 2 * r->fc * s * bessel_i0(r->beta * sqrt(1 - u * u)) / r->i0b;
}

static float dot(const float* a, const float* b, int n) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
    s4 += a[i + 4] * b[i + 4]; s5 += a[i + 5] * b[i + 5]; s6 += a[i + 6] * b[i + 6]; s7 += a[i + 7] * b[i + 7];
  }
  for (; i < n; ++i) s0 += a[i] * b[i];
  return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

static int64_t gcd64(int64_t a, int64_t b) {
  while (b) {
    int64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

size_t st_resample_len(size_t n, int sr_in, int sr_out) {
  double io = (double)sr_in / (double)sr_out;   /* soxr: samples_out = (uint64)(samples_in / io_ratio + .5) */
  return (size_t)((double)n / io + 0.5);
}

int st_resample(const float* x, size_t n, int sr_in, int sr_out, float** y, size_t* ny) {
  *y = NULL;
  *ny = 0;
  if (sr_in <= 0 || sr_out <= 0) {
    st_seterr("resample: bad rates %d -> %d", sr_in, sr_out);
    return -1;
  }
  size_t no = sr_in == sr_out ? n : st_resample_len(n, sr_in, sr_out);
  float* out = (float*)st_malloc(sizeof(float) * (no ? no : 1));
  if (!out) return -1;
  *y = out;
  *ny = no;
  if (sr_in == sr_out) {
    if (n) memcpy(out, x, sizeof(float) * n);
    return 0;
  }
  if (!no) return 0;
  int64_t g = gcd64(sr_in, sr_out), L = sr_out / g, M = sr_in / g;
  double r = sr_out < sr_in ? (double)sr_out / sr_in : 1.0;
  rs_proto P;
  P.fc = 0.5 * r * 0.5 * (RS_PASS + RS_STOP);
  double df = 0.5 * r * (RS_STOP - RS_PASS);
  P.W = 0.5 * (RS_ATTEN - 7.95) / (14.36 * df);
  P.beta = 0.1102 * (RS_ATTEN - 8.7);
  P.i0b = bessel_i0(P.beta);
  int Wi = (int)ceil(P.W), T = 2 * Wi;

  /* zero-padded input: taps of output k cover n0 - Wi + 1 .. n0 + Wi, n0 = floor(k M / L) <= n - 1 + M / L */
  size_t padl = (size_t)Wi, padr = (size_t)Wi + (size_t)(M / L) + 2;
  float* xp = (float*)st_calloc(sizeof(float) * (n + padl + padr));
  if (!xp) goto fail;
  if (n) memcpy(xp + padl, x, sizeof(float) * n);

  if ((size_t)L * (size_t)T <= RS_MAX_TABLE) {
    float* tab = (float*)st_malloc(sizeof(float) * (size_t)L * (size_t)T);
    if (!tab) {
      st_free(xp);
      goto fail;
    }
    for (int64_t p = 0; p < L; ++p)
      for (int i = 0; i < T; ++i) tab[p * T + i] = (float)proto_h(&P, (double)p / (double)L + Wi - 1 - i);
    for (size_t k = 0; k < no; ++k) {
      int64_t km = (int64_t)k * M, n0 = km / L, p = km % L;
      out[k] = dot(xp + padl + n0 - Wi + 1, tab + p * T, T);
    }
    st_free(tab);
  } else {
    /* huge L (nearly coprime rates): linear interpolation in a prototype sampled RS_OS times per input sample */
    size_t nt = (size_t)(P.W * RS_OS) + 3;
    float* proto = (float*)st_malloc(sizeof(float) * nt);
    float* c = (float*)st_malloc(sizeof(float) * (size_t)T);
    if (!proto || !c) {
      st_free(proto);
      st_free(c);
      st_free(xp);
      goto fail;
    }
    for (size_t j = 0; j < nt; ++j) proto[j] = (float)proto_h(&P, (double)j / RS_OS);
    for (size_t k = 0; k < no; ++k) {
      int64_t km = (int64_t)k * M, n0 = km / L;
      double f = (double)(km % L) / (double)L;
      for (int i = 0; i < T; ++i) {
        double a = fabs(f + Wi - 1 - i) * RS_OS;
        size_t j = (size_t)a;
        double w = a - (double)j;
        c[i] = j + 1 < nt ? (float)((1 - w) * proto[j] + w * proto[j + 1]) : 0.f;
      }
      out[k] = dot(xp + padl + n0 - Wi + 1, c, T);
    }
    st_free(proto);
    st_free(c);
  }
  st_free(xp);
  return 0;
fail:
  st_free(out);
  *y = NULL;
  *ny = 0;
  return -1;
}
