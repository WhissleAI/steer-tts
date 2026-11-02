/* Complex FFT (radix-2, decimation in frequency, then bit reversal) with constant flash twiddles. On the ESP32-S3
 * with esp-dsp available the butterflies run in esp-dsp's optimized dsps_fft2r_fc32 kernel (same table layout). */
#include "nano_internal.h"
#include "nano_tables.h"

#if defined(NANO_HAVE_ESP_DSP) && NANO_HAVE_ESP_DSP
#include "dsps_fft2r.h"
#endif

int nano_fft_init(void) { return 0; }

#if !(defined(NANO_HAVE_ESP_DSP) && NANO_HAVE_ESP_DSP)
static void butterflies_c(float* data, int N, const float* w) {
  int ie = 1;
  for (int N2 = N / 2; N2 > 0; N2 >>= 1) {
    int ia = 0;
    for (int j = 0; j < ie; j++) {
      float c = w[2 * j], s = w[2 * j + 1];
      for (int i = 0; i < N2; i++) {
        int m = ia + N2;
        float re = c * data[2 * m] + s * data[2 * m + 1];
        float im = c * data[2 * m + 1] - s * data[2 * m];
        data[2 * m] = data[2 * ia] - re;
        data[2 * m + 1] = data[2 * ia + 1] - im;
        data[2 * ia] = data[2 * ia] + re;
        data[2 * ia + 1] = data[2 * ia + 1] + im;
        ia++;
      }
      ia += N2;
    }
    ie <<= 1;
  }
}
#endif

static void bitrev(float* d, int n) {
  int j = 0;
  for (int i = 0; i < n - 1; i++) {
    if (i < j) {
      float r = d[2 * i], m = d[2 * i + 1];
      d[2 * i] = d[2 * j];
      d[2 * i + 1] = d[2 * j + 1];
      d[2 * j] = r;
      d[2 * j + 1] = m;
    }
    int k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }
}

void nano_fft_c(float* data, int n) {
#if defined(NANO_HAVE_ESP_DSP) && NANO_HAVE_ESP_DSP
  dsps_fft2r_fc32_aes3_(data, n, (float*)nano_fftw1024);
#else
  butterflies_c(data, n, nano_fftw1024);
#endif
  bitrev(data, n);
}
