/* ESP32-S3 int8 kernels on the PIE SIMD unit (128-bit, 16 int8 lanes). Bit-identical to nano_kernels.c:
 *   - dot products: EE.VMULAS.S8.ACCX (16 MACs per instruction into the 40-bit ACCX accumulator), fused with the
 *     next 128-bit load (EE.VMULAS.S8.ACCX.LD.IP);
 *   - GEMM: weight-stationary, every weight row (one output channel) is read once from flash and reused from the
 *     cache for all rows of the chunk, requantized per output channel;
 *   - depthwise conv: 16 channels at a time in QACC (EE.VMULAS.S8.QACC, 16 x 20-bit lanes), unpacked once per output.
 * Pointers and lengths must be 16-byte aligned / multiples of 16 (nano.bin guarantees it for weights, the runtime
 * for activations); anything else falls back to the portable kernels. */
#include "nano_internal.h"

#if NANO_HAVE_SIMD

const char* nano_kernel_impl(void) { return "esp32s3-simd"; }

NANO_INLINE int32_t dot16(const int8_t* a, const int8_t* b, int n16) {
  int32_t r;
  __asm__ volatile(
      "ee.zero.accx\n\t"
      "ee.vld.128.ip q0, %1, 16\n\t"
      "ee.vld.128.ip q1, %2, 16\n\t"
      "addi %3, %3, -1\n\t"
      "loopgtz %3, 1f\n\t"
      "ee.vmulas.s8.accx.ld.ip q0, %1, 16, q0, q1\n\t"
      "ee.vld.128.ip q1, %2, 16\n\t"
      "1:\n\t"
      "ee.vmulas.s8.accx q0, q1\n\t"
      "rur.accx_0 %0\n\t"
      : "=r"(r), "+r"(a), "+r"(b), "+r"(n16)
      :
      : "memory");
  return r;
}

NANO_INLINE int aligned16(const void* p) { return ((uintptr_t)p & 15) == 0; }

int32_t nano_dot_s8(const int8_t* a, const int8_t* b, int n) {
  if (n >= 16 && !(n & 15) && aligned16(a) && aligned16(b)) return dot16(a, b, n >> 4);
  return nano_dot_s8_ref(a, b, n);
}

NANO_INLINE int8_t clamp8(int32_t v, int lo, int hi) { return (int8_t)(v < lo ? lo : v > hi ? hi : v); }

void nano_gemm_s8(const int8_t* in, int in_stride, int rows, const int8_t* w, int K, const int32_t* b,
                  const int32_t* m, const int32_t* s, int8_t* out, int out_stride, int out_ch, int lo, int hi) {
  if (K < 16 || (K & 15) || (in_stride & 15) || !aligned16(in) || !aligned16(w)) {
    nano_gemm_s8_ref(in, in_stride, rows, w, K, b, m, s, out, out_stride, out_ch, lo, hi);
    return;
  }
  int n16 = K >> 4;
  for (int o = 0; o < out_ch; o++) {
    const int8_t* wr = w + (size_t)o * K;
    int32_t bo = b ? b[o] : 0, mo = m[o], so = s[o];
    const int8_t* x = in;
    int8_t* y = out + o;
    for (int r = 0; r < rows; r++) {
      int32_t acc = dot16(x, wr, n16) + bo;
      *y = clamp8(nano_requant(acc, mo, so), lo, hi);
      x += in_stride;
      y += out_stride;
    }
  }
}

/* QACC -> 16 int32 lanes: QACC_L holds lanes 0..7, QACC_H lanes 8..15, 20 bits each, packed little-endian */
NANO_INLINE void qacc_unpack(const uint32_t* raw /* 5 words L, pad, 5 words H at +8 words */, int32_t* lane) {
  for (int h = 0; h < 2; h++) {
    const uint32_t* q = raw + 8 * h;
    uint64_t lo = (uint64_t)q[0] | ((uint64_t)q[1] << 32);   /* bits 0..63 */
    uint64_t mi = (uint64_t)q[2] | ((uint64_t)q[3] << 32);   /* bits 64..127 */
    uint32_t hi = q[4];                                      /* bits 128..159 */
    uint32_t v[8];
    v[0] = (uint32_t)(lo & 0xFFFFF);
    v[1] = (uint32_t)((lo >> 20) & 0xFFFFF);
    v[2] = (uint32_t)((lo >> 40) & 0xFFFFF);
    v[3] = (uint32_t)(((lo >> 60) | (mi << 4)) & 0xFFFFF);
    v[4] = (uint32_t)((mi >> 16) & 0xFFFFF);
    v[5] = (uint32_t)((mi >> 36) & 0xFFFFF);
    v[6] = (uint32_t)(((mi >> 56) | ((uint64_t)hi << 8)) & 0xFFFFF);
    v[7] = (hi >> 12) & 0xFFFFF;
    for (int i = 0; i < 8; i++) lane[8 * h + i] = (int32_t)(v[i] << 12) >> 12;
  }
}

void nano_dw_s8(const int8_t* in, int rows, int C, const int8_t* w, int k, const int32_t* b, const int32_t* m,
                const int32_t* s, int8_t* out, int lo, int hi) {
  if ((C & 15) || !aligned16(in) || !aligned16(w) || k > 16) {   /* 16 taps x 127^2 fits the 20-bit lanes */
    nano_dw_s8_ref(in, rows, C, w, k, b, m, s, out, lo, hi);
    return;
  }
  uint32_t raw[16] __attribute__((aligned(16)));
  int32_t lane[16];
  for (int t = 0; t < rows; t++)
    for (int c0 = 0; c0 < C; c0 += 16) {
      const int8_t* x = in + (size_t)t * C + c0;
      const int8_t* wj = w + c0;
      uint32_t* p = raw;
      __asm__ volatile(
          "ee.zero.qacc\n\t"
          "loopgtz %3, 1f\n\t"
          "ee.vld.128.xp q0, %0, %4\n\t"
          "ee.vld.128.xp q1, %1, %4\n\t"
          "ee.vmulas.s8.qacc q0, q1\n\t"
          "1:\n\t"
          "ee.st.qacc_l.l.128.ip %2, 16\n\t"
          "ee.st.qacc_l.h.32.ip %2, 16\n\t"
          "ee.st.qacc_h.l.128.ip %2, 16\n\t"
          "ee.st.qacc_h.h.32.ip %2, 0\n\t"
          : "+r"(x), "+r"(wj), "+r"(p)
          : "r"(k), "r"(C)
          : "memory");
      qacc_unpack(raw, lane);
      int8_t* y = out + (size_t)t * C + c0;
      for (int l = 0; l < 16; l++) y[l] = clamp8(nano_requant(lane[l] + b[c0 + l], m[c0 + l], s[c0 + l]), lo, hi);
    }
}

#endif
