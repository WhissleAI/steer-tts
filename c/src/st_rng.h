/* torch-compatible random numbers (CPU generator): mt19937 seeded as torch.Generator().manual_seed(seed), and
 * torch.randn(n, generator=g) for float32, bit for bit with the x86-64 AVX2 build of PyTorch (normal_fill_AVX2 with
 * the Cephes-based avx_mathfun log / sincos, emulated here in scalar code; n < 16 uses the double Box-Muller path).
 * See st_rng.c for the exactness notes. */
#ifndef ST_RNG_H
#define ST_RNG_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t s[624];
  int next;
} st_mt19937;

void st_mt_seed(st_mt19937* mt, uint64_t seed);   /* torch: only the low 32 bits of the seed are used */
uint32_t st_mt_next(st_mt19937* mt);

/* out[n] = torch.randn(n, generator=torch.Generator().manual_seed(seed)) (any shape with n elements, contiguous). */
void st_randn_torch(uint64_t seed, float* out, size_t n);

#ifdef __cplusplus
}
#endif
#endif
