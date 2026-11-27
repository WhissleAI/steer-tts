/* st_vocoder internals, shared with the optional GPU path (st_gpu_vocoder.c). */
#ifndef ST_VOCODER_IMPL_H
#define ST_VOCODER_IMPL_H
#include "st_vocoder.h"

#define N_MELS 100

typedef struct {
  float* dw_t;   /* depthwise weights transposed to [7, dim] (owned) */
  const float *dw_w, *dw_b, *ln_w, *ln_b, *gamma, *b1, *b2;
  const st_mat *w1, *w2;
} cnx;

struct st_vocoder {
  int dim, inter, layers, n_fft, hop;
  const st_mat *embed, *out;
  const float *embed_b, *ln_w, *ln_b, *fln_w, *fln_b, *out_b;
  cnx* blk;
  void* gpu;   /* st_gpu_vocoder.c state when attached to a device, else NULL */
};

/* the GPU path of spectra(): re [T, nb] then im [T, nb]; 0 on success */
int st_vocoder_gpu_spectra(st_vocoder* v, const float* mel, int T, float* ri);

#endif
