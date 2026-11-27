/* st_acoustic internals, shared with the optional GPU path (st_gpu_acoustic.c). */
#ifndef ST_ACOUSTIC_IMPL_H
#define ST_ACOUSTIC_IMPL_H
#include "st_acoustic.h"

typedef struct {
  const st_mat* W;
  const float* b;
} lin;

typedef struct {
  int H, N, P;
  const float *norm_w, *kn_w, *qn_w, *k_bias, *q_bias;
  float* decay;   /* -exp(decay_log), [H] (owned) */
  lin in, k, q, v, step, rot, mix, out;
} seqmix;

typedef struct {
  int ada1, ada_seq, ada2;   /* indices into the AdaLN table */
  int seq;                   /* hybrid: a second, sequence-mixing branch */
  lin qkv, o;
  seqmix seq_f, seq_b;
  lin c1, c2;
} block;

typedef struct {
  int n;
  block* l;
  int ada_out;
} stack;

typedef struct {
  lin c1, c2, out;
  int ada1, ada2;
} varpred;

struct st_acoustic {
  int d, heads, ffn, k, g_dim, spk_dim, tag_dim, n_sym, n_tags, spk_enc, state_dim;
  const st_mat *emb, *tag;
  lin spk_proj, cond0, cond2;
  stack enc, dec;
  varpred dur, pitch, energy;
  lin pitch_emb, energy_emb, to_mel;
  int n_ada;
  lin* ada;   /* [n_ada] proj: g -> 2d */
  void* gpu;  /* st_gpu_acoustic.c state when attached to a device, else NULL */
};

/* sinusoidal positions added in place to x [T, d] (Stack.forward) */
void st_acoustic_add_pos(float* x, int T, int d);
/* GPU paths (tab: the AdaLN tables, n_ada rows of [1 + s (d), b (d)]):
 * encode: h [N, d] embeddings + positions in, encoder output out; logdur / pitch / energy [N] out;
 * decode: x [T, d] length-regulated input + positions -> mel [T, 100]. 0 on success. */
int st_acoustic_gpu_encode(st_acoustic* a, float* h, int N, const float* tab, float* logdur, float* pitch,
                           float* energy);
int st_acoustic_gpu_decode(st_acoustic* a, const float* x, int T, const float* tab, float* mel);

#endif
