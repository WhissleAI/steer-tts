/* st_flow internals, shared with the optional GPU path (st_gpu_flow.c). */
#ifndef ST_FLOW_IMPL_H
#define ST_FLOW_IMPL_H
#include "st_flow.h"

enum { MAXL = 64, NMEL = 100, KW = 7, HEADS = 6 };

typedef struct {
  float* wT;   /* [KW, d] depthwise weight, tap-major (owned) */
  const float* dwb;
  int dil;
  const st_mat *mod, *pw1, *pw2;
  const float *modb, *pw1b, *pw2b;
} fblock;

typedef struct {
  const st_mat *mod, *in, *out;
  const float *modb, *inb, *outb;
} fattn;

struct st_flow {
  int d, n_layers, g_dim, prompt, mu_cmn, cin;
  float mean, std;
  st_mat inp_x, inp_c;   /* inp weight split: x columns [0, 300), constant columns [300, 3 * cin) (owned) */
  void *inp_x_buf, *inp_c_buf;
  const float* inp_b;
  const st_mat *t1, *t2, *g1, *g2, *ce1, *ce2, *co, *outw;
  const float *t1b, *t2b, *g1b, *g2b, *ce1b, *ce2b, *cob, *outb;
  float* freq;   /* [d/2] TimeEmb frequencies */
  fblock blk[MAXL];
  fattn att[MAXL];
  int has_att[MAXL];
  void* gpu;   /* st_gpu_flow.c state when attached to a device, else NULL */
};

/* The GPU path of st_flow_sample's ODE loop: x [N,100] (in: x0, out: x after all steps); hc (hcu): constant inp part
 * [N,d]; bmod / amod (bmodu / amodu): per layer [ne][3d] modulation tables (scale, shift, gate) as cond_tables
 * makes them; ts: steps + 1 times; guided: prompt cfg != 0, with the unconditional set. */
int st_flow_gpu_run(st_flow* f, float* x, int N, const float* hc, const float* bmod, const float* amod, const float* hcu,
                    const float* bmodu, const float* amodu, int guided, float cfg, const float* ts, int steps);

#endif
