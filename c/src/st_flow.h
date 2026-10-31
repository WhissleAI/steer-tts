/* Flow-matching mel refiner (steertts/flow.py: Flow.sample, OT-CFM, midpoint ODE) and its prompted variant
 * (pflow.safetensors, zero-shot cloning: the reference clip's mel is prepended as context).
 *
 *   refiner (config.json "refiner".cfg):         FlowNet(d, n_layers, g_dim) - ConvNeXt blocks + AdaLN-zero
 *   prompted flow ("prompted_flow".cfg):          + prompt=True, mu_cmn=True, AttnBlocks at attn_at, ctx_enc / ctx_out
 * The optional blocks rope_at / n_ptok / skips are not used by the released weights and are rejected at load. */
#ifndef ST_FLOW_H
#define ST_FLOW_H
#include "st_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_flow st_flow;

/* w: flow.safetensors / pflow.safetensors (fp32 or the packed int8 copy; kept open by the caller for the flow's
 * lifetime). cfg: the "cfg" object of config.json's "refiner" or "prompted_flow" entry. NULL + st_seterr. */
st_flow* st_flow_load(st_weights* w, const st_json* cfg);
void st_flow_free(st_flow* f);
int st_flow_is_prompted(const st_flow* f);

/* mu [T,100]: the acoustic model's log-mel (time-major); g [g_dim = 128]: the request conditioning (aux["g"]).
 * steps: ODE steps (2 network evaluations each, midpoint), temp: noise scale, seed: torch.Generator seed (the noise
 * is torch.randn over the [1, 100, P+T] tensor, bit-exact).
 * prompt: NULL, or [P,100] log-mel context (st_prompt_mel; prompted flow only). It is prepended to mu as in
 * SteerTTS._render and only the T target frames are written to out [T,100]. With the prompted flow and no prompt
 * the net runs prompt-free (the pipeline does this when the reference clip is shorter than 0.5 s).
 * cfg: classifier-free guidance on the prompt (the pipeline uses 0.0; > 0 costs a second evaluation per step). */
int st_flow_sample(st_flow* f, st_pool* p, const float* mu, int T, const float* g, int steps, float temp, uint64_t seed,
                   const float* prompt, int P, float cfg, float* out);

#ifdef __cplusplus
}
#endif
#endif
