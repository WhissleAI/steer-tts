/* Optional GPU / accelerator backend (ggml): the acoustic model, the flow refiner and the vocoder's network as ggml
 * graphs on any ggml backend (CUDA, Metal, Vulkan, or ggml's own CPU backend for testing). Built only with
 * -DSTEERTTS_GGML=ON; without it every function here reports "built without GPU support" and the engine runs the
 * plain C kernels.
 *
 * Numerics: weights are uploaded as fp32 (int8 / int4 storage dequantized exactly as the CPU kernels do), all
 * compute is fp32. Results match the CPU path to float rounding, not bit for bit (different summation order).
 * The text front end, durations rounding, noise, iSTFT and output levels stay on the CPU. */
#ifndef ST_GPU_H
#define ST_GPU_H
#include "st_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_gpu st_gpu;

/* 1 when the library was built with the ggml backend. */
int st_gpu_built(void);
/* device: "gpu" / "auto" (the first GPU ggml finds), "cuda", "vulkan", "metal" (the first device of that backend),
 * "ggml-cpu" (ggml's CPU backend: runs the GPU graphs on the CPU, for tests), or a ggml device name ("CUDA0", ...).
 * threads: CPU threads for "ggml-cpu". NULL + st_seterr on failure. */
st_gpu* st_gpu_open(const char* device, int threads);
void st_gpu_close(st_gpu* g);
const char* st_gpu_name(const st_gpu* g);   /* "CUDA0 (NVIDIA GeForce RTX 5070)" */
/* Bytes of weights uploaded to the device so far. */
size_t st_gpu_weight_bytes(const st_gpu* g);

struct st_vocoder;
struct st_flow;
struct st_acoustic;
/* Upload a model's weights to the device and route its compute there. 0 on success; on failure the model keeps
 * running on the CPU and st_seterr has the reason. The st_gpu must outlive the models attached to it. */
int st_vocoder_gpu_attach(struct st_vocoder* v, st_gpu* g);
int st_flow_gpu_attach(struct st_flow* f, st_gpu* g);
int st_acoustic_gpu_attach(struct st_acoustic* a, st_gpu* g);
void st_vocoder_gpu_detach(struct st_vocoder* v);
void st_flow_gpu_detach(struct st_flow* f);
void st_acoustic_gpu_detach(struct st_acoustic* a);

#ifdef __cplusplus
}
#endif
#endif
