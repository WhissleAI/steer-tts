/* SteerTTS stand-alone C runtime: shared internals (errors, memory, thread pool, safetensors, JSON, weights,
 * kernels, FFT). C11, no dependencies beyond libc, libm and pthreads.
 *
 * Conventions used by every model file:
 *   - activations are row-major [T, C] (time-major, batch 1) unless a function says otherwise;
 *   - a function that can fail returns int (0 = ok, -1 = error) and records the message with st_seterr();
 *     the public API copies it into the caller's buffer (st_geterr() is thread-local);
 *   - weights are read from safetensors files that stay mmapped for the engine's lifetime; quantized weights
 *     (int8 per output row, int4 per group of g input values) stay quantized and are dequantized panel by panel
 *     inside the matmul. */
#ifndef ST_BASE_H
#define ST_BASE_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* MinGW checks formats against the old msvcrt printf (no %zu); the UCRT these builds link supports C99 formats */
#if defined(__MINGW32__)
#define ST_PRINTF gnu_printf
#else
#define ST_PRINTF printf
#endif

/* ------------------------------------------------------------------ errors */
void st_seterr(const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(ST_PRINTF, 1, 2)))
#endif
    ;
const char* st_geterr(void);
/* strtod for C-syntax decimal numbers that ignores LC_NUMERIC (an embedding app may set a comma-decimal locale).
 * Hexadecimal floats are not accepted. */
double st_strtod(const char* s, char** end);
#define ST_CHECK(x) do { if ((x) != 0) return -1; } while (0)

/* ------------------------------------------------------------------ memory */
void* st_malloc(size_t n);           /* 64-byte aligned, NULL + st_seterr on failure */
void* st_calloc(size_t n);           /* zeroed */
void st_free(void* p);
char* st_strdup(const char* s);
int st_read_file(const char* path, char** data, size_t* len);   /* whole file, NUL-terminated, st_free() it */
double st_time_ms(void);              /* monotonic clock */
size_t st_peak_rss_kb(void);          /* 0 where unsupported */

/* ------------------------------------------------------------------ thread pool */
typedef struct st_pool st_pool;
st_pool* st_pool_create(int n_threads);   /* n <= 1: everything runs on the calling thread */
void st_pool_destroy(st_pool* p);
int st_pool_size(const st_pool* p);       /* 1 for NULL */
/* Linux: the workers' kernel thread ids (not the caller's); the count. 0 elsewhere. */
int st_pool_tids(const st_pool* p, int* out, int max);
/* Runs fn(ctx, task, n_tasks) for task in [0, n_tasks) on the pool (the caller works too) and waits. A call made
 * from inside one of the pool's tasks, or while another thread is using the pool, runs serially on the caller. */
typedef void (*st_task_fn)(void* ctx, int task, int n_tasks);
void st_parallel_for(st_pool* p, int n_tasks, st_task_fn fn, void* ctx);

/* ------------------------------------------------------------------ JSON (read-only DOM) */
typedef enum { ST_JNULL, ST_JBOOL, ST_JNUM, ST_JSTR, ST_JARR, ST_JOBJ } st_jtype;
typedef struct st_json {
  st_jtype type;
  double num;              /* JNUM, JBOOL (0/1) */
  char* str;               /* JSTR: UTF-8, NUL-terminated */
  int n;                   /* JARR / JOBJ: number of items */
  struct st_json* items;   /* JARR / JOBJ values */
  char** keys;             /* JOBJ keys (parallel to items) */
  int is_int;              /* JNUM written without fraction or exponent ("3", not "3.0") */
} st_json;
st_json* st_json_parse(const char* text, size_t len);   /* NULL + st_seterr on a syntax error */
void st_json_free(st_json* j);
const st_json* st_json_get(const st_json* obj, const char* key);   /* NULL if absent / not an object */
const st_json* st_json_at(const st_json* arr, int i);
double st_json_num(const st_json* j, double dflt);
const char* st_json_str(const st_json* j, const char* dflt);

/* ------------------------------------------------------------------ safetensors */
typedef enum { ST_F32, ST_F16, ST_BF16, ST_I8, ST_U8, ST_I16, ST_I32, ST_I64, ST_BOOL, ST_F64 } st_dtype;
typedef struct {
  char* name;
  st_dtype dtype;
  int ndim;
  int64_t shape[8];
  const void* data;        /* into the mapping; may be unaligned */
  size_t nbytes;
} st_tensor;
typedef struct st_file st_file;
st_file* st_file_open(const char* path);       /* mmap (read() fallback); NULL + st_seterr */
void st_file_close(st_file* f);
const st_tensor* st_file_find(const st_file* f, const char* name);   /* NULL if absent */
int st_file_count(const st_file* f);
const st_tensor* st_file_tensor(const st_file* f, int i);
const char* st_file_metadata(const st_file* f, const char* key);   /* __metadata__ value, NULL if absent */
int64_t st_numel(const st_tensor* t);
float st_half_to_float(uint16_t h);
float st_bf16_to_float(uint16_t h);

/* ------------------------------------------------------------------ weights
 * st_weights wraps one safetensors file in either layout used by the releases:
 *   plain   : name -> F32 / F16 / BF16 tensor (v1 fp32 files, converted externals);
 *   packed  : metadata "steertts_quant" = {"layout": {name: {"s": raw|fp16|int8|int4, "shape", "g", "Kp"}}}, data in
 *             name (raw/fp16) or name::q + name::s (int8: int8 [O,K] + fp16 scale per row; int4: two values per byte,
 *             low nibble first, value = (nibble - 8) * scale, groups of g along the flattened input dim padded to
 *             Kp, fp16 scale [O, Kp/g]). Used by the v1 int8/ files and the edge package. */
typedef enum { ST_W_F32, ST_W_I8, ST_W_I4 } st_wtype;
typedef struct {
  st_wtype type;
  int O, K;                /* output rows, flattened input size (a conv weight [O,C,k] has K = C*k) */
  const float* w;          /* F32: [O, K] row-major */
  const int8_t* q8;        /* I8: [O, K] */
  const uint8_t* q4;       /* I4: [O, Kp/2] bytes */
  const float* s;          /* I8: [O]; I4: [O, Kp/g] */
  int g, Kp;
} st_mat;

typedef struct st_weights st_weights;
st_weights* st_weights_open(const char* path);   /* NULL + st_seterr */
void st_weights_close(st_weights* w);
int st_weights_has(const st_weights* w, const char* name);
/* A float tensor (any storage, dequantized if needed) as an owned or mapped f32 array. n_expect > 0 checks the
 * element count. The returned pointer lives as long as the st_weights. NULL + st_seterr if missing. */
const float* st_weights_f32(st_weights* w, const char* name, int64_t n_expect);
/* A >=2-D weight as a matrix [shape[0], prod(shape[1:])], quantized storage kept. NULL + st_seterr if missing or
 * the shape disagrees (O_expect / K_expect <= 0: not checked). */
const st_mat* st_weights_mat(st_weights* w, const char* name, int O_expect, int K_expect);
/* Shape of a tensor in the logical (unpacked) layout; returns ndim, 0 if absent or if the layout shape is invalid
 * (more than 8 dims, a negative or non-integer dim). */
int st_weights_shape(const st_weights* w, const char* name, int64_t shape[8]);
const char* st_weights_path(const st_weights* w);
/* printf-style name helper: a static thread-local buffer, valid until the next call. */
const char* st_name(const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(ST_PRINTF, 1, 2)))
#endif
    ;

/* ------------------------------------------------------------------ kernels */
/* 1 when the CPU and OS support AVX2 + FMA (x86-64) and STEERTTS_NO_SIMD is not set; 0 elsewhere. */
int st_cpu_avx2(void);
/* C[M,N] (+)= A[M,K] * B[N,K]^T (+ bias[N]).  acc: add to C instead of overwriting. */
void st_gemm_nt(st_pool* p, int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                const float* bias, int acc);
/* C[M,N] (+)= A[M,K] * B[K,N] (+ bias[N]). */
void st_gemm_nn(st_pool* p, int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                const float* bias, int acc);
/* y[T,O] (+)= x[T,K] W^T + bias: a linear layer / 1x1 conv with any weight storage. */
void st_linear(st_pool* p, const float* x, int T, int ldx, const st_mat* W, const float* bias, float* y, int ldy,
               int acc);
/* Conv1d on time-major data (groups = 1): x [T, C] (row stride ldx), W [O, C, k] as st_mat (K = C*k), zero padding
 * pad_l / pad_r, stride, dilation; y [T_out, O] with T_out = (T + pad_l + pad_r - dil*(k-1) - 1) / stride + 1, which
 * is returned (-1 on allocation failure). For a grouped conv call it per group with st_mat_rows() and column
 * offsets into x / y. 0 when the padded input is shorter than the kernel span; -1 also for a bad geometry
 * (k, stride, dilation < 1 or negative padding). */
int st_conv1d(st_pool* p, const float* x, int T, int C, int ldx, const st_mat* W, int k, int stride, int dil,
              int pad_l, int pad_r, const float* bias, float* y, int ldy);
/* A view of rows [r0, r0+n) of a matrix (no copy). */
void st_mat_rows(const st_mat* m, int r0, int n, st_mat* out);
/* Row r of a matrix, dequantized: out[K] (embedding lookups). */
void st_mat_row(const st_mat* m, int r, float* out);
/* Depthwise conv1d (groups = C): w [C, k] f32, x [T, C] -> y [T, C], zero padding pad_l (right: same total). */
void st_dwconv1d(const float* x, int T, int C, const float* w, int k, int dil, int pad_l, const float* bias, float* y);
/* In-place row ops on [T, C]. gamma / beta may be NULL. */
void st_layernorm(float* x, int T, int C, int ld, const float* gamma, const float* beta, float eps);
void st_rmsnorm(float* x, int T, int C, int ld, const float* gamma, float eps);
void st_gelu(float* x, size_t n);        /* exact erf GELU (torch default) */
void st_gelu_tanh(float* x, size_t n);   /* tanh approximation */
void st_silu(float* x, size_t n);
void st_relu(float* x, size_t n);
void st_softmax_rows(float* x, int rows, int cols, int ld);
/* Multi-head scaled-dot-product attention, batch 1. q [Tq, H*dh] (stride ldq), k / v [Tk, H*dh] (ldk, ldv),
 * out [Tq, H*dh] (ldo). key_valid: NULL (all valid) or Tk bytes (1 = attend). bias: NULL or a [H?] additive
 * per-(head, q, k) bias callback-free buffer [H, Tq, Tk] (WavLM relative positions). scale <= 0: 1/sqrt(dh). */
/* Returns 0, or -1 (st_seterr, out filled with NaN) if the score scratch cannot be allocated. */
int st_attention(st_pool* p, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv, float* out,
                 int ldo, int Tq, int Tk, int H, int dh, const uint8_t* key_valid, const float* bias, float scale);

/* ------------------------------------------------------------------ FFT / DSP */
typedef struct st_fft st_fft;
st_fft* st_fft_create(int n);              /* n a power of two */
void st_fft_destroy(st_fft* f);
/* in-place complex FFT on interleaved double (re, im) pairs; inverse = 1 gives the unnormalised inverse */
void st_fft_run(const st_fft* f, double* a, int inverse);
/* real -> half spectrum (n/2+1 complex, interleaved) */
void st_rfft(const st_fft* f, const float* x, double* spec, double* work);
/* half spectrum -> real (numpy irfft: imag of DC / Nyquist ignored), normalised by 1/n */
void st_irfft(const st_fft* f, const double* spec, float* y, double* work);
void st_hann_periodic(float* w, int n);

#ifdef __cplusplus
}
#endif
#endif
