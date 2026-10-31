/* safetensors files (mmapped) and the release weight layouts (plain float, packed int8 / int4). */
#define _GNU_SOURCE
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "st_base.h"

/* safetensors data (and the quantized payloads) are little-endian and are read in place */
#if (defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || \
    defined(__BIG_ENDIAN__) || defined(__ARMEB__) || defined(__MIPSEB__)
#error "SteerTTS reads little-endian safetensors in place: big-endian targets are not supported"
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

float st_half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
  if (exp == 0) {
    if (man == 0) bits = sign;
    else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FF;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) bits = sign | 0x7F800000 | (man << 13);
  else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

float st_bf16_to_float(uint16_t h) {
  uint32_t bits = (uint32_t)h << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

/* ------------------------------------------------------------------ safetensors */
struct st_file {
  char* path;
  void* map;
  size_t map_len;
  int mapped;
  st_tensor* t;   /* sorted by name */
  int n;
  st_json* meta;  /* __metadata__ object or NULL */
};

static int dtype_of(const char* s, st_dtype* d, int* size) {
  static const struct { const char* n; st_dtype d; int sz; } tab[] = {
      {"F32", ST_F32, 4}, {"F16", ST_F16, 2}, {"BF16", ST_BF16, 2}, {"I8", ST_I8, 1},  {"U8", ST_U8, 1},
      {"I16", ST_I16, 2}, {"I32", ST_I32, 4}, {"I64", ST_I64, 8},   {"BOOL", ST_BOOL, 1}, {"F64", ST_F64, 8}};
  for (size_t i = 0; i < sizeof tab / sizeof *tab; ++i)
    if (!strcmp(s, tab[i].n)) {
      *d = tab[i].d;
      *size = tab[i].sz;
      return 0;
    }
  return -1;
}

static int cmp_tensor(const void* a, const void* b) {
  return strcmp(((const st_tensor*)a)->name, ((const st_tensor*)b)->name);
}

typedef struct {
  st_tensor t;
  int order;   /* position in the header: a repeated name keeps the last entry, as Python's json does */
} tensor_ord;

static int cmp_tensor_ord(const void* a, const void* b) {
  const tensor_ord *x = (const tensor_ord*)a, *y = (const tensor_ord*)b;
  int c = strcmp(x->t.name, y->t.name);
  return c ? c : (x->order > y->order) - (x->order < y->order);
}

/* a JSON integer (no fraction / exponent) in [0, max] (max <= 2^53, so the double is exact) */
static int json_uint(const st_json* j, double max, uint64_t* out) {
  if (!j || j->type != ST_JNUM || !j->is_int || !(j->num >= 0) || !(j->num <= max)) return -1;
  *out = (uint64_t)j->num;
  return 0;
}

#define ST_MAX_SAFE_INT 9007199254740992.0   /* 2^53 */

st_file* st_file_open(const char* path) {
  st_file* f = (st_file*)st_calloc(sizeof *f);
  st_json* h = NULL;
  tensor_ord* tmp = NULL;
  if (!f) return NULL;
  f->path = st_strdup(path);
  if (!f->path) goto fail;
#ifdef _WIN32
  {
    char* d;
    size_t n;
    if (st_read_file(path, &d, &n)) goto fail;
    f->map = d;
    f->map_len = n;
  }
#else
  {
    int fd = open(path, O_RDONLY);
    struct stat sb;
    if (fd < 0 || fstat(fd, &sb) != 0) {
      if (fd >= 0) close(fd);
      st_seterr("cannot open %s", path);
      goto fail;
    }
    if (sb.st_size < 8) {
      close(fd);
      st_seterr("%s: not a safetensors file", path);
      goto fail;
    }
    f->map_len = (size_t)sb.st_size;
    f->map = mmap(NULL, f->map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (f->map == MAP_FAILED) {
      f->map = NULL;
      st_seterr("cannot map %s", path);
      goto fail;
    }
    f->mapped = 1;
  }
#endif
  if (f->map_len < 8) {
    st_seterr("%s: not a safetensors file", path);
    goto fail;
  }
  uint64_t hl;
  memcpy(&hl, f->map, 8);
  if (hl > f->map_len - 8) {
    st_seterr("%s: bad safetensors header", path);
    goto fail;
  }
  h = st_json_parse((const char*)f->map + 8, (size_t)hl);
  if (!h || h->type != ST_JOBJ) {
    st_seterr("%s: bad safetensors header (%s)", path, h ? "not an object" : st_geterr());
    goto fail;
  }
  const char* base = (const char*)f->map + 8 + hl;
  size_t data_len = f->map_len - 8 - (size_t)hl;
  double max_off = (double)data_len < ST_MAX_SAFE_INT ? (double)data_len : ST_MAX_SAFE_INT;
  tmp = (tensor_ord*)st_calloc(sizeof(tensor_ord) * (size_t)(h->n + 1));
  if (!tmp) goto fail;
  int nt = 0;
  for (int i = 0; i < h->n; ++i) {
    const st_json* v = &h->items[i];
    if (!strcmp(h->keys[i], "__metadata__")) {
      /* keep a private copy: move the node out of the header tree (a repeated key: the last one wins) */
      st_json* m = (st_json*)calloc(1, sizeof(st_json));
      if (!m) {
        st_seterr("out of memory");
        goto fail;
      }
      *m = *v;
      h->items[i].type = ST_JNULL;
      st_json_free(f->meta);
      f->meta = m;
      continue;
    }
    st_tensor* t = &tmp[nt].t;
    int sz;
    const st_json* dt = st_json_get(v, "dtype");
    const st_json* sh = st_json_get(v, "shape");
    const st_json* off = st_json_get(v, "data_offsets");
    if (!dt || !sh || !off || dtype_of(st_json_str(dt, ""), &t->dtype, &sz) || sh->type != ST_JARR || sh->n > 8 ||
        off->type != ST_JARR || off->n != 2) {
      st_seterr("%s: bad entry %s", path, h->keys[i]);
      goto fail;
    }
    t->ndim = sh->n;
    uint64_t numel = 1, a, b;
    for (int d = 0; d < sh->n; ++d) {
      uint64_t dim;
      if (json_uint(&sh->items[d], ST_MAX_SAFE_INT, &dim)) {
        st_seterr("%s: bad shape for %s", path, h->keys[i]);
        goto fail;
      }
      t->shape[d] = (int64_t)dim;
      if (dim && numel > UINT64_MAX / dim) numel = UINT64_MAX;   /* saturate: cannot match b - a below */
      else numel *= dim;
    }
    if (json_uint(&off->items[0], max_off, &a) || json_uint(&off->items[1], max_off, &b) || b < a ||
        b > data_len || numel > UINT64_MAX / (uint64_t)sz || numel * (uint64_t)sz != b - a) {
      st_seterr("%s: bad offsets for %s", path, h->keys[i]);
      goto fail;
    }
    t->data = base + a;
    t->nbytes = (size_t)(b - a);
    t->name = st_strdup(h->keys[i]);
    if (!t->name) goto fail;
    tmp[nt].order = i;
    nt++;
  }
  st_json_free(h);
  h = NULL;
  qsort(tmp, (size_t)nt, sizeof(tensor_ord), cmp_tensor_ord);
  f->t = (st_tensor*)st_calloc(sizeof(st_tensor) * (size_t)(nt + 1));
  if (!f->t) goto fail;
  for (int i = 0; i < nt; ++i) {
    if (i + 1 < nt && !strcmp(tmp[i].t.name, tmp[i + 1].t.name)) {   /* superseded by a later entry */
      st_free(tmp[i].t.name);
      continue;
    }
    f->t[f->n++] = tmp[i].t;
  }
  st_free(tmp);
  return f;
fail:
  if (tmp)
    for (int i = 0; tmp[i].t.name; ++i) st_free(tmp[i].t.name);
  st_free(tmp);
  st_json_free(h);
  st_file_close(f);
  return NULL;
}

void st_file_close(st_file* f) {
  if (!f) return;
  for (int i = 0; i < f->n; ++i) st_free(f->t[i].name);   /* f->t is NULL when f->n == 0 */
  st_free(f->t);
  st_json_free(f->meta);
#ifndef _WIN32
  if (f->mapped && f->map) munmap(f->map, f->map_len);
  else
#endif
    st_free(f->map);
  st_free(f->path);
  st_free(f);
}

const st_tensor* st_file_find(const st_file* f, const char* name) {
  st_tensor key;
  key.name = (char*)name;
  return (const st_tensor*)bsearch(&key, f->t, (size_t)f->n, sizeof(st_tensor), cmp_tensor);
}

int st_file_count(const st_file* f) { return f->n; }
const st_tensor* st_file_tensor(const st_file* f, int i) { return &f->t[i]; }
const char* st_file_metadata(const st_file* f, const char* key) { return st_json_str(st_json_get(f->meta, key), NULL); }

int64_t st_numel(const st_tensor* t) {
  int64_t n = 1;
  for (int d = 0; d < t->ndim; ++d) n *= t->shape[d];
  return n;
}

/* ------------------------------------------------------------------ weights */
typedef struct {
  char* name;
  float* f32;      /* owned converted copy (or NULL) */
  st_mat* mat;     /* owned (or NULL) */
  float* scale;    /* owned f32 scales for quantized mats */
  void* qcopy;     /* owned aligned copy of quantized data (only if needed) */
} wcache;

struct st_weights {
  st_file* f;
  st_json* layout;   /* "layout" object of steertts_quant, or NULL (plain file) */
  st_json* quant;    /* whole steertts_quant metadata */
  wcache* c;
  int nc, capc;
};

st_weights* st_weights_open(const char* path) {
  st_weights* w = (st_weights*)st_calloc(sizeof *w);
  if (!w) return NULL;
  w->f = st_file_open(path);
  if (!w->f) {
    st_free(w);
    return NULL;
  }
  const char* q = st_file_metadata(w->f, "steertts_quant");
  if (q) {
    w->quant = st_json_parse(q, strlen(q));
    if (!w->quant) {
      st_seterr("%s: bad steertts_quant metadata", path);
      st_weights_close(w);
      return NULL;
    }
    w->layout = (st_json*)st_json_get(w->quant, "layout");
  }
  return w;
}

void st_weights_close(st_weights* w) {
  if (!w) return;
  for (int i = 0; i < w->nc; ++i) {
    st_free(w->c[i].name);
    st_free(w->c[i].f32);
    st_free(w->c[i].mat);
    st_free(w->c[i].scale);
    st_free(w->c[i].qcopy);
  }
  free(w->c);
  st_json_free(w->quant);
  st_file_close(w->f);
  st_free(w);
}

const char* st_weights_path(const st_weights* w) { return w->f->path; }

static wcache* cache_get(st_weights* w, const char* name, int create) {
  for (int i = 0; i < w->nc; ++i)
    if (!strcmp(w->c[i].name, name)) return &w->c[i];
  if (!create) return NULL;
  if (w->nc == w->capc) {
    int nc = w->capc ? w->capc * 2 : 64;
    wcache* c = (wcache*)realloc(w->c, sizeof(wcache) * (size_t)nc);
    if (!c) {
      st_seterr("out of memory");
      return NULL;
    }
    w->c = c;
    w->capc = nc;
  }
  wcache* e = &w->c[w->nc];
  memset(e, 0, sizeof *e);
  e->name = st_strdup(name);
  if (!e->name) return NULL;
  w->nc++;
  return e;
}

static const st_tensor* need(const st_weights* w, const char* name) {
  const st_tensor* t = st_file_find(w->f, name);
  if (!t) st_seterr("%s: missing tensor %s", w->f->path, name);
  return t;
}

/* any float tensor -> owned f32 copy (or the mapped data when it is aligned F32) */
static const float* to_f32(const st_tensor* t, float** owned) {
  int64_t n = st_numel(t);
  *owned = NULL;
  if (t->dtype == ST_F32 && ((uintptr_t)t->data & 3) == 0) return (const float*)t->data;
  float* o = (float*)st_malloc(sizeof(float) * (size_t)(n ? n : 1));
  if (!o) return NULL;
  const unsigned char* d = (const unsigned char*)t->data;
  for (int64_t i = 0; i < n; ++i) {
    uint16_t h;
    switch (t->dtype) {
      case ST_F32: memcpy(&o[i], d + 4 * i, 4); break;
      case ST_F16: memcpy(&h, d + 2 * i, 2); o[i] = st_half_to_float(h); break;
      case ST_BF16: memcpy(&h, d + 2 * i, 2); o[i] = st_bf16_to_float(h); break;
      case ST_F64: { double v; memcpy(&v, d + 8 * i, 8); o[i] = (float)v; break; }
      case ST_I64: { int64_t v; memcpy(&v, d + 8 * i, 8); o[i] = (float)v; break; }
      case ST_I32: { int32_t v; memcpy(&v, d + 4 * i, 4); o[i] = (float)v; break; }
      case ST_I8: o[i] = (float)((const int8_t*)d)[i]; break;
      case ST_U8: case ST_BOOL: o[i] = (float)d[i]; break;
      default: st_free(o); st_seterr("unsupported dtype"); return NULL;
    }
  }
  *owned = o;
  return o;
}

static const st_json* layout_of(const st_weights* w, const char* name) {
  return w->layout ? st_json_get(w->layout, name) : NULL;
}

int st_weights_has(const st_weights* w, const char* name) {
  if (layout_of(w, name)) return 1;
  return st_file_find(w->f, name) != NULL;
}

/* *bad = 1 (and st_seterr) when a layout shape exists but is invalid */
static int shape_of(const st_weights* w, const char* name, int64_t shape[8], int* bad) {
  const st_json* L = layout_of(w, name);
  *bad = 0;
  if (L) {
    const st_json* sh = st_json_get(L, "shape");
    if (sh && sh->type == ST_JARR) {
      if (sh->n > 8) {
        st_seterr("%s: layout shape of %s has %d dims", w->f->path, name, sh->n);
        *bad = 1;
        return 0;
      }
      for (int i = 0; i < sh->n; ++i) {
        uint64_t d;
        if (json_uint(&sh->items[i], ST_MAX_SAFE_INT, &d)) {
          st_seterr("%s: bad layout shape for %s", w->f->path, name);
          *bad = 1;
          return 0;
        }
        shape[i] = (int64_t)d;
      }
      return sh->n;
    }
  }
  const st_tensor* t = st_file_find(w->f, name);
  if (!t) return 0;
  for (int i = 0; i < t->ndim; ++i) shape[i] = t->shape[i];
  return t->ndim;
}

int st_weights_shape(const st_weights* w, const char* name, int64_t shape[8]) {
  int bad;
  return shape_of(w, name, shape, &bad);
}

static const float* dequant_full(st_weights* w, const char* name, const st_json* L, int64_t* numel, float** owned);

const float* st_weights_f32(st_weights* w, const char* name, int64_t n_expect) {
  wcache* e = cache_get(w, name, 0);
  int64_t n = 0;
  const float* r = NULL;
  if (e && e->f32) return e->f32;
  const st_json* L = layout_of(w, name);
  float* owned = NULL;
  if (L && strcmp(st_json_str(st_json_get(L, "s"), "raw"), "raw") && strcmp(st_json_str(st_json_get(L, "s"), ""), "fp16")) {
    r = dequant_full(w, name, L, &n, &owned);
  } else {
    const st_tensor* t = need(w, name);
    if (!t) return NULL;
    n = st_numel(t);
    r = to_f32(t, &owned);
  }
  if (!r) return NULL;
  if (n_expect > 0 && n != n_expect) {
    st_free(owned);
    st_seterr("%s: tensor %s has %lld elements, expected %lld", w->f->path, name, (long long)n, (long long)n_expect);
    return NULL;
  }
  if (owned) {
    e = cache_get(w, name, 1);
    if (!e) {
      st_free(owned);
      return NULL;
    }
    e->f32 = owned;
  }
  return r;
}

/* shape [O, ...] of a layout entry or tensor -> O, K (both must fit in an int) */
static int ok_shape(const int64_t* sh, int nd, int* O, int* K) {
  if (nd < 1 || nd > 8 || sh[0] < 0 || sh[0] > INT_MAX) return -1;
  int64_t k = 1;
  for (int i = 1; i < nd; ++i) {
    if (sh[i] < 0) return -1;
    if (sh[i] && k > INT_MAX / sh[i]) return -1;
    k *= sh[i];
  }
  *O = (int)sh[0];
  *K = (int)k;
  return 0;
}

/* an optional integer layout field in [lo, hi]: dflt when absent, -1 when present but not such an integer */
static int64_t layout_int(const st_json* L, const char* key, int64_t dflt, int64_t lo, int64_t hi) {
  const st_json* j = st_json_get(L, key);
  if (!j) return dflt;
  if (j->type != ST_JNUM || !j->is_int || !(j->num >= (double)lo) || !(j->num <= (double)hi)) return -1;
  return (int64_t)j->num;
}

const st_mat* st_weights_mat(st_weights* w, const char* name, int O_expect, int K_expect) {
  wcache* e = cache_get(w, name, 0);
  if (e && e->mat) return e->mat;
  int64_t sh[8];
  int bad, nd = shape_of(w, name, sh, &bad), O = 0, K = 0;
  if (nd == 0) {
    if (!bad) st_seterr("%s: missing tensor %s", w->f->path, name);
    return NULL;
  }
  if (ok_shape(sh, nd, &O, &K)) {
    st_seterr("%s: %s has an unsupported shape for a matrix", w->f->path, name);
    return NULL;
  }
  if ((O_expect > 0 && O != O_expect) || (K_expect > 0 && K != K_expect)) {
    st_seterr("%s: %s is [%d, %d], expected [%d, %d]", w->f->path, name, O, K, O_expect, K_expect);
    return NULL;
  }
  const st_json* L = layout_of(w, name);
  const char* s = L ? st_json_str(st_json_get(L, "s"), "raw") : "raw";
  if (strcmp(s, "raw") && strcmp(s, "fp16") && strcmp(s, "int8") && strcmp(s, "int4")) {
    st_seterr("%s: unknown storage '%s' for %s", w->f->path, s, name);
    return NULL;
  }
  st_mat* m = (st_mat*)st_calloc(sizeof *m);
  if (!m) return NULL;
  m->O = O;
  m->K = K;
  float* scale = NULL;
  void* qcopy = NULL;
  if (!strcmp(s, "int8") || !strcmp(s, "int4")) {
    const st_tensor* q = need(w, st_name("%s::q", name));
    const st_tensor* st = q ? need(w, st_name("%s::s", name)) : NULL;
    if (!q || !st) goto fail;
    float* so;
    const float* sf = to_f32(st, &so);
    if (!sf) goto fail;
    if (!so) {   /* aligned F32 scales in the file: copy anyway, they are tiny */
      so = (float*)st_malloc(st->nbytes);
      if (!so) goto fail;
      memcpy(so, sf, st->nbytes);
    }
    scale = so;
    m->s = scale;
    if (!strcmp(s, "int8")) {
      if (q->dtype != ST_I8 || (int64_t)q->nbytes != (int64_t)O * K || st_numel(st) != O) {
        st_seterr("%s: bad int8 layout for %s", w->f->path, name);
        goto fail;
      }
      m->type = ST_W_I8;
      m->q8 = (const int8_t*)q->data;
    } else {
      /* validate before any division or narrowing: 2 <= g <= 65536, K <= Kp <= INT_MAX, g | Kp, Kp even */
      int64_t g = layout_int(L, "g", 32, 2, 65536), Kp = layout_int(L, "Kp", K, K, INT_MAX);
      if (g < 0 || Kp < 0 || Kp % g || Kp % 2) {
        st_seterr("%s: bad int4 group / padding for %s", w->f->path, name);
        goto fail;
      }
      m->type = ST_W_I4;
      m->g = (int)g;
      m->Kp = (int)Kp;
      if ((q->dtype != ST_U8 && q->dtype != ST_I8) || (int64_t)q->nbytes < (int64_t)O * Kp / 2 ||
          st_numel(st) != (int64_t)O * (Kp / g)) {
        st_seterr("%s: bad int4 layout for %s", w->f->path, name);
        goto fail;
      }
      m->q4 = (const uint8_t*)q->data;
    }
  } else {
    float* owned;
    const st_tensor* t = need(w, name);
    if (!t) goto fail;
    if (st_numel(t) != (int64_t)O * K) {   /* the layout shape must describe the stored tensor */
      st_seterr("%s: %s has %lld elements, layout shape says [%d, %d]", w->f->path, name, (long long)st_numel(t), O, K);
      goto fail;
    }
    const float* f = to_f32(t, &owned);
    if (!f) goto fail;
    m->type = ST_W_F32;
    m->w = f;
    qcopy = owned;
  }
  e = cache_get(w, name, 1);
  if (!e) goto fail;
  e->mat = m;
  e->scale = scale;
  e->qcopy = qcopy;
  return m;
fail:
  st_free(scale);
  st_free(qcopy);
  st_free(m);
  return NULL;
}

static const float* dequant_full(st_weights* w, const char* name, const st_json* L, int64_t* numel, float** owned) {
  const st_mat* m = st_weights_mat(w, name, 0, 0);
  if (!m) return NULL;
  float* o = (float*)st_malloc(sizeof(float) * (size_t)m->O * (size_t)m->K);
  if (!o) return NULL;
  for (int r = 0; r < m->O; ++r)
    for (int k = 0; k < m->K; ++k) {
      float v;
      if (m->type == ST_W_I8) v = m->q8[(size_t)r * m->K + k] * m->s[r];
      else if (m->type == ST_W_I4) {
        size_t idx = (size_t)r * m->Kp + k;
        int nib = (m->q4[idx >> 1] >> ((idx & 1) * 4)) & 15;
        v = (float)(nib - 8) * m->s[(size_t)r * (m->Kp / m->g) + k / m->g];
      } else v = m->w[(size_t)r * m->K + k];
      o[(size_t)r * m->K + k] = v;
    }
  (void)L;
  *numel = (int64_t)m->O * m->K;
  *owned = o;
  return o;
}

const char* st_name(const char* fmt, ...) {
  static _Thread_local char buf[4][512];
  static _Thread_local int k;
  k = (k + 1) & 3;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf[k], sizeof buf[k], fmt, ap);
  va_end(ap);
  return buf[k];
}
