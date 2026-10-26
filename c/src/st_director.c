/* The director (steertts.h: planning a performance). Mirrors the training code (train_director.py) exactly:
 * the same tokenizer (CRC32 of "w:word" and "b:word_word" into 16384 buckets), the same 18 features, an embedding
 * bag (mean) -> tanh(Linear) sentence encoder, a GRU cell over the reading that sees the next sentence and its own
 * previous plan, and a linear head. Weights are 8-bit with a float scale per row (export_director.py). */
#include "steertts.h"
#include "st_director.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKENS 512

typedef struct {
  int rows, cols;
  float* scale;   /* rows */
  int8_t* q;      /* rows * cols */
} mat;

struct steertts_director {
  int buckets, emb, hid, nfeat, ngenre, ntarget;
  mat embm, enc, wih, whh, head;
  float *enc_b, *bih, *bhh, *head_b;
  float cal[25];   /* request = want x cal (rate loud f0_med f0_iqr f0_end): how to ask the paired voice for a change */
  int has_cal;
  void* blob;
};

struct steertts_reading {
  const steertts_director* d;
  float genre[ST_DIRECTOR_GENRES];
  float* h;      /* hid */
  float* prev;   /* ntarget */
  char* ahead;   /* the last look-ahead sentence and its encoding, reused when it becomes the current one */
  float ahead_s[64];
};

/* ---------------------------------------------------------------- loading */

static void set_err(char* err, size_t n, const char* msg) {
  if (err && n) snprintf(err, n, "%s", msg);
}

static int take(const uint8_t** p, const uint8_t* end, size_t n) {
  if ((size_t)(end - *p) < n) return 0;
  *p += n;
  return 1;
}

static int read_mat(mat* m, int rows, int cols, const uint8_t** p, const uint8_t* end) {
  m->rows = rows;
  m->cols = cols;
  m->scale = (float*)*p;
  if (!take(p, end, sizeof(float) * (size_t)rows)) return 0;
  m->q = (int8_t*)*p;
  return take(p, end, (size_t)rows * (size_t)cols);
}

static int read_vec(float** v, int n, const uint8_t** p, const uint8_t* end) {
  *v = (float*)*p;
  return take(p, end, sizeof(float) * (size_t)n);
}

steertts_director* steertts_director_from_memory(const void* data, size_t n, char* err, size_t err_len) {
  if (!data || n < 32 || memcmp(data, "STDR", 4) != 0) {
    set_err(err, err_len, "not a director model (STDR)");
    return NULL;
  }
  steertts_director* d = (steertts_director*)calloc(1, sizeof *d);
  void* blob = malloc(n);
  if (!d || !blob) {
    free(d);
    free(blob);
    set_err(err, err_len, "out of memory");
    return NULL;
  }
  memcpy(blob, data, n);   /* aligned copy: the floats in it can be read in place */
  d->blob = blob;
  const uint8_t* p = (const uint8_t*)blob + 4;
  const uint8_t* end = (const uint8_t*)blob + n;
  uint32_t h[7];
  memcpy(h, p, sizeof h);
  p += sizeof h;
  if (h[0] != 1) {
    set_err(err, err_len, "unsupported director model version");
    steertts_director_free(d);
    return NULL;
  }
  d->buckets = (int)h[1];
  d->emb = (int)h[2];
  d->hid = (int)h[3];
  d->nfeat = (int)h[4];
  d->ngenre = (int)h[5];
  d->ntarget = (int)h[6];
  if (d->nfeat != ST_DIRECTOR_FEATURES || d->ngenre != ST_DIRECTOR_GENRES || d->ntarget != STEERTTS_PLAN_DIMS) {
    set_err(err, err_len, "director model shape does not match this build");
    steertts_director_free(d);
    return NULL;
  }
  int H = d->hid, in_enc = d->emb + d->nfeat + d->ngenre, in_cell = 2 * H + d->ntarget;
  /* floats must be 4-byte aligned: the header is 32 bytes and every block is a multiple of 4 except int8 runs,
   * whose lengths (rows * cols) are multiples of 4 for this architecture */
  if (!read_mat(&d->embm, d->buckets, d->emb, &p, end) || !read_mat(&d->enc, H, in_enc, &p, end) ||
      !read_vec(&d->enc_b, H, &p, end) || !read_mat(&d->wih, 3 * H, in_cell, &p, end) ||
      !read_mat(&d->whh, 3 * H, H, &p, end) || !read_vec(&d->bih, 3 * H, &p, end) || !read_vec(&d->bhh, 3 * H, &p, end) ||
      !read_mat(&d->head, d->ntarget, 2 * H, &p, end) || !read_vec(&d->head_b, d->ntarget, &p, end)) {
    set_err(err, err_len, "director model file is truncated");
    steertts_director_free(d);
    return NULL;
  }
  /* optional: the paired voice's calibration ("CAL1" + 25 floats) */
  if ((size_t)(end - p) >= 4 + sizeof d->cal && memcmp(p, "CAL1", 4) == 0) {
    memcpy(d->cal, p + 4, sizeof d->cal);
    d->has_cal = 1;
  }
  return d;
}

steertts_director* steertts_director_load(const char* path, char* err, size_t err_len) {
  if (!path) {
    set_err(err, err_len, "no director path");
    return NULL;
  }
  char file[4096];
  size_t pl = strlen(path);
  if (pl > 4 && strcmp(path + pl - 4, ".bin") == 0) snprintf(file, sizeof file, "%s", path);
  else snprintf(file, sizeof file, "%s/director.bin", path);   /* a model directory */
  FILE* f = fopen(file, "rb");
  if (!f) {
    char msg[4200];
    snprintf(msg, sizeof msg, "cannot open %s", file);
    set_err(err, err_len, msg);
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void* buf = n > 0 ? malloc((size_t)n) : NULL;
  if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    free(buf);
    set_err(err, err_len, "cannot read the director model");
    return NULL;
  }
  fclose(f);
  steertts_director* d = steertts_director_from_memory(buf, (size_t)n, err, err_len);
  free(buf);
  return d;
}

void steertts_director_free(steertts_director* d) {
  if (!d) return;
  free(d->blob);
  free(d);
}

/* ---------------------------------------------------------------- tokens and features (as train_director.py) */

static const uint32_t crc_table[256] = {
    0x00000000u, 0x77073096u, 0xee0e612cu, 0x990951bau, 0x076dc419u, 0x706af48fu,
    0xe963a535u, 0x9e6495a3u, 0x0edb8832u, 0x79dcb8a4u, 0xe0d5e91eu, 0x97d2d988u,
    0x09b64c2bu, 0x7eb17cbdu, 0xe7b82d07u, 0x90bf1d91u, 0x1db71064u, 0x6ab020f2u,
    0xf3b97148u, 0x84be41deu, 0x1adad47du, 0x6ddde4ebu, 0xf4d4b551u, 0x83d385c7u,
    0x136c9856u, 0x646ba8c0u, 0xfd62f97au, 0x8a65c9ecu, 0x14015c4fu, 0x63066cd9u,
    0xfa0f3d63u, 0x8d080df5u, 0x3b6e20c8u, 0x4c69105eu, 0xd56041e4u, 0xa2677172u,
    0x3c03e4d1u, 0x4b04d447u, 0xd20d85fdu, 0xa50ab56bu, 0x35b5a8fau, 0x42b2986cu,
    0xdbbbc9d6u, 0xacbcf940u, 0x32d86ce3u, 0x45df5c75u, 0xdcd60dcfu, 0xabd13d59u,
    0x26d930acu, 0x51de003au, 0xc8d75180u, 0xbfd06116u, 0x21b4f4b5u, 0x56b3c423u,
    0xcfba9599u, 0xb8bda50fu, 0x2802b89eu, 0x5f058808u, 0xc60cd9b2u, 0xb10be924u,
    0x2f6f7c87u, 0x58684c11u, 0xc1611dabu, 0xb6662d3du, 0x76dc4190u, 0x01db7106u,
    0x98d220bcu, 0xefd5102au, 0x71b18589u, 0x06b6b51fu, 0x9fbfe4a5u, 0xe8b8d433u,
    0x7807c9a2u, 0x0f00f934u, 0x9609a88eu, 0xe10e9818u, 0x7f6a0dbbu, 0x086d3d2du,
    0x91646c97u, 0xe6635c01u, 0x6b6b51f4u, 0x1c6c6162u, 0x856530d8u, 0xf262004eu,
    0x6c0695edu, 0x1b01a57bu, 0x8208f4c1u, 0xf50fc457u, 0x65b0d9c6u, 0x12b7e950u,
    0x8bbeb8eau, 0xfcb9887cu, 0x62dd1ddfu, 0x15da2d49u, 0x8cd37cf3u, 0xfbd44c65u,
    0x4db26158u, 0x3ab551ceu, 0xa3bc0074u, 0xd4bb30e2u, 0x4adfa541u, 0x3dd895d7u,
    0xa4d1c46du, 0xd3d6f4fbu, 0x4369e96au, 0x346ed9fcu, 0xad678846u, 0xda60b8d0u,
    0x44042d73u, 0x33031de5u, 0xaa0a4c5fu, 0xdd0d7cc9u, 0x5005713cu, 0x270241aau,
    0xbe0b1010u, 0xc90c2086u, 0x5768b525u, 0x206f85b3u, 0xb966d409u, 0xce61e49fu,
    0x5edef90eu, 0x29d9c998u, 0xb0d09822u, 0xc7d7a8b4u, 0x59b33d17u, 0x2eb40d81u,
    0xb7bd5c3bu, 0xc0ba6cadu, 0xedb88320u, 0x9abfb3b6u, 0x03b6e20cu, 0x74b1d29au,
    0xead54739u, 0x9dd277afu, 0x04db2615u, 0x73dc1683u, 0xe3630b12u, 0x94643b84u,
    0x0d6d6a3eu, 0x7a6a5aa8u, 0xe40ecf0bu, 0x9309ff9du, 0x0a00ae27u, 0x7d079eb1u,
    0xf00f9344u, 0x8708a3d2u, 0x1e01f268u, 0x6906c2feu, 0xf762575du, 0x806567cbu,
    0x196c3671u, 0x6e6b06e7u, 0xfed41b76u, 0x89d32be0u, 0x10da7a5au, 0x67dd4accu,
    0xf9b9df6fu, 0x8ebeeff9u, 0x17b7be43u, 0x60b08ed5u, 0xd6d6a3e8u, 0xa1d1937eu,
    0x38d8c2c4u, 0x4fdff252u, 0xd1bb67f1u, 0xa6bc5767u, 0x3fb506ddu, 0x48b2364bu,
    0xd80d2bdau, 0xaf0a1b4cu, 0x36034af6u, 0x41047a60u, 0xdf60efc3u, 0xa867df55u,
    0x316e8eefu, 0x4669be79u, 0xcb61b38cu, 0xbc66831au, 0x256fd2a0u, 0x5268e236u,
    0xcc0c7795u, 0xbb0b4703u, 0x220216b9u, 0x5505262fu, 0xc5ba3bbeu, 0xb2bd0b28u,
    0x2bb45a92u, 0x5cb36a04u, 0xc2d7ffa7u, 0xb5d0cf31u, 0x2cd99e8bu, 0x5bdeae1du,
    0x9b64c2b0u, 0xec63f226u, 0x756aa39cu, 0x026d930au, 0x9c0906a9u, 0xeb0e363fu,
    0x72076785u, 0x05005713u, 0x95bf4a82u, 0xe2b87a14u, 0x7bb12baeu, 0x0cb61b38u,
    0x92d28e9bu, 0xe5d5be0du, 0x7cdcefb7u, 0x0bdbdf21u, 0x86d3d2d4u, 0xf1d4e242u,
    0x68ddb3f8u, 0x1fda836eu, 0x81be16cdu, 0xf6b9265bu, 0x6fb077e1u, 0x18b74777u,
    0x88085ae6u, 0xff0f6a70u, 0x66063bcau, 0x11010b5cu, 0x8f659effu, 0xf862ae69u,
    0x616bffd3u, 0x166ccf45u, 0xa00ae278u, 0xd70dd2eeu, 0x4e048354u, 0x3903b3c2u,
    0xa7672661u, 0xd06016f7u, 0x4969474du, 0x3e6e77dbu, 0xaed16a4au, 0xd9d65adcu,
    0x40df0b66u, 0x37d83bf0u, 0xa9bcae53u, 0xdebb9ec5u, 0x47b2cf7fu, 0x30b5ffe9u,
    0xbdbdf21cu, 0xcabac28au, 0x53b39330u, 0x24b4a3a6u, 0xbad03605u, 0xcdd70693u,
    0x54de5729u, 0x23d967bfu, 0xb3667a2eu, 0xc4614ab8u, 0x5d681b02u, 0x2a6f2b94u,
    0xb40bbe37u, 0xc30c8ea1u, 0x5a05df1bu, 0x2d02ef8du};

static uint32_t crc32_of(const char* a, size_t na, const char* b, size_t nb, const char* c, size_t nc) {
  uint32_t x = 0xFFFFFFFFu;
  const char* parts[3] = {a, b, c};
  size_t lens[3] = {na, nb, nc};
  for (int p = 0; p < 3; p++)
    for (size_t i = 0; i < lens[p]; i++) x = crc_table[(x ^ (uint8_t)parts[p][i]) & 0xFF] ^ (x >> 8);
  return x ^ 0xFFFFFFFFu;
}

/* The words of a text: [a-z']+ | [0-9]+ | ! ? … — (lower-cased ASCII; other characters separate). */
static int words_of(const char* t, char words[][64], int max) {
  int n = 0;
  size_t i = 0, L = strlen(t);
  while (i < L && n < max) {
    unsigned char c = (unsigned char)t[i];
    int lower = c < 128 ? tolower(c) : c;
    if ((lower >= 'a' && lower <= 'z') || lower == '\'') {
      int k = 0;
      while (i < L) {
        unsigned char ch = (unsigned char)t[i];
        int lc = ch < 128 ? tolower(ch) : ch;
        if (!((lc >= 'a' && lc <= 'z') || lc == '\'')) break;
        if (k < 63) words[n][k++] = (char)lc;
        i++;
      }
      words[n++][k] = 0;
    } else if (c >= '0' && c <= '9') {
      int k = 0;
      while (i < L && t[i] >= '0' && t[i] <= '9') {
        if (k < 63) words[n][k++] = t[i];
        i++;
      }
      words[n++][k] = 0;
    } else if (c == '!' || c == '?') {
      words[n][0] = (char)c;
      words[n++][1] = 0;
      i++;
    } else if (c == 0xE2 && i + 2 < L && (uint8_t)t[i + 1] == 0x80 &&
               ((uint8_t)t[i + 2] == 0xA6 || (uint8_t)t[i + 2] == 0x94)) {   /* … — */
      memcpy(words[n], t + i, 3);
      words[n++][3] = 0;
      i += 3;
    } else {
      i++;
    }
  }
  return n;
}

int st_director_tokens(const char* text, int* out, int max) {
  char (*words)[64] = (char (*)[64])malloc(sizeof(char[64]) * MAX_TOKENS);
  if (!words) return 0;
  int nw = words_of(text ? text : "", words, MAX_TOKENS);
  int n = 0;
  for (int i = 0; i < nw && n < max; i++) out[n++] = (int)(crc32_of("w:", 2, words[i], strlen(words[i]), "", 0) % 16384u);
  for (int i = 0; i + 1 < nw && n < max; i++) {
    char pair[130];
    int k = snprintf(pair, sizeof pair, "%s_%s", words[i], words[i + 1]);
    out[n++] = (int)(crc32_of("b:", 2, pair, (size_t)k, "", 0) % 16384u);
  }
  free(words);
  return n;
}

static int count_char(const char* t, char c) {
  int n = 0;
  for (; *t; t++) n += *t == c;
  return n;
}

static int has_utf8(const char* t, const char* seq) { return strstr(t, seq) != NULL; }

void st_director_features(const char* text, int dialogue, int has_speaker, int para_len, int para_index,
                             float out[ST_DIRECTOR_FEATURES]) {
  const char* t = text ? text : "";
  size_t L = strlen(t);
  /* the last character after trailing quotes, apostrophes, spaces and brackets */
  size_t e = L;
  for (;;) {
    if (e >= 1 && (t[e - 1] == '"' || t[e - 1] == '\'' || t[e - 1] == ' ' || t[e - 1] == ')')) { e--; continue; }
    if (e >= 3 && (uint8_t)t[e - 3] == 0xE2 && (uint8_t)t[e - 2] == 0x80 &&
        ((uint8_t)t[e - 1] == 0x9D || (uint8_t)t[e - 1] == 0x99)) { e -= 3; continue; }   /* ” ’ */
    break;
  }
  char last = e >= 1 ? t[e - 1] : 0;
  int ellipsis = e >= 3 && (uint8_t)t[e - 3] == 0xE2 && (uint8_t)t[e - 2] == 0x80 && (uint8_t)t[e - 1] == 0xA6;
  size_t r = L;
  while (r > 0 && isspace((unsigned char)t[r - 1])) r--;
  if (r >= 3 && t[r - 1] == '.' && t[r - 2] == '.' && t[r - 3] == '.') ellipsis = 1;
  int words = 0, in_word = 0, upper = 0, lower = 0;
  for (size_t i = 0; i < L; i++) {
    unsigned char c = (unsigned char)t[i];
    int sp = isspace(c);
    if (!sp && !in_word) words++;
    in_word = !sp;
    if (c >= 'A' && c <= 'Z') upper = 1;
    if (c >= 'a' && c <= 'z') lower = 1;
  }
  char head[16] = {0};
  for (int i = 0; i < 15 && t[i]; i++) head[i] = (char)tolower((unsigned char)t[i]);
  int titled = 0;
  const char* heads[] = {"chapter", "book", "part", "canto", "stanza"};
  for (int k = 0; k < 5; k++) {
    size_t n = strlen(heads[k]);
    if (strncmp(head, heads[k], n) == 0 && !isalnum((unsigned char)head[n]) && head[n] != '_') titled = 1;
  }
  float pos = para_len > 1 ? (float)para_index / (float)(para_len - 1) : 0.0f;
  float f[ST_DIRECTOR_FEATURES] = {
      (float)(words < 60 ? words : 60) / 20.0f, (float)count_char(t, ',') / 3.0f,
      (float)(count_char(t, ';') + count_char(t, ':')) / 2.0f, last == '?', last == '!', last == '.', (float)ellipsis,
      (float)(dialogue != 0), (float)(has_speaker != 0), (float)(has_utf8(t, "\xE2\x80\x9C") || strchr(t, '"') != NULL),
      (float)(has_utf8(t, "\xE2\x80\x94") || strstr(t, "--") != NULL), (float)(upper && !lower && words > 1),
      (float)(para_index == 0), (float)(para_index == para_len - 1), (float)(para_len < 12 ? para_len : 12) / 6.0f, pos,
      (float)(words <= 4), (float)titled};
  memcpy(out, f, sizeof f);
}

/* ---------------------------------------------------------------- the network */

static float dot_row(const mat* m, int row, const float* x) {
  const int8_t* q = m->q + (size_t)row * (size_t)m->cols;
  float s = 0;
  for (int j = 0; j < m->cols; j++) s += (float)q[j] * x[j];
  return s * m->scale[row];
}

static float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

/* tanh(enc(mean(emb[tokens]) ++ features ++ genre)) -> s[hid] */
static void encode(const steertts_director* d, const char* text, const float* feat, const float* genre, float* s) {
  int tok[MAX_TOKENS];
  int n = st_director_tokens(text, tok, MAX_TOKENS);
  if (n == 0) tok[n++] = 0;   /* as training: an empty bag is bucket 0 */
  float x[64 + ST_DIRECTOR_FEATURES + ST_DIRECTOR_GENRES];
  int E = d->emb;
  for (int j = 0; j < E; j++) x[j] = 0;
  for (int i = 0; i < n; i++) {
    const int8_t* q = d->embm.q + (size_t)tok[i] * (size_t)E;
    float sc = d->embm.scale[tok[i]];
    for (int j = 0; j < E; j++) x[j] += (float)q[j] * sc;
  }
  for (int j = 0; j < E; j++) x[j] /= (float)n;
  memcpy(x + E, feat, sizeof(float) * ST_DIRECTOR_FEATURES);
  memcpy(x + E + ST_DIRECTOR_FEATURES, genre, sizeof(float) * ST_DIRECTOR_GENRES);
  for (int i = 0; i < d->hid; i++) s[i] = tanhf(dot_row(&d->enc, i, x) + d->enc_b[i]);
}

static steertts_reading* reading_begin(const steertts_director* d, const float* genre) {
  if (!d) return NULL;
  steertts_reading* r = (steertts_reading*)calloc(1, sizeof *r);
  if (!r) return NULL;
  r->d = d;
  if (genre) memcpy(r->genre, genre, sizeof r->genre);
  r->h = (float*)calloc((size_t)d->hid, sizeof(float));
  r->prev = (float*)calloc((size_t)d->ntarget, sizeof(float));
  if (!r->h || !r->prev) {
    steertts_reading_free(r);
    return NULL;
  }
  return r;
}

void steertts_reading_free(steertts_reading* r) {
  if (!r) return;
  free(r->ahead);
  free(r->h);
  free(r->prev);
  free(r);
}

static int reading_next(steertts_reading* r, const char* text, const float* features, const char* next_text,
                        const float* next_features, float plan[STEERTTS_PLAN_DIMS]) {
  if (!r || !text || !features || !plan) return 1;
  const steertts_director* d = r->d;
  int H = d->hid, T = d->ntarget;
  float s[64], nx[64], x[2 * 64 + STEERTTS_PLAN_DIMS], hn[64], hs[2 * 64];
  if (H > 64) return 1;
  if (r->ahead && strcmp(r->ahead, text) == 0) memcpy(s, r->ahead_s, sizeof(float) * (size_t)H);
  else encode(d, text, features, r->genre, s);
  free(r->ahead);
  r->ahead = NULL;
  if (next_text && next_features) {
    encode(d, next_text, next_features, r->genre, nx);
    size_t n = strlen(next_text) + 1;
    r->ahead = (char*)malloc(n);
    if (r->ahead) {
      memcpy(r->ahead, next_text, n);
      memcpy(r->ahead_s, nx, sizeof(float) * (size_t)H);
    }
  } else {
    memset(nx, 0, sizeof(float) * (size_t)H);
  }
  memcpy(x, s, sizeof(float) * (size_t)H);
  memcpy(x + H, nx, sizeof(float) * (size_t)H);
  memcpy(x + 2 * H, r->prev, sizeof(float) * (size_t)T);
  /* GRUCell, PyTorch gate order r, z, n */
  for (int i = 0; i < H; i++) {
    float ir = dot_row(&d->wih, i, x) + d->bih[i], hr = dot_row(&d->whh, i, r->h) + d->bhh[i];
    float iz = dot_row(&d->wih, H + i, x) + d->bih[H + i], hz = dot_row(&d->whh, H + i, r->h) + d->bhh[H + i];
    float in = dot_row(&d->wih, 2 * H + i, x) + d->bih[2 * H + i], hh = dot_row(&d->whh, 2 * H + i, r->h) + d->bhh[2 * H + i];
    float rg = sigm(ir + hr), zg = sigm(iz + hz), ng = tanhf(in + rg * hh);
    hn[i] = (1 - zg) * ng + zg * r->h[i];
  }
  memcpy(r->h, hn, sizeof(float) * (size_t)H);
  memcpy(hs, hn, sizeof(float) * (size_t)H);
  memcpy(hs + H, s, sizeof(float) * (size_t)H);
  for (int k = 0; k < T; k++) plan[k] = dot_row(&d->head, k, hs) + d->head_b[k];
  memcpy(r->prev, plan, sizeof(float) * (size_t)T);
  return 0;
}

/* ---------------------------------------------------------------- the public API */

static const char* const GENRES[ST_DIRECTOR_GENRES] = {"children", "adventure", "fantasy", "mystery", "scifi",
                                                       "gothic", "literary", "humour", "nonfiction", "poetry"};

const char* steertts_director_genre(int i) { return i >= 0 && i < ST_DIRECTOR_GENRES ? GENRES[i] : NULL; }

steertts_reading* steertts_reading_begin(const steertts_director* d, const char* genre) {
  float g[ST_DIRECTOR_GENRES] = {0};
  for (int i = 0; genre && i < ST_DIRECTOR_GENRES; i++)
    if (strcmp(genre, GENRES[i]) == 0) g[i] = 1;
  return reading_begin(d, g);
}

int steertts_reading_next(steertts_reading* r, const steertts_sentence* s, const steertts_sentence* next,
                          float plan[STEERTTS_PLAN_DIMS]) {
  if (!r || !s || !s->text || !plan) return 1;
  float f[ST_DIRECTOR_FEATURES], fn[ST_DIRECTOR_FEATURES];
  st_director_features(s->text, s->dialogue, s->has_speaker, s->para_len, s->para_index, f);
  if (next && next->text) st_director_features(next->text, next->dialogue, next->has_speaker, next->para_len,
                                               next->para_index, fn);
  return reading_next(r, s->text, f, next && next->text ? next->text : NULL, next && next->text ? fn : NULL, plan);
}

int steertts_director_plan(const steertts_director* d, const steertts_sentence* sentences, size_t n, const char* genre,
                           float* plans, char* err, size_t err_len) {
  if (!d || (!sentences && n) || (!plans && n)) {
    set_err(err, err_len, "steertts_director_plan: NULL director, sentences or plans");
    return 1;
  }
  steertts_reading* r = steertts_reading_begin(d, genre);
  if (!r) {
    set_err(err, err_len, "out of memory");
    return 1;
  }
  for (size_t i = 0; i < n; i++) {
    if (!sentences[i].text) {
      steertts_reading_free(r);
      set_err(err, err_len, "steertts_director_plan: a sentence without text");
      return 1;
    }
    steertts_reading_next(r, &sentences[i], i + 1 < n ? &sentences[i + 1] : NULL, plans + i * STEERTTS_PLAN_DIMS);
  }
  steertts_reading_free(r);
  return 0;
}

static float clamp3(float v) { return v < -3 ? -3 : v > 3 ? 3 : v; }

/* how the ending (the final word's pitch, z) maps to the f0_end control (the end slope, z) */
#define END_TO_F0_END 0.8f

void steertts_plan_controls(const steertts_director* d, const float plan[STEERTTS_PLAN_DIMS], float strength,
                            float out[5]) {
  if (!out) return;
  for (int k = 0; k < 5; k++) out[k] = 0;
  if (!plan) return;
  float want[5] = {strength * plan[0], strength * plan[1], strength * plan[2], strength * plan[3],
                   strength * END_TO_F0_END * plan[4]};
  for (int j = 0; j < 5; j++) {
    float v = want[j];
    if (d && d->has_cal) {   /* the voice delivers only part of a requested change (pitch least): ask for more */
      v = 0;
      for (int i = 0; i < 5; i++) v += want[i] * d->cal[i * 5 + j];
    }
    out[j] = clamp3(v);
  }
}

/* "-1.23" without the C locale (a comma-decimal locale must not change the string) */
static size_t put_z(char* b, float v) {
  int c = (int)(v * 100 + (v < 0 ? -0.5f : 0.5f));
  size_t n = 0;
  if (c < 0) { b[n++] = '-'; c = -c; }
  n += (size_t)snprintf(b + n, 16, "%d.%02d", c / 100, c % 100);
  return n;
}

int steertts_plan_ctrl(const steertts_director* d, const float plan[STEERTTS_PLAN_DIMS], float strength, char* ctrl,
                       size_t len) {
  if (!ctrl || !len) return 1;
  ctrl[0] = 0;
  float z[5];
  steertts_plan_controls(d, plan, strength, z);
  static const char* const names[5] = {"rate", "loud", "f0_med", "f0_iqr", "f0_end"};
  char b[160];
  size_t n = 0;
  for (int k = 0; k < 5; k++) {
    n += (size_t)snprintf(b + n, sizeof b - n, "%s%s=", k ? "," : "", names[k]);
    n += put_z(b + n, z[k]);
  }
  if (n + 1 > len) return 1;
  memcpy(ctrl, b, n + 1);
  return 0;
}

int steertts_plan_pause_ms(const float plan[STEERTTS_PLAN_DIMS], float pace) {
  if (!plan) return 0;
  float ms = (expf(plan[5]) - 1) * 1000 / (pace > 0.25f ? pace : 1.0f);
  return ms < 80 ? 80 : ms > 1500 ? 1500 : (int)(ms + 0.5f);
}
