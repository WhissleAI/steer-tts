/* text.to_phonemes / text.normalize (public API of the device front end). */
#include "nf_int.h"

/* ids == NULL: count only (markup word counting), tracking the largest word index in c->part_maxw */
static int out_push(nano_front_ctx* c, int* ids, int* widx, int* n, int cap, int id, int w) {
  if (w > c->part_maxw) c->part_maxw = w;
  if (!ids) {
    (*n)++;
    return 0;
  }
  if (*n >= cap) {
    nf_fail(c, "ValueError: text too long for the device front end");
    return -1;
  }
  ids[*n] = id;
  widx[*n] = w;
  (*n)++;
  return 0;
}

/* normalised text without event tokens (in c->a) -> ids / widx */
static int to_ph_plain(nano_front_ctx* c, int n, int* ids, int* widx, int cap) {
  int* raw = (int*)c->b;
  int nraw = nf_g2p(c, c->a, n, raw, NF_MAX_CP);
  if (nraw < 0) return -1;
  int no = 0;
  c->part_maxw = -1;
  if (out_push(c, ids, widx, &no, cap, NF_SP, -1)) return -1;
  int wi = -1, s0 = -1;   /* phones of the current word: raw[s0 .. k) */
  for (int k = 0; k <= nraw; ++k) {
    int tok = k < nraw ? raw[k] : -1;
    if (tok == -1 || (tok >= 2 && tok <= 8)) {
      if (s0 >= 0) {
        ++wi;
        for (int i = s0; i < k; ++i)
          if (out_push(c, ids, widx, &no, cap, raw[i], wi)) return -1;
        s0 = -1;
        if (tok == -1 && out_push(c, ids, widx, &no, cap, NF_SP, -1)) return -1;
      }
      if (tok >= 2 && tok <= 8 && out_push(c, ids, widx, &no, cap, tok, -1)) return -1;
      continue;
    }
    if (s0 < 0) s0 = k;
  }
  return no;
}

static int decode_norm(nano_front_ctx* c, const char* utf8, size_t len) {
  nfs X = {c->a, 0, NF_MAX_CP, c};
  if (nf_utf8_decode(utf8, len, &X)) return -1;
  nfs O = {c->a, 0, NF_MAX_CP, c};
  nf_normalize(c, c->a, X.n, &O);
  return c->fail ? -1 : O.n;
}

/* \[\s*([A-Za-z_]+)\s*\] at i of the normalised text: end or -1, name span */
static int event_at(nano_front_ctx* c, const u32* s, int n, int i, int* na, int* nb) {
  if (s[i] != '[') return -1;
  int k = i + 1;
  while (k < n && nf_is_space(c, s[k])) k++;
  int a = k;
  while (k < n && ((s[k] >= 'a' && s[k] <= 'z') || (s[k] >= 'A' && s[k] <= 'Z') || s[k] == '_')) k++;
  if (k == a) return -1;
  int b = k;
  while (k < n && nf_is_space(c, s[k])) k++;
  if (k >= n || s[k] != ']') return -1;
  *na = a;
  *nb = b;
  return k + 1;
}

static int phonemes(nano_front_ctx* c, const char* utf8, size_t len, int* ids, int* widx, int cap) {
  int n = decode_norm(c, utf8, len);
  if (n < 0) return -1;
  int has_ev = 0;
  for (int i = 0; i < n && !has_ev; i++)
    if (c->a[i] == '[') {
      int na, nb;
      has_ev = event_at(c, c->a, n, i, &na, &nb) > 0;
    }
  if (!has_ev) return to_ph_plain(c, n, ids, widx, cap);

  /* re.split(event) of the normalised text: text, name, text, ...; each text part is converted on its own
   * (normalised again, as the reference does); the normalised text is rebuilt for every part, so no extra buffer */
  int no = 0, nw = 0, mx = -1;
  c->part_maxw = -1;
  if (out_push(c, ids, widx, &no, cap, NF_SP, -1)) return -1;
  for (int part = 0;; part++) {
    if (part) {
      n = decode_norm(c, utf8, len);
      if (n < 0) return -1;
    }
    /* find the part-th piece */
    int pos = 0, idx = 0, a = 0, b = 0, kind = -1, na = 0, nb = 0;
    for (;;) {
      int e = -1, i = pos;
      for (; i < n; i++)
        if ((e = event_at(c, c->a, n, i, &na, &nb)) > 0) break;
      if (idx == part) {
        a = pos;
        b = i < n ? i : n;
        kind = 0;
        break;
      }
      idx++;
      if (i >= n) break;
      if (idx == part) {
        a = na;
        b = nb;
        kind = 1;
        break;
      }
      idx++;
      pos = e;
    }
    if (kind < 0) break;
    if (kind == 1) {
      char nm[64];
      int L = b - a;
      if (L < 58) {
        for (int k = 0; k < L; k++) {
          u32 ch = c->a[a + k];
          nm[k] = (char)(ch >= 'A' && ch <= 'Z' ? ch + 32 : ch);
        }
        int id = nf_event_symbol(nm, L);
        if (id >= 0 && (out_push(c, ids, widx, &no, cap, id, -1) || out_push(c, ids, widx, &no, cap, NF_SP, -1)))
          return -1;
      }
      continue;
    }
    /* text part: p.strip(" []") empty -> skip; else [ ] -> spaces and convert */
    int x = a, y = b;
    while (x < y && (c->a[x] == ' ' || c->a[x] == '[' || c->a[x] == ']')) x++;
    while (y > x && (c->a[y - 1] == ' ' || c->a[y - 1] == '[' || c->a[y - 1] == ']')) y--;
    if (x == y) continue;
    for (int k = a; k < b; k++)
      if (c->a[k] == '[' || c->a[k] == ']') c->a[k] = ' ';
    char* sub = (char*)c->b;   /* the part as UTF-8, then decoded and normalised again in a */
    if (nf_utf8_encode(c->a + a, b - a, sub, (int)sizeof c->b) < 0) {
      nf_fail(c, "ValueError: text too long for the device front end");
      return -1;
    }
    nfs X = {c->a, 0, NF_MAX_CP, c};
    if (nf_utf8_decode(sub, strlen(sub), &X)) return -1;
    nfs O = {c->a, 0, NF_MAX_CP, c};
    nf_normalize(c, c->a, X.n, &O);
    if (c->fail) return -1;
    int m = to_ph_plain(c, O.n, ids ? ids + no : NULL, ids ? widx + no : NULL, cap - no);
    if (m < 0) return -1;
    if (ids)   /* drop its leading <sp>, shift its word indices */
      for (int i = 1; i < m; i++) {
        ids[no + i - 1] = ids[no + i];
        widx[no + i - 1] = widx[no + i] >= 0 ? widx[no + i] + nw : -1;
      }
    no += m - 1;
    if (c->part_maxw >= 0) mx = nw + c->part_maxw;
    nw = mx + 1;
  }
  c->part_maxw = mx;
  return no;
}

int nano_front_phonemes(nano_front_ctx* c, const char* utf8, int* ids, int* widx, int cap) {
  c->fail = 0;
  c->err[0] = 0;
  return phonemes(c, utf8, strlen(utf8), ids, widx, cap);
}

/* markup._n_words: words of to_phonemes(text) (0 for blank text); -1 on error */
int nf_count_words(nano_front_ctx* c, const char* utf8, size_t len) {
  size_t a = 0;
  while (a < len && (utf8[a] == ' ' || (utf8[a] >= 9 && utf8[a] <= 13))) a++;
  if (phonemes(c, utf8, len, NULL, NULL, 1 << 30) < 0) return -1;
  return c->part_maxw + 1;
}

int nano_front_normalize(nano_front_ctx* c, const char* utf8, char* out, int cap) {
  c->fail = 0;
  c->err[0] = 0;
  int n = decode_norm(c, utf8, strlen(utf8));
  if (n < 0) return -1;
  int r = nf_utf8_encode(c->a, n, out, cap);
  if (r < 0) nf_fail(c, "ValueError: text too long for the device front end");
  return r;
}

int nano_front_gru_word(nano_front_ctx* c, const char* word, int* raw) { return nf_gru_predict(c, word, raw); }
