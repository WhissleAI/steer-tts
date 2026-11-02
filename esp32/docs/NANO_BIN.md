# nano.bin format (version 1)

`nano.bin` is the single weight file of SteerTTS Nano. It is read **in place**: on the ESP32-S3 it sits in a flash
data partition that is memory-mapped, so weights never take RAM. The C definitions are in
[`components/steertts_nano/include/nano_bin.h`](../components/steertts_nano/include/nano_bin.h), a Python writer and
reader in [`tools/nanobin.py`](../tools/nanobin.py) (`expected_tensors(cfg)` is the authoritative tensor list; the
C loader checks every name, dtype and size against the same rules).

All numbers are little-endian.

## Layout

| Offset | Size | Content |
|---|---|---|
| 0 | 256 | header |
| `dir_offset` (256) | 64 x `n_tensors` | tensor directory |
| `data_offset` (16-aligned) | ... | tensor data, each tensor 16-byte aligned, zero padding between tensors |

### Header

| Offset | Type | Field |
|---|---|---|
| 0 | char[4] | magic `STNB` |
| 4 | u32 | version = 1 |
| 8 | u32 | header bytes = 256 |
| 12 | u32 | `n_tensors` |
| 16 | u32 | `dir_offset` |
| 20 | u32 | `data_offset` |
| 24 | u32 | `file_bytes` (total size) |
| 28 | u32 | flags = 0 |
| 32 | i32[48] | `cfg[]` (model configuration, below; unused entries 0) |
| 224 | f32[8] | `fcfg[]` (float configuration, below; unused entries 0) |

`cfg[]` (index: name, v0 value):

| # | Name | v0 | # | Name | v0 |
|---|---|---|---|---|---|
| 0 | sample_rate | 16000 | 17 | pe_k | 3 |
| 1 | n_fft | 1024 | 18 | dec_layers | 4 |
| 2 | hop | 256 | 19 | dec_dim | 160 |
| 3 | win (= n_fft) | 1024 | 20 | dec_ff | 320 |
| 4 | n_mels | 80 | 21 | dec_k | 5 |
| 5 | n_symbols | 118 | 22 | voc_embed_k | 7 |
| 6 | d_cond | 128 | 23 | voc_layers | 4 |
| 7 | n_voices | 16 | 24 | voc_dim | 128 |
| 8 | n_ctrl (12 or 24) | 24 | 25 | voc_ff | 384 |
| 9 | n_tags | 51 | 26 | voc_k | 7 |
| 10 | enc_layers | 4 | 27 | chunk (frames per weight pass; 32 = 0.5 s, 64 = 1 s) | 32 |
| 11 | enc_dim | 192 | 28 | max_phonemes | 200 |
| 12 | enc_ff | 384 | 29 | enc_act (0 ReLU, 1 GELU) | 0 |
| 13 | enc_k | 5 | 30 | dec_act | 0 |
| 14 | pred_layers | 2 | 31 | voc_act | 0 |
| 15 | pred_dim | 192 | 32 | max_dur (frames per phoneme) | 255 |
| 16 | pred_k | 3 | 33 | cond_act (0 none, 1 SiLU) | 1 |
| | | | 34 | first_chunk (0 = chunk) | 16 |

`fcfg[]`: 0 `dur_offset` (1.0), 1 `mag_max` (100), 2 `ws_pitch_div` (3), 3 `ws_energy_div` (10), 4 `ln_eps` (1e-5).

Constraints checked by the loader: every width (`*_dim`, `*_ff`, `n_mels`) is a multiple of 16 and <= 512; every
kernel size is odd and <= 9; `n_fft` is 512, 1024 or 2048 and a multiple of `hop`; at most 12 layers per stack and 4
predictor layers. The streaming halo and the arena size follow from these numbers; nothing is hard-coded.

### Directory entry (64 bytes)

| Offset | Type | Field |
|---|---|---|
| 0 | char[40] | name, NUL-padded |
| 40 | u16 | dtype: 1 int8, 2 int32, 3 float32 |
| 42 | u16 | ndim (1..3) |
| 44 | u32[3] | shape (unused dims 1) |
| 56 | u32 | offset from file start (multiple of 16) |
| 60 | u32 | size in bytes |

## Quantization

Activations are symmetric int8 (zero point 0) with one float scale per tensor: real = q x s. Every int8 tensor that
passes from one layer to the next is in one agreed scale (the loader checks the chain, below).

A **quantized layer** `L` (linear, full conv or depthwise conv) has:

| Tensor | dtype | Shape | Meaning |
|---|---|---|---|
| `L.w` | int8 | linear `[out, in]`; conv `[out, k, in]`; depthwise `[k, ch]` | weights, per output channel scale `ws` |
| `L.b` | int32 | `[out]` | bias in accumulator units: round(b_real / (s_in x ws[o])) |
| `L.m`, `L.s` | int32 | `[out]` | requantization multiplier and shift of s_in x ws[o] / s_out |
| `L.ws` | float32 | `[out]` | weight scales (reference only; the kernels use `m`, `s`) |
| `L.sc` | float32 | `[2]` (`[3]` for `voc.head`) | `sc[0]` = s_in, `sc[1]` = s_out (`voc.head`: `sc[1]` log-magnitude, `sc[2]` phase) |

`m`, `s` are TFLite's `QuantizeMultiplier(real)`: real ~= m x 2^(s - 31), m in [2^30, 2^31) (`nanobin.quantize_multiplier`,
`nanobin.requant_params`). The integer result of a layer is

```
acc  = sum_i x[i] * w[o][i] + b[o]                          (int32; conv: i runs over k taps x in channels)
out  = clamp(RDPOT(SRDHM(acc << max(s,0), m), max(-s,0)), lo, 127)
```

with gemmlowp's `SaturatingRoundingDoublingHighMul` and `RoundingDivideByPOT` (the TFLite / esp-nn bit-exact path),
`lo` = 0 for a ReLU-fused layer, else -128. Depthwise: `acc = sum_j x[t+j][c] * w[j][c] + b[c]`.

Float steps use float32 arithmetic only, in this order, with **round half to even** (`q8(v)`: clamp to [-128, 127],
then `(v + 12582912) - 12582912`):

* **LayerNorm** of an int8 row q (d values, input scale s): `S = sum q`, `SS = sum q^2` (exact int32);
  `mean = S * (1/d)`, `var = max(SS * (1/d) - mean^2, 0)`, `r = 1 / sqrt(var + eps / s^2)`;
  `y_i = q8(((q_i - mean) * r * g_i + b_i) * (1 / s_out))`. AdaLN uses `g = 1 + scale`, `b = shift`.
* **Residual**: `y = q8(x * (s_x / s_y) + h * (s_h / s_y))`.
* **GELU** (only if an `*_act` is 1): an int8 -> int8 table, tanh form, built per layer from (pw1 s_out, pw2 s_in).

## Graph and tensor names

Conditioning (float32, once per utterance):

| Tensor | Shape |
|---|---|
| `cond.voice` | `[n_voices, d_cond]` |
| `cond.tag` | `[n_tags, d_cond]` (row 0 = no style) |
| `cond.ctrl.w`, `cond.ctrl.b` | `[d_cond, n_ctrl]`, `[d_cond]` |

`c = voice[v] + tag[t] + ctrl.w @ ctrl + ctrl.b`, then SiLU if `cond_act` = 1. The API takes the 12 z-scored
controls (rate, loud, f0_med, f0_iqr, f0_end, pause, tilt, voiced, flat, centroid, cpps, h1h2) and a set / unset
flag for each. With `n_ctrl` = 24 the model input is `[value * set (12), set (12)]` (so "not set" differs from "set
to 0", as in the edge model); with `n_ctrl` = 12 it is `value * set`. The sum runs over the inputs in order, starting
from `ctrl.b`, in float32.

**Block** `P` (`enc.i`, `dec.i`: adaptive; `voc.i`: plain LayerNorm), width d, inner ff, kernel k:

```
h = dw(x)                         P.dw     depthwise k, in scale = block input scale, out P.dw.sc[1]
h = norm(h)                       AdaLN:   P.ada.w int8 [2d, d_cond], P.ada.ws f32 [2d], P.ada.b f32 [2d]
                                           ss = (P.ada.w @ c) * ada.ws + ada.b; g = 1 + ss[:d], b = ss[d:]
                                  LN:      P.ln.g, P.ln.b f32 [d]
                                  output scale = P.pw1.sc[0]
h = act(pw1(h))                   P.pw1    [ff, d]   (ReLU fused: P.pw1.sc[1] must equal P.pw2.sc[0])
h = pw2(h)                        P.pw2    [d, ff]   (fold any layer scale into it)
y = x + h                         P.res    f32 [1]   output scale (= next block's P.dw.sc[0])
```

Convolutions are "same" with zero padding (pad k/2 on both sides).

Pipeline:

1. `enc.emb` int8 `[n_symbols, enc_dim]`, quantized at `enc.0.dw.sc[0]`; `enc.0` .. `enc.{L-1}` -> H (scale
   `enc.{L-1}.res`).
2. Predictors `pred.dur`, `pred.f0`, `pred.en`, each on H: `pred.X.c{j}` full conv `[pred_dim, pred_k, in]` + ReLU,
   then LayerNorm `pred.X.c{j}.ln.g/.b`; between layers the LN output is quantized at `pred.X.c{j+1}.sc[0]`; after
   the last layer the LN output stays float and `y = dot(ln, pred.X.out.w) + pred.X.out.b` (`out.w` f32
   `[pred_dim]`, `out.b` f32 `[1]`).
3. Durations, per phoneme, in float32 operations in this order (pace > 1 = faster, as in the edge API):
   `f = (exp(log_dur) - dur_offset) / pace`; `f = f * word_dur` (the word's shift factor, if any);
   `f = min(max(f, 0), 60000)`; `frames = min(round_half_even(f), max_dur)` (round via `(f + 1.5 * 2^23) - 1.5 * 2^23`).
   `exp` is the runtime's deterministic float32 exp (`nano_expf`: Cody-Waite reduction by ln 2, degree-7 Horner
   polynomial, exponent bits), so every platform gets the same frames. Word shifts:
   `f0 += pitch / ws_pitch_div`, `energy += energy / ws_energy_div` for the phonemes of the word.
4. `pe.f0.w`, `pe.en.w` f32 `[pe_k, enc_dim]` + `.b` `[enc_dim]`: conv of the f0 / energy sequence (zero padded),
   `v = H * s_H + pe_f0 + pe_en`, quantized at `dec.in.sc[0]`; `dec.in` `[dec_dim, enc_dim]` -> decoder input
   (scale `dec.0.dw.sc[0]`).
5. Length regulator (repeat phoneme rows by their frame counts), decoder blocks `dec.i`, `dec.mel`
   `[n_mels, dec_dim]` -> int8 log-mel at `voc.embed.sc[0]`.
6. Vocoder: `voc.embed` full conv `[voc_dim, voc_embed_k, n_mels]`, `voc.ln0.g/.b`, blocks `voc.i`,
   `voc.lnf.g/.b` (quantized at `voc.head.sc[0]`), `voc.head` `[n_fft + 2, voc_dim]`: channels 0..n_fft/2 log
   magnitude (scale `sc[1]`), the rest phase (scale `sc[2]`).
7. iSTFT: `S = min(exp(logmag), mag_max) * (cos(phase) + i sin(phase))`, imaginary parts of the DC and Nyquist bins
   ignored, `irfft(S, n_fft)` x periodic Hann, overlap-add with `hop`, divided by the summed squared window,
   "same" padding ((n_fft - hop) / 2 samples trimmed at the start), `n_frames x hop` samples, int16 = round(y x 32767).

The scale-chain checks done at load: block output -> next block input, ReLU `pw1.sc[1] == pw2.sc[0]`, encoder output
-> every predictor input, `dec.in` output -> decoder input, decoder output -> `dec.mel` input, `dec.mel` output ->
`voc.embed` input.

## Streaming and memory

The runtime processes the decoder and vocoder in chunks of `chunk` frames (`first_chunk` for the first one, to cut
latency). Each chunk is **one pass over the decoder and vocoder weights** (weight-stationary int8 GEMM over all frames
of the chunk); the vocoder head runs in groups of 16 frames. Every conv layer keeps its last k-1 input rows (the halo)
between chunks, so the chunked result equals a whole-utterance run sample for sample (`nano_host --check-chunks`).
The encoder runs in chunks of `chunk` phonemes the same way; the predictors run on the whole sentence.

All working memory is one arena; `nano_arena_bytes(model, chunk)` returns its size. v0 sizes: about 122 KB at chunk 32 or
less (there the sentence-level predictor buffers set the size), 130 KB at chunk 48, 146 KB at chunk 64. A larger
chunk means fewer weight passes: at 32 frames the decoder and vocoder weights are read about twice per second of
audio, at 64 once.

## Exporter checklist

* `write_nano_bin(path, cfg, fcfg, tensors)` with exactly `expected_tensors(cfg)`; it checks names, dtypes, shapes.
* Use `requant_params(s_in, ws, s_out)` for every `.m` / `.s`; bias `round(b / (s_in * ws))` as int32.
* Keep the scale chain above (one scale per hand-over tensor); fold BatchNorm / layer scales into weights.
* Golden test: run `nano_host nano.bin --ids "..."` and compare the PCM with the reference; the int8 path is exact
  integer arithmetic, so the int8 tensors must match bit for bit.
