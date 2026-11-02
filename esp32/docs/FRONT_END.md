# Text front end on the device

`components/steertts_nano/front/` turns text with steering markup into the phoneme ids, word indices and per-word
prosody shifts the Nano model reads. It behaves like the SteerTTS edge text front end (and the C library's
`c/src/front`): same normalisation of numbers, dates, times, currencies, units, abbreviations and acronyms, same
CMUdict lexicon, heteronym choice, neural G2P for unknown words, and the same markup. It is written for a
microcontroller without PSRAM:

| | |
|---|---|
| Data | one read-only file, `front.bin` (1.8 MB), read in place from memory-mapped flash |
| Rules | plain C, one matcher per expression of the reference (no regular-expression engine, nothing parsed at run time) |
| RAM | `nano_front_ctx` 9.5 KB + one `nf_segment` 2.1 KB, static, reused for every sentence; no heap |
| Stack | about 2 KB above the caller (5.2 KB measured for the whole benchmark task) |
| Limits | 640 code points per sentence after normalisation, 512 phoneme ids, 32 emphasised words per segment |

```c
#include "nano_front.h"
#include "nano_partition.h"

static nano_front_ctx fc;
static nf_segment seg;
static int ids[NF_MAX_PH], widx[NF_MAX_PH];

nano_partition_map map;
nano_front_open_partition(&fc, "steertts", &map);           /* or nano_front_open(&fc, ptr, len) */
nano_front_markup_begin(&fc, "Hello <slow>there</slow>! [pause 300ms] It's 4:15 pm.",
                        NULL, NULL, NULL, 0, tags, n_tags, NULL, NULL);
while (nano_front_markup_next(&fc, &seg) > 0) {             /* one segment (sentence / style run) at a time */
  if (seg.pause_ms > 0) { /* silence */ continue; }
  int n = nano_front_phonemes(&fc, seg.text, ids, widx, NF_MAX_PH);
  /* nano_request: ids, n, widx, seg.ws (word shifts), seg.tag, seg.ctrl_* */
}
```

The markup is parsed in a stream: each call returns the next segment as soon as its end is known, so a long text
needs no more memory than one sentence.

## front.bin

Built from the edge model's front end assets (`native/front` of `WhissleAI/steer-tts-edge`):

```bash
python tools/make_front_bin.py <steer-tts-edge>/native/front -o front.bin
```

| Section | Size | Content |
|---|---|---|
| lexicon | 950 KB | 81,724 CMUdict entries, sorted, prefix-compressed in blocks of 32 with a block index; binary search over the block heads, then a scan of one block |
| oov | 20 KB | the cached G2P predictions of 10,358 common unknown words, same format |
| gru | 742 KB | the g2p_en GRU encoder-decoder, int8 weights with per-row scales, every array 16-byte aligned |
| hetero | 76 KB | 371 heteronyms and the averaged-perceptron weights / tag dictionary, sorted string tables |
| unicode | 12 KB | the CPython character tables the rules need (\w, \s, \d, case folding, NFD cleaning) |

The file starts with `STFB`, a version (1) and a section table (id, offset, size); see `tools/make_front_bin.py`.

## How the reference is reproduced

* Normalisation is the reference's sequence of about 75 `re.sub` passes, in order. Each pass is a matcher tried at
  every position from left to right (Python's leftmost, non-overlapping semantics) that writes its replacement on
  success. The expression is quoted above each matcher. Where backtracking can make Python pick a shorter
  alternative (comma numbers, Roman numerals, acronyms, times), the matcher tries the alternatives in the same
  order. `\w`, `\d`, `\s` and `\b` use the CPython Unicode tables, so non-ASCII digits and spaces behave as in Python.
* Two buffers of 640 code points ping-pong between the passes.
* Number words (num2words, inflect, g2p_en `normalize_numbers`) are ported with fixed buffers, including the Python
  exceptions the reference raises (`OverflowError` past 10**102, `IndexError` past inflect's scales).
* The tokenizer, CMUdict lookup, heteronym perceptron and the OOV cache follow the edge `EdgeG2p` exactly.
* The GRU reads its int8 weights from flash. The hidden state is int16, split into two int8 halves so that the int8
  SIMD dot product computes the exact int16 x int8 sum. Gates are float32. The state lives in a normalisation buffer
  that is idle during G2P.
* Markup is the reference's tokenizer, run on the caller's UTF-8 text directly: `*x*` emphasis, `<tag>` styles,
  `<slow>` / `<fast>` / `<high>` / `<low>` / `<loud>` / `<soft>` / `<emph>` prosody, `[pause N ms|s]`, `[event]`,
  sentence splitting, and the reference's quirks. It runs as a streaming state machine.

## Conformance

Measured against the reference's golden files: `native/tests/front/golden.jsonl` (normalisation, phoneme ids, word
indices, errors, markup), `golden_g2p.jsonl` (GRU) and the 54 shared normalisation cases of `tests/front_cases.json`.

| Check | Exact | Cases | Rate |
|---|---|---|---|
| normalize | 6,397 | 6,425 | 99.56 % |
| phoneme ids | 6,383 | 6,411 | 99.56 % |
| word indices | 6,383 | 6,411 | 99.56 % |
| errors (inputs the reference rejects) | 14 | 14 | 100 % |
| markup segments | 606 | 607 | 99.84 % |
| markup errors | 7 | 7 | 100 % |
| GRU G2P | 33,265 | 33,265 | 100 % |
| shared normalisation cases | 54 | 54 | 100 % |

Every remaining difference is a deliberate device limit, not a rule difference:

* **28 inputs longer than a sentence buffer** (640 code points after normalisation). These are whole paragraphs of
  650 to 20,000 characters, and 100-digit numbers whose words alone exceed 640 characters. The device returns
  `ValueError: text too long for the device front end` instead of processing them. The markup path splits text into
  sentences first, so this only affects single sentences that are this long. Every case within the limit is exact.
* **1 markup case with more than 32 emphasised words in one segment** (`*a* *a* ...`), above `NF_MAX_WS`.
* **Error messages:** 7 of the 14 rejected inputs give the reference's exact message. The other 7 are rejected with
  a shortened message: the device does not echo 100-digit numbers back.
* Not on the device: free-text `<style prompt="...">`, which needs the prompt model and raises the same `ValueError`
  as the edge build, and the G2P prediction cache (an LRU of up to 16,384 words in the reference). An unknown word
  is predicted again each time it occurs.

Run the test:

```bash
python host/front/golden_to_tsv.py ../native/tests/front ../tests/front_cases.json > golden.tsv
cmake -S host -B build-host -DNANO_FRONT_BIN=$PWD/front.bin -DNANO_FRONT_GOLDEN=$PWD/golden.tsv
cmake --build build-host && ./build-host/test_front front.bin golden.tsv
```

## Speed

On the ESP32-S3 the front end is mostly table lookups in flash.

* A normal sentence takes a few milliseconds: about 70 normalisation passes over ~100 code points, plus one lexicon
  lookup per word (about 12 block heads and one block, a few cache lines).
* A word that is in neither the lexicon nor the OOV table runs the GRU: about 20 steps, each reading a 196 KB matrix
  from flash. Measured on the board: about 0.27 s per unknown word, mostly flash time.
* The benchmark prints `RESULT|front ...` lines for a plain sentence, a sentence with numbers / dates / money, and
  one with unknown words.
