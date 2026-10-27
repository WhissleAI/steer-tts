# The director: reading longer text as a performance

SteerTTS renders one sentence at a time. Each sentence is expressive on its own, but a book read sentence by sentence
has no performance across sentences:
- no register for the genre,
- no lift for dialogue,
- no pause that says "new paragraph",
- no ending that says "more to come" or "that's it".

The **director** supplies that. It is a ~0.44M-parameter model (8-bit, ~0.5 MB, `director.bin` in the edge model
directory) that reads the text in order and plans each sentence, in the units SteerTTS's controls already use:

| Plan | Meaning | Becomes |
|---|---|---|
| `rate` | speaking rate, speaker-relative z | the `rate` control |
| `loud` | loudness, z | the `loud` control |
| `f0_med` | pitch level, z | the `f0_med` control |
| `f0_iqr` | pitch range (liveliness), z | the `f0_iqr` control |
| `end` | the final word's pitch against the sentence's, z | the `f0_end` control (× 0.8) |
| `gap` | log(1 + seconds of silence after the sentence) | the pause before the next sentence (80–1500 ms) |

It is trained on how audiobook narrators actually performed each sentence: LibriVox recordings of Project Gutenberg
books in ten genres. Every sentence was measured with the same DSP code that defines SteerTTS's controls
(`steertts/measure.py`), against the narrator's own average. It sees:
- the sentence (hashed words and word pairs; punctuation, length, quotes, dashes),
- its place in the paragraph,
- whether it is dialogue and whether a speaker is attributed,
- the next sentence,
- an optional genre,
- **its own plans for the sentences before**, so a scene keeps its register.

It runs in about 20 µs per sentence on an M3 Mac and about 60 µs on a Pixel 10, so a whole novel is planned in well
under a second, before the first sentence is rendered.

## Command line

```bash
steertts --model models/steer-tts-edge --perform chapter.txt --genre children -o chapter.wav
steertts --model models/steer-tts-edge --perform chapter.txt --plan          # print the plans, render nothing
steertts --model models/steer-tts-edge --perform poem.txt --genre poetry --strength 0.7 --voice male
```

`--perform FILE` (`-` reads stdin) splits the text into paragraphs at blank lines and into sentences, plans them, and
renders one WAV with the planned pauses between sentences.
- `--strength` scales the plan: 1 is as narrators perform, 0 is no director.
- `--ctrl` still applies and wins over the plan, so a voice's own style stays.
- `--director` points at another `director.bin`.

## C API

```c
#include "steertts.h"

steertts_director* d = steertts_director_load("models/steer-tts-edge", err, sizeof err);   /* or .../director.bin */

steertts_sentence s[] = {   /* text, dialogue, has_speaker, para_len, para_index */
  {"Dorothy lived in the midst of the great Kansas prairies.", 0, 0, 2, 0},
  {"“Where are we going?” she asked.", 1, 1, 2, 1},
};
float plans[2 * STEERTTS_PLAN_DIMS];
steertts_director_plan(d, s, 2, "children", plans, err, sizeof err);   /* genre may be NULL */

for (int i = 0; i < 2; i++) {
  const float* p = plans + i * STEERTTS_PLAN_DIMS;
  char ctrl[160];
  steertts_plan_ctrl(d, p, 1.0f, ctrl, sizeof ctrl);    /* "rate=0.12,loud=-0.30,f0_med=0.61,f0_iqr=0.52,f0_end=0.34" */
  steertts_options o;
  steertts_default_options(&o);
  o.ctrl = ctrl;
  steertts_synthesize(engine, s[i].text, &o, &audio, &n, err, sizeof err);
  int pause_ms = steertts_plan_pause_ms(p, o.pace);     /* silence to leave after it */
  /* ... */
}
steertts_director_free(d);
```

| Function | Purpose |
|---|---|
| `steertts_director_load(path, err, len)` | Load `director.bin`, or the one in a model directory. NULL on failure. |
| `steertts_director_from_memory(data, n, err, len)` | The same from bytes in memory (e.g. an Android asset). |
| `steertts_director_plan(d, sentences, n, genre, plans, err, len)` | Plan a whole text: `n × STEERTTS_PLAN_DIMS` floats. |
| `steertts_reading_begin(d, genre)` / `steertts_reading_next(r, s, next, plan)` / `steertts_reading_free(r)` | Plan sentence by sentence, keeping the context (messages arriving one by one, a reader that cannot see the whole text). `next` is the following sentence when known, else NULL. |
| `steertts_plan_controls(d, plan, strength, out[5])` | The plan as request controls (rate, loud, f0_med, f0_iqr, f0_end), calibrated for the director's voice (below), clamped to ±3. Add them to a voice's own style controls. |
| `steertts_plan_ctrl(d, plan, strength, buf, len)` | The same as a `ctrl` string, independent of the C locale. |
| `steertts_plan_pause_ms(plan, pace)` | The pause after the sentence in ms (80–1500), shorter at a faster pace. |
| `steertts_director_genre(i)` | The genre names: `children`, `adventure`, `fantasy`, `mystery`, `scifi`, `gothic`, `literary`, `humour`, `nonfiction`, `poetry`. |

Notes:
- **Thread safety:** a director is read-only after loading and can be shared across threads. A reading belongs to one
  thread.
- **NULL arguments** are errors, not crashes, as in the rest of the API.

## Python

The C library's Python binding (`c/python/steertts_c.py`, ctypes, no compilation) has the same director:

```python
from steertts_c import Director, Engine

tts = Engine("steer-tts-edge")
d = Director("steer-tts-edge")                      # the edge directory, or director.bin itself
texts = ["“Who are you?” she asked.", "Nobody answered."]
plans = d.plan(texts, genre="children")             # a list of 6 floats per sentence (one paragraph here)
for text, p in zip(texts, plans):
    wav = tts(text, ctrl=d.ctrl(p))                 # the plan as calibrated controls for the voice
    gap = Director.pause_ms(p)                      # silence to leave after it, in ms

r = d.reading("children")                           # sentence by sentence, keeping the context
p = r.next("A new message arrived.")
```

A sentence can be a string (the list is then one paragraph; quotes mark dialogue) or a dict with `text`, `dialogue`,
`has_speaker`, `para_len` and `para_index`. `Director.genres()` lists the genres. `c/tests/test_director_py.py`
checks the binding against the same golden plans as the C test.

## Calibration: asking the voice for what the plan means

The voice doesn't deliver a requested control change in full. Measured on the edge voice (226 random requests,
`steertts/measure.py`), it delivers this share of the change:

| rate | loudness | pitch level | pitch range | ending |
|---|---|---|---|---|
| 75% | 96% | 64% | 43% | 29% |

The cross-effects between controls are small. `director.bin` carries the inverse of this response for its paired
voice, so `steertts_plan_controls` and `steertts_plan_ctrl` ask for what makes the voice perform the plan (pass the
director; with NULL the plan is requested as is).

Rendering 161 held-out sentences with and without their plans, the measured change follows the plan for:
- rate: r = 0.94
- loudness: r = 0.92
- endings: r = 0.54 (0.10 uncalibrated)
- pitch range: r = 0.39 (0.20 uncalibrated)

Small pitch-level plans (a spread of about 0.3 z) stay below the voice's own sentence-to-sentence variation (r ≈ 0.1).

## Using it well

- **Plan a document once, when it opens.** The plans are 6 floats per sentence. Keeping them with the document makes
  seeking, resuming and exporting consistent, because every sentence keeps the plan it got in context.
- **Headings are not planned.** The training data left headings out, so render them with your own delivery (slower,
  a pause after), and plan the body text.
- **Dialogue:** narrators keep their own voice and lift the pitch for dialogue; the director plans that lift. A
  different voice per character is a separate choice.
- **Strength:** 0.6–1.0 suits most listeners. Plans add to the user's own style; they don't replace it.
- **Genre** steadies the register when it is known, for example from a library shelf. Without one the director still
  works: it was trained with the genre hidden a third of the time.

## How well it does

On whole books it never saw in training, the director tracks the narrator's speaking rate, pitch level, pitch range,
sentence endings and pauses. Narrators don't fully agree with each other either, so their agreement on the same
sentences is the ceiling:

| Correlation with the narrator | rate | loud | f0_med | f0_iqr | end | gap |
|---|---|---|---|---|---|---|
| Rule-based moods (no director) | 0.01 | 0.05 | 0.09 | 0.05 | −0.01 | −0.01 |
| Director, format 1 (50-hour pilot) | 0.39 | −0.04 | 0.21 | 0.18 | 0.12 | 0.47 |
| **Director, format 2 (shipped)** | 0.39 | 0.15 | 0.30 | 0.24 | 0.15 | 0.39 |
| Two narrators, same sentence | 0.61 | 0.33 | 0.36 | 0.21 | 0.26 | 0.48 |

The training data is 50 hours: 30 recordings, 28 books, 10 genres, with 5 books read by two or three narrators. Six
books are held out. The shipped model (format 2) adds scene memory (the mean of the last 8 sentence encodings) and
learns from a larger text-only teacher's labels on more books at a low weight: loudness, pitch level and range
improve, and pauses give back a little (0.47 → 0.39). Format-1 files still load.

The C implementation is checked against the trained PyTorch model on 525 held-out sentences in `c/tests/test_director.c`.
