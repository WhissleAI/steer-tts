# The request

Every SteerTTS synthesis, in every build, is one request: text plus optional voice, style tag, controls and markup.
The Python packages take it as keyword arguments or a dict, the C/C++ runtime as a `steertts::Request`, and the full
model also accepts it as JSON ([`request.schema.json`](../request.schema.json), JSON Schema 2020-12). Only `text` is
required.

| Field | Meaning | v1 | edge |
|---|---|:-:|:-:|
| `text` | English text. Numbers, units, currency, dates, times, ordinals, ranges and unknown acronyms are spelled out. | ✓ | ✓ |
| `voice` | A built-in voice (`voice_1688` by default). v1 ships 2,825 voices, edge 200. | ✓ | ✓ |
| `tag` | One of 51 style tags (`none` = unspecified). | ✓ | ✓ |
| `ctrl` | `{control: z}` over the 12 controls below, each in [−5, 5]. | ✓ | ✓ |
| `pace` | Global tempo divisor applied after prediction (>1 is faster). Changes timing only, not style. | ✓ | ✓ |
| `guidance` | Steering scale *s* (default 1). See [Guidance](#guidance). | ✓ | ✓ |
| `word_shift` | `{word_index: {pitch: semitones, energy: dB, dur: factor}}`: per-word prosody edits. Not with `markup`. | ✓ | ✓ |
| `markup` | Parse inline markup in `text` (see [Markup](#markup)). `stream()` always parses it. | ✓ | ✓ |
| `prompt` | Free-text direction ("excited, a bit faster"), mapped to `tag` + `ctrl` by a small text encoder. | ✓ | host only¹ |
| `ref_wav` | Clone a voice from a reference clip (first 8 s, at least 0.5 s). | ✓ | host only¹ |
| `flow`, `seed` | Flow-matching mel refiner (v1 default on) and its sampler seed. | ✓ | — |

¹ The edge build leaves the prompt encoder (MiniLM) and the speaker encoder (WavLM) off the device. Resolve a prompt
on a host with the full model (`SteerTTS.resolve()`) and send the resulting `tag` + `ctrl`; for cloning, compute the
512-d speaker vector on a host and pass it as `spk_vec`.

## Controls

The 12 controls are utterance-level acoustic measurements. Each value is a **z-score against the chosen voice's own
neutral speech**, in units of the pooled within-speaker standard deviation: `0` is this voice speaking normally, `+1`
is one typical step up.

| Control | Measures | + means |
|---|---|---|
| `rate` | articulation rate (phones per second of speech, pauses excluded) | faster |
| `loud` | speech level (dB) | louder |
| `f0_med` | median pitch (semitones) | higher |
| `f0_iqr` | pitch range (semitones) | livelier, less monotone |
| `f0_end` | pitch slope over the final 300 ms (st/s) | rising, question-like |
| `pause` | share of the speech span that is pauses ≥ 100 ms | more pauses |
| `tilt` | energy 1–5 kHz vs 50 Hz–1 kHz (dB) | brighter, more pressed |
| `voiced` | voiced share of speech frames | less breathy (whisper is low) |
| `flat` | spectral flatness | breathier, noisier |
| `centroid` | spectral centroid | brighter |
| `cpps` | cepstral peak prominence (periodicity) | clearer, less breathy |
| `h1h2` | H1–H2 (dB) | breathier |

**The mask.** The model also learns which controls were requested. A control you name is requested; a control you
leave out is *unspecified* and the model picks it from voice, tag and text. So `{"rate": 0}` asks for this voice's
normal rate, while omitting `rate` lets a `fast` tag or the text decide.

**What moves and what doesn't.** `loud`, `f0_med`, `rate`, `h1h2` and `tilt` move strongly and reliably (a request
of ±2 achieves about ±1–1.9 z). `f0_iqr` moves about half as far. `voiced`, `flat` and `cpps` barely move on their
own; use the `whisper` tag for breathy speech. Values beyond ±3 are outside the well-tested range.

## Style tags

`none`, `adoration`, `amazement`, `amusement`, `anger`, `anxiety`, `apologetic`, `bored`, `confidence`, `confusion`,
`contempt`, `contentment`, `cuteness`, `desire`, `determined`, `disappointment`, `disgust`, `distress`, `ecstasy`,
`embarrassment`, `emphasis`, `enunciated`, `excited`, `fast`, `fear`, `frustration`, `guilt`, `happiness`, `hesitant`,
`high_pitch`, `interest`, `irritation`, `laughing`, `loud`, `low_pitch`, `mysterious`, `neutral`, `nostalgia`, `pain`,
`panic`, `pride`, `realization`, `relief`, `sadness`, `sarcasm`, `serenity`, `skeptical`, `slow`, `sympathy`, `tired`,
`whisper`.

Tags are **style switches, not reliable emotions**. Voice-quality and delivery tags act strongly (`whisper`, `loud`,
`slow`, `high_pitch`, `low_pitch`). Of the emotion tags, an independent emotion classifier recognises only `sadness`;
27 of 49 tags are acoustically close to `neutral`. For a dependable effect, combine a tag with explicit controls,
e.g. `tag="sadness", ctrl={"loud": -1, "rate": -0.7}`.

## Guidance

`guidance = s` renders the request and the same request without tag and controls, then extrapolates durations,
pitch, energy and the mel spectrogram: `out = out₀ + s · (out − out₀)`. `s = 1` is off; `s = 1.5–2` strengthens the
requested change (about ×1.7–2 at `s = 2`) at the cost of a second encoder/decoder pass and some leakage into
controls you didn't ask for. It only applies when a tag or controls are given.

## Markup

With `markup` on (and always in `stream()`):

| Markup | Effect |
|---|---|
| `*word*` or `<emph>…</emph>` | Emphasis: +2.5 semitones, +4 dB, ×1.2 duration |
| `<slow>…</slow>`, `<fast>…</fast>` | Per-word duration ×1.4 / ×0.75 |
| `<high>…</high>`, `<low>…</low>` | Per-word pitch +3 / −3 semitones |
| `<loud>…</loud>`, `<soft>…</soft>` | Per-word energy +6 / −6 dB |
| `[pause]`, `[pause 300ms]`, `[pause 1.5s]` | Exact silence (400 ms by default, at most 5 s) |
| `<whisper>…</whisper>` (any tag name) | The span is rendered as its own segment with that tag |
| `<style prompt="…">…</style>` | The span is rendered with a free-text prompt (v1 only; edge raises an error) |
| `[laugh]` `[cough]` `[sigh]` `[sneeze]` `[throat]` | In-voice sounds from the recorded event bank: a real recording of the voice's gender and nearest pitch, re-pitched to the voice, 2 dB under its level, as its own chunk |
| `[dog]` `[knock]` `[door]` `[phone]` `[bell]` … | Environmental sounds from the same bank (full package), 3 dB under the speech. A token the bank lacks reserves about 0.12 s and renders no sound |

The per-word tags nest and stack (`<slow><high>…</high></slow>`), and the sentence is still rendered in one pass, so
the prosody around an edited word stays natural. Text is split into segments at sentence ends and at style-span
boundaries; segments are rendered one at a time, so the first sentence can play while the rest renders, and 10 ms
fades join them without clicks. Unknown tags are dropped and their text is spoken normally.

## Examples

```python
tts("Are you coming with us tonight?", ctrl={"f0_end": 2, "f0_iqr": 1.5})     # rising, livelier
tts("Please keep this between the two of us.", ctrl={"rate": -1.5, "loud": -1.5})   # slow and soft
tts("Please keep this between the two of us.", tag="whisper")
tts("I really thought it would work this time.", tag="sadness", guidance=1.5)
tts("Word shift on one word here.", word_shift={3: {"pitch": 3, "energy": 4, "dur": 1.3}})   # word 3 = "one"
tts("Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow> <whisper>Don't tell anyone.</whisper>",
    markup=True)
```

```bash
steertts-edge "I can't believe it worked." --tag happiness --ctrl rate=1,f0_iqr=1.5 -o happy.wav
```
