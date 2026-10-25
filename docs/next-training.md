# What to improve in the next round of SteerTTS training

A running record of what building real products on the edge model (the Android app: assistant readouts, document
reading, the system voice) shows the model should do better. Each item says what was seen, where, and what the app does
today to work around it, so the workaround can be removed once the model handles it. **Verified** means measured or
heard; **to measure** means a hypothesis from product work that needs an evaluation before it drives training.

Kept up to date as the app is built. Newest first within each section.

## Expressiveness and control

1. **Emotion tags barely change the voice on their own (verified).** `docs/request.md`: an emotion classifier only
   recognises `sadness`; 27 of 49 tags are acoustically close to `neutral`. The app had to pair every mood with control
   shifts to be audible (`Expression.java`, `Style.preset`: e.g. excited = tag + rate +0.7, loud +0.9, f0_iqr +1.4,
   f0_med +0.9). *Train:* tag conditioning strong enough that `sympathy`, `excited`, `amusement`, `apologetic`,
   `anxiety`, `irritation`, `contentment`, `pride` are distinguishable by listeners and a classifier without
   controls. These are the tags the assistant uses most.
2. **`f0_iqr`, `voiced`, `flat`, `cpps`, `centroid`, `pause` follow requests weakly (verified, docs/request.md).** The
   app hides them under "Advanced" with a note. *Train:* larger effect sizes, or drop controls that cannot be learned.
3. **Questions need an explicit rising end (to measure).** The app adds `f0_end +1.4` to every question
   (`Expression.of`, `ContextStyler`). Check whether the model's own question intonation is reliable on yes/no
   questions; if not, learn it from punctuation.
4. **Mood changes inside one utterance (to measure).** A readout is rendered as separate parts ("Sam on WhatsApp" in a
   neutral voice, then the message in a mood and another voice) and concatenated with a pause. Boundaries may sound
   stitched. *Train:* segment-level style and speaker changes within one request with natural transitions (style
   spans already exist in markup; speaker spans do not).
5. **Dialogue voices (to measure).** Books read dialogue in a second voice by rendering quotes as separate requests
   (`Narration.java`). *Train:* an inline speaker switch, and consistent character voices across a chapter.
6. **Nonverbals for chat (to measure).** "haha", "lol", 😂 are currently dropped and only the mood (amusement) is
   applied. The `[laugh]` event token exists; measure its quality before the app inserts it for laughter.

## Voices

7. **The default voice is a slow speaker (verified).** The app ships `DEFAULT_PACE = 1.1` (`Style.java`) so the
   default voice does not drag. *Train:* normalise speaking rate across voices, or pick defaults by rate.
8. **"Metallic" sound on the phone (heard, earlier testing).** Reported when listening on a Pixel; audit the
   vocoder in the int8/edge build against the float model on phone speakers.
9. **Voice metadata is thin (verified).** voices.json has gender, pitch register and pace only; the app invents names
   and descriptions. *Train/export:* richer, listener-validated descriptors (warmth, age range, accent) per voice.

## Text front end (fixed in code, but worth data)

10. **Out-of-dictionary words go to the neural G2P and come out garbled (verified).** Every hyphenated compound and,
    worst, every number 21–99 ("seventy-two" → S EH V AH N T EH F Y UW) until the front end dropped the hyphens.
    *Train:* a G2P robust to compounds, names and brand words; add such words to G2P evaluation.
11. **Real-world text classes (verified, now normalised in the front end):** clock times and AM/PM, phone numbers,
    titles (Dr., Mrs.), °F/°C, currency with K/M/B, versions, Roman numerals after "Chapter", fractions, chat
    abbreviations, emoji. *Train:* include these in training text so the acoustic model has seen their prosody
    (lists of digits, times), not just their words.
12. **Long sentences (to measure).** Documents are split at 240 characters at a comma or semicolon for first-sound
    latency. Measure quality and prosody on 200–400 character inputs; if good, the split can be longer.

## Reading documents

14. **Prosody across sentences (to measure).** The reader and the assistant render one sentence per request (for
    first-sound latency and the sentence highlight), so paragraph-level intonation, declination and topic continuity
    are lost at every boundary. *Train/serve:* context from the previous sentence (text and prosody state) as input.
13. **Headings and list items (to measure).** The reader slows headings with controls (`Narration.HEADING`). A
    `heading`/`list` delivery learned from audiobook and lecture data would sound more natural than control shifts.
