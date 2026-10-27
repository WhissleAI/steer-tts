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
5. **Character voices and delivery (product done with workarounds; quality to measure).** Books are now performed
   (`Cast.java`): the speaker of each paragraph is found from "said X" / pronouns / turn-taking (96% of Alice's dialogue
   paragraphs get a speaker), each character gets a catalogue voice, and speech verbs pick a delivery (whispered ->
   `whisper`, shouted -> `loud` + controls, sobbed -> sadness...). Each quote is a separate request in another voice,
   so the seams between narrator and character may be audible. *Train:* inline speaker switches with natural
   transitions; voices chosen for distinctness (the catalogue has 200, but which pairs are clearly different to a
   listener is unknown); verb-style deliveries (whisper, shout, sob, laugh, mutter) as tags that work without controls.
18. **Context steering is small in the audio (measured 2026-10-27, Android).** The same readout at 10 AM and 11 PM
    (ContextStyler's night rule: softer and slower) measured −24.8 vs −26.7 dB, only 1.9 dB apart, hardly audible on a
    phone speaker. Urgent battery and "calling again" readouts were also only mildly different in the film takes.
    *Train:* a wider, reliable `loud` and energy range (soft and intimate through to urgent) that holds on every
    voice; per-voice calibration of how much a requested change actually lands.
    More evidence (2026-10-27, Mac, The Tell-Tale Heart at 2.5x): scaling the director's plan down to 40 % changed the
    pitch spread only from 4.3 to 4.1 semitones (middle half) and didn't reduce loudness variation (2.4 vs 2.8 dB
    sd). At speed, the voice's own melody dominates and the plan barely moves it.
19. **Short single words lose identity in dark or soft deliveries (heard 2026-10-27, films).** The one-word close
    "Steer." in a low, dark delivery was misheard (as "steel", "stare"); the films switched to a brighter "Meet
    Steer." *Train:* intelligibility of one- and two-word utterances under every tag and low-energy control; add
    such lines to the Whisper round-trip evaluation.
20. **Everything is trained on books, but the apps mostly don't read books (verified by product use).** The
    assistant speaks messages, calls, coding-task results, battery and meeting alerts, and the system voice and
    screen reader speak any app's text. All of these are short, situational lines where the right delivery depends
    on who sent it, what app, the time and whether you're working, which audiobook narration never shows. The apps
    hard-code it today (`ContextStyler`, `Expression` moods, the Mac `Announcer`). *Train:* (a) acoustic data beyond
    narration: announcements, news and briefings (VOA, NASA mission audio, public-domain radio), conversational
    turns (DailyTalk), emotional short lines; (b) a recorded scenario corpus where the same app line is read in
    several contexts (late night, urgent, calling again, good vs bad news) by several speakers; (c) the director's
    second iteration takes app context (content type, app class, sender, urgency, time, presence) as input and
    learns from that corpus plus a generate-and-judge loop. See the director's iteration 2 plan (internal recipe,
    `steering_controls/ITER2.md`).

6. **Nonverbals for chat (to measure).** "haha", "lol", 😂 are currently dropped and only the mood (amusement) is
   applied. The `[laugh]` event token exists; measure its quality before the app inserts it for laughter.

## Voices

7. **The default voice is a slow speaker (verified).** The app ships `DEFAULT_PACE = 1.1` (`Style.java`) so the
   default voice does not drag. *Train:* normalise speaking rate across voices, or pick defaults by rate.
8. **Metallic sound on the phone (heard twice by the user, 2026-10-25; cause not yet proven).** Measured on the
   rendered audio (the same C engine and model the app ships): no clipping (peaks ~0.4), no frame-rate buzz (the
   envelope's modulation peaks at the voice's own pitch). The edge build quantizes **both** the acoustic model and a
   small Vocos-style vocoder (6.2M params) to **int4**; docs/models.md shows the gap: UTMOS 4.08–4.11 against 4.24–4.27
   for v1 (13.5M fp32 Vocos). The app had made it worse by playing as media audio (Pixel processing), now fixed.
   *Next:* A/B the edge vocoder at int4 vs int8/fp16 (same weights) and against v1 Vocos on phone speakers; if int4 is
   the cause, ship the vocoder at int8 (a few MB more), or train a larger edge vocoder. Listen for buzz on sustained
   vowels and sibilants.
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
15. **The engine's own normalisation is narrow, so the app now writes everything out (verified 2026-10-25).** Heard
    by the user: "CHAPTER I" read as the letter I, accented names garbled, a calendar time misread. Measured with the
    C front end: "CHAPTER I." and "II." stay letters (the Roman rule is case-sensitive and needs a keyword), "1990s" →
    "nineteen ninetys", "'80s" → "'eightys", "₹500" → "₹five hundred", "#12345" → "#twelve thousand…", "221B" →
    "two hundred twenty oneB", "Q3" → "Qthree", "Café/Zoë/Dvořák" passed through with accents (the G2P sees no
    letters for é, ë, ř), URLs and emails read character soup, "Mon, Oct 5" stays abbreviated, non-Latin scripts
    reach the G2P. `Spoken.java` (app) now writes all of these as words before the engine (81 unit cases, four
    Gutenberg books checked for false positives, Whisper round-trip on the edge model). *Train:* (a) move this
    normalisation into the shared front end (Python + C, with golden tests) so every client gets it; (b) train on
    normalised text for these classes (times "ten to eleven pee em", digit strings read in groups, ordinals,
    regnal numbers "Henry the eighth") so prosody fits: digit groups need short pauses, times a falling end;
    (c) accented and foreign names: a G2P that reads Latin letters with diacritics (São Paulo, Dvořák, Zoë) instead
    of the app stripping accents; (d) multilingual: Hindi/Hinglish, Devanagari and Cyrillic text is currently dropped
    (silence) — a voice for at least Hinglish messages is a real gap for the assistant.

17. **Brand names and tech words are mispronounced (verified 2026-10-26, edge model, Whisper round-trip).** Found
    recording the Mac demo: "WhatsApp" → "What's Baff", "iMessage" → "a message", "login" → "Logan", "GitHub" →
    "Gethub" (respelled "Git Hub": "JIT Hub"), "npm" → "pen line" (fine as "N P M"). FaceTime, Slack, LinkedIn,
    YouTube, iPhone, macOS, Microsoft Teams, Zoom, PDF were fine. The apps now respell WhatsApp, iMessage and login
    (`Spoken.respell`), but every client would need its own list. *Train:* a lexicon of app, product and developer
    terms (camel-case brands, CLI names) in the G2P training data, and camel-case splitting in the front end.
    Also "whissle.ai": "whissle dot ai" is heard "whistle dot eye" (recogniser, 2026-10-27); the apps now say
    "dot A I". "Whissle" is heard "Whistle", "GitHub" as "Githab". These belong in the lexicon and the shared front
    end. An audit of 53 app names ("Maya on X.", two voices, a recogniser) found 17 heard wrong; respelling fixed 11
    in the apps (Gmail heard "game", PayPal "people", DoorDash, WeChat, Coinbase, Xcode, npm, Grubhub, TikTok,
    Airbnb, Duolingo). No respelling fixed **Zelle** ("aunzel"), **Jira** ("Anjora") or **Calendly** ("Callen lie"):
    these need the G2P itself.

12. **Long sentences (to measure).** Documents are split at 240 characters at a comma or semicolon for first-sound
    latency. Measure quality and prosody on 200–400 character inputs; if good, the split can be longer.

## Reading documents

14. **Prosody across sentences (to measure).** The reader and the assistant render one sentence per request (for
    first-sound latency and the sentence highlight), so paragraph-level intonation, declination and topic continuity
    are lost at every boundary. *Train/serve:* context from the previous sentence (text and prosody state) as input.
13. **Headings and list items (to measure).** The reader slows headings with controls (`Narration.HEADING`). A
    `heading`/`list` delivery learned from audiobook and lecture data would sound more natural than control shifts.
16. **Sentence endings and pauses don't follow the text, and dynamics are flat (measured 2026-10-26, small sample).**
    Two LibriVox narrators read chapter 1 of *The Wonderful Wizard of Oz*. Each sentence was measured with
    `steertts/measure.py` and compared with the edge voice reading the same sentences with no tag or controls.
    - The voice follows the narrators' rate from text alone (r = 0.70), and loudness partly (0.44).
    - It doesn't follow sentence endings (`f0_end` r = 0.07) or pauses (−0.24). Yet the two narrators agree with
      each other on endings (0.48) and rate (0.55), so endings are predictable from text.
    - Across sentences, humans vary loudness 2–3× more than the voice and pauses 1.5–3.5× more.

    *Train:* learn contour and pause from text context, e.g. by adding audiobook data with paragraph context. Plan
    a "director" model, which predicts per-sentence controls from context, trained on the same measurements at scale.
