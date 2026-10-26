# SteerTTS for the Mac

The SteerTTS voice on the Mac: books, PDFs, web pages and any selected text, read aloud on this Mac with nothing sent
anywhere. It is built from the Android app's logic, ported to Swift and tested against the Android test cases.

- **Reading list and player.** Open PDFs, EPUBs, Word documents, HTML, Markdown and text, or a link (its article).
  The reader highlights the sentence being heard and keeps it in view; click a sentence to read from there. The
  player has speed, voice, a sleep timer and AirPlay (HomePod, Apple TV, AirPlay speakers). Media keys, AirPods and
  Control Center work too.
- **Stories are performed** as on Android: dialogue in a second voice, characters in voices of their own, headings
  slower, italics stressed.
- **Free books** from Project Gutenberg: shelves and search. A book you already have opens where you left it.
- **Voices:** 200 named voices, styles and delivery controls, previewed as you change them.
- **Read anywhere:**
  - ⌃⌥R reads the selection in any app (needs Accessibility access).
  - Services > Read with SteerTTS works without any permission.
  - ⌃⌥V reads the clipboard, and ⌃⌥Space plays or pauses.
  - The menu bar item shows what is playing, with controls.
- **Save as audio:** any document as an .m4a audiobook in your voice and style. An 8-hour novel takes about 7 minutes.
- **Text the way people say it:** times, dates, money, units, Roman numerals, codes and accented letters
  (`SteerText/Spoken.swift`, the port of Android's `Spoken.java`).

## Layout

| Module | What |
| --- | --- |
| `Sources/CSteerTTS` | The C engine, compiled from `../c/src` through symlinks |
| `Sources/SteerText` | The text pipeline in plain Swift: `Spoken`, `TextClean`, `Extract` (EPUB/DOCX via `Zip`), `Article`, `PdfLayout`, `Narration`, `Cast`, `Expression`, `Style`, `Voices`, `ContextStyler`, `Doc`. Ports of the Android classes of the same names; keep them in step |
| `Sources/SteerEngine` | `Engine` (the C API), `SentencePlayer` (AVQueuePlayer, one item per sentence, so AirPlay works), `AudioFileWriter` |
| `Sources/SteerTTSApp` | The SwiftUI app |
| `Tests/SteerTextTests` | The Android unit tests, ported, plus Mac-only ones (55 tests) |

## Build

```
swift test                      # the text pipeline
./build.sh --dev                # arm64, ad-hoc signed, dist/SteerTTS.app
./build.sh                      # universal, Developer ID signed, dist/SteerTTS-<version>.dmg
./build.sh --notarize           # also notarize and staple (xcrun notarytool store-credentials WHISSLE_NOTARIZE first)
```

The model comes from `../c/models/steer-tts-edge` (`c/tools/get_models.sh`) and is copied into the app.

Developer hooks, like the Android debug build's adb hooks: start with
`STEERTTS_DEV=1 dist/SteerTTS.app/Contents/MacOS/SteerTTS`, then send commands with `tools/dev.sh`, e.g.
`tools/dev.sh section:voices`, `tools/dev.sh open:/path/book.epub`, `tools/dev.sh snapshot:/tmp/s.png` or
`tools/dev.sh log`. See `Sources/SteerTTSApp/DevHooks.swift`.
