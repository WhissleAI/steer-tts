# SteerTTS on Android

A small demo app that runs the edge model on the phone, offline: type text, pick a voice (female / male defaults or any
of the 200), hear it as it renders. The app's CMake build compiles the C library from [`../c`](../c) into its JNI
library, so there is no prebuilt binary to keep in step.

```bash
c/tools/get_models.sh                        # once: the edge model in c/models/steer-tts-edge (not in git)
cd android && ./gradlew assembleRelease      # or: -PsteerttsModel=/path/to/steer-tts-edge
adb install -r app/build/outputs/apk/release/app-release.apk
```

Needs the Android SDK (platform 34), NDK 27.1.12297006 and CMake 3.22.1 (Android Studio's SDK manager), and a JDK
Gradle 8.2 runs on (17 to 20). The APK is arm64-v8a only, 13.5 MB, of which 12.6 MB is the model; it is unpacked to
app storage on first start.

Driving it from a computer (results in `adb logcat -s SteerTTS`):

```bash
adb shell am start -n ai.whissle.steertts.demo/.MainActivity --es speak "Hello there." --es voice male
adb shell am start -n ai.whissle.steertts.demo/.MainActivity --es bench 0,1,4,6     # 0 = Auto
```

## Using it in your app

Copy `app/src/main/cpp` (JNI bridge + CMake), `SteerTTS.java` and `Player.java`:

```java
SteerTTS tts = new SteerTTS(SteerTTS.installModel(context).getPath(), 0);   // 0 = Auto threads
Player player = new Player();
tts.stream("Hello there. <whisper>Quietly now.</whisper>", tts.genderVoice("male"), null, 1f, (audio, i, ms, text) -> {
    player.write(audio);       // plays while the next sentence renders
    return true;               // false stops the stream
});
player.end();                  // plays out the last sentence
```

Call the engine from one thread (the demo uses one `HandlerThread`); `cancel()` may be called from any thread.

## Speed on a Pixel 10 (Tensor G5)

Render time over 65.6 s of speech (20 sentences), and the time from a request to its first audio. Real-time factor
(RTF) = render time / audio duration; 0.02 is 50x faster than real time.

| Setup | RTF | First audio |
|---|---|---|
| App, Auto (6 threads on the fast cores) | 0.017 – 0.020 | 25 ms |
| App, 4 threads | 0.021 – 0.023 | 31 ms |
| App, 1 thread | 0.038 | 56 ms |
| CLI, 8 threads, not pinned | 0.151 | 345 ms |

What makes the difference is where the threads run. The Tensor G5 has 1 + 5 fast cores and 2 slow ones (a fifth of the
capacity). The engine's threads meet after every layer, so one of them on a slow core holds the others up: 8 unpinned
threads are 7x slower than 4. The JNI bridge pins the engine thread to the cores with at least half the top capacity
(the engine's workers inherit that) and Auto uses one thread per fast core, up to 6.

Compiler flags matter little: `-O3`, `-mtune=cortex-x4` and `-mcpu=cortex-x4` were within 5–10 % of `-O2` (noise
level under the phone's thermal throttling), and `-mcpu=cortex-x4` would exclude older phones, so the app builds with
`-O2`. All of those builds, at 1 to 6 threads, render bit-identical audio; the Mac's render of the same text differs by
at most 3·10⁻⁵ (85 dB SNR, libm rounding). Memory: 29 MB peak.

Playback: `Player` feeds an `AudioTrack` from its own thread, so rendering never waits on the speaker, and drains the
track with `stop()` at the end of each utterance (a streaming track otherwise holds its last partial buffer back until
more audio arrives).
