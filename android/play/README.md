# Publishing SteerTTS on Google Play

Everything the Play Console asks for, prepared. Package **ai.whissle.steertts**, published from Whissle's
organization account (no closed-test requirement).

## Build

```bash
cd android && ./gradlew :app:bundleRelease     # app/build/outputs/bundle/release/app-release.aab
```

The bundle is signed with the release key (`~/.android/steertts-release.jks`, see `~/.gradle/gradle.properties`).
With Play App Signing, Google holds the key that signs what users install, and this key becomes the **upload key**:
keep it backed up, every future upload must be signed with it. Bump `versionCode` in `app/build.gradle` for each
upload. Target SDK 36 (Android 16), min SDK 26 (Android 8), arm64 only.

## Files here

| Play Console field | File |
|---|---|
| App name (30) | `listings/en-US/title.txt` |
| Short description (80) | `listings/en-US/short_description.txt` |
| Full description (4000) | `listings/en-US/full_description.txt` |
| App icon 512×512 | `listings/en-US/graphics/icon.png` |
| Feature graphic 1024×500 | `listings/en-US/graphics/feature-graphic.png` |
| Phone screenshots (6, 1080×1920) | `listings/en-US/graphics/phone-screenshots/` |
| Release notes | `release-notes/en-US.txt` |
| Privacy policy URL | `privacy-policy.md`, public at https://github.com/WhissleAI/steer-tts/blob/main/android/play/privacy-policy.md (or host it on whissle.ai) |
| Promo video (YouTube URL) | upload the 60 s landscape film, paste its link |

## Console checklist (in order)

1. **Create app**: name "SteerTTS: Expressive Voice", default language English (US), App, Free.
2. **Store listing**: paste the texts, upload icon, feature graphic, screenshots, video link.
   Category **Tools**; contact email (a Whissle support address); website https://whissle.ai.
3. **App content**
   - *Privacy policy*: the URL above.
   - *App access*: "All functionality is available without special access" (no login). Optional features need
     Android permissions the reviewer can grant on the app's Assistant tab.
   - *Ads*: No ads.
   - *Content rating* (IARC questionnaire): category "Utility, productivity, communication or other"; answer **No** to
     violence, sexuality, language, controlled substances, gambling; user-generated content shared with others: No;
     shares location: No; digital purchases: No.
   - *Target audience*: 13+ (not designed for children).
   - *News app*: No. *Government app*: No. *Financial features*: none.
   - *Health apps declaration* (Health Connect): reads **Steps, Sleep, Heart rate, Resting heart rate**; read only,
     no writes. Purpose: a spoken personal daily briefing the user turns on ("Last night you slept 7 hours…").
     Data is read on the device when the briefing plays, spoken and discarded; never stored, transferred or sold.
     The rationale screen Health Connect links to is `PrivacyActivity` (VIEW_PERMISSION_USAGE / HEALTH_PERMISSIONS).
     The app is not a medical device and gives no medical advice. Health category: answer **Yes** only for
     "Activity and fitness" / "Sleep" as features of a general app, not a health app.
   - *Data safety*: **No data collected, no data shared.** Rationale: speech, notification text, calendar events,
     Health Connect data, documents and settings are processed and kept on the device; nothing is transmitted to Whissle or third parties. Fetching a
     link the user shares is a user-initiated download from the site the user chose (not collection).
   - *Foreground service types*: **Media playback**, used by the document player (`ReaderService`): the user opens a
     document, shares an article or selects text, and the app reads it aloud as a media-playback foreground service
     with a MediaSession, so playback continues outside the app with play, pause and skip in its notification and on
     the lock screen. User impact if deferred: the reading the user asked for would stop or be delayed.
     A short screen recording is requested: the read-aloud segment of the demo film shows it.
   - *Exact alarms*: the app declares SCHEDULE_EXACT_ALARM (user-granted, not USE_EXACT_ALARM) and falls back to
     inexact alarms.
4. **Release**: Production → Create release → upload `app-release.aab` → release notes → Review → Start rollout.
   First review usually takes a few days.

## Policy notes (already handled in the app)

- Notification access, calendar access and Health Connect are each preceded by an in-app disclosure of what is read
  and that it stays on the phone (Play's prominent-disclosure rule for sensitive data).
- The notification permission (Android 13+) is never requested: the only notification is the player's, and
  media-session notifications are exempt from it.
- No `USE_EXACT_ALARM` (reserved for alarm and calendar apps).
- adb test hooks only work in debuggable builds; the store build ignores them.
- Model licence: the bundled edge model's weights are Whissle's own; the app is free.
