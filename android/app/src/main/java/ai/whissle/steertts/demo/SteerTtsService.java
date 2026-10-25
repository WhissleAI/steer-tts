package ai.whissle.steertts.demo;

import android.content.Intent;
import android.media.AudioFormat;
import android.os.Build;
import android.os.Process;
import android.speech.tts.SynthesisCallback;
import android.speech.tts.SynthesisRequest;
import android.speech.tts.TextToSpeech;
import android.speech.tts.TextToSpeechService;
import android.speech.tts.Voice;
import android.util.Log;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * SteerTTS as Android's text-to-speech engine (Settings > Accessibility > Text-to-speech output > Preferred engine):
 * every app that speaks through android.speech.tts (screen readers, navigation, notification readers) gets the edge
 * model, on the phone. Runs in its own process (":tts"). Text is read as plain text, one sentence at a time, each sent
 * as soon as it is rendered. The user's base style from the app applies, adjusted per utterance by ContextStyler; the
 * style used is broadcast to the app (ACTION_STYLE) so its sliders can show it.
 */
public class SteerTtsService extends TextToSpeechService {
    static final String ACTION_STYLE = "ai.whissle.steertts.demo.STYLE";
    // DEFAULT means "the voice chosen in the app", looked up on every request: apps such as TalkBack ask for the default
    // voice name once, when they connect, and send that name for as long as they run
    static final String FEMALE = "steertts-female", MALE = "steertts-male", DEFAULT = "steertts-default";
    private static final String TAG = "SteerTTS";

    private SteerTTS tts;   // used on the synthesis thread only (onSynthesizeText); cancel() from any thread

    @Override public void onDestroy() {
        SteerTTS e;
        synchronized (this) {
            e = tts;
            tts = null;
        }
        if (e != null) {   // cut the render short; close once it has returned (close waits for it), off the main thread
            e.cancel();
            new Thread(e::close, "steertts-close").start();
        }
        super.onDestroy();
    }

    @Override public void onCreate() {
        super.onCreate();
        Player.initTap(this);
    }

    /** One engine for the binder threads (voice queries) and the synthesis thread alike. */
    private synchronized SteerTTS engine() throws Exception {
        if (tts == null) tts = new SteerTTS(SteerTTS.installModel(this).getPath(), 0);
        return tts;
    }

    // ------------------------------------------------------------------ languages: English (US)
    @Override protected int onIsLanguageAvailable(String lang, String country, String variant) {
        if (!isEnglish(lang)) return TextToSpeech.LANG_NOT_SUPPORTED;
        return country == null || country.isEmpty() || country.equals("USA") || country.equals("US")
                ? TextToSpeech.LANG_COUNTRY_AVAILABLE : TextToSpeech.LANG_AVAILABLE;
    }

    @Override protected String[] onGetLanguage() { return new String[] {"eng", "USA", ""}; }

    @Override protected int onLoadLanguage(String lang, String country, String variant) {
        return onIsLanguageAvailable(lang, country, variant);
    }

    private static boolean isEnglish(String lang) { return lang != null && (lang.equals("eng") || lang.equals("en")); }

    // ------------------------------------------------------------------ voices
    @Override public List<Voice> onGetVoices() {
        List<Voice> out = new ArrayList<>();
        out.add(voice(DEFAULT, "default"));
        out.add(voice(FEMALE, "female"));
        out.add(voice(MALE, "male"));
        try {
            SteerTTS e = engine();
            for (String g : new String[] {"female", "male"})
                for (String v : e.voices(g)) out.add(voice(v, g));
        } catch (Exception ex) {
            Log.e(TAG, "voices", ex);
        }
        return out;
    }

    private static Voice voice(String name, String gender) {
        Set<String> features = new HashSet<>();
        features.add(gender);
        return new Voice(name, Locale.US, Voice.QUALITY_HIGH, Voice.LATENCY_LOW, false, features);
    }

    @Override public String onGetDefaultVoiceNameFor(String lang, String country, String variant) {
        return isEnglish(lang) ? DEFAULT : null;
    }

    @Override public int onIsValidVoiceName(String name) {
        if (DEFAULT.equals(name) || FEMALE.equals(name) || MALE.equals(name)) return TextToSpeech.SUCCESS;
        try {
            return engine().voiceGender(name) != null ? TextToSpeech.SUCCESS : TextToSpeech.ERROR;
        } catch (Exception e) {
            return TextToSpeech.ERROR;
        }
    }

    @Override public int onLoadVoice(String name) { return onIsValidVoiceName(name); }

    // ------------------------------------------------------------------ synthesis
    @Override protected void onStop() {
        if (tts != null) tts.cancel();
    }

    @Override protected void onSynthesizeText(SynthesisRequest req, SynthesisCallback cb) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
        String text = req.getCharSequenceText() != null ? req.getCharSequenceText().toString() : req.getText();
        if (text == null || text.trim().isEmpty()) {
            cb.start(SteerTTS.SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1);
            cb.done();
            return;
        }
        long t0 = System.nanoTime();
        try {
            SteerTTS e = engine();
            e.resetCancel();   // a stop that came in during the previous request must not silence this one
            String caller = callerPackage(req);
            Style base = Style.load(this);
            Style s = ContextStyler.decide(this, base, text, caller);
            String voice = resolveVoice(e, req.getVoiceName(), base.voice);
            // pitch 100 = normal; the system speech rate (100 = normal; screen readers go far higher) scales the user's speed
            float pace = Math.max(0.25f, Math.min(4f, s.pace * req.getSpeechRate() / 100f));
            float pitch = req.getPitch() / 100f;
            if (pitch > 0 && Math.abs(pitch - 1f) > 0.01f)   // semitones / the pooled sd of f0_med (2.09 st)
                s.add("f0_med", (float) (12 * Math.log(pitch) / Math.log(2) / 2.09));
            broadcast(s, voice, text, caller, true);
            if (cb.start(SteerTTS.SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1) != TextToSpeech.SUCCESS) return;
            double first = -1;
            long samples = 0;
            Player.initTap(this);   // the tap directory may have been created after this process started
            java.io.OutputStream tap = Player.newTap();   // demo recordings only (see Player.tapDir)
            Smooth filter = base.smooth ? new Smooth() : null;
            for (String sentence : sentences(Spoken.say(text))) {
                float[] y = e.synthesize(Expression.bmp(sentence), voice, s.tagOrNull(), s.ctrl(), pace, false);
                if (y == null) break;   // stopped
                if (filter != null) filter.process(y);
                if (first < 0) first = (System.nanoTime() - t0) / 1e6;
                samples += y.length;
                Player.writeTap(tap, y);
                if (!send(cb, y)) break;
            }
            Player.closeTap(tap);
            cb.done();
            Log.i(TAG, String.format(Locale.US, "service %s for %s: %.2f s audio, first audio %.0f ms, %d threads [%s]",
                    voice, caller, samples / (double) SteerTTS.SAMPLE_RATE, first, e.threads(), s.reason));
        } catch (Exception ex) {
            Log.e(TAG, "synthesis failed", ex);
            cb.error(TextToSpeech.ERROR_SYNTHESIS);
        } finally {
            broadcast(null, null, null, null, false);
        }
    }

    /**
     * The voice for a request: the app's chosen voice for DEFAULT (or no name); for a gender alias, the chosen voice if it
     * is of that gender, else that gender's default; otherwise the named voice.
     */
    static String resolveVoice(SteerTTS e, String requested, String saved) {
        String mine = saved == null || "female".equals(saved) ? e.genderVoice("female") : "male".equals(saved) ? e.genderVoice("male")
                : e.voiceGender(saved) != null ? saved : e.genderVoice("female");
        if (requested == null || DEFAULT.equals(requested)) return mine;
        if (FEMALE.equals(requested) || MALE.equals(requested)) {
            String g = FEMALE.equals(requested) ? "female" : "male";
            return g.equals(e.voiceGender(mine)) ? mine : e.genderVoice(g);
        }
        return e.voiceGender(requested) != null ? requested : mine;
    }

    private String callerPackage(SynthesisRequest req) {
        if (Build.VERSION.SDK_INT < 24) return null;
        String[] pk = getPackageManager().getPackagesForUid(req.getCallerUid());
        return pk != null && pk.length > 0 ? pk[0] : null;
    }

    /** 16-bit PCM in pieces of at most getMaxBufferSize(); false when the request was stopped. */
    private static boolean send(SynthesisCallback cb, float[] y) {
        int max = Math.max(2, cb.getMaxBufferSize() & ~1);
        byte[] buf = new byte[Math.min(max, y.length * 2)];
        for (int i = 0; i < y.length; ) {
            int k = 0;
            for (; k + 1 < buf.length && i < y.length; ++i) {
                float v = Math.max(-1f, Math.min(1f, y[i]));
                int q = Math.round(v * 32767f);
                buf[k++] = (byte) q;
                buf[k++] = (byte) (q >> 8);
            }
            if (cb.audioAvailable(buf, 0, k) != TextToSpeech.SUCCESS) return false;
        }
        return true;
    }

    /**
     * Text in the pieces it is rendered in: sentences (abbreviations, initials and decimals do not end one), long ones
     * split at a pause; the first piece is kept short (a clause) so the first sound comes quickly.
     */
    static List<String> sentences(String text) {
        List<String> out = new ArrayList<>();
        for (String para : text.split("\\n+")) for (int[] sp : TextClean.spans(para)) out.add(para.substring(sp[0], sp[1]));
        if (!out.isEmpty() && out.get(0).length() > FIRST) {
            String f = out.get(0);
            int cut = -1;
            for (String sep : new String[] {"; ", ", ", " — ", ": "}) {
                int at = f.indexOf(sep, 30);
                if (at > 0 && at <= FIRST && (cut < 0 || at < cut)) cut = at + 1;
            }
            if (cut > 0 && f.length() - cut > 20) {
                out.set(0, f.substring(0, cut).trim());
                out.add(1, f.substring(cut).trim());
            }
        }
        return out;
    }

    private static final int FIRST = 100;   // characters: about a second and a half of speech

    private void broadcast(Style s, String voice, String text, String caller, boolean speaking) {
        Intent i = new Intent(ACTION_STYLE).setPackage(getPackageName()).putExtra("speaking", speaking);
        if (s != null) {
            i.putExtra("style", s.toJson().toString()).putExtra("voice", voice).putExtra("text", text)
                    .putExtra("caller", caller);
        }
        sendBroadcast(i);
    }
}
