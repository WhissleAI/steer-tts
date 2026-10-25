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

import java.text.BreakIterator;
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
    static final String FEMALE = "steertts-female", MALE = "steertts-male";
    private static final String TAG = "SteerTTS";

    private SteerTTS tts;   // used on the synthesis thread only (onSynthesizeText); cancel() from any thread

    @Override public void onDestroy() {
        if (tts != null) tts.close();
        super.onDestroy();
    }

    @Override public void onCreate() {
        super.onCreate();
        Player.initTap(this);
    }

    private SteerTTS engine() throws Exception {
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
        if (!isEnglish(lang)) return null;
        Style base = Style.load(this);
        return "male".equals(base.voice) ? MALE : base.voice != null && base.voice.startsWith("voice_") ? base.voice
                : FEMALE;
    }

    @Override public int onIsValidVoiceName(String name) {
        if (FEMALE.equals(name) || MALE.equals(name)) return TextToSpeech.SUCCESS;
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
            for (String sentence : sentences(text)) {
                float[] y = e.synthesize(sentence, voice, s.ctrl(), pace);
                if (y == null) break;   // stopped
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

    private static String resolveVoice(SteerTTS e, String requested, String saved) {
        String v = requested != null ? requested : saved;
        if (v == null || FEMALE.equals(v) || "female".equals(v)) return e.genderVoice("female");
        if (MALE.equals(v) || "male".equals(v)) return e.genderVoice("male");
        return e.voiceGender(v) != null ? v : e.genderVoice("female");
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

    static List<String> sentences(String text) {
        List<String> out = new ArrayList<>();
        BreakIterator it = BreakIterator.getSentenceInstance(Locale.US);
        it.setText(text);
        for (int a = it.first(), b = it.next(); b != BreakIterator.DONE; a = b, b = it.next()) {
            String s = text.substring(a, b).trim();
            if (!s.isEmpty()) out.add(s);
        }
        return out;
    }

    private void broadcast(Style s, String voice, String text, String caller, boolean speaking) {
        Intent i = new Intent(ACTION_STYLE).setPackage(getPackageName()).putExtra("speaking", speaking);
        if (s != null) {
            i.putExtra("style", s.toJson().toString()).putExtra("voice", voice).putExtra("text", text)
                    .putExtra("caller", caller);
        }
        sendBroadcast(i);
    }
}
