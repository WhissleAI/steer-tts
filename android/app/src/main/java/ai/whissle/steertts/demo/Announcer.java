package ai.whissle.steertts.demo;

import android.content.Context;
import android.content.Intent;
import android.media.AudioAttributes;
import android.media.AudioFocusRequest;
import android.media.AudioManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Process;
import android.os.SystemClock;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The assistant's voice (message readouts, caller and meeting announcements, read aloud), in the app's main process:
 * its own engine on its own thread, the user's style adjusted by ContextStyler, plain text rendered a sentence at a
 * time and played through Player while other audio ducks (transient audio focus). Requests queue; an interrupting one
 * (an incoming call, a new read-aloud) drops what is queued or playing. The renderer stays at most a few seconds ahead
 * of playback, so a long article is never all in memory, and the engine (~30 MB) is closed after two idle minutes.
 */
final class Announcer {
    private static final String TAG = "SteerTTS";
    private static final long IDLE_MS = 120_000, AHEAD = 8L * SteerTTS.SAMPLE_RATE;
    private static Announcer instance;

    static synchronized Announcer get(Context c) {
        if (instance == null) instance = new Announcer(c.getApplicationContext());
        return instance;
    }

    private final Context app;
    private final Handler engineThread;
    private final Player player = new Player();
    private final AudioManager am;
    private final AudioFocusRequest focus;
    private volatile SteerTTS tts;                            // engine thread
    private final AtomicInteger gen = new AtomicInteger();   // bumped by stop(): older requests are dropped
    private final List<Runnable> waiting = new ArrayList<>(); // done callbacks of requests not finished (guarded)
    private final Runnable closeIdle = () -> {
        if (tts != null) tts.close();
        tts = null;
    };

    private Announcer(Context app) {
        this.app = app;
        Player.initTap(app);
        HandlerThread t = new HandlerThread("steertts-announcer", Process.THREAD_PRIORITY_URGENT_AUDIO);
        t.start();
        engineThread = new Handler(t.getLooper());
        am = (AudioManager) app.getSystemService(Context.AUDIO_SERVICE);
        focus = new AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                .setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANT)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setOnAudioFocusChangeListener(change -> { })
                .build();
    }

    /**
     * Speaks text (plain, no markup). caller: the package the text is about, for ContextStyler. done (may be null)
     * runs when it has been heard out, or was dropped.
     */
    void say(String text, String caller, boolean interrupt, Runnable done) {
        if (interrupt) stop();
        int g = gen.get();
        Runnable d = done != null ? done : () -> { };
        synchronized (waiting) {
            if (waiting.isEmpty()) am.requestAudioFocus(focus);
            waiting.add(d);
        }
        Log.i(TAG, "announce [" + caller + "] " + text.substring(0, Math.min(text.length(), 120)));
        engineThread.post(() -> render(text, caller, g, d));
    }

    /** Drops everything queued or playing; their done callbacks run now. */
    void stop() {
        gen.incrementAndGet();
        player.flush();
        List<Runnable> dropped;
        synchronized (waiting) {
            dropped = new ArrayList<>(waiting);
            waiting.clear();
            am.abandonAudioFocusRequest(focus);
        }
        for (Runnable r : dropped) r.run();
    }

    private void render(String text, String caller, int g, Runnable done) {
        engineThread.removeCallbacks(closeIdle);
        try {
            if (tts == null) tts = new SteerTTS(SteerTTS.installModel(app).getPath(), 0);
            Style s = ContextStyler.decide(app, Style.load(app), text, caller);
            String voice = voice(tts, s.voice);
            if (gen.get() == g) show(s, voice, text, caller);
            for (String sentence : SteerTtsService.sentences(text)) {
                while (player.pendingSamples() > AHEAD && gen.get() == g) SystemClock.sleep(50);
                if (gen.get() != g) break;
                float[] y = tts.synthesize(sentence, voice, s.ctrl(), 1f);
                if (y == null || gen.get() != g) break;
                player.write(y);
            }
        } catch (Exception ex) {
            Log.e(TAG, "announce failed", ex);
        }
        if (gen.get() == g) player.end(() -> finished(done));
        else finished(done);
        engineThread.postDelayed(closeIdle, IDLE_MS);
    }

    /** Tells the app's screen what is being said and in what style (SteerTtsService does the same). */
    private void show(Style s, String voice, String text, String caller) {
        app.sendBroadcast(new Intent(SteerTtsService.ACTION_STYLE).setPackage(app.getPackageName())
                .putExtra("speaking", true).putExtra("source", "assistant").putExtra("style", s.toJson().toString())
                .putExtra("voice", voice).putExtra("text", text).putExtra("caller", caller));
    }

    private void finished(Runnable done) {
        synchronized (waiting) {
            if (!waiting.remove(done)) return;   // already run by stop()
            if (waiting.isEmpty()) {
                am.abandonAudioFocusRequest(focus);
                app.sendBroadcast(new Intent(SteerTtsService.ACTION_STYLE).setPackage(app.getPackageName())
                        .putExtra("speaking", false));
            }
        }
        done.run();
    }

    private static String voice(SteerTTS e, String v) {
        if (v == null || v.equals("female")) return e.genderVoice("female");
        if (v.equals("male")) return e.genderVoice("male");
        return e.voiceGender(v) != null ? v : e.genderVoice("female");
    }
}
