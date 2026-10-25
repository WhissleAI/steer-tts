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
 * The assistant's voice (message readouts, caller and meeting announcements, the briefing), in the app's main process:
 * its own engine on its own thread. What it says is a Speech: parts, each rendered with the user's style adjusted for
 * the situation (ContextStyler) and then the part's mood and voice (Expression), a sentence at a time and played through Player while other audio ducks (transient audio focus). Requests queue; an interrupting one
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
        say(new Speech().add(text, Expression.of(text)), caller, interrupt, done);
    }

    /** Speaks parts that may each have their own mood and voice (a readout's framing, then the message). */
    void say(Speech sp, String caller, boolean interrupt, Runnable done) {
        if (interrupt) stop();
        int g = gen.get();
        Runnable d = done != null ? done : () -> { };
        if (sp.isEmpty()) {
            d.run();
            return;
        }
        synchronized (waiting) {
            if (waiting.isEmpty()) am.requestAudioFocus(focus);
            waiting.add(d);
        }
        String text = sp.text();
        Log.i(TAG, "announce [" + caller + "] " + text.substring(0, Math.min(text.length(), 160)) + describe(sp));
        engineThread.post(() -> render(sp, caller, g, d));
    }

    /** " {sad: apologetic; voice_0010}" per part with a mood or a voice of its own (the log; tests read it). */
    private static String describe(Speech sp) {
        StringBuilder b = new StringBuilder();
        for (Speech.Part p : sp.parts)
            if (!p.mood.neutral() || p.voice != null)
                b.append(b.length() > 0 ? ", " : "").append(p.mood.name).append(p.mood.tag != null ? ":" + p.mood.tag : "")
                        .append(p.voice != null ? " " + p.voice : "");
        return b.length() > 0 ? " {" + b + "}" : "";
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

    private void render(Speech sp, String caller, int g, Runnable done) {
        engineThread.removeCallbacks(closeIdle);
        try {
            if (tts == null) tts = new SteerTTS(SteerTTS.installModel(app).getPath(), 0);
            Style base = ContextStyler.decide(app, Style.load(app), sp.text(), caller, false);
            if (gen.get() == g) show(base, sp, caller);
            for (Speech.Part part : sp.parts) {
                Style s = moodStyle(base, part.mood);
                String voice = voice(tts, part.voice != null ? part.voice : s.voice);
                for (String sentence : SteerTtsService.sentences(Expression.bmp(part.text))) {
                    while (player.pendingSamples() > AHEAD && gen.get() == g) SystemClock.sleep(50);
                    if (gen.get() != g) break;
                    float[] y = tts.synthesize(sentence, voice, s.tagOrNull(), s.ctrl(), s.pace, false);
                    if (y == null || gen.get() != g) break;
                    player.write(y);
                }
                if (part.pauseAfterMs > 0 && gen.get() == g) player.write(new float[SteerTTS.SAMPLE_RATE * part.pauseAfterMs / 1000]);
            }
        } catch (Exception ex) {
            Log.e(TAG, "announce failed", ex);
        }
        if (gen.get() == g) player.end(() -> finished(done));
        else finished(done);
        engineThread.postDelayed(closeIdle, IDLE_MS);
    }

    /**
     * The situation's style with a part's mood on top: the mood's tag replaces the user's (except a whisper, which is
     * the user's choice of voice quality) and its control shifts add to the user's.
     */
    static Style moodStyle(Style base, Expression.Mood m) {
        Style s = base.copy();
        if (m.neutral()) return s;
        if (m.tag != null && !"whisper".equals(base.tag)) s.tag = m.tag;
        for (java.util.Map.Entry<String, Float> e : m.ctrl.entrySet()) s.add(e.getKey(), e.getValue());
        s.reason = base.reason.startsWith("neutral") || base.reason.isEmpty() ? m.why : m.why + "; " + base.reason;
        return s;
    }

    /** Tells the app's screen what is being said and in what style (SteerTtsService does the same). */
    /**
     * The app's screen, when it is open (same process): told directly, on the main thread. A broadcast can wait
     * seconds in the queue while the system is busy with the very notification being read out.
     */
    static volatile java.util.function.Consumer<Intent> screen;
    private static final android.os.Handler MAIN = new android.os.Handler(android.os.Looper.getMainLooper());

    private void tell(Intent i) {
        java.util.function.Consumer<Intent> s = screen;
        if (s != null) MAIN.post(() -> s.accept(i));
        else app.sendBroadcast(i);
    }

    private void show(Style base, Speech sp, String caller) {
        Style s = moodStyle(base, sp.mainMood());
        tell(new Intent(SteerTtsService.ACTION_STYLE).setPackage(app.getPackageName())
                .putExtra("speaking", true).putExtra("source", "assistant").putExtra("style", s.toJson().toString())
                .putExtra("text", sp.text()).putExtra("caller", caller));
    }

    private void finished(Runnable done) {
        synchronized (waiting) {
            if (!waiting.remove(done)) return;   // already run by stop()
            if (waiting.isEmpty()) {
                am.abandonAudioFocusRequest(focus);
                tell(new Intent(SteerTtsService.ACTION_STYLE).setPackage(app.getPackageName())
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
