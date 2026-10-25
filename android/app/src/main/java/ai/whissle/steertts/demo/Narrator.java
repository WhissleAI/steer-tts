package ai.whissle.steertts.demo;

import android.content.Context;
import android.content.SharedPreferences;
import android.media.AudioAttributes;
import android.media.AudioFocusRequest;
import android.media.AudioManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Process;
import android.os.SystemClock;
import android.util.Log;

import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Reads a Doc aloud, sentence by sentence, with the controls a listener expects: play, pause, seek to a sentence,
 * skip back and forward, speed. Sentences are rendered a few seconds ahead of playback on the narrator's own thread
 * and engine, and a marker before each one tells the screen which sentence is being heard. Holds media audio focus:
 * a readout or a call (anything that takes focus) pauses it, and it resumes when focus comes back. Position is saved
 * as it goes, so a document resumes where it was left. One per process.
 */
final class Narrator {
    interface Listener {
        void changed();    // playing / position / document changed (main thread)
    }

    private static final String TAG = "SteerTTS";
    private static final long AHEAD = 6L * SteerTTS.SAMPLE_RATE;
    private static Narrator instance;

    static synchronized Narrator get(Context c) {
        if (instance == null) instance = new Narrator(c.getApplicationContext());
        return instance;
    }

    private final Context app;
    private final Handler engineThread;
    private final Player player = new Player(AudioAttributes.USAGE_MEDIA);
    private final AudioManager am;
    private final AudioFocusRequest focus;
    private final SharedPreferences prefs;
    private final CopyOnWriteArrayList<Listener> listeners = new CopyOnWriteArrayList<>();
    private final AtomicInteger gen = new AtomicInteger();
    private SteerTTS tts;   // engine thread
    private Doc doc;
    private volatile boolean playing;
    private boolean resumeOnFocus;
    private int heard;      // the sentence being heard (main thread)

    private Narrator(Context app) {
        this.app = app;
        prefs = app.getSharedPreferences("reader", Context.MODE_PRIVATE);
        HandlerThread t = new HandlerThread("steertts-narrator", Process.THREAD_PRIORITY_URGENT_AUDIO);
        t.start();
        engineThread = new Handler(t.getLooper());
        am = (AudioManager) app.getSystemService(Context.AUDIO_SERVICE);
        focus = new AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
                .setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setWillPauseWhenDucked(true)
                .setOnAudioFocusChangeListener(this::onFocus)
                .build();
    }

    // ---------------------------------------------------------------- state
    Doc doc() { return doc; }
    boolean playing() { return playing; }
    int position() { return doc == null ? 0 : heard; }
    float pace() { return prefs.getFloat("pace", 1.0f); }

    void addListener(Listener l) { listeners.add(l); }
    void removeListener(Listener l) { listeners.remove(l); }

    private void changed() { for (Listener l : listeners) l.changed(); }

    // ---------------------------------------------------------------- controls (main thread)
    void open(Doc d) {
        if (doc != null && doc.id.equals(d.id)) return;
        stop(false);
        doc = d;
        heard = d.pos;
        d.opened = System.currentTimeMillis();
        d.savePos(app);
        changed();
    }

    void play() {
        if (doc == null || doc.sentences.isEmpty()) return;
        if (heard >= doc.sentences.size()) heard = 0;
        if (am.requestAudioFocus(focus) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) return;
        playing = true;
        renderFrom(heard);
        ReaderService.update(app);
        changed();
    }

    void pause() {
        if (!playing) return;
        stop(true);
        changed();
    }

    void toggle() {
        if (playing) pause();
        else play();
    }

    /** Jump to a sentence (keeps playing if it was). */
    void seek(int i) {
        if (doc == null) return;
        heard = Math.max(0, Math.min(i, doc.sentences.size() - 1));
        doc.pos = heard;
        doc.savePos(app);
        if (playing) {
            player.flush();
            renderFrom(heard);
        }
        changed();
    }

    void next() { seek(heard + 1); }
    void prev() { seek(heard - 1); }

    void setPace(float p) {
        prefs.edit().putFloat("pace", p).apply();
        if (playing) seek(heard);   // re-render from the current sentence at the new speed
        else changed();
    }

    /** Stop reading; with keepFocus false also forget audio focus and the notification. */
    private void stop(boolean keepNotification) {
        gen.incrementAndGet();
        playing = false;
        player.flush();
        am.abandonAudioFocusRequest(focus);
        if (doc != null) {
            doc.pos = heard;
            doc.savePos(app);
        }
        ReaderService.update(app);
    }

    void close() {
        stop(false);
        doc = null;
        ReaderService.stop(app);
        changed();
    }

    private void onFocus(int change) {
        if (change == AudioManager.AUDIOFOCUS_GAIN) {
            if (resumeOnFocus && !playing) play();
            resumeOnFocus = false;
        } else if (change == AudioManager.AUDIOFOCUS_LOSS) {
            resumeOnFocus = false;
            pause();
        } else {   // transient loss (a readout, a call, a navigation prompt): pause, resume after
            resumeOnFocus = playing;
            pause();
        }
    }

    // ---------------------------------------------------------------- rendering (engine thread)
    private void renderFrom(int start) {
        int g = gen.incrementAndGet();
        Doc d = doc;
        Style base = Style.load(app).copy();
        base.contextual = false;          // a book is read in one voice, not restyled per sentence
        float pace = pace() * base.pace;
        engineThread.post(() -> {
            try {
                if (tts == null) tts = new SteerTTS(SteerTTS.installModel(app).getPath(), 0);
                String voice = voiceFor(tts, base.voice);
                for (int i = start; i < d.sentences.size(); ++i) {
                    while (player.pendingSamples() > AHEAD && gen.get() == g) SystemClock.sleep(40);
                    if (gen.get() != g) return;
                    float[] y = tts.synthesize(d.sentences.get(i), voice, base.ctrl(), pace);
                    if (y == null || gen.get() != g) return;
                    final int idx = i;
                    player.mark(() -> heardSentence(g, idx));
                    player.write(y);
                }
                if (gen.get() == g) player.mark(() -> finished(g));
            } catch (Exception e) {
                Log.e(TAG, "narrator", e);
            }
        });
    }

    private void heardSentence(int g, int i) {
        if (gen.get() != g || doc == null) return;
        heard = i;
        doc.pos = i;
        if (i % 5 == 0) doc.savePos(app);
        changed();
    }

    private void finished(int g) {
        if (gen.get() != g || doc == null) return;
        // the last sentence's audio is still playing out: stop once it has
        new Handler(android.os.Looper.getMainLooper()).postDelayed(() -> {
            if (gen.get() != g || doc == null) return;
            playing = false;
            heard = doc.sentences.size();
            doc.pos = 0;            // finished: next time starts from the top
            doc.savePos(app);
            am.abandonAudioFocusRequest(focus);
            ReaderService.update(app);
            changed();
        }, 300);
    }

    private static String voiceFor(SteerTTS e, String v) {
        if (v == null || v.equals("female")) return e.genderVoice("female");
        if (v.equals("male")) return e.genderVoice("male");
        return e.voiceGender(v) != null ? v : e.genderVoice("female");
    }
}
