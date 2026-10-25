package ai.whissle.steertts.demo;

import android.animation.ValueAnimator;
import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.Process;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Type text, pick a voice, shape the delivery with a slider per control, hear it: the edge model runs on the phone. The
 * settings here are also the system engine's base style (SteerTtsService); when SteerTTS speaks, for this app or for
 * another one, the sliders move to the style actually used and the reason is shown. Engine calls run on one background
 * thread. adb hooks (results in logcat, tag SteerTTS):
 *   adb shell am start -n ai.whissle.steertts.debug/ai.whissle.steertts.demo.MainActivity --es speak "Hello there." [--es voice male] [--ei threads N]
 *   adb shell am start -n ai.whissle.steertts.debug/ai.whissle.steertts.demo.MainActivity --es bench 0,1,4   (0 = Auto)
 *   adb shell am start -n ai.whissle.steertts.debug/ai.whissle.steertts.demo.MainActivity --es systts "Text"   (through android.speech.tts)
 * and for the assistant:
 *   --es announce "Text"            speak through the assistant's voice (Announcer)
 *   --ei readout_when 0|1|2         when messages are read (AssistantPrefs: headphones / + screen off / any time)
 *   --es readout_allow PACKAGE      read that app's messages too (e.g. com.android.shell for `cmd notification post`)
 *   --ei meetings_check MINUTES     announce now every event starting within MINUTES
 *   --ei demo_hour H                debug builds: context styling acts as if it were H o'clock (-1 = the real time)
 *   --es input "Text"               put text in the box;  --ez reset_card true   clear the now-speaking card
 *   --es say_as Happy               speak the box in a "Say it" delivery (Calm, Happy, Excited, Sad, Whisper)
 */
public class MainActivity extends Activity {
    static final String TAG = "SteerTTS";
    /** 0 = Auto: one thread per fast core, up to 6 (the fastest setting measured on a Pixel 10, see README). */
    static final int[] THREAD_CHOICES = {0, 1, 2, 3, 4, 6, 8};
    static final int STEPS = 60;   // slider steps over [-3, 3]

    private final HandlerThread worker = new HandlerThread("steertts-engine", Process.THREAD_PRIORITY_URGENT_AUDIO);
    private final Handler ui = new Handler(Looper.getMainLooper());
    private Handler engineThread;
    private SteerTTS tts;
    private int ttsWant = Integer.MIN_VALUE;
    private File modelDir;
    private Player player;
    private TextToSpeech systemTts;
    private AssistantPanel assistant;

    private Style base;
    private EditText input;
    private Spinner voicePick, tagPick, threadPick;
    private Switch contextual;
    private TextView status, styleNote;
    private final SeekBar[] bars = new SeekBar[SteerTTS.CONTROLS.length];
    private final TextView[] barValues = new TextView[SteerTTS.CONTROLS.length];
    private final List<String> voiceIds = new ArrayList<>();
    private String[] tags = {"none"};
    private boolean showingApplied;
    private volatile int threadsSel = 0;

    private final BroadcastReceiver styleReceiver = new BroadcastReceiver() {
        @Override public void onReceive(Context c, Intent i) {
            if (i.getBooleanExtra("speaking", false)) {
                try {
                    Style s = Style.fromJson(new JSONObject(i.getStringExtra("style")));
                    showApplied(s, who(i.getStringExtra("caller"), i.getStringExtra("source")), i.getStringExtra("text"));
                } catch (Exception ignored) {
                }
            } else {
                returnToBase(2500);
            }
        }
    };

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        base = Style.load(this);
        buildUi();
        showStyle(base, false);
        Player.initTap(this);
        player = new Player();
        IntentFilter f = new IntentFilter(SteerTtsService.ACTION_STYLE);
        if (Build.VERSION.SDK_INT >= 33) registerReceiver(styleReceiver, f, Context.RECEIVER_NOT_EXPORTED);
        else registerReceiver(styleReceiver, f);
        Announcer.screen = i -> styleReceiver.onReceive(this, i);   // the assistant, in this process: no queue
        worker.start();
        engineThread = new Handler(worker.getLooper());
        engineThread.post(() -> {
            try {
                long t0 = System.nanoTime();
                modelDir = SteerTTS.installModel(this);
                engine();
                say(String.format(Locale.US, "Ready · offline · loaded in %.0f ms", ms(t0)));
                String[] t = tts.tags();
                runOnUiThread(() -> {
                    tags = t;
                    fillVoicesAndTags();
                });
                handleIntent(getIntent());
            } catch (Exception e) {
                say("Load failed: " + e.getMessage());
                Log.e(TAG, "load", e);
            }
        });
    }

    @Override protected void onResume() {
        super.onResume();
        assistant.refresh();
    }

    @Override public void onRequestPermissionsResult(int req, String[] perms, int[] results) {
        assistant.onPermission(req, results.length > 0 && results[0] == android.content.pm.PackageManager.PERMISSION_GRANTED);
    }

    @Override protected void onNewIntent(Intent i) {
        super.onNewIntent(i);
        engineThread.post(() -> handleIntent(i));
    }

    @Override protected void onDestroy() {
        unregisterReceiver(styleReceiver);
        Announcer.screen = null;
        if (systemTts != null) systemTts.shutdown();
        if (tts != null) tts.cancel();
        engineThread.post(() -> {
            if (tts != null) tts.close();
            worker.quit();
        });
        player.release();
        super.onDestroy();
    }

    // ------------------------------------------------------------------ engine thread
    private SteerTTS engine() {
        if (tts == null || ttsWant != threadsSel) {
            if (tts != null) tts.close();
            tts = new SteerTTS(modelDir.getPath(), threadsSel);
            ttsWant = threadsSel;
        }
        return tts;
    }

    private void handleIntent(Intent i) {
        // the adb hooks below are for development and demo recordings only: in a store build any app could send this
        // (exported) activity an intent, so they are ignored unless the build is debuggable
        if (i == null || (getApplicationInfo().flags & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) == 0) return;
        if (i.hasExtra("threads")) threadsSel = i.getIntExtra("threads", 0);
        AssistantPrefs ap = new AssistantPrefs(this);
        if (i.hasExtra("demo_hour")) ContextStyler.setDemoHour(this, i.getIntExtra("demo_hour", -1));
        if (i.getStringExtra("input") != null) {
            String t = i.getStringExtra("input");
            runOnUiThread(() -> input.setText(t));
        }
        if (i.getBooleanExtra("reset_card", false)) runOnUiThread(this::resetCard);
        if (i.getStringExtra("say_as") != null) {
            String p = i.getStringExtra("say_as");
            runOnUiThread(() -> speakAs(p));
        }
        if (i.hasExtra("readout_when")) ap.setWhen(i.getIntExtra("readout_when", AssistantPrefs.PRIVATE));
        if (i.hasExtra("readout_allow")) {
            java.util.Set<String> on = ap.appsOn();
            on.add(i.getStringExtra("readout_allow"));
            ap.setAppsOn(on);
        }
        if (i.hasExtra("meetings_check") && MeetingAnnouncer.allowed(this)) ap.setAnnounceMeetings(true);
        if (i.hasExtra("meetings_check")) MeetingAnnouncer.check(this, i.getIntExtra("meetings_check", 60));
        if (i.getStringExtra("announce") != null)
            Announcer.get(this).say(i.getStringExtra("announce"), getPackageName(), true, null);
        String bench = i.getStringExtra("bench");
        String text = i.getStringExtra("speak");
        String sys = i.getStringExtra("systts");
        if (bench != null) benchmark(bench);
        if (text != null) {
            Style s = base.copy();
            if (i.hasExtra("voice")) s.voice = i.getStringExtra("voice");
            speak(text, s);
        }
        if (sys != null) runOnUiThread(() -> speakThroughSystem(sys));
    }

    private void speak(String text, Style baseStyle) {
        speakStyled(text, ContextStyler.decide(this, baseStyle, text, getPackageName()), null);
    }

    /** Speaks with exactly this style; label names a "Say it" delivery for the now-speaking card. */
    private void speakStyled(String text, Style s, String label) {
        SteerTTS e = engine();
        String voice = resolveVoice(e, s.voice);
        runOnUiThread(() -> showApplied(s, label != null ? "SteerTTS · " + label : "SteerTTS", text.replaceAll("<[^>]+>|\\[[^]]*]|\\*", "")));
        final long t0 = System.nanoTime();
        final double[] first = {-1};
        final long[] samples = {0};
        try {
            e.stream(text, voice, s.tagOrNull(), s.ctrl(), s.pace, (audio, index, synthMs, seg) -> {
                if (first[0] < 0) first[0] = ms(t0);
                samples[0] += audio.length;
                player.write(audio);
                return true;
            });
            player.end();
            double total = ms(t0), audioS = samples[0] / (double) SteerTTS.SAMPLE_RATE;
            String r = String.format(Locale.US,
                    "%s: %.2f s of audio, first sound after %.0f ms, rendered in %.0f ms (%.1fx real time), %d threads",
                    voice, audioS, first[0], total, audioS * 1000 / total, e.threads());
            Log.i(TAG, "speak " + r + " [" + s.reason + "]");
            say(String.format(Locale.US, "First sound in %.0f ms · rendered %.0f× faster than real time", first[0],
                    audioS * 1000 / total));
        } catch (RuntimeException ex) {
            say("Error: " + ex.getMessage());
        }
        runOnUiThread(() -> returnToBase(2500));
    }

    private static String resolveVoice(SteerTTS e, String v) {
        if (v == null || v.equals("female")) return e.genderVoice("female");
        if (v.equals("male")) return e.genderVoice("male");
        return v;
    }

    private static final String[] BENCH = {
            "The weather was calm when we left the harbour this morning.",
            "Please remember to bring the documents to the meeting on Thursday.",
            "I never thought a small garden could grow so many tomatoes.",
            "Thanks for calling. How can I help you today?",
            "If you turn left at the bakery, you will see the station.",
            "Could you send me the report before lunch?",
            "The engineers tested the new design for three weeks before approving it.",
            "When the lights came back on, everyone started clapping."};

    /** For each thread count: real-time factor over BENCH, and time to the first audio of a two-sentence reply. */
    private void benchmark(String spec) {
        StringBuilder out = new StringBuilder("Benchmark (render only, no playback):\n");
        int keep = threadsSel;
        for (String s : spec.split(",")) {
            threadsSel = Integer.parseInt(s.trim());
            long tl = System.nanoTime();
            engine();
            double load = ms(tl);
            for (int w = 0; w < 2; ++w) render(BENCH[w]);   // warm-up
            double synth = 0, audio = 0;
            for (String t : BENCH) {
                long t0 = System.nanoTime();
                audio += render(t);
                synth += ms(t0);
            }
            double[] firsts = new double[5];
            for (int k = 0; k < firsts.length; ++k) {
                double[] f = {-1};
                long t0 = System.nanoTime();
                tts.stream("Hello, how are you? And a second sentence follows.", null, null, null, 1f, (a, i, m, x) -> {
                    f[0] = ms(t0);
                    return false;
                });
                firsts[k] = f[0];
            }
            Arrays.sort(firsts);
            String line = String.format(Locale.US,
                    "threads %d: RTF %.4f (%.1fx real time), first audio %.0f ms, load %.0f ms",
                    tts.threads(), synth / 1000 / audio, audio * 1000 / synth, firsts[2], load);
            Log.i(TAG, "bench " + line);
            out.append(line).append('\n');
            say(out.toString());
        }
        threadsSel = keep;
        engine();
    }

    private double render(String text) {
        final long[] n = {0};
        tts.stream(text, null, null, null, 1f, (a, i, m, x) -> {
            n[0] += a.length;
            return true;
        });
        return n[0] / (double) SteerTTS.SAMPLE_RATE;
    }

    private static double ms(long t0) { return (System.nanoTime() - t0) / 1e6; }

    // ------------------------------------------------------------------ the system engine, as other apps use it
    private void speakThroughSystem(String text) {
        long t0 = System.nanoTime();
        Runnable go = () -> {
            systemTts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                double started = -1;
                @Override public void onStart(String id) { }
                @Override public void onBeginSynthesis(String id, int sr, int fmt, int ch) { }
                @Override public void onAudioAvailable(String id, byte[] audio) {
                    if (started < 0) started = ms(t0);
                }
                @Override public void onDone(String id) {
                    Log.i(TAG, String.format(Locale.US, "systts done: first audio %.0f ms after the request, %.0f ms total",
                            started, ms(t0)));
                }
                @Override public void onError(String id) { Log.e(TAG, "systts error"); }
            });
            int rc = systemTts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "u" + System.nanoTime());
            Log.i(TAG, "systts speak rc " + rc + ", engine " + systemTts.getDefaultEngine());
        };
        if (systemTts != null) go.run();
        else systemTts = new TextToSpeech(this, st -> {
            Log.i(TAG, "systts init " + st);
            if (st == TextToSpeech.SUCCESS) go.run();
        }, getPackageName());   // this engine, whatever the system default is
    }

    // ------------------------------------------------------------------ UI
    // Two pages: home (try it, what is being said now, the assistant's switches; fits one screen) and voice & style
    // (voice, style tag, the 12 controls, threads and benchmark; scrolls). Back returns home.

    private static final String[][] SAMPLES = {
            {"Hello", "Hi! I'm SteerTTS, a voice that runs entirely on your phone. No internet needed."},
            {"Whisper", "Can I tell you a secret? <whisper>I'm running on this phone, offline.</whisper>"},
            {"Emphasis", "This is *really* important. [pause 300ms] Please read it twice."},
            {"Question", "Are you free for coffee at four?"}};

    private FrameLayout pages;
    private View home, studio;
    private TextView nowLabel, nowText, nowWhy, femaleChip, maleChip;
    private Ui u;

    private void say(String s) { runOnUiThread(() -> status.setText(s)); }

    /** Shows a style used for speech: on home what and why, on voice & style the sliders; returnToBase() undoes. */
    private void showApplied(Style s, String who, String text) {
        ui.removeCallbacksAndMessages(null);
        showingApplied = true;
        nowLabel.setText("Now speaking · " + who);
        nowLabel.setTextColor(Ui.ACCENT);
        if (text != null) nowText.setText("“" + text.trim() + "”");
        nowWhy.setText(s.reason);
        nowWhy.setVisibility(View.VISIBLE);
        styleNote.setText("Speaking: " + who + "\n" + s.reason);
        showStyle(s, true);
    }

    private void resetCard() {
        ui.removeCallbacksAndMessages(null);
        showingApplied = false;
        nowLabel.setText("Idle");
        nowLabel.setTextColor(Ui.MUTED);
        nowText.setText("Messages, callers, meetings and other apps' speech appear here as SteerTTS says them.");
        nowWhy.setVisibility(View.GONE);
        showStyle(base, false);
    }

    private void returnToBase(long delayMs) {
        ui.postDelayed(() -> {
            showingApplied = false;
            if (nowLabel.getText().toString().startsWith("Now speaking"))
                nowLabel.setText(nowLabel.getText().toString().replace("Now speaking", "Last spoken"));
            nowLabel.setTextColor(Ui.MUTED);
            styleNote.setText("Your settings: everything SteerTTS says starts from these");
            showStyle(base, true);
        }, delayMs);
    }

    /** "WhatsApp", "Meeting reminder", "Maps (system voice)" for a broadcast's caller and source. */
    private String who(String caller, String source) {
        if ("calendar".equals(caller)) return "Meeting reminder";
        if ("read-aloud".equals(caller)) return "Read aloud";
        if (caller == null || caller.equals(getPackageName())) return "SteerTTS";
        String name = caller;
        try {
            name = getPackageManager().getApplicationLabel(getPackageManager().getApplicationInfo(caller, 0)).toString();
        } catch (Exception ignored) {
        }
        return "assistant".equals(source) ? name : name + " (system voice)";
    }

    private void showStyle(Style s, boolean animate) {
        for (int i = 0; i < bars.length; ++i) {
            int to = Math.round((s.z[i] + Style.LIMIT) / (2 * Style.LIMIT) * STEPS);
            SeekBar bar = bars[i];
            if (!animate) bar.setProgress(to);
            else {
                ValueAnimator a = ValueAnimator.ofInt(bar.getProgress(), to).setDuration(450);
                a.addUpdateListener(v -> bar.setProgress((int) v.getAnimatedValue()));
                a.start();
            }
        }
    }

    private static float zOf(int progress) { return progress * 2 * Style.LIMIT / STEPS - Style.LIMIT; }

    private void saveBase() {
        base.save(this);
        showVoiceChips();
    }

    private void showVoiceChips() {
        u.select(femaleChip, base.voice == null || "female".equals(base.voice));
        u.select(maleChip, "male".equals(base.voice));
    }

    private void fillVoicesAndTags() {
        List<String> labels = new ArrayList<>();
        voiceIds.clear();
        String f = tts.genderVoice("female"), m = tts.genderVoice("male");
        labels.add("Female (" + f + ")");
        voiceIds.add("female");
        labels.add("Male (" + m + ")");
        voiceIds.add("male");
        for (String g : new String[] {"female", "male"})
            for (String v : tts.voices(g)) {
                labels.add(v + "  " + g);
                voiceIds.add(v);
            }
        voicePick.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, labels));
        int vi = voiceIds.indexOf(base.voice == null ? "female" : base.voice);
        voicePick.setSelection(Math.max(vi, 0));
        tagPick.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, tags));
        tagPick.setSelection(Math.max(Arrays.asList(tags).indexOf(base.tag), 0));
    }

    private void buildUi() {
        u = new Ui(this);
        home = buildHome();
        studio = buildStudio();
        pages = new FrameLayout(this);
        pages.setBackgroundColor(Ui.BG);
        pages.addView(home);
        pages.addView(studio);
        studio.setVisibility(View.GONE);
        setContentView(pages);
        // edge to edge (Android 15+ draws apps under the status and navigation bars): keep content clear of them
        if (Build.VERSION.SDK_INT >= 30)
            pages.setOnApplyWindowInsetsListener((v, insets) -> {
                android.graphics.Insets b = insets.getInsets(android.view.WindowInsets.Type.systemBars()
                        | android.view.WindowInsets.Type.displayCutout());
                v.setPadding(b.left, b.top, b.right, b.bottom);
                return android.view.WindowInsets.CONSUMED;
            });
        showVoiceChips();
    }

    // Back from Voice & style returns home. Android 16 (target 36) no longer calls onBackPressed: the predictive back
    // gesture goes to a callback, registered only while the second page is open so back on home leaves the app normally.
    private Object backToHome;

    private void showPage(View page) {
        if (Build.VERSION.SDK_INT >= 33) {
            android.window.OnBackInvokedDispatcher d = getOnBackInvokedDispatcher();
            if (page == studio && backToHome == null) {
                android.window.OnBackInvokedCallback cb = () -> showPage(home);
                d.registerOnBackInvokedCallback(android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT, cb);
                backToHome = cb;
            } else if (page != studio && backToHome != null) {
                d.unregisterOnBackInvokedCallback((android.window.OnBackInvokedCallback) backToHome);
                backToHome = null;
            }
        }
        home.setVisibility(page == home ? View.VISIBLE : View.GONE);
        studio.setVisibility(page == studio ? View.VISIBLE : View.GONE);
    }

    @Override public void onBackPressed() {
        if (studio.getVisibility() == View.VISIBLE) showPage(home);
        else super.onBackPressed();
    }

    private void speakInput() {
        String text = input.getText().toString();
        Style s = base.copy();
        say("Speaking…");
        engineThread.post(() -> speak(text, s));
    }

    /** The input text once, in one of the "Say it" deliveries (the saved style is not changed). */
    private void speakAs(String preset) {
        String text = input.getText().toString();
        Style s = base.preset(preset);
        say("Speaking…");
        engineThread.post(() -> speakStyled(text, s, preset));
    }

    private void stopAll() {
        if (tts != null) tts.cancel();
        player.flush();
        Announcer.get(this).stop();   // readouts and read aloud too
        say("Stopped.");
    }

    private View buildHome() {
        LinearLayout root = u.column();
        root.setPadding(u.dp(16), u.dp(20), u.dp(16), u.dp(16));

        LinearLayout head = u.row();
        LinearLayout titles = u.column();
        titles.addView(u.text("SteerTTS", 28, Ui.INK, true));
        titles.addView(u.text("Expressive voice, fully on Android", 14, Ui.MUTED, false));
        head.addView(titles, Ui.weight(1));
        TextView studioBtn = u.pill("Voice & style", false, v -> showPage(studio));
        head.addView(studioBtn);
        root.addView(head);

        // try it
        LinearLayout tryCard = u.card();
        tryCard.addView(u.label("Try it"));
        input = new EditText(this);
        input.setText(SAMPLES[0][1]);
        input.setTextSize(16);
        input.setTextColor(Ui.INK);
        input.setMinLines(2);
        input.setMaxLines(3);
        input.setGravity(Gravity.TOP);
        input.setBackground(Ui.round(Ui.BG, u.dp(14)));
        input.setPadding(u.dp(12), u.dp(10), u.dp(12), u.dp(10));
        tryCard.addView(input, Ui.matchWrap());

        // two rows of equal-width chips that always fit the card (no scrolling, nothing cut off)
        LinearLayout samples = u.row();
        samples.setPadding(0, u.dp(10), 0, u.dp(4));
        for (String[] smp : SAMPLES)
            samples.addView(fit(u.chip(smp[0], v -> {
                input.setText(smp[1]);
                speakInput();
            })));
        tryCard.addView(samples, Ui.matchWrap());
        TextView sayIt = u.text("SAY IT", 11, Ui.MUTED, true);
        sayIt.setLetterSpacing(0.08f);
        sayIt.setPadding(0, u.dp(6), 0, u.dp(4));
        tryCard.addView(sayIt);
        LinearLayout styles = u.row();
        styles.setPadding(0, 0, 0, u.dp(4));
        for (String p : Style.PRESETS)
            styles.addView(fit(u.chip(p, v -> speakAs(p))));
        tryCard.addView(styles, Ui.matchWrap());

        LinearLayout actions = u.row();
        actions.setPadding(0, u.dp(8), 0, 0);
        femaleChip = u.chip("Female", v -> {
            base.voice = "female";
            saveBase();
            fillVoiceSelection();
        });
        maleChip = u.chip("Male", v -> {
            base.voice = "male";
            saveBase();
            fillVoiceSelection();
        });
        actions.addView(femaleChip);
        actions.addView(maleChip);
        actions.addView(new View(this), Ui.weight(1));
        TextView stop = u.pill("Stop", false, v -> stopAll());
        LinearLayout.LayoutParams sp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        sp.rightMargin = u.dp(8);
        actions.addView(stop, sp);
        actions.addView(u.pill("Speak", true, v -> speakInput()));
        tryCard.addView(actions);
        status = u.text("Loading the voice…", 12.5f, Ui.MUTED, false);
        status.setPadding(0, u.dp(8), 0, 0);
        tryCard.addView(status);
        root.addView(tryCard);

        // now speaking
        LinearLayout now = u.card();
        nowLabel = u.text("Idle", 11.5f, Ui.MUTED, true);
        nowLabel.setLetterSpacing(0.08f);
        nowLabel.setAllCaps(true);
        now.addView(nowLabel);
        nowText = u.text("Messages, callers, meetings and other apps' speech appear here as SteerTTS says them.",
                15, Ui.INK, false);
        nowText.setMaxLines(3);
        nowText.setEllipsize(TextUtils.TruncateAt.END);
        nowText.setPadding(0, u.dp(4), 0, 0);
        now.addView(nowText);
        nowWhy = u.text("", 13, Ui.ACCENT, true);
        nowWhy.setPadding(0, u.dp(6), 0, 0);
        nowWhy.setVisibility(View.GONE);
        now.addView(nowWhy);
        root.addView(now);

        // assistant
        assistant = new AssistantPanel(this, u);
        root.addView(assistant.build());

        LinearLayout bottom = u.row();
        bottom.setPadding(0, u.dp(12), 0, 0);
        bottom.addView(u.pill("Try a readout", false, v -> Announcer.get(this).say(
                "Priya on WhatsApp: Running ten minutes late. Can you order me a coffee?", "com.whatsapp", true, null)),
                Ui.weight(1));
        View gap = new View(this);
        bottom.addView(gap, new LinearLayout.LayoutParams(u.dp(8), 1));
        bottom.addView(u.pill("Phone's TTS settings", false,
                v -> startActivity(new Intent("com.android.settings.TTS_SETTINGS"))), Ui.weight(1));
        root.addView(bottom);

        ScrollView sv = new ScrollView(this);   // fits a phone screen; scrolls only on small ones
        sv.addView(root);
        return sv;
    }

    /** A chip that shares its row's width equally with the others. */
    private TextView fit(TextView chip) {
        LinearLayout.LayoutParams lp = Ui.weight(1);
        lp.rightMargin = u.dp(6);
        chip.setLayoutParams(lp);
        chip.setPadding(u.dp(4), u.dp(7), u.dp(4), u.dp(7));
        chip.setSingleLine(true);
        return chip;
    }

    private void fillVoiceSelection() {
        int vi = voiceIds.indexOf(base.voice == null ? "female" : base.voice);
        if (vi >= 0 && voicePick.getAdapter() != null) voicePick.setSelection(vi);
    }

    private View buildStudio() {
        LinearLayout root = u.column();
        root.setPadding(u.dp(16), u.dp(20), u.dp(16), u.dp(24));

        LinearLayout head = u.row();
        TextView back = u.text("‹", 30, Ui.INK, false);
        back.setPadding(0, 0, u.dp(14), u.dp(4));
        back.setOnClickListener(v -> showPage(home));
        head.addView(back);
        head.addView(u.text("Voice & style", 24, Ui.INK, true));
        root.addView(head);

        LinearLayout voiceCard = u.card();
        voiceCard.addView(u.label("Voice"));
        voicePick = new Spinner(this);
        voicePick.setOnItemSelectedListener(listener(pos -> {
            if (pos < voiceIds.size()) base.voice = voiceIds.get(pos);
            saveBase();
        }));
        voiceCard.addView(labeled("Voice", voicePick));
        tagPick = new Spinner(this);
        tagPick.setOnItemSelectedListener(listener(pos -> {
            base.tag = tags[pos];
            saveBase();
        }));
        voiceCard.addView(labeled("Style", tagPick));
        SeekBar speed = new SeekBar(this);
        TextView speedVal = u.text("", 13, Ui.MUTED, false);
        speedVal.setMinWidth(u.dp(40));
        speedVal.setGravity(Gravity.END);
        int speedSteps = Math.round((Style.MAX_PACE - Style.MIN_PACE) * 20);
        speed.setMax(speedSteps);
        speed.setProgress(Math.round((base.pace - Style.MIN_PACE) * 20));
        speedVal.setText(String.format(Locale.US, "%.2f×", base.pace));
        speed.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar s, int p, boolean fromUser) {
                base.pace = Style.MIN_PACE + p / 20f;
                speedVal.setText(String.format(Locale.US, "%.2f×", base.pace));
            }
            @Override public void onStartTrackingTouch(SeekBar s) { }
            @Override public void onStopTrackingTouch(SeekBar s) { saveBase(); }
        });
        LinearLayout speedRow = labeled("Speed", speed);
        speedRow.addView(speedVal);
        voiceCard.addView(speedRow);
        contextual = u.toggle(base.contextual);
        contextual.setOnCheckedChangeListener((v, on) -> {
            base.contextual = on;
            saveBase();
        });
        voiceCard.addView(u.settingRow("Adapt to context", "Alerts, questions, night, ringer, battery", contextual,
                null));
        root.addView(voiceCard);

        LinearLayout ctl = u.card();
        ctl.addView(u.label("Delivery"));
        styleNote = u.text("Your settings: everything SteerTTS says starts from these", 13, Ui.MUTED, false);
        styleNote.setPadding(0, 0, 0, u.dp(6));
        ctl.addView(styleNote);
        for (int i = 0; i < bars.length; ++i) {
            final int k = i;
            LinearLayout l = u.row();
            TextView name = u.text(SteerTTS.CONTROLS[i], 13.5f, Ui.INK, false);
            name.setMinWidth(u.dp(64));
            TextView value = u.text("", 13, Ui.MUTED, false);
            value.setMinWidth(u.dp(40));
            value.setGravity(Gravity.END);
            barValues[i] = value;
            SeekBar bar = new SeekBar(this);
            bar.setMax(STEPS);
            bar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override public void onProgressChanged(SeekBar s, int p, boolean fromUser) {
                    float z = zOf(p);
                    barValues[k].setText(String.format(Locale.US, "%+.1f", z));
                    if (fromUser) {
                        if (showingApplied) returnToBase(0);
                        base.z[k] = Math.abs(z) < 0.05f ? 0 : z;
                    }
                }
                @Override public void onStartTrackingTouch(SeekBar s) { }
                @Override public void onStopTrackingTouch(SeekBar s) { saveBase(); }
            });
            bars[i] = bar;
            l.addView(name);
            l.addView(bar, Ui.weight(1));
            l.addView(value);
            ctl.addView(l);
        }
        LinearLayout resetRow = u.row();
        resetRow.setPadding(0, u.dp(8), 0, 0);
        resetRow.addView(u.pill("Reset", false, v -> {
            Arrays.fill(base.z, 0f);
            saveBase();
            showStyle(base, true);
        }));
        resetRow.addView(new View(this), Ui.weight(1));
        resetRow.addView(u.pill("Speak", true, v -> speakInput()));
        ctl.addView(resetRow);
        root.addView(ctl);

        LinearLayout perf = u.card();
        perf.addView(u.label("Performance"));
        threadPick = new Spinner(this);
        List<String> tl = new ArrayList<>();
        for (int t : THREAD_CHOICES) tl.add(t == 0 ? "Auto" : String.valueOf(t));
        threadPick.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, tl));
        threadPick.setOnItemSelectedListener(listener(pos -> threadsSel = THREAD_CHOICES[pos]));
        perf.addView(labeled("Threads", threadPick));
        perf.addView(u.pill("Run benchmark", false, v -> {
            showPage(home);
            say("Benchmarking…");
            engineThread.post(() -> benchmark("0,1,2,4,6"));
        }));
        root.addView(perf);

        ScrollView sv = new ScrollView(this);
        sv.addView(root);
        return sv;
    }

    private interface OnPick { void picked(int pos); }

    private static AdapterView.OnItemSelectedListener listener(OnPick p) {
        return new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> a, View v, int pos, long id) { p.picked(pos); }
            @Override public void onNothingSelected(AdapterView<?> a) { }
        };
    }

    private LinearLayout labeled(String label, View v) {
        LinearLayout l = u.row();
        TextView t = u.text(label, 14, Ui.MUTED, false);
        t.setMinWidth(u.dp(64));
        l.addView(t);
        l.addView(v, Ui.weight(1));
        return l;
    }
}
