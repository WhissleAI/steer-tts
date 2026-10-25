package ai.whissle.steertts.demo;

import android.animation.ValueAnimator;
import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.Process;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
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
 *   adb shell am start -n ai.whissle.steertts.demo/.MainActivity --es speak "Hello there." [--es voice male] [--ei threads N]
 *   adb shell am start -n ai.whissle.steertts.demo/.MainActivity --es bench 0,1,4   (0 = Auto)
 *   adb shell am start -n ai.whissle.steertts.demo/.MainActivity --es systts "Text"   (through android.speech.tts)
 * and for the assistant:
 *   --es announce "Text"            speak through the assistant's voice (Announcer)
 *   --ei readout_when 0|1|2         when messages are read (AssistantPrefs: headphones / + screen off / any time)
 *   --es readout_allow PACKAGE      read that app's messages too (e.g. com.android.shell for `cmd notification post`)
 *   --ei meetings_check MINUTES     announce now every event starting within MINUTES
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
                    String who = i.getStringExtra("caller");
                    showApplied(s, "System TTS" + (who != null ? " for " + who : "") + ", " + i.getStringExtra("voice"));
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
        player = new Player();
        IntentFilter f = new IntentFilter(SteerTtsService.ACTION_STYLE);
        if (Build.VERSION.SDK_INT >= 33) registerReceiver(styleReceiver, f, Context.RECEIVER_NOT_EXPORTED);
        else registerReceiver(styleReceiver, f);
        worker.start();
        engineThread = new Handler(worker.getLooper());
        engineThread.post(() -> {
            try {
                long t0 = System.nanoTime();
                modelDir = SteerTTS.installModel(this);
                engine();
                say(String.format(Locale.US, "Ready: model loaded in %.0f ms, %d threads.", ms(t0), tts.threads()));
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
        if (i == null) return;
        if (i.hasExtra("threads")) threadsSel = i.getIntExtra("threads", 0);
        AssistantPrefs ap = new AssistantPrefs(this);
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
        SteerTTS e = engine();
        Style s = ContextStyler.decide(this, baseStyle, text, getPackageName());
        String voice = resolveVoice(e, s.voice);
        runOnUiThread(() -> showApplied(s, "This app, " + voice));
        final long t0 = System.nanoTime();
        final double[] first = {-1};
        final long[] samples = {0};
        try {
            e.stream(text, voice, s.tagOrNull(), s.ctrl(), 1.0f, (audio, index, synthMs, seg) -> {
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
            say(r);
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
    private void say(String s) { runOnUiThread(() -> status.setText(s)); }

    /** Moves the sliders to a style used for speech and says why; returnToBase() moves them back. */
    private void showApplied(Style s, String who) {
        ui.removeCallbacksAndMessages(null);
        showingApplied = true;
        styleNote.setText("Speaking: " + who + "\n" + s.reason);
        showStyle(s, true);
    }

    private void returnToBase(long delayMs) {
        ui.postDelayed(() -> {
            showingApplied = false;
            styleNote.setText("Your settings (drag to change; system speech starts from these)");
            showStyle(base, true);
        }, delayMs);
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
        int pad = dp(16);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad * 2, pad, pad);

        TextView title = new TextView(this);
        title.setText("SteerTTS on device");
        title.setTextSize(22);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(title);
        assistant = new AssistantPanel(this);
        root.addView(assistant.build());

        TextView voiceTitle = new TextView(this);
        voiceTitle.setText("Voice");
        voiceTitle.setTextSize(18);
        voiceTitle.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(voiceTitle);

        input = new EditText(this);
        input.setMinLines(2);
        input.setGravity(Gravity.TOP);
        input.setText("Hello there. <whisper>This part is quiet.</whisper> [pause 300ms] And *this* is not!");
        root.addView(input, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout row = new LinearLayout(this);
        Button speak = new Button(this), stop = new Button(this), bench = new Button(this);
        speak.setText("Speak");
        stop.setText("Stop");
        bench.setText("Bench");
        speak.setOnClickListener(v -> {
            String text = input.getText().toString();
            Style s = base.copy();
            say("Speaking...");
            engineThread.post(() -> speak(text, s));
        });
        stop.setOnClickListener(v -> {
            if (tts != null) tts.cancel();
            player.flush();
            Announcer.get(this).stop();   // readouts and read aloud too
            say("Stopped.");
        });
        bench.setOnClickListener(v -> {
            say("Benchmarking...");
            engineThread.post(() -> benchmark("0,1,2,4,6"));
        });
        for (Button x : new Button[] {speak, stop, bench})
            row.addView(x, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        root.addView(row);

        status = new TextView(this);
        status.setText("Loading the model...");
        status.setPadding(0, dp(4), 0, dp(8));
        root.addView(status);

        voicePick = new Spinner(this);
        voicePick.setOnItemSelectedListener(listener(pos -> {
            if (pos < voiceIds.size()) base.voice = voiceIds.get(pos);
            saveBase();
        }));
        root.addView(labeled("Voice", voicePick));
        tagPick = new Spinner(this);
        tagPick.setOnItemSelectedListener(listener(pos -> {
            base.tag = tags[pos];
            saveBase();
        }));
        root.addView(labeled("Style", tagPick));

        contextual = new Switch(this);
        contextual.setText("Context styling (alerts, questions, night, ringer, battery)");
        contextual.setChecked(base.contextual);
        contextual.setOnCheckedChangeListener((v, on) -> {
            base.contextual = on;
            saveBase();
        });
        contextual.setPadding(0, dp(8), 0, dp(8));
        root.addView(contextual);

        styleNote = new TextView(this);
        styleNote.setText("Your settings (drag to change; system speech starts from these)");
        styleNote.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(styleNote);

        for (int i = 0; i < bars.length; ++i) {
            final int k = i;
            LinearLayout l = new LinearLayout(this);
            l.setGravity(Gravity.CENTER_VERTICAL);
            TextView name = new TextView(this);
            name.setText(SteerTTS.CONTROLS[i]);
            name.setMinWidth(dp(72));
            TextView value = new TextView(this);
            value.setMinWidth(dp(44));
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
            l.addView(bar, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
            l.addView(value);
            root.addView(l);
        }

        LinearLayout row2 = new LinearLayout(this);
        Button reset = new Button(this), sys = new Button(this);
        reset.setText("Reset controls");
        reset.setOnClickListener(v -> {
            Arrays.fill(base.z, 0f);
            saveBase();
            showStyle(base, true);
        });
        sys.setText("System TTS settings");
        sys.setOnClickListener(v -> startActivity(new Intent("com.android.settings.TTS_SETTINGS")));
        row2.addView(reset, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        row2.addView(sys, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        root.addView(row2);

        threadPick = new Spinner(this);
        List<String> tl = new ArrayList<>();
        for (int t : THREAD_CHOICES) tl.add(t == 0 ? "Auto" : String.valueOf(t));
        threadPick.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, tl));
        threadPick.setOnItemSelectedListener(listener(pos -> threadsSel = THREAD_CHOICES[pos]));
        root.addView(labeled("Threads", threadPick));

        ScrollView sv = new ScrollView(this);
        sv.addView(root);
        setContentView(sv);
    }

    private interface OnPick { void picked(int pos); }

    private static AdapterView.OnItemSelectedListener listener(OnPick p) {
        return new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> a, View v, int pos, long id) { p.picked(pos); }
            @Override public void onNothingSelected(AdapterView<?> a) { }
        };
    }

    private LinearLayout labeled(String label, View v) {
        LinearLayout l = new LinearLayout(this);
        l.setGravity(Gravity.CENTER_VERTICAL);
        TextView t = new TextView(this);
        t.setText(label);
        t.setMinWidth(dp(72));
        l.addView(t);
        l.addView(v, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        return l;
    }

    private int dp(int x) { return Math.round(x * getResources().getDisplayMetrics().density); }
}
