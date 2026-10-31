package ai.whissle.steertts.demo;

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
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The app: three tabs. Listen (what is being said and played, things to listen to, the queue), Assistant (what
 * SteerTTS does for you) and Voices (try it, voice, style, delivery); Settings behind the gear. The first start shows
 * OnboardingActivity. Previews on the Voices tab run on this activity's engine thread. Debug builds only, adb hooks
 * (results in logcat, tag SteerTTS), on ai.whissle.steertts.debug/ai.whissle.steertts.demo.MainActivity, started with
 * -f 0x24000000 (clear top, single top) so they arrive even when the player is open above it (tools/device_test.sh):
 *   --es speak "Hello there." [--es voice male]      --es announce "Text"   (the assistant's voice)
 *   --es readout "Sam|WhatsApp|message"              a readout as if that message arrived (mood, person voice)
 *   --es call "Mom|3"                                a call announcement, the 3rd ring from Mom
 *   --es digest_add "Sam|WhatsApp|message"           a message waiting for catch-up;  --ez catchup true   catch up now
 *   --ez sample true                                 the Assistant tab's sample;      --ez briefing true   the briefing
 *   --es pdf_layout FILE   (in files/) log the paragraphs a PDF is read as
 *   --ez auto_next false (the reader stops at the end)   --ez messages_on true (readouts and calls on)   --ei readout_when 0|1|2   --es readout_allow PACKAGE   --ei meetings_check MINUTES   --ei demo_hour H
 *   --es tab listen|assistant|voices   --ez skip_welcome true   --ez settings true
 */
public class MainActivity extends Activity implements VoicesScreen.Host {
    static final String TAG = "SteerTTS";
    private static final String[] TABS = {"listen", "assistant", "voices"};
    private static final String[] TAB_NAMES = {"Listen", "Assistant", "Voices"};
    private static final int[] TAB_ICONS = {R.drawable.ic_listen, R.drawable.ic_assistant, R.drawable.ic_voices};

    private final HandlerThread worker = new HandlerThread("steertts-engine", Process.THREAD_PRIORITY_URGENT_AUDIO);
    private final Handler ui = new Handler(Looper.getMainLooper());
    private Handler engineThread;
    private SteerTTS tts;
    private File modelDir;
    private Player player;

    private Ui u;
    private Style base;
    private FrameLayout content;
    private final Map<String, View> tabViews = new LinkedHashMap<>();
    private final Map<String, LinearLayout> tabButtons = new LinkedHashMap<>();
    private String tab = "listen";
    private Object backToFirst;

    private ListenScreen listen;
    private AssistantScreen assistant;
    private VoicesScreen voices;
    private TextView status;

    private final BroadcastReceiver styleReceiver = new BroadcastReceiver() {
        @Override public void onReceive(Context c, Intent i) {
            if (listen == null) return;
            if (i.getBooleanExtra("speaking", false)) {
                ui.removeCallbacksAndMessages(null);
                String why = "";
                try {
                    why = Style.fromJson(new JSONObject(i.getStringExtra("style"))).reason;
                } catch (Exception ignored) {
                }
                String text = i.getStringExtra("text");
                listen.speaking(who(i.getStringExtra("caller"), i.getStringExtra("source")),
                        text == null ? "" : "“" + text.replaceAll("<[^>]+>|\\[[^]]*]|\\*", "").trim() + "”",
                        why.startsWith("neutral") || why.startsWith("your settings") ? "" : why);
            } else {
                ui.postDelayed(() -> listen.speaking(null, null, null), 1500);
            }
        }
    };

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        Ui.init(this);
        u = new Ui(this);
        base = Style.load(this);
        if (b == null && !OnboardingActivity.done(this) && !getIntent().getBooleanExtra("skip_welcome", false))
            startActivity(new Intent(this, OnboardingActivity.class));
        build();
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
                tts = new SteerTTS(modelDir.getPath(), 0);
                say(String.format(Locale.US, "Ready · offline · loaded in %.0f ms", ms(t0)));
            } catch (Exception e) {
                say("Couldn't load the voice: " + e.getMessage());
                Log.e(TAG, "load", e);
            }
        });
        String t = getIntent().getStringExtra("tab");
        selectTab(t != null ? t : "listen");
        if (b == null && getIntent().getBooleanExtra("setup_screen", false) && assistant != null) assistant.setupScreen();
        handleIntent(getIntent());
    }

    private boolean started;

    /** The Listen tab follows the narrator only while the app is on screen (it would rebuild every sentence otherwise). */
    @Override protected void onStart() {
        super.onStart();
        started = true;
        if (listen != null) Narrator.get(this).addListener(listen);
    }

    @Override protected void onStop() {
        started = false;
        if (listen != null) Narrator.get(this).removeListener(listen);
        super.onStop();
    }

    @Override protected void onResume() {
        super.onResume();
        base = Style.load(this);
        Smooth.on = base.smooth;
        if (listen != null) listen.refresh();
        if (assistant != null) assistant.refresh();
        if (voices != null) voices.refresh();
    }

    @Override public void onRequestPermissionsResult(int req, String[] perms, int[] results) {
        boolean granted = false;
        for (int r : results) granted |= r == android.content.pm.PackageManager.PERMISSION_GRANTED;
        if (assistant != null) assistant.onPermission(req, granted);
    }

    @Override protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req == ListenScreen.PICK && res == RESULT_OK && data != null) {
            selectTab("listen");
            listen.picked(data.getData());
        }
    }

    @Override protected void onNewIntent(Intent i) {
        super.onNewIntent(i);
        setIntent(i);
        if (i.getStringExtra("tab") != null) selectTab(i.getStringExtra("tab"));
        if (i.getBooleanExtra("setup_screen", false) && assistant != null) assistant.setupScreen();
        handleIntent(i);
    }

    @Override protected void onDestroy() {
        unregisterReceiver(styleReceiver);
        Announcer.screen = null;
        if (listen != null) Narrator.get(this).removeListener(listen);
        if (tts != null) tts.cancel();
        engineThread.post(() -> {
            if (tts != null) tts.close();
            worker.quit();
        });
        player.release();
        super.onDestroy();
    }

    // ---------------------------------------------------------------- tabs
    private void build() {
        LinearLayout root = u.column();
        root.setBackgroundColor(Ui.BG);
        content = new FrameLayout(this);
        root.addView(content, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        LinearLayout bar = u.row();
        bar.setBackgroundColor(Ui.CARD);
        bar.setPadding(u.dp(12), u.dp(8), u.dp(12), u.dp(8));
        for (int i = 0; i < TABS.length; ++i) {
            String t = TABS[i];
            LinearLayout b = u.column();
            b.setGravity(Gravity.CENTER);
            b.setMinimumHeight(u.dp(56));
            LinearLayout pillBox = u.row();
            pillBox.setGravity(Gravity.CENTER);
            pillBox.setPadding(u.dp(18), u.dp(4), u.dp(18), u.dp(4));
            pillBox.addView(u.img(TAB_ICONS[i], Ui.MUTED, 24));
            b.addView(pillBox);
            TextView label = u.text(TAB_NAMES[i], 12.5f, Ui.MUTED, true);
            label.setPadding(0, u.dp(4), 0, 0);
            label.setGravity(android.view.Gravity.CENTER_HORIZONTAL);
            b.addView(label);
            b.setOnClickListener(v -> selectTab(t));
            b.setContentDescription(TAB_NAMES[i]);
            b.setFocusable(true);
            bar.addView(b, Ui.weight(1));
            tabButtons.put(t, b);
        }
        root.addView(bar);
        setContentView(root);
        if (Build.VERSION.SDK_INT >= 30)
            root.setOnApplyWindowInsetsListener((v, insets) -> {
                android.graphics.Insets ins = insets.getInsets(android.view.WindowInsets.Type.systemBars()
                        | android.view.WindowInsets.Type.displayCutout());
                android.graphics.Insets ime = insets.getInsets(android.view.WindowInsets.Type.ime());
                v.setPadding(ins.left, ins.top, ins.right, Math.max(ins.bottom, ime.bottom));
                bar.setVisibility(ime.bottom > 0 ? View.GONE : View.VISIBLE);
                return android.view.WindowInsets.CONSUMED;
            });
    }

    void selectTab(String t) {
        if ("home".equals(t) || "read".equals(t)) t = "listen";   // the old tab names
        if (!Arrays.asList(TABS).contains(t)) t = "listen";
        tab = t;
        View v = tabViews.get(t);
        if (v == null) {
            v = buildTab(t);
            tabViews.put(t, v);
            content.addView(v);
        }
        for (Map.Entry<String, View> e : tabViews.entrySet()) e.getValue().setVisibility(e.getKey().equals(t) ? View.VISIBLE : View.GONE);
        for (Map.Entry<String, LinearLayout> e : tabButtons.entrySet()) {
            boolean on = e.getKey().equals(t);
            LinearLayout pillBox = (LinearLayout) e.getValue().getChildAt(0);
            pillBox.setBackground(on ? Ui.round(Ui.ACCENT_SOFT, u.dp(16)) : null);
            ((android.widget.ImageView) pillBox.getChildAt(0)).setImageTintList(android.content.res.ColorStateList.valueOf(on ? Ui.ACCENT : Ui.MUTED));
            ((TextView) e.getValue().getChildAt(1)).setTextColor(on ? Ui.INK : Ui.MUTED);
            e.getValue().setSelected(on);
        }
        // back from another tab returns to Listen (Android 16 delivers back to this callback, not onBackPressed)
        if (Build.VERSION.SDK_INT >= 33) {
            android.window.OnBackInvokedDispatcher d = getOnBackInvokedDispatcher();
            if (!t.equals("listen") && backToFirst == null) {
                android.window.OnBackInvokedCallback cb = () -> selectTab("listen");
                d.registerOnBackInvokedCallback(android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT, cb);
                backToFirst = cb;
            } else if (t.equals("listen") && backToFirst != null) {
                d.unregisterOnBackInvokedCallback((android.window.OnBackInvokedCallback) backToFirst);
                backToFirst = null;
            }
        }
        if (t.equals("listen")) listen.refresh();
        if (t.equals("assistant")) assistant.refresh();
    }

    @SuppressWarnings("deprecation")
    @Override public void onBackPressed() {
        if (!tab.equals("listen")) selectTab("listen");
        else super.onBackPressed();
    }

    private View buildTab(String t) {
        View.OnClickListener gear = v -> openSettings();
        switch (t) {
            case "assistant":
                assistant = new AssistantScreen(this, u, gear);
                return assistant.build();
            case "voices":
                voices = new VoicesScreen(this, u, this);
                return voices.build();
            default:
                listen = new ListenScreen(this, u, gear);
                View lv = listen.build();
                if (started) Narrator.get(this).addListener(listen);
                return lv;
        }
    }

    /** "WhatsApp", "Meeting reminder", "Maps · system voice" for a broadcast's caller and source. */
    private String who(String caller, String source) {
        if ("calendar".equals(caller)) return "Meeting reminder";
        if ("read-aloud".equals(caller)) return "Read aloud";
        if ("briefing".equals(caller)) return "Morning briefing";
        if ("catch-up".equals(caller)) return "Catching you up";
        if ("sample".equals(caller)) return "Sample";
        if ("onboarding".equals(caller) || caller == null || caller.equals(getPackageName())) return "SteerTTS";
        String name = caller;
        try {
            name = getPackageManager().getApplicationLabel(getPackageManager().getApplicationInfo(caller, 0)).toString();
        } catch (Exception ignored) {
        }
        return "assistant".equals(source) ? name : name + " · system voice";
    }

    // ---------------------------------------------------------------- VoicesScreen.Host
    @Override public Style base() { return base; }
    @Override public void saveBase() { base.save(this); }
    @Override public void status(TextView t) { status = t; }
    @Override public void openSettings() { startActivity(new Intent(this, SettingsActivity.class)); }

    @Override public void preview(String text, Style s) {
        stop();
        engineThread.post(() -> speakStyled(text, s));
    }

    @Override public void stop() {
        if (tts != null) tts.cancel();
        player.flush();
        engineThread.removeCallbacksAndMessages(null);
    }

    private void say(String s) { runOnUiThread(() -> { if (status != null) status.setText(s); }); }

    // ---------------------------------------------------------------- engine thread
    /** Speaks with exactly this style (inline markup allowed). */
    private void speakStyled(String text, Style s) {
        if (tts == null) return;
        String voice = resolveVoice(tts, s.voice);
        final long t0 = System.nanoTime();
        final double[] first = {-1};
        final long[] samples = {0};
        try {
            tts.stream(Expression.bmp(text), voice, s.tagOrNull(), s.ctrl(), s.pace, (audio, index, synthMs, seg) -> {
                if (first[0] < 0) first[0] = ms(t0);
                samples[0] += audio.length;
                player.write(audio);
                return true;
            });
            player.end();
            double total = ms(t0), audioS = samples[0] / (double) SteerTTS.SAMPLE_RATE;
            Log.i(TAG, String.format(Locale.US, "speak %s: %.2f s of audio, first sound after %.0f ms, rendered in %.0f ms, %d threads [%s]",
                    voice, audioS, first[0], total, tts.threads(), s.reason));
            if (first[0] >= 0) say(String.format(Locale.US, "First sound in %.0f ms · offline", first[0]));
        } catch (RuntimeException ex) {
            Log.e(TAG, "speak", ex);
            say("Couldn't speak that one. Try a shorter text.");
        }
    }

    private static String resolveVoice(SteerTTS e, String v) {
        if (v == null || v.equals("female")) return e.genderVoice("female");
        if (v.equals("male")) return e.genderVoice("male");
        return e.voiceGender(v) != null ? v : e.genderVoice("female");
    }

    private static double ms(long t0) { return (System.nanoTime() - t0) / 1e6; }

    // ---------------------------------------------------------------- debug hooks
    private void handleIntent(Intent i) {
        // the adb hooks below are for development and tests only: in a store build any app could send this (exported)
        // activity an intent, so they are ignored unless the build is debuggable
        if (i == null || i.getExtras() == null || (getApplicationInfo().flags & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) == 0) return;
        Log.i(TAG, "hook " + i.getExtras().keySet());
        AssistantPrefs ap = new AssistantPrefs(this);
        if (i.hasExtra("demo_hour")) ContextStyler.setDemoHour(this, i.getIntExtra("demo_hour", -1));
        if (i.hasExtra("readout_when")) ap.setWhen(i.getIntExtra("readout_when", AssistantPrefs.PRIVATE));
        if (i.hasExtra("readout_allow")) {
            java.util.Set<String> on = ap.appsOn();
            on.add(i.getStringExtra("readout_allow"));
            ap.setAppsOn(on);
        }
        if (i.hasExtra("auto_next")) Narrator.get(this).setAutoNext(i.getBooleanExtra("auto_next", true));
        if (i.getBooleanExtra("messages_on", false)) {
            ap.setReadMessages(true);
            ap.setAnnounceCalls(true);
        }
        if (i.hasExtra("meetings_check")) MeetingAnnouncer.check(this, i.getIntExtra("meetings_check", 60));
        if (i.getBooleanExtra("settings", false)) openSettings();
        if (i.getStringExtra("announce") != null) Announcer.get(this).say(i.getStringExtra("announce"), getPackageName(), true, null);
        if (i.getStringExtra("readout") != null) {
            String[] p = i.getStringExtra("readout").split("\\|", 3);
            List<String> texts = new ArrayList<>();
            texts.add(p[2]);
            Announcer.get(this).say(Readout.message(p[0], null, p[1], texts, ap.personVoices(), Style.load(this).voice), "readout-test", false, null);
        }
        if (i.getStringExtra("call") != null) {
            String[] p = i.getStringExtra("call").split("\\|");
            Announcer.get(this).say(Readout.call(p[0], null, p.length > 1 ? Integer.parseInt(p[1]) : 1), "call-test", true, null);
        }
        if (i.getStringExtra("digest_add") != null) {
            String[] p = i.getStringExtra("digest_add").split("\\|", 3);
            Digest.add("test:" + System.nanoTime(), p[0], p[1], p[2]);
        }
        if (i.getStringExtra("pdf_layout") != null && Build.VERSION.SDK_INT >= 35) {   // what PdfText + PdfLayout read
            try {
                PdfText.Result r = PdfText.read(new File(getFilesDir(), i.getStringExtra("pdf_layout")));
                for (String p : r.paragraphs) Log.i(TAG, "pdf para [" + p.replace(String.valueOf(Narration.HEAD), "# ") + "]");
            } catch (Exception e) {
                Log.e(TAG, "pdf_layout", e);
            }
        }
        if (i.hasExtra("screen_read_in") && ScreenReader.instance != null) ScreenReader.instance.readIn(i.getIntExtra("screen_read_in", 4000));
        if (i.getBooleanExtra("perform", false) && Narrator.get(this).doc() != null) {   // how the current book is performed
            Doc d = Narrator.get(this).doc();
            Cast c = d.cast();
            for (int k = 0; k < Math.min(d.sentences.size(), 400); ++k) {
                Cast.Line l = c.lines[d.paraOf[k]];
                for (Narration.Seg sg : Narration.segments(d.sentences.get(k), l))
                    if (sg.dialogue) Log.i(TAG, "perform " + (l.speaker == null ? "-" : l.speaker) + " [" + sg.mood.name + "] " + sg.text.substring(0, Math.min(60, sg.text.length())));
            }
        }
        if (i.getBooleanExtra("catchup", false)) ReadoutListener.catchUp(this, true);
        if (i.getBooleanExtra("sample", false)) AssistantScreen.sample(this);
        if (i.getBooleanExtra("briefing", false)) Briefing.play(this);
        String text = i.getStringExtra("speak");
        if (text != null) {
            Style s = base.copy();
            if (i.hasExtra("voice")) s.voice = i.getStringExtra("voice");
            Style st = ContextStyler.decide(this, s, text, getPackageName());
            engineThread.post(() -> speakStyled(text, st));
        }
    }
}
