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
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.text.TextUtils;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.File;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * The app: five tabs. Home (try the voice, what is being said now, status), Read (documents, books, pages), Voices
 * (voice, style, speed, delivery), Assistant (readouts, calls, meetings, the daily briefing) and Settings. The first
 * start shows OnboardingActivity. Engine calls for this screen run on one background thread. Debug builds only, adb
 * hooks (results in logcat, tag SteerTTS), on ai.whissle.steertts.debug/ai.whissle.steertts.demo.MainActivity:
 *   --es speak "Hello there." [--es voice male] [--ei threads N]     --es bench 0,1,4 (0 = Auto)     --es systts "Text"
 *   --es announce "Text"            speak through the assistant's voice (Announcer)
 *   --ei readout_when 0|1|2         when messages are read (AssistantPrefs: headphones / + screen off / any time)
 *   --es readout_allow PACKAGE      read that app's messages too (e.g. com.android.shell for `cmd notification post`)
 *   --ei meetings_check MINUTES     announce now every event starting within MINUTES
 *   --ei demo_hour H                context styling acts as if it were H o'clock (-1 = the real time)
 *   --es input "Text"               put text in the box;  --ez reset_card true   clear the now-speaking card
 *   --es say_as Happy               speak the box in a "Say it" delivery (Happy, Whisper, Excited, Sad, Calm)
 *   --es tab read|voices|assistant|settings|home     --ez skip_welcome true     --ez briefing true   play the briefing
 */
public class MainActivity extends Activity implements VoicesScreen.Host, SettingsScreen.Host {
    static final String TAG = "SteerTTS";
    private static final String[] TABS = {"home", "read", "voices", "assistant", "settings"};
    private static final String[] TAB_NAMES = {"Home", "Read", "Voices", "Assistant", "Settings"};
    private static final String[][] SAMPLES = {
            {"Hello", "Hi! I'm SteerTTS, a voice that runs entirely on your phone. No internet needed."},
            {"Whisper", "Can I tell you a secret? <whisper>I'm running on this phone, offline.</whisper>"},
            {"Emphasis", "This is *really* important. [pause 300ms] Please read it twice."},
            {"Question", "Are you free for coffee at four?"}};

    private final HandlerThread worker = new HandlerThread("steertts-engine", Process.THREAD_PRIORITY_URGENT_AUDIO);
    private final Handler ui = new Handler(Looper.getMainLooper());
    private Handler engineThread;
    private SteerTTS tts;
    private int ttsWant = Integer.MIN_VALUE;
    private volatile int threadsSel = 0;
    private File modelDir;
    private Player player;
    private TextToSpeech systemTts;

    private Ui u;
    private Style base;
    private FrameLayout content;
    private final Map<String, View> tabViews = new LinkedHashMap<>();
    private final Map<String, TextView> tabButtons = new LinkedHashMap<>();
    private String tab = "home";
    private Object backToHome;

    private AssistantPanel assistant;
    private LibraryScreen library;
    private VoicesScreen voices;
    private SettingsScreen settings;

    private EditText input;
    private TextView status, nowLabel, nowText, nowWhy, voiceChip;
    private LinearLayout phoneVoiceRow, assistantRow;

    private final BroadcastReceiver styleReceiver = new BroadcastReceiver() {
        @Override public void onReceive(Context c, Intent i) {
            if (i.getBooleanExtra("speaking", false)) {
                try {
                    Style s = Style.fromJson(new JSONObject(i.getStringExtra("style")));
                    showApplied(s, who(i.getStringExtra("caller"), i.getStringExtra("source")), i.getStringExtra("text"));
                } catch (Exception ignored) {
                }
            } else {
                settle(2500);
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
                engine();
                say(String.format(Locale.US, "Ready · offline · loaded in %.0f ms", ms(t0)));
                handleIntent(getIntent());
            } catch (Exception e) {
                say("Couldn't load the voice: " + e.getMessage());
                Log.e(TAG, "load", e);
            }
        });
        String t = getIntent().getStringExtra("tab");
        selectTab(t != null ? t : "home");
    }

    @Override protected void onResume() {
        super.onResume();
        base = Style.load(this);
        refreshHome();
        if (assistant != null) assistant.refresh();
        if (library != null) library.refresh();
        if (voices != null) voices.refresh();
        if (settings != null) settings.refresh();
    }

    @Override public void onRequestPermissionsResult(int req, String[] perms, int[] results) {
        boolean granted = false;
        for (int r : results) granted |= r == android.content.pm.PackageManager.PERMISSION_GRANTED;
        if (assistant != null) assistant.onPermission(req, granted);
    }

    @Override protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req == LibraryScreen.PICK && res == RESULT_OK && data != null) {
            if (library == null) selectTab("read");
            library.picked(data.getData());
        }
    }

    @Override protected void onNewIntent(Intent i) {
        super.onNewIntent(i);
        setIntent(i);
        if (i.getStringExtra("tab") != null) selectTab(i.getStringExtra("tab"));
        engineThread.post(() -> handleIntent(i));
    }

    @Override protected void onDestroy() {
        unregisterReceiver(styleReceiver);
        Announcer.screen = null;
        if (library != null) Narrator.get(this).removeListener(library);
        if (systemTts != null) systemTts.shutdown();
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
        View line = new View(this);
        line.setBackgroundColor(Ui.LINE);
        root.addView(line, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, Math.max(1, u.dp(0.5f))));
        LinearLayout bar = u.row();
        bar.setBackgroundColor(Ui.CARD);
        bar.setPadding(u.dp(6), u.dp(6), u.dp(6), u.dp(6));
        for (int i = 0; i < TABS.length; ++i) {
            String t = TABS[i];
            TextView b = u.text(TAB_NAMES[i], 13, Ui.MUTED, true);
            b.setGravity(Gravity.CENTER);
            b.setMinHeight(u.dp(48));
            b.setSingleLine(true);
            b.setOnClickListener(v -> selectTab(t));
            b.setContentDescription(TAB_NAMES[i] + " tab");
            LinearLayout.LayoutParams lp = Ui.weight(1);
            lp.leftMargin = lp.rightMargin = u.dp(2);
            bar.addView(b, lp);
            tabButtons.put(t, b);
        }
        root.addView(bar);
        setContentView(root);
        if (Build.VERSION.SDK_INT >= 30)
            root.setOnApplyWindowInsetsListener((v, insets) -> {
                android.graphics.Insets ins = insets.getInsets(android.view.WindowInsets.Type.systemBars()
                        | android.view.WindowInsets.Type.displayCutout() | android.view.WindowInsets.Type.ime());
                v.setPadding(ins.left, ins.top, ins.right, ins.bottom);
                return android.view.WindowInsets.CONSUMED;
            });
    }

    void selectTab(String t) {
        if (!Arrays.asList(TABS).contains(t)) t = "home";
        tab = t;
        View v = tabViews.get(t);
        if (v == null) {
            v = buildTab(t);
            tabViews.put(t, v);
            content.addView(v);
        }
        for (Map.Entry<String, View> e : tabViews.entrySet()) e.getValue().setVisibility(e.getKey().equals(t) ? View.VISIBLE : View.GONE);
        for (Map.Entry<String, TextView> e : tabButtons.entrySet()) {
            boolean on = e.getKey().equals(t);
            e.getValue().setTextColor(on ? Ui.ACCENT : Ui.MUTED);
            e.getValue().setBackground(on ? Ui.round(Ui.ACCENT_SOFT, u.dp(16)) : null);
            e.getValue().setSelected(on);
        }
        // back from another tab returns home (Android 16 delivers back to this callback, not onBackPressed)
        if (Build.VERSION.SDK_INT >= 33) {
            android.window.OnBackInvokedDispatcher d = getOnBackInvokedDispatcher();
            if (!t.equals("home") && backToHome == null) {
                android.window.OnBackInvokedCallback cb = () -> selectTab("home");
                d.registerOnBackInvokedCallback(android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT, cb);
                backToHome = cb;
            } else if (t.equals("home") && backToHome != null) {
                d.unregisterOnBackInvokedCallback((android.window.OnBackInvokedCallback) backToHome);
                backToHome = null;
            }
        }
        if (t.equals("read")) library.refresh();
        if (t.equals("home")) refreshHome();
    }

    @SuppressWarnings("deprecation")
    @Override public void onBackPressed() {
        if (!tab.equals("home")) selectTab("home");
        else super.onBackPressed();
    }

    private View buildTab(String t) {
        switch (t) {
            case "read":
                library = new LibraryScreen(this, u);
                View lv = library.build();
                Narrator.get(this).addListener(library);
                return lv;
            case "voices":
                voices = new VoicesScreen(this, u, this);
                return voices.build();
            case "assistant": {
                LinearLayout root = u.column();
                root.setPadding(u.dp(16), u.dp(20), u.dp(16), u.dp(24));
                root.addView(u.header("Assistant", "It speaks up for you. Everything stays on your phone."));
                assistant = new AssistantPanel(this, u);
                root.addView(assistant.build());
                assistant.refresh();
                ScrollView sv = new ScrollView(this);
                sv.addView(root);
                return sv;
            }
            case "settings":
                settings = new SettingsScreen(this, u, this);
                return settings.build();
            default:
                return buildHome();
        }
    }

    // ---------------------------------------------------------------- home
    private View buildHome() {
        LinearLayout root = u.column();
        root.setPadding(u.dp(16), u.dp(20), u.dp(16), u.dp(24));
        root.addView(u.header("SteerTTS", "An expressive voice, fully on your phone"));

        LinearLayout st = u.card();
        phoneVoiceRow = u.settingRow("Phone's voice", "", u.chevron(), v -> SettingsScreen.openTtsSettings(this));
        st.addView(phoneVoiceRow);
        st.addView(u.divider());
        assistantRow = u.settingRow("Assistant", "", u.chevron(), v -> selectTab("assistant"));
        st.addView(assistantRow);
        root.addView(st);

        LinearLayout tryCard = u.card();
        tryCard.addView(u.label("Try it"));
        input = new EditText(this);
        input.setText(SAMPLES[0][1]);
        input.setTextSize(16);
        input.setTextColor(Ui.INK);
        input.setHintTextColor(Ui.MUTED);
        input.setHint("Type anything to hear it");
        input.setMinLines(2);
        input.setMaxLines(4);
        input.setGravity(Gravity.TOP);
        input.setBackground(Ui.round(Ui.BG, u.dp(14)));
        input.setPadding(u.dp(12), u.dp(10), u.dp(12), u.dp(10));
        tryCard.addView(input, Ui.matchWrap());
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
        for (String p : Style.PRESETS) styles.addView(fit(u.chip(p, v -> speakAs(p))));
        tryCard.addView(styles, Ui.matchWrap());
        LinearLayout actions = u.row();
        actions.setPadding(0, u.dp(12), 0, 0);
        voiceChip = u.chip("Voice", v -> selectTab("voices"));
        voiceChip.setContentDescription("Change the voice");
        actions.addView(voiceChip);
        actions.addView(new View(this), Ui.spacer());
        LinearLayout.LayoutParams sp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        sp.rightMargin = u.dp(8);
        actions.addView(u.pill("Stop", false, v -> stopAll()), sp);
        actions.addView(u.pill("Speak", true, v -> speakInput()));
        tryCard.addView(actions);
        status = u.text("Loading the voice…", 12.5f, Ui.MUTED, false);
        status.setPadding(0, u.dp(8), 0, 0);
        status.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        tryCard.addView(status);
        root.addView(tryCard);

        LinearLayout now = u.card();
        nowLabel = u.text("", 11.5f, Ui.MUTED, true);
        nowLabel.setLetterSpacing(0.08f);
        now.addView(nowLabel);
        nowText = u.text("", 15, Ui.INK, false);
        nowText.setMaxLines(3);
        nowText.setEllipsize(TextUtils.TruncateAt.END);
        nowText.setPadding(0, u.dp(4), 0, 0);
        now.addView(nowText);
        nowWhy = u.text("", 13, Ui.ACCENT, true);
        nowWhy.setPadding(0, u.dp(6), 0, 0);
        now.addView(nowWhy);
        root.addView(now);
        resetCard();

        LinearLayout quick = u.card();
        quick.addView(u.label("Listen"));
        quick.addView(u.settingRow("Read what you copied", "The clipboard, read aloud", u.chevron(), v -> {
            selectTab("read");
            library.readClipboard();
        }));
        quick.addView(u.divider());
        quick.addView(u.settingRow("Open a document", "PDF, e-book, Word or web page", u.chevron(), v -> {
            selectTab("read");
            library.openFile();
        }));
        quick.addView(u.divider());
        quick.addView(u.settingRow("My daily briefing", "Your day, sleep and steps, spoken now", u.chevron(), v -> Briefing.play(this)));
        root.addView(quick);

        ScrollView sv = new ScrollView(this);
        sv.addView(root);
        refreshHome();
        return sv;
    }

    private void refreshHome() {
        if (phoneVoiceRow == null) return;
        boolean on = SettingsScreen.isPhoneVoice(this);
        TextView s = (TextView) phoneVoiceRow.getTag();
        s.setText(on ? "✓ SteerTTS: other apps speak with it" : "Not set yet: tap to make SteerTTS your phone's voice");
        s.setTextColor(on ? Ui.ACCENT : Ui.MUTED);
        AssistantPrefs p = new AssistantPrefs(this);
        StringBuilder a = new StringBuilder();
        boolean access = AssistantPanel.listenerOn(this);
        if (access && p.readMessages()) a.append("Readouts");
        if (access && p.announceCalls()) a.append(a.length() > 0 ? " · " : "").append("Callers");
        if (p.announceMeetings() && MeetingAnnouncer.allowed(this)) a.append(a.length() > 0 ? " · " : "").append("Meetings");
        if (Briefing.daily(this)) a.append(a.length() > 0 ? " · " : "").append("Briefing");
        TextView as = (TextView) assistantRow.getTag();
        as.setText(a.length() > 0 ? "✓ " + a : "Off: tap to choose what it reads to you");
        as.setTextColor(a.length() > 0 ? Ui.ACCENT : Ui.MUTED);
        refreshVoiceChip();
    }

    private void refreshVoiceChip() {
        if (voiceChip == null) return;
        voiceChip.setText("Voice: " + Voices.get(this).nameOf(base.voice, null) + "  ▾");
    }

    private TextView fit(TextView chip) {
        LinearLayout.LayoutParams lp = Ui.weight(1);
        lp.rightMargin = u.dp(6);
        chip.setLayoutParams(lp);
        chip.setPadding(u.dp(4), u.dp(7), u.dp(4), u.dp(7));
        chip.setSingleLine(true);
        chip.setEllipsize(TextUtils.TruncateAt.END);
        return chip;
    }

    private void say(String s) { runOnUiThread(() -> { if (status != null) status.setText(s); }); }

    /** What is being said now, for whom, and why it sounds that way. */
    private void showApplied(Style s, String who, String text) {
        if (nowLabel == null) return;
        ui.removeCallbacksAndMessages(null);
        nowLabel.setText(("Now speaking · " + who).toUpperCase(Locale.ROOT));
        nowLabel.setTextColor(Ui.ACCENT);
        if (text != null) nowText.setText("“" + text.trim() + "”");
        nowWhy.setText(s.reason);
        nowWhy.setVisibility(s.reason == null || s.reason.isEmpty() ? View.GONE : View.VISIBLE);
    }

    private void resetCard() {
        if (nowLabel == null) return;
        ui.removeCallbacksAndMessages(null);
        nowLabel.setText("NOTHING PLAYING");
        nowLabel.setTextColor(Ui.MUTED);
        nowText.setText("Messages, callers, meetings and other apps' speech show up here as SteerTTS says them.");
        nowWhy.setVisibility(View.GONE);
    }

    private void settle(long delayMs) {
        ui.postDelayed(() -> {
            if (nowLabel == null) return;
            nowLabel.setText(nowLabel.getText().toString().replace("NOW SPEAKING", "LAST SPOKEN"));
            nowLabel.setTextColor(Ui.MUTED);
        }, delayMs);
    }

    /** "WhatsApp", "Meeting reminder", "Maps (system voice)" for a broadcast's caller and source. */
    private String who(String caller, String source) {
        if ("calendar".equals(caller)) return "Meeting reminder";
        if ("read-aloud".equals(caller)) return "Read aloud";
        if ("briefing".equals(caller)) return "Daily briefing";
        if ("onboarding".equals(caller) || caller == null || caller.equals(getPackageName())) return "SteerTTS";
        String name = caller;
        try {
            name = getPackageManager().getApplicationLabel(getPackageManager().getApplicationInfo(caller, 0)).toString();
        } catch (Exception ignored) {
        }
        return "assistant".equals(source) ? name : name + " (system voice)";
    }

    // ---------------------------------------------------------------- VoicesScreen.Host
    @Override public Style base() { return base; }

    @Override public void saveBase() {
        base.save(this);
        refreshVoiceChip();
    }

    @Override public void preview(String text, Style s) {
        if (tts != null) tts.cancel();
        player.flush();
        engineThread.removeCallbacksAndMessages(null);
        engineThread.post(() -> speakStyled(text, s, null));
    }

    // ---------------------------------------------------------------- SettingsScreen.Host
    @Override public void setThreads(int n) { threadsSel = n; }
    @Override public int threads() { return threadsSel; }
    @Override public void benchmark(java.util.function.Consumer<String> result) { engineThread.post(() -> result.accept(benchmark("0,1"))); }
    @Override public void testSystemVoice() { speakThroughSystem("This is SteerTTS speaking through Android's text to speech."); }
    @Override public void showWelcome() { startActivity(new Intent(this, OnboardingActivity.class)); }

    // ---------------------------------------------------------------- engine thread
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
            runOnUiThread(() -> { if (input != null) input.setText(t); });
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
        if (i.getBooleanExtra("briefing", false)) Briefing.play(this);
        if (i.getStringExtra("announce") != null)
            Announcer.get(this).say(i.getStringExtra("announce"), getPackageName(), true, null);
        String bench = i.getStringExtra("bench");
        String text = i.getStringExtra("speak");
        String sys = i.getStringExtra("systts");
        if (bench != null) say(benchmark(bench));
        if (text != null) {
            Style s = base.copy();
            if (i.hasExtra("voice")) s.voice = i.getStringExtra("voice");
            speakStyled(text, ContextStyler.decide(this, s, text, getPackageName()), null);
        }
        if (sys != null) runOnUiThread(() -> speakThroughSystem(sys));
    }

    private void speakInput() {
        String text = input.getText().toString();
        if (text.trim().isEmpty()) return;
        Style s = ContextStyler.decide(this, base.copy(), text, getPackageName());
        say("Speaking…");
        engineThread.post(() -> speakStyled(text, s, null));
    }

    /** The input text once, in one of the "Say it" deliveries (the saved style is not changed). */
    private void speakAs(String preset) {
        String text = input.getText().toString();
        if (text.trim().isEmpty()) return;
        Style s = base.preset(preset);
        say("Speaking…");
        engineThread.post(() -> speakStyled(text, s, preset));
    }

    private void stopAll() {
        if (tts != null) tts.cancel();
        player.flush();
        Announcer.get(this).stop();   // readouts too
        Narrator.get(this).pause();
        say("Stopped.");
    }

    /** Speaks with exactly this style; label names a "Say it" delivery for the now-speaking card. */
    private void speakStyled(String text, Style s, String label) {
        if (modelDir == null) return;
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
            Log.i(TAG, String.format(Locale.US, "speak %s: %.2f s of audio, first sound after %.0f ms, rendered in %.0f ms, %d threads [%s]",
                    voice, audioS, first[0], total, e.threads(), s.reason));
            if (first[0] >= 0)
                say(String.format(Locale.US, "First sound in %.0f ms · %.0f× faster than real time · offline", first[0], audioS * 1000 / total));
        } catch (RuntimeException ex) {
            Log.e(TAG, "speak", ex);
            say("Couldn't speak that one. Try a shorter text.");
        }
        runOnUiThread(() -> settle(2500));
    }

    private static String resolveVoice(SteerTTS e, String v) {
        if (v == null || v.equals("female")) return e.genderVoice("female");
        if (v.equals("male")) return e.genderVoice("male");
        return e.voiceGender(v) != null ? v : e.genderVoice("female");
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
    private String benchmark(String spec) {
        if (modelDir == null) return "The voice is still loading.";
        StringBuilder out = new StringBuilder();
        int keep = threadsSel;
        for (String s : spec.split(",")) {
            threadsSel = Integer.parseInt(s.trim());
            long tl = System.nanoTime();
            engine();
            double load = ms(tl);
            for (int w = 0; w < 2; ++w) render(BENCH[w]);
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
            String line = String.format(Locale.US, "%s: first audio %.0f ms, %.0f× real time (load %.0f ms)",
                    tts.threads() == 1 ? "1 thread" : tts.threads() + " threads", firsts[2], audio * 1000 / synth, load);
            Log.i(TAG, "bench " + line);
            out.append(line).append('\n');
        }
        threadsSel = keep;
        engine();
        return out.toString().trim();
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

    // ---------------------------------------------------------------- the system engine, as other apps use it
    private void speakThroughSystem(String text) {
        long t0 = System.nanoTime();
        Runnable go = () -> {
            systemTts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                double started = -1;
                @Override public void onStart(String id) { }
                @Override public void onBeginSynthesis(String id, int sr, int fmt, int ch) { }
                @Override public void onAudioAvailable(String id, byte[] audio) { if (started < 0) started = ms(t0); }
                @Override public void onDone(String id) {
                    Log.i(TAG, String.format(Locale.US, "systts done: first audio %.0f ms after the request, %.0f ms total", started, ms(t0)));
                }
                @Override public void onError(String id) { Log.e(TAG, "systts error"); }
            });
            systemTts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "u" + System.nanoTime());
        };
        if (systemTts != null) go.run();
        else systemTts = new TextToSpeech(this, st -> { if (st == TextToSpeech.SUCCESS) go.run(); }, getPackageName());
    }
}
