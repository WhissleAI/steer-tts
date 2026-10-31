package ai.whissle.steertts.demo;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.net.Uri;
import android.provider.Settings;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;

/** The Settings tab: appearance, the phone's voice, privacy, about, and the developer tools (threads, benchmark). */
final class SettingsScreen {
    interface Host {
        void setThreads(int n);
        int threads();
        void benchmark(java.util.function.Consumer<String> result);
        void testSystemVoice();
        void showWelcome();
        void compareSmooth();   // the same sentence, raw then smoothed
    }

    static final int[] THREADS = {0, 1, 2, 3, 4, 6, 8};

    private final Activity a;
    private final Ui u;
    private final Host host;
    private LinearLayout voiceRow;
    private final List<TextView> themeChips = new ArrayList<>();

    SettingsScreen(Activity a, Ui u, Host host) {
        this.a = a;
        this.u = u;
        this.host = host;
    }

    static boolean isPhoneVoice(Activity a) {
        return a.getPackageName().equals(Settings.Secure.getString(a.getContentResolver(), Settings.Secure.TTS_DEFAULT_SYNTH));
    }

    static void openTtsSettings(Activity a) {
        try {
            a.startActivity(new Intent("com.android.settings.TTS_SETTINGS"));
        } catch (Exception e) {
            a.startActivity(new Intent(Settings.ACTION_SETTINGS));
        }
    }

    View build() {
        LinearLayout root = u.column();
        root.setPadding(u.dp(18), u.dp(6), u.dp(18), u.dp(28));

        LinearLayout sound = u.card();
        sound.addView(u.label("Sound"));
        Style st = Style.load(a);
        android.widget.Switch sm = u.toggle(st.smooth);
        sm.setOnCheckedChangeListener((v, on) -> {
            Style s2 = Style.load(a);
            s2.smooth = on;
            s2.save(a);
            Smooth.on = on;
        });
        sound.addView(u.settingRow("Smoother sound", "Softens a metallic edge some phone speakers bring out", sm, null));
        sound.addView(u.divider());
        sound.addView(u.settingRow("Hear the difference", "The same sentence, first as it is, then smoothed", u.chevron(),
                v -> host.compareSmooth()));
        root.addView(sound);

        LinearLayout look = u.card();
        look.addView(u.label("Appearance"));
        LinearLayout chips = u.row();
        String[] names = {"System", "Light", "Dark"};
        for (int i = 0; i < 3; ++i) {
            final int mode = i;
            TextView ch = u.chip(names[i], v -> {
                Ui.setThemePref(a, mode);
                a.recreate();
            });
            themeChips.add(ch);
            chips.addView(ch);
        }
        look.addView(chips);
        root.addView(look);

        LinearLayout reader = u.card();
        reader.addView(u.label("Reading"));
        Narrator n = Narrator.get(a);
        android.widget.Switch dv = u.toggle(n.dialogueVoice());
        dv.setOnCheckedChangeListener((v, on) -> n.setDialogueVoice(on));
        reader.addView(u.settingRow("Dialogue in a second voice", "Words in quotation marks are read by another voice", dv, null));
        reader.addView(u.divider());
        android.widget.Switch an = u.toggle(n.autoNext());
        an.setOnCheckedChangeListener((v, on) -> n.setAutoNext(on));
        reader.addView(u.settingRow("Play the next one", "When a document ends, start the next in your list", an, null));
        root.addView(reader);

        LinearLayout sys = u.card();
        sys.addView(u.label("Your phone"));
        voiceRow = u.settingRow("Phone's voice", "", u.chevron(), v -> openTtsSettings(a));
        sys.addView(voiceRow);
        sys.addView(u.divider());
        sys.addView(u.settingRow("Your data", "What SteerTTS reads, and why it never leaves your phone", u.chevron(),
                v -> a.startActivity(new Intent(a, PrivacyActivity.class))));
        root.addView(sys);

        LinearLayout about = u.card();
        about.addView(u.label("About"));
        String ver;
        try {
            ver = a.getPackageManager().getPackageInfo(a.getPackageName(), 0).versionName;
        } catch (Exception e) {
            ver = "";
        }
        about.addView(u.settingRow("SteerTTS " + ver, "An expressive voice that runs entirely on your phone. Made by Whissle.", null, null));
        about.addView(u.divider());
        about.addView(u.settingRow("Whissle", "whissle.ai", u.chevron(), v -> a.startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse("https://whissle.ai")))));
        about.addView(u.divider());
        about.addView(u.settingRow("Privacy policy", "The full policy", u.chevron(),
                v -> a.startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(PrivacyActivity.POLICY)))));
        about.addView(u.divider());
        about.addView(u.settingRow("Licences", "Open-source notices", u.chevron(), v -> new AlertDialog.Builder(a)
                .setTitle("Licences")
                .setMessage("SteerTTS app and engine: CC BY-NC 4.0 + Whissle terms (non-commercial), github.com/WhissleAI/steer-tts.\n\n"
                        + "The SteerTTS voice model is Whissle's own and is included for use in this app.\n\n"
                        + "Pronunciation data derived from the CMU Pronouncing Dictionary (BSD licence).")
                .setPositiveButton("OK", null).show()));
        about.addView(u.divider());
        about.addView(u.settingRow("Show the welcome again", "The short tour from the first start", u.chevron(), v -> host.showWelcome()));
        root.addView(about);

        LinearLayout dev = u.card();
        TextView dh = u.text("Developer  ▾", 14, Ui.ACCENT, true);
        dh.setMinHeight(u.dp(44));
        dh.setGravity(android.view.Gravity.CENTER_VERTICAL);
        dev.setPadding(dev.getPaddingLeft(), u.dp(6), dev.getPaddingRight(), u.dp(6));
        LinearLayout box = u.column();
        box.setVisibility(View.GONE);
        dh.setOnClickListener(v -> {
            boolean open = box.getVisibility() != View.VISIBLE;
            box.setVisibility(open ? View.VISIBLE : View.GONE);
            dh.setText(open ? "Developer  ▴" : "Developer  ▾");
        });
        dev.addView(dh);
        Spinner threads = new Spinner(a);
        List<String> tl = new ArrayList<>();
        int sel = 0;
        for (int i = 0; i < THREADS.length; ++i) {
            tl.add(THREADS[i] == 0 ? "Auto (one per fast core)" : THREADS[i] + (THREADS[i] == 1 ? " thread" : " threads"));
            if (THREADS[i] == host.threads()) sel = i;
        }
        threads.setAdapter(new ArrayAdapter<>(a, android.R.layout.simple_spinner_dropdown_item, tl));
        threads.setSelection(sel);
        threads.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(android.widget.AdapterView<?> p, View v, int pos, long id) { host.setThreads(THREADS[pos]); }
            @Override public void onNothingSelected(android.widget.AdapterView<?> p) { }
        });
        box.addView(u.text("Benchmark threads", 13, Ui.MUTED, false));
        box.addView(threads);
        TextView result = u.text("", 13, Ui.MUTED, false);
        result.setPadding(0, u.dp(8), 0, 0);
        LinearLayout br = u.row();
        br.addView(u.pill("Run benchmark", false, v -> {
            result.setText("Benchmarking…");
            host.benchmark(r -> a.runOnUiThread(() -> result.setText(r)));
        }));
        br.addView(new View(a), new LinearLayout.LayoutParams(u.dp(8), 1));
        br.addView(u.pill("Test system voice", false, v -> host.testSystemVoice()));
        box.addView(br);
        box.addView(result);
        dev.addView(box);
        root.addView(dev);

        refresh();
        ScrollView sv = new ScrollView(a);
        sv.addView(root);
        return sv;
    }

    void refresh() {
        boolean on = isPhoneVoice(a);
        TextView s = (TextView) voiceRow.getTag();
        s.setText(on ? "✓ SteerTTS is your phone's voice: other apps speak with it" : "Not set: tap, then choose SteerTTS as the preferred engine");
        s.setTextColor(on ? Ui.ACCENT : Ui.MUTED);
        int t = Ui.themePref(a);
        for (int i = 0; i < 3; ++i) u.select(themeChips.get(i), i == t);
    }
}
