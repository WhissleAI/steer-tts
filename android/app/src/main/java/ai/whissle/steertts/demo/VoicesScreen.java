package ai.whissle.steertts.demo;

import android.app.Activity;
import android.view.Gravity;
import android.view.View;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The Voices tab: pick a voice (named, described, previewed, favourited, filtered), a style, the speed and the
 * delivery. The four main controls are the ones the model follows reliably; the rest sit under Advanced. Every
 * change is saved as the user's base style (used by the app, the assistant, the reader and the system voice) and
 * previewed when the finger lifts.
 */
final class VoicesScreen {
    interface Host {
        Style base();
        void saveBase();                          // persist the base style and refresh the other screens
        void preview(String text, Style s);       // speak now, interrupting a previous preview
        void stop();                              // stop whatever this screen is saying
        void status(TextView t);                  // where to show "First sound in 41 ms"
        void openSettings();
    }

    private static final String PREVIEW = "Hi, this is how I sound. Is this the voice you want?";
    private static final String[][] MAIN = {{"f0_med", "Pitch", "Lower", "Higher"}, {"f0_iqr", "Expressiveness", "Flatter", "Livelier"},
            {"loud", "Loudness", "Softer", "Louder"}, {"f0_end", "Endings", "Fall", "Rise"}};
    private static final String[][] ADVANCED = {{"rate", "Speaking rate", "Slower", "Faster"}, {"tilt", "Brightness", "Darker", "Brighter"},
            {"h1h2", "Breathiness", "Pressed", "Breathy"}, {"centroid", "Timbre", "Duller", "Sharper"}, {"pause", "Pauses", "Fewer", "More"},
            {"voiced", "Voicing", "Less", "More"}, {"flat", "Steadiness", "Varied", "Steady"}, {"cpps", "Clarity", "Rougher", "Clearer"}};
    private static final String[] FILTERS = {"All", "Female", "Male", "★ Favourites", "Deep", "Warm", "Bright", "Relaxed", "Brisk"};
    static final int STEPS = 60;

    private final Activity a;
    private final Ui u;
    private final Host host;
    private final Voices voices;
    private LinearLayout list, currentCard;
    private final List<TextView> filterChips = new ArrayList<>(), styleChips = new ArrayList<>();
    private final List<SeekBar> bars = new ArrayList<>();
    private final List<String> barControls = new ArrayList<>();
    private final List<TextView> barValues = new ArrayList<>();
    private SeekBar speed;
    private TextView speedVal;
    private Switch contextual;
    private String filter = "All";
    private android.widget.EditText input;
    private TextView more;
    private boolean showAll;
    private static final int SHOWN = 12;

    VoicesScreen(Activity a, Ui u, Host host) {
        this.a = a;
        this.u = u;
        this.host = host;
        voices = Voices.get(a);
    }

    View build() {
        LinearLayout root = u.column();
        root.setPadding(u.dp(18), u.dp(14), u.dp(18), u.dp(28));
        root.addView(u.topBar("Voices", v -> host.openSettings()));
        root.addView(tryCard());

        currentCard = u.card();
        root.addView(currentCard);

        LinearLayout style = u.card();
        style.addView(u.label("Style"));
        LinearLayout r1 = u.row(), r2 = u.row();
        for (int i = 0; i < Style.STYLES.length; ++i) {
            String s = Style.STYLES[i];
            TextView ch = u.chip(s, v -> {
                Style b = host.base();
                b.applyPreset(s);
                host.saveBase();
                showStyle();
                host.preview(PREVIEW, ContextStyler.decide(a, b, PREVIEW, a.getPackageName()));
            });
            LinearLayout.LayoutParams lp = Ui.weight(1);
            lp.rightMargin = u.dp(6);
            lp.topMargin = u.dp(6);
            ch.setLayoutParams(lp);
            styleChips.add(ch);
            (i < 4 ? r1 : r2).addView(ch);
        }
        style.addView(r1, Ui.matchWrap());
        style.addView(r2, Ui.matchWrap());
        root.addView(style);

        LinearLayout delivery = u.card();
        delivery.addView(u.label("Delivery"));
        speed = new SeekBar(a);
        speed.setMax(Math.round((Style.MAX_PACE - Style.MIN_PACE) * 20));
        speed.setContentDescription("Speed");
        speedVal = u.text("", 13, Ui.MUTED, false);
        speed.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar s, int p, boolean fromUser) {
                speedVal.setText(String.format(Locale.US, "%.2f×", Style.MIN_PACE + p / 20f));
                if (fromUser) host.base().pace = Style.MIN_PACE + p / 20f;
            }
            @Override public void onStartTrackingTouch(SeekBar s) { }
            @Override public void onStopTrackingTouch(SeekBar s) { changedControl(); }
        });
        delivery.addView(sliderRow("Speed", speed, speedVal, "Slower", "Faster"));
        for (String[] c : MAIN) delivery.addView(control(c));
        TextView adv = u.text("Advanced  ▾", 14, Ui.ACCENT, true);
        adv.setPadding(0, u.dp(12), 0, u.dp(6));
        adv.setMinHeight(u.dp(44));
        LinearLayout advBox = u.column();
        advBox.setVisibility(View.GONE);
        TextView note = u.text("These shape the voice too, but the model follows them less reliably.", 12.5f, Ui.MUTED, false);
        advBox.addView(note);
        for (String[] c : ADVANCED) advBox.addView(control(c));
        adv.setOnClickListener(v -> {
            boolean open = advBox.getVisibility() != View.VISIBLE;
            advBox.setVisibility(open ? View.VISIBLE : View.GONE);
            adv.setText(open ? "Advanced  ▴" : "Advanced  ▾");
        });
        delivery.addView(adv);
        delivery.addView(advBox);
        contextual = u.toggle(host.base().contextual);
        contextual.setOnCheckedChangeListener((v, on) -> {
            host.base().contextual = on;
            host.saveBase();
        });
        delivery.addView(u.settingRow("Adapt to the moment", "Alerts faster and louder, questions rising, softer late at night",
                contextual, null));
        LinearLayout btns = u.row();
        btns.setPadding(0, u.dp(8), 0, 0);
        btns.addView(u.pill("Reset", false, v -> {
            Style b = host.base();
            b.applyPreset("Natural");
            b.pace = Style.DEFAULT_PACE;
            host.saveBase();
            refresh();
        }));
        btns.addView(new View(a), Ui.spacer());
        btns.addView(u.pill("Preview", true, v -> host.preview(PREVIEW, ContextStyler.decide(a, host.base(), PREVIEW, a.getPackageName()))));
        delivery.addView(btns);
        root.addView(delivery);

        root.addView(u.section("Choose a voice"));
        HorizontalScrollView fs = new HorizontalScrollView(a);
        fs.setHorizontalScrollBarEnabled(false);
        LinearLayout feat = u.row();
        for (Voices.Voice v : voices.featured()) feat.addView(featuredCard(v));
        fs.addView(feat);
        root.addView(fs);

                HorizontalScrollView fsc = new HorizontalScrollView(a);
        fsc.setHorizontalScrollBarEnabled(false);
        LinearLayout filters = u.row();
        for (String f : FILTERS) {
            TextView ch = u.chip(f, v -> {
                filter = f;
                fillList();
            });
            filterChips.add(ch);
            filters.addView(ch);
        }
        fsc.addView(filters);
        root.addView(fsc);
        LinearLayout listCard = u.card();
        list = u.column();
        listCard.addView(list);
        root.addView(listCard);
        more = u.pill("", false, v -> {
            showAll = !showAll;
            fillList();
        });
        LinearLayout.LayoutParams ml = Ui.matchWrap();
        ml.topMargin = u.dp(10);
        root.addView(more, ml);


        refresh();
        ScrollView sv = new ScrollView(a);
        sv.addView(root);
        return sv;
    }

    /** Type anything and hear it: in your voice and style, or once in a "Say it" delivery. */
    private View tryCard() {
        LinearLayout c = u.card();
        c.setPadding(u.dp(18), u.dp(16), u.dp(18), u.dp(16));
        input = new android.widget.EditText(a);
        input.setText("Hi! I'm SteerTTS, a voice that runs entirely on your phone. No internet needed.");
        input.setTextSize(16);
        input.setTextColor(Ui.INK);
        input.setHintTextColor(Ui.MUTED);
        input.setHint("Type anything to hear it");
        input.setMinLines(2);
        input.setMaxLines(4);
        input.setGravity(Gravity.TOP);
        input.setBackground(Ui.round(Ui.BG, u.dp(16)));
        input.setPadding(u.dp(14), u.dp(12), u.dp(14), u.dp(12));
        c.addView(input, Ui.matchWrap());
        HorizontalScrollView hs = new HorizontalScrollView(a);
        hs.setHorizontalScrollBarEnabled(false);
        LinearLayout chips = u.row();
        chips.setPadding(0, u.dp(10), 0, 0);
        for (String p : Style.PRESETS)
            chips.addView(u.chip(p, v -> {
                String t = input.getText().toString().trim();
                if (!t.isEmpty()) host.preview(t, host.base().preset(p));
            }));
        hs.addView(chips);
        c.addView(hs);
        LinearLayout r = u.row();
        r.setPadding(0, u.dp(10), 0, 0);
        TextView st = u.text("", 12.5f, Ui.MUTED, false);
        st.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        host.status(st);
        r.addView(st, Ui.weight(1));
        LinearLayout.LayoutParams sp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        sp.rightMargin = u.dp(8);
        r.addView(u.pill("Stop", false, v -> host.stop()), sp);
        r.addView(u.pill("Speak", true, v -> {
            String t = input.getText().toString().trim();
            if (!t.isEmpty()) host.preview(t, ContextStyler.decide(a, host.base().copy(), t, a.getPackageName()));
        }));
        c.addView(r, Ui.matchWrap());
        return c;
    }

    void refresh() {
        showCurrent();
        fillList();
        showStyle();
        Style b = host.base();
        speed.setProgress(Math.round((b.pace - Style.MIN_PACE) * 20));
        for (int i = 0; i < bars.size(); ++i) bars.get(i).setProgress(progressOf(zOf(b, barControls.get(i))));
        contextual.setChecked(b.contextual);
    }

    // ---------------------------------------------------------------- voice
    private String currentId() {
        String v = host.base().voice;
        if (v == null || "female".equals(v)) return Voices.FEATURED[0];
        if ("male".equals(v)) return Voices.FEATURED[1];
        return v;
    }

    private void select(String id) {
        host.base().voice = id;
        host.saveBase();
        showCurrent();
        fillList();
        Voices.Voice v = voices.voice(id);
        host.preview(named(v != null ? v.name : "your new voice"), host.base().copy());
    }

    private void showCurrent() {
        currentCard.removeAllViews();
        Voices.Voice v = voices.voice(currentId());
        currentCard.addView(u.label("Your voice"));
        LinearLayout r = u.row();
        LinearLayout t = u.column();
        t.addView(u.text(v != null ? v.name : currentId(), 26, Ui.INK, true));
        t.addView(u.text(v != null ? v.gender() + " · " + v.describe() + " · " + host.base().preset : "", 14, Ui.MUTED, false));
        r.addView(t, Ui.weight(1));
        r.addView(u.icon(R.drawable.ic_play, "Preview your voice", 52, true,
                x -> host.preview(PREVIEW, ContextStyler.decide(a, host.base(), PREVIEW, a.getPackageName()))));
        currentCard.addView(r);
    }

    private View featuredCard(Voices.Voice v) {
        LinearLayout c = u.column();
        c.setBackground(Ui.round(Ui.CARD, u.dp(18)));
        c.setPadding(u.dp(14), u.dp(12), u.dp(10), u.dp(12));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(u.dp(150), LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.rightMargin = u.dp(10);
        lp.topMargin = u.dp(6);
        c.setLayoutParams(lp);
        LinearLayout top = u.row();
        top.addView(u.text(v.name, 18, Ui.INK, true), Ui.weight(1));
        top.addView(u.icon(R.drawable.ic_play, "Preview " + v.name, 40, false, x -> previewVoice(v)));
        c.addView(top);
        c.addView(u.text(v.gender() + "\n" + v.describe(), 12.5f, Ui.MUTED, false));
        c.setOnClickListener(x -> select(v.id));
        c.setContentDescription(v.name + ", " + v.gender() + ", " + v.describe() + ". Double tap to use this voice.");
        return c;
    }

    private static String named(String name) { return "Hi, I'm " + name + ". This is how I sound."; }

    private void previewVoice(Voices.Voice v) {
        Style s = host.base().copy();
        s.voice = v.id;
        host.preview(named(v.name), s);
    }

    private void fillList() {
        for (int i = 0; i < FILTERS.length; ++i) u.select(filterChips.get(i), FILTERS[i].equals(filter));
        list.removeAllViews();
        String cur = currentId();
        int n = 0, total = 0;
        for (Voices.Voice v : voices.all()) {
            if (!matches(v)) continue;
            total++;
            if (!showAll && n >= SHOWN) continue;
            if (n++ > 0) list.addView(u.divider());
            list.addView(row(v, v.id.equals(cur)));
        }
        if (n == 0) list.addView(u.text("No voices here yet. Tap ☆ on a voice to keep it in Favourites.", 14, Ui.MUTED, false));
        if (more != null) {
            more.setVisibility(total > SHOWN ? View.VISIBLE : View.GONE);
            more.setText(showAll ? "Show fewer" : "Show all " + total + " voices");
        }
    }

    private boolean matches(Voices.Voice v) {
        switch (filter) {
            case "Female": return "female".equals(v.gender);
            case "Male": return "male".equals(v.gender);
            case "★ Favourites": return voices.isFavourite(v.id);
            case "Deep": return "low".equals(v.pitch);
            case "Warm": return "mid".equals(v.pitch);
            case "Bright": return "high".equals(v.pitch);
            case "Relaxed": return "slow".equals(v.pace);
            case "Brisk": return "fast".equals(v.pace);
            default: return true;
        }
    }

    private View row(Voices.Voice v, boolean current) {
        LinearLayout r = u.settingRow((current ? "✓  " : "") + v.name, v.gender() + " · " + v.describe(), null, x -> select(v.id));
        r.setContentDescription(v.name + ", " + v.gender() + ", " + v.describe() + (current ? ", selected" : ""));
        TextView star = u.text(voices.isFavourite(v.id) ? "★" : "☆", 24, voices.isFavourite(v.id) ? Ui.RED : Ui.MUTED, false);
        star.setGravity(Gravity.CENTER);
        star.setMinWidth(u.dp(48));
        star.setMinHeight(u.dp(48));
        star.setContentDescription(voices.isFavourite(v.id) ? "Remove from favourites" : "Add to favourites");
        star.setOnClickListener(x -> {
            voices.toggleFavourite(v.id);
            boolean f = voices.isFavourite(v.id);
            star.setText(f ? "★" : "☆");
            star.setTextColor(f ? Ui.RED : Ui.MUTED);
            star.setContentDescription(f ? "Remove from favourites" : "Add to favourites");
        });
        r.addView(u.icon(R.drawable.ic_play, "Preview " + v.name, 40, false, x -> previewVoice(v)));
        r.addView(star);
        if (current) ((TextView) ((LinearLayout) r.getChildAt(0)).getChildAt(0)).setTextColor(Ui.ACCENT);
        return r;
    }

    // ---------------------------------------------------------------- style and delivery
    private void showStyle() {
        String p = host.base().preset;
        for (int i = 0; i < Style.STYLES.length; ++i) u.select(styleChips.get(i), Style.STYLES[i].equals(p));
    }

    private TextView section(String s) {
        TextView t = u.label(s);
        t.setPadding(u.dp(4), u.dp(18), 0, 0);
        return t;
    }

    private View control(String[] c) {
        SeekBar bar = new SeekBar(a);
        bar.setMax(STEPS);
        bar.setContentDescription(c[1] + ", from " + c[2].toLowerCase(Locale.ROOT) + " to " + c[3].toLowerCase(Locale.ROOT));
        TextView val = u.text("", 13, Ui.MUTED, false);
        bar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar s, int p, boolean fromUser) {
                float z = p * 2 * Style.LIMIT / STEPS - Style.LIMIT;
                val.setText(String.format(Locale.US, "%+.1f", z));
                if (fromUser) setZ(host.base(), c[0], Math.abs(z) < 0.05f ? 0 : z);
            }
            @Override public void onStartTrackingTouch(SeekBar s) { }
            @Override public void onStopTrackingTouch(SeekBar s) {
                host.base().preset = "Custom";
                changedControl();
                showStyle();
            }
        });
        bars.add(bar);
        barControls.add(c[0]);
        barValues.add(val);
        return sliderRow(c[1], bar, val, c[2], c[3]);
    }

    private LinearLayout sliderRow(String label, SeekBar bar, TextView val, String lo, String hi) {
        LinearLayout box = u.column();
        box.setPadding(0, u.dp(6), 0, u.dp(2));
        LinearLayout top = u.row();
        top.addView(u.text(label, 15, Ui.INK, true), Ui.weight(1));
        top.addView(val);
        box.addView(top);
        box.addView(bar);
        LinearLayout ends = u.row();
        ends.addView(u.text(lo, 12, Ui.MUTED, false), Ui.weight(1));
        ends.addView(u.text(hi, 12, Ui.MUTED, false));
        box.addView(ends);
        return box;
    }

    private void changedControl() {
        host.saveBase();
        showCurrent();
        host.preview(PREVIEW, ContextStyler.decide(a, host.base(), PREVIEW, a.getPackageName()));
    }

    private static int progressOf(float z) { return Math.round((z + Style.LIMIT) / (2 * Style.LIMIT) * STEPS); }

    private static float zOf(Style s, String control) {
        for (int i = 0; i < SteerTTS.CONTROLS.length; ++i) if (SteerTTS.CONTROLS[i].equals(control)) return s.z[i];
        return 0;
    }

    private static void setZ(Style s, String control, float z) {
        for (int i = 0; i < SteerTTS.CONTROLS.length; ++i) if (SteerTTS.CONTROLS[i].equals(control)) s.z[i] = Style.clamp(z);
    }
}
