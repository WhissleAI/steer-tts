package ai.whissle.steertts.demo;

import android.app.UiModeManager;
import android.content.Context;
import android.content.res.ColorStateList;
import android.content.res.Configuration;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.os.Build;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.Switch;
import android.widget.TextView;

/**
 * The app's look, in code (no AndroidX): a light and a dark palette in Whissle's colours, cards, text styles, pill
 * buttons, chips, settings rows and the tab bar. init() picks the palette for the current configuration (system
 * setting, or the user's choice in Settings on Android 12+) and must run before any view is built.
 */
final class Ui {
    static int BG, CARD, INK, MUTED, LINE, ACCENT, ACCENT_SOFT, ON_ACCENT, WARN, RED;
    static boolean dark;

    /** 0 = follow the system, 1 = light (the default), 2 = dark (stored in the "ui" preferences). */
    static int themePref(Context c) { return c.getSharedPreferences("ui", Context.MODE_PRIVATE).getInt("theme", 1); }

    static void setThemePref(Context c, int mode) {
        c.getSharedPreferences("ui", Context.MODE_PRIVATE).edit().putInt("theme", mode).apply();
        if (Build.VERSION.SDK_INT >= 31) {
            UiModeManager m = c.getSystemService(UiModeManager.class);
            m.setApplicationNightMode(mode == 1 ? UiModeManager.MODE_NIGHT_NO : mode == 2 ? UiModeManager.MODE_NIGHT_YES
                    : UiModeManager.MODE_NIGHT_AUTO);
        }
    }

    static void init(Context c) {
        int pref = themePref(c);
        if (!c.getSharedPreferences("ui", Context.MODE_PRIVATE).contains("theme")) setThemePref(c, pref);   // first start: light
        dark = pref == 2 || pref == 0
                && (c.getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES;
        if (dark) {
            BG = 0xFF0F1513; CARD = 0xFF19211E; INK = 0xFFEAF1EE; MUTED = 0xFF98A7A1; LINE = 0xFF2A3531;
            ACCENT = 0xFF6FD8B8; ACCENT_SOFT = 0xFF1D3A31; ON_ACCENT = 0xFF06231B; WARN = 0xFFFFB4AB; RED = 0xFFFF7A7E;
        } else {
            BG = 0xFFF4F6F5; CARD = 0xFFFFFFFF; INK = 0xFF13201C; MUTED = 0xFF5F6E68; LINE = 0xFFE2E8E5;
            ACCENT = 0xFF1A6A55; ACCENT_SOFT = 0xFFE1F1EB; ON_ACCENT = 0xFFFFFFFF; WARN = 0xFFB3261E; RED = 0xFFF14E52;
        }
    }

    private final Context c;

    Ui(Context c) { this.c = c; }

    int dp(float x) { return Math.round(x * c.getResources().getDisplayMetrics().density); }

    static GradientDrawable round(int color, float radiusPx) {
        GradientDrawable g = new GradientDrawable();
        g.setColor(color);
        g.setCornerRadius(radiusPx);
        return g;
    }

    LinearLayout column() {
        LinearLayout l = new LinearLayout(c);
        l.setOrientation(LinearLayout.VERTICAL);
        return l;
    }

    LinearLayout row() {
        LinearLayout l = new LinearLayout(c);
        l.setGravity(Gravity.CENTER_VERTICAL);
        return l;
    }

    /** A rounded card with padding, spaced from the one above. */
    LinearLayout card() {
        LinearLayout l = column();
        l.setBackground(round(CARD, dp(20)));
        l.setPadding(dp(16), dp(14), dp(16), dp(14));
        if (!dark) l.setElevation(dp(1));
        LinearLayout.LayoutParams lp = matchWrap();
        lp.topMargin = dp(12);
        l.setLayoutParams(lp);
        return l;
    }

    TextView text(String s, float sp, int color, boolean bold) {
        TextView t = new TextView(c);
        t.setText(s);
        t.setTextSize(sp);
        t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        return t;
    }

    /** A small uppercase section label ("ASSISTANT"). */
    TextView label(String s) {
        TextView t = text(s.toUpperCase(), 11.5f, ACCENT, true);
        t.setLetterSpacing(0.08f);
        t.setPadding(0, 0, 0, dp(6));
        return t;
    }

    /** A screen title with an optional line under it. */
    LinearLayout header(String title, String sub) {
        LinearLayout l = column();
        TextView t = text(title, 28, INK, true);
        if (Build.VERSION.SDK_INT >= 28) t.setAccessibilityHeading(true);
        l.addView(t);
        if (sub != null) l.addView(text(sub, 14, MUTED, false));
        return l;
    }

    /** A filled (primary) or tonal pill button, at least 48 dp tall. */
    TextView pill(String s, boolean primary, View.OnClickListener l) {
        TextView b = text(s, 15, primary ? ON_ACCENT : ACCENT, true);
        b.setGravity(Gravity.CENTER);
        b.setMinHeight(dp(48));
        b.setPadding(dp(18), dp(10), dp(18), dp(10));
        b.setBackground(new RippleDrawable(ColorStateList.valueOf(0x33000000),
                round(primary ? ACCENT : ACCENT_SOFT, dp(24)), null));
        b.setOnClickListener(l);
        b.setFocusable(true);
        return b;
    }

    /** A round icon button with a text glyph (▶ ❚❚ ⏮ ⏭ ★ …) and a spoken label for screen readers. */
    TextView glyph(String glyph, String spoken, float sizeDp, boolean primary, View.OnClickListener l) {
        TextView b = text(glyph, sizeDp * 0.42f, primary ? ON_ACCENT : ACCENT, true);
        b.setGravity(Gravity.CENTER);
        b.setContentDescription(spoken);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(dp(sizeDp), dp(sizeDp));
        b.setLayoutParams(lp);
        b.setBackground(new RippleDrawable(ColorStateList.valueOf(0x33000000),
                round(primary ? ACCENT : ACCENT_SOFT, dp(sizeDp / 2)), null));
        b.setOnClickListener(l);
        b.setFocusable(true);
        return b;
    }

    /** A round icon button (a vector drawable) with a spoken label for screen readers. */
    android.widget.ImageView icon(int res, String spoken, float sizeDp, boolean primary, View.OnClickListener l) {
        android.widget.ImageView b = new android.widget.ImageView(c);
        b.setImageResource(res);
        b.setImageTintList(ColorStateList.valueOf(primary ? ON_ACCENT : ACCENT));
        b.setScaleType(android.widget.ImageView.ScaleType.CENTER_INSIDE);
        int pad = dp(sizeDp * 0.24f);
        b.setPadding(pad, pad, pad, pad);
        b.setContentDescription(spoken);
        b.setLayoutParams(new LinearLayout.LayoutParams(dp(sizeDp), dp(sizeDp)));
        b.setBackground(new RippleDrawable(ColorStateList.valueOf(0x33000000),
                round(primary ? ACCENT : ACCENT_SOFT, dp(sizeDp / 2)), null));
        b.setOnClickListener(l);
        b.setFocusable(true);
        return b;
    }

    /** A selectable chip; select(chip, on) restyles it. */
    TextView chip(String s, View.OnClickListener l) {
        TextView b = text(s, 13.5f, INK, true);
        b.setGravity(Gravity.CENTER);
        b.setMinHeight(dp(40));
        b.setPadding(dp(14), dp(7), dp(14), dp(7));
        b.setOnClickListener(l);
        b.setFocusable(true);
        select(b, false);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.rightMargin = dp(8);
        b.setLayoutParams(lp);
        return b;
    }

    void select(TextView chip, boolean on) {
        GradientDrawable g = round(on ? ACCENT_SOFT : CARD, dp(16));
        g.setStroke(dp(1), on ? ACCENT : LINE);
        chip.setBackground(new RippleDrawable(ColorStateList.valueOf(0x22000000), g, null));
        chip.setTextColor(on ? ACCENT : INK);
        chip.setSelected(on);
    }

    /** A settings row: title, a secondary line (row.getTag() is that TextView, for updates), and an optional end view. */
    LinearLayout settingRow(String title, String sub, View end, View.OnClickListener tap) {
        LinearLayout r = row();
        r.setMinimumHeight(dp(56));
        r.setPadding(0, dp(9), 0, dp(9));
        LinearLayout texts = column();
        texts.addView(text(title, 15.5f, INK, true));
        TextView s = text(sub, 13, MUTED, false);
        s.setMaxLines(2);
        s.setEllipsize(TextUtils.TruncateAt.END);
        texts.addView(s);
        r.addView(texts, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        if (end != null) r.addView(end);
        if (tap != null) {
            r.setOnClickListener(tap);
            r.setFocusable(true);
            r.setBackground(new RippleDrawable(ColorStateList.valueOf(0x14000000), null, round(0xFFFFFFFF, dp(8))));
        }
        r.setTag(s);
        return r;
    }

    Switch toggle(boolean on) {
        Switch s = new Switch(c);
        s.setChecked(on);
        s.setMinHeight(dp(48));
        return s;
    }

    TextView chevron() { return text("›", 22, MUTED, false); }

    View divider() {
        View v = new View(c);
        v.setBackgroundColor(LINE);
        v.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, Math.max(1, dp(0.7f))));
        return v;
    }

    /** A thin progress bar (0..1) drawn as two rounded blocks. */
    View progress(float p) {
        LinearLayout track = row();
        track.setBackground(round(LINE, dp(3)));
        View fill = new View(c);
        fill.setBackground(round(ACCENT, dp(3)));
        track.addView(fill, new LinearLayout.LayoutParams(0, dp(6), Math.max(0.0001f, p)));
        track.addView(new View(c), new LinearLayout.LayoutParams(0, dp(6), Math.max(0.0001f, 1 - p)));
        return track;
    }

    static LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    /** For an empty View that pushes the rest of a row to the end (zero height: a bare View fills any height it is offered). */
    static LinearLayout.LayoutParams spacer() { return new LinearLayout.LayoutParams(0, 0, 1); }

    static LinearLayout.LayoutParams weight(float w) {
        return new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, w);
    }
}
