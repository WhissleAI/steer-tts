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
        l.setBackground(round(CARD, dp(22)));
        l.setPadding(dp(16), dp(14), dp(16), dp(14));
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

    // ---------------------------------------------------------------- icons, sections, tiles
    /** A tinted vector icon. */
    android.widget.ImageView img(int res, int color, float sizeDp) {
        android.widget.ImageView v = new android.widget.ImageView(c);
        v.setImageResource(res);
        v.setImageTintList(ColorStateList.valueOf(color));
        v.setLayoutParams(new LinearLayout.LayoutParams(dp(sizeDp), dp(sizeDp)));
        v.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        return v;
    }

    /** An icon in a filled circle. */
    android.widget.FrameLayout badge(int res, int bg, int fg, float sizeDp) {
        android.widget.FrameLayout f = new android.widget.FrameLayout(c);
        f.setBackground(round(bg, dp(sizeDp / 2)));
        android.widget.ImageView i = img(res, fg, sizeDp * 0.5f);
        f.addView(i, new android.widget.FrameLayout.LayoutParams(dp(sizeDp * 0.5f), dp(sizeDp * 0.5f), Gravity.CENTER));
        f.setLayoutParams(new LinearLayout.LayoutParams(dp(sizeDp), dp(sizeDp)));
        return f;
    }

    /** A screen's top: its title, and the settings gear at the end. */
    LinearLayout topBar(String title, View.OnClickListener settings) {
        LinearLayout r = row();
        r.setPadding(0, dp(4), 0, dp(6));
        TextView t = text(title, 30, INK, true);
        if (Build.VERSION.SDK_INT >= 28) t.setAccessibilityHeading(true);
        r.addView(t, weight(1));
        if (settings != null) {
            android.widget.ImageView g = img(R.drawable.ic_settings, MUTED, 48);
            int pad = dp(12);
            g.setPadding(pad, pad, pad, pad);
            g.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_YES);
            g.setContentDescription("Settings");
            g.setBackground(new RippleDrawable(ColorStateList.valueOf(0x22000000), null, round(0xFFFFFFFF, dp(24))));
            g.setOnClickListener(settings);
            r.addView(g);
        }
        return r;
    }

    /** A section title: calmer than a coloured label, it reads as a heading. */
    TextView section(String s) {
        TextView t = text(s, 17, INK, true);
        t.setPadding(dp(4), dp(22), 0, dp(4));
        if (Build.VERSION.SDK_INT >= 28) t.setAccessibilityHeading(true);
        return t;
    }

    /** A round action with its label underneath (Paste, File, Link, Type). */
    LinearLayout action(int res, String label, View.OnClickListener l) {
        LinearLayout col = column();
        col.setGravity(Gravity.CENTER_HORIZONTAL);
        android.widget.FrameLayout b = badge(res, ACCENT_SOFT, ACCENT, 58);
        col.addView(b);
        TextView t = text(label, 13, INK, true);
        t.setPadding(0, dp(6), 0, 0);
        t.setGravity(Gravity.CENTER_HORIZONTAL);
        col.addView(t);
        col.setOnClickListener(l);
        col.setFocusable(true);
        col.setContentDescription(label);
        col.setBackground(new RippleDrawable(ColorStateList.valueOf(0x18000000), null, round(0xFFFFFFFF, dp(16))));
        col.setPadding(dp(4), dp(6), dp(4), dp(6));
        return col;
    }

    /** A feature tile (Assistant): icon, title, a status line; Tile.set() shows on / off. */
    static final class Tile {
        final LinearLayout root;
        final TextView title, status, state;
        final android.widget.FrameLayout icon;
        final Ui u;
        final int res;

        Tile(Ui u, int res, String name) {
            this.u = u;
            this.res = res;
            root = u.column();
            root.setPadding(u.dp(14), u.dp(14), u.dp(14), u.dp(14));
            root.setMinimumHeight(u.dp(132));
            LinearLayout top = u.row();
            icon = u.badge(res, ACCENT_SOFT, ACCENT, 40);
            top.addView(icon);
            top.addView(new View(u.c), spacer());
            state = u.text("", 12, MUTED, true);
            state.setPadding(u.dp(9), u.dp(3), u.dp(9), u.dp(3));
            top.addView(state);
            root.addView(top, matchWrap());
            title = u.text(name, 15.5f, INK, true);
            title.setPadding(0, u.dp(10), 0, u.dp(2));
            root.addView(title);
            status = u.text("", 12.5f, MUTED, false);
            status.setMaxLines(3);
            status.setEllipsize(TextUtils.TruncateAt.END);
            root.addView(status);
            root.setFocusable(true);
        }

        void set(boolean on, String statusText) {
            GradientDrawable g = round(on ? ACCENT_SOFT : CARD, u.dp(22));
            if (!on) g.setStroke(u.dp(1), LINE);
            root.setBackground(new RippleDrawable(ColorStateList.valueOf(0x18000000), g, null));
            ((GradientDrawable) icon.getBackground()).setColor(on ? ACCENT : ACCENT_SOFT);
            ((android.widget.ImageView) icon.getChildAt(0)).setImageTintList(ColorStateList.valueOf(on ? ON_ACCENT : ACCENT));
            state.setText(on ? "On" : "Off");
            state.setTextColor(on ? ON_ACCENT : MUTED);
            state.setBackground(round(on ? ACCENT : LINE, u.dp(10)));
            status.setText(statusText);
            root.setContentDescription(title.getText() + ", " + (on ? "on" : "off") + ". " + statusText);
        }
    }

    /** A little equaliser that moves while something is being said. */
    static final class Bars extends View {
        private final android.graphics.Paint paint = new android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG);
        private final android.animation.ValueAnimator anim = android.animation.ValueAnimator.ofFloat(0, (float) (2 * Math.PI));
        private float phase;

        Bars(Context c, int color) {
            super(c);
            paint.setColor(color);
            anim.setDuration(1100);
            anim.setRepeatCount(android.animation.ValueAnimator.INFINITE);
            anim.setInterpolator(new android.view.animation.LinearInterpolator());
            anim.addUpdateListener(a -> {
                phase = (float) a.getAnimatedValue();
                invalidate();
            });
            setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_NO);
        }

        void play(boolean on) {
            if (on && !anim.isStarted()) anim.start();
            else if (!on) {
                anim.cancel();
                phase = 0;
                invalidate();
            }
        }

        @Override protected void onDraw(android.graphics.Canvas cv) {
            int n = 4;
            float w = getWidth(), h = getHeight(), bw = w / (n * 2 - 1);
            for (int i = 0; i < n; ++i) {
                float a = anim.isStarted() ? 0.35f + 0.65f * (float) Math.abs(Math.sin(phase + i * 0.9)) : 0.35f;
                float bh = h * a, x = i * 2 * bw, y = (h - bh) / 2;
                cv.drawRoundRect(x, y, x + bw, y + bh, bw / 2, bw / 2, paint);
            }
        }
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
