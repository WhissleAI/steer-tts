package ai.whissle.steertts.demo;

import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.Switch;
import android.widget.TextView;

/** The app's look, in code (no AndroidX): colours, cards, text styles, pill buttons, chips and settings rows. */
final class Ui {
    static final int BG = 0xFFF5F4FA, CARD = 0xFFFFFFFF, INK = 0xFF1C1B22, MUTED = 0xFF6E6C7A, LINE = 0xFFE6E4EE,
            ACCENT = 0xFF5B4BDB, ACCENT_SOFT = 0xFFECE9FD, WARN = 0xFFB3261E;

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

    /** A white rounded card with padding, spaced from the one above. */
    LinearLayout card() {
        LinearLayout l = column();
        l.setBackground(round(CARD, dp(20)));
        l.setPadding(dp(16), dp(14), dp(16), dp(14));
        l.setElevation(dp(1));
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

    /** A filled (primary) or tonal pill button. */
    TextView pill(String s, boolean primary, View.OnClickListener l) {
        TextView b = text(s, 15, primary ? 0xFFFFFFFF : ACCENT, true);
        b.setGravity(Gravity.CENTER);
        b.setPadding(dp(18), dp(10), dp(18), dp(10));
        b.setBackground(new RippleDrawable(ColorStateList.valueOf(0x33000000),
                round(primary ? ACCENT : ACCENT_SOFT, dp(24)), null));
        b.setOnClickListener(l);
        return b;
    }

    /** A selectable chip; select(chip, on) restyles it. */
    TextView chip(String s, View.OnClickListener l) {
        TextView b = text(s, 13.5f, INK, true);
        b.setGravity(Gravity.CENTER);
        b.setPadding(dp(14), dp(7), dp(14), dp(7));
        b.setOnClickListener(l);
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
    }

    /** A settings row: title, a secondary line (returned for updates via row.getTag()), and an optional end view. */
    LinearLayout settingRow(String title, String sub, View end, View.OnClickListener tap) {
        LinearLayout r = row();
        r.setPadding(0, dp(9), 0, dp(9));
        LinearLayout texts = column();
        texts.addView(text(title, 15.5f, INK, true));
        TextView s = text(sub, 13, MUTED, false);
        s.setSingleLine(true);
        s.setEllipsize(TextUtils.TruncateAt.END);
        texts.addView(s);
        r.addView(texts, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        if (end != null) r.addView(end);
        if (tap != null) {
            r.setOnClickListener(tap);
            r.setBackground(new RippleDrawable(ColorStateList.valueOf(0x14000000), null, round(0xFFFFFFFF, dp(8))));
        }
        r.setTag(s);
        return r;
    }

    Switch toggle(boolean on) {
        Switch s = new Switch(c);
        s.setChecked(on);
        return s;
    }

    TextView chevron() { return text("›", 22, MUTED, false); }

    View divider() {
        View v = new View(c);
        v.setBackgroundColor(LINE);
        v.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, Math.max(1, dp(0.7f))));
        return v;
    }

    static LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    static LinearLayout.LayoutParams weight(float w) {
        return new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, w);
    }
}
