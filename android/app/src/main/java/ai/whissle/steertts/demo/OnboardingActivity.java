package ai.whissle.steertts.demo;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

/**
 * The first start, three pages: what SteerTTS is (with a live sample), making it the phone's voice (one tap to the
 * system setting, a tick when done), and what the assistant can do (each feature asks for its permission later, on the
 * Assistant tab, with its own explanation). Shown once; Settings can show it again.
 */
public class OnboardingActivity extends Activity {
    static boolean done(Context c) { return c.getSharedPreferences("ui", MODE_PRIVATE).getBoolean("onboarded", false); }

    private Ui u;
    private int page;
    private FrameLayout body;
    private TextView next, dots;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        Ui.init(this);
        u = new Ui(this);
        LinearLayout root = u.column();
        root.setBackgroundColor(Ui.BG);
        root.setPadding(u.dp(24), u.dp(24), u.dp(24), u.dp(24));
        LinearLayout top = u.row();
        top.addView(new View(this), Ui.spacer());
        TextView skip = u.text("Skip", 15, Ui.MUTED, true);
        skip.setPadding(u.dp(12), u.dp(12), u.dp(4), u.dp(12));
        skip.setOnClickListener(v -> finishTour(null));
        top.addView(skip);
        root.addView(top);
        body = new FrameLayout(this);
        root.addView(body, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        dots = u.text("", 18, Ui.ACCENT, true);
        dots.setGravity(Gravity.CENTER);
        root.addView(dots, Ui.matchWrap());
        next = u.pill("Next", true, v -> {
            if (page < 2) show(page + 1);
            else finishTour("assistant");
        });
        LinearLayout.LayoutParams lp = Ui.matchWrap();
        lp.topMargin = u.dp(14);
        root.addView(next, lp);
        setContentView(root);
        if (Build.VERSION.SDK_INT >= 30)
            root.setOnApplyWindowInsetsListener((v, ins) -> {
                android.graphics.Insets bars = ins.getInsets(android.view.WindowInsets.Type.systemBars());
                v.setPadding(bars.left + u.dp(24), bars.top + u.dp(12), bars.right + u.dp(24), bars.bottom + u.dp(16));
                return android.view.WindowInsets.CONSUMED;
            });
        show(0);
    }

    @Override protected void onResume() {
        super.onResume();
        if (page == 1) show(1);   // back from the system setting: show the tick
    }

    private void show(int p) {
        page = p;
        body.removeAllViews();
        LinearLayout c = u.column();
        c.setGravity(Gravity.CENTER_HORIZONTAL);
        ImageView mark = new ImageView(this);
        mark.setImageResource(R.mipmap.ic_launcher);
        LinearLayout.LayoutParams ml = new LinearLayout.LayoutParams(u.dp(112), u.dp(112));
        ml.topMargin = u.dp(36);
        ml.bottomMargin = u.dp(28);
        c.addView(mark, ml);
        if (p == 0) {
            title(c, "Meet SteerTTS");
            text(c, "An expressive voice that runs entirely on your phone. No account, no internet, nothing leaves the device.");
            TextView hear = u.pill("▶  Hear it", false, v -> Announcer.get(this).say(
                    "Hi! I'm " + Voices.get(this).nameOf(Style.load(this).voice, null) + ", one of two hundred voices in SteerTTS. I can read your messages, books and articles, "
                            + "and I sound the way you want me to.", "onboarding", true, null));
            LinearLayout.LayoutParams lp = Ui.matchWrap();
            lp.topMargin = u.dp(28);
            c.addView(hear, lp);
        } else if (p == 1) {
            title(c, "Make it your phone's voice");
            text(c, "Set SteerTTS as Android's text-to-speech engine and every app that speaks (navigation, screen readers, "
                    + "readers) uses this voice.");
            boolean on = SettingsScreen.isPhoneVoice(this);
            TextView b = u.pill(on ? "✓  SteerTTS is your phone's voice" : "Open the setting", !on, v -> SettingsScreen.openTtsSettings(this));
            LinearLayout.LayoutParams lp = Ui.matchWrap();
            lp.topMargin = u.dp(28);
            c.addView(b, lp);
            if (!on) text(c, "Choose “Preferred engine”, then SteerTTS.");
        } else {
            title(c, "It speaks up for you");
            text(c, "Read new messages aloud when you have headphones in. Say who's calling. Remind you before meetings. "
                    + "Brief you each morning on your day, your sleep and your steps. Read any article or e-book.");
            text(c, "You choose what it may do on the Assistant tab. Each feature explains what it reads before asking.");
        }
        body.addView(c);
        dots.setText(p == 0 ? "●  ○  ○" : p == 1 ? "○  ●  ○" : "○  ○  ●");
        dots.setContentDescription("Page " + (p + 1) + " of 3");
        next.setText(p == 2 ? "Set up the assistant" : "Next");
    }

    private void title(LinearLayout c, String s) {
        TextView t = u.text(s, 28, Ui.INK, true);
        t.setGravity(Gravity.CENTER);
        c.addView(t, Ui.matchWrap());
    }

    private void text(LinearLayout c, String s) {
        TextView t = u.text(s, 16, Ui.MUTED, false);
        t.setGravity(Gravity.CENTER);
        t.setLineSpacing(0, 1.2f);
        t.setPadding(0, u.dp(12), 0, 0);
        c.addView(t, Ui.matchWrap());
    }

    private void finishTour(String tab) {
        getSharedPreferences("ui", MODE_PRIVATE).edit().putBoolean("onboarded", true).apply();
        if (tab == null) {   // skipped: back to whatever is underneath (it may be a document opened from another app)
            finish();
            return;
        }
        Intent i = new Intent(this, MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        if (tab != null) i.putExtra("tab", tab);
        startActivity(i);
        finish();
    }
}
