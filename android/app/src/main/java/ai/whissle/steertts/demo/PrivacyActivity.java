package ai.whissle.steertts.demo;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.widget.LinearLayout;
import android.widget.ScrollView;

/**
 * What SteerTTS does with the data it can read, in plain words. Android shows this screen when the user asks why the
 * app wants Health Connect access (VIEW_PERMISSION_USAGE / HEALTH_PERMISSIONS), and Settings links to it.
 */
public class PrivacyActivity extends Activity {
    static final String POLICY = "https://github.com/WhissleAI/steer-tts/blob/main/android/play/privacy-policy.md";

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        Ui.init(this);
        Ui u = new Ui(this);
        LinearLayout root = u.column();
        root.setPadding(u.dp(20), u.dp(24), u.dp(20), u.dp(24));
        root.addView(u.header("Your data stays on your phone", null));
        String[][] items = {
                {"Health", "For the daily briefing, SteerTTS reads last night's sleep, your steps and resting heart rate "
                        + "from Health Connect. It speaks them and keeps nothing."},
                {"Notifications", "For message readouts and caller announcements, it reads new notifications from the "
                        + "apps you choose, speaks them and keeps nothing."},
                {"Calendar", "For meeting reminders and the briefing, it reads the title, time and place of upcoming "
                        + "events."},
                {"Documents and links", "Files you open are turned into text on the phone. A link you share is "
                        + "downloaded from that website, like a browser would."},
                {"No accounts, no tracking", "SteerTTS has no account, no analytics and no ads, and never sends what "
                        + "it reads anywhere."}};
        for (String[] it : items) {
            LinearLayout card = u.card();
            card.addView(u.text(it[0], 16, Ui.INK, true));
            card.addView(u.text(it[1], 14.5f, Ui.MUTED, false));
            root.addView(card);
        }
        android.widget.TextView full = u.pill("Read the full privacy policy", false,
                v -> startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(POLICY))));
        LinearLayout.LayoutParams lp = Ui.matchWrap();
        lp.topMargin = u.dp(16);
        root.addView(full, lp);
        ScrollView sv = new ScrollView(this);
        sv.setBackgroundColor(Ui.BG);
        sv.addView(root);
        setContentView(sv);
        if (Build.VERSION.SDK_INT >= 30)
            sv.setOnApplyWindowInsetsListener((v, ins) -> {
                android.graphics.Insets bars = ins.getInsets(android.view.WindowInsets.Type.systemBars());
                v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
                return android.view.WindowInsets.CONSUMED;
            });
    }
}
