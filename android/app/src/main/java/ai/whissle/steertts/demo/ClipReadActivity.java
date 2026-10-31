package ai.whissle.steertts.demo;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.os.Bundle;
import android.widget.Toast;

/**
 * "Read copied text" from Quick Settings: no window of its own. Android lets an app read the clipboard only while it
 * has focus, so the clipboard is read when this (invisible) activity gains it; the text (or the article behind a link)
 * goes into the reading list and plays in the background, with the player notification for control.
 */
public class ClipReadActivity extends Activity {
    private boolean done;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
    }

    @Override public void onWindowFocusChanged(boolean focus) {
        super.onWindowFocusChanged(focus);
        if (!focus || done) return;
        done = true;
        ClipboardManager cm = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        ClipData clip = cm.getPrimaryClip();
        CharSequence t = clip != null && clip.getItemCount() > 0 ? clip.getItemAt(0).coerceToText(this) : null;
        if (t == null || t.toString().trim().isEmpty()) {
            Toast.makeText(this, "Nothing copied yet", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }
        String s = t.toString().trim();
        Toast.makeText(this, "Reading what you copied", Toast.LENGTH_SHORT).show();
        android.content.Context app = getApplicationContext();
        new Thread(() -> {
            try {
                Doc d = Article.isLink(s) ? Extract.fromUrl(s) : Extract.fromText(s, null);
                if (!d.sentences.isEmpty()) {
                    String id = d.saveOrExisting(app);
                    Doc open = id.equals(d.id) ? d : Doc.load(app, id);   // a link already in the list: where it was left
                    runOnUiThread(() -> {
                        Narrator n = Narrator.get(app);
                        n.open(open != null ? open : d);
                        n.play();
                    });
                }
            } catch (Exception | OutOfMemoryError e) {
                android.util.Log.e(MainActivity.TAG, "read copied", e);
                String msg = e instanceof Extract.Unsupported ? e.getMessage() : "Couldn't read that";
                runOnUiThread(() -> Toast.makeText(app, msg, Toast.LENGTH_LONG).show());
            }
            runOnUiThread(this::finish);
        }, "steertts-clip").start();
    }
}
