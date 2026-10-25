package ai.whissle.steertts.demo;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Toast;

/**
 * "Read with SteerTTS": in any app's text-selection menu (ACTION_PROCESS_TEXT), share sheet (text, a link, or a file)
 * and "Open with" for documents (PDF, EPUB, Word, HTML, Markdown, text). Turns what it was given into a Doc on a
 * background thread, adds it to the reading list and opens the player, playing. No window of its own.
 */
public class ReadAloudActivity extends Activity {
    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        // a see-through window that does not take touches: the app underneath stays usable while the text is prepared
        getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
        if (b != null) return;   // recreated (rotation): the work started the first time
        Intent i = getIntent();
        String action = i.getAction();
        CharSequence text = Intent.ACTION_PROCESS_TEXT.equals(action)
                ? i.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT) : i.getCharSequenceExtra(Intent.EXTRA_TEXT);
        Uri stream = Intent.ACTION_VIEW.equals(action) ? i.getData()
                : Intent.ACTION_SEND.equals(action) ? stream(i) : null;
        String subject = i.getStringExtra(Intent.EXTRA_SUBJECT);
        if (stream == null && (text == null || text.toString().trim().isEmpty())) {
            Toast.makeText(this, "Nothing to read", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }
        Toast.makeText(this, "Getting it ready…", Toast.LENGTH_SHORT).show();
        String mime = i.getType();
        android.content.Context app = getApplicationContext();
        new Thread(() -> {
            String error = null;
            try {
                Doc d;
                if (stream != null) {
                    d = Extract.fromUri(app, stream, mime);
                } else {
                    String s = text.toString().trim();
                    d = Article.isLink(s) ? Extract.fromUrl(s) : Extract.fromText(s, subject);
                }
                if (d.sentences.isEmpty()) throw new Extract.Unsupported("There's no text to read in this one.");
                String id = d.saveOrExisting(app);
                // the player in SteerTTS's own task, not inside the app that shared
                runOnUiThread(() -> startActivity(ReaderActivity.open(this, id, true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)));
            } catch (Extract.Unsupported e) {
                error = e.getMessage();
            } catch (Exception | OutOfMemoryError e) {
                android.util.Log.e(MainActivity.TAG, "read aloud", e);
                error = "Couldn't open that one.";
            }
            String msg = error;
            runOnUiThread(() -> {
                if (msg != null) Toast.makeText(app, msg, Toast.LENGTH_LONG).show();
                finish();
            });
        }, "steertts-extract").start();
    }

    @SuppressWarnings("deprecation")
    private static Uri stream(Intent i) {
        return android.os.Build.VERSION.SDK_INT >= 33 ? i.getParcelableExtra(Intent.EXTRA_STREAM, Uri.class)
                : i.getParcelableExtra(Intent.EXTRA_STREAM);
    }
}
