package ai.whissle.steertts.demo;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.widget.Toast;

/**
 * "Read with SteerTTS": in any app's text-selection menu (ACTION_PROCESS_TEXT) and share sheet (ACTION_SEND, text or
 * a link). No window: hands the text to ReadAloudService and finishes.
 */
public class ReadAloudActivity extends Activity {
    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        Intent i = getIntent();
        CharSequence text = Intent.ACTION_PROCESS_TEXT.equals(i.getAction())
                ? i.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT) : i.getCharSequenceExtra(Intent.EXTRA_TEXT);
        if (text != null && text.toString().trim().length() > 0) {
            startForegroundService(new Intent(this, ReadAloudService.class)
                    .putExtra(ReadAloudService.EXTRA_TEXT, text.toString()));
            Toast.makeText(this, "Reading aloud", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Nothing to read", Toast.LENGTH_SHORT).show();
        }
        finish();
    }
}
