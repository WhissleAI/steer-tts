package ai.whissle.steertts.demo;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;

/** The sentence Android's text-to-speech settings play for "Play" / "Listen to an example" (English only). */
public class GetSampleText extends Activity {
    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        setResult(TextToSpeech.LANG_AVAILABLE, new Intent().putExtra(TextToSpeech.Engine.EXTRA_SAMPLE_TEXT,
                "Hi, I'm SteerTTS, your phone's voice. Every word you hear is made right here on this device."));
        finish();
    }
}
