package ai.whissle.steertts.demo;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;

import java.util.ArrayList;

/** Answers TextToSpeech.Engine.ACTION_CHECK_TTS_DATA: the model ships in the APK, so English (US) is always there. */
public class CheckVoiceData extends Activity {
    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        ArrayList<String> available = new ArrayList<>();
        available.add("eng-USA");
        setResult(TextToSpeech.Engine.CHECK_VOICE_DATA_PASS,
                new Intent().putStringArrayListExtra(TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES, available)
                        .putStringArrayListExtra(TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES, new ArrayList<>()));
        finish();
    }
}
