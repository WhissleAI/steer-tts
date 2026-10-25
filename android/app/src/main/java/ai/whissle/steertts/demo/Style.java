package ai.whissle.steertts.demo;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.Locale;

/**
 * A delivery style: voice, style tag and the 12 controls (z-scores against the voice's own neutral speech, SteerTTS.CONTROLS
 * order; 0 = left to the model). The user's base style is saved as a small JSON file, which the TTS service (its own
 * process) reads on every request, so a change in the app applies to the next system utterance.
 */
final class Style {
    static final float LIMIT = 3f;   // beyond +-3 is outside the model's well-tested range

    String voice;                    // a voice name, "female" / "male", or null for the female default
    String tag = "none";
    final float[] z = new float[SteerTTS.CONTROLS.length];
    boolean contextual = true;       // let ContextStyler adjust system utterances
    String reason = "";              // why a context style looks the way it does (shown in the app)

    Style copy() {
        Style s = new Style();
        s.voice = voice;
        s.tag = tag;
        System.arraycopy(z, 0, s.z, 0, z.length);
        s.contextual = contextual;
        s.reason = reason;
        return s;
    }

    void add(String control, float dz) {
        for (int i = 0; i < z.length; ++i)
            if (SteerTTS.CONTROLS[i].equals(control)) z[i] = clamp(z[i] + dz);
    }

    static float clamp(float v) { return Math.max(-LIMIT, Math.min(LIMIT, v)); }

    /** "rate=1.20,loud=-0.50" with the controls that are set, or null. */
    String ctrl() {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < z.length; ++i)
            if (Math.abs(z[i]) >= 0.05f)
                b.append(b.length() > 0 ? "," : "").append(SteerTTS.CONTROLS[i]).append('=')
                        .append(String.format(Locale.US, "%.2f", z[i]));
        return b.length() > 0 ? b.toString() : null;
    }

    String tagOrNull() { return tag == null || tag.equals("none") ? null : tag; }

    JSONObject toJson() {
        try {
            JSONObject o = new JSONObject();
            o.put("voice", voice == null ? JSONObject.NULL : voice);
            o.put("tag", tag);
            JSONArray a = new JSONArray();
            for (float v : z) a.put((double) v);
            o.put("z", a);
            o.put("contextual", contextual);
            o.put("reason", reason);
            return o;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    static Style fromJson(JSONObject o) {
        Style s = new Style();
        s.voice = o.isNull("voice") ? null : o.optString("voice", null);
        s.tag = o.optString("tag", "none");
        JSONArray a = o.optJSONArray("z");
        for (int i = 0; a != null && i < Math.min(a.length(), s.z.length); ++i) s.z[i] = clamp((float) a.optDouble(i, 0));
        s.contextual = o.optBoolean("contextual", true);
        s.reason = o.optString("reason", "");
        return s;
    }

    private static File file(Context c) { return new File(c.getFilesDir(), "style.json"); }

    static Style load(Context c) {
        try (FileInputStream in = new FileInputStream(file(c))) {
            return fromJson(new JSONObject(new String(in.readAllBytes())));
        } catch (Exception e) {
            return new Style();
        }
    }

    void save(Context c) {
        File tmp = new File(c.getFilesDir(), "style.json.tmp");
        try (FileOutputStream out = new FileOutputStream(tmp)) {
            out.write(toJson().toString().getBytes());
        } catch (Exception e) {
            return;
        }
        tmp.renameTo(file(c));   // atomic: the service never reads half a file
    }
}
