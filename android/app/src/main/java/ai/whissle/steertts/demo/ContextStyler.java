package ai.whissle.steertts.demo;

import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.media.AudioManager;
import android.os.BatteryManager;

import java.util.Calendar;
import java.util.Locale;
import java.util.regex.Pattern;

/**
 * Adjusts the user's base style to what is being said and the phone's state: alerts faster, louder and brighter;
 * questions rising at the end; good news livelier; navigation and screen-reader speech brisk and clear; late at night,
 * or with the ringer off, softer (not for alerts; never slower: the default voice is already a slow speaker). Adjustments add to the base controls (z-scores), so the
 * user's own settings stay the starting point. The rules lean on loud, f0_med, rate, tilt and h1h2: the controls the
 * model follows reliably (docs/request.md).
 */
final class ContextStyler {
    // other apps' words (system voice): only clear alerts and clear good news; ordinary UI words ("done", "error",
    // "timer", "security") are left alone, they are labels a screen reader reads all day
    private static final Pattern ALERT = Pattern.compile(
            "\\b(urgent|warning|emergency|alarm|critical|danger|fraud alert|low battery|battery (is )?low|wake up|incoming call)\\b");
    private static final Pattern GOOD = Pattern.compile(
            "\\b(congratulations|congrats|great news|happy birthday|well done)\\b");

    static final String MAPS = "com.google.android.apps.maps", WAZE = "com.waze",
            TALKBACK = "com.google.android.marvin.talkback";
    // caller may also be one of the Announcer's sources: "calendar", "read-aloud", or the package a message is from

    private ContextStyler() {}

    /** The style for one utterance. caller: the requesting app's package (null if unknown). */
    static Style decide(Context ctx, Style base, String text, String caller) {
        return decide(ctx, base, text, caller, !TALKBACK.equals(caller));   // a screen reader's labels carry no mood
    }

    /**
     * textCues false: only the situation (the requesting app, time of day, ringer, battery); the words are left to the
     * caller's own moods (the Announcer's parts carry Expression moods, which would otherwise count twice).
     */
    static Style decide(Context ctx, Style base, String text, String caller, boolean textCues) {
        Style s = base.copy();
        StringBuilder why = new StringBuilder();
        if (!base.contextual) {
            s.reason = "your settings (context styling off)";
            return s;
        }
        String t = text.toLowerCase(Locale.ROOT);
        boolean alert = (textCues && (ALERT.matcher(t).find() || text.trim().endsWith("!!")))
                || (caller != null && (caller.contains("deskclock") || caller.contains("alarm")));
        if (alert) {
            s.add("rate", 0.8f);
            s.add("loud", 1.2f);
            s.add("f0_med", 0.5f);
            s.add("tilt", 0.8f);
            s.add("pause", -0.5f);
            note(why, "alert: faster, louder, brighter");
        } else if (textCues && GOOD.matcher(t).find()) {
            s.add("f0_iqr", 1.2f);
            s.add("f0_med", 0.4f);
            s.add("loud", 0.3f);
            note(why, "good news: livelier");
        }
        if (textCues && text.trim().endsWith("?")) {
            s.add("f0_end", 1.5f);
            note(why, "question: rising end");
        }
        if (MAPS.equals(caller) || WAZE.equals(caller)) {
            s.add("rate", 0.3f);
            s.add("loud", 0.6f);
            note(why, "navigation: clear");
        } else if (TALKBACK.equals(caller)) {
            s.add("rate", 1.0f);
            note(why, "screen reader: brisk");
        }
        boolean navigation = MAPS.equals(caller) || WAZE.equals(caller);
        if (!alert && !navigation) {   // directions while driving stay clear at night and on silent
            int hour = hour(ctx);
            AudioManager am = (AudioManager) ctx.getSystemService(Context.AUDIO_SERVICE);
            boolean quietRinger = am != null && am.getRingerMode() != AudioManager.RINGER_MODE_NORMAL;
            if (hour >= 22 || hour < 7) {
                s.add("loud", -1.2f);
                s.add("h1h2", 0.6f);
                note(why, "night: softer");
            } else if (quietRinger) {
                s.add("loud", -0.8f);
                note(why, "ringer off: softer");
            }
            if (lowBattery(ctx)) {
                s.add("rate", -0.3f);
                note(why, "battery low: calmer");
            }
        }
        s.reason = why.length() > 0 ? why.toString() : "neutral: your settings";
        return s;
    }

    /** The hour context styling uses: the real one, or (debug builds only) a demo override from setDemoHour. */
    static int hour(Context ctx) {
        if ((ctx.getApplicationInfo().flags & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0) {
            java.io.File f = new java.io.File(ctx.getFilesDir(), "demo_hour");
            try (java.io.FileInputStream in = new java.io.FileInputStream(f)) {
                int h = Integer.parseInt(new String(in.readAllBytes()).trim());
                if (h >= 0 && h < 24) return h;
            } catch (Exception ignored) {   // no override
            }
        }
        return Calendar.getInstance().get(Calendar.HOUR_OF_DAY);
    }

    /** Demo recordings: act as if it were `hour` o'clock (a file, so the TTS service process sees it too); -1 clears. */
    static void setDemoHour(Context ctx, int hour) {
        java.io.File f = new java.io.File(ctx.getFilesDir(), "demo_hour");
        if (hour < 0) {
            f.delete();
            return;
        }
        try (java.io.FileOutputStream out = new java.io.FileOutputStream(f)) {
            out.write(String.valueOf(hour).getBytes());
        } catch (java.io.IOException ignored) {
        }
    }

    private static boolean lowBattery(Context ctx) {
        Intent b = ctx.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        if (b == null) return false;
        int level = b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1), scale = b.getIntExtra(BatteryManager.EXTRA_SCALE, 100);
        int plugged = b.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0);
        return plugged == 0 && level >= 0 && level * 100 < 15 * scale;
    }

    private static void note(StringBuilder b, String s) { b.append(b.length() > 0 ? "; " : "").append(s); }
}
