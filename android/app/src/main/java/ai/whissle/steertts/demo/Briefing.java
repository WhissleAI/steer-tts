package ai.whissle.steertts.demo;

import android.Manifest;
import android.app.AlarmManager;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.os.Build;
import android.os.OutcomeReceiver;
import android.provider.CalendarContract;
import android.util.Log;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * A spoken daily briefing: a greeting, last night's sleep, steps (today and yesterday) and resting heart rate from
 * Health Connect (Android 14+, read-only, each part only if permitted and present), and today's calendar. On demand,
 * or every day at a chosen time (an alarm; this class is also its receiver). Health data is read on the phone, spoken,
 * and never stored or sent.
 */
public class Briefing extends BroadcastReceiver {
    static final String[] HEALTH_PERMISSIONS = Build.VERSION.SDK_INT >= 34 ? new String[] {
            android.health.connect.HealthPermissions.READ_STEPS, android.health.connect.HealthPermissions.READ_SLEEP,
            android.health.connect.HealthPermissions.READ_RESTING_HEART_RATE,
            android.health.connect.HealthPermissions.READ_HEART_RATE} : new String[0];
    private static final String TAG = "SteerTTS", ACTION = "ai.whissle.steertts.BRIEFING";
    private static final Executor HEALTH = Executors.newSingleThreadExecutor();   // Health Connect answers, shared

    // ---------------------------------------------------------------- settings
    static SharedPreferences prefs(Context c) { return c.getSharedPreferences("briefing", Context.MODE_PRIVATE); }
    static boolean daily(Context c) { return prefs(c).getBoolean("daily", false); }
    static int minuteOfDay(Context c) { return prefs(c).getInt("at", 7 * 60 + 30); }

    static void setDaily(Context c, boolean on, int minuteOfDay) {
        prefs(c).edit().putBoolean("daily", on).putInt("at", minuteOfDay).apply();
        schedule(c);
    }

    static boolean healthAvailable() { return Build.VERSION.SDK_INT >= 34; }

    static boolean healthGranted(Context c) {
        if (!healthAvailable()) return false;
        for (String p : HEALTH_PERMISSIONS)
            if (c.checkSelfPermission(p) == PackageManager.PERMISSION_GRANTED) return true;
        return false;
    }

    // ---------------------------------------------------------------- the words
    static final Expression.Mood RESTED = new Expression.Mood("rested", "contentment", "a good night: warm", "f0_iqr", 0.6, "h1h2", 0.5);
    static final Expression.Mood SHORT_NIGHT = new Expression.Mood("short night", "sympathy", "a short night: gentle", "rate", -0.3, "loud", -0.3, "f0_iqr", -0.2);
    static final Expression.Mood ACTIVE = new Expression.Mood("active", "pride", "an active day: upbeat", "f0_iqr", 0.9, "f0_med", 0.4, "rate", 0.2);

    /**
     * The briefing, each line in its own mood: the greeting bright in the morning and calm in the evening, sleep warm
     * after a good night and gentle after a short one, steps upbeat after an active day, the calendar clear and brisk;
     * then the messages that waited overnight (the Digest). Blocks up to a few seconds while Health Connect answers.
     */
    static Speech compose(Context c) {
        Speech sp = new Speech();
        int h = LocalTime.now().getHour();
        if (h < 12) sp.add("Good morning.", Expression.HAPPY, null, 250);
        else if (h < 18) sp.add("Good afternoon.", null, null, 250);
        else sp.add("Good evening.", Expression.CALM, null, 250);
        int before = sp.parts.size();
        if (healthGranted(c)) health(c, sp);
        String cal = calendar(c);
        if (cal != null) sp.add(cal, cal.contains("clear") ? Expression.CALM : Expression.CONFIDENT, null, 250);
        AssistantPrefs prefs = new AssistantPrefs(c);
        if (Digest.size() > 0) for (Speech.Part p : Digest.take(prefs.personVoices(), Style.load(c).voice).parts) sp.parts.add(p);
        if (sp.parts.size() == before)
            sp.add("There's nothing new to tell you. Connect Health Connect or your calendar on the Assistant tab to hear more.");
        return sp;
    }

    private static void health(Context c, Speech sp) {
        if (Build.VERSION.SDK_INT < 34) return;
        android.health.connect.HealthConnectManager hc = c.getSystemService(android.health.connect.HealthConnectManager.class);
        if (hc == null) return;
        ZoneId z = ZoneId.systemDefault();
        Instant now = Instant.now(), today = LocalDate.now().atStartOfDay(z).toInstant(), yesterday = today.minus(Duration.ofDays(1));
        Long sleep = agg(c, hc, android.health.connect.datatypes.SleepSessionRecord.SLEEP_DURATION_TOTAL,
                LocalDateTime.of(LocalDate.now().minusDays(1), LocalTime.of(18, 0)).atZone(z).toInstant(), now,
                android.health.connect.HealthPermissions.READ_SLEEP);
        if (sleep != null && sleep > 30 * 60_000L) {
            long m = sleep / 60_000L;
            if (m >= 7 * 60) sp.add("You slept " + hours(m) + " last night. Nicely done.", RESTED, null, 200);
            else if (m < 6 * 60) sp.add("You only slept " + hours(m) + " last night. Go easy on yourself today.", SHORT_NIGHT, null, 200);
            else sp.add("Last night you slept " + hours(m) + ".", null, null, 200);
        }
        Long stepsToday = agg(c, hc, android.health.connect.datatypes.StepsRecord.STEPS_COUNT_TOTAL, today, now,
                android.health.connect.HealthPermissions.READ_STEPS);
        Long stepsYesterday = agg(c, hc, android.health.connect.datatypes.StepsRecord.STEPS_COUNT_TOTAL, yesterday, today,
                android.health.connect.HealthPermissions.READ_STEPS);
        if (stepsYesterday != null && stepsYesterday >= 8000)
            sp.add(String.format(Locale.US, "Yesterday you walked %,d steps. Great job.", stepsYesterday), ACTIVE, null, 200);
        else if (stepsYesterday != null && stepsYesterday > 0)
            sp.add(String.format(Locale.US, "Yesterday you walked %,d steps.", stepsYesterday), null, null, 200);
        if (stepsToday != null && stepsToday >= 500)
            sp.add(String.format(Locale.US, "So far today, %,d.", stepsToday), null, null, 200);
        Long rhr = agg(c, hc, android.health.connect.datatypes.RestingHeartRateRecord.BPM_AVG, now.minus(Duration.ofDays(2)), now,
                android.health.connect.HealthPermissions.READ_RESTING_HEART_RATE);
        if (rhr != null && rhr > 0) sp.add("Your resting heart rate is " + rhr + " beats per minute.", null, null, 200);
    }

    private static String hours(long minutes) {
        long h = minutes / 60, m = minutes % 60;
        String hs = h == 1 ? "1 hour" : h + " hours";
        return m == 0 ? hs : hs + " and " + (m == 1 ? "1 minute" : m + " minutes");
    }

    /** One aggregate from Health Connect, or null (no permission, no data, error, or no answer within 4 s). */
    private static <T> T agg(Context c, Object hcObj, Object typeObj, Instant from, Instant to, String perm) {
        if (Build.VERSION.SDK_INT < 34 || c.checkSelfPermission(perm) != PackageManager.PERMISSION_GRANTED) return null;
        android.health.connect.HealthConnectManager hc = (android.health.connect.HealthConnectManager) hcObj;
        @SuppressWarnings("unchecked")
        android.health.connect.datatypes.AggregationType<T> type = (android.health.connect.datatypes.AggregationType<T>) typeObj;
        android.health.connect.AggregateRecordsRequest<T> req = new android.health.connect.AggregateRecordsRequest.Builder<T>(
                new android.health.connect.TimeInstantRangeFilter.Builder().setStartTime(from).setEndTime(to).build())
                .addAggregationType(type).build();
        final Object[] out = {null};
        CountDownLatch done = new CountDownLatch(1);
        Executor ex = HEALTH;
        hc.aggregate(req, ex, new OutcomeReceiver<android.health.connect.AggregateRecordsResponse<T>, android.health.connect.HealthConnectException>() {
            @Override public void onResult(android.health.connect.AggregateRecordsResponse<T> r) {
                out[0] = r.get(type);
                done.countDown();
            }
            @Override public void onError(android.health.connect.HealthConnectException e) {
                Log.w(TAG, "health " + perm + ": " + e.getMessage());
                done.countDown();
            }
        });
        try {
            done.await(2500, TimeUnit.MILLISECONDS);
        } catch (InterruptedException ignored) {
        }
        @SuppressWarnings("unchecked") T t = (T) out[0];
        return t;
    }

    /** "You have 3 events today; the first, Design review, is at 10 AM." or null. */
    private static String calendar(Context c) {
        if (c.checkSelfPermission(Manifest.permission.READ_CALENDAR) != PackageManager.PERMISSION_GRANTED) return null;
        long now = System.currentTimeMillis();
        long end = LocalDate.now().plusDays(1).atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli();
        String first = null;
        long firstAt = 0;
        int n = 0;
        try (Cursor cur = CalendarContract.Instances.query(c.getContentResolver(), new String[] {
                CalendarContract.Instances.TITLE, CalendarContract.Instances.BEGIN, CalendarContract.Instances.ALL_DAY,
                CalendarContract.Instances.SELF_ATTENDEE_STATUS}, now, end)) {
            while (cur != null && cur.moveToNext()) {
                if (cur.getInt(2) != 0 || cur.getInt(3) == CalendarContract.Attendees.ATTENDEE_STATUS_DECLINED) continue;
                long b = cur.getLong(1);
                if (b < now) continue;
                n++;
                if (first == null || b < firstAt) {
                    first = cur.getString(0);
                    firstAt = b;
                }
            }
        } catch (Exception e) {
            return null;
        }
        if (n == 0) return "Your calendar is clear for the rest of today.";
        LocalTime t = Instant.ofEpochMilli(firstAt).atZone(ZoneId.systemDefault()).toLocalTime();
        String when = MeetingAnnouncer.spokenTime(t);
        String title = first == null || first.trim().isEmpty() ? "an event" : first.trim();
        return n == 1 ? "You have one more event today: " + title + ", at " + when + "."
                : "You have " + n + " more events today; the next, " + title + ", is at " + when + ".";
    }

    // ---------------------------------------------------------------- playing it
    static void play(Context c) {
        Context app = c.getApplicationContext();
        new Thread(() -> Announcer.get(app).say(compose(app), "briefing", true, null), "steertts-briefing").start();
    }

    @Override public void onReceive(Context c, Intent i) {
        PendingResult r = goAsync();
        Context app = c.getApplicationContext();
        new Thread(() -> {
            try {
                if (ACTION.equals(i.getAction()) && daily(app) && AssistantPrefs.mayAnnounceNow(app)) {
                    CountDownLatch spoken = new CountDownLatch(1);
                    Announcer.get(app).say(compose(app), "briefing", false, spoken::countDown);
                    spoken.await(25, TimeUnit.SECONDS);   // stay alive while it speaks, within the broadcast's limits
                }
            } catch (InterruptedException ignored) {
            } finally {
                schedule(app);
                r.finish();
            }
        }).start();
    }

    private static PendingIntent alarm(Context c) {
        return PendingIntent.getBroadcast(c, 7, new Intent(c, Briefing.class).setAction(ACTION),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
    }

    /** The next daily alarm (or none). */
    static void schedule(Context c) {
        AlarmManager am = (AlarmManager) c.getSystemService(Context.ALARM_SERVICE);
        if (!daily(c)) {
            am.cancel(alarm(c));
            return;
        }
        int m = minuteOfDay(c);
        LocalDateTime next = LocalDate.now().atTime(m / 60, m % 60);
        if (!next.isAfter(LocalDateTime.now())) next = next.plusDays(1);
        long at = next.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
        if (Build.VERSION.SDK_INT < 31 || am.canScheduleExactAlarms()) am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, at, alarm(c));
        else am.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, at, alarm(c));
    }
}
