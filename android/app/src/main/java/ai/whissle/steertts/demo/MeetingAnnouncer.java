package ai.whissle.steertts.demo;

import android.Manifest;
import android.app.AlarmManager;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.ContentObserver;
import android.database.Cursor;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.CalendarContract;
import android.util.Log;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Announces calendar events a few minutes before they start ("In 5 minutes: standup with Arun, on Google Meet."),
 * from the phone's calendars (READ_CALENDAR). One alarm is kept set: exact at the next announcement time, else a
 * recheck every 30 minutes; it is re-armed after each check, at boot, on app update, when the app opens and, while the
 * notification listener runs, whenever a calendar changes. All-day, declined and cancelled events are skipped.
 */
public class MeetingAnnouncer extends BroadcastReceiver {
    static final String ACTION = "ai.whissle.steertts.demo.MEETINGS";
    private static final String TAG = "SteerTTS";
    private static final long MIN = 60_000, RECHECK = 30 * MIN, DAY = 24 * 60 * MIN;
    private static final String[] COLS = {CalendarContract.Instances.EVENT_ID, CalendarContract.Instances.BEGIN,
            CalendarContract.Instances.TITLE, CalendarContract.Instances.EVENT_LOCATION,
            CalendarContract.Instances.DESCRIPTION, CalendarContract.Instances.ALL_DAY,
            CalendarContract.Instances.SELF_ATTENDEE_STATUS, CalendarContract.Instances.STATUS};
    private static Handler watcher;

    @Override public void onReceive(Context c, Intent i) {
        PendingResult r = goAsync();
        new Thread(() -> {
            try {
                check(c, 0);
            } finally {
                r.finish();
            }
        }).start();
    }

    /**
     * Speaks the events due for announcement and re-arms the alarm. leadMinutes > 0 (adb test hook): announce every
     * event starting within that many minutes, announced before or not. Blocks until spoken (at most 20 s).
     */
    static synchronized void check(Context c, int leadMinutes) {   // one at a time: never two threads announcing one event
        AssistantPrefs prefs = new AssistantPrefs(c);
        if (!prefs.announceMeetings() || !allowed(c)) {
            cancel(c);
            return;
        }
        long now = System.currentTimeMillis(), lead = (leadMinutes > 0 ? leadMinutes : prefs.leadMinutes()) * MIN;
        long next = now + RECHECK;
        boolean exact = false;
        Set<String> done = prefs.announced(), keep = new HashSet<>();
        List<String> due = new ArrayList<>();
        try (Cursor cur = CalendarContract.Instances.query(c.getContentResolver(), COLS, now - MIN,
                now + Math.max(DAY, lead + MIN))) {
            while (cur != null && cur.moveToNext()) {
                long begin = cur.getLong(1);
                String id = cur.getLong(0) + ":" + begin;
                if (done.contains(id)) keep.add(id);
                if (cur.getInt(5) != 0
                        || cur.getInt(6) == CalendarContract.Attendees.ATTENDEE_STATUS_DECLINED
                        || cur.getInt(7) == CalendarContract.Events.STATUS_CANCELED
                        || begin <= now) continue;
                long at = begin - lead;
                if (at <= now + MIN) {
                    if (leadMinutes > 0 || !done.contains(id)) due.add(phrase(cur, begin, now));
                    if (leadMinutes == 0) keep.add(id);   // a test run leaves the real announcement due
                } else if (at < next) {
                    next = at;
                    exact = true;
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "calendar", e);
        }
        prefs.setAnnounced(keep);
        schedule(c, next, exact);
        if (due.isEmpty() || !AssistantPrefs.mayAnnounceNow(c)) return;
        CountDownLatch spoken = new CountDownLatch(1);
        Announcer.get(c).say(String.join(" ", due), "calendar", false, spoken::countDown);
        try {
            spoken.await(20, TimeUnit.SECONDS);
        } catch (InterruptedException ignored) {
        }
    }

    private static String phrase(Cursor cur, long begin, long now) {
        long m = Math.round((begin - now) / (double) MIN);
        String when = m <= 0 ? "Starting now" : m == 1 ? "In 1 minute" : "In " + m + " minutes";
        String title = cur.getString(2) == null || cur.getString(2).trim().isEmpty() ? "a meeting" : cur.getString(2).trim();
        String loc = cur.getString(3) == null ? "" : cur.getString(3).trim();
        String where = (loc + " " + (cur.getString(4) == null ? "" : cur.getString(4))).toLowerCase(Locale.ROOT);
        String place = where.contains("meet.google.com") ? ", on Google Meet"
                : where.contains("zoom.us") ? ", on Zoom"
                : where.contains("teams.microsoft") ? ", on Teams"
                : !loc.isEmpty() && !loc.contains("://") ? ", at " + loc : "";
        return when + ": " + title + place + ".";
    }

    /** "10 AM", "9:55 AM", "noon": a clock time as it is said aloud. */
    static String spokenTime(java.time.LocalTime t) {
        int h = t.getHour(), m = t.getMinute();
        if (h == 12 && m == 0) return "noon";
        if (h == 0 && m == 0) return "midnight";
        int h12 = h % 12 == 0 ? 12 : h % 12;
        return h12 + (m == 0 ? "" : m < 10 ? ":0" + m : ":" + m) + (h < 12 ? " AM" : " PM");
    }

    static boolean allowed(Context c) {
        return c.checkSelfPermission(Manifest.permission.READ_CALENDAR) == PackageManager.PERMISSION_GRANTED;
    }

    private static PendingIntent alarm(Context c) {
        return PendingIntent.getBroadcast(c, 0, new Intent(c, MeetingAnnouncer.class).setAction(ACTION),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
    }

    private static void schedule(Context c, long at, boolean exact) {
        AlarmManager am = (AlarmManager) c.getSystemService(Context.ALARM_SERVICE);
        if (exact && (Build.VERSION.SDK_INT < 31 || am.canScheduleExactAlarms()))
            am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, at, alarm(c));
        else am.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, at, alarm(c));
    }

    static void cancel(Context c) {
        ((AlarmManager) c.getSystemService(Context.ALARM_SERVICE)).cancel(alarm(c));
    }

    /** Re-arms the alarm off the calling thread (the app opened, a setting changed). */
    static void reschedule(Context c) {
        Context app = c.getApplicationContext();
        new Thread(() -> check(app, 0)).start();
    }

    /** Rechecks when any calendar changes (an event added for ten minutes from now), while the process lives. */
    static synchronized void watchCalendar(Context c) {
        if (watcher != null || !allowed(c)) return;
        HandlerThread t = new HandlerThread("steertts-calendar");
        t.start();
        watcher = new Handler(t.getLooper());
        Context app = c.getApplicationContext();
        Runnable recheck = () -> check(app, 0);
        app.getContentResolver().registerContentObserver(CalendarContract.CONTENT_URI, true, new ContentObserver(watcher) {
            @Override public void onChange(boolean self) {
                watcher.removeCallbacks(recheck);
                watcher.postDelayed(recheck, 5000);   // a sync writes many rows: once, after it settles
            }
        });
        watcher.post(recheck);
    }
}
