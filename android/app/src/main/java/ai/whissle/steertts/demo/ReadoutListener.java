package ai.whissle.steertts.demo;

import android.app.Notification;
import android.app.NotificationManager;
import android.app.Person;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.media.AudioManager;
import android.os.BatteryManager;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Parcelable;
import android.service.notification.NotificationListenerService;
import android.service.notification.StatusBarNotification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Notification access (Settings > Notifications > Device & app notifications): reads new messages aloud from the apps
 * the user picked ("Priya on WhatsApp", then her message in its mood and her voice; Readout), when
 * AssistantPrefs.mayReadNow allows; what cannot be read then waits in the Digest for "catch me up", which also runs
 * when headphones connect. Announces incoming
 * calls by the caller's name from the ringing call's own notification (the dialer's or a VoIP app's), so no phone or
 * contacts permission is needed. Messaging apps repost a conversation's notification with its whole recent history on
 * every message: only messages newer than the last one read, and not read before, are spoken; the user's own replies
 * never. Silent
 * notifications, ongoing ones and group summaries are skipped, and nothing is read in Do Not Disturb.
 */
public class ReadoutListener extends NotificationListenerService {
    private static final int MAX_MESSAGES = 3;
    private static final Set<String> DIALERS = new HashSet<>(java.util.Arrays.asList(
            "com.google.android.dialer", "com.android.dialer", "com.samsung.android.incallui", "com.android.incallui"));

    private final Map<String, Long> lastMessage = new HashMap<>();   // notification key -> newest message time read
    private final Map<String, Set<String>> readMessages = new HashMap<>();   // key -> "sender\ntext" already read
    private final Map<String, String> lastText = new HashMap<>();    // notification key -> text read (no history)
    private final Set<String> callsAnnounced = new HashSet<>();

    private android.media.AudioDeviceCallback headphones;
    private long connectedAt;

    @Override public void onListenerConnected() {
        MeetingAnnouncer.watchCalendar(this);
        seed();
        if (batteryWatch == null) watchBattery();
        android.media.AudioManager am = getSystemService(android.media.AudioManager.class);
        connectedAt = android.os.SystemClock.elapsedRealtime();
        headphones = new android.media.AudioDeviceCallback() {
            @Override public void onAudioDevicesAdded(android.media.AudioDeviceInfo[] added) {
                // registering reports the devices already connected: only later arrivals count
                if (android.os.SystemClock.elapsedRealtime() - connectedAt < 2000) return;
                for (android.media.AudioDeviceInfo d : added) if (d.isSink() && personal(d.getType())) {
                    new android.os.Handler(android.os.Looper.getMainLooper()).postDelayed(() -> catchUp(ReadoutListener.this, false), 2500);
                    return;
                }
            }
        };
        am.registerAudioDeviceCallback(headphones, new android.os.Handler(android.os.Looper.getMainLooper()));
    }

    /**
     * After a restart (an app update, the process killed) the notifications already showing were read before:
     * remember them, so the next repost of a conversation reads only what is new, not its recent history again.
     */
    private void seed() {
        try {
            for (StatusBarNotification sbn : getActiveNotifications()) {
                Bundle x = sbn.getNotification().extras;
                Parcelable[] msgs = x.getParcelableArray(Notification.EXTRA_MESSAGES);
                if (msgs != null) {
                    long newest = 0;
                    Set<String> read = readMessages.computeIfAbsent(sbn.getKey(), k -> new HashSet<>());
                    for (Parcelable p : msgs)
                        if (p instanceof Bundle) {
                            Bundle m = (Bundle) p;
                            newest = Math.max(newest, m.getLong("time"));
                            read.add(senderOf(m) + "\n" + m.getCharSequence("text"));
                        }
                    lastMessage.put(sbn.getKey(), newest);
                } else {
                    CharSequence t = x.getCharSequence(Notification.EXTRA_TITLE), b = x.getCharSequence(Notification.EXTRA_BIG_TEXT);
                    if (b == null) b = x.getCharSequence(Notification.EXTRA_TEXT);
                    lastText.put(sbn.getKey(), (t != null ? t : "") + "\n" + (b != null ? b : ""));
                }
                if (Notification.CATEGORY_CALL.equals(sbn.getNotification().category)) callsAnnounced.add(sbn.getKey());
            }
        } catch (RuntimeException ignored) {   // the listener may be unbound again already
        }
    }

    @Override public void onListenerDisconnected() {
        if (batteryWatch != null) {
            try { unregisterReceiver(batteryWatch); } catch (RuntimeException ignored) { }
            batteryWatch = null;
        }
        if (headphones != null) getSystemService(android.media.AudioManager.class).unregisterAudioDeviceCallback(headphones);
        headphones = null;
    }

    private static boolean personal(int type) {
        return type == android.media.AudioDeviceInfo.TYPE_BLUETOOTH_A2DP || type == android.media.AudioDeviceInfo.TYPE_WIRED_HEADPHONES
                || type == android.media.AudioDeviceInfo.TYPE_WIRED_HEADSET || type == android.media.AudioDeviceInfo.TYPE_USB_HEADSET
                || (Build.VERSION.SDK_INT >= 31 && type == android.media.AudioDeviceInfo.TYPE_BLE_HEADSET);
    }

    /**
     * Reads the messages that waited (Digest). asked false (headphones connecting): only when there is something and
     * the user wants it; asked true (the tile, the app): always, "all caught up" included.
     */
    static void catchUp(android.content.Context c, boolean asked) {
        AssistantPrefs prefs = new AssistantPrefs(c);
        if (!asked && (!prefs.catchUp() || Digest.size() == 0 || !AssistantPrefs.mayAnnounceNow(c))) return;
        Announcer.get(c).say(Digest.take(prefs.personVoices(), Style.load(c).voice), "catch-up", false, null);
    }

    @Override public void onNotificationRemoved(StatusBarNotification sbn) {
        Digest.removeKey(sbn.getKey());
        lastMessage.remove(sbn.getKey());
        readMessages.remove(sbn.getKey());
        lastText.remove(sbn.getKey());
        callsAnnounced.remove(sbn.getKey());
    }

    @Override public void onNotificationPosted(StatusBarNotification sbn, RankingMap ranking) {
        String pkg = sbn.getPackageName();
        if (pkg.equals(getPackageName())) return;
        Notification n = sbn.getNotification();
        AssistantPrefs prefs = new AssistantPrefs(this);
        if (Notification.CATEGORY_CALL.equals(n.category)) {
            Ranking cr = new Ranking();
            // in Do Not Disturb only a call that is allowed to ring (a starred contact, a repeat caller) is announced
            boolean rings = !AssistantPrefs.doNotDisturb(this) || (ranking != null && ranking.getRanking(sbn.getKey(), cr) && cr.matchesInterruptionFilter());
            if (prefs.announceCalls() && rings) call(sbn, n, pkg);
            return;
        }
        if (other(sbn, n, pkg, prefs, ranking)) return;
        if (!prefs.readMessages()) return;
        if ((n.flags & (Notification.FLAG_ONGOING_EVENT | Notification.FLAG_GROUP_SUMMARY
                | Notification.FLAG_FOREGROUND_SERVICE)) != 0) return;
        Ranking r = new Ranking();
        if (ranking != null && ranking.getRanking(sbn.getKey(), r)
                && r.getImportance() < NotificationManager.IMPORTANCE_DEFAULT) return;   // silent: not worth a voice
        Said said = describe(sbn.getKey(), n, pkg, prefs);
        if (said == null || said.speech.isEmpty()) return;
        prefs.noteSeen(pkg);
        if (!prefs.appOn(pkg)) return;
        if (getCurrentInterruptionFilter() > INTERRUPTION_FILTER_ALL || !prefs.mayReadNow(this)) {
            for (String[] it : said.items) Digest.add(sbn.getKey(), it[0], label(pkg), it[1]);   // for "catch me up" later
            return;
        }
        Announcer.get(this).say(said.speech, pkg, false, null);
    }

    /** A readout, and what the digest keeps if it cannot be spoken now ({person, words} per message). */
    private static final class Said {
        final Speech speech;
        final List<String[]> items;

        Said(Speech speech, List<String[]> items) {
            this.speech = speech;
            this.items = items;
        }
    }

    /**
     * Things worth hearing besides messages: missed calls, alarms / timers / reminders, deliveries and rides, email
     * subjects. Returns true when the notification was one of these (spoken, kept for catch-up, or deliberately not read).
     */
    private boolean other(StatusBarNotification sbn, Notification n, String pkg, AssistantPrefs prefs, RankingMap ranking) {
        Bundle x = n.extras;
        boolean alarmLike = Notification.CATEGORY_ALARM.equals(n.category) || Notification.CATEGORY_REMINDER.equals(n.category)
                || Notification.CATEGORY_MISSED_CALL.equals(n.category);
        if (!alarmLike) {   // matched by app: its quiet or ongoing notices ("Upcoming alarm", "Your order", progress) are not news
            Ranking r = new Ranking();
            boolean quiet = ranking != null && ranking.getRanking(sbn.getKey(), r) && r.getImportance() < NotificationManager.IMPORTANCE_DEFAULT;
            boolean listed = AssistantPrefs.REMINDER_APPS.contains(pkg) || AssistantPrefs.DELIVERY_APPS.contains(pkg)
                    || AssistantPrefs.EMAIL_APPS.contains(pkg);
            if (listed && (quiet || (n.flags & (Notification.FLAG_ONGOING_EVENT | Notification.FLAG_FOREGROUND_SERVICE)) != 0)) return true;
        }
        CharSequence ct = x.getCharSequence(Notification.EXTRA_TITLE), cb = x.getCharSequence(Notification.EXTRA_TEXT);
        String title = ct == null ? null : ct.toString(), text = cb == null ? null : cb.toString(), app = label(pkg);
        if ((n.flags & Notification.FLAG_GROUP_SUMMARY) != 0) return false;
        String said = title + "\n" + text;
        Speech sp = null;
        boolean announce = false;   // reminders and missed calls follow the announcement rule, the rest the readout rule
        if (Notification.CATEGORY_MISSED_CALL.equals(n.category) || (DIALERS.contains(pkg) && title != null
                && title.toLowerCase(java.util.Locale.ROOT).startsWith("missed call"))) {
            if (!prefs.announceCalls()) return true;
            String who = DIALERS.contains(pkg) && title != null && title.toLowerCase(java.util.Locale.ROOT).startsWith("missed call") ? text : title;
            sp = Readout.missedCall(who);
            announce = true;
        } else if (Notification.CATEGORY_ALARM.equals(n.category) || Notification.CATEGORY_REMINDER.equals(n.category)
                || AssistantPrefs.REMINDER_APPS.contains(pkg)) {
            if (AssistantPrefs.CALENDAR_APPS.contains(pkg)) return true;   // meetings have their own reminders
            if (!prefs.reminders() || (title == null && text == null)) return true;
            sp = Readout.reminder(title, text, app);
            announce = true;
        } else if (AssistantPrefs.DELIVERY_APPS.contains(pkg)) {
            if (!prefs.deliveries() || (title == null && text == null)) return true;
            sp = Readout.update(title, text, app);
        } else if (AssistantPrefs.EMAIL_APPS.contains(pkg)) {
            if (!prefs.email() || title == null) return true;
            sp = Readout.email(title, text);
        } else {
            return false;
        }
        if (said.equals(lastText.put(sbn.getKey(), said))) return true;   // the same words again (a progress update)
        prefs.noteSeen(pkg);
        boolean now = !AssistantPrefs.doNotDisturb(this) && (announce ? AssistantPrefs.mayAnnounceNow(this) : prefs.mayReadNow(this));
        if (now) Announcer.get(this).say(sp, pkg, false, null);
        else if (!announce) Digest.add(sbn.getKey(), title != null ? title : app, app, text != null ? text : "");
        return true;
    }

    // ---------------------------------------------------------------- battery
    private BroadcastReceiver batteryWatch;
    private int lastLevel = -1;

    private void watchBattery() {
        batteryWatch = new BroadcastReceiver() {
            @Override public void onReceive(Context c, Intent i) {
                int level = i.getIntExtra(BatteryManager.EXTRA_LEVEL, -1), scale = i.getIntExtra(BatteryManager.EXTRA_SCALE, 100);
                if (level < 0 || scale <= 0) return;
                int pct = Math.round(100f * level / scale);
                boolean charging = i.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0;
                int before = lastLevel;
                lastLevel = pct;
                if (before < 0 || !new AssistantPrefs(c).battery() || !AssistantPrefs.mayAnnounceNow(c)) return;
                if (!charging && before > 15 && pct <= 15) Announcer.get(c).say(Readout.battery(pct, false), "battery", false, null);
                else if (charging && before < 100 && pct >= 100) Announcer.get(c).say(Readout.battery(100, true), "battery", false, null);
            }
        };
        registerReceiver(batteryWatch, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
    }

    /** "Incoming call from Mom" while a call rings (the notification with a full-screen intent), once per call. */
    private void call(StatusBarNotification sbn, Notification n, String pkg) {
        if (n.fullScreenIntent == null || !callsAnnounced.add(sbn.getKey())) return;
        CharSequence who = n.extras.getCharSequence(Notification.EXTRA_TITLE);
        String name = who != null && who.length() > 0 ? who.toString() : "an unknown caller";
        int count = Readout.ringCount(name, System.currentTimeMillis());
        AudioManager am = getSystemService(AudioManager.class);
        if (AssistantPrefs.inCall(am) || !(AssistantPrefs.privateOutput(am) || am.getRingerMode() == AudioManager.RINGER_MODE_NORMAL)) return;
        Announcer.get(this).say(Readout.call(name, DIALERS.contains(pkg) ? null : label(pkg), count), pkg, true, null);
    }

    /** What to say for a new notification, or null if there is nothing new. */
    private Said describe(String key, Notification n, String pkg, AssistantPrefs prefs) {
        Bundle x = n.extras;
        String app = label(pkg);
        Parcelable[] msgs = x.getParcelableArray(Notification.EXTRA_MESSAGES);
        if (msgs != null && msgs.length > 0) {
            long last = lastMessage.getOrDefault(key, 0L), newest = last;
            List<String[]> fresh = new ArrayList<>();
            Set<String> read = readMessages.computeIfAbsent(key, k -> new HashSet<>());
            for (Parcelable p : msgs) {
                if (!(p instanceof Bundle)) continue;
                Bundle m = (Bundle) p;
                long t = m.getLong("time");
                CharSequence body = m.getCharSequence("text");
                CharSequence from = senderOf(m);
                if (t <= last || body == null) continue;
                newest = Math.max(newest, t);
                if (from == null || !read.add(from + "\n" + body)) continue;   // the user's own, or read already
                fresh.add(new String[] {from.toString(), body.toString()});
            }
            lastMessage.put(key, newest);
            if (fresh.isEmpty()) return null;
            if (fresh.size() > MAX_MESSAGES) fresh = fresh.subList(fresh.size() - MAX_MESSAGES, fresh.size());
            CharSequence conv = x.getCharSequence(Notification.EXTRA_CONVERSATION_TITLE);
            boolean group = Build.VERSION.SDK_INT >= 28 ? x.getBoolean(Notification.EXTRA_IS_GROUP_CONVERSATION)
                    : conv != null;
            Speech sp = Readout.conversation(fresh, group && conv != null ? conv.toString() : null, app,
                    prefs.personVoices(), Style.load(this).voice);
            return new Said(sp, fresh);
        }
        CharSequence title = x.getCharSequence(Notification.EXTRA_TITLE);
        CharSequence body = x.getCharSequence(Notification.EXTRA_BIG_TEXT);
        if (body == null) body = x.getCharSequence(Notification.EXTRA_TEXT);
        if (title == null && body == null) return null;
        String said = (title != null ? title : "") + "\n" + (body != null ? body : "");
        if (said.equals(lastText.put(key, said))) return null;   // an update with the same words
        String t = title == null ? null : title.toString(), b = body == null ? null : body.toString();
        List<String[]> one = new ArrayList<>();
        one.add(new String[] {t != null ? t : app, b != null ? b : t});
        return new Said(Readout.notice(t, b, app), one);
    }

    private static CharSequence senderOf(Bundle m) {
        CharSequence s = m.getCharSequence("sender");
        if (s == null && Build.VERSION.SDK_INT >= 28) {
            Person p = m.getParcelable("sender_person");
            if (p != null) s = p.getName();
        }
        return s;
    }

    private String label(String pkg) {
        PackageManager pm = getPackageManager();
        try {
            return pm.getApplicationLabel(pm.getApplicationInfo(pkg, 0)).toString();
        } catch (PackageManager.NameNotFoundException e) {
            return "your phone";
        }
    }
}
