package ai.whissle.steertts.demo;

import android.app.Notification;
import android.app.NotificationManager;
import android.app.Person;
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

    @Override public void onListenerDisconnected() {
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
            if (prefs.announceCalls()) call(sbn, n, pkg);
            return;
        }
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
            Digest.add(sbn.getKey(), said.person, label(pkg), said.words);   // for "catch me up" later
            return;
        }
        Announcer.get(this).say(said.speech, pkg, false, null);
    }

    /** A readout, and what the digest keeps if it cannot be spoken now. */
    private static final class Said {
        final Speech speech;
        final String person, words;

        Said(Speech speech, String person, String words) {
            this.speech = speech;
            this.person = person;
            this.words = words;
        }
    }

    /** "Incoming call from Mom" while a call rings (the notification with a full-screen intent), once per call. */
    private void call(StatusBarNotification sbn, Notification n, String pkg) {
        if (n.fullScreenIntent == null || !callsAnnounced.add(sbn.getKey())) return;
        CharSequence who = n.extras.getCharSequence(Notification.EXTRA_TITLE);
        String name = who != null && who.length() > 0 ? who.toString() : "an unknown caller";
        int count = Readout.ringCount(name, System.currentTimeMillis());
        if (!AssistantPrefs.mayAnnounceNow(this)) return;
        Announcer.get(this).say(Readout.call(name, DIALERS.contains(pkg) ? null : label(pkg), count), pkg, true, null);
    }

    /** What to say for a new notification, or null if there is nothing new. */
    private Said describe(String key, Notification n, String pkg, AssistantPrefs prefs) {
        Bundle x = n.extras;
        String app = label(pkg);
        Parcelable[] msgs = x.getParcelableArray(Notification.EXTRA_MESSAGES);
        if (msgs != null && msgs.length > 0) {
            long last = lastMessage.getOrDefault(key, 0L), newest = last;
            List<String> texts = new ArrayList<>();
            Set<String> read = readMessages.computeIfAbsent(key, k -> new HashSet<>());
            String sender = null;
            for (Parcelable p : msgs) {
                if (!(p instanceof Bundle)) continue;
                Bundle m = (Bundle) p;
                long t = m.getLong("time");
                CharSequence body = m.getCharSequence("text");
                CharSequence from = senderOf(m);
                if (t <= last || body == null) continue;
                newest = Math.max(newest, t);
                if (from == null || !read.add(from + "\n" + body)) continue;   // the user's own, or read already
                sender = from.toString();
                texts.add(body.toString());
            }
            lastMessage.put(key, newest);
            if (texts.isEmpty()) return null;
            if (texts.size() > MAX_MESSAGES) texts = texts.subList(texts.size() - MAX_MESSAGES, texts.size());
            CharSequence conv = x.getCharSequence(Notification.EXTRA_CONVERSATION_TITLE);
            boolean group = Build.VERSION.SDK_INT >= 28 ? x.getBoolean(Notification.EXTRA_IS_GROUP_CONVERSATION)
                    : conv != null;
            Speech sp = Readout.message(sender, group && conv != null ? conv.toString() : null, app, texts,
                    prefs.personVoices(), Style.load(this).voice);
            return new Said(sp, sender, String.join(" ", texts));
        }
        CharSequence title = x.getCharSequence(Notification.EXTRA_TITLE);
        CharSequence body = x.getCharSequence(Notification.EXTRA_BIG_TEXT);
        if (body == null) body = x.getCharSequence(Notification.EXTRA_TEXT);
        if (title == null && body == null) return null;
        String said = (title != null ? title : "") + "\n" + (body != null ? body : "");
        if (said.equals(lastText.put(key, said))) return null;   // an update with the same words
        String t = title == null ? null : title.toString(), b = body == null ? null : body.toString();
        return new Said(Readout.notice(t, b, app), t != null ? t : app, b != null ? b : t);
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
