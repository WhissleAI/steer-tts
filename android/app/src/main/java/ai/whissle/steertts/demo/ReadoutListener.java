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
 * the user picked ("Priya on WhatsApp: running late"), when AssistantPrefs.mayReadNow allows, and announces incoming
 * calls by the caller's name from the ringing call's own notification (the dialer's or a VoIP app's), so no phone or
 * contacts permission is needed. Messaging apps repost a conversation's notification with its whole recent history on
 * every message: only messages newer than the last one read, and not read before, are spoken; the user's own replies
 * never. Silent
 * notifications, ongoing ones and group summaries are skipped, and nothing is read in Do Not Disturb.
 */
public class ReadoutListener extends NotificationListenerService {
    private static final int MAX_CHARS = 300, MAX_MESSAGES = 3;
    private static final Set<String> DIALERS = new HashSet<>(java.util.Arrays.asList(
            "com.google.android.dialer", "com.android.dialer", "com.samsung.android.incallui", "com.android.incallui"));

    private final Map<String, Long> lastMessage = new HashMap<>();   // notification key -> newest message time read
    private final Map<String, Set<String>> readMessages = new HashMap<>();   // key -> "sender\ntext" already read
    private final Map<String, String> lastText = new HashMap<>();    // notification key -> text read (no history)
    private final Set<String> callsAnnounced = new HashSet<>();

    @Override public void onListenerConnected() { MeetingAnnouncer.watchCalendar(this); }

    @Override public void onNotificationRemoved(StatusBarNotification sbn) {
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
        String text = describe(sbn.getKey(), n, pkg);
        if (text == null) return;
        prefs.noteSeen(pkg);
        if (!prefs.appOn(pkg) || getCurrentInterruptionFilter() > INTERRUPTION_FILTER_ALL || !prefs.mayReadNow(this))
            return;
        Announcer.get(this).say(text, pkg, false, null);
    }

    /** "Incoming call from Mom" while a call rings (the notification with a full-screen intent), once per call. */
    private void call(StatusBarNotification sbn, Notification n, String pkg) {
        if (n.fullScreenIntent == null || !callsAnnounced.add(sbn.getKey())) return;
        if (!AssistantPrefs.mayAnnounceNow(this)) return;
        CharSequence who = n.extras.getCharSequence(Notification.EXTRA_TITLE);
        String name = who != null && who.length() > 0 ? who.toString() : "an unknown caller";
        String text = DIALERS.contains(pkg) ? "Incoming call from " + name + "."
                : "Incoming " + label(pkg) + " call from " + name + ".";
        Announcer.get(this).say(text, pkg, true, null);
    }

    /** The words to say for a new notification, or null if there is nothing new. */
    private String describe(String key, Notification n, String pkg) {
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
            String who = group && conv != null ? sender + " in " + conv : sender;
            return cap(who + " on " + app + ": " + join(texts));
        }
        CharSequence title = x.getCharSequence(Notification.EXTRA_TITLE);
        CharSequence body = x.getCharSequence(Notification.EXTRA_BIG_TEXT);
        if (body == null) body = x.getCharSequence(Notification.EXTRA_TEXT);
        if (title == null && body == null) return null;
        String said = (title != null ? title : "") + "\n" + (body != null ? body : "");
        if (said.equals(lastText.put(key, said))) return null;   // an update with the same words
        if (title == null) return cap(app + ": " + body);
        if (body == null) return cap(app + ": " + title);
        return cap(title + " on " + app + ": " + body);
    }

    private static CharSequence senderOf(Bundle m) {
        CharSequence s = m.getCharSequence("sender");
        if (s == null && Build.VERSION.SDK_INT >= 28) {
            Person p = m.getParcelable("sender_person");
            if (p != null) s = p.getName();
        }
        return s;
    }

    private static String join(List<String> texts) {
        StringBuilder b = new StringBuilder();
        for (String t : texts) {
            String s = t.trim();
            if (s.isEmpty()) continue;
            if (b.length() > 0) b.append(' ');
            b.append(s);
            char c = s.charAt(s.length() - 1);
            if (c != '.' && c != '!' && c != '?') b.append('.');   // one message = one sentence at least
        }
        return b.toString();
    }

    private static String cap(String s) {
        s = s.replaceAll("\\s+", " ").trim();
        if (s.length() <= MAX_CHARS) return s;
        int cut = s.lastIndexOf(' ', MAX_CHARS);
        return s.substring(0, cut > 0 ? cut : MAX_CHARS) + ", and more.";
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
