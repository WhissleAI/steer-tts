package ai.whissle.steertts.demo;

import android.content.Context;
import android.content.SharedPreferences;
import android.media.AudioDeviceInfo;
import android.media.AudioManager;
import android.os.Build;
import android.os.PowerManager;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/** The assistant's settings (main process only), and whether speaking out loud is fine right now. */
final class AssistantPrefs {
    /** When message readouts may play. */
    static final int PRIVATE = 0, PRIVATE_OR_SCREEN_OFF = 1, ANY_TIME = 2;
    static final String[] WHEN_LABELS = {"Headphones or car only", "Headphones, car, or screen off",
            "Any time (not with the ringer off)"};
    static final String[] WHEN_SHORT = {"Headphones or car", "Headphones, car or screen off", "Any time"};
    static final int[] LEAD_MINUTES = {1, 2, 5, 10, 15};

    /** Read by default: messaging apps. Any other app that posts a message-like notification can be switched on. */
    static final Set<String> MESSAGING = new HashSet<>(Arrays.asList(
            "com.google.android.apps.messaging", "com.whatsapp", "com.whatsapp.w4b", "org.telegram.messenger",
            "org.thoughtcrime.securesms", "com.Slack", "com.facebook.orca", "com.instagram.android", "com.discord",
            "com.microsoft.teams", "com.google.android.apps.dynamite", "com.snapchat.android"));

    private final SharedPreferences p;

    AssistantPrefs(Context c) { p = c.getSharedPreferences("assistant", Context.MODE_PRIVATE); }

    boolean readMessages() { return p.getBoolean("read_messages", true); }
    void setReadMessages(boolean on) { p.edit().putBoolean("read_messages", on).apply(); }

    int when() { return p.getInt("when", PRIVATE); }
    void setWhen(int w) { p.edit().putInt("when", w).apply(); }

    boolean announceCalls() { return p.getBoolean("calls", true); }
    void setAnnounceCalls(boolean on) { p.edit().putBoolean("calls", on).apply(); }

    boolean announceMeetings() { return p.getBoolean("meetings", true); }
    void setAnnounceMeetings(boolean on) { p.edit().putBoolean("meetings", on).apply(); }

    boolean reminders() { return p.getBoolean("reminders", true); }
    void setReminders(boolean on) { p.edit().putBoolean("reminders", on).apply(); }
    boolean deliveries() { return p.getBoolean("deliveries", true); }
    void setDeliveries(boolean on) { p.edit().putBoolean("deliveries", on).apply(); }
    boolean email() { return p.getBoolean("email", false); }
    void setEmail(boolean on) { p.edit().putBoolean("email", on).apply(); }
    boolean battery() { return p.getBoolean("battery", true); }
    void setBattery(boolean on) { p.edit().putBoolean("battery", on).apply(); }

    /** Apps whose notifications are reminders, alarms and timers (calendars are left to the meeting reminders). */
    static final Set<String> REMINDER_APPS = new HashSet<>(Arrays.asList(
            "com.google.android.deskclock", "com.android.deskclock", "com.sec.android.app.clockpackage",
            "com.google.android.apps.tasks", "com.google.android.keep", "com.todoist", "com.microsoft.todos",
            "com.ticktick.task", "com.anydo"));
    /** Deliveries, food and rides. */
    static final Set<String> DELIVERY_APPS = new HashSet<>(Arrays.asList(
            "com.ubercab", "com.ubercab.eats", "me.lyft.android", "com.dd.doordash", "com.instacart.client",
            "com.grubhub.android", "com.amazon.mShop.android.shopping", "in.swiggy.android", "com.application.zomato",
            "com.olacabs.customer", "com.fedex.ida.android", "com.ups.mobile.android", "com.deliveroo.orderapp",
            "com.global.foodpanda.android", "com.grabtaxi.passenger", "ee.mtakso.client", "com.postmates.android"));
    static final Set<String> EMAIL_APPS = new HashSet<>(Arrays.asList(
            "com.google.android.gm", "com.microsoft.office.outlook", "com.yahoo.mobile.client.android.mail",
            "com.samsung.android.email.provider", "ch.protonmail.android", "com.readdle.spark"));
    static final Set<String> CALENDAR_APPS = new HashSet<>(Arrays.asList(
            "com.google.android.calendar", "com.samsung.android.calendar", "com.microsoft.office.outlook.calendar"));

    /** Each person's messages in a voice of their own (Voices.forPerson). */
    boolean personVoices() { return p.getBoolean("person_voices", true); }
    void setPersonVoices(boolean on) { p.edit().putBoolean("person_voices", on).apply(); }

    /** Messages that arrived while they could not be read out are offered when headphones connect. */
    boolean catchUp() { return p.getBoolean("catch_up", true); }
    void setCatchUp(boolean on) { p.edit().putBoolean("catch_up", on).apply(); }

    int leadMinutes() { return p.getInt("lead", 5); }
    void setLeadMinutes(int m) { p.edit().putInt("lead", m).apply(); }

    Set<String> appsOn() { return new HashSet<>(p.getStringSet("apps_on", MESSAGING)); }
    void setAppsOn(Set<String> s) { p.edit().putStringSet("apps_on", new HashSet<>(s)).apply(); }
    boolean appOn(String pkg) { return p.getStringSet("apps_on", MESSAGING).contains(pkg); }

    /** Apps seen posting something readable, offered in the app list. */
    Set<String> seen() { return new HashSet<>(p.getStringSet("seen", new HashSet<>())); }
    void noteSeen(String pkg) {
        Set<String> s = p.getStringSet("seen", new HashSet<>());
        if (s.contains(pkg)) return;
        Set<String> n = new HashSet<>(s);
        n.add(pkg);
        p.edit().putStringSet("seen", n).apply();
    }

    /** Meetings already announced, as "eventId:begin". */
    Set<String> announced() { return new HashSet<>(p.getStringSet("announced", new HashSet<>())); }
    void setAnnounced(Set<String> s) { p.edit().putStringSet("announced", s).apply(); }

    // ------------------------------------------------------------------ the phone's state

    /**
     * Speech would be heard by the user only: headphones, a headset, a car. On Android 13+ the route the assistant's
     * audio would actually take is checked (a connected but unused headset does not count); a Bluetooth speaker that
     * reports itself as one is never private. (Classic Bluetooth speakers and cars both report as A2DP and cannot be
     * told apart without the Bluetooth permission; they count as private.)
     */
    static boolean privateOutput(AudioManager am) {
        AudioDeviceInfo[] devices;
        if (Build.VERSION.SDK_INT >= 33) {
            java.util.List<AudioDeviceInfo> route = am.getAudioDevicesForAttributes(new android.media.AudioAttributes.Builder()
                    .setUsage(android.media.AudioAttributes.USAGE_ASSISTANT).build());
            devices = route.toArray(new AudioDeviceInfo[0]);
        } else {
            devices = am.getDevices(AudioManager.GET_DEVICES_OUTPUTS);
        }
        for (AudioDeviceInfo d : devices) {
            switch (d.getType()) {
                case AudioDeviceInfo.TYPE_WIRED_HEADPHONES:
                case AudioDeviceInfo.TYPE_WIRED_HEADSET:
                case AudioDeviceInfo.TYPE_BLUETOOTH_A2DP:
                case AudioDeviceInfo.TYPE_BLUETOOTH_SCO:
                case AudioDeviceInfo.TYPE_USB_HEADSET:
                    return true;
                default:
                    if (Build.VERSION.SDK_INT >= 31 && d.getType() == AudioDeviceInfo.TYPE_BLE_HEADSET) return true;
            }
        }
        return false;
    }

    /** Do Not Disturb (any filter but "all"). */
    static boolean doNotDisturb(Context c) {
        android.app.NotificationManager nm = c.getSystemService(android.app.NotificationManager.class);
        return nm != null && nm.getCurrentInterruptionFilter() > android.app.NotificationManager.INTERRUPTION_FILTER_ALL;
    }

    static boolean inCall(AudioManager am) {
        int m = am.getMode();
        return m == AudioManager.MODE_IN_CALL || m == AudioManager.MODE_IN_COMMUNICATION;
    }

    /** May a message be read out now, by the user's "when" setting? Never during a call; out loud, not on silent. */
    boolean mayReadNow(Context c) {
        AudioManager am = (AudioManager) c.getSystemService(Context.AUDIO_SERVICE);
        if (inCall(am) || am.getMode() == AudioManager.MODE_RINGTONE) return false;
        if (privateOutput(am)) return true;
        boolean ringerOn = am.getRingerMode() == AudioManager.RINGER_MODE_NORMAL;
        PowerManager pm = (PowerManager) c.getSystemService(Context.POWER_SERVICE);
        switch (when()) {
            case ANY_TIME: return ringerOn;
            case PRIVATE_OR_SCREEN_OFF: return ringerOn && !pm.isInteractive();
            default: return false;
        }
    }

    /** Announcements (meetings, the briefing) play through headphones, or out loud with the ringer on; not in DND. */
    static boolean mayAnnounceNow(Context c) {
        AudioManager am = (AudioManager) c.getSystemService(Context.AUDIO_SERVICE);
        return !inCall(am) && !doNotDisturb(c) && (privateOutput(am) || am.getRingerMode() == AudioManager.RINGER_MODE_NORMAL);
    }
}
