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

    /** Headphones, a headset, a car or any other Bluetooth / wired / USB output: speech is heard by the user only. */
    static boolean privateOutput(AudioManager am) {
        for (AudioDeviceInfo d : am.getDevices(AudioManager.GET_DEVICES_OUTPUTS)) {
            switch (d.getType()) {
                case AudioDeviceInfo.TYPE_WIRED_HEADPHONES:
                case AudioDeviceInfo.TYPE_WIRED_HEADSET:
                case AudioDeviceInfo.TYPE_BLUETOOTH_A2DP:
                case AudioDeviceInfo.TYPE_BLUETOOTH_SCO:
                case AudioDeviceInfo.TYPE_USB_HEADSET:
                    return true;
                default:
                    if (Build.VERSION.SDK_INT >= 31 && (d.getType() == AudioDeviceInfo.TYPE_BLE_HEADSET
                            || d.getType() == AudioDeviceInfo.TYPE_BLE_SPEAKER)) return true;
            }
        }
        return false;
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

    /** Announcements (calls, meetings) play through headphones, or out loud with the ringer on. */
    static boolean mayAnnounceNow(Context c) {
        AudioManager am = (AudioManager) c.getSystemService(Context.AUDIO_SERVICE);
        return !inCall(am) && (privateOutput(am) || am.getRingerMode() == AudioManager.RINGER_MODE_NORMAL);
    }
}
