package ai.whissle.steertts.demo;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.NotificationManager;
import android.content.ComponentName;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.provider.Settings;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.Switch;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * The Assistant tab: one row each for message readouts, incoming calls, meeting reminders and read aloud, a switch per
 * feature, details (which apps, when, how early) in a dialog behind the row; and the daily briefing (on/off, time,
 * Health Connect, play now). A feature without its permission shows "Off" and asks, with an explanation first, when
 * switched on. refresh() after the user returns from a permission screen.
 */
final class AssistantPanel {
    static final int REQ_CALENDAR = 1, REQ_NOTIFY = 2, REQ_HEALTH = 3, REQ_BRIEF_CALENDAR = 4;

    private final Activity a;
    private final Ui ui;
    private final AssistantPrefs prefs;
    private LinearLayout readRow, callRow, meetRow, briefRow, timeRow, healthRow;
    private Switch read, calls, meetings, brief;

    AssistantPanel(Activity a, Ui ui) {
        this.a = a;
        this.ui = ui;
        prefs = new AssistantPrefs(a);
    }

    View build() {
        LinearLayout box = ui.column();
        LinearLayout card = ui.card();
        card.addView(ui.label("Speaks up"));

        read = ui.toggle(prefs.readMessages());
        readRow = ui.settingRow("Message readouts", "", read, v -> readoutOptions());
        read.setOnCheckedChangeListener((v, on) -> {
            prefs.setReadMessages(on);
            if (on && !listenerOn()) askListenerAccess(read);
            refresh();
        });
        card.addView(readRow);
        card.addView(ui.divider());

        calls = ui.toggle(prefs.announceCalls());
        callRow = ui.settingRow("Incoming calls", "", calls, null);
        calls.setOnCheckedChangeListener((v, on) -> {
            prefs.setAnnounceCalls(on);
            if (on && !listenerOn()) askListenerAccess(calls);
            refresh();
        });
        card.addView(callRow);
        card.addView(ui.divider());

        meetings = ui.toggle(prefs.announceMeetings() && MeetingAnnouncer.allowed(a));
        meetRow = ui.settingRow("Meeting reminders", "", meetings, v -> leadOptions());
        meetings.setOnCheckedChangeListener((v, on) -> {
            prefs.setAnnounceMeetings(on);
            if (on && !MeetingAnnouncer.allowed(a)) askCalendarAccess(REQ_CALENDAR, meetings);
            MeetingAnnouncer.reschedule(a);
            refresh();
        });
        card.addView(meetRow);
        card.addView(ui.divider());

        card.addView(ui.settingRow("Read aloud anywhere", "Select text or tap Share in any app → Read with SteerTTS",
                null, null));
        box.addView(card);

        LinearLayout bc = ui.card();
        bc.addView(ui.label("Daily briefing"));
        brief = ui.toggle(Briefing.daily(a));
        briefRow = ui.settingRow("Every morning", "", brief, null);
        brief.setOnCheckedChangeListener((v, on) -> {
            Briefing.setDaily(a, on, Briefing.minuteOfDay(a));
            refresh();
        });
        bc.addView(briefRow);
        bc.addView(ui.divider());
        timeRow = ui.settingRow("Time", "", ui.chevron(), v -> new android.app.TimePickerDialog(a, (tp, h, m) -> {
            Briefing.setDaily(a, true, h * 60 + m);
            brief.setChecked(true);
            refresh();
        }, Briefing.minuteOfDay(a) / 60, Briefing.minuteOfDay(a) % 60, android.text.format.DateFormat.is24HourFormat(a)).show());
        bc.addView(timeRow);
        bc.addView(ui.divider());
        healthRow = ui.settingRow("Sleep, steps and heart rate", "", ui.chevron(), v -> askHealthAccess());
        bc.addView(healthRow);
        bc.addView(ui.divider());
        bc.addView(ui.settingRow("Play it now", "Hear today's briefing", ui.chevron(), v -> {
            if (!MeetingAnnouncer.allowed(a) && !a.getSharedPreferences("briefing", Activity.MODE_PRIVATE).getBoolean("asked_calendar", false)) {
                a.getSharedPreferences("briefing", Activity.MODE_PRIVATE).edit().putBoolean("asked_calendar", true).apply();
                askCalendarAccess(REQ_BRIEF_CALENDAR, null);
            } else {
                Briefing.play(a);
            }
        }));
        box.addView(bc);
        return box;
    }

    void refresh() {
        boolean access = listenerOn();
        sub(readRow, !access ? "Off · reads new messages aloud when you have headphones in"
                : AssistantPrefs.WHEN_SHORT[prefs.when()] + " · " + appsOnCount() + " apps · tap for options", false);
        sub(callRow, access ? "Says who is calling" : "Off · says who is calling", false);
        if (!MeetingAnnouncer.allowed(a)) {
            prefs.setAnnounceMeetings(false);
            meetings.setChecked(false);
            sub(meetRow, "Off · a heads-up before each meeting", false);
        } else {
            int m = prefs.leadMinutes();
            sub(meetRow, m + (m == 1 ? " minute" : " minutes") + " before each event", false);
        }
        if (access && waitingFor != null) {   // back from the system screen with access: the switch that asked stays on
            Switch s = waitingFor;
            waitingFor = null;
            s.setChecked(true);
        } else if (!access && waitingFor == null) {   // without access the switches cannot do anything
            read.setChecked(false);
            calls.setChecked(false);
        }
        MeetingAnnouncer.reschedule(a);
        int at = Briefing.minuteOfDay(a);
        String time = MeetingAnnouncer.spokenTime(java.time.LocalTime.of(at / 60, at % 60));
        sub(briefRow, Briefing.daily(a) ? "On · at " + time + ", when headphones or the car are connected"
                : "Off · your day, sleep and steps, spoken", false);
        sub(timeRow, time, false);
        sub(healthRow, !Briefing.healthAvailable() ? "Needs Android 14 or later"
                : Briefing.healthGranted(a) ? "✓ From Health Connect" : "Tap to connect Health Connect", false);
        if (Briefing.healthGranted(a)) ((TextView) healthRow.getTag()).setTextColor(Ui.ACCENT);
    }

    void onPermission(int req, boolean granted) {
        if (req == REQ_HEALTH) {
            refresh();
            return;
        }
        if (req == REQ_BRIEF_CALENDAR) {
            Briefing.play(a);
            refresh();
            return;
        }
        if (req != REQ_CALENDAR) return;
        prefs.setAnnounceMeetings(granted);
        meetings.setChecked(granted);
        refresh();
    }

    private void sub(LinearLayout row, String s, boolean warn) {
        TextView t = (TextView) row.getTag();
        t.setText(s);
        t.setTextColor(warn ? Ui.WARN : Ui.MUTED);
    }

    // ------------------------------------------------------------------ disclosures (before every system permission screen)
    private Switch waitingFor;   // the switch that sent the user to the notification-access screen

    /** What notification access is used for, in the app's words, before the system screen (Play's disclosure rule). */
    private void askListenerAccess(Switch from) {
        new AlertDialog.Builder(a)
                .setTitle("Let SteerTTS read notifications?")
                .setMessage("To read new messages aloud and say who is calling, SteerTTS needs notification access.\n\n"
                        + "It reads the text of new notifications from the apps you choose and speaks it on this phone. "
                        + "Nothing is stored, and nothing leaves your phone.\n\n"
                        + "Next, Android shows its notification access screen: switch SteerTTS on there.")
                .setPositiveButton("Continue", (d, w) -> {
                    waitingFor = from;
                    openListenerSettings();
                })
                .setNegativeButton("Not now", (d, w) -> from.setChecked(false))
                .setOnCancelListener(d -> from.setChecked(false))
                .show();
    }

    private void askCalendarAccess(int req, Switch from) {
        new AlertDialog.Builder(a)
                .setTitle("Let SteerTTS see your calendar?")
                .setMessage("To announce meetings before they start and tell you about your day, SteerTTS reads the "
                        + "title, time and place of your upcoming events. It stays on this phone and is never sent anywhere.")
                .setPositiveButton("Continue",
                        (d, w) -> a.requestPermissions(new String[] {Manifest.permission.READ_CALENDAR}, req))
                .setNegativeButton("Not now", (d, w) -> {
                    if (from != null) from.setChecked(false);
                    else if (req == REQ_BRIEF_CALENDAR) Briefing.play(a);
                })
                .setOnCancelListener(d -> { if (from != null) from.setChecked(false); })
                .show();
    }

    /** What the briefing reads from Health Connect, before Android's own permission screen. */
    private void askHealthAccess() {
        if (!Briefing.healthAvailable()) return;
        if (Briefing.healthGranted(a)) {
            try {
                a.startActivity(new Intent("android.health.connect.action.MANAGE_HEALTH_PERMISSIONS")
                        .putExtra(Intent.EXTRA_PACKAGE_NAME, a.getPackageName()));
            } catch (Exception e) {
                a.startActivity(new Intent("android.health.connect.action.HEALTH_HOME_SETTINGS"));
            }
            return;
        }
        new AlertDialog.Builder(a)
                .setTitle("Add your health to the briefing?")
                .setMessage("SteerTTS can read last night's sleep, your steps and your resting heart rate from Health "
                        + "Connect, and say them in your briefing.\n\nIt only reads; it stores nothing and sends nothing "
                        + "anywhere. Next, choose what to allow.")
                .setPositiveButton("Continue", (d, w) -> a.requestPermissions(Briefing.HEALTH_PERMISSIONS, REQ_HEALTH))
                .setNegativeButton("Not now", null)
                .show();
    }

    private ComponentName listener() { return new ComponentName(a, ReadoutListener.class); }

    private boolean listenerOn() { return listenerOn(a); }

    static boolean listenerOn(android.content.Context c) {
        ComponentName l = new ComponentName(c, ReadoutListener.class);
        if (Build.VERSION.SDK_INT >= 27)
            return c.getSystemService(NotificationManager.class).isNotificationListenerAccessGranted(l);
        String s = Settings.Secure.getString(c.getContentResolver(), "enabled_notification_listeners");
        return s != null && s.contains(l.flattenToString());
    }

    private void openListenerSettings() {
        Intent i = Build.VERSION.SDK_INT >= 30
                ? new Intent(Settings.ACTION_NOTIFICATION_LISTENER_DETAIL_SETTINGS)
                        .putExtra(Settings.EXTRA_NOTIFICATION_LISTENER_COMPONENT_NAME, listener().flattenToString())
                : new Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS);
        try {
            a.startActivity(i);
        } catch (Exception e) {
            a.startActivity(new Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS));
        }
    }

    // ------------------------------------------------------------------ dialogs

    private void readoutOptions() {
        new AlertDialog.Builder(a)
                .setTitle("Read messages")
                .setSingleChoiceItems(AssistantPrefs.WHEN_LABELS, prefs.when(), (d, w) -> {
                    prefs.setWhen(w);
                    refresh();
                })
                .setPositiveButton("Done", null)
                .setNeutralButton("Choose apps…", (d, w) -> pickApps())
                .show();
    }

    private void leadOptions() {
        String[] labels = new String[AssistantPrefs.LEAD_MINUTES.length];
        int sel = 0;
        for (int i = 0; i < labels.length; ++i) {
            int m = AssistantPrefs.LEAD_MINUTES[i];
            labels[i] = m + (m == 1 ? " minute before" : " minutes before");
            if (m == prefs.leadMinutes()) sel = i;
        }
        new AlertDialog.Builder(a)
                .setTitle("Announce meetings")
                .setSingleChoiceItems(labels, sel, (d, w) -> {
                    prefs.setLeadMinutes(AssistantPrefs.LEAD_MINUTES[w]);
                    MeetingAnnouncer.reschedule(a);
                    refresh();
                    d.dismiss();
                })
                .show();
    }

    private int appsOnCount() {
        int n = 0;
        for (String p : prefs.appsOn()) n += installed(p) ? 1 : 0;
        return n;
    }

    private boolean installed(String p) {
        try {
            a.getPackageManager().getApplicationInfo(p, 0);
            return true;
        } catch (PackageManager.NameNotFoundException e) {
            return false;
        }
    }

    /** The messaging apps installed plus every app seen posting something readable, each on or off. */
    private void pickApps() {
        PackageManager pm = a.getPackageManager();
        Set<String> pkgs = new TreeSet<>(prefs.seen());
        pkgs.addAll(AssistantPrefs.MESSAGING);
        List<String> ids = new ArrayList<>(), labels = new ArrayList<>();
        for (String p : pkgs) {
            try {
                ApplicationInfo ai = pm.getApplicationInfo(p, 0);
                ids.add(p);
                labels.add(pm.getApplicationLabel(ai).toString());
            } catch (PackageManager.NameNotFoundException ignored) {   // not installed
            }
        }
        Set<String> on = prefs.appsOn();
        boolean[] checked = new boolean[ids.size()];
        for (int i = 0; i < checked.length; ++i) checked[i] = on.contains(ids.get(i));
        new AlertDialog.Builder(a)
                .setTitle("Read messages from")
                .setMultiChoiceItems(labels.toArray(new String[0]), checked, (d, i, c) -> {
                    if (c) on.add(ids.get(i));
                    else on.remove(ids.get(i));
                })
                .setPositiveButton("Done", (d, w) -> {
                    prefs.setAppsOn(on);
                    refresh();
                })
                .setNegativeButton("Cancel", null)
                .show();
    }
}
