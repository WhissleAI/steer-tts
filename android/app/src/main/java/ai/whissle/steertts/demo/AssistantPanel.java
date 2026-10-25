package ai.whissle.steertts.demo;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.NotificationManager;
import android.content.ComponentName;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.os.Build;
import android.provider.Settings;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * The app screen's assistant settings: notification access (message readouts, incoming calls), which apps and when,
 * meeting announcements (calendar permission) and how early, plus how to use read aloud. refresh() after the user
 * returns from a permission screen.
 */
final class AssistantPanel {
    static final int REQ_CALENDAR = 1, REQ_NOTIFY = 2;

    private final Activity a;
    private final AssistantPrefs prefs;
    private TextView access;
    private Button grant;
    private Switch meetings;

    AssistantPanel(Activity a) {
        this.a = a;
        prefs = new AssistantPrefs(a);
    }

    View build() {
        LinearLayout l = new LinearLayout(a);
        l.setOrientation(LinearLayout.VERTICAL);
        l.setPadding(0, dp(8), 0, dp(16));

        TextView title = new TextView(a);
        title.setText("Assistant");
        title.setTextSize(18);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        l.addView(title);

        access = new TextView(a);
        l.addView(access);
        grant = new Button(a);
        grant.setOnClickListener(v -> openListenerSettings());
        l.addView(grant);

        Switch read = toggle("Read new messages aloud", prefs.readMessages(), prefs::setReadMessages);
        l.addView(read);
        Spinner when = new Spinner(a);
        when.setAdapter(new ArrayAdapter<>(a, android.R.layout.simple_spinner_dropdown_item, AssistantPrefs.WHEN_LABELS));
        when.setSelection(prefs.when());
        when.setOnItemSelectedListener(picked(prefs::setWhen));
        l.addView(labeled("When", when));
        Button apps = new Button(a);
        apps.setText("Apps to read…");
        apps.setOnClickListener(v -> pickApps());
        l.addView(apps);

        l.addView(toggle("Announce incoming calls (caller's name)", prefs.announceCalls(), prefs::setAnnounceCalls));

        meetings = toggle("Announce meetings before they start", prefs.announceMeetings(), on -> {
            prefs.setAnnounceMeetings(on);
            if (on && !MeetingAnnouncer.allowed(a))
                a.requestPermissions(new String[] {Manifest.permission.READ_CALENDAR}, REQ_CALENDAR);
            MeetingAnnouncer.reschedule(a);
        });
        l.addView(meetings);
        Spinner lead = new Spinner(a);
        List<String> leads = new ArrayList<>();
        int sel = 0;
        for (int i = 0; i < AssistantPrefs.LEAD_MINUTES.length; ++i) {
            int m = AssistantPrefs.LEAD_MINUTES[i];
            leads.add(m + (m == 1 ? " minute before" : " minutes before"));
            if (m == prefs.leadMinutes()) sel = i;
        }
        lead.setAdapter(new ArrayAdapter<>(a, android.R.layout.simple_spinner_dropdown_item, leads));
        lead.setSelection(sel);
        lead.setOnItemSelectedListener(picked(pos -> {
            prefs.setLeadMinutes(AssistantPrefs.LEAD_MINUTES[pos]);
            MeetingAnnouncer.reschedule(a);
        }));
        l.addView(labeled("Meetings", lead));

        Button test = new Button(a);
        test.setText("Try a readout");
        test.setOnClickListener(v -> Announcer.get(a).say(
                "Priya on WhatsApp: Running ten minutes late. Can you order me a coffee?", "com.whatsapp", true, null));
        l.addView(test);

        TextView hint = new TextView(a);
        hint.setText("Read aloud: select text in any app and pick \"Read with SteerTTS\" from the menu, or share text "
                + "or a link to SteerTTS.");
        hint.setPadding(0, dp(4), 0, 0);
        l.addView(hint);

        if (Build.VERSION.SDK_INT >= 33
                && a.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED)
            a.requestPermissions(new String[] {Manifest.permission.POST_NOTIFICATIONS}, REQ_NOTIFY);   // read-aloud's Stop
        return l;
    }

    void refresh() {
        boolean on = listenerOn();
        access.setText(on ? "Notification access: on (messages and calls can be read out)"
                : "Notification access: off. Grant it to hear messages and callers.");
        grant.setText(on ? "Notification access settings" : "Grant notification access");
        if (prefs.announceMeetings() && !MeetingAnnouncer.allowed(a)) meetings.setChecked(false);
        if (!MeetingAnnouncer.allowed(a)) prefs.setAnnounceMeetings(false);
        MeetingAnnouncer.reschedule(a);
    }

    void onPermission(int req, boolean granted) {
        if (req != REQ_CALENDAR) return;
        prefs.setAnnounceMeetings(granted);
        meetings.setChecked(granted);
        MeetingAnnouncer.reschedule(a);
    }

    private ComponentName listener() { return new ComponentName(a, ReadoutListener.class); }

    private boolean listenerOn() {
        if (Build.VERSION.SDK_INT >= 27)
            return a.getSystemService(NotificationManager.class).isNotificationListenerAccessGranted(listener());
        String s = Settings.Secure.getString(a.getContentResolver(), "enabled_notification_listeners");
        return s != null && s.contains(listener().flattenToString());
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
                .setPositiveButton("Done", (d, w) -> prefs.setAppsOn(on))
                .setNegativeButton("Cancel", null)
                .show();
    }

    private interface OnToggle { void toggled(boolean on); }
    private interface OnPick { void picked(int pos); }

    private Switch toggle(String label, boolean on, OnToggle t) {
        Switch s = new Switch(a);
        s.setText(label);
        s.setChecked(on);
        s.setPadding(0, dp(6), 0, dp(6));
        s.setOnCheckedChangeListener((v, c) -> t.toggled(c));
        return s;
    }

    private static AdapterView.OnItemSelectedListener picked(OnPick p) {
        return new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> v, View view, int pos, long id) { p.picked(pos); }
            @Override public void onNothingSelected(AdapterView<?> v) { }
        };
    }

    private LinearLayout labeled(String label, View v) {
        LinearLayout l = new LinearLayout(a);
        l.setGravity(android.view.Gravity.CENTER_VERTICAL);
        TextView t = new TextView(a);
        t.setText(label);
        t.setMinWidth(dp(72));
        l.addView(t);
        l.addView(v, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        return l;
    }

    private int dp(int x) { return Math.round(x * a.getResources().getDisplayMetrics().density); }
}
