package ai.whissle.steertts.demo;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.NotificationManager;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.provider.Settings;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * The Assistant tab: a tile per thing SteerTTS does for you (messages, calls, meetings, the morning briefing, catching
 * up, a voice for each person), showing at a glance what is on. A tile opens a sheet that says what the feature does,
 * turns it on or off and holds its options. What a tile shows is worked out from the saved setting and the permission
 * it needs, never the other way round: only the user's own tap changes a setting. Turning on a feature that needs a
 * permission explains it first (Play's disclosure rule), and the feature switches on when the permission comes back.
 */
final class AssistantScreen {
    static final int REQ_CALENDAR = 1, REQ_HEALTH = 3;
    private static final int MESSAGES = 0, CALLS = 1, MEETINGS = 2, REMINDERS = 3, DELIVERIES = 4, EMAIL = 5, BATTERY = 6,
            BRIEFING = 7, CATCHUP = 8, VOICES = 9, SCREEN = 10, N = 11;

    private final Activity a;
    private final Ui u;
    private final AssistantPrefs prefs;
    private final View.OnClickListener settings;
    private final Ui.Tile[] tiles = new Ui.Tile[N];
    private TextView catchUpLabel;
    private LinearLayout access, recent;
    private int pending = -1;   // the feature waiting for a permission screen to come back
    private boolean askedListener, retryListener, askedScreen;
    private int calendarFor = -1;   // who asked for calendar access: MEETINGS, BRIEFING (its sheet) or PLAY_BRIEFING
    private static final int PLAY_BRIEFING = 100;

    AssistantScreen(Activity a, Ui u, View.OnClickListener settings) {
        this.a = a;
        this.u = u;
        this.settings = settings;
        prefs = new AssistantPrefs(a);
    }

    View build() {
        LinearLayout root = u.column();
        root.setPadding(u.dp(18), u.dp(14), u.dp(18), u.dp(28));
        root.addView(u.topBar("Assistant", settings));

        // what it sounds like: three readouts that show the voice following the moment
        LinearLayout demo = u.card();
        demo.setPadding(u.dp(20), u.dp(18), u.dp(20), u.dp(18));
        demo.setBackground(Ui.round(Ui.ACCENT_SOFT, u.dp(26)));
        LinearLayout dr = u.row();
        LinearLayout dt = u.column();
        dt.addView(u.text("It speaks the way people mean it", 18, Ui.INK, true));
        TextView dd = u.text("A sorry sounds gentle, good news sounds bright, a third call sounds urgent. Hear a sample.", 13.5f, Ui.MUTED, false);
        dd.setPadding(0, u.dp(4), u.dp(8), 0);
        dt.addView(dd);
        dr.addView(dt, Ui.weight(1));
        dr.addView(u.icon(R.drawable.ic_play, "Play a sample", 56, true, v -> sample(a)));
        demo.addView(dr);
        root.addView(demo);

        // without notification access most tiles cannot work: one clear way to fix that, instead of a wall of "Off"
        access = u.card();
        access.setPadding(u.dp(20), u.dp(16), u.dp(20), u.dp(16));
        LinearLayout ar = u.row();
        ar.addView(u.badge(R.drawable.ic_message, Ui.ACCENT, Ui.ON_ACCENT, 44));
        LinearLayout at = u.column();
        at.setPadding(u.dp(14), 0, u.dp(8), 0);
        at.addView(u.text("One step to turn it on", 16, Ui.INK, true));
        at.addView(u.text("Allow notification access so SteerTTS can read messages, calls, reminders and deliveries to you.",
                13.5f, Ui.MUTED, false));
        ar.addView(at, Ui.weight(1));
        access.addView(ar);
        TextView allow = u.pill("Allow", true, v -> askListenerAccess(-1));
        LinearLayout.LayoutParams al = Ui.matchWrap();
        al.topMargin = u.dp(12);
        access.addView(allow, al);
        root.addView(access);

        int[] icons = {R.drawable.ic_message, R.drawable.ic_call, R.drawable.ic_calendar, R.drawable.ic_alarm, R.drawable.ic_delivery,
                R.drawable.ic_email, R.drawable.ic_battery, R.drawable.ic_sun, R.drawable.ic_inbox,
                R.drawable.ic_people, R.drawable.ic_speaker};
        // three short groups instead of one long grid
        Object[][] groups = {{"Speaks up when", new int[] {MESSAGES, EMAIL, CALLS, MEETINGS, REMINDERS, DELIVERIES, BATTERY}},
                {"Every day", new int[] {BRIEFING, CATCHUP}}, {"More", new int[] {SCREEN, VOICES}}};
        for (Object[] g : groups) {
            root.addView(u.section((String) g[0]));
            int[] fs = (int[]) g[1];
            for (int i = 0; i < fs.length; i += 2) {
                LinearLayout r = u.row();
                r.setGravity(android.view.Gravity.TOP);
                for (int k = i; k < Math.min(fs.length, i + 2); ++k) {
                    final int f = fs[k];
                    tiles[f] = new Ui.Tile(u, icons[f], TITLES[f]);
                    tiles[f].root.setOnClickListener(v -> sheet(f));
                    LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1);
                    if (k == i) lp.rightMargin = u.dp(6);
                    else lp.leftMargin = u.dp(6);
                    lp.topMargin = u.dp(12);
                    r.addView(tiles[f].root, lp);
                }
                if (fs.length - i == 1) r.addView(new View(a), new LinearLayout.LayoutParams(0, 1, 1));   // keep the tile half width
                root.addView(r, Ui.matchWrap());
            }
        }

        LinearLayout acts = u.row();
        acts.setPadding(0, u.dp(18), 0, 0);
        TextView cu = u.pill("Catch me up", true, v -> ReadoutListener.catchUp(a, true));
        catchUpLabel = cu;
        LinearLayout.LayoutParams l1 = Ui.weight(1);
        l1.rightMargin = u.dp(6);
        acts.addView(cu, l1);
        LinearLayout.LayoutParams l2 = Ui.weight(1);
        l2.leftMargin = u.dp(6);
        acts.addView(u.pill("Play briefing", false, v -> playBriefing()), l2);
        root.addView(acts, Ui.matchWrap());

        // what happened with the last notifications, and why something was not read
        root.addView(u.section("Recently"));
        recent = u.card();
        root.addView(recent);
        Recent.init(a);
        Recent.onChange = this::showRecent;

        TextView tip = u.text("Read aloud anywhere: select text or tap Share in any app, then “Read with SteerTTS”. "
                + "Add the “Read this screen”, “Read copied text” and “Catch me up” tiles to Quick Settings for one-tap access.", 13, Ui.MUTED, false);
        tip.setPadding(u.dp(4), u.dp(18), u.dp(4), 0);
        root.addView(tip);

        refresh();
        ScrollView sv = new ScrollView(a);
        sv.addView(root);
        return sv;
    }

    // ---------------------------------------------------------------- state (always from prefs + permissions)
    boolean on(int f) {
        switch (f) {
            case MESSAGES: return prefs.readMessages() && listenerOn(a);
            case REMINDERS: return prefs.reminders() && listenerOn(a);
            case DELIVERIES: return prefs.deliveries() && listenerOn(a);
            case EMAIL: return prefs.email() && listenerOn(a);
            case BATTERY: return prefs.battery() && listenerOn(a);
            case SCREEN: return screenReaderOn(a);
            case CALLS: return prefs.announceCalls() && listenerOn(a);
            case MEETINGS: return prefs.announceMeetings() && MeetingAnnouncer.allowed(a);
            case BRIEFING: return Briefing.daily(a);
            case CATCHUP: return prefs.catchUp() && listenerOn(a);
            default: return prefs.personVoices();
        }
    }

    private String status(int f) {
        boolean on = on(f);
        switch (f) {
            case MESSAGES:
                return on ? AssistantPrefs.WHEN_SHORT[prefs.when()] + " · " + appsOnCount() + " apps" : "New messages read out as they arrive";
            case CALLS: return on ? "Who is calling, missed calls, and when it's urgent" : "Hear who's calling without looking";
            case REMINDERS: return on ? "Alarms, timers and to-do reminders, read out" : "Hear what an alarm or reminder is for";
            case DELIVERIES: return on ? "Rides, food and parcels, as they move" : "“Your driver is arriving”, read out";
            case EMAIL: return on ? "Who it's from, the subject and how it starts" : "Hear who emailed and what about";
            case BATTERY: return on ? "When it's low, and when it's full" : "A heads-up before the battery runs out";
            case SCREEN: return on ? "Tap “Read this screen” in Quick Settings, in any app" : "Any app's page, read aloud, with a tiny control";
            case MEETINGS: {
                int m = prefs.leadMinutes();
                return on ? m + (m == 1 ? " minute" : " minutes") + " before each event" : "A heads-up before each meeting";
            }
            case BRIEFING: {
                int at = Briefing.minuteOfDay(a);
                return on ? "Every day at " + MeetingAnnouncer.spokenTime(java.time.LocalTime.of(at / 60, at % 60))
                        : "Your day, sleep and steps, each morning";
            }
            case CATCHUP: {
                int n = Digest.size();
                return on ? (n > 0 ? n + " waiting · read when headphones connect" : "Missed messages read when headphones connect")
                        : "Missed messages, when you're ready";
            }
            default: return on ? "Each sender has a voice of their own" : "Know who it's from by the voice";
        }
    }

    void refresh() {
        if (tiles[0] == null) return;
        if (askedListener) {   // back from the notification access screen
            askedListener = false;
            int f = pending;
            pending = -1;
            if (listenerOn(a)) {
                if (f >= 0 && needsListener(f)) {
                    set(f, true);
                    if (f == MESSAGES || f == EMAIL) askWhen();
                }
            } else if (Build.VERSION.SDK_INT >= 33 && !fromStore()) {
                restrictedHelp(f);   // most likely Android's "restricted setting" for apps installed from a file
            }
        } else if (askedScreen) {   // back from the accessibility settings
            askedScreen = false;
            if (!screenReaderOn(a) && Build.VERSION.SDK_INT >= 33 && !fromStore()) restrictedHelp(SCREEN);
        } else if (retryListener) {   // back from App info after allowing restricted settings: straight to the switch
            retryListener = false;
            if (!listenerOn(a)) {
                askedListener = true;
                openListenerSettings();
            }
        }
        for (int f = 0; f < N; ++f) tiles[f].set(on(f), status(f));
        access.setVisibility(listenerOn(a) ? View.GONE : View.VISIBLE);
        showRecent();
        int n = Digest.size();
        catchUpLabel.setText(n > 0 ? "Catch me up · " + n : "Catch me up");
        MeetingAnnouncer.reschedule(a);
    }

    private void showRecent() {
        if (recent == null) return;
        recent.removeAllViews();
        List<Recent.Event> ev = Recent.all();
        if (ev.isEmpty()) {
            recent.addView(u.text("Nothing yet. When a message, an email or a call comes in, you'll see here whether SteerTTS "
                    + "read it out, and why if it didn't.", 13.5f, Ui.MUTED, false));
            return;
        }
        java.text.DateFormat tf = android.text.format.DateFormat.getTimeFormat(a);
        for (int i = 0; i < Math.min(5, ev.size()); ++i) {
            Recent.Event e = ev.get(i);
            if (i > 0) recent.addView(u.divider());
            LinearLayout r = u.row();
            r.setPadding(0, u.dp(8), 0, u.dp(8));
            r.setGravity(android.view.Gravity.TOP);
            r.addView(u.badge(e.spoken ? R.drawable.ic_check : R.drawable.ic_inbox, e.spoken ? Ui.ACCENT_SOFT : Ui.LINE, e.spoken ? Ui.ACCENT : Ui.MUTED, 30));
            LinearLayout t = u.column();
            t.setPadding(u.dp(12), 0, 0, 0);
            t.addView(u.text(e.app + (e.who.isEmpty() ? "" : " · " + e.who) + "  ·  " + when(e.time, tf), 14, Ui.INK, true));
            t.addView(u.text(e.outcome, 13, e.spoken ? Ui.ACCENT : Ui.MUTED, false));
            r.addView(t, Ui.weight(1));
            if (e.fix != null) {
                r.setOnClickListener(v -> fix(e));
                r.addView(u.chevron());
            }
            recent.addView(r);
        }
    }

    /** "2:14 PM" today, "Yesterday 9:02 AM", else "Oct 21". */
    private static String when(long t, java.text.DateFormat tf) {
        java.time.LocalDate d = java.time.Instant.ofEpochMilli(t).atZone(java.time.ZoneId.systemDefault()).toLocalDate();
        java.time.LocalDate today = java.time.LocalDate.now();
        String time = tf.format(new java.util.Date(t));
        if (d.equals(today)) return time;
        if (d.equals(today.minusDays(1))) return "Yesterday " + time;
        return d.format(java.time.format.DateTimeFormatter.ofPattern("MMM d", java.util.Locale.getDefault()));
    }

    /** A tap on a "not read" row: fix it right there. */
    private void fix(Recent.Event e) {
        switch (e.fix) {
            case "app":
                new AlertDialog.Builder(a).setTitle("Read messages from " + e.app + "?")
                        .setPositiveButton("Add " + e.app, (d, w) -> {
                            java.util.Set<String> on = prefs.appsOn();
                            on.add(e.pkg);
                            prefs.setAppsOn(on);
                            refresh();
                        }).setNegativeButton("Not now", null).show();
                break;
            case "when": askWhen(); break;
            case "messages": toggle(MESSAGES); break;
            case "email": toggle(EMAIL); break;
            case "calls": toggle(CALLS); break;
            default: break;
        }
    }

    /** Right after Messages or Email is switched on: when may it speak? (A choice, not a hidden default.) */
    private void askWhen() {
        new AlertDialog.Builder(a)
                .setTitle("When should SteerTTS read them out?")
                .setSingleChoiceItems(AssistantPrefs.WHEN_LABELS, prefs.when(), (d, w) -> {
                    prefs.setWhen(w);
                    refresh();
                    d.dismiss();
                })
                .setNegativeButton("Keep this", null)
                .show();
    }

    private void set(int f, boolean v) {
        switch (f) {
            case MESSAGES: prefs.setReadMessages(v); break;
            case REMINDERS: prefs.setReminders(v); break;
            case DELIVERIES: prefs.setDeliveries(v); break;
            case EMAIL: prefs.setEmail(v); break;
            case BATTERY: prefs.setBattery(v); break;
            case CALLS: prefs.setAnnounceCalls(v); break;
            case MEETINGS: prefs.setAnnounceMeetings(v); MeetingAnnouncer.reschedule(a); break;
            case BRIEFING: Briefing.setDaily(a, v, Briefing.minuteOfDay(a)); break;
            case CATCHUP: prefs.setCatchUp(v); break;
            default: prefs.setPersonVoices(v); break;
        }
    }

    /** The user's own tap: on (asking for the permission first if needed) or off. */
    private void toggle(int f) {
        boolean on = on(f);
        if (on && f == SCREEN) {   // only Android's settings can switch an accessibility service off
            a.startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS));
            return;
        }
        if (on) {
            set(f, false);
        } else if (f == SCREEN) {
            askScreenAccess();
            return;
        } else if (needsListener(f) && !listenerOn(a)) {
            askListenerAccess(f);
            return;
        } else if (f == MEETINGS && !MeetingAnnouncer.allowed(a)) {
            askCalendarAccess(MEETINGS);
            return;
        } else {
            set(f, true);
            if (f == MESSAGES || f == EMAIL) askWhen();
        }
        refresh();
    }

    /** Features that work from notifications (or, for the battery, from the notification listener's connection). */
    private static boolean needsListener(int f) { return f != MEETINGS && f != BRIEFING && f != VOICES && f != SCREEN; }

    /** "Read this screen" is switched on in Android's accessibility settings. */
    static boolean screenReaderOn(Context c) {
        String s = Settings.Secure.getString(c.getContentResolver(), Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES);
        return s != null && s.contains(new ComponentName(c, ScreenReader.class).flattenToString());
    }

    /** From the "Read this screen" tile before setup: show the sheet. */
    void setupScreen() { sheet(SCREEN); }

    private void askScreenAccess() {
        new AlertDialog.Builder(a)
                .setTitle("Let SteerTTS read your screen?")
                .setMessage("To read any app's page aloud, SteerTTS uses Android's accessibility access. It reads the "
                        + "visible text of the app in front of you only when you start it (the Quick Settings tile or the "
                        + "accessibility button), can scroll that page to keep reading, and never stores or sends what it reads.\n\n"
                        + "Next, Android shows its accessibility settings: open “SteerTTS: read this screen” and switch it on.")
                .setPositiveButton("Continue", (d, w) -> {
                    askedScreen = true;
                    Intent i = new Intent("android.settings.ACCESSIBILITY_DETAILS_SETTINGS")
                            .putExtra(Intent.EXTRA_COMPONENT_NAME, new ComponentName(a, ScreenReader.class).flattenToString());
                    try {
                        a.startActivity(i);
                    } catch (Exception e) {
                        a.startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS));
                    }
                })
                .setNegativeButton("Not now", null)
                .show();
    }

    void onPermission(int req, boolean granted) {
        if (req == REQ_CALENDAR) {
            if (granted && calendarFor == MEETINGS) set(MEETINGS, true);   // only the feature that asked is switched on
            if (calendarFor == PLAY_BRIEFING) Briefing.play(a);           // with the calendar if allowed, without if not
            calendarFor = -1;
        }
        refresh();
    }

    private static final String[] TITLES = {"Messages", "Calls", "Meetings", "Reminders & timers", "Deliveries & rides",
            "Email", "Battery", "Morning briefing", "Catch me up", "A voice for each person", "Read any screen"};

    // ---------------------------------------------------------------- the sheet behind a tile
    private void sheet(int f) {
        String[] titles = TITLES;
        String[] about = {
                "New messages from the apps you choose are read aloud as they arrive, in the mood they were written: a "
                        + "sorry sounds gentle, good news sounds bright. By default only with headphones or in the car, "
                        + "never during a call or in Do Not Disturb.",
                "While the phone rings, SteerTTS says who is calling. If the same person calls again within a few "
                        + "minutes, it says so, and on the third call it tells you it may be urgent.",
                "A spoken heads-up a few minutes before each event on your calendar, with where it happens.",
                "When an alarm rings, a timer ends or a to-do app reminds you, SteerTTS says what it is for: “Reminder: "
                        + "call the dentist.” Works with the Clock, Google Tasks, Keep, Todoist and other reminder apps.",
                "Updates from ride, food and delivery apps (Uber, Lyft, DoorDash, Instacart, Amazon, Swiggy, Zomato and "
                        + "more) are read out as they happen: “Your driver is arriving.”",
                "New email from the apps in your list (Gmail, Outlook and other mail apps) is announced with who it's from, "
                        + "the subject and its first sentence, only when you can hear it privately (or as “When” allows).",
                "A spoken heads-up when the battery falls to 15 percent, and when it's fully charged.",
                "Every morning, or whenever you ask: your day ahead, last night's sleep and your steps, and any messages "
                        + "that waited overnight. Each line sounds like what it says.",
                "Messages that arrived while SteerTTS couldn't read them out wait here. When you put headphones in, "
                        + "it tells you who wrote and what they said. They wait in memory only, never written to storage.",
                "Each person's messages are read in a voice of their own, always the same one, so you know who wrote "
                        + "without hearing the name. The assistant's own voice says who it's from.",
                "Any app's page read aloud: an article, a recipe, a long message. Open it, then tap “Read this screen” in "
                        + "Quick Settings (or the accessibility button). A small control floats on top to pause, skip and "
                        + "keep going: with auto-continue on, SteerTTS scrolls the page and keeps reading."};
        LinearLayout box = u.column();
        box.setPadding(u.dp(24), u.dp(8), u.dp(24), 0);
        TextView t = u.text(about[f], 15, Ui.MUTED, false);
        t.setLineSpacing(0, 1.2f);
        box.addView(t);
        List<View> options = new ArrayList<>();
        switch (f) {
            case MESSAGES:
                options.add(u.settingRow("When", AssistantPrefs.WHEN_SHORT[prefs.when()], u.chevron(), v -> readoutOptions()));
                options.add(u.settingRow("Apps", appsOnCount() + " apps", u.chevron(), v -> pickApps()));
                break;
            case MEETINGS:
                options.add(u.settingRow("How early", prefs.leadMinutes() + " minutes before", u.chevron(), v -> leadOptions()));
                break;
            case BRIEFING: {
                int at = Briefing.minuteOfDay(a);
                options.add(u.settingRow("Time", MeetingAnnouncer.spokenTime(java.time.LocalTime.of(at / 60, at % 60)), u.chevron(),
                        v -> new android.app.TimePickerDialog(a, (tp, h, m) -> {
                            Briefing.setDaily(a, true, h * 60 + m);
                            refresh();
                        }, at / 60, at % 60, android.text.format.DateFormat.is24HourFormat(a)).show()));
                options.add(u.settingRow("Sleep, steps and heart rate", !Briefing.healthAvailable() ? "Needs Android 14 or later"
                        : Briefing.healthGranted(a) ? "✓ From Health Connect" : "Connect Health Connect", u.chevron(), v -> askHealthAccess()));
                options.add(u.settingRow("Calendar", MeetingAnnouncer.allowed(a) ? "✓ Today's events" : "Allow calendar access",
                        u.chevron(), v -> { if (!MeetingAnnouncer.allowed(a)) askCalendarAccess(BRIEFING); }));
                options.add(u.settingRow("Play it now", "Hear today's briefing", u.chevron(), v -> playBriefing()));
                break;
            }
            case CATCHUP:
                options.add(u.settingRow("Catch me up now", Digest.size() + " waiting", u.chevron(), v -> ReadoutListener.catchUp(a, true)));
                break;
            case VOICES:
                options.add(u.settingRow("Hear an example", "Two people, two voices", u.chevron(), v -> sample(a)));
                break;
            default:
                break;
        }
        if (!options.isEmpty()) {
            LinearLayout card = u.column();
            card.setPadding(0, u.dp(12), 0, 0);
            for (View o : options) card.addView(o);
            box.addView(card);
        }
        ScrollView sv = new ScrollView(a);
        sv.addView(box);
        boolean on = on(f);
        new AlertDialog.Builder(a).setTitle(titles[f]).setView(sv)
                .setPositiveButton(on ? "Turn off" : "Turn on", (d, w) -> toggle(f))
                .setNegativeButton("Close", null)
                .setOnDismissListener(d -> refresh())
                .show();
    }

    /** Three readouts that show the voice following the moment (and, with person voices on, who is speaking). */
    static void sample(Context c) {
        AssistantPrefs p = new AssistantPrefs(c);
        String own = Style.load(c).voice;
        List<String> sam = new ArrayList<>(), mom = new ArrayList<>();
        sam.add("I'm so sorry, the flight got cancelled. Call me when you can?");
        mom.add("HAPPY BIRTHDAY!!! 🎉 We're so proud of you!");
        Announcer an = Announcer.get(c);
        an.say(Readout.message("Sam", null, "WhatsApp", sam, p.personVoices(), own).pause(500), "sample", true, null);
        an.say(Readout.message("Mom", null, "Messages", mom, p.personVoices(), own).pause(500), "sample", false, null);
        an.say(Readout.call("Alex", null, 3), "sample", false, null);
    }

    private void playBriefing() {
        if (!MeetingAnnouncer.allowed(a) && !a.getSharedPreferences("briefing", Context.MODE_PRIVATE).getBoolean("asked_calendar", false)) {
            a.getSharedPreferences("briefing", Context.MODE_PRIVATE).edit().putBoolean("asked_calendar", true).apply();
            askCalendarAccess(PLAY_BRIEFING);
            return;
        }
        Briefing.play(a);
    }

    // ---------------------------------------------------------------- disclosures (before every system permission screen)
    private void askListenerAccess(int f) {
        new AlertDialog.Builder(a)
                .setTitle("Let SteerTTS read notifications?")
                .setMessage("To read new messages aloud and say who is calling, SteerTTS needs notification access.\n\n"
                        + "It reads the text of new notifications from the apps you choose and speaks it on this phone. "
                        + "Message text is never stored and nothing leaves your phone. A short history (which app, who, and "
                        + "whether it was read) is kept on the phone for the Recently list.\n\n"
                        + "Next, Android shows its notification access screen: switch SteerTTS on there.")
                .setPositiveButton("Continue", (d, w) -> {
                    pending = f;
                    askedListener = true;
                    openListenerSettings();
                })
                .setNegativeButton("Not now", null)
                .show();
    }

    /** Installed by an app store (or adb): Android's restricted settings do not apply. */
    private boolean fromStore() {
        try {
            String by = Build.VERSION.SDK_INT >= 30
                    ? a.getPackageManager().getInstallSourceInfo(a.getPackageName()).getInstallingPackageName()
                    : a.getPackageManager().getInstallerPackageName(a.getPackageName());
            return by != null && (by.equals("com.android.vending") || by.equals("com.android.shell")
                    || by.equals("com.sec.android.app.samsungapps") || by.equals("com.amazon.venezia") || by.equals("com.huawei.appmarket"));
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Android 13+ blocks notification access for apps installed from a file ("App was denied access" / "Restricted
     * setting"). The user can lift it in App info; then the access switch works.
     */
    private void restrictedHelp(int f) {
        new AlertDialog.Builder(a)
                .setTitle(f == SCREEN ? "Android blocked screen reading" : "Android blocked notification access")
                .setMessage("SteerTTS was installed from a file, so Android asks you to confirm before it can "
                        + (f == SCREEN ? "read the screen." : "read notifications.") + "\n\n1. Tap Open app info.\n2. Tap the ⋮ menu at the top right, then “Allow "
                        + "restricted settings”, and confirm.\n3. Come back and turn SteerTTS on again.")
                .setPositiveButton("Open app info", (d, w) -> {
                    pending = f;
                    retryListener = f != SCREEN;
                    a.startActivity(new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                            android.net.Uri.fromParts("package", a.getPackageName(), null)));
                })
                .setNegativeButton("Not now", null)
                .show();
    }

    private void askCalendarAccess(int forWhat) {
        calendarFor = forWhat;
        new AlertDialog.Builder(a)
                .setTitle("Let SteerTTS see your calendar?")
                .setMessage("To announce meetings before they start and tell you about your day, SteerTTS reads the "
                        + "title, time and place of your upcoming events. It stays on this phone and is never sent anywhere.")
                .setPositiveButton("Continue", (d, w) -> a.requestPermissions(new String[] {Manifest.permission.READ_CALENDAR}, REQ_CALENDAR))
                .setNegativeButton("Not now", (d, w) -> {
                    if (calendarFor == PLAY_BRIEFING) Briefing.play(a);   // play it without the calendar
                    calendarFor = -1;
                })
                .show();
    }

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

    static boolean listenerOn(Context c) {
        ComponentName l = new ComponentName(c, ReadoutListener.class);
        if (Build.VERSION.SDK_INT >= 27)
            return c.getSystemService(NotificationManager.class).isNotificationListenerAccessGranted(l);
        String s = Settings.Secure.getString(c.getContentResolver(), "enabled_notification_listeners");
        return s != null && s.contains(l.flattenToString());
    }

    private void openListenerSettings() {
        ComponentName l = new ComponentName(a, ReadoutListener.class);
        Intent i = Build.VERSION.SDK_INT >= 30
                ? new Intent(Settings.ACTION_NOTIFICATION_LISTENER_DETAIL_SETTINGS)
                        .putExtra(Settings.EXTRA_NOTIFICATION_LISTENER_COMPONENT_NAME, l.flattenToString())
                : new Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS);
        try {
            a.startActivity(i);
        } catch (Exception e) {
            a.startActivity(new Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS));
        }
    }

    // ---------------------------------------------------------------- option dialogs
    private void readoutOptions() {
        new AlertDialog.Builder(a)
                .setTitle("Read messages")
                .setSingleChoiceItems(AssistantPrefs.WHEN_LABELS, prefs.when(), (d, w) -> {
                    prefs.setWhen(w);
                    refresh();
                    d.dismiss();
                })
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
