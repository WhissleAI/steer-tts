package ai.whissle.steertts.demo;

import java.util.ArrayList;
import java.util.List;

/**
 * What the assistant did with the last notifications, for the Assistant tab: spoken, kept for catch-up, or not read
 * and why ("Email is off", "no headphones", "Do Not Disturb"). Answers "why didn't it say anything?", also after the
 * app was closed: kept in a small private file, and only the app, the person or title and the outcome, never the
 * message itself.
 */
final class Recent {
    private Recent() {}

    static final class Event {
        final long time;
        final String app, who, outcome;
        final boolean spoken;
        String pkg, fix;   // what a tap on the row offers: "app" (add pkg to the list), "when", "messages", "email", "calls"…

        Event(String app, String who, String outcome, boolean spoken) { this(System.currentTimeMillis(), app, who, outcome, spoken); }

        Event(long time, String app, String who, String outcome, boolean spoken) {
            this.time = time;
            this.app = app;
            this.who = who;
            this.outcome = outcome;
            this.spoken = spoken;
        }
    }

    private static final List<Event> events = new ArrayList<>();
    static volatile Runnable onChange;
    private static java.io.File store;   // null until init(): then events are loaded once and saved on every change

    static synchronized void init(android.content.Context c) {
        if (store != null) return;
        store = new java.io.File(c.getApplicationContext().getFilesDir(), "recent.json");
        try (java.io.FileInputStream in = new java.io.FileInputStream(store)) {
            org.json.JSONArray a = new org.json.JSONArray(new String(in.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8));
            List<Event> loaded = new ArrayList<>();
            for (int i = 0; i < a.length(); ++i) {
                org.json.JSONObject o = a.getJSONObject(i);
                Event e = new Event(o.getLong("t"), o.getString("app"), o.optString("who"), o.getString("out"), o.optBoolean("spoken"));
                e.pkg = o.optString("pkg", null);
                e.fix = o.optString("fix", null);
                loaded.add(e);
            }
            List<Event> merged = new ArrayList<>(events);   // anything added before init is newer
            merged.addAll(loaded);
            events.clear();
            events.addAll(merged.subList(0, Math.min(merged.size(), 12)));
        } catch (Exception ignored) {   // nothing saved yet
        }
    }

    private static void save() {
        if (store == null) return;
        try {
            org.json.JSONArray a = new org.json.JSONArray();
            for (Event e : events) {
                org.json.JSONObject o = new org.json.JSONObject().put("t", e.time).put("app", e.app).put("who", e.who)
                        .put("out", e.outcome).put("spoken", e.spoken);
                if (e.pkg != null) o.put("pkg", e.pkg);
                if (e.fix != null) o.put("fix", e.fix);
                a.put(o);
            }
            try (java.io.FileOutputStream out = new java.io.FileOutputStream(store)) {
                out.write(a.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
            }
        } catch (Exception ignored) {
        }
    }

    static synchronized void add(String app, String who, String outcome, boolean spoken) { add(app, who, outcome, spoken, null, null); }

    static synchronized void add(String app, String who, String outcome, boolean spoken, String pkg, String fix) {
        Event e = new Event(app, who == null ? "" : who, outcome, spoken);
        e.pkg = pkg;
        e.fix = fix;
        events.add(0, e);
        while (events.size() > 12) events.remove(events.size() - 1);
        save();
        Runnable r = onChange;
        if (r != null) new android.os.Handler(android.os.Looper.getMainLooper()).post(r);
    }

    static synchronized List<Event> all() { return new ArrayList<>(events); }
}
