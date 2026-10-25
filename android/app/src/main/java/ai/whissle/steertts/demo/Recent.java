package ai.whissle.steertts.demo;

import java.util.ArrayList;
import java.util.List;

/**
 * What the assistant did with the last notifications, for the Assistant tab: spoken, kept for catch-up, or not read
 * and why ("Email is off", "no headphones", "Do Not Disturb"). Answers "why didn't it say anything?". Memory only, and
 * only the app, the person or title and the outcome: never the message itself.
 */
final class Recent {
    private Recent() {}

    static final class Event {
        final long time;
        final String app, who, outcome;
        final boolean spoken;
        String pkg, fix;   // what a tap on the row offers: "app" (add pkg to the list), "when", "messages", "email", "calls"…

        Event(String app, String who, String outcome, boolean spoken) {
            this.time = System.currentTimeMillis();
            this.app = app;
            this.who = who;
            this.outcome = outcome;
            this.spoken = spoken;
        }
    }

    private static final List<Event> events = new ArrayList<>();
    static volatile Runnable onChange;

    static synchronized void add(String app, String who, String outcome, boolean spoken) { add(app, who, outcome, spoken, null, null); }

    static synchronized void add(String app, String who, String outcome, boolean spoken, String pkg, String fix) {
        Event e = new Event(app, who == null ? "" : who, outcome, spoken);
        e.pkg = pkg;
        e.fix = fix;
        events.add(0, e);
        while (events.size() > 12) events.remove(events.size() - 1);
        Runnable r = onChange;
        if (r != null) new android.os.Handler(android.os.Looper.getMainLooper()).post(r);
    }

    static synchronized List<Event> all() { return new ArrayList<>(events); }
}
