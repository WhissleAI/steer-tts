package ai.whissle.steertts.demo;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Messages that arrived while they could not be read out (no headphones, Do Not Disturb), for "catch me up": when
 * headphones connect, from the Quick Settings tile, or the Assistant tab. The words are held in memory only, never
 * written to storage; only the notifications' keys are saved, so after a restart (an app update, the process ended)
 * the listener rebuilds the list from the notifications still showing (restore). A message leaves when its
 * notification is dismissed or opened (the user has seen it), when it has been caught up on, or after twelve hours.
 */
final class Digest {
    private Digest() {}

    static final class Item {
        final String key, person, app, text;
        final long time;

        Item(String key, String person, String app, String text, long time) {
            this.key = key;
            this.person = person;
            this.app = app;
            this.text = text;
            this.time = time;
        }
    }

    private static final long MAX_AGE = 12 * 3600_000L;
    private static final int MAX_ITEMS = 40;
    private static final List<Item> items = new ArrayList<>();

    private static java.io.File store;
    private static final java.util.Map<String, Long> saved = new java.util.HashMap<>();   // key -> when it was kept

    /** Loads the saved keys once (what was kept before a restart, for restore). */
    static synchronized void init(android.content.Context c) {
        if (store != null) return;
        store = new java.io.File(c.getApplicationContext().getFilesDir(), "kept.keys");
        try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader(store))) {
            for (String line; (line = r.readLine()) != null; ) {
                int tab = line.lastIndexOf('\t');
                if (tab > 0) saved.put(line.substring(0, tab), Long.parseLong(line.substring(tab + 1)));
            }
        } catch (Exception ignored) {   // nothing kept
        }
    }

    /** Keys kept before a restart and not restored yet. */
    static synchronized java.util.Set<String> savedKeys() { return new java.util.HashSet<>(saved.keySet()); }

    /** A message kept before a restart, read again from its notification (keeps its first time, not added twice). */
    static synchronized void restore(String key, String person, String app, String text) {
        Long t = saved.get(key);
        if (t == null) return;
        for (Item i : items) if (i.key.equals(key) && i.text.equals(text)) return;
        items.add(new Item(key, person, app, text, t));
    }

    /** After a restore: forget keys whose notifications are gone (seen or dismissed while the app was not running). */
    static synchronized void forgetExcept(java.util.Set<String> active) {
        saved.keySet().retainAll(active);
        persist();
    }

    private static void persist() {
        if (store == null) return;
        try (java.io.FileWriter w = new java.io.FileWriter(store)) {
            for (java.util.Map.Entry<String, Long> e : saved.entrySet()) w.write(e.getKey() + "\t" + e.getValue() + "\n");
        } catch (Exception ignored) {
        }
    }

    static synchronized void add(String key, String person, String app, String text) {
        long now = System.currentTimeMillis();
        items.add(new Item(key, person, app, text, now));
        while (items.size() > MAX_ITEMS) items.remove(0);
        if (!saved.containsKey(key)) {
            saved.put(key, now);
            persist();
        }
    }

    /** The notification is gone: the user has seen or dismissed it. */
    static synchronized void removeKey(String key) {
        items.removeIf(i -> i.key.equals(key));
        if (saved.remove(key) != null) persist();
    }

    static synchronized int size() {
        prune();
        return items.size();
    }

    static synchronized void clear() {
        items.clear();
        saved.clear();
        persist();
    }

    private static void prune() {
        long old = System.currentTimeMillis() - MAX_AGE;
        items.removeIf(i -> i.time < old);
        if (saved.values().removeIf(t -> t < old)) persist();
    }

    /**
     * "You have 4 new messages, from Sam and Mom." then each person's messages, newest last, in their mood (and voice,
     * when people have voices of their own). Empties the digest.
     */
    static synchronized Speech take(boolean personVoices, String ownVoice) {
        prune();
        Speech sp = new Speech();
        if (items.isEmpty()) return sp.add("You're all caught up. No new messages.");
        Map<String, List<Item>> byPerson = new LinkedHashMap<>();
        for (Item i : items) byPerson.computeIfAbsent(i.person, k -> new ArrayList<>()).add(i);
        int n = items.size();
        List<String> names = new ArrayList<>(byPerson.keySet());
        sp.add((n == 1 ? "You have one new message" : "You have " + n + " new messages") + ", from " + names(names) + ".",
                Expression.CONFIDENT, null, 350);
        int people = 0;
        for (Map.Entry<String, List<Item>> e : byPerson.entrySet()) {
            if (++people > 6) {
                sp.add("And more from " + (byPerson.size() - 6) + " others.");
                break;
            }
            List<Item> l = e.getValue();
            List<Item> last = l.subList(Math.max(0, l.size() - 3), l.size());
            StringBuilder said = new StringBuilder();
            for (Item i : last) said.append(said.length() > 0 ? " " : "").append(Readout.sentence(i.text));
            String body = said.toString();
            sp.add((l.size() > last.size() ? "The latest from " : "From ") + e.getKey() + ".", null, null, 150);
            sp.add(body, Expression.of(body), personVoices ? Voices.forPerson(e.getKey(), ownVoice) : null, 400);
        }
        items.clear();
        saved.clear();
        persist();
        return sp;
    }

    static String names(List<String> names) {
        if (names.size() == 1) return names.get(0);
        if (names.size() <= 3)
            return String.join(", ", names.subList(0, names.size() - 1)) + " and " + names.get(names.size() - 1);
        return names.get(0) + ", " + names.get(1) + " and " + (names.size() - 2) + " others";
    }
}
