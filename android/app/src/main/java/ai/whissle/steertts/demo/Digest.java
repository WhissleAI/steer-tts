package ai.whissle.steertts.demo;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Messages that arrived while they could not be read out (no headphones, Do Not Disturb), for "catch me up": when
 * headphones connect, from the Quick Settings tile, or the Assistant tab. Memory only (never written to storage); a
 * message leaves when its notification is dismissed or opened (the user has seen it), when it has been caught up on,
 * or after twelve hours.
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

    static synchronized void add(String key, String person, String app, String text) {
        items.add(new Item(key, person, app, text, System.currentTimeMillis()));
        while (items.size() > MAX_ITEMS) items.remove(0);
    }

    /** The notification is gone: the user has seen or dismissed it. */
    static synchronized void removeKey(String key) { items.removeIf(i -> i.key.equals(key)); }

    static synchronized int size() {
        prune();
        return items.size();
    }

    static synchronized void clear() { items.clear(); }

    private static void prune() {
        long old = System.currentTimeMillis() - MAX_AGE;
        items.removeIf(i -> i.time < old);
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
        return sp;
    }

    static String names(List<String> names) {
        if (names.size() == 1) return names.get(0);
        if (names.size() <= 3)
            return String.join(", ", names.subList(0, names.size() - 1)) + " and " + names.get(names.size() - 1);
        return names.get(0) + ", " + names.get(1) + " and " + (names.size() - 2) + " others";
    }
}
