package ai.whissle.steertts.demo;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * What the assistant says for a message or a call, as a Speech (plain Java, unit tested): the framing ("Sam on
 * WhatsApp") in the assistant's voice, then the message in its own mood, and in the sender's voice when people have
 * voices of their own. A message that is only emoji is described ("Sam sent a thumbs up"). Calls get more urgent when
 * the same caller rings again within a quarter of an hour.
 */
final class Readout {
    private Readout() {}

    static final int MAX_CHARS = 300;

    /**
     * person: the sender (null if unknown); group: the conversation's title when it is a group; app: the app's name;
     * texts: the new messages, oldest first.
     */
    static Speech message(String person, String group, String app, List<String> texts, boolean personVoices, String ownVoice) {
        Speech sp = new Speech();
        StringBuilder body = new StringBuilder();
        String onlyEmoji = null;
        for (String t : texts) {
            String d = Expression.describeEmoji(t);
            if (d != null && texts.size() == 1) onlyEmoji = d;
            else if (d == null) body.append(body.length() > 0 ? " " : "").append(sentence(t));
        }
        String who = person == null ? app : group != null ? person + " in " + group : person;
        if (onlyEmoji != null) return sp.add(who + (person == null ? ": " : " sent ") + onlyEmoji + ".");
        String said = cap(body.toString());
        if (said.isEmpty()) return sp;
        String framing = person == null ? app + "." : who + " on " + app + ".";
        sp.add(framing, null, null, 180);
        // the mood is read from the original words (emoji, "lol" and capitals carry it), the words spoken are cleaned
        StringBuilder raw = new StringBuilder();
        for (String t : texts) raw.append(t).append(' ');
        String voice = personVoices && person != null ? Voices.forPerson(person, ownVoice) : null;
        sp.add(said, Expression.of(raw.toString()), voice, 0);
        return sp;
    }

    /** A notification that is not a conversation: its title (a subject, a headline) and text. */
    static Speech notice(String title, String text, String app) {
        Speech sp = new Speech();
        String t = title == null ? null : Expression.speakable(title), b = text == null ? null : cap(sentence(text));
        if ((t == null || t.isEmpty()) && (b == null || b.isEmpty())) return sp;
        sp.add(t == null || t.isEmpty() ? app + "." : t + ", on " + app + ".", null, null, 150);
        if (b != null && !b.isEmpty()) sp.add(b, Expression.of(text), null, 0);
        return sp;
    }

    /** One message as at least one sentence, in words a voice can say. */
    static String sentence(String text) {
        String s = Expression.speakable(text);
        if (s.isEmpty()) return s;
        char c = s.charAt(s.length() - 1);
        return c == '.' || c == '!' || c == '?' ? s : s + ".";
    }

    static String cap(String s) {
        s = s.replaceAll("\\s+", " ").trim();
        if (s.length() <= MAX_CHARS) return s;
        int cut = s.lastIndexOf(' ', MAX_CHARS);
        return s.substring(0, cut > 0 ? cut : MAX_CHARS) + ", and more.";
    }

    // ---------------------------------------------------------------- calls
    private static final long REPEAT_WINDOW = 15 * 60_000L;
    private static final Map<String, Deque<Long>> rings = new HashMap<>();

    /** How many times this caller has rung within the window, this call included. */
    static synchronized int ringCount(String caller, long now) {
        Deque<Long> d = rings.computeIfAbsent(caller.toLowerCase(java.util.Locale.ROOT), k -> new ArrayDeque<>());
        while (!d.isEmpty() && now - d.peekFirst() > REPEAT_WINDOW) d.pollFirst();
        d.addLast(now);
        return d.size();
    }

    /** "Incoming call from Mom." / "Mom is calling again." / "Mom is calling for the third time. It may be urgent." */
    static Speech call(String caller, String via, int count) {
        String how = via == null ? "" : via + " ";
        if (count <= 1) return new Speech().add("Incoming " + how + "call from " + caller + ".", Expression.CONFIDENT);
        if (count == 2)
            return new Speech().add(caller + " is calling again.", Expression.CONFIDENT.plus("again: more present", "loud", 0.5, "f0_med", 0.3));
        return new Speech().add(caller + " is calling for the " + ordinal(count) + " time. It may be urgent.", Expression.URGENT);
    }

    private static String ordinal(int n) {
        String[] w = {"", "first", "second", "third", "fourth", "fifth", "sixth"};
        return n < w.length ? w[n] : n + "th";
    }
}
