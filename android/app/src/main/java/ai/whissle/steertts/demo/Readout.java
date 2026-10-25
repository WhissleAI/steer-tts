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

    /**
     * New messages of one conversation, oldest first, as {sender, text} pairs: each run from one person gets its own
     * framing and voice ("Sam in Family on WhatsApp. …  Then Mom. …"), so nobody's words are read as someone else's.
     */
    static Speech conversation(List<String[]> msgs, String group, String app, boolean personVoices, String ownVoice) {
        Speech sp = new Speech();
        int i = 0;
        boolean first = true;
        while (i < msgs.size()) {
            String who = msgs.get(i)[0];
            List<String> run = new java.util.ArrayList<>();
            while (i < msgs.size() && msgs.get(i)[0].equals(who)) run.add(msgs.get(i++)[1]);
            Speech part = message(who, group, app, run, personVoices, ownVoice);
            if (!first && !part.parts.isEmpty()) {
                Speech.Part f = part.parts.get(0);   // later senders: "Then Mom." instead of the full framing again
                String framing = f.text.endsWith(" on " + app + ".") ? "Then " + who + "." : f.text;
                part.parts.set(0, new Speech.Part(framing, f.mood, f.voice, f.pauseAfterMs));
            }
            for (Speech.Part p : part.parts) sp.parts.add(p);
            first = false;
        }
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

    // ---------------------------------------------------------------- other things worth hearing
    /** "You missed a call from Mom." */
    static Speech missedCall(String who) {
        return new Speech().add("You missed a call from " + (who == null || who.trim().isEmpty() ? "an unknown number" : who.trim()) + ".",
                Expression.CONFIDENT);
    }

    /**
     * An alarm, a timer or a reminder: "Reminder: call the dentist." / "Timer: time's up." The title says what it is,
     * the text adds the detail; a clock's bare "Alarm 7:00 AM" is read as it is.
     */
    static Speech reminder(String title, String text, String app) {
        String t = clean(title), b = clean(text);
        String what = t.isEmpty() ? app : t;
        String said = b.isEmpty() || b.equalsIgnoreCase(what) ? what : what + ": " + b;
        String low = said.toLowerCase(java.util.Locale.ROOT);
        String head = low.startsWith("reminder") || low.startsWith("timer") || low.startsWith("alarm") ? "" : "Reminder: ";
        return new Speech().add(sentence(head + said), Expression.CONFIDENT.plus("a reminder: clear", "rate", -0.1));
    }

    /** A delivery or ride update: "Uber: Your driver is arriving." */
    static Speech update(String title, String text, String app) {
        String t = clean(title), b = clean(text);
        Speech sp = new Speech().add(app + ".", null, null, 150);
        String said = t.isEmpty() ? b : b.isEmpty() || b.startsWith(t) ? t : t + ". " + b;
        return sp.add(cap(sentence(said)), Expression.of(said));
    }

    /** "Email from Priya: Contract signed." (the subject only, never the body). */
    static Speech email(String sender, String subject) {
        String who = clean(sender), sub = clean(subject);
        if (who.isEmpty()) who = "someone";
        Speech sp = new Speech().add("Email from " + who + ".", null, null, 150);
        return sub.isEmpty() ? sp : sp.add(cap(sentence(sub)), Expression.of(sub));
    }

    /** "Battery at 15 percent. Time to charge." / "Fully charged." */
    static Speech battery(int level, boolean charging) {
        if (charging && level >= 100) return new Speech().add("Your phone is fully charged.", Expression.HAPPY);
        return new Speech().add("Battery at " + level + " percent. Time to charge.", Expression.CONFIDENT.plus("low battery: clear", "loud", 0.3));
    }

    private static String clean(String s) { return s == null ? "" : Expression.speakable(s).trim(); }

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
