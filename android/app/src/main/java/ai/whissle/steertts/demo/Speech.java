package ai.whissle.steertts.demo;

import java.util.ArrayList;
import java.util.List;

/**
 * One thing the assistant says, as parts that may sound different: "Sam on WhatsApp" in the assistant's own voice,
 * then Sam's message in Sam's voice and the message's mood ("I'm so sorry..." gently), then a short pause. The
 * Announcer renders each part with the user's style, adjusted for the situation (ContextStyler) and then the part's
 * mood and voice.
 */
final class Speech {
    static final class Part {
        final String text;
        final Expression.Mood mood;
        final String voice;        // null = the user's voice
        final int pauseAfterMs;

        Part(String text, Expression.Mood mood, String voice, int pauseAfterMs) {
            this.text = text;
            this.mood = mood == null ? Expression.NEUTRAL : mood;
            this.voice = voice;
            this.pauseAfterMs = pauseAfterMs;
        }
    }

    final List<Part> parts = new ArrayList<>();

    static Speech of(String text) { return new Speech().add(text); }

    Speech add(String text) { return add(text, null, null, 0); }

    Speech add(String text, Expression.Mood mood) { return add(text, mood, null, 0); }

    Speech add(String text, Expression.Mood mood, String voice, int pauseAfterMs) {
        if (text != null && !text.trim().isEmpty()) parts.add(new Part(text.trim(), mood, voice, pauseAfterMs));
        return this;
    }

    Speech pause(int ms) {
        if (!parts.isEmpty()) {
            Part p = parts.remove(parts.size() - 1);
            parts.add(new Part(p.text, p.mood, p.voice, p.pauseAfterMs + ms));
        }
        return this;
    }

    boolean isEmpty() { return parts.isEmpty(); }

    /** All the words, for the screen and the log. */
    String text() {
        StringBuilder b = new StringBuilder();
        for (Part p : parts) b.append(b.length() > 0 ? " " : "").append(p.text);
        return b.toString();
    }

    /** The strongest mood in it (for the now-speaking card). */
    Expression.Mood mainMood() {
        for (Part p : parts) if (!p.mood.neutral()) return p.mood;
        return Expression.NEUTRAL;
    }
}
