package ai.whissle.steertts.demo;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * How a sentence of a document is performed (plain Java, unit tested): a heading slower with a pause after it;
 * dialogue (words in quotation marks) apart from the narration, in the dialogue voice and the line's own mood
 * ("Run!" excited, "I'm so sorry" gentle), narration in the reader's voice; words the document set in italics or bold
 * stressed (engine markup *word*). Scare quotes ("so-called") stay narration.
 */
final class Narration {
    private Narration() {}

    static final char EM_ON = '', EM_OFF = '', HEAD = '';
    static final Expression.Mood HEADING = new Expression.Mood("heading", null, "a heading: slower", "rate", -0.3, "f0_iqr", 0.3, "loud", 0.2);

    static final class Seg {
        final String text;          // engine text, markup allowed (*word*)
        final boolean dialogue, heading;
        final Expression.Mood mood;
        final int pauseAfterMs;

        Seg(String text, boolean dialogue, boolean heading, Expression.Mood mood, int pauseAfterMs) {
            this.text = text;
            this.dialogue = dialogue;
            this.heading = heading;
            this.mood = mood;
            this.pauseAfterMs = pauseAfterMs;
        }
    }

    static List<Seg> segments(String sentence) { return segments(sentence, null); }

    /**
     * line: the paragraph's reading from Cast (who speaks, how, the scene), or null. Dialogue gets the speaker's
     * delivery (whispered, shouted…) or, without one, the mood of its own words; narration gets the scene's mood.
     */
    static List<Seg> segments(String sentence, Cast.Line line) {
        List<Seg> out = new ArrayList<>();
        if (sentence.isEmpty()) return out;
        if (sentence.charAt(0) == HEAD) {
            out.add(new Seg(markup(sentence.substring(1)), false, true, HEADING, 650));
            return out;
        }
        Matcher m = Cast.QUOTE.matcher(sentence);
        Expression.Mood scene = line != null ? line.scene : Expression.NEUTRAL;
        int at = 0;
        while (m.find()) {
            String inner = Cast.inner(m);
            if (!isDialogue(inner)) continue;
            narration(out, sentence.substring(at, m.start()), scene);
            Expression.Mood own = Expression.of(display(inner));
            Expression.Mood mood = line == null || line.delivery == null ? own
                    : line.delivery.tag == null ? (own.neutral() ? line.delivery : own.plus(line.delivery.why, flat(line.delivery)))
                    : line.delivery;
            out.add(new Seg(markup(inner), true, false, mood, 120));
            at = m.end();
        }
        narration(out, sentence.substring(at), scene);
        return out;
    }

    private static Object[] flat(Expression.Mood m) {
        Object[] kv = new Object[m.ctrl.size() * 2];
        int i = 0;
        for (java.util.Map.Entry<String, Float> e : m.ctrl.entrySet()) {
            kv[i++] = e.getKey();
            kv[i++] = e.getValue();
        }
        return kv;
    }

    private static void narration(List<Seg> out, String s, Expression.Mood scene) {
        String t = s.trim();
        if (t.replaceAll("[\\p{Punct}\\s“”‘’—–]", "").isEmpty()) return;   // only a comma or a dash between quotes
        out.add(new Seg(markup(t), false, false, scene, 120));
    }

    /** Spoken words, not a scare quote or a title: three words or more, or a line ending in punctuation. */
    static boolean isDialogue(String inner) {
        String t = inner.trim();
        return t.split("\\s+").length >= 3 || t.matches(".*[.!?,]$");
    }

    /**
     * Engine text: characters the engine would read as markup ([ ] < > *) made plain, then emphasis markers as *...*
     * when they pair up in this piece (a sentence split inside an italic run just loses the stress).
     */
    static String markup(String s) {
        String t = s.replace('[', '(').replace(']', ')').replace('<', ' ').replace('>', ' ').replace('*', ' ');
        int on = 0, off = 0;
        for (int i = 0; i < t.length(); ++i) {
            if (t.charAt(i) == EM_ON) on++;
            else if (t.charAt(i) == EM_OFF) off++;
        }
        if (on == off && on > 0) t = t.replace(EM_ON, '*').replace(EM_OFF, '*').replaceAll("\\*\\s*\\*", " ");
        else t = strip(t);
        return t.replaceAll("\\s{2,}", " ").trim();
    }

    /** For the screen: the markers removed. */
    static String display(String s) { return strip(s).trim(); }

    static String strip(String s) {
        return s.indexOf(EM_ON) < 0 && s.indexOf(EM_OFF) < 0 && s.indexOf(HEAD) < 0 ? s
                : s.replace(String.valueOf(EM_ON), "").replace(String.valueOf(EM_OFF), "").replace(String.valueOf(HEAD), "");
    }
}
