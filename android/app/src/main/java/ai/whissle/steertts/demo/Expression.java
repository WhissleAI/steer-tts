package ai.whissle.steertts.demo;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * How a piece of text should sound, read from the text itself: the words ("so sorry", "congrats", "haha"), emoji,
 * exclamation marks, a message in capitals, a question. The result is a Mood: a style tag together with the control
 * shifts that make it audible (an emotion tag alone changes little on the edge model; see Style.preset), plus why.
 * Also turns chat text into something a voice can say: "👍" alone becomes "a thumbs up", emoji elsewhere and
 * "lol" are left to the delivery, "idk" becomes "I don't know". Plain Java (no Android), so it is unit tested.
 */
final class Expression {
    private Expression() {}

    /** A delivery: tag (null = keep the speaker's), control shifts to add (z-score units), and why. */
    static final class Mood {
        final String name, tag, why;
        final Map<String, Float> ctrl;

        Mood(String name, String tag, String why, Object... kv) {
            this.name = name;
            this.tag = tag;
            this.why = why;
            ctrl = new LinkedHashMap<>();
            for (int i = 0; i + 1 < kv.length; i += 2) ctrl.put((String) kv[i], ((Number) kv[i + 1]).floatValue());
        }

        boolean neutral() { return tag == null && ctrl.isEmpty(); }

        /** This mood with more shifts on top (a question's rising end, alert loudness). */
        Mood plus(String why2, Object... kv) {
            Object[] all = new Object[ctrl.size() * 2 + kv.length];
            int i = 0;
            for (Map.Entry<String, Float> e : ctrl.entrySet()) {
                all[i++] = e.getKey();
                all[i++] = e.getValue();
            }
            System.arraycopy(kv, 0, all, i, kv.length);
            Mood m = new Mood(name, tag, why.isEmpty() ? why2 : why + "; " + why2);
            for (int k = 0; k + 1 < all.length; k += 2) m.ctrl.merge((String) all[k], ((Number) all[k + 1]).floatValue(), Float::sum);
            return m;
        }
    }

    static final Mood NEUTRAL = new Mood("neutral", null, "");
    static final Mood URGENT = new Mood("urgent", "determined", "urgent: faster, louder", "rate", 0.8, "loud", 1.2, "f0_med", 0.5, "tilt", 0.8, "pause", -0.5);
    static final Mood SAD = new Mood("sad", "sadness", "sad news: slower, lower", "rate", -0.6, "f0_med", -0.7, "f0_iqr", -0.7, "loud", -0.6);
    static final Mood SORRY = new Mood("sorry", "apologetic", "an apology: gentle", "rate", -0.3, "f0_med", -0.4, "f0_iqr", -0.3, "loud", -0.4);
    static final Mood WORRIED = new Mood("worried", "anxiety", "worried: quicker, higher", "rate", 0.4, "f0_med", 0.5, "f0_iqr", 0.6);
    static final Mood ANNOYED = new Mood("annoyed", "irritation", "annoyed: firm", "loud", 0.6, "f0_iqr", 0.5, "rate", 0.3, "f0_end", -0.6);
    static final Mood EXCITED = new Mood("excited", "excited", "excited: livelier, faster", "rate", 0.7, "loud", 0.9, "f0_iqr", 1.4, "f0_med", 0.9);
    static final Mood AMUSED = new Mood("amused", "amusement", "amused: playful", "f0_iqr", 1.0, "f0_med", 0.5, "rate", 0.3);
    static final Mood HAPPY = new Mood("happy", "happiness", "happy: brighter", "f0_iqr", 1.1, "f0_med", 0.6, "rate", 0.3, "tilt", 0.4);
    static final Mood LOVING = new Mood("loving", "contentment", "affectionate: warm", "f0_iqr", 0.6, "h1h2", 0.6, "rate", -0.2);
    static final Mood GRATEFUL = new Mood("grateful", "contentment", "thankful: warm", "f0_iqr", 0.8, "f0_med", 0.3);
    static final Mood SECRET = new Mood("secret", "whisper", "a secret: whispered");
    static final Mood CALM = new Mood("calm", "serenity", "calm", "rate", -0.3, "f0_iqr", -0.5, "loud", -0.3);
    static final Mood CONFIDENT = new Mood("confident", "confidence", "clear", "f0_med", -0.3, "loud", 0.3);

    /** Cues per mood, strongest first in tie breaks: words (one point) and emoji (two). */
    private static final Object[][] CUES = {
            {URGENT, "\\b(urgent(ly)?|asap|emergency|immediately|right now|help me|call me now|911)\\b", "🚨|🆘"},
            {SAD, "\\b(passed away|died|funeral|heartbroken|devastated|so sad|crying|bad news|rip)\\b", "😢|😭|💔|😞|😔|☹️|😿"},
            {SORRY, "\\b(sorry|apologi[sz]e|apologies|my bad|forgive me|unfortunately|(got |been |is |was )?cancell?ed)\\b", "🥺"},
            {WORRIED, "\\b(worried|scared|nervous|afraid|anxious|are you ok(ay)?|is everything ok(ay)?|hope you'?re ok(ay)?)\\b", "😟|😰|😨|😧"},
            {ANNOYED, "\\b(ugh+|annoy(ed|ing)|angry|furious|mad at|wtf|ridiculous|seriously\\?|so done)\\b", "😡|😠|🤬|🙄|😤"},
            {EXCITED, "\\b(congrats|congratulations|amazing|awesome|incredible|can'?t wait|woo+hoo+|yay+|omg|so excited|we won|"
                    + "got the job|happy birthday|let'?s go+|finally)\\b", "🎉|🥳|🎊|🔥|🤩|🎂|🚀|💯"},
            {AMUSED, "\\b(ha(ha)+h?|he(he)+|lol+|lmao+|rofl|too funny|hilarious)\\b", "😂|🤣|😆|😹|😜|😝"},
            {LOVING, "\\b(love you|love u|luv u|miss you|miss u|xoxo|xx)\\b", "❤️|❤|😍|🥰|😘|💕|💖|💗|💞"},
            {GRATEFUL, "\\b(thank you|thanks|thx|ty|appreciate (it|you))\\b", "🙏"},
            {HAPPY, "\\b(great|good news|glad|nice one|well done|perfect|wonderful|good morning|sounds good)\\b", "😊|🙂|😄|😁|😃|☺️|😀|👍|✨"},
            {SECRET, "\\b(don'?t tell|secret|between us|keep it quiet)\\b", "🤫"},
    };
    private static final Pattern[][] PATTERNS = new Pattern[CUES.length][];
    static {
        for (int i = 0; i < CUES.length; ++i)
            PATTERNS[i] = new Pattern[] {Pattern.compile((String) CUES[i][1], Pattern.CASE_INSENSITIVE), Pattern.compile((String) CUES[i][2])};
    }

    /** The mood of a message or a line of dialogue. */
    static Mood of(String text) {
        if (text == null || text.trim().isEmpty()) return NEUTRAL;
        int best = -1, bestScore = 0;
        int[] score = new int[CUES.length];
        for (int i = 0; i < CUES.length; ++i) {
            score[i] = count(PATTERNS[i][0], text) + 2 * count(PATTERNS[i][1], text);
        }
        int bangs = count(Pattern.compile("!"), text);
        if (bangs >= 2 || text.contains("!!")) score[index(EXCITED)] += score[index(ANNOYED)] > 0 ? 0 : 1;
        if (shouting(text)) {
            if (score[index(ANNOYED)] > 0) score[index(ANNOYED)] += 1;
            else score[index(EXCITED)] += 1;
        }
        for (int i = 0; i < CUES.length; ++i)
            if (score[i] > bestScore) {
                bestScore = score[i];
                best = i;
            }
        Mood m = best < 0 ? NEUTRAL : (Mood) CUES[best][0];
        if (shouting(text)) m = m.plus("in capitals: louder", "loud", 0.6);
        if (question(text) && m != URGENT) m = m.plus("question: rising end", "f0_end", 1.4);
        return m;
    }

    private static int index(Mood m) {
        for (int i = 0; i < CUES.length; ++i) if (CUES[i][0] == m) return i;
        return -1;
    }

    private static int count(Pattern p, String s) {
        Matcher m = p.matcher(s);
        int n = 0;
        while (m.find()) n++;
        return n;
    }

    /** Mostly capitals, and long enough to be shouting rather than an acronym. */
    static boolean shouting(String s) {
        int up = 0, letters = 0;
        for (int i = 0; i < s.length(); ++i) {
            char c = s.charAt(i);
            if (Character.isLetter(c)) {
                letters++;
                if (Character.isUpperCase(c)) up++;
            }
        }
        return letters >= 8 && up * 10 >= letters * 8;
    }

    static boolean question(String s) {
        String t = s.trim().replaceAll("[\\s\\p{So}\\p{Sk}\\uFE0F\\u200D]+$", "");
        return t.endsWith("?");
    }

    // ---------------------------------------------------------------- what the voice can say
    private static final String[][] EMOJI_WORDS = {
            {"👍", "a thumbs up"}, {"👎", "a thumbs down"}, {"❤️", "a heart"}, {"❤", "a heart"}, {"😂", "a laughing face"},
            {"🤣", "a laughing face"}, {"😍", "a heart-eyes face"}, {"🎉", "a party popper"}, {"🙏", "folded hands"},
            {"😢", "a crying face"}, {"😭", "a crying face"}, {"😮", "a surprised face"}, {"😡", "an angry face"},
            {"👋", "a wave"}, {"🔥", "a fire emoji"}, {"👏", "applause"}, {"😘", "a kiss"}, {"🥰", "a smiling face with hearts"},
            {"😊", "a smile"}, {"🙂", "a smile"}, {"👌", "an OK hand"}, {"💯", "a hundred points emoji"}};

    private static final String[][] CHAT = {
            {"idk", "I don't know"}, {"btw", "by the way"}, {"brb", "be right back"}, {"omw", "on my way"},
            {"tmrw", "tomorrow"}, {"tmr", "tomorrow"}, {"thx", "thanks"}, {"ty", "thank you"}, {"pls", "please"},
            {"plz", "please"}, {"u", "you"}, {"ur", "your"}, {"omg", "oh my god"}, {"imo", "in my opinion"},
            {"ttyl", "talk to you later"}, {"np", "no problem"}, {"wyd", "what are you doing"}, {"hbu", "how about you"},
            {"gm", "good morning"}, {"gn", "good night"}, {"bday", "birthday"}, {"rn", "right now"}, {"ily", "I love you"}};
    private static final Pattern LAUGH = Pattern.compile("(?i)(^|\\s)(lol+|lmao+|rofl)(?=\\s|[.,!?]|$)");
    private static final Pattern EMOJI = Pattern.compile("[\\x{1F000}-\\x{1FAFF}\\x{2600}-\\x{27BF}\\x{2B00}-\\x{2BFF}\\x{1F1E6}-\\x{1F1FF}"
            + "\\x{FE0F}\\x{200D}\\x{20E3}\\x{E0020}-\\x{E007F}]+");

    /** Only emoji (and spaces): such a message is described ("sent a thumbs up"). */
    static boolean emojiOnly(String s) {
        String t = s.trim();
        return !t.isEmpty() && EMOJI.matcher(t).replaceAll("").trim().isEmpty();
    }

    /** "a thumbs up" for a message that is only emoji; null if it is not. */
    static String describeEmoji(String s) {
        if (!emojiOnly(s)) return null;
        String t = s.trim();
        for (String[] e : EMOJI_WORDS) if (t.startsWith(e[0])) return e[1];
        return "an emoji";
    }

    /**
     * Chat text as words: emoji dropped (the mood carries them), "lol" dropped, chat abbreviations spelled out, letters
     * repeated for effect shortened ("sooooo" -> "soo"), characters outside the Basic Multilingual Plane removed (they
     * would reach the engine as broken UTF-8). Not for documents (no abbreviation expansion there).
     */
    static String speakable(String s) {
        String t = EMOJI.matcher(s).replaceAll(" ");
        t = LAUGH.matcher(t).replaceAll("$1");
        t = t.replaceAll("(\\p{L})\\1{2,}", "$1$1");
        for (String[] c : CHAT) t = t.replaceAll("(?i)(?<![\\w'])" + Pattern.quote(c[0]) + "(?![\\w'])", Matcher.quoteReplacement(c[1]));
        return bmp(t).replaceAll("\\s+([,.!?])", "$1").replaceAll("\\s{2,}", " ").trim();
    }

    /** Removes code points above U+FFFF (JNI passes them to C as invalid UTF-8). */
    static String bmp(String s) {
        StringBuilder b = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            b.append(cp > 0xFFFF ? ' ' : (char) cp);
            i += Character.charCount(cp);
        }
        return b.toString();
    }

    /** "excited: livelier, faster" -> shown in the app as why it sounds the way it does. */
    static String describe(Mood m) { return m.neutral() ? "" : m.why.toLowerCase(Locale.ROOT); }
}
