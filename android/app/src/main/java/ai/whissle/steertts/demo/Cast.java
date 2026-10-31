package ai.whissle.steertts.demo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Who speaks in a story, and how (plain Java, unit tested). Read once per book, paragraph by paragraph, the way a
 * reader follows it: "…,” said Tom / Tom said / she whispered / said the Queen name the speaker of a paragraph's
 * dialogue; "he" and "she" stand for the last man or woman named; in an exchange with no names, the two speakers take
 * turns. Each character gets a voice of their own (Voices), the speech verb says how the line is delivered (whispered,
 * shouted, sobbed, laughed), and a light mood is read from the narration (a tense scene, a sad one, a happy one).
 */
final class Cast {
    /** One paragraph: its speaker (null when unknown or no dialogue), how the line is said, and the scene's mood. */
    static final class Line {
        final String speaker, gender;
        final Expression.Mood delivery, scene;

        Line(String speaker, String gender, Expression.Mood delivery, Expression.Mood scene) {
            this.speaker = speaker;
            this.gender = gender;
            this.delivery = delivery;
            this.scene = scene;
        }
    }

    final Line[] lines;
    final Map<String, String> gender = new LinkedHashMap<>();   // character -> "female" / "male" / null (unknown)

    private Cast(int n) { lines = new Line[n]; }

    // ---------------------------------------------------------------- dialogue
    /** Curly double quotes, straight double quotes, and curly single quotes (an apostrophe inside a word is not a close). */
    // straight single quotes too ('Now, my dears,' said old Mrs. Rabbit): an apostrophe between letters (don't,
    // McGregor's) is never a quote mark
    static final Pattern QUOTE = Pattern.compile("“([^”]+)”|\"([^\"]+)\"|‘((?:[^’]|’(?=\\p{L}))+)’(?!\\p{L})"
            + "|(?<![\\p{L}\\d])'((?:[^']|'(?=\\p{L}))+?)'(?!\\p{L})");

    static boolean hasDialogue(String p) {
        Matcher m = QUOTE.matcher(p);
        while (m.find()) if (Narration.isDialogue(inner(m))) return true;
        return false;
    }

    static String inner(Matcher m) {
        for (int g = 1; g <= 4; ++g) if (m.group(g) != null) return m.group(g);
        return "";
    }

    /** The paragraph with its quoted words blanked out: what the narrator says around the dialogue. */
    static String outside(String p) {
        Matcher m = QUOTE.matcher(p);
        StringBuffer b = new StringBuffer();
        while (m.find()) m.appendReplacement(b, Narration.isDialogue(inner(m)) ? " ¶ " : Matcher.quoteReplacement(m.group()));
        m.appendTail(b);
        return b.toString();
    }

    // ---------------------------------------------------------------- speech verbs
    private static final Expression.Mood WHISPER = new Expression.Mood("whispered", "whisper", "whispered");
    private static final Expression.Mood SHOUT = new Expression.Mood("shouted", "loud", "shouted: loud", "loud", 1.4, "rate", 0.3, "f0_med", 0.6, "f0_iqr", 0.8);
    private static final Expression.Mood MUTTER = new Expression.Mood("muttered", "irritation", "muttered: low", "loud", -0.8, "f0_med", -0.4, "rate", 0.2);
    private static final Expression.Mood STAMMER = new Expression.Mood("stammered", "hesitant", "hesitant", "rate", -0.4, "pause", 0.6);
    private static final Expression.Mood GASP = new Expression.Mood("gasped", "amazement", "surprised", "f0_med", 0.8, "f0_iqr", 1.0);
    private static final Expression.Mood PLEAD = new Expression.Mood("pleaded", "distress", "pleading", "f0_med", 0.4, "f0_iqr", 0.6, "rate", -0.2);
    private static final Expression.Mood SIGH = new Expression.Mood("sighed", "tired", "tired", "rate", -0.4, "f0_iqr", -0.5, "loud", -0.4);
    private static final Expression.Mood GROWL = new Expression.Mood("growled", "anger", "angry", "loud", 0.8, "f0_med", -0.6, "f0_end", -0.8);
    private static final Expression.Mood ASK = new Expression.Mood("asked", null, "a question", "f0_end", 1.2);

    private static final Object[][] VERBS = {
            {WHISPER, "whisper(ed|s)?|murmur(ed|s)?|breathed"},
            {SHOUT, "shout(ed|s)?|yell(ed|s)?|scream(ed|s)?|roar(ed|s)?|bellow(ed|s)?|cried out|exclaim(ed|s)?|called out"},
            {Expression.SAD, "sob(bed|s)?|wept|weep(s)?|wail(ed|s)?"},
            {Expression.AMUSED, "laugh(ed|s)?|giggl(ed|es)|chuckl(ed|es)|grinn?ed"},
            {GROWL, "growl(ed|s)?|snarl(ed|s)?|snapp?ed|hiss(ed|es)?|thunder(ed|s)?"},
            {MUTTER, "mutter(ed|s)?|grumbl(ed|es)|mumbl(ed|es)"},
            {STAMMER, "stammer(ed|s)?|stutter(ed|s)?|faltered"},
            {GASP, "gasp(ed|s)?"},
            {PLEAD, "plead(ed|s)?|pled|begg?ed|implor(ed|es)"},
            {SIGH, "sigh(ed|s)?"},
            {ASK, "ask(ed|s)?|enquir(ed|es)|inquir(ed|es)|wonder(ed|s)?"},
            {null, "said|says|say|replied|replies|answered|added|continued|went on|told|remarked|repeated|agreed|explained|"
                    + "began|called|cried|insisted|announced|declared|suggested|interrupted|thought|admitted|observed"},
    };
    private static final String VERB;
    static {
        StringBuilder b = new StringBuilder();
        for (Object[] v : VERBS) b.append(b.length() > 0 ? "|" : "").append(((String) v[1]).replace("(", "(?:"));   // no groups of their own
        VERB = "(?:" + b + ")";
    }
    private static final Pattern[] VERB_PATTERNS = new Pattern[VERBS.length];
    static {
        for (int i = 0; i < VERBS.length; ++i) VERB_PATTERNS[i] = Pattern.compile("(?i)^(?:" + VERBS[i][1] + ")$");
    }

    static Expression.Mood delivery(String verb) {
        if (verb == null) return null;
        String v = verb.toLowerCase(Locale.ROOT).trim();
        for (int i = 0; i < VERBS.length; ++i) if (VERB_PATTERNS[i].matcher(v).matches()) return (Expression.Mood) VERBS[i][0];
        return null;
    }

    // a name: one to three capitalised words, or "the" + one or two capitalised words ("the Queen", "the March Hare")
    private static final String NAME = "((?:the |The )?(?:Mr\\.|Mrs\\.|Ms\\.|Miss|Dr\\.|Sir|Lady|Lord|Aunt|Uncle|Captain|King|Queen|Prince|Princess)?\\s?"
            + "\\p{Lu}[\\p{L}’'-]+(?: \\p{Lu}[\\p{L}’'-]+){0,2})";
    private static final String ADJ = "(?:(?:old|young|little|poor|dear|kind|good|wise|tall|small|big)\\s+)?";   // "said old Mrs. Rabbit"
    private static final String WHO = ADJ + "(" + NAME.substring(1, NAME.length() - 1) + "|he|she|He|She|I)";
    // after the quote: "”, said Tom" / "” said the Queen" / "” Tom said" / "” she whispered"
    private static final Pattern AFTER_VS = Pattern.compile("^\\s*[,.!?—–-]?\\s*(" + VERB + ")\\s+" + WHO + "(?=[\\s,.;!?]|$)");
    private static final Pattern AFTER_SV = Pattern.compile("^\\s*[,.!?—–-]?\\s*" + WHO + "\\s+(?:\\w+ly\\s+)?(" + VERB + ")\\b");
    // an unnamed speaker: "said our strange visitor", "said a voice": someone, for turn-taking
    private static final Pattern AFTER_SOMEONE = Pattern.compile("^\\s*[,.!?—–-]?\\s*(" + VERB + ")\\s+(?:our|the|a|an|his|her|my|their|old|young)\\s+\\p{Ll}");
    // before the quote: "Tom said, “" / "She whispered: “"
    private static final Pattern BEFORE_SV = Pattern.compile(WHO + "\\s+(?:\\w+ly\\s+)?(" + VERB + ")[^.!?]{0,40}$");

    private static final Set<String> NOT_NAMES = new HashSet<>(java.util.Arrays.asList(
            "The", "Then", "But", "And", "It", "They", "We", "You", "There", "This", "That", "What", "When", "Where", "Why",
            "How", "Yes", "No", "Oh", "Well", "So", "If", "As", "At", "In", "On", "Now", "Here", "Just", "All", "One",
            "His", "Her", "Its", "Our", "Their", "My", "Your", "A", "An", "Nobody", "Everyone", "Someone", "Somebody"));

    private static final Set<String> FEMALE = new HashSet<>(java.util.Arrays.asList(
            "alice", "mary", "anna", "emma", "elizabeth", "jane", "lucy", "sarah", "sophie", "grace", "mia", "olivia", "ava",
            "lily", "ella", "hannah", "kate", "katie", "laura", "rose", "ruby", "amy", "beth", "meg", "jo", "wendy", "dorothy",
            "susan", "margaret", "helen", "clara", "heidi", "matilda", "hermione", "ginny", "luna", "molly", "petunia", "anne",
            "diana", "marianne", "eleanor", "catherine", "isabella", "harriet", "fanny", "emily", "charlotte", "maria",
            "nora", "zoe", "chloe", "julia", "rachel", "nancy", "betty", "linda", "karen", "lisa", "amelia", "isla",
            "mother", "mum", "mom", "mama", "grandma", "granny", "sister", "aunt", "queen", "princess", "duchess", "lady",
            "girl", "woman", "witch", "mrs", "miss", "ms", "madam", "wife", "daughter", "hen", "mrs."));
    private static final Set<String> MALE = new HashSet<>(java.util.Arrays.asList(
            "tom", "john", "james", "peter", "harry", "ron", "jack", "sam", "ben", "george", "william", "henry", "charles",
            "edward", "robert", "david", "michael", "paul", "mark", "luke", "matthew", "oliver", "noah", "leo", "max", "finn",
            "arthur", "frank", "fred", "bob", "jim", "joe", "huck", "oliver", "pip", "darcy", "bingley", "heathcliff",
            "sherlock", "watson", "holmes", "frodo", "bilbo", "gandalf", "aragorn", "sam", "merry", "pippin", "dumbledore",
            "snape", "hagrid", "draco", "neville", "father", "dad", "papa", "grandpa", "grandfather", "brother", "uncle",
            "king", "prince", "duke", "lord", "sir", "boy", "man", "wizard", "mr", "mr.", "husband", "son", "captain",
            "rabbit", "hatter", "caterpillar", "mouse", "dodo", "gryphon", "knave"));

    /** "female", "male" or null for a name, from titles, family words and common first names. */
    static String genderOf(String name) {
        for (String w : name.toLowerCase(Locale.ROOT).replace("the ", "").split("[\\s.]+")) {
            if (FEMALE.contains(w)) return "female";
            if (MALE.contains(w)) return "male";
        }
        return null;
    }

    /** "the Queen" -> "Queen", "the Rabbit’s" -> "Rabbit", "Mr. Hall" stays: one key per character. */
    static String canonical(String name) {
        return name.replaceFirst("^(the|The)\\s+", "").replaceFirst("[’']s$", "").trim();
    }

    /**
     * The character a name refers to: an earlier one whose name contains this one or is contained in it ("Rabbit" and
     * "White Rabbit", "Mock" and "Mock Turtle") is the same character; otherwise a new one.
     */
    private String resolve(String name) {
        List<String> words = java.util.Arrays.asList(name.split(" "));
        for (String known : gender.keySet()) {
            List<String> kw = java.util.Arrays.asList(known.split(" "));
            if (kw.containsAll(words) || words.containsAll(kw)) return known;
        }
        return name;
    }

    // ---------------------------------------------------------------- scenes
    private static final Expression.Mood TENSE = new Expression.Mood("tense", "mysterious", "a tense scene", "rate", -0.15, "loud", -0.25, "f0_iqr", -0.2);
    private static final Expression.Mood SOMBRE = new Expression.Mood("sombre", "sadness", "a sad scene", "rate", -0.25, "f0_med", -0.25, "f0_iqr", -0.3);
    private static final Expression.Mood BRIGHT = new Expression.Mood("bright", "contentment", "a happy scene", "f0_iqr", 0.35, "f0_med", 0.15);
    private static final Expression.Mood ACTION = new Expression.Mood("action", null, "action", "rate", 0.3, "loud", 0.2);
    private static final Pattern P_TENSE = Pattern.compile("(?i)\\b(dark(ness)?|shadow|creep|crept|silence|silent|whisper|fear|afraid|terrif|horror|dread|ghost|cold|trembl|shiver|eerie|strange noise|footsteps)\\w*");
    private static final Pattern P_SOMBRE = Pattern.compile("(?i)\\b(tears|wept|weeping|grief|griev|funeral|died|death|dead|mourn|lonely|alone|sorrow|sad)\\w*");
    private static final Pattern P_BRIGHT = Pattern.compile("(?i)\\b(laugh|smil|joy|happ|delight|sunshine|cheer|merr|danced|sang|wonderful)\\w*");
    private static final Pattern P_ACTION = Pattern.compile("(?i)\\b(ran|run|rushed|dashed|leapt|jumped|crash|suddenly|chased|fled|raced|grabbed|burst)\\w*");

    /** A light mood for narration, from the paragraph's words (two cues of a kind; the strongest kind wins). */
    static Expression.Mood scene(String p) {
        int t = count(P_TENSE, p), s = count(P_SOMBRE, p), b = count(P_BRIGHT, p), a = count(P_ACTION, p);
        int max = Math.max(Math.max(t, s), Math.max(b, a));
        if (max < 2) return Expression.NEUTRAL;
        return max == s ? SOMBRE : max == t ? TENSE : max == a ? ACTION : BRIGHT;
    }

    private static int count(Pattern p, String s) {
        Matcher m = p.matcher(s);
        int n = 0;
        while (m.find()) n++;
        return n;
    }

    // ---------------------------------------------------------------- the reading
    static Cast analyze(List<String> paragraphs) {
        Cast c = new Cast(paragraphs.size());
        String lastFemale = null, lastMale = null, last = null, beforeLast = null;
        boolean prevDialogue = false, prevPrevDialogue = false;
        for (int i = 0; i < paragraphs.size(); ++i) {
            String p = Narration.display(paragraphs.get(i));
            Expression.Mood scene = scene(p);
            if (!hasDialogue(p)) {
                // names in narration are remembered for "he" / "she" later
                for (String n : namesIn(p)) {
                    String g = c.genderFor(n);
                    if ("female".equals(g)) lastFemale = c.resolve(n);
                    else if ("male".equals(g)) lastMale = c.resolve(n);
                }
                c.lines[i] = new Line(null, null, null, scene);
                prevPrevDialogue = prevDialogue;
                prevDialogue = false;
                continue;
            }
            String who = null, verb = null;
            Matcher q = QUOTE.matcher(p);
            int searchFrom = 0;
            while (q.find() && who == null) {
                if (!Narration.isDialogue(inner(q))) continue;
                String after = p.substring(q.end());
                Matcher m;
                if ((m = AFTER_VS.matcher(after)).find()) { verb = m.group(1); who = m.group(2); }
                else if ((m = AFTER_SOMEONE.matcher(after)).find()) { verb = m.group(1); who = "?"; }
                else if ((m = AFTER_SV.matcher(after)).find()) { who = m.group(1); verb = m.group(2); }
                else if ((m = BEFORE_SV.matcher(p.substring(searchFrom, q.start()))).find()) { who = m.group(1); verb = m.group(2); }
                searchFrom = q.end();
            }
            String speaker = null;
            if (who != null) {
                String w = who.trim();
                if (w.equals("~")) { /* speaker already found in the narration */ }
                else if (w.equals("?")) speaker = null;   // someone unnamed
                else if (w.equalsIgnoreCase("she")) speaker = lastFemale;
                else if (w.equalsIgnoreCase("he")) speaker = lastMale;
                else if (w.equals("I")) speaker = "Narrator";
                else if (!NOT_NAMES.contains(w.split(" ")[0])) speaker = c.resolve(canonical(w));
            }
            if (who == null) {   // no "said X": the narration around the quote names the speaker, or "she" / "he" does
                String narr = outside(p);
                int best = Integer.MAX_VALUE;
                for (String known : c.gender.keySet()) {
                    Matcher km = c.namePattern(known).matcher(narr);
                    if (km.find() && km.start() < best) {
                        best = km.start();
                        speaker = known;
                    }
                }
                if (speaker == null) {
                    Matcher pm = Pattern.compile("\\b(she|he|She|He)\\b").matcher(narr);
                    if (pm.find()) speaker = pm.group(1).equalsIgnoreCase("she") ? lastFemale : lastMale;
                }
                if (speaker != null) who = "~";   // found in the narration: no turn-taking guess below
            }
            if (who == null && prevDialogue) {
                if (prevPrevDialogue && beforeLast != null && !beforeLast.equals(last))
                    speaker = beforeLast.equals("?") ? null : beforeLast;   // a back-and-forth without names: turns
                else if (last != null && !last.equals("?"))
                    speaker = last;   // no exchange going on: the last speaker goes on (a monologue over paragraphs)
            }
            String g = null;
            if (speaker != null && !speaker.equals("Narrator")) {
                g = c.genderFor(speaker);
                if (g == null && who != null && (who.equalsIgnoreCase("she") || who.equalsIgnoreCase("he")))
                    g = who.equalsIgnoreCase("she") ? "female" : "male";
                c.gender.putIfAbsent(speaker, g);
                if (g != null && c.gender.get(speaker) == null) c.gender.put(speaker, g);
                if ("female".equals(g)) lastFemale = speaker;
                else if ("male".equals(g)) lastMale = speaker;
            }
            String turn = speaker != null ? speaker : "?";   // the unnamed count as one more person in the exchange
            if (!turn.equals(last)) {
                beforeLast = last;
                last = turn;
            }
            c.lines[i] = new Line(speaker, g, delivery(verb), scene);
            prevPrevDialogue = prevDialogue;
            prevDialogue = true;
        }
        return c;
    }

    private final Map<String, Pattern> namePatterns = new HashMap<>();

    private Pattern namePattern(String name) {
        return namePatterns.computeIfAbsent(name, k -> Pattern.compile("\\b" + Pattern.quote(k) + "\\b"));
    }

    private String genderFor(String name) {
        String known = gender.get(name);
        return known != null ? known : genderOf(name);
    }

    private static final Pattern NAME_IN_TEXT = Pattern.compile(
            "(?<![.!?]\\s)(?<!^)\\b((?:(?:Mr|Mrs|Ms|Dr)\\.|Miss|Sir|Lady|Lord)\\s)?(\\p{Lu}[a-z]{2,})\\b");
    private static final Set<String> TITLES = new HashSet<>(java.util.Arrays.asList("Mr", "Mrs", "Ms", "Miss", "Dr", "Sir", "Lady", "Lord"));

    /** Capitalised words that are not the first word of a sentence: likely names. */
    private static List<String> namesIn(String p) {
        List<String> out = new ArrayList<>();
        Matcher m = NAME_IN_TEXT.matcher(p);
        while (m.find()) {
            String n = m.group(2);
            if (NOT_NAMES.contains(n) || TITLES.contains(n)) continue;
            out.add(m.group(1) != null ? m.group(1) + n : n);   // "Mrs. Hudson", not "Mrs"
        }
        return out;
    }

    // ---------------------------------------------------------------- voices
    private final Map<String, String> voices = new HashMap<>();

    /**
     * The voice of a character: one of the catalogue voices of their gender (either when unknown), never the
     * narrator's, and different from the characters before them as long as there are voices to go round.
     */
    synchronized String voiceFor(String speaker, String narrator, List<String> female, List<String> male) {
        if (speaker == null) return null;
        if (speaker.equals("Narrator")) return narrator;
        String v = voices.get(speaker);
        if (v != null) return v;
        String g = gender.get(speaker);
        List<String> pool = new ArrayList<>("male".equals(g) ? male : "female".equals(g) ? female : female);
        if (g == null) pool.addAll(male);
        pool.remove(narrator);
        pool.removeAll(voices.values());
        if (pool.isEmpty()) pool = new ArrayList<>("male".equals(g) ? male : female);
        int h = 0;
        for (int i = 0; i < speaker.length(); ++i) h = 31 * h + Character.toLowerCase(speaker.charAt(i));
        v = pool.get(Math.floorMod(h, pool.size()));
        voices.put(speaker, v);
        return v;
    }
}
