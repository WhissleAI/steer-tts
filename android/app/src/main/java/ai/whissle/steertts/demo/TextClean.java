package ai.whissle.steertts.demo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Document text -> speakable sentences. Text pulled out of PDFs, e-books and web pages carries layout that should not
 * be read aloud: lines wrapped mid-sentence, words hyphenated across lines, page numbers, running headers and footers,
 * ligature characters, citation markers, long URLs. clean() removes that layout; sentences() then splits the text where
 * a reader would pause: at sentence ends, but not after abbreviations ("Dr.", "e.g.", "U.S."), initials or decimals,
 * and long sentences are split again at a semicolon, colon or comma so the first sound never waits on a paragraph.
 */
final class TextClean {
    private TextClean() {}

    private static final Map<String, String> LIGATURES = new HashMap<>();
    static {
        LIGATURES.put("\uFB00", "ff");
        LIGATURES.put("\uFB01", "fi");
        LIGATURES.put("\uFB02", "fl");
        LIGATURES.put("\uFB03", "ffi");
        LIGATURES.put("\uFB04", "ffl");
        LIGATURES.put("\uFB05", "st");
        LIGATURES.put("\uFB06", "st");
    }

    /** Pages of text (one string per PDF page; for other sources a single "page") -> clean paragraphs. */
    static List<String> clean(List<String> pages) {
        List<List<String>> lines = new ArrayList<>();
        for (String p : pages) lines.add(splitLines(normalizeChars(p)));
        Set<String> running = runningLines(lines);
        List<String> paras = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (List<String> page : lines) {
            for (String raw : page) {
                String ln = raw.trim();
                if (ln.isEmpty()) {                          // a blank line ends a paragraph
                    flush(cur, paras);
                    continue;
                }
                if (isPageNumber(ln) || running.contains(key(ln))) continue;
                if (cur.length() > 0) {
                    char last = cur.charAt(cur.length() - 1);
                    if (last == '-' && cur.length() > 1 && Character.isLetter(cur.charAt(cur.length() - 2))
                            && Character.isLowerCase(ln.charAt(0))) {
                        cur.setLength(cur.length() - 1);     // "exam-" + "ple" -> "example"
                    } else if (isBullet(ln) || (endsSentence(cur) && looksLikeHeading(ln))) {
                        flush(cur, paras);
                    } else {
                        cur.append(' ');
                    }
                }
                cur.append(isBullet(ln) ? ln.replaceFirst("^[•·▪◦‣\\-–*]\\s*", "") : ln);
            }
            // a page break inside a sentence continues it; after a sentence end it starts a new paragraph
            if (cur.length() > 0 && endsSentence(cur)) flush(cur, paras);
        }
        flush(cur, paras);
        List<String> out = new ArrayList<>();
        for (String p : paras) {
            String s = speakable(p);
            if (!s.isEmpty()) out.add(s);
        }
        return out;
    }

    static List<String> clean(String text) {
        List<String> one = new ArrayList<>();
        one.add(text);
        return clean(one);
    }

    private static void flush(StringBuilder cur, List<String> paras) {
        String s = cur.toString().trim();
        if (!s.isEmpty()) paras.add(s);
        cur.setLength(0);
    }

    static String normalizeChars(String s) {
        for (Map.Entry<String, String> e : LIGATURES.entrySet()) s = s.replace(e.getKey(), e.getValue());
        return s.replace("\u00AD", "")                     // soft hyphen
                .replace('\u00A0', ' ').replace('\u2009', ' ').replace('\u202F', ' ')
                .replace("\r\n", "\n").replace('\r', '\n').replace('\f', '\n');
    }

    private static List<String> splitLines(String s) {
        List<String> out = new ArrayList<>();
        for (String l : s.split("\n", -1)) out.add(l);
        return out;
    }

    private static final Pattern PAGE_NO = Pattern.compile(
            "(?i)^(page\\s+)?\\d{1,4}(\\s*(/|of)\\s*\\d{1,4})?$|^[-–—]\\s*\\d{1,4}\\s*[-–—]$|^[ivxlcdm]{1,6}$");

    private static boolean isPageNumber(String ln) { return PAGE_NO.matcher(ln).matches(); }

    private static String key(String ln) { return ln.replaceAll("\\d+", "#").toLowerCase(Locale.ROOT); }

    /** Lines that repeat at the top or bottom of most pages (running headers, footers), digits ignored. */
    private static Set<String> runningLines(List<List<String>> pages) {
        Set<String> out = new java.util.HashSet<>();
        if (pages.size() < 3) return out;
        Map<String, Integer> n = new HashMap<>();
        for (List<String> page : pages) {
            List<String> edge = new ArrayList<>();
            List<String> nonEmpty = new ArrayList<>();
            for (String l : page) if (!l.trim().isEmpty()) nonEmpty.add(l.trim());
            for (int i = 0; i < Math.min(2, nonEmpty.size()); ++i) edge.add(nonEmpty.get(i));
            for (int i = Math.max(0, nonEmpty.size() - 2); i < nonEmpty.size(); ++i) edge.add(nonEmpty.get(i));
            for (String l : new java.util.HashSet<>(edge)) if (l.length() < 90) n.merge(key(l), 1, Integer::sum);
        }
        for (Map.Entry<String, Integer> e : n.entrySet()) if (e.getValue() >= Math.max(3, pages.size() / 2)) out.add(e.getKey());
        return out;
    }

    private static boolean isBullet(String ln) { return ln.matches("^[•·▪◦‣]\\s*.+|^[-–*]\\s+\\S.*"); }

    private static boolean endsSentence(CharSequence s) {
        for (int i = s.length() - 1; i >= 0; --i) {
            char c = s.charAt(i);
            if (c == '"' || c == '\'' || c == ')' || c == '”' || c == '’' || c == ']') continue;
            return c == '.' || c == '!' || c == '?' || c == ':';
        }
        return false;
    }

    /** A short line without a final full stop, mostly capitalised words: a heading. */
    private static boolean looksLikeHeading(String ln) {
        if (ln.length() > 70 || endsSentence(ln)) return false;
        String[] w = ln.split("\\s+");
        int caps = 0;
        for (String x : w) if (!x.isEmpty() && Character.isUpperCase(x.charAt(0))) caps++;
        return w.length <= 10 && caps * 2 >= w.length;
    }

    private static final Pattern URL = Pattern.compile("\\b(?:https?://|www\\.)([\\w.-]+\\.[a-z]{2,})(?:/[^\\s)\\]]*)?", Pattern.CASE_INSENSITIVE);
    private static final Pattern CITE = Pattern.compile("\\s?\\[(\\d{1,3}([,–-]\\s?\\d{1,3})*|citation needed|[a-z])\\]");

    /** One paragraph, ready for the voice: links read as their site, citation markers gone, spaces tidy. */
    static String speakable(String p) {
        Matcher m = URL.matcher(p);
        StringBuffer b = new StringBuffer();
        while (m.find()) m.appendReplacement(b, Matcher.quoteReplacement(m.group(1).replaceFirst("^www\\.", "")));
        m.appendTail(b);
        String s = CITE.matcher(b.toString()).replaceAll("");
        s = s.replaceAll("\\s+([,.;:!?])", "$1").replaceAll("\\s{2,}", " ").trim();
        return s;
    }

    /**
     * A book's story without its wrapping: Project Gutenberg's licence header and footer (between its "*** START OF"
     * and "*** END OF" lines) and a table of contents (a "Contents" heading followed by short lines, up to the next
     * heading or long paragraph).
     */
    static List<String> bookBody(List<String> paras) {
        int from = 0, to = paras.size();
        for (int i = 0; i < paras.size(); ++i) {
            String p = paras.get(i).toUpperCase(Locale.ROOT);
            if (p.contains("*** START OF THE PROJECT GUTENBERG") || p.contains("*** START OF THIS PROJECT GUTENBERG")) from = i + 1;
            if ((p.contains("*** END OF THE PROJECT GUTENBERG") || p.contains("*** END OF THIS PROJECT GUTENBERG")) && i >= from) {
                to = i;
                break;
            }
        }
        List<String> out = new ArrayList<>();
        for (int i = from; i < to; ++i) {
            String p = paras.get(i), plain = Narration.display(p).trim();
            if (plain.matches("(?i)(table of )?contents\\.?")) {   // skip the table of contents
                int j = i + 1;
                // contents entries are short lines; the story starts at the first real heading or a paragraph
                while (j < to && !paras.get(j).startsWith(String.valueOf(Narration.HEAD))
                        && Narration.display(paras.get(j)).length() < 80 && j - i < 200) j++;
                i = j - 1;
                continue;
            }
            out.add(p);
        }
        return out;
    }

    // ---------------------------------------------------------------- sentences
    private static final Set<String> ABBREV = new java.util.HashSet<>(java.util.Arrays.asList(
            "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "e.g", "i.e", "fig", "figs", "no", "nos",
            "vol", "vols", "pp", "p", "ch", "sec", "ed", "eds", "inc", "ltd", "co", "corp", "dept", "est", "approx", "u.s",
            "u.k", "a.m", "p.m", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
            "mon", "tue", "wed", "thu", "fri", "sat", "sun", "gen", "gov", "sen", "rep", "rev", "capt", "col", "lt",
            "sgt", "ave", "blvd", "rd", "al", "cf", "ibid", "op", "ca"));
    private static final int MAX = 240;    // characters: split longer sentences at a pause

    /** Clean paragraphs -> sentences. */
    static List<String> sentences(List<String> paragraphs) {
        List<String> out = new ArrayList<>();
        for (String p : paragraphs) for (int[] sp : spans(p)) out.add(p.substring(sp[0], sp[1]));
        return out;
    }

    /** Where each sentence of a paragraph is: [start, end) character ranges, trimmed, in order. */
    static List<int[]> spans(String p) {
        List<int[]> out = new ArrayList<>();
        for (int[] s : splitParagraph(p)) out.addAll(splitLong(p, s[0], s[1]));
        return out;
    }

    private static int[] trimmed(String p, int a, int b) {
        while (a < b && Character.isWhitespace(p.charAt(a))) a++;
        while (b > a && Character.isWhitespace(p.charAt(b - 1))) b--;
        return new int[] {a, b};
    }

    private static List<int[]> splitParagraph(String p) {
        List<int[]> out = new ArrayList<>();
        int start = 0, n = p.length();
        for (int i = 0; i < n; ++i) {
            char c = p.charAt(i);
            if (c != '.' && c != '!' && c != '?') continue;
            int j = i + 1;
            while (j < n && (".!?\"'”’)]".indexOf(p.charAt(j)) >= 0 || p.charAt(j) == Narration.EM_OFF)) j++;
            if (j < n && !Character.isWhitespace(p.charAt(j))) continue;          // "3.5", "e.g.x", "U.S.A"
            int k = j;
            while (k < n && Character.isWhitespace(p.charAt(k))) k++;
            if (insideQuote(p, start, j)) continue;   // "Run! They're coming!" stays one line of dialogue
            int look = k;   // past emphasis and heading markers, to the first real character
            while (look < n && (p.charAt(look) == Narration.EM_ON || p.charAt(look) == Narration.HEAD)) look++;
            if (look < n) {
                char next = p.charAt(look);
                if (!(Character.isUpperCase(next) || Character.isDigit(next) || "\"'“‘(".indexOf(next) >= 0)) continue;
            }
            if (c == '.') {
                int w = i - 1;
                while (w >= start && !Character.isWhitespace(p.charAt(w))) w--;
                String word = p.substring(w + 1, i).toLowerCase(Locale.ROOT).replaceAll("^[(\"'“‘]+", "");
                if (ABBREV.contains(word) || word.matches("[a-z]")) continue;    // "Dr.", "J. Smith"
            }
            int[] t = trimmed(p, start, j);
            if (t[1] > t[0]) out.add(t);
            start = k;
            i = k - 1;
        }
        int[] t = trimmed(p, start, n);
        if (t[1] > t[0]) out.add(t);
        return out;
    }

    /** Whether a quotation opened in p[from, to) is still open at to (curly quotes counted, straight ones paired). */
    private static boolean insideQuote(String p, int from, int to) {
        int curly = 0, straight = 0, single = 0;
        for (int i = from; i < to; ++i) {
            char c = p.charAt(i);
            boolean letterBefore = i > 0 && Character.isLetter(p.charAt(i - 1));
            boolean letterAfter = i + 1 < p.length() && Character.isLetter(p.charAt(i + 1));
            if (c == '“' || (c == '‘' && !letterBefore)) curly++;
            else if (c == '”' || (c == '’' && curly > 0 && !letterAfter)) curly--;   // "don’t" is an apostrophe
            else if (c == '"') straight++;
            else if (c == '\'' && !letterBefore && letterAfter) single++;                // 'Now, my dears,'
            else if (c == '\'' && single > 0 && !letterAfter) single--;
        }
        return curly > 0 || straight % 2 == 1 || single > 0;   // (single quotes: ‘…’ opened and not yet closed count too)
    }

    /** A sentence longer than MAX, split again at a pause (; : — ,), as character ranges of p. */
    private static List<int[]> splitLong(String p, int a, int b) {
        List<int[]> out = new ArrayList<>();
        while (b - a > MAX) {
            String s = p.substring(a, b);
            int cut = -1;
            for (String sep : new String[] {"; ", ": ", " — ", ", "}) {
                int at = s.lastIndexOf(sep, MAX);
                if (at > MAX / 3) {
                    cut = at + sep.length() - 1;
                    break;
                }
            }
            if (cut < 0) cut = s.lastIndexOf(' ', MAX);
            if (cut <= 0) break;
            int[] head = trimmed(p, a, a + cut + 1);
            if (head[1] > head[0]) out.add(head);
            int[] rest = trimmed(p, a + cut + 1, b);
            a = rest[0];
            b = rest[1];
        }
        if (b > a) out.add(new int[] {a, b});
        return out;
    }
}
