package ai.whissle.steertts.demo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Reading order of a PDF from where its lines sit on the page (plain Java, unit tested; PdfText feeds it the lines and
 * their boxes from the PDF itself). Running headers, footers and page numbers are recognised by position and repetition
 * and dropped; a page in two columns is read column by column (a title or abstract across the top first); lines are
 * joined into paragraphs by spacing, indentation and size; a line set clearly larger than the text is a heading; a word
 * hyphenated at a line end is joined. Every paragraph keeps the boxes of its lines, so the reader can show the very
 * line being read on the page.
 */
final class PdfLayout {
    private PdfLayout() {}

    /** A line as the PDF has it: page index, text, box in page points (y down), and the page size. */
    static final class Line {
        final int page;
        final String text;
        final float l, t, r, b, pw, ph;

        Line(int page, String text, float l, float t, float r, float b, float pw, float ph) {
            this.page = page;
            this.text = text;
            this.l = l;
            this.t = t;
            this.r = r;
            this.b = b;
            this.pw = pw;
            this.ph = ph;
        }

        float h() { return b - t; }
        float w() { return r - l; }
    }

    /** Where part of a paragraph is on a page: characters [start, end) of the paragraph, box as fractions of the page. */
    static final class Span {
        final int page, start, end;
        final float l, t, r, b;

        Span(int page, int start, int end, float l, float t, float r, float b) {
            this.page = page;
            this.start = start;
            this.end = end;
            this.l = l;
            this.t = t;
            this.r = r;
            this.b = b;
        }
    }

    static final class Para {
        final StringBuilder text = new StringBuilder();
        final List<Span> spans = new ArrayList<>();
        boolean heading;

        String text() { return (heading ? String.valueOf(Narration.HEAD) : "") + text; }
    }

    private static final Pattern PAGE_NO = Pattern.compile(
            "(?i)^(page\\s+)?\\d{1,4}(\\s*(/|of)\\s*\\d{1,4})?$|^[-–—]\\s*\\d{1,4}\\s*[-–—]$|^[ivxlcdm]{1,6}$");
    private static final Pattern BULLET = Pattern.compile("^([•·▪◦‣●○■□–-]|\\d{1,2}[.)]|[a-z][.)])\\s+.*");

    static List<Para> layout(List<Line> all) {
        // ---- 1. running headers, footers, page numbers
        int pages = 0;
        for (Line ln : all) pages = Math.max(pages, ln.page + 1);
        Map<String, Integer> edgeCount = new HashMap<>();
        for (Line ln : all) if (edge(ln)) edgeCount.merge(key(ln), 1, Integer::sum);
        List<Line> body = new ArrayList<>();
        for (Line ln : all) {
            String t = ln.text.trim();
            if (t.isEmpty()) continue;
            if (edge(ln) && (PAGE_NO.matcher(t).matches() || (pages >= 2 && edgeCount.get(key(ln)) >= Math.max(2, pages / 2)))) continue;
            body.add(ln);
        }
        float median = medianHeight(body);
        body.removeIf(ln -> edge(ln) && ln.h() < median * 0.85f && ln.text.trim().split("\\s+").length <= 12);   // small print at the edge: a header
        // ---- 2. reading order, page by page
        List<Line> ordered = new ArrayList<>();
        for (int p = 0; p < pages; ++p) {
            List<Line> page = new ArrayList<>();
            for (Line ln : body) if (ln.page == p) page.add(ln);
            ordered.addAll(readingOrder(page));
        }
        // ---- 3. paragraphs
        List<Para> out = new ArrayList<>();
        Para cur = null;
        Line prev = null;
        for (Line ln : ordered) {
            String t = ln.text.trim().replaceAll("\\s+", " ");
            boolean heading = isHeading(ln, t, median, prev);
            if (cur == null || prev == null || breaksBefore(prev, ln, t, median, heading, cur.heading)) {
                if (cur != null && cur.text.length() > 0) out.add(cur);
                cur = new Para();
                cur.heading = heading;
            }
            int start = cur.text.length();
            if (start > 0) {
                char last = cur.text.charAt(start - 1);
                if (last == '-' && start > 1 && Character.isLetter(cur.text.charAt(start - 2)) && !t.isEmpty() && Character.isLowerCase(t.charAt(0))) {
                    cur.text.setLength(start - 1);   // "exam-" + "ple"
                    start--;
                } else {
                    cur.text.append(' ');
                    start++;
                }
            }
            cur.text.append(t);
            cur.spans.add(new Span(ln.page, start, cur.text.length(), ln.l / ln.pw, ln.t / ln.ph, ln.r / ln.pw, ln.b / ln.ph));
            prev = ln;
        }
        if (cur != null && cur.text.length() > 0) out.add(cur);
        return out;
    }

    private static boolean edge(Line ln) { return ln.b < ln.ph * 0.09f || ln.t > ln.ph * 0.91f; }

    private static String key(Line ln) {
        return (ln.t < ln.ph / 2 ? "top:" : "bottom:") + ln.text.trim().replaceAll("\\d+", "#").toLowerCase(Locale.ROOT);
    }

    private static float medianHeight(List<Line> lines) {
        if (lines.isEmpty()) return 10;
        List<Float> h = new ArrayList<>();
        for (Line ln : lines) h.add(ln.h());
        Collections.sort(h);
        return h.get(h.size() / 2);
    }

    /**
     * Two columns when enough lines sit wholly on each side of the page's middle and few cross it; then: lines across
     * the page above the columns, the left column, the right column, lines across the page below. Otherwise top to
     * bottom, with lines that share a row (a role and its dates) left to right.
     */
    static List<Line> readingOrder(List<Line> page) {
        List<Line> out = new ArrayList<>();
        if (page.isEmpty()) return out;
        float pw = page.get(0).pw, mid = pw / 2;
        List<Line> left = new ArrayList<>(), right = new ArrayList<>(), across = new ArrayList<>();
        for (Line ln : page) {
            if (ln.r <= mid + pw * 0.02f) left.add(ln);
            else if (ln.l >= mid - pw * 0.02f) right.add(ln);
            else across.add(ln);
        }
        boolean columns = left.size() >= 6 && right.size() >= 6 && across.size() * 4 <= page.size()
                && wide(left, pw) && wide(right, pw);
        if (!columns) {
            List<Line> all = new ArrayList<>(page);
            all.sort((a, b) -> sameRow(a, b) ? Float.compare(a.l, b.l) : Float.compare(a.t, b.t));
            return mergeRows(all);
        }
        float colTop = Float.MAX_VALUE, colBottom = 0;
        for (Line ln : left) { colTop = Math.min(colTop, ln.t); colBottom = Math.max(colBottom, ln.b); }
        for (Line ln : right) { colTop = Math.min(colTop, ln.t); colBottom = Math.max(colBottom, ln.b); }
        List<Line> above = new ArrayList<>(), below = new ArrayList<>();
        for (Line ln : across) (ln.t < colTop + 1 ? above : below).add(ln);
        above.sort((a, b) -> Float.compare(a.t, b.t));
        below.sort((a, b) -> Float.compare(a.t, b.t));
        left.sort((a, b) -> Float.compare(a.t, b.t));
        right.sort((a, b) -> Float.compare(a.t, b.t));
        out.addAll(above);
        out.addAll(left);
        out.addAll(right);
        out.addAll(below);
        return out;
    }

    /** A column of real text: most of its lines are wider than a quarter of the page (not a column of dates). */
    private static boolean wide(List<Line> col, float pw) {
        int n = 0;
        for (Line ln : col) if (ln.w() > pw * 0.25f) n++;
        return n * 2 >= col.size();
    }

    private static boolean sameRow(Line a, Line b) {
        float overlap = Math.min(a.b, b.b) - Math.max(a.t, b.t);
        return overlap > 0.6f * Math.min(a.h(), b.h());
    }

    /** Lines on one row (a job title on the left, its dates on the right) become one line, read left to right. */
    private static List<Line> mergeRows(List<Line> sorted) {
        List<Line> out = new ArrayList<>();
        for (Line ln : sorted) {
            if (!out.isEmpty()) {
                Line p = out.get(out.size() - 1);
                if (p.page == ln.page && sameRow(p, ln) && ln.l > p.r) {
                    String sep = p.text.trim().matches(".*[,.;:!?]$") ? " " : ", ";
                    out.set(out.size() - 1, new Line(p.page, p.text.trim() + sep + ln.text.trim(), p.l, Math.min(p.t, ln.t), ln.r,
                            Math.max(p.b, ln.b), p.pw, p.ph));
                    continue;
                }
            }
            out.add(ln);
        }
        return out;
    }

    private static final Pattern NUMBERED = Pattern.compile("^(\\d+(\\.\\d+)*\\.?|[IVX]+\\.|Chapter \\w+|Part \\w+)\\s+\\p{Lu}.*");

    /**
     * A heading: short, not a sentence, and either clearly larger than the text, numbered like a section ("2 Method",
     * "3.1 Data"), or a title-case line set apart by space above ("Skills"). Box heights alone mislead: a word without
     * descenders measures smaller.
     */
    private static boolean isHeading(Line ln, String t, float median, Line prev) {
        String[] w = t.split(" ");
        if (t.length() > 90 || w.length > 12) return false;
        if (t.matches(".*[.,;:]$") && !NUMBERED.matcher(t).matches()) return false;
        if (ln.h() >= median * 1.25f) return true;
        if (w.length <= 8 && NUMBERED.matcher(t).matches()) return true;
        boolean apart = prev == null || prev.page != ln.page || ln.t - prev.b > median * 1.1f;
        int caps = 0;
        for (String x : w) if (!x.isEmpty() && Character.isUpperCase(x.charAt(0))) caps++;
        return apart && w.length <= 5 && caps * 2 >= w.length && !t.matches(".*\\d{4}.*") && !t.contains("@");
    }

    private static boolean breaksBefore(Line prev, Line ln, String t, float median, boolean heading, boolean prevHeading) {
        if (prev.page != ln.page) {
            // a page break inside a sentence continues the paragraph
            String p = prev.text.trim();
            return heading || prevHeading || p.matches(".*[.!?:”\"]$");
        }
        if (heading != prevHeading) return true;
        if (heading) return Math.abs(ln.h() - prev.h()) > median * 0.2f || ln.t - prev.b > median * 0.8f;
        float gap = ln.t - prev.b;
        if (gap > Math.max(median * 0.75f, prev.h() * 0.6f)) return true;          // a blank line between paragraphs
        if (ln.t < prev.t - median) return prev.text.trim().matches(".*[.!?:”\"]$");   // a new column: continues mid-sentence
        if (BULLET.matcher(t).matches()) return true;
        String p = prev.text.trim();
        boolean ended = p.matches(".*[.!?:”\"]$");
        if (ended && ln.l > prev.l + median * 0.9f) return true;                   // a first-line indent
        return ended && prev.r < ln.r - prev.pw * 0.12f && ln.w() > prev.w();       // a short last line
    }
}
