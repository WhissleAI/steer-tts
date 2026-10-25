package ai.whissle.steertts.demo;

import android.view.accessibility.AccessibilityNodeInfo;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * A short spoken summary for what does not read well aloud on a screen: an infobox or two-column table ("Details:
 * Author, Lewis Carroll. Genre, children's fiction. And 7 more."), a data table ("A table of 12 rows: Region, Q2, Q3
 * and Change. The first row: North, $1.1M…"), a long list of links ("A list of 24 links, starting with…"), a reference
 * list (skipped, with a word). The phrasing is plain Java (unit tested); of() reads the rows from the accessibility
 * tree (Android describes tables and lists as collections).
 */
final class Summary {
    private Summary() {}

    /** A summary for a table or list node, "" to skip it in silence, or null to read it as it is. */
    static String of(AccessibilityNodeInfo n) {
        AccessibilityNodeInfo.CollectionInfo ci = n.getCollectionInfo();
        if (ci == null) return null;
        Map<Integer, Map<Integer, String>> cells = new TreeMap<>();
        List<AccessibilityNodeInfo> stack = new ArrayList<>();
        for (int i = n.getChildCount() - 1; i >= 0; --i) stack.add(n.getChild(i));
        int seen = 0;
        while (!stack.isEmpty() && seen < 2000) {
            AccessibilityNodeInfo x = stack.remove(stack.size() - 1);
            if (x == null) continue;
            seen++;
            AccessibilityNodeInfo.CollectionItemInfo it = x.getCollectionItemInfo();
            if (it != null) {
                String t = text(x).trim();
                if (!t.isEmpty()) cells.computeIfAbsent(it.getRowIndex(), k -> new TreeMap<>()).merge(it.getColumnIndex(), t, (a, b) -> a + " " + b);
                continue;
            }
            for (int i = x.getChildCount() - 1; i >= 0; --i) stack.add(x.getChild(i));
        }
        List<List<String>> rows = new ArrayList<>();
        for (Map<Integer, String> r : cells.values()) rows.add(new ArrayList<>(r.values()));
        return phrase(rows, ci.getColumnCount());
    }

    /** All the text under a node, in order. */
    static String text(AccessibilityNodeInfo n) {
        StringBuilder b = new StringBuilder();
        List<AccessibilityNodeInfo> stack = new ArrayList<>();
        stack.add(n);
        while (!stack.isEmpty()) {
            AccessibilityNodeInfo x = stack.remove(stack.size() - 1);
            if (x == null) continue;
            if (x.getChildCount() == 0 && x.getText() != null) {
                String s = x.getText().toString();
                if (b.length() > 0 && !Character.isWhitespace(b.charAt(b.length() - 1)) && !s.startsWith(" ")) b.append(' ');
                b.append(s);
            }
            for (int i = x.getChildCount() - 1; i >= 0; --i) stack.add(x.getChild(i));
        }
        return b.toString().replaceAll("\\s+", " ");
    }

    // ---------------------------------------------------------------- phrasing (plain Java)
    /** rows of cells (strings), and the table's column count (1 or 0 for a list). */
    static String phrase(List<List<String>> rows, int columns) {
        int n = rows.size();
        if (n == 0) return null;
        int width = 0;
        for (List<String> r : rows) width = Math.max(width, r.size());
        if (columns <= 1 && width <= 1) {   // a list
            List<String> items = new ArrayList<>();
            int len = 0;
            for (List<String> r : rows) if (!r.isEmpty()) {
                items.add(r.get(0));
                len += r.get(0).length();
            }
            if (items.isEmpty()) return null;
            if (references(items)) return items.size() >= 3 ? "A list of " + items.size() + " references, skipped." : null;
            if (footer(items)) return "";   // a site's footer or menu: nothing worth saying
            if (items.size() < 6 || len / items.size() > 60) return null;   // a short list, or one of sentences: read it
            return "A list of " + items.size() + " items, starting with " + join(firstWords(items.subList(0, 3), 6)) + ".";
        }
        if (n < 3) return null;   // a small table reads fine
        if (width == 2) return details(rows);
        return table(rows);
    }

    static String details(List<List<String>> rows) {
        StringBuilder b = new StringBuilder("Details: ");
        int shown = 0;
        for (List<String> r : rows) {
            if (r.size() < 2 || shown >= 4) continue;
            b.append(clip(r.get(0), 5)).append(", ").append(clip(r.get(1), 8)).append(". ");
            shown++;
        }
        int rest = rows.size() - shown;
        if (rest > 0) b.append("And ").append(rest).append(rest == 1 ? " more." : " more.");
        return b.toString().trim();
    }

    static String table(List<List<String>> rows) {
        List<String> head = rows.get(0);
        StringBuilder b = new StringBuilder("A table of ").append(rows.size() - 1).append(" rows");
        if (!head.isEmpty()) b.append(", with columns ").append(join(firstWords(head.subList(0, Math.min(head.size(), 5)), 4)));
        b.append(". ");
        if (rows.size() > 1) b.append("The first row: ").append(String.join(", ", firstWords(rows.get(1).subList(0, Math.min(rows.get(1).size(), 5)), 5))).append(".");
        return b.toString().trim();
    }

    private static boolean footer(List<String> items) {
        int hits = 0;
        for (String s : items)
            if (s.matches("(?i).*\\b(privacy|terms|cookies?|contact|about( us)?|legal|disclaimer|accessibility|help|sign in|log in|"
                    + "careers|developers|statistics|mobile view|code of conduct)\\b.*")) hits++;
        return hits >= 2;
    }

    private static boolean references(List<String> items) {
        int refs = 0;
        for (String s : items) if (s.matches("(?s)^\\^.*|.*\\b(Retrieved|doi|ISBN|Archived|pp?\\.\\s?\\d).*")) refs++;
        return refs * 2 >= items.size();
    }

    private static List<String> firstWords(List<String> in, int n) {
        List<String> out = new ArrayList<>();
        for (String s : in) out.add(clip(s, n));
        return out;
    }

    private static String clip(String s, int words) {
        String[] w = s.trim().split("\\s+");
        if (w.length <= words) return s.trim().replaceAll("[.;:,]+$", "");
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < words; ++i) b.append(i > 0 ? " " : "").append(w[i]);
        return b.toString().replaceAll("[.;:,]+$", "");
    }

    static String join(List<String> xs) {
        if (xs.size() <= 1) return xs.isEmpty() ? "" : xs.get(0);
        return String.join(", ", xs.subList(0, xs.size() - 1)) + " and " + xs.get(xs.size() - 1);
    }
}
