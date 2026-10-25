package ai.whissle.steertts.demo;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * What to read for shared text: the text itself, or, when it is (little more than) a link, that page's article: its
 * title and paragraphs (inside <article> when the page has one), without scripts, menus, headers and footers. A rough
 * extraction, not a reader mode, but enough for news and blog pages.
 */
final class Article {
    private static final Pattern URL_RE = Pattern.compile("https?://\\S+");
    private static final int MAX_BYTES = 3 << 20, MAX_CHARS = 20000;

    final String title, text;

    private Article(String title, String text) {
        this.title = title;
        this.text = text;
    }

    /** Shared text that is a link with at most a short caption ("Interesting read https://..."). */
    static boolean isLink(String s) {
        Matcher m = URL_RE.matcher(s);
        return m.find() && s.length() - m.group().length() < 100;
    }

    static Article resolve(String shared) {
        if (!isLink(shared)) return new Article(null, shared);
        Matcher m = URL_RE.matcher(shared);
        m.find();
        try {
            return extract(fetch(m.group()));
        } catch (Exception e) {
            return new Article(null, "Sorry, I couldn't open that link.");
        }
    }

    private static String fetch(String url) throws Exception {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setConnectTimeout(8000);
        c.setReadTimeout(8000);
        c.setRequestProperty("User-Agent", "Mozilla/5.0 (Linux; Android 16) SteerTTS");
        try (InputStream in = c.getInputStream()) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[1 << 16];
            for (int n; out.size() < MAX_BYTES && (n = in.read(buf)) > 0; ) out.write(buf, 0, n);
            Charset cs = StandardCharsets.UTF_8;
            String ct = c.getContentType();
            Matcher cm = ct == null ? null : Pattern.compile("charset=([\\w-]+)").matcher(ct);
            if (cm != null && cm.find()) try { cs = Charset.forName(cm.group(1)); } catch (Exception ignored) { }
            return out.toString(cs.name());
        } finally {
            c.disconnect();
        }
    }

    static Article extract(String html) {
        String h = html.replaceAll("(?is)<(script|style|noscript|svg|nav|header|footer|aside|form|figure)\\b.*?</\\1>", " ");
        String title = first(h, "(?is)<meta[^>]+property=[\"']og:title[\"'][^>]+content=[\"']([^\"']+)");
        if (title == null) title = first(h, "(?is)<title[^>]*>(.*?)</title>");
        String body = paragraphs(first(h, "(?is)<article\\b[^>]*>(.*?)</article>"));
        if (body.length() < 500) body = paragraphs(h);
        if (body.length() > MAX_CHARS) {
            int cut = body.lastIndexOf(". ", MAX_CHARS);
            body = body.substring(0, cut > 0 ? cut + 1 : MAX_CHARS);
        }
        title = title == null ? null : clean(title);
        if (body.isEmpty()) return new Article(title, title != null ? title + ". I couldn't find an article on that page."
                : "I couldn't find an article on that page.");
        return new Article(title, (title != null ? title + ".\n" : "") + body);
    }

    private static String paragraphs(String h) {
        if (h == null) return "";
        StringBuilder b = new StringBuilder();
        Matcher m = Pattern.compile("(?is)<p\\b[^>]*>(.*?)</p>").matcher(h);
        while (m.find()) {
            String p = clean(m.group(1));
            if (p.length() >= 40) b.append(p).append('\n');
        }
        return b.toString().trim();
    }

    private static String first(String h, String re) {
        if (h == null) return null;
        Matcher m = Pattern.compile(re).matcher(h);
        return m.find() ? m.group(1) : null;
    }

    private static String clean(String s) {
        s = s.replaceAll("(?s)<[^>]*>", " ").replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'").replace("&apos;", "'")
                .replace("&rsquo;", "'").replace("&lsquo;", "'").replace("&ldquo;", "\"").replace("&rdquo;", "\"")
                .replace("&mdash;", ", ").replace("&ndash;", "-").replace("&hellip;", "...");
        Matcher m = Pattern.compile("&#(x?)([0-9a-fA-F]+);").matcher(s);
        StringBuffer b = new StringBuffer();
        while (m.find()) {
            String ch;
            try {
                ch = new String(Character.toChars(Integer.parseInt(m.group(2), m.group(1).isEmpty() ? 10 : 16)));
            } catch (IllegalArgumentException e) {   // not a number, or not a code point
                ch = " ";
            }
            m.appendReplacement(b, Matcher.quoteReplacement(ch));
        }
        m.appendTail(b);
        return b.toString().replaceAll("\\s+", " ").trim();
    }
}
