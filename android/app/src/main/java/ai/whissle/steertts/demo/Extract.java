package ai.whissle.steertts.demo;

import android.content.ContentResolver;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * A file, a link or plain text -> a Doc to listen to. Formats: PDF (the platform's text extraction, Android 15+),
 * EPUB (the spine's XHTML chapters, in reading order), DOCX (the body's paragraphs), HTML, Markdown and plain text;
 * links are fetched and their article read (Article). Everything goes through TextClean.
 */
final class Extract {
    private Extract() {}

    static final String[] MIME_TYPES = {"application/pdf", "application/epub+zip",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain", "text/html",
            "text/markdown", "text/x-markdown"};

    static final class Unsupported extends IOException {
        Unsupported(String m) { super(m); }
    }

    static Doc fromText(String text, String title) {
        List<String> paras = TextClean.clean(text);
        String t = title != null ? title : firstWords(paras);
        return Doc.create(t, "text", null, paras);
    }

    static Doc fromUrl(String shared) {
        Article a = Article.resolve(shared);
        List<String> paras = new ArrayList<>();
        for (String p : a.text.split("\n")) if (!p.trim().isEmpty()) paras.add(TextClean.speakable(p.trim()));
        return Doc.create(a.title != null ? a.title : firstWords(paras), "web", shared, paras);
    }

    static Doc fromUri(Context c, Uri uri, String mimeHint) throws IOException {
        ContentResolver cr = c.getContentResolver();
        String name = displayName(c, uri);
        String mime = mimeHint != null ? mimeHint : cr.getType(uri);
        String ext = name.contains(".") ? name.substring(name.lastIndexOf('.') + 1).toLowerCase(Locale.ROOT) : "";
        String title = name.contains(".") ? name.substring(0, name.lastIndexOf('.')) : name;
        if ("application/pdf".equals(mime) || "pdf".equals(ext)) return pdf(c, uri, title);
        byte[] data = read(cr.openInputStream(uri), 200 << 20);
        if ("application/epub+zip".equals(mime) || "epub".equals(ext)) return epub(data, title, uri.toString());
        if ("docx".equals(ext) || (mime != null && mime.contains("wordprocessingml"))) return docx(data, title, uri.toString());
        String text = new String(data, StandardCharsets.UTF_8);
        if ("html".equals(ext) || "htm".equals(ext) || "xhtml".equals(ext) || (mime != null && mime.contains("html"))) {
            String t = htmlTitle(text);
            return Doc.create(t != null ? t : title, "html", uri.toString(), TextClean.clean(htmlToText(text)));
        }
        if ("md".equals(ext) || "markdown".equals(ext) || (mime != null && mime.contains("markdown")))
            return Doc.create(title, "markdown", uri.toString(), TextClean.clean(markdownToText(text)));
        if (mime != null && !mime.startsWith("text/") && !"".equals(ext) && !"txt".equals(ext))
            throw new Unsupported("SteerTTS can read PDF, EPUB, Word (.docx), web pages and text files.");
        return Doc.create(title, "text", uri.toString(), TextClean.clean(text));
    }

    // ---------------------------------------------------------------- PDF
    private static Doc pdf(Context c, Uri uri, String title) throws IOException {
        if (Build.VERSION.SDK_INT < 35) throw new Unsupported("Reading PDFs needs Android 15 or newer.");
        List<String> pages = new ArrayList<>();
        try (ParcelFileDescriptor fd = c.getContentResolver().openFileDescriptor(uri, "r");
             android.graphics.pdf.PdfRenderer r = new android.graphics.pdf.PdfRenderer(fd)) {
            for (int i = 0; i < r.getPageCount(); ++i) {
                try (android.graphics.pdf.PdfRenderer.Page p = r.openPage(i)) {
                    StringBuilder b = new StringBuilder();
                    for (android.graphics.pdf.content.PdfPageTextContent t : p.getTextContents()) b.append(t.getText()).append('\n');
                    pages.add(b.toString());
                }
            }
        }
        List<String> paras = TextClean.clean(pages);
        if (paras.isEmpty()) throw new Unsupported("This PDF has no text to read (it may be scanned pages).");
        return Doc.create(title, "pdf", uri.toString(), paras);
    }

    // ---------------------------------------------------------------- EPUB
    private static Doc epub(byte[] data, String fallbackTitle, String source) throws IOException {
        Map<String, byte[]> files = unzip(data);
        String container = str(files.get("META-INF/container.xml"));
        String opfPath = first(container, "full-path=\"([^\"]+)\"");
        if (opfPath == null) throw new Unsupported("This EPUB could not be opened.");
        String opf = str(files.get(opfPath));
        String base = opfPath.contains("/") ? opfPath.substring(0, opfPath.lastIndexOf('/') + 1) : "";
        Map<String, String> manifest = new HashMap<>();
        Matcher item = Pattern.compile("<item\\b[^>]*>").matcher(opf);
        while (item.find()) {
            String tag = item.group();
            String id = first(tag, "\\bid=\"([^\"]+)\""), href = first(tag, "\\bhref=\"([^\"]+)\"");
            if (id != null && href != null) manifest.put(id, href);
        }
        List<String> paras = new ArrayList<>();
        Matcher ref = Pattern.compile("<itemref\\b[^>]*idref=\"([^\"]+)\"").matcher(opf);
        while (ref.find()) {
            String href = manifest.get(ref.group(1));
            if (href == null) continue;
            byte[] x = files.get(resolve(base, Uri.decode(href)));
            if (x != null) paras.addAll(TextClean.clean(htmlToText(str(x))));
        }
        if (paras.isEmpty()) throw new Unsupported("This EPUB has no text to read.");
        String title = clean(first(opf, "<dc:title[^>]*>([\\s\\S]*?)</dc:title>"));
        String author = clean(first(opf, "<dc:creator[^>]*>([\\s\\S]*?)</dc:creator>"));
        String t = title != null && !title.isEmpty() ? title : fallbackTitle;
        return Doc.create(author != null && !author.isEmpty() ? t + " · " + author : t, "epub", source, paras);
    }

    private static String resolve(String base, String href) {
        String p = base + href;
        List<String> out = new ArrayList<>();
        for (String s : p.split("/")) {
            if (s.equals("..")) { if (!out.isEmpty()) out.remove(out.size() - 1); }
            else if (!s.equals(".") && !s.isEmpty()) out.add(s);
        }
        int hash = String.join("/", out).indexOf('#');
        String r = String.join("/", out);
        return hash >= 0 ? r.substring(0, hash) : r;
    }

    // ---------------------------------------------------------------- DOCX
    private static Doc docx(byte[] data, String fallbackTitle, String source) throws IOException {
        Map<String, byte[]> files = unzip(data);
        String doc = str(files.get("word/document.xml"));
        if (doc.isEmpty()) throw new Unsupported("This Word file could not be opened.");
        StringBuilder text = new StringBuilder();
        for (String p : doc.split("</w:p>")) {
            StringBuilder para = new StringBuilder();
            Matcher m = Pattern.compile("<w:t(?:\\s[^>]*)?>([^<]*)</w:t>|<w:tab/>|<w:br/>").matcher(p);
            while (m.find()) para.append(m.group(1) != null ? decode(m.group(1)) : " ");
            if (para.toString().trim().length() > 0) text.append(para).append("\n\n");
        }
        String title = clean(first(str(files.get("docProps/core.xml")), "<dc:title>([\\s\\S]*?)</dc:title>"));
        return Doc.create(title != null && !title.isEmpty() ? title : fallbackTitle, "docx", source, TextClean.clean(text.toString()));
    }

    // ---------------------------------------------------------------- HTML / Markdown
    /** Visible text of an HTML document, one block element per paragraph. */
    static String htmlToText(String html) {
        String h = html.replaceAll("(?is)<(script|style|noscript|svg|head|nav|header|footer|aside|form|figure|sup)\\b.*?</\\1>", " ");
        h = h.replaceAll("(?i)<br\\s*/?>", "\n");
        // italics and bold become stress markers, a heading starts with a heading marker (Narration reads both)
        h = h.replaceAll("(?i)<(em|i|strong|b)\\b[^>]*>", String.valueOf(Narration.EM_ON));
        h = h.replaceAll("(?i)</(em|i|strong|b)\\s*>", String.valueOf(Narration.EM_OFF));
        h = h.replaceAll("(?i)<h[1-6]\\b[^>]*>", "\n\n" + Narration.HEAD);
        h = h.replaceAll("(?i)</?(p|div|h[1-6]|li|blockquote|section|article|tr|dt|dd|pre|title)\\b[^>]*>", "\n\n");
        h = h.replaceAll("(?s)<[^>]*>", " ");
        return decode(h).replaceAll("[ \\t]+", " ").replaceAll("\\n[ \\t]+", "\n").replaceAll("\\n{3,}", "\n\n");
    }

    private static String htmlTitle(String html) { return clean(first(html, "(?is)<title[^>]*>(.*?)</title>")); }

    static String markdownToText(String md) {
        String t = md.replaceAll("(?s)```.*?```", " ");
        t = t.replaceAll("!\\[[^]]*]\\([^)]*\\)", "");                    // images
        t = t.replaceAll("\\[([^]]+)]\\([^)]*\\)", "$1");                 // links -> their text
        t = t.replaceAll("(?m)^\\s{0,3}#{1,6}\\s*(.*?)\\s*#*$", "\n" + Narration.HEAD + "$1.\n");  // headings
        t = t.replaceAll("(\\*\\*|__)(\\S(?:.*?\\S)?)\\1", Narration.EM_ON + "$2" + Narration.EM_OFF);  // **bold**
        t = t.replaceAll("(?<![\\w*])([*_])(\\S(?:.*?\\S)?)\\1(?![\\w*])", Narration.EM_ON + "$2" + Narration.EM_OFF);  // *italic*
        t = t.replaceAll("(?m)^\\s*>\\s?", "");
        t = t.replaceAll("(\\*\\*|__|\\*|_|`)", "");
        t = t.replaceAll("(?m)^\\s*([-*+]|\\d+\\.)\\s+", "• ");
        return t;
    }

    // ---------------------------------------------------------------- helpers
    private static String displayName(Context c, Uri uri) {
        try (Cursor cur = c.getContentResolver().query(uri, new String[] {OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (cur != null && cur.moveToFirst() && cur.getString(0) != null) return cur.getString(0);
        } catch (Exception ignored) {
        }
        String last = uri.getLastPathSegment();
        return last != null ? last : "Document";
    }

    private static byte[] read(InputStream in, int max) throws IOException {
        if (in == null) throw new IOException("cannot open the file");
        try (InputStream s = in) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[1 << 16];
            for (int n; (n = s.read(buf)) > 0; ) {
                out.write(buf, 0, n);
                if (out.size() > max) throw new Unsupported("This file is too large to read.");
            }
            return out.toByteArray();
        }
    }

    private static Map<String, byte[]> unzip(byte[] data) throws IOException {
        Map<String, byte[]> out = new LinkedHashMap<>();
        try (ZipInputStream z = new ZipInputStream(new java.io.ByteArrayInputStream(data))) {
            for (ZipEntry e; (e = z.getNextEntry()) != null; ) {
                if (e.isDirectory()) continue;
                ByteArrayOutputStream b = new ByteArrayOutputStream();
                byte[] buf = new byte[1 << 15];
                for (int n; (n = z.read(buf)) > 0; ) b.write(buf, 0, n);
                out.put(e.getName(), b.toByteArray());
            }
        }
        return out;
    }

    private static String str(byte[] b) { return b == null ? "" : new String(b, StandardCharsets.UTF_8); }

    private static String first(String s, String re) {
        if (s == null) return null;
        Matcher m = Pattern.compile(re).matcher(s);
        return m.find() ? m.group(1) : null;
    }

    private static String clean(String s) { return s == null ? null : decode(s.replaceAll("<[^>]*>", " ")).replaceAll("\\s+", " ").trim(); }

    static String decode(String s) {
        s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"")
                .replace("&#39;", "'").replace("&apos;", "'").replace("&rsquo;", "’").replace("&lsquo;", "‘")
                .replace("&ldquo;", "“").replace("&rdquo;", "”").replace("&mdash;", "—").replace("&ndash;", "–")
                .replace("&hellip;", "…");
        Matcher m = Pattern.compile("&#(x?)([0-9a-fA-F]+);").matcher(s);
        StringBuffer b = new StringBuffer();
        while (m.find()) {
            String ch;
            try {
                ch = new String(Character.toChars(Integer.parseInt(m.group(2), m.group(1).isEmpty() ? 10 : 16)));
            } catch (IllegalArgumentException e) {
                ch = " ";
            }
            m.appendReplacement(b, Matcher.quoteReplacement(ch));
        }
        m.appendTail(b);
        return b.toString();
    }

    private static String firstWords(List<String> paras) {
        String s = paras.isEmpty() ? "Text" : paras.get(0);
        return s.length() <= 48 ? s : s.substring(0, s.lastIndexOf(' ', 48) > 0 ? s.lastIndexOf(' ', 48) : 48) + "…";
    }
}
