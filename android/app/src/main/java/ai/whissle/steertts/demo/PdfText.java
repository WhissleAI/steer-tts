package ai.whissle.steertts.demo;

import android.content.Context;
import android.graphics.RectF;
import android.graphics.pdf.PdfRenderer;
import android.graphics.pdf.content.PdfPageTextContent;
import android.graphics.pdf.models.selection.PageSelection;
import android.graphics.pdf.models.selection.SelectionBoundary;
import android.os.ParcelFileDescriptor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A PDF's lines with their boxes, from the PDF itself (Android 15+ PdfRenderer: each page's text, then each line's
 * box by selecting its characters), handed to PdfLayout for the reading order. The PDF is kept in the app's storage
 * (files/pdf/<doc id>.pdf) so the reader can show its pages and the line being read.
 */
final class PdfText {
    private PdfText() {}

    static File file(Context c, String docId) {
        File d = new File(c.getFilesDir(), "pdf");
        d.mkdirs();
        return new File(d, docId + ".pdf");
    }

    /** Paragraphs (heading marker first for headings) and, per paragraph, where its lines are. */
    static final class Result {
        final List<String> paragraphs = new ArrayList<>();
        final List<List<PdfLayout.Span>> spans = new ArrayList<>();
    }

    static Result read(File pdf) throws IOException {
        List<PdfLayout.Line> lines = new ArrayList<>();
        try (ParcelFileDescriptor fd = ParcelFileDescriptor.open(pdf, ParcelFileDescriptor.MODE_READ_ONLY);
             PdfRenderer r = new PdfRenderer(fd)) {
            for (int p = 0; p < r.getPageCount(); ++p) {
                try (PdfRenderer.Page page = r.openPage(p)) {
                    StringBuilder all = new StringBuilder();
                    for (PdfPageTextContent c : page.getTextContents()) {
                        if (all.length() > 0) all.append('\n');
                        all.append(c.getText());
                    }
                    String text = all.toString();
                    int at = 0;
                    for (String raw : text.split("\n", -1)) {
                        int start = at;
                        at += raw.length() + 1;
                        String t = TextClean.normalizeChars(raw).replace('\r', ' ').trim();
                        if (t.isEmpty()) continue;
                        RectF box = box(page, start, start + raw.length());
                        if (box == null) continue;   // text without a place on the page (hidden, or unmapped)
                        lines.add(new PdfLayout.Line(p, t, box.left, box.top, box.right, box.bottom, page.getWidth(), page.getHeight()));
                    }
                }
            }
        }
        Result out = new Result();
        for (PdfLayout.Para para : PdfLayout.layout(lines)) {
            String t = para.text();
            int shift = para.heading ? 1 : 0;   // the heading marker before the text
            List<PdfLayout.Span> spans = new ArrayList<>();
            for (PdfLayout.Span s : para.spans) spans.add(new PdfLayout.Span(s.page, s.start + shift, s.end + shift, s.l, s.t, s.r, s.b));
            out.paragraphs.add(t);
            out.spans.add(spans);
        }
        return out;
    }

    /** The box of characters [a, b) of a page's text (the union of what the selection returns), or null. */
    private static RectF box(PdfRenderer.Page page, int a, int b) {
        if (b <= a) return null;
        try {
            PageSelection sel = page.selectContent(new SelectionBoundary(a), new SelectionBoundary(b));
            if (sel == null) return null;
            RectF u = null;
            for (PdfPageTextContent c : sel.getSelectedTextContents())
                for (RectF r : c.getBounds()) {
                    if (u == null) u = new RectF(r);
                    else u.union(r);
                }
            return u;
        } catch (RuntimeException e) {
            return null;
        }
    }
}
