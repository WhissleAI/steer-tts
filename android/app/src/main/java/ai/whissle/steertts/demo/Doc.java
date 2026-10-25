package ai.whissle.steertts.demo;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Something to listen to: a title, its clean paragraphs (TextClean), the sentences they split into, and where the
 * listener is. Stored as one JSON file per document in files/library, so the reading list, progress and resume survive
 * restarts; the original file is not kept.
 */
final class Doc {
    final String id, title, kind, source;
    final List<String> paragraphs;
    final List<String> sentences;
    final List<List<PdfLayout.Span>> spans;   // PDFs: where each paragraph's lines are on the pages (null otherwise)
    final List<List<float[]>> regions;        // PDFs: per sentence, boxes {page, l, t, r, b} as page fractions
    final int[] paraOf;                       // the paragraph each sentence is in
    private volatile Cast cast;

    /** Who speaks where, and how (Cast.analyze on first use: call off the main thread for a long book). */
    Cast cast() {
        Cast c = cast;
        if (c == null) cast = c = Cast.analyze(paragraphs);
        return c;
    }
    final long added;
    int pos;            // the sentence to read next
    long opened;
    boolean done;       // heard to the end (pos is back at 0)
    boolean transient_;  // read once and not kept (a screen being read): never saved, no position file

    private Doc(String id, String title, String kind, String source, List<String> paragraphs, List<List<PdfLayout.Span>> spans,
                long added, int pos, long opened) {
        this.id = id;
        this.title = title;
        this.kind = kind;
        this.source = source;
        this.paragraphs = paragraphs;
        this.spans = spans != null && spans.size() == paragraphs.size() ? spans : null;
        this.sentences = new ArrayList<>();
        this.regions = this.spans != null ? new ArrayList<>() : null;
        List<Integer> para = new ArrayList<>();
        for (int k = 0; k < paragraphs.size(); ++k) {
            String p = paragraphs.get(k);
            for (int[] sp : TextClean.spans(p)) {
                sentences.add(p.substring(sp[0], sp[1]));
                para.add(k);
                if (regions != null) regions.add(boxes(this.spans.get(k), sp[0], sp[1]));
            }
        }
        this.paraOf = new int[para.size()];
        for (int i = 0; i < paraOf.length; ++i) paraOf[i] = para.get(i);
        this.added = added;
        this.pos = Math.max(0, Math.min(pos, Math.max(0, sentences.size() - 1)));
        this.opened = opened;
    }

    static Doc create(String title, String kind, String source, List<String> paragraphs) {
        long now = System.currentTimeMillis();
        return new Doc(UUID.randomUUID().toString(), title, kind, source, paragraphs, null, now, 0, now);
    }

    /** Something to read now and not keep (the text of a screen). */
    static Doc createTransient(String title, List<String> paragraphs) {
        long now = System.currentTimeMillis();
        Doc d = new Doc("screen-" + now, title, "screen", "", paragraphs, null, now, 0, now);
        d.transient_ = true;
        return d;
    }

    static Doc createPdf(String id, String title, String source, List<String> paragraphs, List<List<PdfLayout.Span>> spans) {
        long now = System.currentTimeMillis();
        return new Doc(id, title, "pdf", source, paragraphs, spans, now, 0, now);
    }

    /** The part of each line a sentence covers (characters [a, b) of its paragraph), as page boxes. */
    static List<float[]> boxes(List<PdfLayout.Span> lines, int a, int b) {
        List<float[]> out = new ArrayList<>();
        for (PdfLayout.Span l : lines) {
            int s = Math.max(a, l.start), e = Math.min(b, l.end);
            if (e <= s) continue;
            float n = Math.max(1, l.end - l.start), w = l.r - l.l;
            out.add(new float[] {l.page, l.l + w * (s - l.start) / n, l.t, l.l + w * (e - l.start) / n, l.b});
        }
        return out;
    }

    /** The first page this sentence is on (PDFs), or -1. */
    int pageOf(int sentence) {
        if (regions == null || sentence < 0 || sentence >= regions.size() || regions.get(sentence).isEmpty()) return -1;
        return (int) regions.get(sentence).get(0)[0];
    }

    float progress() { return done ? 1f : sentences.isEmpty() ? 0 : Math.min(1f, pos / (float) sentences.size()); }

    /**
     * What the reading list shows, without the text: read from a small <id>.meta file and the position file, so listing
     * the library never parses whole books (that would block the screen with a long list).
     */
    static final class Meta {
        final String id, title, kind;
        final long added;
        long opened;
        int pos, count;
        long chars;
        boolean done;

        Meta(String id, String title, String kind, long added, int count, long chars) {
            this.id = id;
            this.title = title;
            this.kind = kind;
            this.added = added;
            this.count = count;
            this.chars = chars;
        }

        float progress() { return done ? 1f : count == 0 ? 0 : Math.min(1f, pos / (float) count); }

        int minutesLeft(float pace) {
            float left = chars * (1 - progress());
            return Math.max(1, Math.round(left / 15f / Math.max(0.5f, pace) / 60f));
        }
    }

    Meta meta() {
        long chars = 0;
        for (String x : sentences) chars += x.length();
        Meta m = new Meta(id, title, kind, added, sentences.size(), chars);
        m.opened = opened;
        m.pos = pos;
        m.done = done;
        return m;
    }

    private static void writeMeta(Context c, Meta m) {
        try (FileOutputStream out = new FileOutputStream(new File(dir(c), m.id + ".meta"))) {
            out.write(new JSONObject().put("title", m.title).put("kind", m.kind).put("added", m.added).put("count", m.count)
                    .put("chars", m.chars).toString().getBytes("UTF-8"));
        } catch (Exception ignored) {
        }
    }

    /** The library, most recently opened first, as metadata (cheap). */
    static List<Meta> list(Context c) {
        List<Meta> out = new ArrayList<>();
        File[] fs = dir(c).listFiles((d, n) -> n.endsWith(".json"));
        if (fs == null) return out;
        for (File f : fs) {
            String id = f.getName().substring(0, f.getName().length() - 5);
            Meta m = null;
            File mf = new File(dir(c), id + ".meta");
            if (mf.exists()) {
                try (FileInputStream in = new FileInputStream(mf)) {
                    JSONObject o = new JSONObject(new String(in.readAllBytes(), "UTF-8"));
                    m = new Meta(id, o.getString("title"), o.optString("kind", "text"), o.optLong("added"), o.optInt("count"), o.optLong("chars"));
                } catch (Exception ignored) {
                }
            }
            if (m == null) {   // a document saved before metadata files existed: read it once
                Doc d = load(c, id);
                if (d == null) continue;
                m = d.meta();
                writeMeta(c, m);
            }
            readPos(c, m);
            out.add(m);
        }
        out.sort((a, b) -> Long.compare(b.opened, a.opened));
        return out;
    }

    private static void readPos(Context c, Meta m) {
        File meta = new File(dir(c), m.id + ".pos");
        if (!meta.exists()) return;
        try (FileInputStream in = new FileInputStream(meta)) {
            String[] p = new String(in.readAllBytes()).trim().split(" ");
            m.pos = Integer.parseInt(p[0]);
            if (p.length > 1) m.opened = Long.parseLong(p[1]);
            if (p.length > 2) m.done = "1".equals(p[2]);
        } catch (Exception ignored) {
        }
    }

    /** The reading queue: documents not finished, the oldest added first (what plays next after one ends). */
    static List<Meta> upNext(Context c, String exceptId) {
        List<Meta> out = new ArrayList<>();
        for (Meta d : list(c)) if (!d.done && !d.id.equals(exceptId)) out.add(d);
        out.sort((a, b) -> Long.compare(a.added, b.added));
        return out;
    }

    /** Rough listening time left, from the characters still to read (about 15 characters a second at 1x). */
    int minutesLeft(float pace) {
        long chars = 0;
        for (int i = pos; i < sentences.size(); ++i) chars += sentences.get(i).length();
        return Math.max(1, Math.round(chars / 15f / Math.max(0.5f, pace) / 60f));
    }

    // ---------------------------------------------------------------- store
    private static File dir(Context c) {
        File d = new File(c.getFilesDir(), "library");
        d.mkdirs();
        return d;
    }

    void save(Context c) {
        try {
            JSONObject o = new JSONObject();
            o.put("id", id).put("title", title).put("kind", kind).put("source", source == null ? "" : source)
                    .put("added", added).put("pos", pos).put("opened", opened);
            JSONArray a = new JSONArray();
            for (String p : paragraphs) a.put(p);
            o.put("paragraphs", a);
            if (spans != null) {
                JSONArray sa = new JSONArray();
                for (List<PdfLayout.Span> ps : spans) {
                    JSONArray pa = new JSONArray();
                    for (PdfLayout.Span x : ps)
                        pa.put(new JSONArray().put(x.page).put(x.start).put(x.end).put((double) x.l).put((double) x.t).put((double) x.r).put((double) x.b));
                    sa.put(pa);
                }
                o.put("spans", sa);
            }
            File tmp = new File(dir(c), id + ".tmp");
            try (FileOutputStream out = new FileOutputStream(tmp)) { out.write(o.toString().getBytes("UTF-8")); }
            tmp.renameTo(new File(dir(c), id + ".json"));
            writeMeta(c, meta());
        } catch (Exception e) {
            android.util.Log.e("SteerTTS", "save doc", e);
        }
    }

    /** Just the position (cheap, called as reading moves on). */
    void savePos(Context c) {
        if (transient_) return;
        File meta = new File(dir(c), id + ".pos");
        try (FileOutputStream out = new FileOutputStream(meta)) {
            out.write((pos + " " + opened + " " + (done ? 1 : 0)).getBytes());
        } catch (Exception ignored) {
        }
    }

    static Doc load(Context c, String id) {
        try (FileInputStream in = new FileInputStream(new File(dir(c), id + ".json"))) {
            JSONObject o = new JSONObject(new String(in.readAllBytes(), "UTF-8"));
            JSONArray a = o.getJSONArray("paragraphs");
            List<String> paras = new ArrayList<>();
            for (int i = 0; i < a.length(); ++i) paras.add(a.getString(i));
            int pos = o.optInt("pos");
            long opened = o.optLong("opened");
            boolean done = false;
            File meta = new File(dir(c), id + ".pos");
            if (meta.exists()) {
                try (FileInputStream m = new FileInputStream(meta)) {
                    String[] p = new String(m.readAllBytes()).trim().split(" ");
                    pos = Integer.parseInt(p[0]);
                    if (p.length > 1) opened = Long.parseLong(p[1]);
                    if (p.length > 2) done = "1".equals(p[2]);
                } catch (Exception ignored) {
                }
            }
            List<List<PdfLayout.Span>> spans = null;
            JSONArray sa = o.optJSONArray("spans");
            if (sa != null) {
                spans = new ArrayList<>();
                for (int i = 0; i < sa.length(); ++i) {
                    JSONArray pa = sa.getJSONArray(i);
                    List<PdfLayout.Span> ps = new ArrayList<>();
                    for (int j = 0; j < pa.length(); ++j) {
                        JSONArray x = pa.getJSONArray(j);
                        ps.add(new PdfLayout.Span(x.getInt(0), x.getInt(1), x.getInt(2), (float) x.getDouble(3), (float) x.getDouble(4),
                                (float) x.getDouble(5), (float) x.getDouble(6)));
                    }
                    spans.add(ps);
                }
            }
            Doc d = new Doc(o.getString("id"), o.getString("title"), o.optString("kind", "text"), o.optString("source"),
                    paras, spans, o.optLong("added"), pos, opened);
            d.done = done;
            return d;
        } catch (Exception e) {
            return null;
        }
    }

    static void delete(Context c, String id) {
        PdfText.file(c, id).delete();
        new File(dir(c), id + ".json").delete();
        new File(dir(c), id + ".pos").delete();
        new File(dir(c), id + ".meta").delete();
    }

    /** Everything in the library, most recently opened first (light: titles and progress only need the full load). */
    static List<Doc> all(Context c) {
        List<Doc> out = new ArrayList<>();
        File[] fs = dir(c).listFiles((d, n) -> n.endsWith(".json"));
        if (fs != null)
            for (File f : fs) {
                Doc d = load(c, f.getName().substring(0, f.getName().length() - 5));
                if (d != null) out.add(d);
            }
        out.sort((a, b) -> Long.compare(b.opened, a.opened));
        return out;
    }
}
