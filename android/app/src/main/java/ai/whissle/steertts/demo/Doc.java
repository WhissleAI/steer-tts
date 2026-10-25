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
    final long added;
    int pos;            // the sentence to read next
    long opened;

    private Doc(String id, String title, String kind, String source, List<String> paragraphs, long added, int pos, long opened) {
        this.id = id;
        this.title = title;
        this.kind = kind;
        this.source = source;
        this.paragraphs = paragraphs;
        this.sentences = TextClean.sentences(paragraphs);
        this.added = added;
        this.pos = Math.max(0, Math.min(pos, Math.max(0, sentences.size() - 1)));
        this.opened = opened;
    }

    static Doc create(String title, String kind, String source, List<String> paragraphs) {
        long now = System.currentTimeMillis();
        return new Doc(UUID.randomUUID().toString(), title, kind, source, paragraphs, now, 0, now);
    }

    float progress() { return sentences.isEmpty() ? 0 : Math.min(1f, pos / (float) sentences.size()); }

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
            File tmp = new File(dir(c), id + ".tmp");
            try (FileOutputStream out = new FileOutputStream(tmp)) { out.write(o.toString().getBytes("UTF-8")); }
            tmp.renameTo(new File(dir(c), id + ".json"));
        } catch (Exception e) {
            android.util.Log.e("SteerTTS", "save doc", e);
        }
    }

    /** Just the position (cheap, called as reading moves on). */
    void savePos(Context c) {
        File meta = new File(dir(c), id + ".pos");
        try (FileOutputStream out = new FileOutputStream(meta)) {
            out.write((pos + " " + opened).getBytes());
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
            File meta = new File(dir(c), id + ".pos");
            if (meta.exists()) {
                try (FileInputStream m = new FileInputStream(meta)) {
                    String[] p = new String(m.readAllBytes()).trim().split(" ");
                    pos = Integer.parseInt(p[0]);
                    if (p.length > 1) opened = Long.parseLong(p[1]);
                } catch (Exception ignored) {
                }
            }
            return new Doc(o.getString("id"), o.getString("title"), o.optString("kind", "text"), o.optString("source"),
                    paras, o.optLong("added"), pos, opened);
        } catch (Exception e) {
            return null;
        }
    }

    static void delete(Context c, String id) {
        new File(dir(c), id + ".json").delete();
        new File(dir(c), id + ".pos").delete();
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
