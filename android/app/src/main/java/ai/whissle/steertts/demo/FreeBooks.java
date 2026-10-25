package ai.whissle.steertts.demo;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.os.Build;
import android.os.Bundle;
import android.text.InputType;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.EditText;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Free books from Project Gutenberg (70,000+ public-domain books): shelves (popular, children's, fairy tales, adventure,
 * mystery, poetry…) and search, from its OPDS catalogue; a book is downloaded as an EPUB, added to the reading list
 * and opened in the player, where stories are performed (Cast: a voice per character, how each line is said).
 */
public class FreeBooks extends Activity {
    private static final String BASE = "https://www.gutenberg.org/ebooks/search.opds/";
    private static final String[][] SHELVES = {
            {"Popular", ""}, {"Children's", "s.children l.en"}, {"Fairy tales", "s.fairy l.en"}, {"Adventure", "s.adventure l.en"},
            {"Mystery", "s.detective l.en"}, {"Science fiction", "s.science fiction l.en"}, {"Poetry", "s.poetry l.en"},
            {"Humour", "s.humor l.en"}, {"Classics", "s.fiction l.en"}};

    static final class Book {
        final String id, title, author;

        Book(String id, String title, String author) {
            this.id = id;
            this.title = title;
            this.author = author;
        }
    }

    private Ui u;
    private LinearLayout list;
    private TextView status, more;
    private final List<TextView> chips = new ArrayList<>();
    private String query = "";
    private int start = 1;
    private int request;   // the latest request: older answers are dropped

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        Ui.init(this);
        u = new Ui(this);
        LinearLayout root = u.column();
        root.setBackgroundColor(Ui.BG);
        LinearLayout bar = u.row();
        bar.setPadding(u.dp(6), u.dp(6), u.dp(18), u.dp(2));
        android.widget.ImageView back = u.img(R.drawable.ic_back, Ui.INK, 48);
        int pad = u.dp(12);
        back.setPadding(pad, pad, pad, pad);
        back.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_YES);
        back.setContentDescription("Back");
        back.setOnClickListener(v -> finish());
        bar.addView(back);
        bar.addView(u.text("Free books", 24, Ui.INK, true));
        root.addView(bar);

        LinearLayout body = u.column();
        body.setPadding(u.dp(18), 0, u.dp(18), u.dp(28));
        TextView sub = u.text("Classics and children's stories from Project Gutenberg, free to keep. Stories are read with "
                + "a voice for each character.", 13.5f, Ui.MUTED, false);
        body.addView(sub);
        EditText search = new EditText(this);
        search.setHint("Search titles and authors");
        search.setSingleLine(true);
        search.setInputType(InputType.TYPE_CLASS_TEXT);
        search.setImeOptions(EditorInfo.IME_ACTION_SEARCH);
        search.setTextColor(Ui.INK);
        search.setHintTextColor(Ui.MUTED);
        search.setBackground(Ui.round(Ui.CARD, u.dp(24)));
        search.setPadding(u.dp(18), u.dp(12), u.dp(18), u.dp(12));
        search.setOnEditorActionListener((v, id, e) -> {
            if (id == EditorInfo.IME_ACTION_SEARCH || (e != null && e.getKeyCode() == KeyEvent.KEYCODE_ENTER)) {
                String q = v.getText().toString().trim();
                if (!q.isEmpty()) load(q + " l.en", true, -1);
                return true;
            }
            return false;
        });
        LinearLayout.LayoutParams sl = Ui.matchWrap();
        sl.topMargin = u.dp(14);
        body.addView(search, sl);
        HorizontalScrollView hs = new HorizontalScrollView(this);
        hs.setHorizontalScrollBarEnabled(false);
        LinearLayout row = u.row();
        row.setPadding(0, u.dp(12), 0, u.dp(4));
        for (int i = 0; i < SHELVES.length; ++i) {
            final int k = i;
            TextView ch = u.chip(SHELVES[i][0], v -> load(SHELVES[k][1], true, k));
            chips.add(ch);
            row.addView(ch);
        }
        hs.addView(row);
        body.addView(hs);
        status = u.text("", 13.5f, Ui.MUTED, false);
        status.setPadding(u.dp(4), u.dp(10), 0, 0);
        body.addView(status);
        list = u.column();
        body.addView(list, Ui.matchWrap());
        more = u.pill("Show more", false, v -> load(query, false, -2));
        more.setVisibility(View.GONE);
        LinearLayout.LayoutParams ml = Ui.matchWrap();
        ml.topMargin = u.dp(12);
        body.addView(more, ml);
        ScrollView sv = new ScrollView(this);
        sv.addView(body);
        root.addView(sv, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1));
        setContentView(root);
        if (Build.VERSION.SDK_INT >= 30)
            root.setOnApplyWindowInsetsListener((v, ins) -> {
                android.graphics.Insets bars = ins.getInsets(android.view.WindowInsets.Type.systemBars() | android.view.WindowInsets.Type.ime());
                v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
                return android.view.WindowInsets.CONSUMED;
            });
        load(SHELVES[1][1], true, 1);   // children's stories first
    }

    /** shelf: the chip to select (-1 for a search, -2 to keep the selection when showing more). */
    private void load(String q, boolean fresh, int shelf) {
        if (shelf >= -1) for (int i = 0; i < chips.size(); ++i) u.select(chips.get(i), i == shelf);
        if (fresh) {
            query = q;
            start = 1;
            list.removeAllViews();
        }
        int r = ++request;
        status.setText("Loading…");
        more.setVisibility(View.GONE);
        String url = BASE + "?sort_order=downloads&start_index=" + start + (q.isEmpty() ? "" : "&query=" + enc(q));
        new Thread(() -> {
            List<Book> books;
            try {
                books = parse(get(url, 1 << 20));
            } catch (Exception e) {
                books = null;
            }
            List<Book> got = books;
            runOnUiThread(() -> {
                if (r != request || isDestroyed()) return;
                if (got == null) {
                    status.setText("Couldn't reach Project Gutenberg. Check the connection and try again.");
                    return;
                }
                status.setText(got.isEmpty() && start == 1 ? "No books found." : "");
                for (Book bk : got) list.addView(row(bk));
                start += got.size();
                more.setVisibility(got.size() >= 25 ? View.VISIBLE : View.GONE);
            });
        }, "steertts-books").start();
    }

    private View row(Book b) {
        LinearLayout r = u.row();
        r.setPadding(u.dp(14), u.dp(12), u.dp(8), u.dp(12));
        r.setBackground(Ui.round(Ui.CARD, u.dp(18)));
        LinearLayout t = u.column();
        TextView title = u.text(b.title, 15.5f, Ui.INK, true);
        title.setMaxLines(2);
        title.setEllipsize(TextUtils.TruncateAt.END);
        t.addView(title);
        if (!b.author.isEmpty()) t.addView(u.text(b.author, 13, Ui.MUTED, false));
        r.addView(t, Ui.weight(1));
        r.addView(u.icon(R.drawable.ic_play, "Listen to " + b.title, 44, false, v -> open(b)));
        r.setOnClickListener(v -> open(b));
        r.setContentDescription(b.title + (b.author.isEmpty() ? "" : ", by " + b.author) + ". Double tap to listen.");
        LinearLayout.LayoutParams lp = Ui.matchWrap();
        lp.topMargin = u.dp(8);
        r.setLayoutParams(lp);
        return r;
    }

    /** Download the book (EPUB without images; with images if that is all there is), add it, play it. */
    private void open(Book b) {
        @SuppressWarnings("deprecation")
        ProgressDialog pd = ProgressDialog.show(this, null, "Getting “" + b.title + "”…", true);
        new Thread(() -> {
            Doc d = null;
            String error = null;
            try {
                byte[] epub;
                try {
                    epub = get("https://www.gutenberg.org/ebooks/" + b.id + ".epub.noimages", 64 << 20);
                } catch (Exception e) {
                    epub = get("https://www.gutenberg.org/ebooks/" + b.id + ".epub3.images", 64 << 20);
                }
                d = Extract.epub(epub, b.title, "https://www.gutenberg.org/ebooks/" + b.id);
                d.save(getApplicationContext());
            } catch (Extract.Unsupported e) {
                error = e.getMessage();
            } catch (Exception | OutOfMemoryError e) {
                error = "Couldn't download this book. Check the connection and try again.";
            }
            Doc done = d;
            String msg = error;
            runOnUiThread(() -> {
                if (isDestroyed()) return;
                pd.dismiss();
                if (done != null) startActivity(ReaderActivity.open(this, done.id, true));
                else new AlertDialog.Builder(this).setTitle("Couldn't get the book").setMessage(msg).setPositiveButton("OK", null).show();
            });
        }, "steertts-book").start();
    }

    // ---------------------------------------------------------------- the catalogue
    private static final Pattern ENTRY = Pattern.compile("(?s)<entry>(.*?)</entry>");
    private static final Pattern ID = Pattern.compile("/ebooks/(\\d+)\\.opds");
    private static final Pattern TITLE = Pattern.compile("(?s)<title>(.*?)</title>");
    private static final Pattern AUTHOR = Pattern.compile("(?s)<content type=\"text\">(.*?)</content>");

    /** The books in an OPDS page (entries that are not books, such as "sort by", have no numeric id). */
    static List<Book> parse(byte[] xml) {
        String s = new String(xml, java.nio.charset.StandardCharsets.UTF_8);
        List<Book> out = new ArrayList<>();
        Matcher e = ENTRY.matcher(s);
        while (e.find()) {
            String x = e.group(1);
            Matcher id = ID.matcher(x), t = TITLE.matcher(x), a = AUTHOR.matcher(x);
            if (!id.find() || !t.find()) continue;
            String title = Extract.decode(t.group(1)).replaceAll("\\s+", " ").trim();
            String author = a.find() ? Extract.decode(a.group(1)).replaceAll("\\s+", " ").trim() : "";
            out.add(new Book(id.group(1), title, author));
        }
        return out;
    }

    private static String enc(String q) {
        try {
            return URLEncoder.encode(q, "UTF-8");
        } catch (Exception e) {
            return q;
        }
    }

    static byte[] get(String url, int max) throws Exception {
        HttpURLConnection c = null;
        for (int hop = 0; hop < 5; ++hop) {
            c = (HttpURLConnection) new URL(url).openConnection();
            c.setInstanceFollowRedirects(false);
            c.setConnectTimeout(10000);
            c.setReadTimeout(20000);
            c.setRequestProperty("User-Agent", "SteerTTS (Android; free books from Project Gutenberg)");
            int code = c.getResponseCode();
            String loc = c.getHeaderField("Location");
            if (code / 100 != 3 || loc == null) break;
            c.disconnect();
            url = new URL(new URL(url), loc).toString().replaceFirst("^http://", "https://");
        }
        if (c.getResponseCode() >= 400) throw new java.io.IOException("HTTP " + c.getResponseCode());
        try (InputStream in = c.getInputStream()) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[1 << 16];
            for (int n; (n = in.read(buf)) > 0; ) {
                out.write(buf, 0, n);
                if (out.size() > max) throw new Extract.Unsupported("This book is too large to download.");
            }
            return out.toByteArray();
        } finally {
            c.disconnect();
        }
    }
}
