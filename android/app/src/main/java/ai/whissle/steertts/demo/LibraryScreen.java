package ai.whissle.steertts.demo;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.text.InputType;
import android.view.View;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.util.List;

/**
 * The Read tab: everything to listen to. Open a file (PDF, EPUB, Word, web page, text), paste text, read the
 * clipboard or a link; continue the current document; and the reading list with progress. Documents open in the
 * player (ReaderActivity).
 */
final class LibraryScreen implements Narrator.Listener {
    static final int PICK = 41;

    private final Activity a;
    private final Ui u;
    private LinearLayout nowCard, listBox;

    LibraryScreen(Activity a, Ui u) {
        this.a = a;
        this.u = u;
    }

    View build() {
        LinearLayout root = u.column();
        root.setPadding(u.dp(16), u.dp(20), u.dp(16), u.dp(24));
        root.addView(u.header("Read", "Listen to documents, books and web pages"));

        nowCard = u.card();
        root.addView(nowCard);

        LinearLayout add = u.card();
        add.addView(u.label("Listen to"));
        add.addView(u.settingRow("A file", "PDF, EPUB e-book, Word, web page or text", u.chevron(), v -> openFile()));
        add.addView(u.divider());
        add.addView(u.settingRow("What you copied", "Read the clipboard", u.chevron(), v -> readClipboard()));
        add.addView(u.divider());
        add.addView(u.settingRow("Text you paste or type", "Anything: notes, an email, a chapter", u.chevron(), v -> pasteText()));
        add.addView(u.divider());
        add.addView(u.settingRow("A web page", "Paste a link; the article is read, not the menus", u.chevron(), v -> openLink()));
        root.addView(add);

        TextView tip = u.text("Tip: in any app, select text or tap Share, then choose “Read with SteerTTS”.", 13, Ui.MUTED, false);
        tip.setPadding(u.dp(4), u.dp(12), u.dp(4), 0);
        root.addView(tip);

        TextView lh = u.label("Your list");
        lh.setPadding(u.dp(4), u.dp(20), 0, 0);
        root.addView(lh);
        listBox = u.column();
        root.addView(listBox);

        ScrollView sv = new ScrollView(a);
        sv.addView(root);
        return sv;
    }

    void refresh() {
        changed();
        listBox.removeAllViews();
        List<Doc> docs = Doc.all(a);
        if (docs.isEmpty()) {
            LinearLayout c = u.card();
            c.addView(u.text("Nothing here yet. Open a PDF or an e-book, or share an article to SteerTTS.", 14, Ui.MUTED, false));
            listBox.addView(c);
        }
        for (Doc d : docs) listBox.addView(docRow(d));
    }

    @Override public void changed() {
        Narrator n = Narrator.get(a);
        Doc d = n.doc();
        nowCard.removeAllViews();
        nowCard.setVisibility(d == null ? View.GONE : View.VISIBLE);
        if (d == null) return;
        nowCard.addView(u.label(n.playing() ? "Now reading" : "Continue listening"));
        LinearLayout r = u.row();
        LinearLayout t = u.column();
        TextView title = u.text(d.title, 17, Ui.INK, true);
        title.setMaxLines(2);
        t.addView(title);
        t.addView(u.text(d.minutesLeft(n.pace() * Style.load(a).pace) + " min left · " + Math.round(100 * d.progress()) + "%", 13, Ui.MUTED, false));
        r.addView(t, Ui.weight(1));
        r.addView(u.icon(n.playing() ? R.drawable.ic_pause : R.drawable.ic_play, n.playing() ? "Pause" : "Play", 52, true, v -> n.toggle()));
        nowCard.addView(r);
        View bar = u.progress(d.progress());
        LinearLayout.LayoutParams lp = Ui.matchWrap();
        lp.topMargin = u.dp(10);
        nowCard.addView(bar, lp);
        nowCard.setOnClickListener(v -> a.startActivity(ReaderActivity.open(a, d.id, false)));
    }

    private View docRow(Doc d) {
        LinearLayout c = u.card();
        c.setPadding(u.dp(16), u.dp(12), u.dp(16), u.dp(14));
        LinearLayout r = u.row();
        LinearLayout t = u.column();
        TextView title = u.text(d.title, 16, Ui.INK, true);
        title.setMaxLines(2);
        t.addView(title);
        String kind = d.kind.equals("epub") ? "E-book" : d.kind.equals("pdf") ? "PDF" : d.kind.equals("docx") ? "Word"
                : d.kind.equals("web") ? "Web page" : "Text";
        t.addView(u.text(kind + " · " + (d.progress() > 0 ? Math.round(100 * d.progress()) + "% · " : "")
                + d.minutesLeft(Narrator.get(a).pace() * Style.load(a).pace) + " min", 13, Ui.MUTED, false));
        r.addView(t, Ui.weight(1));
        r.addView(u.icon(R.drawable.ic_play, "Play " + d.title, 44, false, v -> a.startActivity(ReaderActivity.open(a, d.id, true))));
        c.addView(r);
        if (d.progress() > 0) {
            LinearLayout.LayoutParams lp = Ui.matchWrap();
            lp.topMargin = u.dp(8);
            c.addView(u.progress(d.progress()), lp);
        }
        c.setOnClickListener(v -> a.startActivity(ReaderActivity.open(a, d.id, false)));
        c.setOnLongClickListener(v -> {
            new AlertDialog.Builder(a).setTitle("Remove from your list?").setMessage(d.title)
                    .setPositiveButton("Remove", (x, w) -> {
                        if (Narrator.get(a).doc() != null && Narrator.get(a).doc().id.equals(d.id)) Narrator.get(a).close();
                        Doc.delete(a, d.id);
                        refresh();
                    })
                    .setNegativeButton("Keep", null).show();
            return true;
        });
        return c;
    }

    // ---------------------------------------------------------------- sources
    void openFile() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*")
                .putExtra(Intent.EXTRA_MIME_TYPES, Extract.MIME_TYPES);
        a.startActivityForResult(i, PICK);
    }

    /** From MainActivity.onActivityResult. */
    void picked(Uri uri) {
        if (uri == null) return;
        prepare(() -> Extract.fromUri(a, uri, null));
    }

    void readClipboard() {
        ClipboardManager cm = (ClipboardManager) a.getSystemService(Context.CLIPBOARD_SERVICE);
        ClipData clip = cm.getPrimaryClip();
        CharSequence t = clip != null && clip.getItemCount() > 0 ? clip.getItemAt(0).coerceToText(a) : null;
        if (t == null || t.toString().trim().isEmpty()) {
            Toast.makeText(a, "The clipboard is empty. Copy some text first.", Toast.LENGTH_SHORT).show();
            return;
        }
        String s = t.toString().trim();
        prepare(() -> Article.isLink(s) ? Extract.fromUrl(s) : Extract.fromText(s, null));
    }

    private void pasteText() {
        EditText e = new EditText(a);
        e.setMinLines(5);
        e.setHint("Paste or type what you'd like to hear");
        e.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        new AlertDialog.Builder(a).setTitle("Text to listen to").setView(e)
                .setPositiveButton("Listen", (d, w) -> {
                    String s = e.getText().toString().trim();
                    if (!s.isEmpty()) prepare(() -> Extract.fromText(s, null));
                })
                .setNegativeButton("Cancel", null).show();
    }

    private void openLink() {
        EditText e = new EditText(a);
        e.setHint("https://…");
        e.setInputType(InputType.TYPE_TEXT_VARIATION_URI);
        new AlertDialog.Builder(a).setTitle("Web page").setView(e)
                .setPositiveButton("Listen", (d, w) -> {
                    String s = e.getText().toString().trim();
                    if (!s.isEmpty()) prepare(() -> Extract.fromUrl(s.startsWith("http") ? s : "https://" + s));
                })
                .setNegativeButton("Cancel", null).show();
    }

    interface Source { Doc get() throws Exception; }

    /** Extracts off the main thread (with a progress note), saves to the list and opens the player. */
    void prepare(Source s) {
        @SuppressWarnings("deprecation")
        ProgressDialog pd = ProgressDialog.show(a, null, "Getting it ready…", true);
        new Thread(() -> {
            try {
                Doc d = s.get();
                if (d.sentences.isEmpty()) throw new Extract.Unsupported("There's no text to read in this one.");
                d.save(a);
                a.runOnUiThread(() -> {
                    pd.dismiss();
                    a.startActivity(ReaderActivity.open(a, d.id, true));
                });
            } catch (Exception ex) {
                a.runOnUiThread(() -> {
                    pd.dismiss();
                    new AlertDialog.Builder(a).setTitle("Couldn't read that")
                            .setMessage(ex instanceof Extract.Unsupported ? ex.getMessage() : "Something went wrong opening it.")
                            .setPositiveButton("OK", null).show();
                });
            }
        }, "steertts-extract").start();
    }
}
