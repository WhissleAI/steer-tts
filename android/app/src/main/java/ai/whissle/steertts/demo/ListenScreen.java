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
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

/**
 * The Listen tab, the app's home: what SteerTTS is saying right now (a readout, a call), what is playing (or how to
 * start), four ways to add something to hear (paste, a file, a link, typed text) and the queue that plays next. One
 * gentle setup nudge appears until SteerTTS is the phone's voice. Documents open in the player (ReaderActivity).
 */
final class ListenScreen implements Narrator.Listener {
    static final int PICK = 41;

    private final Activity a;
    private final Ui u;
    private final View.OnClickListener settings;
    private LinearLayout live, hero, nudge, queue, finished;
    private TextView liveWho, liveText, liveWhy;
    private Ui.Bars bars;
    private boolean showFinished;

    ListenScreen(Activity a, Ui u, View.OnClickListener settings) {
        this.a = a;
        this.u = u;
        this.settings = settings;
    }

    View build() {
        LinearLayout root = u.column();
        root.setPadding(u.dp(18), u.dp(14), u.dp(18), u.dp(28));
        root.addView(u.topBar("Listen", settings));

        // what the assistant is saying now (hidden when quiet)
        live = u.card();
        LinearLayout lr = u.row();
        bars = new Ui.Bars(a, Ui.ACCENT);
        lr.addView(bars, new LinearLayout.LayoutParams(u.dp(22), u.dp(22)));
        liveWho = u.text("", 13, Ui.ACCENT, true);
        liveWho.setPadding(u.dp(10), 0, 0, 0);
        lr.addView(liveWho, Ui.weight(1));
        live.addView(lr);
        liveText = u.text("", 16, Ui.INK, false);
        liveText.setMaxLines(3);
        liveText.setEllipsize(TextUtils.TruncateAt.END);
        liveText.setPadding(0, u.dp(8), 0, 0);
        live.addView(liveText);
        liveWhy = u.text("", 12.5f, Ui.MUTED, false);
        liveWhy.setPadding(0, u.dp(6), 0, 0);
        live.addView(liveWhy);
        live.setVisibility(View.GONE);
        live.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        root.addView(live);

        hero = u.column();
        root.addView(hero, Ui.matchWrap());

        LinearLayout acts = u.row();
        acts.setPadding(0, u.dp(18), 0, 0);
        acts.addView(u.action(R.drawable.ic_paste, "Paste", v -> readClipboard()), Ui.weight(1));
        acts.addView(u.action(R.drawable.ic_file, "File", v -> openFile()), Ui.weight(1));
        acts.addView(u.action(R.drawable.ic_link, "Link", v -> openLink()), Ui.weight(1));
        acts.addView(u.action(R.drawable.ic_text, "Type", v -> pasteText()), Ui.weight(1));
        root.addView(acts, Ui.matchWrap());

        nudge = u.column();
        root.addView(nudge, Ui.matchWrap());

        queue = u.column();
        root.addView(queue, Ui.matchWrap());
        finished = u.column();
        root.addView(finished, Ui.matchWrap());

        ScrollView sv = new ScrollView(a);
        sv.setFillViewport(true);
        sv.addView(root);
        return sv;
    }

    void refresh() {
        changed();
        nudge();
        queue.removeAllViews();
        finished.removeAllViews();
        Narrator n = Narrator.get(a);
        String current = n.doc() != null ? n.doc().id : "";
        if (n.doc() == null)
            for (Doc x : Doc.all(a))
                if (!x.done && x.pos > 0) {   // the hero offers it (changed()): not listed twice
                    current = x.id;
                    break;
                }
        List<Doc> next = Doc.upNext(a, current), done = new ArrayList<>();
        for (Doc d : Doc.all(a)) if (d.done && !d.id.equals(current)) done.add(d);
        if (!next.isEmpty()) {
            queue.addView(u.section("Up next"));
            LinearLayout card = u.card();
            card.setPadding(u.dp(6), u.dp(4), u.dp(6), u.dp(4));
            for (int i = 0; i < next.size(); ++i) {
                if (i > 0) card.addView(u.divider());
                card.addView(docRow(next.get(i)));
            }
            queue.addView(card);
        }
        if (!done.isEmpty()) {
            TextView h = u.section((showFinished ? "▾  " : "▸  ") + "Finished · " + done.size());
            h.setTextColor(Ui.MUTED);
            h.setOnClickListener(v -> {
                showFinished = !showFinished;
                refresh();
            });
            finished.addView(h);
            if (showFinished) {
                LinearLayout card = u.card();
                card.setPadding(u.dp(6), u.dp(4), u.dp(6), u.dp(4));
                for (int i = 0; i < done.size(); ++i) {
                    if (i > 0) card.addView(u.divider());
                    card.addView(docRow(done.get(i)));
                }
                finished.addView(card);
            }
        }
    }

    /** What the assistant is saying (from MainActivity's style receiver); who null = quiet. */
    void speaking(String who, String text, String why) {
        if (live == null) return;
        if (who == null) {
            bars.play(false);
            live.setVisibility(View.GONE);
            return;
        }
        live.setVisibility(View.VISIBLE);
        liveWho.setText(who);
        liveText.setText(text);
        liveWhy.setText(why == null || why.isEmpty() ? "" : "Sounds " + why);
        liveWhy.setVisibility(why == null || why.isEmpty() ? View.GONE : View.VISIBLE);
        bars.play(true);
    }

    @Override public void changed() {
        if (hero == null) return;
        Narrator n = Narrator.get(a);
        Doc d = n.doc();
        boolean resume = false;
        if (d == null) {   // after a restart: offer the document opened most recently and not finished
            for (Doc x : Doc.all(a))
                if (!x.done && x.pos > 0) {
                    d = x;
                    resume = true;
                    break;
                }
        }
        final Doc doc = d;
        final boolean loaded = !resume;
        hero.removeAllViews();
        LinearLayout c = u.card();
        c.setPadding(u.dp(20), u.dp(20), u.dp(20), u.dp(20));
        c.setBackground(Ui.round(Ui.ACCENT_SOFT, u.dp(26)));
        if (d == null) {
            c.addView(u.badge(R.drawable.ic_listen, Ui.ACCENT, Ui.ON_ACCENT, 48));
            TextView t = u.text("Listen to anything", 22, Ui.INK, true);
            t.setPadding(0, u.dp(14), 0, u.dp(4));
            c.addView(t);
            c.addView(u.text("Copy some text, open a PDF or an e-book, or share a page to SteerTTS from any app. "
                    + "It's read in a natural voice, right on your phone.", 14.5f, Ui.MUTED, false));
            hero.addView(c);
            return;
        }
        TextView label = u.text(n.playing() ? "NOW PLAYING" : "CONTINUE", 11.5f, Ui.ACCENT, true);
        label.setLetterSpacing(0.1f);
        c.addView(label);
        LinearLayout r = u.row();
        LinearLayout t = u.column();
        TextView title = u.text(Narration.display(d.title), 20, Ui.INK, true);
        title.setMaxLines(2);
        title.setEllipsize(TextUtils.TruncateAt.END);
        title.setPadding(0, u.dp(6), 0, u.dp(4));
        t.addView(title);
        String sleep = n.sleepLabel();
        t.addView(u.text(d.minutesLeft(n.pace() * Style.load(a).pace) + " min left · " + Math.round(100 * d.progress()) + "%"
                + (sleep != null ? " · sleep in " + sleep.toLowerCase() : ""), 13.5f, Ui.MUTED, false));
        r.addView(t, Ui.weight(1));
        r.addView(u.icon(n.playing() ? R.drawable.ic_pause : R.drawable.ic_play, n.playing() ? "Pause" : "Play", 60, true, v -> {
            if (!loaded) n.open(doc);
            n.toggle();
        }));
        c.addView(r);
        View bar = u.progress(d.progress());
        LinearLayout.LayoutParams lp = Ui.matchWrap();
        lp.topMargin = u.dp(14);
        c.addView(bar, lp);
        c.setOnClickListener(v -> a.startActivity(ReaderActivity.open(a, doc.id, false)));
        c.setContentDescription((n.playing() ? "Now playing " : "Continue ") + d.title + ". Opens the player.");
        hero.addView(c);
    }

    /** "Make SteerTTS your phone's voice", until it is or the user closes it. */
    private void nudge() {
        nudge.removeAllViews();
        if (SettingsScreen.isPhoneVoice(a) || a.getSharedPreferences("ui", Context.MODE_PRIVATE).getBoolean("nudge_voice_off", false)) return;
        LinearLayout c = u.card();
        LinearLayout r = u.row();
        r.addView(u.badge(R.drawable.ic_speaker, Ui.ACCENT_SOFT, Ui.ACCENT, 40));
        LinearLayout t = u.column();
        t.setPadding(u.dp(12), 0, u.dp(4), 0);
        t.addView(u.text("Make it your phone's voice", 15, Ui.INK, true));
        t.addView(u.text("Maps, screen readers and other apps will speak with SteerTTS.", 13, Ui.MUTED, false));
        r.addView(t, Ui.weight(1));
        android.widget.ImageView x = u.img(R.drawable.ic_close, Ui.MUTED, 40);
        x.setPadding(u.dp(10), u.dp(10), u.dp(10), u.dp(10));
        x.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_YES);
        x.setContentDescription("Dismiss");
        x.setOnClickListener(v -> {
            a.getSharedPreferences("ui", Context.MODE_PRIVATE).edit().putBoolean("nudge_voice_off", true).apply();
            nudge();
        });
        r.addView(x);
        c.addView(r);
        c.setOnClickListener(v -> SettingsScreen.openTtsSettings(a));
        LinearLayout.LayoutParams lp = Ui.matchWrap();
        lp.topMargin = u.dp(18);
        nudge.addView(c, lp);
    }

    private View docRow(Doc d) {
        LinearLayout r = u.row();
        r.setPadding(u.dp(10), u.dp(12), u.dp(6), u.dp(12));
        r.addView(u.badge(d.kind.equals("web") ? R.drawable.ic_link : d.kind.equals("text") ? R.drawable.ic_text : R.drawable.ic_file,
                Ui.ACCENT_SOFT, Ui.ACCENT, 40));
        LinearLayout t = u.column();
        t.setPadding(u.dp(12), 0, u.dp(8), 0);
        TextView title = u.text(Narration.display(d.title), 15.5f, Ui.INK, true);
        title.setMaxLines(2);
        title.setEllipsize(TextUtils.TruncateAt.END);
        t.addView(title);
        String kind = d.kind.equals("epub") ? "E-book" : d.kind.equals("pdf") ? "PDF" : d.kind.equals("docx") ? "Word"
                : d.kind.equals("web") ? "Web page" : "Text";
        String meta = d.done ? kind + " · finished"
                : kind + " · " + (d.progress() > 0 ? Math.round(100 * d.progress()) + "% · " : "")
                + d.minutesLeft(Narrator.get(a).pace() * Style.load(a).pace) + " min";
        t.addView(u.text(meta, 13, Ui.MUTED, false));
        r.addView(t, Ui.weight(1));
        r.addView(u.icon(R.drawable.ic_play, "Play " + d.title, 44, false, v -> a.startActivity(ReaderActivity.open(a, d.id, true))));
        r.setOnClickListener(v -> a.startActivity(ReaderActivity.open(a, d.id, false)));
        r.setOnLongClickListener(v -> {
            new AlertDialog.Builder(a).setTitle("Remove from your list?").setMessage(d.title)
                    .setPositiveButton("Remove", (x, w) -> {
                        if (Narrator.get(a).doc() != null && Narrator.get(a).doc().id.equals(d.id)) Narrator.get(a).close();
                        Doc.delete(a, d.id);
                        refresh();
                    })
                    .setNegativeButton("Keep", null).show();
            return true;
        });
        r.setContentDescription(d.title + ", " + meta + ". Long press to remove.");
        return r;
    }

    // ---------------------------------------------------------------- sources
    void openFile() {
        a.startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*")
                .putExtra(Intent.EXTRA_MIME_TYPES, Extract.MIME_TYPES), PICK);
    }

    /** From MainActivity.onActivityResult. */
    void picked(Uri uri) {
        if (uri != null) prepare(a, () -> Extract.fromUri(a, uri, null));
    }

    void readClipboard() {
        ClipboardManager cm = (ClipboardManager) a.getSystemService(Context.CLIPBOARD_SERVICE);
        ClipData clip = cm.getPrimaryClip();
        CharSequence t = clip != null && clip.getItemCount() > 0 ? clip.getItemAt(0).coerceToText(a) : null;
        if (t == null || t.toString().trim().isEmpty()) {
            Toast.makeText(a, "Nothing copied yet. Copy some text or a link first.", Toast.LENGTH_SHORT).show();
            return;
        }
        String s = t.toString().trim();
        prepare(a, () -> Article.isLink(s) ? Extract.fromUrl(s) : Extract.fromText(s, null));
    }

    private void pasteText() {
        EditText e = new EditText(a);
        e.setMinLines(4);
        e.setGravity(Gravity.TOP);
        e.setHint("Type or paste what you'd like to hear");
        e.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        new AlertDialog.Builder(a).setTitle("Listen to text").setView(pad(e))
                .setPositiveButton("Listen", (d, w) -> {
                    String s = e.getText().toString().trim();
                    if (!s.isEmpty()) prepare(a, () -> Extract.fromText(s, null));
                })
                .setNegativeButton("Cancel", null).show();
    }

    private void openLink() {
        EditText e = new EditText(a);
        e.setHint("https://…");
        e.setSingleLine(true);
        e.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        new AlertDialog.Builder(a).setTitle("Listen to a web page").setMessage("The article is read, not the menus and ads.")
                .setView(pad(e))
                .setPositiveButton("Listen", (d, w) -> {
                    String s = e.getText().toString().trim();
                    if (!s.isEmpty()) prepare(a, () -> Extract.fromUrl(s.startsWith("http") ? s : "https://" + s));
                })
                .setNegativeButton("Cancel", null).show();
    }

    private View pad(View v) {
        android.widget.FrameLayout f = new android.widget.FrameLayout(a);
        f.setPadding(u.dp(20), u.dp(8), u.dp(20), 0);
        f.addView(v);
        return f;
    }

    interface Source { Doc get() throws Exception; }

    /** Extracts off the main thread (with a progress note), saves to the list and opens the player, playing. */
    static void prepare(Activity a, Source s) {
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
