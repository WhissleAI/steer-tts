package ai.whissle.steertts.demo;

import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AbsListView;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.SeekBar;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;

/**
 * The player: the document as a list of sentences, the one being heard highlighted and followed as reading moves on
 * (unless the listener is scrolling); tap a sentence to read from there. Below: progress through the document with
 * the time left, previous / play-pause / next, speed, and the voice. Extras: "doc" (a library id), "autoplay".
 */
public class ReaderActivity extends Activity implements Narrator.Listener {
    static final float[] SPEEDS = {0.8f, 1.0f, 1.25f, 1.5f, 2.0f};

    private Narrator narrator;
    private Ui u;
    private ListView list;
    private Adapter adapter;
    private TextView title, left, voiceLine, sleepBtn;
    private SeekBar seek;
    private ImageView playBtn;
    private TextView speedBtn;
    private long userScrolledAt;
    private String shown;   // the id of the document the screen was laid out for
    private android.widget.FrameLayout stage;
    private PdfPagesView pages;   // the PDF itself, for PDFs
    private TextView viewChip;

    static Intent open(android.content.Context c, String docId, boolean autoplay) {
        return new Intent(c, ReaderActivity.class).putExtra("doc", docId).putExtra("autoplay", autoplay)
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);   // one player, never a stack
    }

    /** Another document opened while the player is up: show and play that one instead. */
    @Override protected void onNewIntent(Intent i) {
        super.onNewIntent(i);
        setIntent(i);
        Doc d = i.getStringExtra("doc") == null ? null : Doc.load(this, i.getStringExtra("doc"));
        if (d != null) narrator.open(d);
        if (narrator.doc() != null) changed();
        if (i.getBooleanExtra("autoplay", false) && !narrator.playing()) narrator.play();
    }

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        Ui.init(this);
        u = new Ui(this);
        narrator = Narrator.get(this);
        String id = b == null ? getIntent().getStringExtra("doc") : null;   // recreated (rotation): keep what is playing now
        if (id != null) {
            Doc d = Doc.load(this, id);
            if (d != null) narrator.open(d);
        }
        if (narrator.doc() == null) {
            finish();
            return;
        }
        build();
        if (b == null && getIntent().getBooleanExtra("autoplay", false) && !narrator.playing()) narrator.play();
        getIntent().removeExtra("autoplay");
    }

    @Override protected void onStart() {
        super.onStart();
        if (narrator.doc() == null) return;
        narrator.addListener(this);
        changed();
        list.setSelectionFromTop(narrator.position(), u.dp(120));
    }

    @Override protected void onStop() {
        narrator.removeListener(this);
        super.onStop();
    }

    @Override protected void onDestroy() {
        if (pages != null) pages.close();
        super.onDestroy();
    }

    private void build() {
        Doc d = narrator.doc();
        LinearLayout root = u.column();
        root.setBackgroundColor(Ui.BG);

        LinearLayout top = u.row();
        top.setPadding(u.dp(8), u.dp(8), u.dp(16), u.dp(4));
        TextView back = u.text("‹", 30, Ui.INK, false);
        back.setContentDescription("Back");
        back.setPadding(u.dp(10), 0, u.dp(14), u.dp(4));
        back.setOnClickListener(v -> finish());
        top.addView(back);
        title = u.text("", 19, Ui.INK, false);
        title.setText(heading(d.title));
        title.setMaxLines(3);
        title.setEllipsize(android.text.TextUtils.TruncateAt.END);
        top.addView(title, Ui.weight(1));
        root.addView(top);

        list = new ListView(this);
        list.setDivider(null);
        list.setClipToPadding(false);
        list.setPadding(u.dp(12), u.dp(8), u.dp(12), u.dp(24));
        adapter = new Adapter();
        list.setAdapter(adapter);
        list.setOnItemClickListener((p, v, pos, idt) -> {
            narrator.seek(pos);
            if (!narrator.playing()) narrator.play();
        });
        list.setOnScrollListener(new AbsListView.OnScrollListener() {
            @Override public void onScrollStateChanged(AbsListView v, int s) {
                if (s != SCROLL_STATE_IDLE) userScrolledAt = System.currentTimeMillis();
            }
            @Override public void onScroll(AbsListView v, int f, int n, int t) { }
        });
        stage = new android.widget.FrameLayout(this);
        stage.addView(list);
        root.addView(stage, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        viewChip = u.chip("Text", v -> showPages(!pagesShown()));
        viewChip.setContentDescription("Switch between the PDF pages and the text");
        top.addView(viewChip);
        setUpPages(d);

        LinearLayout panel = u.column();
        panel.setBackground(Ui.round(Ui.CARD, u.dp(28)));
        panel.setPadding(u.dp(18), u.dp(8), u.dp(18), u.dp(6));
        seek = new SeekBar(this);
        seek.setMax(Math.max(1, d.sentences.size() - 1));
        seek.setContentDescription("Position in the document");
        seek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar s, int p, boolean fromUser) { }
            @Override public void onStartTrackingTouch(SeekBar s) { }
            @Override public void onStopTrackingTouch(SeekBar s) { narrator.seek(s.getProgress()); }
        });
        panel.addView(seek);
        left = u.text("", 13, Ui.MUTED, false);
        left.setGravity(Gravity.CENTER);
        panel.addView(left, Ui.matchWrap());

        // one row: sleep timer, back, play, forward, speed (the page above gets the room)
        LinearLayout controls = u.row();
        controls.setGravity(Gravity.CENTER_VERTICAL);
        controls.setPadding(0, u.dp(6), 0, u.dp(2));
        sleepBtn = u.chip("", v -> sleepMenu());
        sleepBtn.setMinWidth(u.dp(64));
        sleepBtn.setGravity(Gravity.CENTER);
        controls.addView(sleepBtn);
        controls.addView(new View(this), Ui.spacer());
        controls.addView(u.icon(R.drawable.ic_prev, "Previous sentence", 48, false, v -> narrator.prev()));
        playBtn = u.icon(R.drawable.ic_play, "Play", 64, true, v -> narrator.toggle());
        LinearLayout.LayoutParams pl = (LinearLayout.LayoutParams) playBtn.getLayoutParams();
        pl.leftMargin = pl.rightMargin = u.dp(18);
        controls.addView(playBtn);
        controls.addView(u.icon(R.drawable.ic_next, "Next sentence", 48, false, v -> narrator.next()));
        controls.addView(new View(this), Ui.spacer());
        speedBtn = u.chip("", v -> {   // a tap moves to the next speed
            float p = narrator.pace();
            int k = 0;
            for (int i = 0; i < SPEEDS.length; ++i) if (Math.abs(SPEEDS[i] - p) < 0.01f) k = i;
            narrator.setPace(SPEEDS[(k + 1) % SPEEDS.length]);
            showSpeed();
        });
        speedBtn.setMinWidth(u.dp(64));
        speedBtn.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams sl = (LinearLayout.LayoutParams) speedBtn.getLayoutParams();
        sl.rightMargin = 0;
        controls.addView(speedBtn);
        panel.addView(controls, Ui.matchWrap());
        voiceLine = u.text("", 13, Ui.ACCENT, true);
        voiceLine.setGravity(Gravity.CENTER);
        voiceLine.setMinHeight(u.dp(36));
        voiceLine.setOnClickListener(v -> startActivity(new Intent(this, MainActivity.class).putExtra("tab", "voices")
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP)));
        panel.addView(voiceLine, Ui.matchWrap());
        root.addView(panel);

        setContentView(root);
        if (Build.VERSION.SDK_INT >= 30)
            root.setOnApplyWindowInsetsListener((v, ins) -> {
                android.graphics.Insets bars = ins.getInsets(android.view.WindowInsets.Type.systemBars());
                v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
                return android.view.WindowInsets.CONSUMED;
            });
        showSpeed();
    }

    // ---------------------------------------------------------------- the PDF's own pages
    /** For a PDF that kept its file: the page view, shown by default (the reader's choice is remembered). */
    private void setUpPages(Doc d) {
        if (pages != null) {
            stage.removeView(pages);
            pages.close();
            pages = null;
        }
        java.io.File f = PdfText.file(this, d.id);
        boolean can = d.regions != null && f.exists();
        viewChip.setVisibility(can ? View.VISIBLE : View.GONE);
        if (!can) {
            list.setVisibility(View.VISIBLE);
            return;
        }
        pages = new PdfPagesView(this, f, this::tappedPage);
        pages.setPadding(u.dp(10), u.dp(8), u.dp(10), u.dp(24));
        pages.setOnScrollListener(new AbsListView.OnScrollListener() {
            @Override public void onScrollStateChanged(AbsListView v, int st) {
                if (st != SCROLL_STATE_IDLE) userScrolledAt = System.currentTimeMillis();
            }
            @Override public void onScroll(AbsListView v, int f1, int n, int t) { }
        });
        stage.addView(pages);
        showPages(getSharedPreferences("reader", MODE_PRIVATE).getBoolean("pdf_pages", true));
    }

    private boolean pagesShown() { return pages != null && pages.getVisibility() == View.VISIBLE; }

    private void showPages(boolean on) {
        if (pages == null) return;
        getSharedPreferences("reader", MODE_PRIVATE).edit().putBoolean("pdf_pages", on).apply();
        pages.setVisibility(on ? View.VISIBLE : View.GONE);
        list.setVisibility(on ? View.GONE : View.VISIBLE);
        viewChip.setText(on ? "Text" : "Page");
        userScrolledAt = 0;
        if (playBtn != null) changed();   // not while the screen is still being built
    }

    /** Read from the sentence printed where the page was tapped (or the nearest one on that page). */
    private void tappedPage(int page, float fx, float fy) {
        Doc d = narrator.doc();
        if (d == null || d.regions == null) return;
        int best = -1;
        float bestDist = Float.MAX_VALUE;
        for (int i = 0; i < d.regions.size(); ++i)
            for (float[] b : d.regions.get(i)) {
                if ((int) b[0] != page) continue;
                float dy = fy < b[2] ? b[2] - fy : fy > b[4] ? fy - b[4] : 0, dx = fx < b[1] ? b[1] - fx : fx > b[3] ? fx - b[3] : 0;
                float dist = dy * 4 + dx;   // the line matters more than the column
                if (dist < bestDist) {
                    bestDist = dist;
                    best = i;
                }
            }
        if (best < 0 || bestDist > 0.08f) return;
        userScrolledAt = 0;
        narrator.seek(best);
        if (!narrator.playing()) narrator.play();
    }

    private void sleepMenu() {
        String[] labels = {"Off", "In 15 minutes", "In 30 minutes", "In 45 minutes", "In an hour", "At the end of this document"};
        int[] minutes = {-1, 15, 30, 45, 60, 0};
        new android.app.AlertDialog.Builder(this).setTitle("Sleep timer")
                .setItems(labels, (d, w) -> {
                    narrator.setSleep(minutes[w]);
                    changed();
                })
                .show();
    }

    private static String trim(float s) { return String.valueOf(s).replaceAll("\\.?0+$", ""); }

    private void showSpeed() {
        float p = narrator.pace();
        speedBtn.setText(trim(p) + "×");
        speedBtn.setContentDescription("Speed " + trim(p) + ". Tap for the next speed.");
        u.select(speedBtn, Math.abs(p - 1f) > 0.01f);
    }

    @Override public void changed() {
        Doc d = narrator.doc();
        if (d == null) {
            finish();
            return;
        }
        if (!d.id.equals(shown)) {   // the narrator moved to another document
            boolean first = shown == null;
            shown = d.id;
            if (!first) setUpPages(d);
            title.setText(heading(d.title));
            seek.setMax(Math.max(1, d.sentences.size() - 1));
            list.setSelectionFromTop(narrator.position(), u.dp(120));
        }
        boolean playing = narrator.playing();
        playBtn.setImageResource(playing ? R.drawable.ic_pause : R.drawable.ic_play);
        playBtn.setContentDescription(playing ? "Pause" : "Play");
        int at = Math.min(narrator.position(), Math.max(0, d.sentences.size() - 1));
        seek.setProgress(at);
        left.setText(at >= d.sentences.size() - 1 && !playing && narrator.position() >= d.sentences.size()
                ? "Finished" : d.minutesLeft(narrator.pace() * Style.load(this).pace) + " min left · "
                + Math.round(100 * d.progress()) + "%");
        voiceLine.setText(Voices.get(this).nameOf(Style.load(this).voice, null) + "  ·  change voice");
        String sl = narrator.sleepLabel();
        sleepBtn.setText(sl == null ? "☾" : "☾ " + sl.replace("End of document", "end"));
        sleepBtn.setContentDescription(sl == null ? "Sleep timer" : "Sleep timer: " + sl);
        u.select(sleepBtn, sl != null);
        adapter.notifyDataSetChanged();
        if (pagesShown()) {
            List<float[]> boxes = d.regions != null && at < d.regions.size() ? d.regions.get(at) : null;
            pages.highlight(boxes);
            if (System.currentTimeMillis() - userScrolledAt > 4000 && boxes != null && !boxes.isEmpty()) pages.reveal(boxes.get(0));
        } else if (playing && System.currentTimeMillis() - userScrolledAt > 4000)
            list.smoothScrollToPositionFromTop(at, u.dp(140), 350);
    }

    private final class Adapter extends BaseAdapter {
        @Override public int getCount() { return narrator.doc() == null ? 0 : narrator.doc().sentences.size(); }
        @Override public Object getItem(int i) { return narrator.doc().sentences.get(i); }
        @Override public long getItemId(int i) { return i; }

        @Override public View getView(int i, View v, ViewGroup parent) {
            TextView t = v instanceof TextView ? (TextView) v : u.text("", 18, Ui.INK, false);
            t.setLineSpacing(0, 1.25f);
            t.setPadding(u.dp(12), u.dp(6), u.dp(12), u.dp(6));
            String raw = narrator.doc().sentences.get(i);
            boolean heading = !raw.isEmpty() && raw.charAt(0) == Narration.HEAD;
            t.setText(Narration.display(raw));
            t.setTextSize(heading ? 21 : 18);
            t.setTypeface(heading ? android.graphics.Typeface.create("sans-serif-medium", android.graphics.Typeface.NORMAL)
                    : android.graphics.Typeface.DEFAULT);
            boolean cur = i == narrator.position();
            t.setBackground(cur ? Ui.round(Ui.ACCENT_SOFT, u.dp(12)) : null);
            t.setTextColor(cur ? Ui.INK : Ui.MUTED);
            return t;
        }
    }

    /** The title, and the author on a smaller, quieter second line. */
    private static CharSequence heading(String full) {
        String[] ta = ListenScreen.titleAndAuthor(full);
        android.text.SpannableString s = new android.text.SpannableString(ta[1] == null ? ta[0] : ta[0] + "\n" + ta[1]);
        s.setSpan(new android.text.style.StyleSpan(android.graphics.Typeface.BOLD), 0, ta[0].length(), 0);
        if (ta[1] != null) {
            int a = ta[0].length() + 1;
            s.setSpan(new android.text.style.RelativeSizeSpan(0.74f), a, s.length(), 0);
            s.setSpan(new android.text.style.ForegroundColorSpan(Ui.MUTED), a, s.length(), 0);
        }
        return s;
    }
}
