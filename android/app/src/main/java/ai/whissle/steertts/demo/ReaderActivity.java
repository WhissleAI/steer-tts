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
    private final List<TextView> speedChips = new ArrayList<>();
    private long userScrolledAt;
    private String shown;   // the id of the document the screen was laid out for

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
        String id = getIntent().getStringExtra("doc");
        if (id != null) {
            Doc d = Doc.load(this, id);
            if (d != null) narrator.open(d);
        }
        if (narrator.doc() == null) {
            finish();
            return;
        }
        build();
        if (getIntent().getBooleanExtra("autoplay", false) && !narrator.playing()) narrator.play();
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
        title = u.text(d.title, 19, Ui.INK, true);
        title.setMaxLines(2);
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
        root.addView(list, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        LinearLayout panel = u.column();
        panel.setBackground(Ui.round(Ui.CARD, u.dp(28)));
        panel.setPadding(u.dp(20), u.dp(14), u.dp(20), u.dp(16));
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

        LinearLayout controls = u.row();
        controls.setGravity(Gravity.CENTER);
        controls.setPadding(0, u.dp(10), 0, u.dp(10));
        controls.addView(u.icon(R.drawable.ic_prev, "Previous sentence", 56, false, v -> narrator.prev()));
        playBtn = u.icon(R.drawable.ic_play, "Play", 76, true, v -> narrator.toggle());
        LinearLayout.LayoutParams pl = (LinearLayout.LayoutParams) playBtn.getLayoutParams();
        pl.leftMargin = pl.rightMargin = u.dp(28);
        controls.addView(playBtn);
        controls.addView(u.icon(R.drawable.ic_next, "Next sentence", 56, false, v -> narrator.next()));
        panel.addView(controls);

        LinearLayout speeds = u.row();
        speeds.setGravity(Gravity.CENTER);
        for (float s : SPEEDS) {
            TextView ch = u.chip(trim(s) + "×", v -> {
                narrator.setPace(s);
                showSpeed();
            });
            ch.setContentDescription("Speed " + trim(s));
            speedChips.add(ch);
            speeds.addView(ch);
        }
        panel.addView(speeds);
        LinearLayout foot = u.row();
        foot.setPadding(0, u.dp(10), 0, 0);
        sleepBtn = u.chip("", v -> sleepMenu());
        foot.addView(sleepBtn);
        foot.addView(new View(this), Ui.spacer());
        voiceLine = u.text("", 13.5f, Ui.ACCENT, true);
        voiceLine.setMinHeight(u.dp(40));
        voiceLine.setGravity(Gravity.CENTER_VERTICAL);
        voiceLine.setOnClickListener(v -> startActivity(new Intent(this, MainActivity.class).putExtra("tab", "voices")
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP)));
        foot.addView(voiceLine);
        panel.addView(foot, Ui.matchWrap());
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
        for (int i = 0; i < SPEEDS.length; ++i) u.select(speedChips.get(i), Math.abs(SPEEDS[i] - p) < 0.01f);
    }

    @Override public void changed() {
        Doc d = narrator.doc();
        if (d == null) {
            finish();
            return;
        }
        if (!d.id.equals(shown)) {   // the narrator moved to another document
            shown = d.id;
            title.setText(d.title);
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
        voiceLine.setText("Voice: " + Voices.get(this).nameOf(Style.load(this).voice, null) + "  ›");
        String sl = narrator.sleepLabel();
        sleepBtn.setText(sl == null ? "☾  Sleep timer" : "☾  " + sl);
        u.select(sleepBtn, sl != null);
        adapter.notifyDataSetChanged();
        if (playing && System.currentTimeMillis() - userScrolledAt > 4000)
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
}
