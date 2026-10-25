package ai.whissle.steertts.demo;

import android.accessibilityservice.AccessibilityButtonController;
import android.accessibilityservice.AccessibilityService;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * "Read this screen", in any app: the text on screen (paragraphs in reading order; buttons, bars and the status bar left
 * out) is read aloud by the Narrator, the paragraph being read is outlined on the screen, and a small control floats
 * on top while it reads: play / pause, back, skip, auto-continue (at the end of what is visible, scroll on and keep
 * reading what is new) and close. Started from the "Read this screen" Quick Settings tile, the accessibility button
 * or shortcut, or the app. An accessibility service (it can read other apps' windows and scroll them); it reads only
 * when asked, and keeps nothing.
 */
public class ScreenReader extends AccessibilityService implements Narrator.Listener {
    static volatile ScreenReader instance;

    private final Handler main = new Handler(Looper.getMainLooper());
    private WindowManager wm;
    private View controls;
    private Outline outline;
    private TextView playBtn, autoBtn;
    private boolean auto = true;
    private final Set<String> heard = new HashSet<>();   // paragraphs already read on this page (auto-continue skips them)
    private final List<Rect> boxes = new ArrayList<>();  // where each paragraph of the current reading is on screen
    private final List<String> raw = new ArrayList<>();  // each paragraph's text as the screen has it (to find it again)
    private boolean following;                           // a follow-along scroll is under way
    private String readingPkg;                           // the app whose screen is being read
    private String reading;   // the transient document's id while this reader owns the narrator
    private int lastPara = -1;
    private int tries;   // auto-continue: looks after a scroll that found nothing new

    static boolean enabled() { return instance != null; }

    @Override protected void onServiceConnected() {
        instance = this;
        wm = getSystemService(WindowManager.class);
        if (Build.VERSION.SDK_INT >= 26) {
            getAccessibilityButtonController().registerAccessibilityButtonCallback(new AccessibilityButtonController.AccessibilityButtonCallback() {
                @Override public void onClicked(AccessibilityButtonController c) { toggle(); }
            });
        }
    }

    @Override public boolean onUnbind(android.content.Intent i) {
        stop();
        instance = null;
        return super.onUnbind(i);
    }

    private final Runnable relocate = this::settle;

    /**
     * Only while reading: the user scrolled the page (find the paragraphs again), or went to another app (stop: what is
     * being read is no longer on screen).
     */
    @Override public void onAccessibilityEvent(AccessibilityEvent e) {
        if (reading == null || e == null) return;
        CharSequence pkg = e.getPackageName();
        if (e.getEventType() == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED && pkg != null && readingPkg != null
                && !pkg.toString().equals(readingPkg) && !pkg.toString().equals(getPackageName())
                && !pkg.toString().equals("com.android.systemui")) {
            log("left the page for " + pkg + ": stop");
            stop();
        } else if (e.getEventType() == AccessibilityEvent.TYPE_VIEW_SCROLLED && !following) {
            main.removeCallbacks(relocate);
            main.postDelayed(relocate, 250);
        }
    }

    @Override public void onInterrupt() { stop(); }

    /** Start reading the screen, or stop if already reading. */
    void toggle() {
        if (reading != null) stop();
        else read(false);
    }

    /** Debug hook: read whatever app is in front after ms (time to switch to it). */
    void readIn(long ms) { main.postDelayed(() -> read(false), ms); }

    /** From the Quick Settings tile: close the shade first, then read the app underneath. */
    void readAfterShade() {
        if (Build.VERSION.SDK_INT >= 31) performGlobalAction(GLOBAL_ACTION_DISMISS_NOTIFICATION_SHADE);
        main.postDelayed(() -> read(false), 700);
    }

    // ---------------------------------------------------------------- reading
    /** Read what is on screen now. more: auto-continue after a scroll (only paragraphs not read yet). */
    private void read(boolean more) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) {
            if (!more) say("There's nothing to read on this screen.");
            else stop();
            return;
        }
        CharSequence pkg = root.getPackageName();
        if (pkg != null && (pkg.toString().equals(getPackageName()) || pkg.toString().equals("com.android.systemui"))) {
            if (!more) say("Open the page you want to hear, then start reading.");
            return;
        }
        List<Item> items = new ArrayList<>();
        collect(root, items, 0);
        items.sort((a, b) -> Math.abs(a.box.top - b.box.top) > 12 ? Integer.compare(a.box.top, b.box.top) : Integer.compare(a.box.left, b.box.left));
        if (!more) heard.clear();
        List<String> paras = new ArrayList<>();
        boxes.clear();
        raw.clear();
        Set<String> seen = new HashSet<>();
        for (Item it : items) {
            String t = it.text.replaceAll("\\s+", " ").trim();
            if (!seen.add(t) || heard.contains(t)) continue;
            paras.add(TextClean.speakable(t));
            boxes.add(it.box);
            raw.add(t);
            heard.add(t);
        }
        if (paras.isEmpty()) {
            if (more && tries < 2) {   // the page may still be settling, or scrolled too little: look again, scroll once more
                tries++;
                main.postDelayed(tries == 1 ? () -> read(true) : this::ended, 900);
                return;
            }
            if (more) {
                say("That's the end of the page.");
                stop();
            } else {
                say("There's no text to read on this screen.");
            }
            return;
        }
        tries = 0;
        if (SteerTTS.debug(this)) for (String x : paras) android.util.Log.i(MainActivity.TAG, "screen para [" + x.substring(0, Math.min(150, x.length())) + "]");
        String app = label(pkg == null ? "" : pkg.toString());
        readingPkg = pkg == null ? null : pkg.toString();
        Doc d = Doc.createTransient(app, paras);
        reading = d.id;
        lastPara = -1;
        Narrator n = Narrator.get(this);
        n.addListener(this);
        if (more) n.continueTransient(d, this::ended);
        else {
            n.openTransient(d, this::ended);
            n.play();
        }
        showControls();
    }

    /**
     * The end of what was visible: scroll on and read what is new, or stop. A swipe first (it moves what the user sees,
     * as a finger would; a browser may accept a "scroll" action on some inner box and not move the page), then the
     * page's own scroll action; each is checked by whether new text came into view.
     */
    private void ended() {
        if (!auto || reading == null) {
            stop();
            return;
        }
        if (hasNew()) {   // follow-along already brought new text into view: straight on, no page turn
            log("auto-continue: new text on screen");
            read(true);
            return;
        }
        String before = visibleSignature();
        swipe(() -> main.postDelayed(() -> {
            if (moved(before)) {
                log("auto-continue: swiped on");
                read(true);
                return;
            }
            AccessibilityNodeInfo root = getRootInActiveWindow();
            AccessibilityNodeInfo list = root == null ? null : scrollable(root);
            boolean acted = list != null && ((Build.VERSION.SDK_INT >= 23
                    && list.performAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_DOWN.getId()))
                    || list.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD));
            main.postDelayed(() -> {
                if (acted && moved(before)) {
                    log("auto-continue: scrolled on");
                    read(true);
                } else {
                    log("auto-continue: nothing more (acted " + acted + ")");
                    say("That's the end of the page.");
                    stop();
                }
            }, 900);
        }, 900));
    }

    /**
     * Follow along: when the paragraph now being read sits low on the screen, scroll it gently up near the top while
     * the voice goes on, so the text ahead comes into view and reading never stops for a page turn.
     */
    private void follow(int para) {
        if (!auto || following || para < 0 || para >= boxes.size() || Build.VERSION.SDK_INT < 24) return;
        Rect b = boxes.get(para);
        int h = getResources().getDisplayMetrics().heightPixels;
        if (b.isEmpty() || b.bottom < h * 0.78f || b.top < h * 0.22f) return;
        int distance = (int) Math.min(b.top - h * 0.15f, h * 0.55f);
        if (distance < h * 0.1f) return;
        following = true;
        android.graphics.Path path = new android.graphics.Path();
        float x = getResources().getDisplayMetrics().widthPixels * 0.4f, y0 = h * 0.72f;
        path.moveTo(x, y0);
        path.lineTo(x, y0 - distance);
        long ms = Math.max(450, (long) (distance / 0.9f));   // slow enough not to fling the page past
        boolean ok = dispatchGesture(new android.accessibilityservice.GestureDescription.Builder()
                .addStroke(new android.accessibilityservice.GestureDescription.StrokeDescription(path, 0, ms)).build(),
                new GestureResultCallback() {
                    @Override public void onCompleted(android.accessibilityservice.GestureDescription g) { settle(); }
                    @Override public void onCancelled(android.accessibilityservice.GestureDescription g) { settle(); }
                }, main);
        log("follow: scroll " + distance + " px " + ok);
        if (!ok) following = false;
    }

    /** After a scroll (ours or the user's): find the paragraphs again where they now are. */
    private void settle() {
        main.postDelayed(() -> {
            following = false;
            AccessibilityNodeInfo root = getRootInActiveWindow();
            if (root == null || reading == null) return;
            List<Item> items = new ArrayList<>();
            collect(root, items, 0);
            for (int i = 0; i < raw.size(); ++i) {
                Rect at = new Rect();
                for (Item it : items) if (it.text.replaceAll("\\s+", " ").trim().equals(raw.get(i))) { at = it.box; break; }
                boxes.set(i, at);   // empty: scrolled out of view
            }
            if (outline != null && lastPara >= 0 && lastPara < boxes.size()) outline.show(boxes.get(lastPara).isEmpty() ? null : boxes.get(lastPara));
        }, 350);
    }

    /** Paragraphs on screen that have not been read yet. */
    private boolean hasNew() {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return false;
        List<Item> items = new ArrayList<>();
        collect(root, items, 0);
        for (Item it : items) if (!heard.contains(it.text.replaceAll("\\s+", " ").trim())) return true;
        return false;
    }

    /** The visible text, joined: compared before and after a scroll. */
    private String visibleSignature() {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return "";
        List<Item> items = new ArrayList<>();
        collect(root, items, 0);
        StringBuilder b = new StringBuilder();
        for (Item it : items) b.append(it.text).append('|').append(it.box.top).append('\n');
        return b.toString();
    }

    private boolean moved(String before) { return !visibleSignature().equals(before); }

    /** A swipe up the middle of the screen (away from the floating control); then runs next. */
    private void swipe(Runnable next) {
        if (Build.VERSION.SDK_INT < 24) {
            next.run();
            return;
        }
        android.util.DisplayMetrics dm = getResources().getDisplayMetrics();
        android.graphics.Path path = new android.graphics.Path();
        float x = dm.widthPixels * 0.4f;
        path.moveTo(x, dm.heightPixels * 0.70f);
        path.lineTo(x, dm.heightPixels * 0.28f);
        boolean ok = dispatchGesture(new android.accessibilityservice.GestureDescription.Builder()
                .addStroke(new android.accessibilityservice.GestureDescription.StrokeDescription(path, 0, 400)).build(),
                new GestureResultCallback() {
                    @Override public void onCompleted(android.accessibilityservice.GestureDescription g) { next.run(); }
                    @Override public void onCancelled(android.accessibilityservice.GestureDescription g) { next.run(); }
                }, main);
        log("auto-continue: swipe dispatched " + ok);
        if (!ok) next.run();
    }

    private void log(String s) { if (SteerTTS.debug(this)) android.util.Log.i(MainActivity.TAG, s); }

    private static final class Item {
        final String text;
        final Rect box;

        Item(String text, Rect box) {
            this.text = text;
            this.box = box;
        }
    }

    /**
     * Paragraphs, the way the page shows them. A web page gives a paragraph as a container of text runs and links
     * (" (also known as ", "Lewis Carroll", ") is an 1865 English "…); a container whose children are all such inline
     * text is one paragraph, its pieces joined in order. Elsewhere a node's own text is a paragraph. Buttons, fields
     * and labels of one or two words are left out.
     */
    private void collect(AccessibilityNodeInfo n, List<Item> out, int depth) {
        if (n == null || depth > 80) return;
        String summary = Summary.of(n);
        if (summary != null) {   // a table, an infobox or a long list: a short summary instead of every cell
            Rect r = new Rect();
            n.getBoundsInScreen(r);
            if (n.isVisibleToUser() && !summary.isEmpty()) out.add(new Item(summary, r));
            return;
        }
        if (n.getChildCount() > 0 && inline(n, 0)) {
            if (toolbar(n)) return;
            StringBuilder b = new StringBuilder();
            Rect box = new Rect();
            int[] links = new int[2];   // characters in links, links
            if (text(n, b, box, 0, links) && n.isVisibleToUser() && !navigation(b.toString(), links)) add(b.toString(), box, out);
            return;
        }
        CharSequence t = n.getText();
        if (n.getChildCount() == 0 && t != null && n.isVisibleToUser() && !control(n) && !image(n)) {
            Rect r = new Rect();
            n.getBoundsInScreen(r);
            add(t.toString(), r, out);
        }
        for (int i = 0; i < n.getChildCount(); ++i) collect(n.getChild(i), out, depth + 1);
    }

    /** Mostly links and no sentence of its own: a navigation box or a footer ("Peter Rabbit Fictional character The…"). */
    private static boolean navigation(String text, int[] links) {
        String s = text.trim();
        return links[1] >= 4 && links[0] >= s.replaceAll("\\s", "").length() * 0.75f;
    }

    // licences, copyright and site boilerplate, which nobody wants read aloud
    private static final java.util.regex.Pattern LEGAL = java.util.regex.Pattern.compile(
            "(?i)(CC BY-SA|Creative Commons|all rights reserved|©|terms of (use|service)|privacy policy|cookie (policy|settings)|"
                    + "rendered with Parsoid|content is available under|this page was last edited)");

    private static void add(String text, Rect box, List<Item> out) {
        String s = text.replaceAll("\\s+", " ").trim();
        if (s.length() < 2 || box.height() <= 0) return;
        if (LEGAL.matcher(s).find() && s.length() < 400) return;
        boolean words = s.split(" ").length >= 3 || s.matches(".*[.!?:]$");   // a label of one or two words is chrome
        if (words) out.add(new Item(s, box));
    }

    private static boolean control(AccessibilityNodeInfo n) {
        CharSequence t = n.getText();
        return n.isEditable() || n.isCheckable() || (n.isClickable() && (t == null || t.length() < 40) && n.getChildCount() == 0
                && !"android.widget.TextView".contentEquals(n.getClassName() == null ? "" : n.getClassName()));
    }

    /** Only text inside: text leaves, or small wrappers of them (a link), no pictures, no fields. */
    private static boolean inline(AccessibilityNodeInfo n, int depth) {
        if (depth > 3 || n.isEditable() || image(n)) return false;
        if (n.getChildCount() == 0) return n.getText() != null && n.getText().length() > 0;
        int texts = 0;
        for (int i = 0; i < n.getChildCount(); ++i) {
            AccessibilityNodeInfo c = n.getChild(i);
            if (c == null) continue;
            if (!inline(c, depth + 1)) return false;
            texts++;
        }
        return texts > 0;
    }

    private static boolean image(AccessibilityNodeInfo n) {
        CharSequence c = n.getClassName();
        return c != null && (c.toString().endsWith(".Image") || c.toString().endsWith("ImageView") || c.toString().endsWith("ImageButton"));
    }

    /** A row of short labels that are all buttons or links (a toolbar: "Language Save Edit Download PDF"). */
    private static boolean toolbar(AccessibilityNodeInfo n) {
        int leaves = 0, labels = 0;
        List<AccessibilityNodeInfo> stack = new ArrayList<>();
        stack.add(n);
        while (!stack.isEmpty()) {
            AccessibilityNodeInfo x = stack.remove(stack.size() - 1);
            if (x == null) continue;
            if (x.getChildCount() == 0 && x.getText() != null) {
                leaves++;
                boolean link = x.isClickable() || (x.getParent() != null && x.getParent().isClickable());
                if (link && x.getText().length() < 24) labels++;
            }
            for (int i = 0; i < x.getChildCount(); ++i) stack.add(x.getChild(i));
        }
        return leaves >= 2 && labels == leaves;
    }

    /** The text leaves of an inline container, in order, and the box around the visible ones. */
    private static boolean text(AccessibilityNodeInfo n, StringBuilder b, Rect box, int depth, int[] links) {
        if (n.getChildCount() == 0) {
            CharSequence t = n.getText();
            if (t == null) return false;
            String s = t.toString();
            if (n.isClickable() || (n.getParent() != null && n.getParent().isClickable())) {
                links[0] += s.trim().length();
                links[1]++;
            }
            if (b.length() > 0 && !Character.isWhitespace(b.charAt(b.length() - 1)) && !s.isEmpty()
                    && Character.isLetterOrDigit(s.charAt(0)) && Character.isLetterOrDigit(b.charAt(b.length() - 1))) b.append(' ');
            b.append(s);
            Rect r = new Rect();
            n.getBoundsInScreen(r);
            if (r.height() > 0 && r.width() > 0) {
                if (box.isEmpty()) box.set(r);
                else box.union(r);
            }
            return true;
        }
        boolean any = false;
        for (int i = 0; i < n.getChildCount(); ++i) any |= text(n.getChild(i), b, box, depth + 1, links);
        return any;
    }

    private static boolean scrolls(AccessibilityNodeInfo x) {
        if (x.isScrollable()) return true;
        for (AccessibilityNodeInfo.AccessibilityAction a : x.getActionList()) {
            if (a.getId() == AccessibilityNodeInfo.ACTION_SCROLL_FORWARD) return true;
            if (Build.VERSION.SDK_INT >= 23 && a.getId() == AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_DOWN.getId()) return true;
        }
        return false;
    }

    /** The biggest scrollable view on screen: the page. */
    private static AccessibilityNodeInfo scrollable(AccessibilityNodeInfo n) {
        AccessibilityNodeInfo best = null;
        long bestArea = 0;
        List<AccessibilityNodeInfo> stack = new ArrayList<>();
        stack.add(n);
        while (!stack.isEmpty()) {
            AccessibilityNodeInfo x = stack.remove(stack.size() - 1);
            if (x == null) continue;
            if (scrolls(x) && x.isVisibleToUser()) {
                Rect r = new Rect();
                x.getBoundsInScreen(r);
                long a = (long) r.width() * r.height();
                if (a > bestArea) {
                    bestArea = a;
                    best = x;
                }
            }
            for (int i = 0; i < x.getChildCount(); ++i) stack.add(x.getChild(i));
        }
        return best;
    }

    private void say(String s) { Announcer.get(this).say(s, "screen", true, null); }

    void stop() {
        Narrator n = Narrator.get(this);
        n.removeListener(this);
        if (reading != null && n.doc() != null && n.doc().id.equals(reading)) n.close();
        reading = null;
        hideControls();
    }

    @Override public void changed() {
        Narrator n = Narrator.get(this);
        if (reading == null) return;
        if (n.doc() == null || !n.doc().id.equals(reading)) {   // something else is playing now
            reading = null;
            n.removeListener(this);
            hideControls();
            return;
        }
        if (playBtn != null) {
            boolean on = n.playing() || n.suspended();
            playBtn.setText(on ? "❚❚" : "▶");
            playBtn.setContentDescription(on ? "Pause" : "Play");
        }
        int pos = n.position();
        int para = pos < n.doc().paraOf.length ? n.doc().paraOf[pos] : -1;
        if (para != lastPara && outline != null) {
            lastPara = para;
            outline.show(para >= 0 && para < boxes.size() ? boxes.get(para) : null);
            follow(para);
        }
    }

    private void skipParagraph(int dir) {
        Narrator n = Narrator.get(this);
        Doc d = n.doc();
        if (d == null || reading == null || !d.id.equals(reading)) return;
        int p = n.position() < d.paraOf.length ? d.paraOf[n.position()] : 0;
        int target = Math.max(0, p + dir);
        for (int i = 0; i < d.paraOf.length; ++i)
            if (d.paraOf[i] == target) {
                n.seek(i);
                if (!n.playing()) n.play();
                return;
            }
        if (dir > 0) ended();   // past the last paragraph on screen: on to the next part
    }

    // ---------------------------------------------------------------- the floating control and the outline
    private void showControls() {
        if (controls != null) return;
        Ui.init(this);
        Ui u = new Ui(this);
        outline = new Outline(this);
        WindowManager.LayoutParams full = new WindowManager.LayoutParams(WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE | WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                PixelFormat.TRANSLUCENT);
        if (Build.VERSION.SDK_INT >= 28) full.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
        wm.addView(outline, full);

        LinearLayout bar = u.row();
        bar.setPadding(u.dp(6), u.dp(4), u.dp(6), u.dp(4));
        bar.setBackground(Ui.round(0xEE13201C, u.dp(26)));
        bar.setElevation(u.dp(6));
        playBtn = button(u, "❚❚", "Pause", v -> Narrator.get(this).toggle());
        TextView back = button(u, "⏮", "Previous paragraph", v -> skipParagraph(-1));
        TextView next = button(u, "⏭", "Next paragraph", v -> skipParagraph(1));
        autoBtn = button(u, "⟳", "", v -> {
            auto = !auto;
            showAuto();
        });
        TextView speed = button(u, "", "", null);
        speed.setTextSize(14);
        showSpeed(speed);
        speed.setOnClickListener(v -> {   // a tap: the next speed
            float[] sp = ReaderActivity.SPEEDS;
            float p = Narrator.get(this).pace();
            int k = 0;
            for (int i = 0; i < sp.length; ++i) if (Math.abs(sp[i] - p) < 0.01f) k = i;
            Narrator.get(this).setPace(sp[(k + 1) % sp.length]);
            showSpeed(speed);
        });
        TextView close = button(u, "✕", "Stop reading", v -> stop());
        bar.addView(back);
        bar.addView(playBtn);
        bar.addView(next);
        bar.addView(autoBtn);
        bar.addView(speed);
        bar.addView(close);
        showAuto();
        WindowManager.LayoutParams lp = new WindowManager.LayoutParams(WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS, PixelFormat.TRANSLUCENT);
        lp.gravity = Gravity.BOTTOM | Gravity.END;
        lp.x = u.dp(16);
        lp.y = u.dp(120);
        draggable(bar, lp);
        controls = bar;
        wm.addView(controls, lp);
    }

    private void showSpeed(TextView b) {
        float p = Narrator.get(this).pace();
        String t = String.valueOf(p).replaceAll("\\.?0+$", "") + "×";
        b.setText(t);
        b.setContentDescription("Speed " + t + ". Tap for the next speed.");
    }

    private void showAuto() {
        if (autoBtn == null) return;
        autoBtn.setAlpha(auto ? 1f : 0.45f);
        autoBtn.setContentDescription(auto ? "Auto-continue on: scrolls on and keeps reading" : "Auto-continue off");
    }

    private TextView button(Ui u, String glyph, String spoken, View.OnClickListener l) {
        TextView b = u.text(glyph, 18, 0xFFFFFFFF, true);
        b.setGravity(Gravity.CENTER);
        b.setMinWidth(u.dp(44));
        b.setMinHeight(u.dp(44));
        b.setContentDescription(spoken);
        b.setOnClickListener(l);
        return b;
    }

    /** The bar can be moved out of the way with a finger. */
    private void draggable(View bar, WindowManager.LayoutParams lp) {
        final float[] down = new float[4];
        bar.setOnTouchListener((v, e) -> {
            switch (e.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    down[0] = e.getRawX();
                    down[1] = e.getRawY();
                    down[2] = lp.x;
                    down[3] = lp.y;
                    return false;
                case MotionEvent.ACTION_MOVE:
                    lp.x = (int) (down[2] - (e.getRawX() - down[0]));
                    lp.y = (int) (down[3] - (e.getRawY() - down[1]));
                    wm.updateViewLayout(bar, lp);
                    return Math.abs(e.getRawX() - down[0]) + Math.abs(e.getRawY() - down[1]) > 20;
                default:
                    return false;
            }
        });
    }

    private void hideControls() {
        try {
            if (controls != null) wm.removeView(controls);
            if (outline != null) wm.removeView(outline);
        } catch (RuntimeException ignored) {
        }
        controls = null;
        outline = null;
        playBtn = autoBtn = null;
    }

    private String label(String pkg) {
        try {
            return getPackageManager().getApplicationLabel(getPackageManager().getApplicationInfo(pkg, 0)).toString();
        } catch (Exception e) {
            return "This screen";
        }
    }

    /** The outline around the paragraph being read (the window passes every touch through). */
    private static final class Outline extends View {
        private final Paint stroke = new Paint(Paint.ANTI_ALIAS_FLAG), fill = new Paint(Paint.ANTI_ALIAS_FLAG);
        private Rect box;

        Outline(Context c) {
            super(c);
            float d = c.getResources().getDisplayMetrics().density;
            stroke.setStyle(Paint.Style.STROKE);
            stroke.setStrokeWidth(2.5f * d);
            stroke.setColor(0xFF1A6A55);
            fill.setColor(0x221A6A55);
        }

        void show(Rect r) {
            box = r;
            invalidate();
        }

        @Override protected void onDraw(Canvas c) {
            if (box == null) return;
            int[] at = new int[2];
            getLocationOnScreen(at);
            float d = getResources().getDisplayMetrics().density, pad = 4 * d;
            RectF r = new RectF(box.left - at[0] - pad, box.top - at[1] - pad, box.right - at[0] + pad, box.bottom - at[1] + pad);
            c.drawRoundRect(r, 10 * d, 10 * d, fill);
            c.drawRoundRect(r, 10 * d, 10 * d, stroke);
        }
    }
}
