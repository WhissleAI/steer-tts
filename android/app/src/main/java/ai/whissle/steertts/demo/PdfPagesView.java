package ai.whissle.steertts.demo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.pdf.PdfRenderer;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.ParcelFileDescriptor;
import android.util.LruCache;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AbsListView;
import android.widget.BaseAdapter;
import android.widget.ListView;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * The PDF itself, page under page, with the sentence being read highlighted where it is printed. Pages are rendered
 * on their own thread at the screen's width and a few are kept; tapping a line asks the reader to read from there.
 */
final class PdfPagesView extends ListView {
    interface OnTap { void tapped(int page, float fx, float fy); }

    private final HandlerThread thread = new HandlerThread("steertts-pdf-pages");
    private final Handler worker, main = new Handler(Looper.getMainLooper());
    private final LruCache<Integer, Bitmap> cache = new LruCache<Integer, Bitmap>(5);
    private final List<Integer> pending = new ArrayList<>();
    private final Paint mark = new Paint(Paint.ANTI_ALIAS_FLAG), shade = new Paint();
    private final OnTap onTap;
    private ParcelFileDescriptor fd;
    private PdfRenderer renderer;   // worker thread only
    private float[] ratio = new float[0];
    private List<float[]> boxes = new ArrayList<>();
    private int widthPx;

    PdfPagesView(Context c, File pdf, OnTap onTap) {
        super(c);
        this.onTap = onTap;
        setDivider(null);
        setDividerHeight(0);
        setClipToPadding(false);
        setBackgroundColor(Ui.dark ? 0xFF2A302E : 0xFFE9ECEB);
        mark.setColor((Ui.ACCENT & 0x00FFFFFF) | 0x55000000);
        shade.setColor(0xFFF2F2F2);
        thread.start();
        worker = new Handler(thread.getLooper());
        worker.post(() -> {
            try {
                fd = ParcelFileDescriptor.open(pdf, ParcelFileDescriptor.MODE_READ_ONLY);
                renderer = new PdfRenderer(fd);
                float[] r = new float[renderer.getPageCount()];
                for (int i = 0; i < r.length; ++i)
                    try (PdfRenderer.Page p = renderer.openPage(i)) { r[i] = p.getHeight() / (float) p.getWidth(); }
                main.post(() -> {
                    ratio = r;
                    adapter.notifyDataSetChanged();
                });
            } catch (Exception e) {
                android.util.Log.e(MainActivity.TAG, "pdf pages", e);
            }
        });
        setAdapter(adapter);
    }

    /** The boxes {page, l, t, r, b} (page fractions) of the sentence being read. */
    void highlight(List<float[]> b) {
        boxes = b == null ? new ArrayList<>() : b;
        for (int i = 0; i < getChildCount(); ++i) getChildAt(i).invalidate();
    }

    /** Scroll so the box's page, and the box itself, is in view. */
    void reveal(float[] box) {
        if (box == null || ratio.length == 0) return;
        int page = (int) box[0];
        int pageH = Math.round(width() * ratio[Math.min(page, ratio.length - 1)]);
        int y = Math.round(pageH * box[2]);
        setSelectionFromTop(page, getHeight() / 3 - y);
    }

    void close() {
        worker.post(() -> {
            try {
                if (renderer != null) renderer.close();
                if (fd != null) fd.close();
            } catch (Exception ignored) {
            }
            thread.quitSafely();
        });
    }

    private int width() {
        int w = getWidth() - getPaddingLeft() - getPaddingRight();
        return w > 0 ? w : getResources().getDisplayMetrics().widthPixels;
    }

    private void render(int page, int w) {
        synchronized (pending) {
            if (pending.contains(page)) return;
            pending.add(page);
        }
        worker.post(() -> {
            Bitmap bm = null;
            try {
                if (renderer != null) {
                    try (PdfRenderer.Page p = renderer.openPage(page)) {
                        bm = Bitmap.createBitmap(w, Math.max(1, Math.round(w * p.getHeight() / (float) p.getWidth())), Bitmap.Config.ARGB_8888);
                        bm.eraseColor(Color.WHITE);
                        p.render(bm, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY);
                    }
                }
            } catch (Exception e) {
                android.util.Log.e(MainActivity.TAG, "render page " + page, e);
            }
            Bitmap done = bm;
            main.post(() -> {
                synchronized (pending) { pending.remove((Integer) page); }
                if (done != null) {
                    cache.put(page, done);
                    for (int i = 0; i < getChildCount(); ++i) {
                        View v = getChildAt(i);
                        if (v instanceof PageView && ((PageView) v).page == page) v.invalidate();
                    }
                }
            });
        });
    }

    private final BaseAdapter adapter = new BaseAdapter() {
        @Override public int getCount() { return ratio.length; }
        @Override public Object getItem(int i) { return i; }
        @Override public long getItemId(int i) { return i; }

        @Override public View getView(int i, View v, ViewGroup parent) {
            PageView pv = v instanceof PageView ? (PageView) v : new PageView(getContext());
            pv.page = i;
            pv.setContentDescription("Page " + (i + 1) + " of " + ratio.length);
            pv.requestLayout();
            pv.invalidate();
            return pv;
        }
    };

    private final class PageView extends View {
        int page;
        private float downX, downY;

        PageView(Context c) {
            super(c);
            setLayoutParams(new AbsListView.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        }

        @Override protected void onMeasure(int ws, int hs) {
            int w = MeasureSpec.getSize(ws);
            float r = page < ratio.length ? ratio[page] : 1.3f;
            int gap = Math.round(getResources().getDisplayMetrics().density * 10);
            setMeasuredDimension(w, Math.round(w * r) + gap);
        }

        @Override protected void onDraw(Canvas c) {
            int w = getWidth(), gap = Math.round(getResources().getDisplayMetrics().density * 10), h = getHeight() - gap;
            widthPx = w;
            Bitmap bm = cache.get(page);
            if (bm == null || bm.getWidth() != w) {
                c.drawRect(0, 0, w, h, shade);
                render(page, w);
                if (bm != null) c.drawBitmap(bm, null, new RectF(0, 0, w, h), null);
            } else {
                c.drawBitmap(bm, 0, 0, null);
            }
            float pad = getResources().getDisplayMetrics().density * 3;
            for (float[] b : boxes)
                if ((int) b[0] == page) c.drawRoundRect(b[1] * w - pad, b[2] * h - pad, b[3] * w + pad, b[4] * h + pad, pad * 2, pad * 2, mark);
        }

        @Override public boolean onTouchEvent(MotionEvent e) {
            if (e.getAction() == MotionEvent.ACTION_DOWN) {
                downX = e.getX();
                downY = e.getY();
                return true;
            }
            if (e.getAction() == MotionEvent.ACTION_UP && Math.abs(e.getX() - downX) < 20 && Math.abs(e.getY() - downY) < 20) {
                int gap = Math.round(getResources().getDisplayMetrics().density * 10);
                onTap.tapped(page, e.getX() / getWidth(), e.getY() / (getHeight() - gap));
                performClick();
                return true;
            }
            return false;
        }

        @Override public boolean performClick() { return super.performClick(); }
    }
}
