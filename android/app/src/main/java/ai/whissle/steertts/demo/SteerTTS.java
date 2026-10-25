package ai.whissle.steertts.demo;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** SteerTTS edge engine (the C library through JNI). Use one instance from one thread at a time; close() it. */
public final class SteerTTS implements AutoCloseable {
    static { System.loadLibrary("steertts_jni"); }

    /** Receives each rendered sentence (24 kHz mono float). Return false to stop the stream. */
    public interface Sink {
        boolean onChunk(float[] audio, int index, double synthMs, String text);
    }

    public static final int SAMPLE_RATE = 24000;

    /** Debug builds only: lines with private content (messages, names, titles) never reach a store build's log. */
    static boolean debug(Context c) {
        return (c.getApplicationInfo().flags & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0;
    }
    private long h;

    /** The 12 prosody controls, in the order Style uses (z-scores, about -3 .. 3 in practice). */
    public static final String[] CONTROLS = {"rate", "loud", "f0_med", "f0_iqr", "f0_end", "pause", "tilt", "voiced",
            "flat", "centroid", "cpps", "h1h2"};

    /**
     * threads <= 0: one per fast core, up to 6. Before every request the engine thread is pinned to the fast cores the
     * process may use at that moment (Android gives background processes fewer, slower ones) and the worker pool is
     * resized to match, so the engine works in the foreground and in a background service alike.
     */
    public SteerTTS(String modelDir, int threads) { h = nativeCreate(modelDir, threads); }

    /** Worker threads in use. */
    public int threads() { return nativeThreads(h); }

    /**
     * Streams text sentence by sentence (inline markup parsed: *word*, [pause 300ms], <whisper>..</whisper>). voice,
     * tag and ctrl ("rate=1,loud=-0.5") may be null.
     */
    public void stream(String text, String voice, String tag, String ctrl, float pace, Sink sink) {
        nativeStream(h, text, voice, tag, ctrl, pace, sink);
    }

    /** One render of plain text (no markup); null if cancel() was called before it started. */
    public float[] synthesize(String text, String voice, String ctrl, float pace) {
        return nativeSynthesize(h, text, voice, null, ctrl, pace, false);
    }

    /** One render with a style tag (may be null); markup: parse *word* and [pause 300ms] in text. */
    public float[] synthesize(String text, String voice, String tag, String ctrl, float pace, boolean markup) {
        return nativeSynthesize(h, text, voice, tag, ctrl, pace, markup);
    }

    /** Forget an earlier cancel(): call at the start of a new request (a cancel during the last one stays set otherwise). */
    public void resetCancel() { nativeResetCancel(h); }

    /** Stops a running stream() from another thread (it returns after the current sentence). */
    public void cancel() { nativeCancel(h); }

    public String[] voices(String gender) { return nativeVoices(h, gender); }

    /** Style tags ("none", "happiness", "whisper", ...). */
    public String[] tags() { return nativeTags(h); }

    /** "female", "male" or null. */
    public String voiceGender(String voice) { return nativeVoiceGender(h, voice); }

    /** The default voice of "female" / "male". */
    public String genderVoice(String gender) { return nativeGenderVoice(h, gender); }

    @Override public void close() {
        if (h != 0) nativeDestroy(h);
        h = 0;
    }

    /**
     * Copies the bundled model (assets/model) to app storage once per app version; returns its directory. The app and
     * its TTS service (another process) may both call this at first start: a file lock makes one copy, the other waits.
     */
    public static File installModel(Context ctx) throws IOException {
        // a file lock serializes processes; threads of one process must be serialized first (a second lock on the same
        // file from this process throws OverlappingFileLockException instead of waiting)
        synchronized (SteerTTS.class) {
            if (installed != null) return installed;
            installed = installLocked(ctx);
            return installed;
        }
    }

    private static File installed;   // this process has checked (and if needed copied) the model

    private static File installLocked(Context ctx) throws IOException {
        try (java.io.RandomAccessFile lf = new java.io.RandomAccessFile(new File(ctx.getFilesDir(), "model.lock"), "rw");
             java.nio.channels.FileLock lock = lf.getChannel().lock()) {
            return installModelLocked(ctx);
        }
    }

    private static File installModelLocked(Context ctx) throws IOException {
        File dir = new File(ctx.getFilesDir(), "model");
        File stamp = new File(dir, ".installed");
        String version;
        try {
            version = String.valueOf(ctx.getPackageManager().getPackageInfo(ctx.getPackageName(), 0).lastUpdateTime);
        } catch (Exception e) {
            version = "0";
        }
        if (stamp.exists() && version.equals(readAll(stamp))) return dir;
        copyTree(ctx.getAssets(), "model", dir);
        try (OutputStream o = new FileOutputStream(stamp)) { o.write(version.getBytes()); }
        return dir;
    }

    private static void copyTree(AssetManager am, String path, File dst) throws IOException {
        String[] kids = am.list(path);
        if (kids != null && kids.length > 0) {
            if (!dst.isDirectory() && !dst.mkdirs()) throw new IOException("cannot create " + dst);
            for (String k : kids) copyTree(am, path + "/" + k, new File(dst, k));
            return;
        }
        byte[] buf = new byte[1 << 16];
        try (InputStream in = am.open(path); OutputStream out = new FileOutputStream(dst)) {
            for (int n; (n = in.read(buf)) > 0; ) out.write(buf, 0, n);
        }
    }

    private static String readAll(File f) throws IOException {
        try (InputStream in = new java.io.FileInputStream(f)) { return new String(in.readAllBytes()); }
    }

    private static native long nativeCreate(String dir, int threads);
    private static native void nativeDestroy(long h);
    private static native int nativeThreads(long h);
    private static native void nativeStream(long h, String text, String voice, String tag, String ctrl, float pace,
                                            Sink sink);
    private static native float[] nativeSynthesize(long h, String text, String voice, String tag, String ctrl, float pace,
                                                   boolean markup);
    private static native String nativeVoiceGender(long h, String voice);
    private static native String[] nativeTags(long h);
    private static native void nativeCancel(long h);
    private static native void nativeResetCancel(long h);
    private static native String[] nativeVoices(long h, String gender);
    private static native String nativeGenderVoice(long h, String gender);
}
