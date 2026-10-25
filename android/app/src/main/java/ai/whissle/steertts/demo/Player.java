package ai.whissle.steertts.demo;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Plays float chunks on its own thread, so rendering never waits for playback. end() marks the end of an utterance:
 * the track is drained with stop(), which plays everything written (a streaming AudioTrack otherwise holds a partly
 * filled buffer back until more audio arrives, and the last word would play only with the next line). end(done) also
 * runs done once the utterance has been heard out.
 */
final class Player {
    private static final float[] END = new float[0];
    private final LinkedBlockingQueue<Object> q = new LinkedBlockingQueue<>();   // float[] audio, END, Runnable
    private final AtomicLong pending = new AtomicLong();   // samples queued, not yet handed to the track
    private final AudioTrack track;
    private final Thread thread;
    private long playStartNs, frames, heardAtNs;   // player thread: the current utterance
    private java.io.OutputStream tap;               // player thread: the current utterance's copy, see tapDir

    /**
     * Demo videos (screen recorders cannot capture an assistant's voice): when set, each utterance played is also
     * written to tapDir/<wall-clock ms at its start>.f32 (raw 24 kHz float), to be put back under the recording.
     * Set from files/tap, which only `adb shell run-as` (debug builds) can create.
     */
    static volatile java.io.File tapDir;

    static void initTap(android.content.Context c) {
        java.io.File d = new java.io.File(c.getFilesDir(), "tap");
        if (d.isDirectory()) tapDir = d;
    }

    Player() {
        int min = AudioTrack.getMinBufferSize(SteerTTS.SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_FLOAT);
        track = new AudioTrack.Builder()
                .setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANT)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(new AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                        .setSampleRate(SteerTTS.SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setBufferSizeInBytes(Math.max(min, SteerTTS.SAMPLE_RATE * 4 / 5))   // 200 ms
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build();
        thread = new Thread(this::run, "steertts-player");
        thread.start();
    }

    void write(float[] audio) {
        pending.addAndGet(audio.length);
        q.add(audio);
    }

    void end() { q.add(END); }

    /** end(), then done on the player thread once everything written has played (dropped by flush()). */
    void end(Runnable done) {
        q.add(END);
        q.add(done);
    }

    /** Audio written but not yet in the track, in samples: lets a writer stay only a few seconds ahead. */
    long pendingSamples() { return Math.max(0, pending.get()); }

    /** Drops queued audio and silences the track now. */
    void flush() {
        q.clear();
        pending.set(0);
        track.pause();
        track.flush();
    }

    void release() {
        thread.interrupt();
        track.release();
    }

    private void openTap() {
        closeTap();
        java.io.File d = tapDir;
        if (d == null) return;
        try {
            tap = new java.io.BufferedOutputStream(new java.io.FileOutputStream(
                    new java.io.File(d, System.currentTimeMillis() + ".f32")));
        } catch (java.io.IOException ignored) {
        }
    }

    private void writeTap(float[] a) {
        if (tap == null) return;
        java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(a.length * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        b.asFloatBuffer().put(a);
        try {
            tap.write(b.array());
        } catch (java.io.IOException ignored) {
        }
    }

    private void closeTap() {
        if (tap == null) return;
        try {
            tap.close();
        } catch (java.io.IOException ignored) {
        }
        tap = null;
    }

    private void run() {
        try {
            while (!Thread.interrupted()) {
                Object o = q.take();
                if (o instanceof Runnable) {
                    long wait = (heardAtNs - System.nanoTime()) / 1000000;
                    if (wait > 0) Thread.sleep(wait);
                    ((Runnable) o).run();
                    continue;
                }
                float[] a = (float[]) o;
                if (a == END) {
                    closeTap();
                    track.stop();   // plays out what was written, then stops (stop() itself returns at once)
                    heardAtNs = playStartNs + frames * 1000000000L / SteerTTS.SAMPLE_RATE + 250000000L;
                    continue;
                }
                if (track.getPlayState() != AudioTrack.PLAYSTATE_PLAYING) {
                    track.play();
                    playStartNs = System.nanoTime();
                    frames = 0;
                    openTap();
                }
                track.write(a, 0, a.length, AudioTrack.WRITE_BLOCKING);
                frames += a.length;
                writeTap(a);
                pending.addAndGet(-a.length);
            }
        } catch (InterruptedException ignored) {
        } catch (IllegalStateException ignored) {   // released
        }
    }
}
