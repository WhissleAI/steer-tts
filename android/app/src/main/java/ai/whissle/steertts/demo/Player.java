package ai.whissle.steertts.demo;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;

import java.util.concurrent.LinkedBlockingQueue;

/**
 * Plays float chunks on its own thread, so rendering never waits for playback. end() marks the end of an utterance:
 * the track is drained with stop(), which plays everything written (a streaming AudioTrack otherwise holds a partly
 * filled buffer back until more audio arrives, and the last word would play only with the next line).
 */
final class Player {
    private static final float[] END = new float[0];
    private final LinkedBlockingQueue<float[]> q = new LinkedBlockingQueue<>();
    private final AudioTrack track;
    private final Thread thread;

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

    void write(float[] audio) { q.add(audio); }

    void end() { q.add(END); }

    /** Drops queued audio and silences the track now. */
    void flush() {
        q.clear();
        track.pause();
        track.flush();
    }

    void release() {
        thread.interrupt();
        track.release();
    }

    private void run() {
        try {
            while (!Thread.interrupted()) {
                float[] a = q.take();
                if (a == END) {
                    track.stop();   // plays out what was written, then stops
                    continue;
                }
                if (track.getPlayState() != AudioTrack.PLAYSTATE_PLAYING) track.play();
                track.write(a, 0, a.length, AudioTrack.WRITE_BLOCKING);
            }
        } catch (InterruptedException ignored) {
        } catch (IllegalStateException ignored) {   // released
        }
    }
}
