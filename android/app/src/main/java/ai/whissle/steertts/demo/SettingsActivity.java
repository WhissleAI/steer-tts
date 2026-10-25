package ai.whissle.steertts.demo;

import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.ScrollView;

import java.util.Arrays;
import java.util.Locale;

/** Settings, behind the gear on every tab: SettingsScreen, plus the developer tools' engine (benchmark, system voice). */
public class SettingsActivity extends Activity implements SettingsScreen.Host {
    private final HandlerThread worker = new HandlerThread("steertts-bench");
    private Handler engineThread;
    private volatile int threadsSel = 0;
    private TextToSpeech systemTts;
    private SettingsScreen screen;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        Ui.init(this);
        Ui u = new Ui(this);
        worker.start();
        engineThread = new Handler(worker.getLooper());
        LinearLayout root = u.column();
        root.setBackgroundColor(Ui.BG);
        LinearLayout bar = u.row();
        bar.setPadding(u.dp(6), u.dp(6), u.dp(18), u.dp(6));
        android.widget.ImageView back = u.img(R.drawable.ic_back, Ui.INK, 48);
        int pad = u.dp(12);
        back.setPadding(pad, pad, pad, pad);
        back.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_YES);
        back.setContentDescription("Back");
        back.setOnClickListener(v -> finish());
        bar.addView(back);
        bar.addView(u.text("Settings", 24, Ui.INK, true));
        root.addView(bar);
        screen = new SettingsScreen(this, u, this);
        View content = screen.build();
        root.addView(content, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1));
        setContentView(root);
        if (Build.VERSION.SDK_INT >= 30)
            root.setOnApplyWindowInsetsListener((v, ins) -> {
                android.graphics.Insets bars = ins.getInsets(android.view.WindowInsets.Type.systemBars());
                v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
                return android.view.WindowInsets.CONSUMED;
            });
    }

    @Override protected void onResume() {
        super.onResume();
        screen.refresh();
    }

    @Override protected void onDestroy() {
        if (systemTts != null) systemTts.shutdown();
        worker.quitSafely();
        super.onDestroy();
    }

    @Override public void setThreads(int n) { threadsSel = n; }
    @Override public int threads() { return threadsSel; }
    @Override public void showWelcome() { startActivity(new Intent(this, OnboardingActivity.class)); }

    @Override public void benchmark(java.util.function.Consumer<String> result) {
        engineThread.post(() -> {
            try (SteerTTS tts = new SteerTTS(SteerTTS.installModel(this).getPath(), threadsSel)) {
                result.accept(Bench.run(tts));
            } catch (Exception e) {
                result.accept("Benchmark failed: " + e.getMessage());
            }
        });
    }

    @Override public void testSystemVoice() {
        String text = "This is SteerTTS speaking through Android's text to speech.";
        Runnable go = () -> systemTts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "u" + System.nanoTime());
        if (systemTts != null) go.run();
        else systemTts = new TextToSpeech(this, st -> { if (st == TextToSpeech.SUCCESS) go.run(); }, getPackageName());
    }

    /** Real-time factor over a few sentences, and time to the first audio of a two-sentence reply. */
    static final class Bench {
        private static final String[] TEXTS = {
                "The weather was calm when we left the harbour this morning.",
                "Please remember to bring the documents to the meeting on Thursday.",
                "I never thought a small garden could grow so many tomatoes.",
                "Thanks for calling. How can I help you today?",
                "If you turn left at the bakery, you will see the station.",
                "Could you send me the report before lunch?"};

        static String run(SteerTTS tts) {
            for (int w = 0; w < 2; ++w) render(tts, TEXTS[w]);
            double synth = 0, audio = 0;
            for (String t : TEXTS) {
                long t0 = System.nanoTime();
                audio += render(tts, t);
                synth += ms(t0);
            }
            double[] firsts = new double[5];
            for (int k = 0; k < firsts.length; ++k) {
                double[] f = {-1};
                long t0 = System.nanoTime();
                tts.stream("Hello, how are you? And a second sentence follows.", null, null, null, 1f, (a, i, m, x) -> {
                    f[0] = ms(t0);
                    return false;
                });
                firsts[k] = f[0];
            }
            Arrays.sort(firsts);
            String line = String.format(Locale.US, "%s: first audio %.0f ms, %.0f× real time",
                    tts.threads() == 1 ? "1 thread" : tts.threads() + " threads", firsts[2], audio * 1000 / synth);
            Log.i(MainActivity.TAG, "bench " + line);
            return line;
        }

        private static double render(SteerTTS tts, String text) {
            final long[] n = {0};
            tts.stream(text, null, null, null, 1f, (a, i, m, x) -> {
                n[0] += a.length;
                return true;
            });
            return n[0] / (double) SteerTTS.SAMPLE_RATE;
        }

        private static double ms(long t0) { return (System.nanoTime() - t0) / 1e6; }
    }
}
