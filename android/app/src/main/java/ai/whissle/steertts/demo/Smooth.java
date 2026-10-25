package ai.whissle.steertts.demo;

/**
 * "Smoother sound": an optional gentle filter on the voice (24 kHz) for the metallic edge some phones bring out in the
 * edge vocoder's top end: a high shelf of -4 dB from about 4.5 kHz, then a 2nd-order low-pass at 9 kHz. Two biquads,
 * state kept across the chunks of one stream. Off by default; the user compares in Settings (raw, then smoothed).
 */
final class Smooth {
    /** The user's choice (Style.smooth), for the players in this process. */
    static volatile boolean on;

    private final Biquad shelf = Biquad.highShelf(24000, 4500, -4.0, 0.7), low = Biquad.lowPass(24000, 9000, 0.707);

    void process(float[] x) {
        for (int i = 0; i < x.length; ++i) x[i] = (float) low.step(shelf.step(x[i]));
    }

    void reset() {
        shelf.reset();
        low.reset();
    }

    /** RBJ audio-EQ-cookbook biquad, direct form I. */
    static final class Biquad {
        final double b0, b1, b2, a1, a2;
        double x1, x2, y1, y2;

        Biquad(double b0, double b1, double b2, double a0, double a1, double a2) {
            this.b0 = b0 / a0;
            this.b1 = b1 / a0;
            this.b2 = b2 / a0;
            this.a1 = a1 / a0;
            this.a2 = a2 / a0;
        }

        static Biquad lowPass(double fs, double f0, double q) {
            double w = 2 * Math.PI * f0 / fs, c = Math.cos(w), al = Math.sin(w) / (2 * q);
            return new Biquad((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + al, -2 * c, 1 - al);
        }

        static Biquad highShelf(double fs, double f0, double gainDb, double slope) {
            double a = Math.pow(10, gainDb / 40), w = 2 * Math.PI * f0 / fs, c = Math.cos(w), s = Math.sin(w);
            double al = s / 2 * Math.sqrt((a + 1 / a) * (1 / slope - 1) + 2), sa = 2 * Math.sqrt(a) * al;
            return new Biquad(a * ((a + 1) + (a - 1) * c + sa), -2 * a * ((a - 1) + (a + 1) * c), a * ((a + 1) + (a - 1) * c - sa),
                    (a + 1) - (a - 1) * c + sa, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - sa);
        }

        double step(double x) {
            double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = y;
            return y;
        }

        void reset() { x1 = x2 = y1 = y2 = 0; }
    }
}
