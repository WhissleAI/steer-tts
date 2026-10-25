package ai.whissle.steertts.demo;

import java.text.Normalizer;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Text as it is said: every number, time, date, amount, unit, code, Roman numeral, link and symbol written out in words
 * before the engine sees it, and letters the engine cannot read made plain (é -> e, ß -> ss; scripts it has no voice
 * for are dropped). The engine's own front end reads bare numbers and a few patterns; what people actually write
 * (a calendar's "Tue 10:00", "CHAPTER IV.", "₹1,299", "1990s", "Zoë", "john@gmail.com", "code 482913") it misreads.
 * Plain Java, unit tested (SpokenTest). Output has no digits, so say(say(x)) == say(x).
 */
final class Spoken {
    private Spoken() {}

    /** Dates like 3/4/2026: day first (most of the world) or month first (US). */
    static boolean dayFirst = !"US".equals(Locale.getDefault().getCountry());

    private static final Pattern MARKUP = Pattern.compile("\\[[^\\]\\n]{1,40}\\]|</?[a-z]+[^<>\\n]{0,40}>");

    /** Text for the engine. markup: leave the engine's [pause 300ms] and <whisper> tags as they are. */
    static String say(String s, boolean markup) {
        if (s == null || s.isEmpty()) return "";
        if (!markup) return say(s);
        StringBuilder b = new StringBuilder();
        Matcher m = MARKUP.matcher(s);
        int at = 0;
        while (m.find()) {
            b.append(say(s.substring(at, m.start()))).append(' ').append(m.group()).append(' ');
            at = m.end();
        }
        b.append(say(s.substring(at)));
        return b.toString().replaceAll(" {2,}", " ").trim();
    }

    static String say(String s) {
        if (s == null || s.isEmpty()) return "";
        s = letters(s);
        s = links(s);
        s = phones(s);
        s = dates(s);
        s = times(s);
        s = amounts(s);
        s = numbers(s);
        s = symbols(s);
        s = s.replaceAll("[ \\t]{2,}", " ").replaceAll(" +([,.;:!?])", "$1");
        s = s.replaceAll("([,;:])(?:\\s*[,;:])+", "$1").replaceAll("^[\\s,;:.]+", "");   // left where a script was dropped
        return s.trim();
    }

    // ---------------------------------------------------------------- languages the voice can't speak
    /**
     * The language of text the voice cannot read (its letters are in a script other than Latin), as a name to say
     * ("Hindi", "Arabic"), or null when there is something readable. "Hi भाई" is readable; "नमस्ते भाई" is Hindi.
     */
    static String unspoken(String s) {
        if (s == null || s.trim().isEmpty() || !say(s).replaceAll("[^\\p{L}\\p{N}]", "").isEmpty()) return null;
        java.util.Map<Character.UnicodeScript, Integer> n = new java.util.EnumMap<>(Character.UnicodeScript.class);
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            i += Character.charCount(cp);
            if (Character.isLetter(cp)) n.merge(Character.UnicodeScript.of(cp), 1, Integer::sum);
        }
        if (n.isEmpty()) return null;   // only emoji or symbols: described elsewhere
        Character.UnicodeScript top = java.util.Collections.max(n.entrySet(), java.util.Map.Entry.comparingByValue()).getKey();
        switch (top) {
            case DEVANAGARI: return "Hindi";
            case BENGALI: return "Bengali";
            case GURMUKHI: return "Punjabi";
            case GUJARATI: return "Gujarati";
            case TAMIL: return "Tamil";
            case TELUGU: return "Telugu";
            case KANNADA: return "Kannada";
            case MALAYALAM: return "Malayalam";
            case ORIYA: return "Odia";
            case ARABIC: return "Arabic";
            case HEBREW: return "Hebrew";
            case CYRILLIC: return "Russian";
            case GREEK: return "Greek";
            case HAN: return "Chinese";
            case HIRAGANA: case KATAKANA: return "Japanese";
            case HANGUL: return "Korean";
            case THAI: return "Thai";
            default: return "another language";
        }
    }

    // ---------------------------------------------------------------- letters
    private static final String[][] BEFORE_FOLD = {
            {"™", ""}, {"®", ""}, {"©", " copyright "}, {"℠", ""}, {"²", "²"}, {"№", " number "}, {"§§", " sections "},
            {"§", " section "}, {"¶", " "}, {"ß", "ss"}, {"ẞ", "SS"}, {"Æ", "Ae"}, {"æ", "ae"}, {"Œ", "Oe"}, {"œ", "oe"},
            {"Ø", "O"}, {"ø", "o"}, {"Đ", "D"}, {"đ", "d"}, {"Ł", "L"}, {"ł", "l"}, {"Þ", "Th"}, {"þ", "th"}, {"Ð", "D"},
            {"ð", "d"}, {"ı", "i"}, {"Ŋ", "Ng"}, {"ŋ", "ng"}, {"ſ", "s"}, {"∶", ":"}, {"꞉", ":"}, {"⁄", "/"}, {"∕", "/"},
            {"−", "-"}, {"‐", "-"}, {"‑", "-"}, {"‒", "–"}, {"―", "—"}, {"′", "'"}, {"″", "\""}, {"‘", "‘"}};
    private static final String[][] GREEK = {
            {"α", "alpha"}, {"β", "beta"}, {"γ", "gamma"}, {"δ", "delta"}, {"ε", "epsilon"}, {"ζ", "zeta"}, {"η", "eta"},
            {"θ", "theta"}, {"ι", "iota"}, {"κ", "kappa"}, {"λ", "lambda"}, {"μ", "mu"}, {"ν", "nu"}, {"ξ", "xi"},
            {"π", "pi"}, {"ρ", "rho"}, {"σ", "sigma"}, {"ς", "sigma"}, {"τ", "tau"}, {"υ", "upsilon"}, {"φ", "phi"},
            {"χ", "chi"}, {"ψ", "psi"}, {"ω", "omega"}, {"Γ", "gamma"}, {"Δ", "delta"}, {"Θ", "theta"}, {"Λ", "lambda"},
            {"Π", "pi"}, {"Σ", "sigma"}, {"Φ", "phi"}, {"Ψ", "psi"}, {"Ω", "omega"}};

    /** Compatibility forms folded (ligatures, full-width, ½), accents removed, digits of any script made 0-9. */
    static String letters(String s) {
        StringBuilder b = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); ) {   // emoji and anything else outside the BMP: the mood carries them
            int cp = s.codePointAt(i);
            i += Character.charCount(cp);
            if (cp > 0xFFFF) b.append(' ');
            else if (Character.getType(cp) == Character.DECIMAL_DIGIT_NUMBER && (cp < '0' || cp > '9'))
                b.append((char) ('0' + Character.getNumericValue(cp)));
            else if (cp == 0x200B || cp == 0x200C || cp == 0x200D || cp == 0xFEFF || cp == 0x00AD || cp == 0xFE0F) { /* invisible */ }
            else if (Character.isSpaceChar(cp) || cp == '\t') b.append(' ');
            else b.append((char) cp);
        }
        s = b.toString();
        for (String[] r : BEFORE_FOLD) s = s.replace(r[0], r[1]);
        s = s.replaceAll("(?<=[0-9a-zA-Z])²", " squared").replaceAll("(?<=[0-9a-zA-Z])³", " cubed");
        s = s.replaceAll("[µμ](?=(m|g|s|l|L)\\b)", "micro");
        s = s.replaceAll("(?<=\\d)\\s?Ω", " ohms");
        for (String[] g : GREEK) s = s.replace(g[0], " " + g[1] + " ");
        s = s.replace("½", " 1/2").replace("¼", " 1/4").replace("¾", " 3/4").replace("⅓", " 1/3").replace("⅔", " 2/3");
        s = Normalizer.normalize(s, Normalizer.Form.NFKD).replaceAll("\\p{M}+", "");
        s = s.replaceAll("[\\p{L}&&[^\\p{IsLatin}]]+", " ");   // a script the voice cannot read
        s = s.replaceAll("(?<=[\\w?!.,;:’'])_+|_+(?=[\\w“‘'\"])", "");   // _emphasis_ in plain-text books
        return s.replace("⁄", "/");
    }

    // ---------------------------------------------------------------- links, emails, handles
    private static final Pattern URL = Pattern.compile("(?i)\\b(?:https?://|www\\.)([a-z0-9.-]+\\.[a-z]{2,})(?::\\d+)?(/[^\\s<>\"]*)?");
    private static final Pattern EMAIL = Pattern.compile("(?i)\\b([a-z0-9._%+-]+)@([a-z0-9-]+(?:\\.[a-z0-9-]+)*\\.[a-z]{2,})\\b");
    private static final Pattern DOMAIN = Pattern.compile("(?i)\\b([a-z0-9-]{2,}(?:\\.[a-z0-9-]{2,})*)\\.(com|org|net|io|ai|co|in|uk|edu|gov|app|dev|me|info|us|ca|de|fr|au)\\b(?![.\\w]*@)");
    private static final Pattern HANDLE = Pattern.compile("(?<![\\w@])@([A-Za-z0-9_]{2,30})\\b");
    private static final Pattern HASHTAG = Pattern.compile("(?<![\\w#&])#([A-Za-z][A-Za-z0-9_]{1,40})\\b");

    static String links(String s) {
        s = sub(s, URL, m -> " " + host(m.group(1)) + " ");
        s = sub(s, EMAIL, m -> " " + m.group(1).replace(".", " dot ").replace("_", " underscore ").replace("-", " dash ")
                .replace("+", " plus ") + " at " + host(m.group(2)) + " ");
        s = sub(s, DOMAIN, m -> host(m.group(0)));
        s = sub(s, HANDLE, m -> "at " + m.group(1).replace("_", " "));
        return sub(s, HASHTAG, m -> "hashtag " + m.group(1).replace("_", " "));
    }

    private static String host(String h) {
        h = h.toLowerCase(Locale.ROOT).replaceFirst("^www\\.", "").replaceFirst("^m\\.", "");
        return h.replace("-", " ").replace(".", " dot ");
    }

    // ---------------------------------------------------------------- phone numbers
    private static final Pattern PHONE = Pattern.compile(
            "(?<![\\w.,/:-])(\\+\\d{1,3}[\\s.-]?)?(?:\\(\\d{2,4}\\)[\\s.-]?|\\d{2,5}[\\s.-])\\d{3,5}(?:[\\s.-]\\d{3,5})?(?![\\w,/:-]|[.,]\\d)");
    private static final Pattern PLUS_PHONE = Pattern.compile("(?<![\\w])\\+\\d{1,3}[\\s.-]?\\d{6,12}\\b");

    static String phones(String s) {
        s = sub(s, PHONE, m -> {
            String all = m.group();
            int digits = all.replaceAll("\\D", "").length();
            if (digits < 9 || digits > 13) return all;
            boolean bare = m.group(1) == null && !all.contains("(");
            String[] parts = all.trim().split("[\\s.-]+");
            // "in 2015 10000 people": two numbers side by side, not a phone number (but "98765 43210" is one)
            if (bare && parts.length < 3 && !all.contains("-") && !(parts.length == 2 && parts[0].length() == 5 && parts[1].length() == 5)) return all;
            return phone(all);
        });
        return sub(s, PLUS_PHONE, m -> phone(m.group()));
    }

    private static String phone(String all) {
        StringBuilder b = new StringBuilder();
        String t = all.trim();
        if (t.startsWith("+")) {
            b.append("plus ");
            t = t.substring(1);
        }
        for (String g : t.split("[\\s.()-]+")) {
            if (g.isEmpty()) continue;
            if (g.length() > 5) for (String part : groups(g)) b.append(digits(part)).append(", ");
            else b.append(digits(g)).append(", ");
        }
        return " " + b.toString().replaceAll(", $", "") + " ";
    }

    /** A long run of digits in groups a listener can follow: 3-3-4, 4-4… */
    private static String[] groups(String d) {
        java.util.List<String> out = new java.util.ArrayList<>();
        int i = 0, n = d.length();
        while (n - i > 4) {
            int take = (n - i == 7 || n - i == 10 || n - i == 6) ? 3 : n - i == 8 ? 4 : 3;
            out.add(d.substring(i, i + take));
            i += take;
        }
        if (i < n) out.add(d.substring(i));
        return out.toArray(new String[0]);
    }

    // ---------------------------------------------------------------- dates
    static final String[] MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August", "September",
            "October", "November", "December"};
    private static final String MON = "(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
            + "Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|JAN|FEB|MAR|APR|MAY|JUNE?|JULY?|AUG|SEPT?|OCT|NOV|DEC)";
    private static final String DAY = "(Mon(?:day)?|Tue(?:s(?:day)?)?|Wed(?:nesday)?|Thu(?:r(?:s(?:day)?)?)?|Fri(?:day)?|Sat(?:urday)?|"
            + "Sun(?:day)?|MON|TUE|WED|THU|FRI|SAT|SUN)";
    private static final String[] DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};
    private static final Pattern ISO = Pattern.compile("\\b(\\d{4})-(\\d{1,2})-(\\d{1,2})(?![\\d-])");
    private static final Pattern NUMERIC = Pattern.compile("(?<![\\d/.])(\\d{1,2})([/.-])(\\d{1,2})\\2(\\d{4}|\\d{2})(?![\\d/.]|\\.\\d)");
    private static final Pattern DAY_MD = Pattern.compile("\\b" + DAY + "\\.?,?\\s+(\\d{1,2})/(\\d{1,2})(?![\\d/])");
    private static final Pattern MONTH_DAY = Pattern.compile("\\b" + MON + "\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?(?![\\d:.]\\d|\\d|:)\\b");
    private static final Pattern DAY_MONTH = Pattern.compile("(?<![\\d.,/])(\\d{1,2})(?:st|nd|rd|th)?\\s+(?:of\\s+)?" + MON + "\\b\\.?");
    private static final Pattern MONTH_YEAR = Pattern.compile("\\b" + MON + "\\.?\\s+(?=\\d{4}\\b)");
    private static final Pattern WEEKDAY = Pattern.compile("\\b" + DAY + "\\b\\.?(?=,?\\s+(?:\\d|" + MON + "|the\\b|at\\b))");

    static String dates(String s) {
        s = sub(s, ISO, m -> {
            int y = Integer.parseInt(m.group(1)), mo = Integer.parseInt(m.group(2)), d = Integer.parseInt(m.group(3));
            return valid(mo, d) ? date(mo, d) + ", " + year(y) : m.group();
        });
        s = sub(s, NUMERIC, m -> {
            int a = Integer.parseInt(m.group(1)), c = Integer.parseInt(m.group(3));
            String ys = m.group(4);
            if (m.group(2).equals(".") && ys.length() == 2) return m.group();   // 1.2.10: a version
            int y = Integer.parseInt(ys);
            if (ys.length() == 2) y += y < 50 ? 2000 : 1900;
            int mo, d;
            if (a > 12) { d = a; mo = c; }
            else if (c > 12) { mo = a; d = c; }
            else if (dayFirst) { d = a; mo = c; }
            else { mo = a; d = c; }
            return valid(mo, d) ? date(mo, d) + ", " + year(y) : m.group();
        });
        s = sub(s, DAY_MD, m -> {
            int a = Integer.parseInt(m.group(2)), c = Integer.parseInt(m.group(3));
            int mo = dayFirst ? c : a, d = dayFirst ? a : c;
            if (mo > 12) { int t = mo; mo = d; d = t; }
            return valid(mo, d) ? weekday(m.group(1)) + ", " + date(mo, d) : m.group();
        });
        s = sub(s, MONTH_DAY, m -> {
            int mo = month(m.group(1)), d = Integer.parseInt(m.group(2));
            return valid(mo, d) ? date(mo, d) : m.group();
        });
        s = sub(s, DAY_MONTH, m -> {
            int mo = month(m.group(2)), d = Integer.parseInt(m.group(1));
            return valid(mo, d) ? "the " + ordinal(d) + " of " + MONTHS[mo - 1] : m.group();
        });
        s = sub(s, MONTH_YEAR, m -> MONTHS[month(m.group(1)) - 1] + " ");
        s = s.replaceAll("(?i)\\b(the) the (?=\\w+ of )", "$1 ");
        return sub(s, WEEKDAY, m -> weekday(m.group(1)));
    }

    private static boolean valid(int mo, int d) { return mo >= 1 && mo <= 12 && d >= 1 && d <= 31; }

    private static String date(int mo, int d) { return MONTHS[mo - 1] + " " + ordinal(d); }

    private static int month(String m) {
        String k = m.substring(0, 3).toLowerCase(Locale.ROOT);
        for (int i = 0; i < 12; ++i) if (MONTHS[i].toLowerCase(Locale.ROOT).startsWith(k)) return i + 1;
        return 1;
    }

    private static String weekday(String w) {
        String k = w.substring(0, 2).toLowerCase(Locale.ROOT);
        for (String d : DAYS) if (d.toLowerCase(Locale.ROOT).startsWith(k)) return d;
        return w;
    }

    // ---------------------------------------------------------------- times
    private static final String AMPM = "(?:\\s?([AaPp])\\.?\\s?[Mm]\\.?(?![A-Za-z]))";
    private static final Pattern CLOCK = Pattern.compile("(?<![\\d:.,])([01]?\\d|2[0-3]):([0-5]\\d)(?::([0-5]\\d))?(?![\\d:])" + AMPM + "?");
    private static final Pattern DOT_CLOCK = Pattern.compile("(?<![\\d:.,])(1[0-2]|0?[1-9])\\.([0-5]\\d)" + AMPM);
    private static final Pattern HOUR_AMPM = Pattern.compile("(?<![\\d:.,])(1[0-2]|0?[1-9])" + AMPM);

    private static final Pattern CLOCK_RANGE = Pattern.compile("(?<![\\d:.,])(1[0-2]|0?[1-9])(?::([0-5]\\d))?\\s?[-–—]\\s?(1[0-2]|0?[1-9])(?::([0-5]\\d))?" + AMPM);
    private static final Pattern DAY_RANGE = Pattern.compile("\\b" + DAY + "\\.?\\s?[-–—]\\s?" + DAY + "\\b\\.?");
    private static final Pattern HM = Pattern.compile("(?<![\\w.])(\\d{1,2})\\s?h\\s?(\\d{1,2})\\s?m(?:in)?\\b");

    static String times(String s) {
        // "10:00 – 11:00 PM", "9-11am": the second time's AM or PM is the first one's too
        s = sub(s, CLOCK_RANGE, m -> {
            int h1 = Integer.parseInt(m.group(1)), m1 = m.group(2) == null ? 0 : Integer.parseInt(m.group(2));
            int h2 = Integer.parseInt(m.group(3)), m2 = m.group(4) == null ? 0 : Integer.parseInt(m.group(4));
            return cardinal(h1) + (m1 == 0 ? "" : " " + minutes(m1)) + " to " + clock(h2, m2, m.group(5));
        });
        s = sub(s, DAY_RANGE, m -> weekday(m.group(1)) + " to " + weekday(m.group(2)));
        s = sub(s, HM, m -> {
            int h = Integer.parseInt(m.group(1)), mi = Integer.parseInt(m.group(2));
            return cardinal(h) + (h == 1 ? " hour " : " hours ") + cardinal(mi) + (mi == 1 ? " minute" : " minutes");
        });
        // ranges: 10:00-11:00, 2-3pm, 10am-2pm
        s = s.replaceAll("(\\d:\\d\\d(?:\\s?[AaPp]\\.?\\s?[Mm]\\.?)?)\\s?[-–—]\\s?(?=\\d)", "$1 to ");
        s = s.replaceAll("(?<![\\d:.])(\\d{1,2})\\s?[-–—]\\s?(?=\\d{1,2}(?:[:.]\\d\\d)?\\s?[AaPp]\\.?\\s?[Mm]\\b)", "$1 to ");
        s = s.replaceAll("([AaPp]\\.?[Mm]\\.?)\\s?[-–—]\\s?(?=\\d)", "$1 to ");
        s = sub(s, CLOCK, m -> {
            int h = Integer.parseInt(m.group(1)), mi = Integer.parseInt(m.group(2));
            String ap = m.group(4);
            if (m.group(3) != null && ap == null) return duration(h, mi, Integer.parseInt(m.group(3)));
            return clock(h, mi, ap);
        });
        s = sub(s, DOT_CLOCK, m -> clock(Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2)), m.group(3)));
        return sub(s, HOUR_AMPM, m -> clock(Integer.parseInt(m.group(1)), 0, m.group(2)));
    }

    /** ap: "a", "p" or null (then a 24-hour time past noon or at midnight says so). */
    static String clock(int h, int mi, String ap) {
        boolean pm;
        if (ap != null) pm = ap.equalsIgnoreCase("p");
        else if (h == 0 || h >= 13) pm = h >= 12;
        else {
            String hw = cardinal(h);
            return mi == 0 ? hw + " o'clock" : hw + " " + minutes(mi);
        }
        int h12 = h % 12 == 0 ? 12 : h % 12;
        return cardinal(h12) + (mi == 0 ? "" : " " + minutes(mi)) + (pm ? " pee em" : " ay em");
    }

    private static String minutes(int mi) { return mi < 10 ? "oh " + cardinal(mi) : cardinal(mi); }

    private static String duration(int h, int m, int s) {
        StringBuilder b = new StringBuilder();
        if (h > 0) b.append(cardinal(h)).append(h == 1 ? " hour" : " hours");
        if (m > 0) b.append(b.length() > 0 ? ", " : "").append(cardinal(m)).append(m == 1 ? " minute" : " minutes");
        if (s > 0 || b.length() == 0) b.append(b.length() > 0 ? " and " : "").append(cardinal(s)).append(s == 1 ? " second" : " seconds");
        return b.toString();
    }

    // ---------------------------------------------------------------- money
    private static final Map<String, String[]> CURRENCY = new HashMap<>();   // symbol -> {one, many, cent, cents}
    static {
        String[] dollar = {"dollar", "dollars", "cent", "cents"};
        for (String k : new String[] {"$", "US$", "USD", "A$", "AU$", "AUD", "C$", "CA$", "CAD", "NZ$", "S$", "SGD", "HK$"})
            CURRENCY.put(k, dollar);
        CURRENCY.put("€", new String[] {"euro", "euros", "cent", "cents"});
        CURRENCY.put("EUR", CURRENCY.get("€"));
        CURRENCY.put("£", new String[] {"pound", "pounds", "penny", "pence"});
        CURRENCY.put("GBP", CURRENCY.get("£"));
        String[] rupee = {"rupee", "rupees", "paisa", "paise"};
        for (String k : new String[] {"₹", "Rs.", "Rs", "INR", "rs.", "Rs.", "₨"}) CURRENCY.put(k, rupee);
        CURRENCY.put("¥", new String[] {"yen", "yen", "", ""});
        CURRENCY.put("JPY", CURRENCY.get("¥"));
        CURRENCY.put("₩", new String[] {"won", "won", "", ""});
        CURRENCY.put("₽", new String[] {"ruble", "rubles", "kopek", "kopeks"});
        CURRENCY.put("₱", new String[] {"peso", "pesos", "centavo", "centavos"});
        CURRENCY.put("₺", new String[] {"lira", "lira", "kurus", "kurus"});
        CURRENCY.put("₦", new String[] {"naira", "naira", "kobo", "kobo"});
        CURRENCY.put("₿", new String[] {"bitcoin", "bitcoin", "", ""});
        CURRENCY.put("¢", new String[] {"cent", "cents", "", ""});
    }
    private static final String AMOUNT = "(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?";
    private static final String SCALE = "(?:\\s?(k|K|m|M|mn|bn|b|B|million|billion|trillion|thousand|lakh|lakhs|crore|crores|cr)\\b\\.?)?";
    private static final Pattern MONEY_BEFORE = Pattern.compile(
            "(US\\$|AU?\\$|CA?\\$|NZ\\$|S\\$|HK\\$|\\$|€|£|₹|₨|¥|₩|₽|₱|₺|₦|₿|\\bRs\\.?|\\b(?:USD|EUR|GBP|INR|JPY|AUD|CAD|SGD)\\b)\\s?"
                    + AMOUNT + SCALE);
    private static final Pattern MONEY_AFTER = Pattern.compile("(?<![\\w.])" + AMOUNT + SCALE + "\\s?(USD|EUR|GBP|INR|JPY|AUD|CAD|SGD|¢|€)(?![\\w])");

    static String amounts(String s) {
        s = sub(s, MONEY_BEFORE, m -> money(m.group(1), m.group(2), m.group(3), m.group(4)));
        return sub(s, MONEY_AFTER, m -> money(m.group(4), m.group(1), m.group(2), m.group(3)));
    }

    private static String money(String sym, String whole, String frac, String scale) {
        String[] c = CURRENCY.get(sym.trim());
        if (c == null) c = CURRENCY.get("$");
        long w = Long.parseLong(whole.replace(",", ""));
        if (scale != null) {
            String sc = scaleWord(scale);
            String n = frac != null ? cardinal(w) + " point " + digits(frac) : cardinal(w);
            return n + " " + sc + " " + c[1];
        }
        if (frac == null || c[2].isEmpty() || frac.length() != 2) {
            String n = frac != null ? cardinal(w) + " point " + digits(frac) : cardinal(w);
            return n + " " + (w == 1 && frac == null ? c[0] : c[1]);
        }
        int cents = Integer.parseInt(frac);
        if (w == 0) return cardinal(cents) + " " + (cents == 1 ? c[2] : c[3]);
        String out = cardinal(w) + " " + (w == 1 ? c[0] : c[1]);
        return cents == 0 ? out : out + " and " + cardinal(cents) + " " + (cents == 1 ? c[2] : c[3]);
    }

    private static String scaleWord(String s) {
        switch (s.toLowerCase(Locale.ROOT)) {
            case "k": case "thousand": return "thousand";
            case "m": case "mn": case "million": return "million";
            case "b": case "bn": case "billion": return "billion";
            case "trillion": return "trillion";
            case "lakh": case "lakhs": return "lakh";
            default: return "crore";
        }
    }

    // ---------------------------------------------------------------- numbers in context
    private static final String[][] UNITS = {   // unit, singular, plural (case-sensitive, after a number)
            {"km/h", "kilometer per hour", "kilometers per hour"}, {"kmph", "kilometer per hour", "kilometers per hour"},
            {"kph", "kilometer per hour", "kilometers per hour"}, {"mph", "mile per hour", "miles per hour"},
            {"km", "kilometer", "kilometers"}, {"kms", "kilometer", "kilometers"}, {"cm", "centimeter", "centimeters"},
            {"mm", "millimeter", "millimeters"}, {"micrometers", "micrometer", "micrometers"}, {"nm", "nanometer", "nanometers"},
            {"kg", "kilogram", "kilograms"}, {"kgs", "kilogram", "kilograms"}, {"mg", "milligram", "milligrams"},
            {"gm", "gram", "grams"}, {"gms", "gram", "grams"}, {"lb", "pound", "pounds"}, {"lbs", "pound", "pounds"},
            {"oz", "ounce", "ounces"}, {"mi", "mile", "miles"}, {"ft", "foot", "feet"}, {"sq ft", "square foot", "square feet"},
            {"yd", "yard", "yards"}, {"ml", "milliliter", "milliliters"}, {"mL", "milliliter", "milliliters"},
            {"L", "liter", "liters"}, {"ltr", "liter", "liters"}, {"hrs", "hour", "hours"}, {"hr", "hour", "hours"},
            {"h", "hour", "hours"}, {"mins", "minute", "minutes"}, {"min", "minute", "minutes"}, {"secs", "second", "seconds"},
            {"sec", "second", "seconds"}, {"ms", "millisecond", "milliseconds"}, {"yrs", "year", "years"},
            {"yr", "year", "years"}, {"mos", "month", "months"}, {"wks", "week", "weeks"}, {"wk", "week", "weeks"},
            {"TB", "terabyte", "terabytes"}, {"GB", "gigabyte", "gigabytes"}, {"MB", "megabyte", "megabytes"},
            {"KB", "kilobyte", "kilobytes"}, {"kB", "kilobyte", "kilobytes"}, {"Gbps", "gigabit per second", "gigabits per second"},
            {"Mbps", "megabit per second", "megabits per second"}, {"GHz", "gigahertz", "gigahertz"},
            {"MHz", "megahertz", "megahertz"}, {"kHz", "kilohertz", "kilohertz"}, {"Hz", "hertz", "hertz"},
            {"kWh", "kilowatt hour", "kilowatt hours"}, {"kW", "kilowatt", "kilowatts"}, {"W", "watt", "watts"},
            {"mAh", "milliamp hour", "milliamp hours"}, {"V", "volt", "volts"}, {"pts", "point", "points"},
            {"px", "pixel", "pixels"}, {"fps", "frame per second", "frames per second"}, {"bpm", "beat per minute", "beats per minute"},
            {"°C", "degree Celsius", "degrees Celsius"}, {"°F", "degree Fahrenheit", "degrees Fahrenheit"},
            {"° C", "degree Celsius", "degrees Celsius"}, {"° F", "degree Fahrenheit", "degrees Fahrenheit"},
            {"°", "degree", "degrees"}, {"%", "percent", "percent"}, {"‰", "per mille", "per mille"},
            {"x", "times", "times"}, {"k", "thousand", "thousand"}, {"K", "thousand", "thousand"},
            {"M", "million", "million"}, {"bn", "billion", "billion"}, {"cr", "crore", "crore"}};
    private static final Pattern UNIT;
    private static final Map<String, String[]> UNIT_WORDS = new HashMap<>();
    static {
        StringBuilder alt = new StringBuilder();
        String[][] sorted = UNITS.clone();
        java.util.Arrays.sort(sorted, (a, b) -> b[0].length() - a[0].length());
        for (String[] u : sorted) {
            UNIT_WORDS.put(u[0], u);
            alt.append(alt.length() > 0 ? "|" : "").append(Pattern.quote(u[0]));
        }
        UNIT = Pattern.compile("(?<![\\w.,])(-?" + "(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)\\s?(" + alt + ")(?![\\w°%])");
    }
    private static final Pattern ORDINAL = Pattern.compile("(?i)(?<![\\w.])(\\d+)(st|nd|rd|th)\\b");
    private static final Pattern DECADE = Pattern.compile("(?<![\\w'’])(1[1-9]|20)(\\d)0'?s\\b");
    private static final Pattern SHORT_DECADE = Pattern.compile("(?<![\\w])['’]?([2-9])0'?s\\b");
    private static final Pattern NUMBER_SIGN = Pattern.compile("(?:#|\\bNo\\.\\s?|\\bno\\.\\s?|\\bNos?\\.\\s?)(\\d+)\\b");
    private static final Pattern CODE = Pattern.compile(
            "(?i)\\b(code|otp|pin|passcode|password|ref(?:erence)?|order|tracking|confirmation|booking|pnr|account|a/c|acct|"
                    + "invoice|ticket|transaction|txn|id|policy|member(?:ship)?|card|ending in|ends in)\\b"
                    + "((?:\\s*(?:[:#.-]|number|no\\.?|num|id|is|was|ending in|ends in|ending))*\\s*)(\\d(?:[\\d-]*\\d)?)\\b");
    private static final Pattern TIMES = Pattern.compile("(\\d)(\\s?)[x×]\\s?(\\d)");
    private static final Pattern MATH = Pattern.compile("(\\d)\\s*([+=÷^<>≤≥≠≈±])\\s*(?=-?\\d)");
    private static final Pattern RANGE = Pattern.compile("(\\d)\\s?[-–—]\\s?(?=\\d)");
    private static final Pattern NEGATIVE = Pattern.compile("(?<![\\w.])-(?=\\d)");
    private static final Pattern FRACTION = Pattern.compile("(?<![\\d/])(?:(\\d+) )?(\\d{1,2})/(\\d{1,2})(?![\\d/])");
    private static final Pattern RATIO = Pattern.compile("(\\d):(?=\\d)");
    private static final Pattern MIXED = Pattern.compile("(?<=[A-Za-z])(?=\\d)|(?<=\\d)(?=[A-Za-z])");
    private static final Pattern NUM = Pattern.compile("(?<![\\w.])(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?(?![\\w])|(?<![\\w\\d])\\.(\\d+)\\b");
    private static final Pattern VERSION = Pattern.compile("(?<![\\w.])[vV]?(\\d+(?:\\.\\d+){2,})(?![\\w])");

    static String numbers(String s) {
        s = roman(s);
        s = s.replaceAll("(?<=\\b\\d{1,3})\\.(?=[A-Z]\\.?\\d)", " ").replaceAll("(?<=\\b[A-Z])\\.(?=\\d)", " ");   // section 1.E.8
        s = s.replaceAll("(?i)\\bversion v(?=\\d)", "version ").replaceAll("(?<![\\w.])v(?=\\d+(?:\\.\\d+)*\\b)", "version ");
        s = s.replaceAll("\\b(?:c|ca|circa)\\.\\s?(?=\\d{3,4}\\b)", "circa ");
        s = sub(s, VERSION, m -> " " + m.group(1).replace(".", " point ") + " ");
        s = sub(s, CODE, m -> {
            if (m.group(3).replace("-", "").length() < 4) return m.group();
            StringBuilder b = new StringBuilder();
            for (String g : m.group(3).split("-")) b.append(b.length() > 0 ? ", " : "").append(spaced(g));
            return m.group(1) + m.group(2).replace("#", " number ").replaceAll("(?i)\\bno\\b\\.?", "number") + b;
        });
        s = sub(s, NUMBER_SIGN, m -> "number " + (m.group(1).length() >= 4 ? spaced(m.group(1)) : m.group(1)));
        s = sub(s, DECADE, m -> plural(year(Integer.parseInt(m.group(1) + m.group(2) + "0"))));
        s = s.replaceAll("(?<=\\s|^|\\()['’](?=\\d\\d\\b(?!['’]))", "");   // ’90, the class of '82
        s = sub(s, SHORT_DECADE, m -> plural(cardinal(Integer.parseInt(m.group(1)) * 10)));
        s = sub(s, ORDINAL, m -> ordinal(Long.parseLong(m.group(1))));
        s = sub(s, TIMES, m -> m.group(1) + (m.group(2).isEmpty() ? " by " : " times ") + m.group(3));
        s = sub(s, UNIT, m -> {
            String n = m.group(1);
            String[] u = UNIT_WORDS.get(m.group(2));
            boolean one = n.equals("1") || n.equals("-1");
            return n + " " + (one ? u[1] : u[2]);
        });
        s = sub(s, FRACTION, m -> {
            int a = Integer.parseInt(m.group(2)), b = Integer.parseInt(m.group(3));
            String whole = m.group(1) != null ? m.group(1) + " and " : "";
            if (m.group(1) == null && !(b <= 10 && a < b)) {   // 3/14: a date (24/7 is not)
                int mo = dayFirst ? b : a, d = dayFirst ? a : b;
                if (valid(mo, d)) return date(mo, d);
            }
            if (b >= 2 && b <= 10 && a < b) {
                String f = a == 1 && b == 2 ? (whole.isEmpty() ? "one half" : "a half")
                        : a == 1 && b == 4 && !whole.isEmpty() ? "a quarter" : cardinal(a) + " " + denominator(b, a != 1);
                return whole + f;
            }
            return (m.group(1) != null ? m.group(1) + " " : "") + a + " " + b;   // 24/7, 9/11
        });
        s = sub(s, MATH, m -> m.group(1) + " " + op(m.group(2)) + " ");
        s = sub(s, RANGE, m -> m.group(1) + " to ");
        s = sub(s, NEGATIVE, m -> "minus ");
        s = sub(s, RATIO, m -> m.group(1) + " ");
        s = s.replaceAll("(?<=[A-Za-z])-(?=\\d)|(?<=\\d)-(?=[A-Za-z])", " ");   // COVID-19, 3-D
        s = MIXED.matcher(s).replaceAll(" ");
        s = s.replaceAll("~\\s?(?=\\d)", "about ");
        String in = s;
        return sub(s, NUM, m -> {
            if (m.group(3) != null) return "point " + digits(m.group(3));
            String w = m.group(1);
            String frac = m.group(2);
            if (frac != null) return number(w) + " point " + digits(frac);
            if (w.contains(",")) return cardinal(Long.parseLong(w.replace(",", "")));
            if (w.length() > 1 && w.startsWith("0")) return digits(w);   // 007, 0123
            if (w.length() >= 7 && !w.matches("\\d+000")) return spaced(w);   // an account or phone number
            int v = w.length() <= 9 ? Integer.parseInt(w) : -1;
            if (w.length() == 4 && v >= 1100 && v <= 2099 && yearAt(in, m.end())) return year(v);
            return cardinal(Long.parseLong(w));
        });
    }

    private static final Set<String> COUNTED = new HashSet<>(java.util.Arrays.asList("people", "men", "women", "children",
            "persons", "feet", "times", "words", "pages", "users", "votes", "points", "students", "copies", "items",
            "employees", "jobs", "deaths", "cases", "homes", "acres", "calories", "steps", "meters", "metres", "miles",
            "dollars", "rupees", "euros", "pounds", "km", "kilometers", "of"));

    /** A four-digit number read as a year ("in 2015", "2015 was"), not a count ("2015 people"). */
    private static boolean yearAt(String s, int end) {
        Matcher w = Pattern.compile("^\\s+([A-Za-z]+)").matcher(s.substring(end, Math.min(s.length(), end + 24)));
        return !w.find() || !COUNTED.contains(w.group(1).toLowerCase(Locale.ROOT));
    }

    private static String op(String o) {
        switch (o) {
            case "+": return "plus";
            case "=": return "equals";
            case "÷": return "divided by";
            case "^": return "to the power of";
            case "<": return "is less than";
            case ">": return "is greater than";
            case "≤": return "is at most";
            case "≥": return "is at least";
            case "≠": return "is not";
            case "≈": return "is about";
            default: return "plus or minus";
        }
    }

    private static String denominator(int b, boolean many) {
        String[] one = {"", "", "half", "third", "quarter", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"};
        String w = one[b];
        return many ? (b == 2 ? "halves" : w + "s") : w;
    }

    private static String number(String w) { return cardinal(Long.parseLong(w.replace(",", ""))); }

    private static String plural(String words) {
        if (words.endsWith("y")) return words.substring(0, words.length() - 1) + "ies";
        if (words.endsWith("x")) return words + "es";
        return words + "s";
    }

    // ---------------------------------------------------------------- Roman numerals
    private static final String ROMAN = "(M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))";
    private static final Pattern ROMAN_AFTER = Pattern.compile("(?i)\\b(chapter|chap\\.|ch\\.|part|book|volume|vol\\.|act|scene|"
            + "section|article|phase|season|episode|level|stage|round|appendix|canto|psalm|stave|lesson|unit|world war|"
            + "series|clause|schedule|plate|figure|table|chapitre)\\s+(?=[ivxlcdm]+\\b)" + ROMAN + "\\b(?!['’])");
    private static final Pattern ROMAN_HEADING = Pattern.compile("^(\\s*)" + "(?=[IVXLC]+\\.)" + "(M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\\.(?=\\s+[\"“‘'A-Z(]|\\s*$)");
    private static final String RULERS = "King|Queen|Pope|Emperor|Empress|Tsar|Czar|Tsarina|Prince|Princess|Duke|Duchess|Sultan|"
            + "Louis|Henry|Charles|Elizabeth|George|Edward|William|Richard|James|John|Paul|Pius|Leo|Benedict|Gregory|Innocent|"
            + "Clement|Urban|Boniface|Napoleon|Peter|Catherine|Mary|Philip|Frederick|Ludwig|Alexander|Nicholas|Ivan|Ramesses|"
            + "Ptolemy|Constantine|Justinian|Otto|Rudolf|Ferdinand|Wilhelm|Gustav|Gustavus|Carl|Harald|Olav|Haakon|"
            + "Christian|Felipe|Juan Carlos|Willem|Rama|Amenhotep|Thutmose|Selim|Mehmed|Suleiman|Murad|Victor Emmanuel|Stephen";
    private static final Pattern REGNAL = Pattern.compile("\\b(" + RULERS + ")\\s+(?=[IVXL]+\\b)" + ROMAN + "\\b(?=['’]s\\b|\\s*(?:$|[,.;:!?)\"”—–-])|\\s+[a-z])");
    private static final Pattern SUFFIX = Pattern.compile("\\b([A-Z][a-z]+\\s+[A-Z][a-z]+),?\\s+(II|III|IV|V|VI)\\b(?=\\s*(?:$|[,.;:!?)]|\\s+[a-z]))");
    private static final Pattern SEQUEL = Pattern.compile("\\b([A-Z][a-z]+)\\s+(?=[IVXL]{2,}\\b)" + ROMAN + "\\b(?!['’])");
    private static final Set<String> NOT_ROMAN = new HashSet<>(java.util.Arrays.asList("CD", "DC", "CV", "MD", "DM", "LCD",
            "CLI", "MIX", "MIDI", "DIV", "XL", "XXL", "XXX", "LI", "MI", "DL", "ML", "CC", "CI", "DI", "LL", "MM", "MC", "VC",
            "IC", "CM", "DIL", "MIL", "CIV", "LCM", "XXXL", "MCC", "MLC", "VL", "IL", "ID", "IM", "DV"));

    static String roman(String s) {
        s = s.replaceAll("\\bWW ?(II|2)\\b", "World War two").replaceAll("\\bWW ?(I|1)\\b", "World War one");
        String whole = s.trim().replaceAll("\\.$", "");
        if (whole.matches("[IVXLC]{1,7}") && romanValue(whole) > 0) return capital(cardinal(romanValue(whole))) + ".";
        String in = s;
        s = sub(s, ROMAN_AFTER, m -> {
            String r = m.group(2);
            int v = romanValue(r.toUpperCase(Locale.ROOT));
            // "Chapter I", "PART I." but not "for the most part I think"
            if (r.equalsIgnoreCase("i") && (Character.isLowerCase(m.group(1).charAt(0)) || followedByWord(in, m.end()))) return m.group();
            return v > 0 ? m.group(1) + " " + cardinal(v) : m.group();
        });
        s = sub(s, ROMAN_HEADING, m -> {
            int v = romanValue(m.group(2));
            return v > 0 ? m.group(1) + capital(cardinal(v)) + "." : m.group();
        });
        String in2 = s;
        s = sub(s, REGNAL, m -> {
            int v = romanValue(m.group(2));
            if (m.group(2).equals("I") && followedByWord(in2, m.end()) && !in2.startsWith(" of ", m.end())) return m.group();   // "Mary I went"
            return v > 0 ? m.group(1) + " the " + ordinal(v) : m.group();
        });
        s = sub(s, SUFFIX, m -> {
            int v = romanValue(m.group(2));
            return v > 0 ? m.group(1) + " the " + ordinal(v) : m.group();
        });
        return sub(s, SEQUEL, m -> {
            String r = m.group(2);
            int v = romanValue(r);
            return v > 1 && !NOT_ROMAN.contains(r) ? m.group(1) + " " + cardinal(v) : m.group();
        });
    }

    /** A lowercase word comes next ("part I think"): the I is a pronoun. */
    private static boolean followedByWord(String s, int end) {
        return s.substring(end).matches("(?s)\\s+[a-z].*");
    }

    static int romanValue(String r) {
        if (r == null || r.isEmpty()) return 0;
        int[] val = new int[128];
        val['I'] = 1; val['V'] = 5; val['X'] = 10; val['L'] = 50; val['C'] = 100; val['D'] = 500; val['M'] = 1000;
        int total = 0;
        for (int i = 0; i < r.length(); ++i) {
            int c = r.charAt(i), v = c < 128 ? val[c] : 0;
            if (v == 0) return 0;
            int next = i + 1 < r.length() && r.charAt(i + 1) < 128 ? val[r.charAt(i + 1)] : 0;
            total += v < next ? -v : v;
        }
        return total;
    }

    // ---------------------------------------------------------------- symbols left over
    static String symbols(String s) {
        s = s.replaceAll("\\bw/o\\b", "without").replaceAll("\\bw/(?=\\s?\\w)", "with ").replaceAll("\\bb/c\\b", "because");
        s = s.replaceAll("(?i)\\ba\\.k\\.a\\.?|\\baka\\b", "also known as");
        s = s.replaceAll("\\bvs\\b\\.?", "versus").replaceAll("\\bapprox\\b\\.?", "approximately");
        s = s.replaceAll("(?<=\\b[A-Z][a-z]{2,20}) (St|Ave|Rd|Blvd|Ln)(\\.?)(?=\\s*(?:$|[,;\\n]))", " $1@@$2");
        s = s.replace("St@@", "Street").replace("Ave@@", "Avenue").replace("Rd@@", "Road").replace("Blvd@@", "Boulevard")
                .replace("Ln@@", "Lane");
        s = s.replaceAll("\\btsps?\\b\\.?", "teaspoon").replaceAll("\\btbsps?\\b\\.?", "tablespoon");
        s = s.replaceAll("\\bMt\\.(?=\\s+[A-Z])", "Mount").replaceAll("\\bFt\\.(?=\\s+[A-Z])", "Fort");
        s = s.replaceAll("\\s*[→⇒➔⟶]\\s*", " to ").replaceAll("\\s*[←⇐]\\s*", " from ");
        s = s.replaceAll("\\s*[•·‣▪●◦|¦]\\s*", ", ").replaceAll("^,\\s*", "");
        s = s.replaceAll("[✓✔☑]", " ").replaceAll("[★☆✗✘✕]", " ");
        s = s.replace("±", " plus or minus ").replace("≈", " about ").replace("≠", " is not ").replace("≤", " at most ")
                .replace("≥", " at least ").replace("×", " times ").replace("÷", " divided by ").replace("°", " degrees ");
        s = s.replaceAll("(?<=\\s|^)@(?=\\s)", "at").replaceAll("(?<=\\w)@(?=\\w)", " at ");
        s = s.replaceAll("(?<=\\s|^)\\+(?=\\s)", "plus").replaceAll("(?<=\\s|^)=(?=\\s)", "equals");
        s = s.replaceAll("(?<=\\s|^)%", "percent");
        s = s.replaceAll("[_\\\\^{}]+", " ").replaceAll("(?<=\\w)/(?=\\w)", " ");
        s = s.replaceAll("(?<=\\s|^)#(?=\\s|$)", "");
        return s;
    }

    // ---------------------------------------------------------------- number words
    private static final String[] ONES = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    private static final String[] TENS = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
    private static final String[] BIG = {"", " thousand", " million", " billion", " trillion", " quadrillion"};

    static String cardinal(long n) {
        if (n < 0) return "minus " + cardinal(-n);
        if (n < 1000) return below1000((int) n);
        StringBuilder b = new StringBuilder();
        int g = 0;
        String out = "";
        while (n > 0 && g < BIG.length) {
            int part = (int) (n % 1000);
            if (part > 0) out = below1000(part) + BIG[g] + (out.isEmpty() ? "" : " " + out);
            n /= 1000;
            g++;
        }
        return b.append(out).toString();
    }

    private static String below1000(int n) {
        if (n < 20) return ONES[n];
        if (n < 100) return TENS[n / 10] + (n % 10 == 0 ? "" : " " + ONES[n % 10]);
        return ONES[n / 100] + " hundred" + (n % 100 == 0 ? "" : " " + below1000(n % 100));
    }

    static String ordinal(long n) {
        String c = cardinal(n);
        int sp = c.lastIndexOf(' ');
        String head = c.substring(0, sp + 1), last = c.substring(sp + 1);
        switch (last) {
            case "one": return head + "first";
            case "two": return head + "second";
            case "three": return head + "third";
            case "five": return head + "fifth";
            case "eight": return head + "eighth";
            case "nine": return head + "ninth";
            case "twelve": return head + "twelfth";
            default: return head + (last.endsWith("y") ? last.substring(0, last.length() - 1) + "ieth" : last + "th");
        }
    }

    static String year(int y) {
        if (y < 1000 || y > 9999) return cardinal(y);
        int hi = y / 100, lo = y % 100;
        if (y >= 2000 && y < 2010) return cardinal(y);
        if (lo == 0) return hi % 10 == 0 ? cardinal(y) : cardinal(hi) + " hundred";
        return cardinal(hi) + " " + (lo < 10 ? "oh " + ONES[lo] : below1000(lo));
    }

    static String digits(String d) {
        StringBuilder b = new StringBuilder();
        for (char c : d.toCharArray()) if (c >= '0' && c <= '9') b.append(b.length() > 0 ? " " : "").append(ONES[c - '0']);
        return b.toString();
    }

    /** Digit by digit, in groups with a short pause between them. */
    private static String spaced(String d) {
        if (d.length() <= 4) return digits(d);
        StringBuilder b = new StringBuilder();
        for (String g : groups(d)) b.append(b.length() > 0 ? ", " : "").append(digits(g));
        return b.toString();
    }

    private static String capital(String s) { return s.isEmpty() ? s : Character.toUpperCase(s.charAt(0)) + s.substring(1); }

    private static String sub(String s, Pattern p, Function<Matcher, String> f) {
        Matcher m = p.matcher(s);
        if (!m.find()) return s;
        StringBuffer b = new StringBuffer();
        do m.appendReplacement(b, Matcher.quoteReplacement(f.apply(m))); while (m.find());
        m.appendTail(b);
        return b.toString();
    }
}
