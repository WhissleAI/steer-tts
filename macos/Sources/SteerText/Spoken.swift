import Foundation

/// Text as it is said: every number, time, date, amount, unit, code, Roman numeral, link and symbol written out in words
/// before the engine sees it, and letters the engine cannot read made plain (é -> e, ß -> ss; scripts it has no voice
/// for are dropped). The engine's own front end reads bare numbers and a few patterns; what people actually write
/// (a calendar's "Tue 10:00", "CHAPTER IV.", "₹1,299", "1990s", "Zoë", "john@gmail.com", "code 482913") it misreads.
/// Plain Swift ported line for line from the Android app's Spoken.java, unit tested (SpokenTests). Output has no digits,
/// so say(say(x)) == say(x).
public enum Spoken {
    /// Dates like 3/4/2026: day first (most of the world) or month first (US).
    public static var dayFirst: Bool = defaultDayFirst(Locale.current)

    /// The default for a locale: month first only in the US, as the Java app's Locale.getDefault().getCountry().
    public static func defaultDayFirst(_ locale: Locale) -> Bool { locale.region?.identifier != "US" }

    private static let MARKUP = "\\[[^\\]\\n]{1,40}\\]|</?[a-z]+[^<>\\n]{0,40}>"

    /// Text for the engine. markup: leave the engine's [pause 300ms] and <whisper> tags as they are.
    public static func say(_ s: String?, markup: Bool) -> String {
        guard let s = s, !s.isEmpty else { return "" }
        if !markup { return say(s) }
        var b = ""
        var at = 0
        for m in rx(MARKUP).all(s) {
            b += say(s.jsub(at, m.start())) + " " + m.group(0)! + " "
            at = m.end()
        }
        b += say(s.jsub(at))
        return jtrim(b.replaceAll(" {2,}", " "))
    }

    public static func say(_ s: String?) -> String {
        guard var s = s, !s.isEmpty else { return "" }
        s = letters(s)
        s = links(s)
        s = phones(s)
        s = dates(s)
        s = times(s)
        s = amounts(s)
        s = numbers(s)
        s = symbols(s)
        s = s.replaceAll("[ \\t]{2,}", " ").replaceAll(" +([,.;:!?])", "$1")
        s = s.replaceAll("([,;:])(?:\\s*[,;:])+", "$1").replaceAll("^[\\s,;:.]+", "")   // left where a script was dropped
        return jtrim(s)
    }

    // ---------------------------------------------------------------- languages the voice can't speak
    /// Scripts in java.lang.Character.UnicodeScript's order (so a tie goes the same way as Collections.max), with the
    /// name to say for the ones the app names.
    private static let SCRIPTS: [(script: String, name: String?)] = [
        ("Common", nil), ("Latin", nil), ("Greek", "Greek"), ("Cyrillic", "Russian"), ("Armenian", nil),
        ("Hebrew", "Hebrew"), ("Arabic", "Arabic"), ("Syriac", nil), ("Thaana", nil), ("Devanagari", "Hindi"),
        ("Bengali", "Bengali"), ("Gurmukhi", "Punjabi"), ("Gujarati", "Gujarati"), ("Oriya", "Odia"), ("Tamil", "Tamil"),
        ("Telugu", "Telugu"), ("Kannada", "Kannada"), ("Malayalam", "Malayalam"), ("Sinhala", nil), ("Thai", "Thai"),
        ("Lao", nil), ("Tibetan", nil), ("Myanmar", nil), ("Georgian", nil), ("Hangul", "Korean"), ("Ethiopic", nil),
        ("Cherokee", nil), ("Canadian_Aboriginal", nil), ("Ogham", nil), ("Runic", nil), ("Khmer", nil),
        ("Mongolian", nil), ("Hiragana", "Japanese"), ("Katakana", "Japanese"), ("Bopomofo", nil), ("Han", "Chinese"),
        ("Yi", nil), ("Inherited", nil), ("Tagalog", nil), ("Hanunoo", nil), ("Buhid", nil), ("Tagbanwa", nil),
        ("Limbu", nil), ("Tai_Le", nil), ("Linear_B", nil), ("Ugaritic", nil), ("Shavian", nil), ("Osmanya", nil),
        ("Cypriot", nil), ("Braille", nil), ("Buginese", nil), ("Coptic", nil), ("New_Tai_Lue", nil), ("Glagolitic", nil),
        ("Tifinagh", nil), ("Syloti_Nagri", nil), ("Old_Persian", nil), ("Kharoshthi", nil), ("Balinese", nil),
        ("Cuneiform", nil), ("Phoenician", nil), ("Phags_Pa", nil), ("Nko", nil), ("Sundanese", nil), ("Batak", nil),
        ("Lepcha", nil), ("Ol_Chiki", nil), ("Vai", nil), ("Saurashtra", nil), ("Kayah_Li", nil), ("Rejang", nil),
        ("Lycian", nil), ("Carian", nil), ("Lydian", nil), ("Cham", nil), ("Tai_Tham", nil), ("Tai_Viet", nil),
        ("Avestan", nil), ("Egyptian_Hieroglyphs", nil), ("Samaritan", nil), ("Mandaic", nil), ("Lisu", nil),
        ("Bamum", nil), ("Javanese", nil), ("Meetei_Mayek", nil), ("Imperial_Aramaic", nil), ("Old_South_Arabian", nil),
        ("Inscriptional_Parthian", nil), ("Inscriptional_Pahlavi", nil), ("Old_Turkic", nil), ("Brahmi", nil),
        ("Kaithi", nil), ("Meroitic_Hieroglyphs", nil), ("Meroitic_Cursive", nil), ("Sora_Sompeng", nil), ("Chakma", nil),
        ("Sharada", nil), ("Takri", nil), ("Miao", nil)]

    /// The language of text the voice cannot read (its letters are in a script other than Latin), as a name to say
    /// ("Hindi", "Arabic"), or nil when there is something readable. "Hi भाई" is readable; "नमस्ते भाई" is Hindi.
    public static func unspoken(_ s: String?) -> String? {
        guard let s = s, !jtrim(s).isEmpty, say(s).replaceAll("[^\\p{L}\\p{N}]", "").isEmpty else { return nil }
        var n: [Int: Int] = [:]   // index in SCRIPTS (SCRIPTS.count: a script not listed) -> letters
        for u in s.unicodeScalars where isLetter(u) {
            let one = String(u)
            let k = SCRIPTS.firstIndex { rx("\\p{Script=\($0.script)}").whole(one) } ?? SCRIPTS.count
            n[k, default: 0] += 1
        }
        if n.isEmpty { return nil }   // only emoji or symbols: described elsewhere
        // the most letters; on a tie the script first in order, as Collections.max over an EnumMap
        let top = n.max { a, b in a.value != b.value ? a.value < b.value : a.key > b.key }!.key
        return top < SCRIPTS.count ? SCRIPTS[top].name ?? "another language" : "another language"
    }

    // ---------------------------------------------------------------- letters
    private static let BEFORE_FOLD: [(String, String)] = [
        ("™", ""), ("®", ""), ("©", " copyright "), ("℠", ""), ("²", "²"), ("№", " number "), ("§§", " sections "),
        ("§", " section "), ("¶", " "), ("ß", "ss"), ("ẞ", "SS"), ("Æ", "Ae"), ("æ", "ae"), ("Œ", "Oe"), ("œ", "oe"),
        ("Ø", "O"), ("ø", "o"), ("Đ", "D"), ("đ", "d"), ("Ł", "L"), ("ł", "l"), ("Þ", "Th"), ("þ", "th"), ("Ð", "D"),
        ("ð", "d"), ("ı", "i"), ("Ŋ", "Ng"), ("ŋ", "ng"), ("ſ", "s"), ("∶", ":"), ("꞉", ":"), ("⁄", "/"), ("∕", "/"),
        ("−", "-"), ("‐", "-"), ("‑", "-"), ("‒", "–"), ("―", "—"), ("′", "'"), ("″", "\""), ("‘", "‘")]
    private static let GREEK: [(String, String)] = [
        ("α", "alpha"), ("β", "beta"), ("γ", "gamma"), ("δ", "delta"), ("ε", "epsilon"), ("ζ", "zeta"), ("η", "eta"),
        ("θ", "theta"), ("ι", "iota"), ("κ", "kappa"), ("λ", "lambda"), ("μ", "mu"), ("ν", "nu"), ("ξ", "xi"),
        ("π", "pi"), ("ρ", "rho"), ("σ", "sigma"), ("ς", "sigma"), ("τ", "tau"), ("υ", "upsilon"), ("φ", "phi"),
        ("χ", "chi"), ("ψ", "psi"), ("ω", "omega"), ("Γ", "gamma"), ("Δ", "delta"), ("Θ", "theta"), ("Λ", "lambda"),
        ("Π", "pi"), ("Σ", "sigma"), ("Φ", "phi"), ("Ψ", "psi"), ("Ω", "omega")]

    /// Compatibility forms folded (ligatures, full-width, ½), accents removed, digits of any script made 0-9.
    public static func letters(_ input: String) -> String {
        var b = String.UnicodeScalarView()
        for u in input.unicodeScalars {   // emoji and anything else outside the BMP: the mood carries them
            let cp = u.value
            if cp > 0xFFFF { b.append(" ") }
            else if u.properties.generalCategory == .decimalNumber && (cp < 0x30 || cp > 0x39),
                    let v = u.properties.numericValue {
                b.append(Unicode.Scalar(0x30 + UInt32(Int(v)))!)
            }
            else if cp == 0x200B || cp == 0x200C || cp == 0x200D || cp == 0xFEFF || cp == 0x00AD || cp == 0xFE0F { /* invisible */ }
            else if isSpaceChar(u) || cp == 0x09 { b.append(" ") }
            else { b.append(u) }
        }
        var s = String(b)
        for (a, r) in BEFORE_FOLD { s = lit(s, a, r) }
        s = s.replaceAll("(?<=[0-9a-zA-Z])²", " squared").replaceAll("(?<=[0-9a-zA-Z])³", " cubed")
        s = s.replaceAll("[µμ](?=(m|g|s|l|L)\\b)", "micro")
        s = s.replaceAll("(?<=\\d)\\s?Ω", " ohms")
        for (a, r) in GREEK { s = lit(s, a, " " + r + " ") }
        s = lit(lit(lit(lit(lit(s, "½", " 1/2"), "¼", " 1/4"), "¾", " 3/4"), "⅓", " 1/3"), "⅔", " 2/3")
        s = s.decomposedStringWithCompatibilityMapping.replaceAll("\\p{M}+", "")
        s = s.replaceAll("[\\p{L}&&[^\\p{Script=Latin}]]+", " ")   // a script the voice cannot read
        s = s.replaceAll("(?<=[\\w?!.,;:’'])_+|_+(?=[\\w“‘'\"])", "")   // _emphasis_ in plain-text books
        return lit(s, "⁄", "/")
    }

    // ---------------------------------------------------------------- links, emails, handles
    private static let URL = "(?i)\\b(?:https?://|www\\.)([a-z0-9.-]+\\.[a-z]{2,})(?::\\d+)?(/[^\\s<>\"]*)?"
    private static let EMAIL = "(?i)\\b([a-z0-9._%+-]+)@([a-z0-9-]+(?:\\.[a-z0-9-]+)*\\.[a-z]{2,})\\b"
    private static let DOMAIN = "(?i)\\b([a-z0-9-]{2,}(?:\\.[a-z0-9-]{2,})*)\\.(com|org|net|io|ai|co|in|uk|edu|gov|app|dev|me|info|us|ca|de|fr|au)\\b(?![.\\w]*@)"
    private static let HANDLE = "(?<![\\w@])@([A-Za-z0-9_]{2,30})\\b"
    private static let HASHTAG = "(?<![\\w#&])#([A-Za-z][A-Za-z0-9_]{1,40})\\b"

    public static func links(_ s: String) -> String {
        var s = sub(s, URL) { " " + host($0.group(1)!) + " " }
        s = sub(s, EMAIL) { m in
            " " + lit(lit(lit(lit(m.group(1)!, ".", " dot "), "_", " underscore "), "-", " dash "), "+", " plus ")
                + " at " + host(m.group(2)!) + " "
        }
        s = sub(s, DOMAIN) { host($0.group(0)!) }
        s = sub(s, HANDLE) { "at " + lit($0.group(1)!, "_", " ") }
        return sub(s, HASHTAG) { "hashtag " + lit($0.group(1)!, "_", " ") }
    }

    private static func host(_ h: String) -> String {
        let h = h.lowercased().replaceFirst("^www\\.", "").replaceFirst("^m\\.", "")
        return lit(lit(h, "-", " "), ".", " dot ")
    }

    // ---------------------------------------------------------------- phone numbers
    private static let PHONE =
        "(?<![\\w.,/:-])(\\+\\d{1,3}[\\s.-]?)?(?:\\(\\d{2,4}\\)[\\s.-]?|\\d{2,5}[\\s.-])\\d{3,5}(?:[\\s.-]\\d{3,5})?(?![\\w,/:-]|[.,]\\d)"
    private static let PLUS_PHONE = "(?<![\\w])\\+\\d{1,3}[\\s.-]?\\d{6,12}\\b"

    public static func phones(_ s: String) -> String {
        let s = sub(s, PHONE) { m in
            let all = m.group(0)!
            let digits = all.replaceAll("\\D", "").jlength
            if digits < 9 || digits > 13 { return all }
            let bare = m.group(1) == nil && !all.contains("(")
            let parts = jtrim(all).split(regex: "[\\s.-]+")
            // "in 2015 10000 people": two numbers side by side, not a phone number (but "98765 43210" is one)
            if bare && parts.count < 3 && !all.contains("-")
                && !(parts.count == 2 && parts[0].jlength == 5 && parts[1].jlength == 5) { return all }
            return phone(all)
        }
        return sub(s, PLUS_PHONE) { phone($0.group(0)!) }
    }

    private static func phone(_ all: String) -> String {
        var b = ""
        var t = jtrim(all)
        if t.hasPrefix("+") {
            b += "plus "
            t = t.jsub(1)
        }
        for g in t.split(regex: "[\\s.()-]+") {
            if g.isEmpty { continue }
            if g.jlength > 5 { for part in groups(g) { b += digits(part) + ", " } }
            else { b += digits(g) + ", " }
        }
        return " " + b.replaceAll(", $", "") + " "
    }

    /// A long run of digits in groups a listener can follow: 3-3-4, 4-4…
    private static func groups(_ d: String) -> [String] {
        var out: [String] = []
        var i = 0
        let n = d.jlength
        while n - i > 4 {
            let take = (n - i == 7 || n - i == 10 || n - i == 6) ? 3 : n - i == 8 ? 4 : 3
            out.append(d.jsub(i, i + take))
            i += take
        }
        if i < n { out.append(d.jsub(i)) }
        return out
    }

    // ---------------------------------------------------------------- dates
    public static let MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                                "October", "November", "December"]
    private static let MON = "(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
        + "Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|JAN|FEB|MAR|APR|MAY|JUNE?|JULY?|AUG|SEPT?|OCT|NOV|DEC)"
    private static let DAY = "(Mon(?:day)?|Tue(?:s(?:day)?)?|Wed(?:nesday)?|Thu(?:r(?:s(?:day)?)?)?|Fri(?:day)?|Sat(?:urday)?|"
        + "Sun(?:day)?|MON|TUE|WED|THU|FRI|SAT|SUN)"
    private static let DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    private static let ISO = "\\b(\\d{4})-(\\d{1,2})-(\\d{1,2})(?![\\d-])"
    private static let NUMERIC = "(?<![\\d/.])(\\d{1,2})([/.-])(\\d{1,2})\\2(\\d{4}|\\d{2})(?![\\d/.]|\\.\\d)"
    private static let DAY_MD = "\\b" + DAY + "\\.?,?\\s+(\\d{1,2})/(\\d{1,2})(?![\\d/])"
    private static let MONTH_DAY = "\\b" + MON + "\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?(?![\\d:.]\\d|\\d|:)\\b"
    private static let DAY_MONTH = "(?<![\\d.,/])(\\d{1,2})(?:st|nd|rd|th)?\\s+(?:of\\s+)?" + MON + "\\b\\.?"
    private static let MONTH_YEAR = "\\b" + MON + "\\.?\\s+(?=\\d{4}\\b)"
    private static let WEEKDAY = "\\b" + DAY + "\\b\\.?(?=,?\\s+(?:\\d|" + MON + "|the\\b|at\\b))"

    public static func dates(_ s: String) -> String {
        var s = sub(s, ISO) { m in
            let y = pint(m.group(1)!), mo = pint(m.group(2)!), d = pint(m.group(3)!)
            return valid(mo, d) ? date(mo, d) + ", " + year(y) : m.group(0)!
        }
        s = sub(s, NUMERIC) { m in
            let a = pint(m.group(1)!), c = pint(m.group(3)!)
            let ys = m.group(4)!
            if m.group(2)! == "." && ys.jlength == 2 { return m.group(0)! }   // 1.2.10: a version
            var y = pint(ys)
            if ys.jlength == 2 { y += y < 50 ? 2000 : 1900 }
            let mo: Int, d: Int
            if a > 12 { d = a; mo = c }
            else if c > 12 { mo = a; d = c }
            else if dayFirst { d = a; mo = c }
            else { mo = a; d = c }
            return valid(mo, d) ? date(mo, d) + ", " + year(y) : m.group(0)!
        }
        s = sub(s, DAY_MD) { m in
            let a = pint(m.group(2)!), c = pint(m.group(3)!)
            var mo = dayFirst ? c : a, d = dayFirst ? a : c
            if mo > 12 { let t = mo; mo = d; d = t }
            return valid(mo, d) ? weekday(m.group(1)!) + ", " + date(mo, d) : m.group(0)!
        }
        s = sub(s, MONTH_DAY) { m in
            let mo = month(m.group(1)!), d = pint(m.group(2)!)
            return valid(mo, d) ? date(mo, d) : m.group(0)!
        }
        s = sub(s, DAY_MONTH) { m in
            let mo = month(m.group(2)!), d = pint(m.group(1)!)
            return valid(mo, d) ? "the " + ordinal(d) + " of " + MONTHS[mo - 1] : m.group(0)!
        }
        s = sub(s, MONTH_YEAR) { MONTHS[month($0.group(1)!) - 1] + " " }
        s = s.replaceAll("(?i)\\b(the) the (?=\\w+ of )", "$1 ")
        return sub(s, WEEKDAY) { weekday($0.group(1)!) }
    }

    private static func valid(_ mo: Int, _ d: Int) -> Bool { mo >= 1 && mo <= 12 && d >= 1 && d <= 31 }

    private static func date(_ mo: Int, _ d: Int) -> String { MONTHS[mo - 1] + " " + ordinal(d) }

    private static func month(_ m: String) -> Int {
        let k = m.jsub(0, 3).lowercased()
        for i in 0..<12 where MONTHS[i].lowercased().hasPrefix(k) { return i + 1 }
        return 1
    }

    private static func weekday(_ w: String) -> String {
        let k = w.jsub(0, 2).lowercased()
        for d in DAYS where d.lowercased().hasPrefix(k) { return d }
        return w
    }

    // ---------------------------------------------------------------- times
    private static let AMPM = "(?:\\s?([AaPp])\\.?\\s?[Mm]\\.?(?![A-Za-z]))"
    private static let CLOCK = "(?<![\\d:.,])([01]?\\d|2[0-3]):([0-5]\\d)(?::([0-5]\\d))?(?![\\d:])" + AMPM + "?"
    private static let DOT_CLOCK = "(?<![\\d:.,])(1[0-2]|0?[1-9])\\.([0-5]\\d)" + AMPM
    private static let HOUR_AMPM = "(?<![\\d:.,])(1[0-2]|0?[1-9])" + AMPM

    private static let CLOCK_RANGE = "(?<![\\d:.,])(1[0-2]|0?[1-9])(?::([0-5]\\d))?\\s?[-–—]\\s?(1[0-2]|0?[1-9])(?::([0-5]\\d))?" + AMPM
    private static let DAY_RANGE = "\\b" + DAY + "\\.?\\s?[-–—]\\s?" + DAY + "\\b\\.?"
    private static let HM = "(?<![\\w.])(\\d{1,2})\\s?h\\s?(\\d{1,2})\\s?m(?:in)?\\b"

    public static func times(_ s: String) -> String {
        // "10:00 – 11:00 PM", "9-11am": the second time's AM or PM is the first one's too
        var s = sub(s, CLOCK_RANGE) { m in
            let h1 = pint(m.group(1)!), m1 = m.group(2).map(pint) ?? 0
            let h2 = pint(m.group(3)!), m2 = m.group(4).map(pint) ?? 0
            return cardinal(h1) + (m1 == 0 ? "" : " " + minutes(m1)) + " to " + clock(h2, m2, m.group(5))
        }
        s = sub(s, DAY_RANGE) { weekday($0.group(1)!) + " to " + weekday($0.group(2)!) }
        s = sub(s, HM) { m in
            let h = pint(m.group(1)!), mi = pint(m.group(2)!)
            return cardinal(h) + (h == 1 ? " hour " : " hours ") + cardinal(mi) + (mi == 1 ? " minute" : " minutes")
        }
        // ranges: 10:00-11:00, 2-3pm, 10am-2pm
        s = s.replaceAll("(\\d:\\d\\d(?:\\s?[AaPp]\\.?\\s?[Mm]\\.?)?)\\s?[-–—]\\s?(?=\\d)", "$1 to ")
        s = s.replaceAll("(?<![\\d:.])(\\d{1,2})\\s?[-–—]\\s?(?=\\d{1,2}(?:[:.]\\d\\d)?\\s?[AaPp]\\.?\\s?[Mm]\\b)", "$1 to ")
        s = s.replaceAll("([AaPp]\\.?[Mm]\\.?)\\s?[-–—]\\s?(?=\\d)", "$1 to ")
        s = sub(s, CLOCK) { m in
            let h = pint(m.group(1)!), mi = pint(m.group(2)!)
            let ap = m.group(4)
            if let sec = m.group(3), ap == nil { return duration(h, mi, pint(sec)) }
            return clock(h, mi, ap)
        }
        s = sub(s, DOT_CLOCK) { clock(pint($0.group(1)!), pint($0.group(2)!), $0.group(3)) }
        return sub(s, HOUR_AMPM) { clock(pint($0.group(1)!), 0, $0.group(2)) }
    }

    /// ap: "a", "p" or nil (then a 24-hour time past noon or at midnight says so).
    public static func clock(_ h: Int, _ mi: Int, _ ap: String?) -> String {
        let pm: Bool
        if let ap = ap { pm = ap.lowercased() == "p" }
        else if h == 0 || h >= 13 { pm = h >= 12 }
        else {
            let hw = cardinal(h)
            return mi == 0 ? hw + " o'clock" : hw + " " + minutes(mi)
        }
        let h12 = h % 12 == 0 ? 12 : h % 12
        return cardinal(h12) + (mi == 0 ? "" : " " + minutes(mi)) + (pm ? " pee em" : " ay em")
    }

    private static func minutes(_ mi: Int) -> String { mi < 10 ? "oh " + cardinal(mi) : cardinal(mi) }

    private static func duration(_ h: Int, _ m: Int, _ s: Int) -> String {
        var b = ""
        if h > 0 { b += cardinal(h) + (h == 1 ? " hour" : " hours") }
        if m > 0 { b += (b.isEmpty ? "" : ", ") + cardinal(m) + (m == 1 ? " minute" : " minutes") }
        if s > 0 || b.isEmpty { b += (b.isEmpty ? "" : " and ") + cardinal(s) + (s == 1 ? " second" : " seconds") }
        return b
    }

    // ---------------------------------------------------------------- money
    private static let CURRENCY: [String: [String]] = {   // symbol -> {one, many, cent, cents}
        var c: [String: [String]] = [:]
        let dollar = ["dollar", "dollars", "cent", "cents"]
        for k in ["$", "US$", "USD", "A$", "AU$", "AUD", "C$", "CA$", "CAD", "NZ$", "S$", "SGD", "HK$"] { c[k] = dollar }
        c["€"] = ["euro", "euros", "cent", "cents"]
        c["EUR"] = c["€"]
        c["£"] = ["pound", "pounds", "penny", "pence"]
        c["GBP"] = c["£"]
        let rupee = ["rupee", "rupees", "paisa", "paise"]
        for k in ["₹", "Rs.", "Rs", "INR", "rs.", "Rs.", "₨"] { c[k] = rupee }
        c["¥"] = ["yen", "yen", "", ""]
        c["JPY"] = c["¥"]
        c["₩"] = ["won", "won", "", ""]
        c["₽"] = ["ruble", "rubles", "kopek", "kopeks"]
        c["₱"] = ["peso", "pesos", "centavo", "centavos"]
        c["₺"] = ["lira", "lira", "kurus", "kurus"]
        c["₦"] = ["naira", "naira", "kobo", "kobo"]
        c["₿"] = ["bitcoin", "bitcoin", "", ""]
        c["¢"] = ["cent", "cents", "", ""]
        return c
    }()
    private static let AMOUNT = "(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?"
    private static let SCALE = "(?:\\s?(k|K|m|M|mn|bn|b|B|million|billion|trillion|thousand|lakh|lakhs|crore|crores|cr)\\b\\.?)?"
    private static let MONEY_BEFORE =
        "(US\\$|AU?\\$|CA?\\$|NZ\\$|S\\$|HK\\$|\\$|€|£|₹|₨|¥|₩|₽|₱|₺|₦|₿|\\bRs\\.?|\\b(?:USD|EUR|GBP|INR|JPY|AUD|CAD|SGD)\\b)\\s?"
        + AMOUNT + SCALE
    private static let MONEY_AFTER = "(?<![\\w.])" + AMOUNT + SCALE + "\\s?(USD|EUR|GBP|INR|JPY|AUD|CAD|SGD|¢|€)(?![\\w])"

    public static func amounts(_ s: String) -> String {
        let s = sub(s, MONEY_BEFORE) { money($0.group(1)!, $0.group(2)!, $0.group(3), $0.group(4)) }
        return sub(s, MONEY_AFTER) { money($0.group(4)!, $0.group(1)!, $0.group(2), $0.group(3)) }
    }

    private static func money(_ sym: String, _ whole: String, _ frac: String?, _ scale: String?) -> String {
        let c = CURRENCY[jtrim(sym)] ?? CURRENCY["$"]!
        let w = plong(lit(whole, ",", ""))
        if let scale = scale {
            let sc = scaleWord(scale)
            let n = frac != nil ? cardinal(w) + " point " + digits(frac!) : cardinal(w)
            return n + " " + sc + " " + c[1]
        }
        guard let frac = frac, !c[2].isEmpty, frac.jlength == 2 else {
            let n = frac != nil ? cardinal(w) + " point " + digits(frac!) : cardinal(w)
            return n + " " + (w == 1 && frac == nil ? c[0] : c[1])
        }
        let cents = pint(frac)
        if w == 0 { return cardinal(cents) + " " + (cents == 1 ? c[2] : c[3]) }
        let out = cardinal(w) + " " + (w == 1 ? c[0] : c[1])
        return cents == 0 ? out : out + " and " + cardinal(cents) + " " + (cents == 1 ? c[2] : c[3])
    }

    private static func scaleWord(_ s: String) -> String {
        switch s.lowercased() {
        case "k", "thousand": return "thousand"
        case "m", "mn", "million": return "million"
        case "b", "bn", "billion": return "billion"
        case "trillion": return "trillion"
        case "lakh", "lakhs": return "lakh"
        default: return "crore"
        }
    }

    // ---------------------------------------------------------------- numbers in context
    private static let UNITS: [[String]] = [   // unit, singular, plural (case-sensitive, after a number)
        ["km/h", "kilometer per hour", "kilometers per hour"], ["kmph", "kilometer per hour", "kilometers per hour"],
        ["kph", "kilometer per hour", "kilometers per hour"], ["mph", "mile per hour", "miles per hour"],
        ["km", "kilometer", "kilometers"], ["kms", "kilometer", "kilometers"], ["cm", "centimeter", "centimeters"],
        ["mm", "millimeter", "millimeters"], ["micrometers", "micrometer", "micrometers"], ["nm", "nanometer", "nanometers"],
        ["kg", "kilogram", "kilograms"], ["kgs", "kilogram", "kilograms"], ["mg", "milligram", "milligrams"],
        ["gm", "gram", "grams"], ["gms", "gram", "grams"], ["lb", "pound", "pounds"], ["lbs", "pound", "pounds"],
        ["oz", "ounce", "ounces"], ["mi", "mile", "miles"], ["ft", "foot", "feet"], ["sq ft", "square foot", "square feet"],
        ["yd", "yard", "yards"], ["ml", "milliliter", "milliliters"], ["mL", "milliliter", "milliliters"],
        ["L", "liter", "liters"], ["ltr", "liter", "liters"], ["hrs", "hour", "hours"], ["hr", "hour", "hours"],
        ["h", "hour", "hours"], ["mins", "minute", "minutes"], ["min", "minute", "minutes"], ["secs", "second", "seconds"],
        ["sec", "second", "seconds"], ["ms", "millisecond", "milliseconds"], ["yrs", "year", "years"],
        ["yr", "year", "years"], ["mos", "month", "months"], ["wks", "week", "weeks"], ["wk", "week", "weeks"],
        ["TB", "terabyte", "terabytes"], ["GB", "gigabyte", "gigabytes"], ["MB", "megabyte", "megabytes"],
        ["KB", "kilobyte", "kilobytes"], ["kB", "kilobyte", "kilobytes"], ["Gbps", "gigabit per second", "gigabits per second"],
        ["Mbps", "megabit per second", "megabits per second"], ["GHz", "gigahertz", "gigahertz"],
        ["MHz", "megahertz", "megahertz"], ["kHz", "kilohertz", "kilohertz"], ["Hz", "hertz", "hertz"],
        ["kWh", "kilowatt hour", "kilowatt hours"], ["kW", "kilowatt", "kilowatts"], ["W", "watt", "watts"],
        ["mAh", "milliamp hour", "milliamp hours"], ["V", "volt", "volts"], ["pts", "point", "points"],
        ["px", "pixel", "pixels"], ["fps", "frame per second", "frames per second"], ["bpm", "beat per minute", "beats per minute"],
        ["°C", "degree Celsius", "degrees Celsius"], ["°F", "degree Fahrenheit", "degrees Fahrenheit"],
        ["° C", "degree Celsius", "degrees Celsius"], ["° F", "degree Fahrenheit", "degrees Fahrenheit"],
        ["°", "degree", "degrees"], ["%", "percent", "percent"], ["‰", "per mille", "per mille"],
        ["x", "times", "times"], ["k", "thousand", "thousand"], ["K", "thousand", "thousand"],
        ["M", "million", "million"], ["bn", "billion", "billion"], ["cr", "crore", "crore"]]
    private static let UNIT_WORDS: [String: [String]] = {
        var w: [String: [String]] = [:]
        for u in UNITS { w[u[0]] = u }
        return w
    }()
    private static let UNIT: String = {
        // longest first (a stable sort, as Java's Arrays.sort of objects), so "km/h" wins over "km"
        let sorted = UNITS.enumerated().sorted { a, b in
            a.element[0].count != b.element[0].count ? a.element[0].count > b.element[0].count : a.offset < b.offset
        }.map(\.element)
        let alt = sorted.map { NSRegularExpression.escapedPattern(for: $0[0]) }.joined(separator: "|")
        return "(?<![\\w.,])(-?" + "(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)\\s?(" + alt + ")(?![\\w°%])"
    }()
    private static let ORDINAL = "(?i)(?<![\\w.])(\\d+)(st|nd|rd|th)\\b"
    private static let DECADE = "(?<![\\w'’])(1[1-9]|20)(\\d)0'?s\\b"
    private static let SHORT_DECADE = "(?<![\\w])['’]?([2-9])0'?s\\b"
    private static let NUMBER_SIGN = "(?:#|\\bNo\\.\\s?|\\bno\\.\\s?|\\bNos?\\.\\s?)(\\d+)\\b"
    // ICU reads "[:" as the start of a POSIX class, so the colon in the set is escaped (Java: [:#.-])
    private static let CODE =
        "(?i)\\b(code|otp|pin|passcode|password|ref(?:erence)?|order|tracking|confirmation|booking|pnr|account|a/c|acct|"
        + "invoice|ticket|transaction|txn|id|policy|member(?:ship)?|card|ending in|ends in)\\b"
        + "((?:\\s*(?:[\\:#.-]|number|no\\.?|num|id|is|was|ending in|ends in|ending))*\\s*)(\\d(?:[\\d-]*\\d)?)\\b"
    private static let TIMES = "(\\d)(\\s?)[x×]\\s?(\\d)"
    private static let MATH = "(\\d)\\s*([+=÷^<>≤≥≠≈±])\\s*(?=-?\\d)"
    private static let RANGE = "(\\d)\\s?[-–—]\\s?(?=\\d)"
    private static let NEGATIVE = "(?<![\\w.])-(?=\\d)"
    private static let FRACTION = "(?<![\\d/])(?:(\\d+) )?(\\d{1,2})/(\\d{1,2})(?![\\d/])"
    private static let RATIO = "(\\d):(?=\\d)"
    private static let MIXED = "(?<=[A-Za-z])(?=\\d)|(?<=\\d)(?=[A-Za-z])"
    private static let NUM = "(?<![\\w.])(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?(?![\\w])|(?<![\\w\\d])\\.(\\d+)\\b"
    private static let VERSION = "(?<![\\w.])[vV]?(\\d+(?:\\.\\d+){2,})(?![\\w])"

    public static func numbers(_ s: String) -> String {
        var s = roman(s)
        s = s.replaceAll("(?<=\\b\\d{1,3})\\.(?=[A-Z]\\.?\\d)", " ").replaceAll("(?<=\\b[A-Z])\\.(?=\\d)", " ")   // section 1.E.8
        s = s.replaceAll("(?i)\\bversion v(?=\\d)", "version ").replaceAll("(?<![\\w.])v(?=\\d+(?:\\.\\d+)*\\b)", "version ")
        s = s.replaceAll("\\b(?:c|ca|circa)\\.\\s?(?=\\d{3,4}\\b)", "circa ")
        s = sub(s, VERSION) { " " + lit($0.group(1)!, ".", " point ") + " " }
        s = sub(s, CODE) { m in
            let code = m.group(3)!
            if lit(code, "-", "").jlength < 4 { return m.group(0)! }
            var b = ""
            for g in code.split(regex: "-") { b += (b.isEmpty ? "" : ", ") + spaced(g) }
            return m.group(1)! + lit(m.group(2)!, "#", " number ").replaceAll("(?i)\\bno\\b\\.?", "number") + b
        }
        s = sub(s, NUMBER_SIGN) { m in "number " + (m.group(1)!.jlength >= 4 ? spaced(m.group(1)!) : m.group(1)!) }
        s = sub(s, DECADE) { plural(year(pint($0.group(1)! + $0.group(2)! + "0"))) }
        s = s.replaceAll("(?<=\\s|^|\\()['’](?=\\d\\d\\b(?!['’]))", "")   // ’90, the class of '82
        s = sub(s, SHORT_DECADE) { plural(cardinal(pint($0.group(1)!) * 10)) }
        s = sub(s, ORDINAL) { ordinal(plong($0.group(1)!)) }
        s = sub(s, TIMES) { m in m.group(1)! + (m.group(2)!.isEmpty ? " by " : " times ") + m.group(3)! }
        s = sub(s, UNIT) { m in
            let n = m.group(1)!
            let u = UNIT_WORDS[m.group(2)!]!
            let one = n == "1" || n == "-1"
            return n + " " + (one ? u[1] : u[2])
        }
        s = sub(s, FRACTION) { m in
            let a = pint(m.group(2)!), b = pint(m.group(3)!)
            let whole = m.group(1) != nil ? m.group(1)! + " and " : ""
            if m.group(1) == nil && !(b <= 10 && a < b) {   // 3/14: a date (24/7 is not)
                let mo = dayFirst ? b : a, d = dayFirst ? a : b
                if valid(mo, d) { return date(mo, d) }
            }
            if b >= 2 && b <= 10 && a < b {
                let f = a == 1 && b == 2 ? (whole.isEmpty ? "one half" : "a half")
                    : a == 1 && b == 4 && !whole.isEmpty ? "a quarter" : cardinal(a) + " " + denominator(b, a != 1)
                return whole + f
            }
            return (m.group(1) != nil ? m.group(1)! + " " : "") + String(a) + " " + String(b)   // 24/7, 9/11
        }
        s = sub(s, MATH) { m in m.group(1)! + " " + op(m.group(2)!) + " " }
        s = sub(s, RANGE) { $0.group(1)! + " to " }
        s = sub(s, NEGATIVE) { _ in "minus " }
        s = sub(s, RATIO) { $0.group(1)! + " " }
        s = s.replaceAll("(?<=[A-Za-z])-(?=\\d)|(?<=\\d)-(?=[A-Za-z])", " ")   // COVID-19, 3-D
        s = s.replaceAll(MIXED, " ")
        s = s.replaceAll("~\\s?(?=\\d)", "about ")
        let input = s
        return sub(s, NUM) { m in
            if let f = m.group(3) { return "point " + digits(f) }
            let w = m.group(1)!
            if let frac = m.group(2) { return number(w) + " point " + digits(frac) }
            if w.contains(",") { return cardinal(plong(lit(w, ",", ""))) }
            if w.jlength > 1 && w.hasPrefix("0") { return digits(w) }   // 007, 0123
            if w.jlength >= 7 && !w.matches("\\d+000") { return spaced(w) }   // an account or phone number
            let v = w.jlength <= 9 ? pint(w) : -1
            if w.jlength == 4 && v >= 1100 && v <= 2099 && yearAt(input, m.end()) { return year(v) }
            return cardinal(plong(w))
        }
    }

    private static let COUNTED: Set<String> = ["people", "men", "women", "children",
        "persons", "feet", "times", "words", "pages", "users", "votes", "points", "students", "copies", "items",
        "employees", "jobs", "deaths", "cases", "homes", "acres", "calories", "steps", "meters", "metres", "miles",
        "dollars", "rupees", "euros", "pounds", "km", "kilometers", "of"]

    /// A four-digit number read as a year ("in 2015", "2015 was"), not a count ("2015 people").
    private static func yearAt(_ s: String, _ end: Int) -> Bool {
        guard let w = rx("^\\s+([A-Za-z]+)").find(s.jsub(end, min(s.jlength, end + 24))) else { return true }
        return !COUNTED.contains(w.group(1)!.lowercased())
    }

    private static func op(_ o: String) -> String {
        switch o {
        case "+": return "plus"
        case "=": return "equals"
        case "÷": return "divided by"
        case "^": return "to the power of"
        case "<": return "is less than"
        case ">": return "is greater than"
        case "≤": return "is at most"
        case "≥": return "is at least"
        case "≠": return "is not"
        case "≈": return "is about"
        default: return "plus or minus"
        }
    }

    private static func denominator(_ b: Int, _ many: Bool) -> String {
        let one = ["", "", "half", "third", "quarter", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        let w = one[b]
        return many ? (b == 2 ? "halves" : w + "s") : w
    }

    private static func number(_ w: String) -> String { cardinal(plong(lit(w, ",", ""))) }

    private static func plural(_ words: String) -> String {
        if words.hasSuffix("y") { return String(words.dropLast()) + "ies" }
        if words.hasSuffix("x") { return words + "es" }
        return words + "s"
    }

    // ---------------------------------------------------------------- Roman numerals
    private static let ROMAN = "(M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))"
    private static let ROMAN_AFTER = "(?i)\\b(chapter|chap\\.|ch\\.|part|book|volume|vol\\.|act|scene|"
        + "section|article|phase|season|episode|level|stage|round|appendix|canto|psalm|stave|lesson|unit|world war|"
        + "series|clause|schedule|plate|figure|table|chapitre)\\s+(?=[ivxlcdm]+\\b)" + ROMAN + "\\b(?!['’])"
    private static let ROMAN_HEADING = "^(\\s*)" + "(?=[IVXLC]+\\.)" + "(M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\\.(?=\\s+[\"“‘'A-Z(]|\\s*$)"
    private static let RULERS = "King|Queen|Pope|Emperor|Empress|Tsar|Czar|Tsarina|Prince|Princess|Duke|Duchess|Sultan|"
        + "Louis|Henry|Charles|Elizabeth|George|Edward|William|Richard|James|John|Paul|Pius|Leo|Benedict|Gregory|Innocent|"
        + "Clement|Urban|Boniface|Napoleon|Peter|Catherine|Mary|Philip|Frederick|Ludwig|Alexander|Nicholas|Ivan|Ramesses|"
        + "Ptolemy|Constantine|Justinian|Otto|Rudolf|Ferdinand|Wilhelm|Gustav|Gustavus|Carl|Harald|Olav|Haakon|"
        + "Christian|Felipe|Juan Carlos|Willem|Rama|Amenhotep|Thutmose|Selim|Mehmed|Suleiman|Murad|Victor Emmanuel|Stephen"
    private static let REGNAL = "\\b(" + RULERS + ")\\s+(?=[IVXL]+\\b)" + ROMAN + "\\b(?=['’]s\\b|\\s*(?:$|[,.;:!?)\"”—–-])|\\s+[a-z])"
    private static let SUFFIX = "\\b([A-Z][a-z]+\\s+[A-Z][a-z]+),?\\s+(II|III|IV|V|VI)\\b(?=\\s*(?:$|[,.;:!?)]|\\s+[a-z]))"
    private static let SEQUEL = "\\b([A-Z][a-z]+)\\s+(?=[IVXL]{2,}\\b)" + ROMAN + "\\b(?!['’])"
    private static let NOT_ROMAN: Set<String> = ["CD", "DC", "CV", "MD", "DM", "LCD",
        "CLI", "MIX", "MIDI", "DIV", "XL", "XXL", "XXX", "LI", "MI", "DL", "ML", "CC", "CI", "DI", "LL", "MM", "MC", "VC",
        "IC", "CM", "DIL", "MIL", "CIV", "LCM", "XXXL", "MCC", "MLC", "VL", "IL", "ID", "IM", "DV"]

    public static func roman(_ s: String) -> String {
        var s = s.replaceAll("\\bWW ?(II|2)\\b", "World War two").replaceAll("\\bWW ?(I|1)\\b", "World War one")
        let whole = jtrim(s).replaceAll("\\.$", "")
        if whole.matches("[IVXLC]{1,7}") && romanValue(whole) > 0 { return capital(cardinal(romanValue(whole))) + "." }
        let input = s
        s = sub(s, ROMAN_AFTER) { m in
            let r = m.group(2)!
            let v = romanValue(r.uppercased())
            // "Chapter I", "PART I." but not "for the most part I think"
            if r.lowercased() == "i" && (isLowerCase(m.group(1)!.jchar(0)) || followedByWord(input, m.end())) { return m.group(0)! }
            return v > 0 ? m.group(1)! + " " + cardinal(v) : m.group(0)!
        }
        s = sub(s, ROMAN_HEADING) { m in
            let v = romanValue(m.group(2)!)
            return v > 0 ? m.group(1)! + capital(cardinal(v)) + "." : m.group(0)!
        }
        let in2 = s
        s = sub(s, REGNAL) { m in
            let v = romanValue(m.group(2)!)
            if m.group(2)! == "I" && followedByWord(in2, m.end()) && !in2.jsub(m.end()).hasPrefix(" of ") { return m.group(0)! }   // "Mary I went"
            return v > 0 ? m.group(1)! + " the " + ordinal(v) : m.group(0)!
        }
        s = sub(s, SUFFIX) { m in
            let v = romanValue(m.group(2)!)
            return v > 0 ? m.group(1)! + " the " + ordinal(v) : m.group(0)!
        }
        return sub(s, SEQUEL) { m in
            let r = m.group(2)!
            let v = romanValue(r)
            return v > 1 && !NOT_ROMAN.contains(r) ? m.group(1)! + " " + cardinal(v) : m.group(0)!
        }
    }

    /// A lowercase word comes next ("part I think"): the I is a pronoun.
    private static func followedByWord(_ s: String, _ end: Int) -> Bool {
        s.jsub(end).matches("(?s)\\s+[a-z].*")
    }

    public static func romanValue(_ r: String?) -> Int {
        guard let r = r, !r.isEmpty else { return 0 }
        func val(_ c: unichar) -> Int {
            switch c {
            case 0x49: return 1      // I
            case 0x56: return 5      // V
            case 0x58: return 10     // X
            case 0x4C: return 50     // L
            case 0x43: return 100    // C
            case 0x44: return 500    // D
            case 0x4D: return 1000   // M
            default: return 0
            }
        }
        let ns = r as NSString
        var total = 0
        for i in 0..<ns.length {
            let v = val(ns.character(at: i))
            if v == 0 { return 0 }
            let next = i + 1 < ns.length ? val(ns.character(at: i + 1)) : 0
            total += v < next ? -v : v
        }
        return total
    }

    // ---------------------------------------------------------------- symbols left over
    public static func symbols(_ s: String) -> String {
        var s = s.replaceAll("\\bw/o\\b", "without").replaceAll("\\bw/(?=\\s?\\w)", "with ").replaceAll("\\bb/c\\b", "because")
        s = s.replaceAll("(?i)\\ba\\.k\\.a\\.?|\\baka\\b", "also known as")
        s = s.replaceAll("\\bvs\\b\\.?", "versus").replaceAll("\\bapprox\\b\\.?", "approximately")
        s = s.replaceAll("(?<=\\b[A-Z][a-z]{2,20}) (St|Ave|Rd|Blvd|Ln)(\\.?)(?=\\s*(?:$|[,;\\n]))", " $1@@$2")
        s = lit(lit(lit(lit(lit(s, "St@@", "Street"), "Ave@@", "Avenue"), "Rd@@", "Road"), "Blvd@@", "Boulevard"),
                "Ln@@", "Lane")
        s = s.replaceAll("\\btsps?\\b\\.?", "teaspoon").replaceAll("\\btbsps?\\b\\.?", "tablespoon")
        s = s.replaceAll("\\bMt\\.(?=\\s+[A-Z])", "Mount").replaceAll("\\bFt\\.(?=\\s+[A-Z])", "Fort")
        s = s.replaceAll("\\s*[→⇒➔⟶]\\s*", " to ").replaceAll("\\s*[←⇐]\\s*", " from ")
        s = s.replaceAll("\\s*[•·‣▪●◦|¦]\\s*", ", ").replaceAll("^,\\s*", "")
        s = s.replaceAll("[✓✔☑]", " ").replaceAll("[★☆✗✘✕]", " ")
        for (a, r) in [("±", " plus or minus "), ("≈", " about "), ("≠", " is not "), ("≤", " at most "),
                       ("≥", " at least "), ("×", " times "), ("÷", " divided by "), ("°", " degrees ")] { s = lit(s, a, r) }
        s = s.replaceAll("(?<=\\s|^)@(?=\\s)", "at").replaceAll("(?<=\\w)@(?=\\w)", " at ")
        s = s.replaceAll("(?<=\\s|^)\\+(?=\\s)", "plus").replaceAll("(?<=\\s|^)=(?=\\s)", "equals")
        s = s.replaceAll("(?<=\\s|^)%", "percent")
        s = s.replaceAll("[_\\\\^{}]+", " ").replaceAll("(?<=\\w)/(?=\\w)", " ")
        s = s.replaceAll("(?<=\\s|^)#(?=\\s|$)", "")
        return s
    }

    // ---------------------------------------------------------------- number words
    private static let ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                               "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    private static let TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    private static let BIG = ["", " thousand", " million", " billion", " trillion", " quadrillion"]

    public static func cardinal(_ n: Int) -> String {
        if n < 0 { return "minus " + cardinal(n == Int.min ? Int.max : -n) }
        if n < 1000 { return below1000(n) }
        var n = n
        var g = 0
        var out = ""
        while n > 0 && g < BIG.count {
            let part = n % 1000
            if part > 0 { out = below1000(part) + BIG[g] + (out.isEmpty ? "" : " " + out) }
            n /= 1000
            g += 1
        }
        return out
    }

    private static func below1000(_ n: Int) -> String {
        if n < 20 { return ONES[n] }
        if n < 100 { return TENS[n / 10] + (n % 10 == 0 ? "" : " " + ONES[n % 10]) }
        return ONES[n / 100] + " hundred" + (n % 100 == 0 ? "" : " " + below1000(n % 100))
    }

    public static func ordinal(_ n: Int) -> String {
        let c = cardinal(n)
        let sp = (c as NSString).range(of: " ", options: .backwards).location
        let cut = sp == NSNotFound ? 0 : sp + 1
        let head = c.jsub(0, cut), last = c.jsub(cut)
        switch last {
        case "one": return head + "first"
        case "two": return head + "second"
        case "three": return head + "third"
        case "five": return head + "fifth"
        case "eight": return head + "eighth"
        case "nine": return head + "ninth"
        case "twelve": return head + "twelfth"
        default: return head + (last.hasSuffix("y") ? String(last.dropLast()) + "ieth" : last + "th")
        }
    }

    public static func year(_ y: Int) -> String {
        if y < 1000 || y > 9999 { return cardinal(y) }
        let hi = y / 100, lo = y % 100
        if y >= 2000 && y < 2010 { return cardinal(y) }
        if lo == 0 { return hi % 10 == 0 ? cardinal(y) : cardinal(hi) + " hundred" }
        return cardinal(hi) + " " + (lo < 10 ? "oh " + ONES[lo] : below1000(lo))
    }

    public static func digits(_ d: String) -> String {
        var b = ""
        for c in d.utf16 where c >= 0x30 && c <= 0x39 { b += (b.isEmpty ? "" : " ") + ONES[Int(c - 0x30)] }
        return b
    }

    /// Digit by digit, in groups with a short pause between them.
    private static func spaced(_ d: String) -> String {
        if d.jlength <= 4 { return digits(d) }
        var b = ""
        for g in groups(d) { b += (b.isEmpty ? "" : ", ") + digits(g) }
        return b
    }

    private static func capital(_ s: String) -> String { s.isEmpty ? s : s.jsub(0, 1).uppercased() + s.jsub(1) }

    /// Every match of p replaced by f's text, used literally (Java's appendReplacement with quoteReplacement).
    private static func sub(_ s: String, _ p: String, _ f: (JMatch) -> String) -> String { rx(p).replaceAll(s, f) }

    // ---------------------------------------------------------------- Java's library, as Spoken uses it
    /// String.replace(target, replacement): literal UTF-16 matching, no Unicode equivalence.
    private static func lit(_ s: String, _ a: String, _ r: String) -> String {
        s.replacingOccurrences(of: a, with: r, options: .literal)
    }

    /// Java's String.trim(): strips every char up to U+0020 from both ends.
    private static func jtrim(_ s: String) -> String {
        let u = Array(s.utf16)
        var a = 0, b = u.count
        while a < b && u[a] <= 0x20 { a += 1 }
        while b > a && u[b - 1] <= 0x20 { b -= 1 }
        return a == 0 && b == u.count ? s : s.jsub(a, b)
    }

    /// Integer/Long.parseInt of a run of digits (any script's, as Java's), saturating where Java would throw on overflow.
    private static func plong(_ s: String) -> Int {
        var v = 0
        for u in s.unicodeScalars {
            guard let d = u.properties.numericValue, u.properties.generalCategory == .decimalNumber else { continue }
            let (m, o1) = v.multipliedReportingOverflow(by: 10)
            let (t, o2) = m.addingReportingOverflow(Int(d))
            if o1 || o2 { return Int.max }
            v = t
        }
        return v
    }

    private static func pint(_ s: String) -> Int { plong(s) }

    /// Character.isLetter: general category L.
    private static func isLetter(_ u: Unicode.Scalar) -> Bool {
        switch u.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return false
        }
    }

    /// Character.isSpaceChar: Zs, Zl or Zp.
    private static func isSpaceChar(_ u: Unicode.Scalar) -> Bool {
        switch u.properties.generalCategory {
        case .spaceSeparator, .lineSeparator, .paragraphSeparator: return true
        default: return false
        }
    }

    /// Character.isLowerCase on a UTF-16 unit.
    private static func isLowerCase(_ c: unichar) -> Bool {
        guard let u = Unicode.Scalar(c) else { return false }
        return u.properties.isLowercase
    }
}
