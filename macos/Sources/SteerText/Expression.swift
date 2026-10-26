import Foundation

/// How a piece of text should sound, read from the text itself: the words ("so sorry", "congrats", "haha"), emoji,
/// exclamation marks, a message in capitals, a question. The result is a Mood: a style tag together with the control
/// shifts that make it audible (an emotion tag alone changes little on the edge model; see Style.preset), plus why.
/// Also turns chat text into something a voice can say: "👍" alone becomes "a thumbs up", emoji elsewhere and
/// "lol" are left to the delivery, "idk" becomes "I don't know". Ported from Expression.java (plain, unit tested).
public enum Expression {
    /// A delivery: tag (nil = keep the speaker's), control shifts to add (z-score units), and why.
    /// A class, as in Java: moods are compared by identity (`m !== Expression.URGENT`).
    public final class Mood {
        public let name: String
        public let tag: String?
        public let why: String
        /// Control -> shift. `ctrlKeys` keeps the insertion order (Java's LinkedHashMap).
        public let ctrl: [String: Float]
        public let ctrlKeys: [String]

        public convenience init(_ name: String, _ tag: String?, _ why: String, _ kv: (String, Float)...) {
            self.init(name, tag, why, pairs: kv)
        }

        public init(_ name: String, _ tag: String?, _ why: String, pairs kv: [(String, Float)]) {
            self.name = name
            self.tag = tag
            self.why = why
            var c: [String: Float] = [:]
            var keys: [String] = []
            for (k, v) in kv {
                if c[k] == nil { keys.append(k) }
                c[k] = v   // put: a repeated key replaces the value, keeps its place
            }
            ctrl = c
            ctrlKeys = keys
        }

        public var neutral: Bool { tag == nil && ctrl.isEmpty }

        /// The shifts as ordered pairs (Narration's flat()).
        public var pairs: [(String, Float)] { ctrlKeys.map { ($0, ctrl[$0]!) } }

        /// This mood with more shifts on top (a question's rising end, alert loudness).
        public func plus(_ why2: String, _ kv: (String, Float)...) -> Mood { plus(why2, pairs: kv) }

        public func plus(_ why2: String, pairs kv: [(String, Float)]) -> Mood {
            var c: [String: Float] = [:]
            var keys: [String] = []
            for (k, v) in pairs + kv {   // merge with Float::sum
                if let old = c[k] { c[k] = old + v } else { c[k] = v; keys.append(k) }
            }
            return Mood(name, tag, why.isEmpty ? why2 : why + "; " + why2, pairs: keys.map { ($0, c[$0]!) })
        }
    }

    public static let NEUTRAL = Mood("neutral", nil, "")
    public static let URGENT = Mood("urgent", "determined", "urgent: faster, louder", ("rate", 0.8), ("loud", 1.2), ("f0_med", 0.5), ("tilt", 0.8), ("pause", -0.5))
    public static let SAD = Mood("sad", "sadness", "sad news: slower, lower", ("rate", -0.6), ("f0_med", -0.7), ("f0_iqr", -0.7), ("loud", -0.6))
    public static let SORRY = Mood("sorry", "apologetic", "an apology: gentle", ("rate", -0.3), ("f0_med", -0.4), ("f0_iqr", -0.3), ("loud", -0.4))
    public static let WORRIED = Mood("worried", "anxiety", "worried: quicker, higher", ("rate", 0.4), ("f0_med", 0.5), ("f0_iqr", 0.6))
    public static let ANNOYED = Mood("annoyed", "irritation", "annoyed: firm", ("loud", 0.6), ("f0_iqr", 0.5), ("rate", 0.3), ("f0_end", -0.6))
    public static let EXCITED = Mood("excited", "excited", "excited: livelier, faster", ("rate", 0.7), ("loud", 0.9), ("f0_iqr", 1.4), ("f0_med", 0.9))
    public static let AMUSED = Mood("amused", "amusement", "amused: playful", ("f0_iqr", 1.0), ("f0_med", 0.5), ("rate", 0.3))
    public static let HAPPY = Mood("happy", "happiness", "happy: brighter", ("f0_iqr", 1.1), ("f0_med", 0.6), ("rate", 0.3), ("tilt", 0.4))
    public static let LOVING = Mood("loving", "contentment", "affectionate: warm", ("f0_iqr", 0.6), ("h1h2", 0.6), ("rate", -0.2))
    public static let GRATEFUL = Mood("grateful", "contentment", "thankful: warm", ("f0_iqr", 0.8), ("f0_med", 0.3))
    public static let SECRET = Mood("secret", "whisper", "a secret: whispered")
    public static let CALM = Mood("calm", "serenity", "calm", ("rate", -0.3), ("f0_iqr", -0.5), ("loud", -0.3))
    public static let CONFIDENT = Mood("confident", "confidence", "clear", ("f0_med", -0.3), ("loud", 0.3))

    /// Cues per mood, strongest first in tie breaks: words (one point) and emoji (two).
    // phrases rather than single words: "great" alone is often sarcastic, "died" is often a phone, "right now" is not urgent
    private static let CUES: [(Mood, String, String)] = [
        (URGENT, "\\b(urgent(ly)?|asap|emergency|911|help me|call me (right )?now|it'?s urgent|please call( me)? back now)\\b", "🚨|🆘"),
        (SAD, "\\b(passed away|funeral|heartbroken|devastated|so sad|bad news|lost (her|his|my|our) (mom|dad|mother|father|grand\\w*|wife|husband|dog|cat))\\b", "😢|😭|💔|😞|😔|☹️|😿"),
        (SORRY, "\\b(so sorry|i'?m sorry|sorry (about|for|to)|apologi[sz]e|apologies|my bad|forgive me|unfortunately|(got |been |was )cancell?ed)\\b", "🥺"),
        (WORRIED, "\\b(worried|scared|nervous|afraid|anxious|are you ok(ay)?|is everything ok(ay)?|hope you'?re ok(ay)?)\\b", "😟|😰|😨|😧"),
        (ANNOYED, "\\b(ugh+|so annoy(ed|ing)|so angry|furious|wtf|ridiculous|so done with)\\b|\\bseriously\\?+", "😡|😠|🤬|🙄|😤"),
        (EXCITED, "\\b(congrats|congratulations|amazing news|so excited|can'?t wait|woo+hoo+|yay+|we won|got the job|got in|happy birthday|let'?s go+|we did it)\\b", "🎉|🥳|🎊|🤩|🎂|🚀"),
        (AMUSED, "\\b(ha(ha)+h?|he(he)+|lol+|lmao+|rofl|too funny|hilarious|so funny)\\b", "😂|🤣|😆|😹|😜|😝"),
        (LOVING, "\\b(love you|love u|luv u|miss you|miss u|xoxo)\\b", "❤️|❤|😍|🥰|😘|💕|💖|💗|💞"),
        (GRATEFUL, "\\b(thank you|thanks so much|thanks a lot|many thanks|appreciate (it|you))\\b", "🙏"),
        (HAPPY, "\\b(good news|so glad|so happy|well done|wonderful|nice one|great news|good morning)\\b", "😊|🙂|😄|😁|😃|☺️|😀|✨"),
        (SECRET, "\\b(don'?t tell (anyone|anybody)|it'?s a secret|between us|keep it quiet)\\b", "🤫"),
    ]
    private static let NOT = "(?<!\\bnot )(?<!n't )(?<!\\bno )"   // "not great news", "isn't amazing news"
    private static let PATTERNS: [(NSRegularExpression, NSRegularExpression)] = CUES.map {
        (try! NSRegularExpression(pattern: NOT + "(?:" + $0.1 + ")", options: [.caseInsensitive]), rx($0.2))
    }

    /// The mood of a message or a line of dialogue.
    public static func of(_ text: String?) -> Mood {
        guard let text = text, !jtrim(text).isEmpty else { return NEUTRAL }
        var best = -1, bestScore = 0
        var score = [Int](repeating: 0, count: CUES.count)
        for i in 0..<CUES.count {
            score[i] = PATTERNS[i].0.all(text).count + 2 * PATTERNS[i].1.all(text).count
        }
        let bangs = rx("!").all(text).count
        if bangs >= 2 || text.contains("!!") { score[index(EXCITED)] += score[index(ANNOYED)] > 0 ? 0 : 1 }
        if shouting(text) {
            if score[index(ANNOYED)] > 0 { score[index(ANNOYED)] += 1 } else { score[index(EXCITED)] += 1 }
        }
        for i in 0..<CUES.count where score[i] > bestScore {
            bestScore = score[i]
            best = i
        }
        var m = best < 0 ? NEUTRAL : CUES[best].0
        if shouting(text) { m = m.plus("in capitals: louder", ("loud", 0.6)) }
        if question(text) && m !== URGENT { m = m.plus("question: rising end", ("f0_end", 1.4)) }
        return m
    }

    private static func index(_ m: Mood) -> Int { CUES.firstIndex { $0.0 === m } ?? -1 }

    /// Mostly capitals, and long enough to be shouting rather than an acronym.
    public static func shouting(_ s: String) -> Bool {
        var up = 0, letters = 0
        for c in s.utf16 {   // per UTF-16 unit, as Java's charAt (a surrogate is not a letter)
            guard let u = Unicode.Scalar(c) else { continue }
            switch u.properties.generalCategory {   // Character.isLetter: L* (CharacterSet.letters also has marks)
            case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: break
            default: continue
            }
            do {
                letters += 1
                if u.properties.isUppercase { up += 1 }
            }
        }
        return letters >= 8 && up * 10 >= letters * 8
    }

    public static func question(_ s: String) -> Bool {
        let t = jtrim(s).replaceAll("[\\s\\p{So}\\p{Sk}\\uFE0F\\u200D]+$", "")
        return t.hasSuffix("?")
    }

    // ---------------------------------------------------------------- what the voice can say
    private static let EMOJI_WORDS: [(String, String)] = [
        ("👍", "a thumbs up"), ("👎", "a thumbs down"), ("❤️", "a heart"), ("❤", "a heart"), ("😂", "a laughing face"),
        ("🤣", "a laughing face"), ("😍", "a heart-eyes face"), ("🎉", "a party popper"), ("🙏", "folded hands"),
        ("😢", "a crying face"), ("😭", "a crying face"), ("😮", "a surprised face"), ("😡", "an angry face"),
        ("👋", "a wave"), ("🔥", "a fire emoji"), ("👏", "applause"), ("😘", "a kiss"), ("🥰", "a smiling face with hearts"),
        ("😊", "a smile"), ("🙂", "a smile"), ("👌", "an OK hand"), ("💯", "a hundred points emoji")]

    private static let CHAT: [(String, String)] = [
        ("idk", "I don't know"), ("btw", "by the way"), ("brb", "be right back"), ("omw", "on my way"),
        ("tmrw", "tomorrow"), ("tmr", "tomorrow"), ("thx", "thanks"), ("ty", "thank you"), ("pls", "please"),
        ("plz", "please"), ("u", "you"), ("ur", "your"), ("omg", "oh my god"), ("imo", "in my opinion"),
        ("ttyl", "talk to you later"), ("np", "no problem"), ("wyd", "what are you doing"), ("hbu", "how about you"),
        ("gm", "good morning"), ("gn", "good night"), ("bday", "birthday"), ("rn", "right now"), ("ily", "I love you")]
    private static let LAUGH = "(?i)(^|\\s)(lol+|lmao+|rofl)(?=\\s|[.,!?]|$)"
    private static let EMOJI = "[\\x{1F000}-\\x{1FAFF}\\x{2600}-\\x{27BF}\\x{2B00}-\\x{2BFF}\\x{1F1E6}-\\x{1F1FF}"
        + "\\x{FE0F}\\x{200D}\\x{20E3}\\x{E0020}-\\x{E007F}]+"

    /// Only emoji (and spaces): such a message is described ("sent a thumbs up").
    public static func emojiOnly(_ s: String) -> Bool {
        let t = jtrim(s)
        return !t.isEmpty && jtrim(t.replaceAll(EMOJI, "")).isEmpty
    }

    /// "a thumbs up" for a message that is only emoji; nil if it is not.
    public static func describeEmoji(_ s: String) -> String? {
        if !emojiOnly(s) { return nil }
        let t = jtrim(s) as NSString
        // startsWith on UTF-16 units, as Java ("❤" also starts "❤️")
        for e in EMOJI_WORDS where t.hasPrefix(e.0) { return e.1 }
        return "an emoji"
    }

    /// Chat text as words: emoji dropped (the mood carries them), "lol" dropped, chat abbreviations spelled out, letters
    /// repeated for effect shortened ("sooooo" -> "soo"), characters outside the Basic Multilingual Plane removed (they
    /// would reach the engine as broken UTF-8). Not for documents (no abbreviation expansion there).
    public static func speakable(_ s: String) -> String {
        var t = s.replaceAll(EMOJI, " ")
        t = t.replaceAll(LAUGH, "$1")
        t = t.replaceAll("(\\p{L})\\1{2,}", "$1$1")
        for c in CHAT {
            t = t.replaceAll("(?i)(?<![\\w'])" + NSRegularExpression.escapedPattern(for: c.0) + "(?![\\w'])",
                             NSRegularExpression.escapedTemplate(for: c.1))
        }
        return jtrim(bmp(t).replaceAll("\\s+([,.!?])", "$1").replaceAll("\\s{2,}", " "))
    }

    /// Replaces code points above U+FFFF with a space (the C engine got them as invalid UTF-8 through JNI on Android).
    public static func bmp(_ s: String) -> String {
        var out = String.UnicodeScalarView()
        for u in s.unicodeScalars { out.append(u.value > 0xFFFF ? " " : u) }
        return String(out)
    }

    /// "excited: livelier, faster" -> shown in the app as why it sounds the way it does.
    public static func describe(_ m: Mood) -> String { m.neutral ? "" : m.why.lowercased() }
}

/// Java's String.trim(): code units <= U+0020 off both ends.
fileprivate func jtrim(_ s: String) -> String {
    let u = Array(s.utf16)
    var a = 0, b = u.count
    while a < b && u[a] <= 0x20 { a += 1 }
    while b > a && u[b - 1] <= 0x20 { b -= 1 }
    return a == 0 && b == u.count ? s : s.jsub(a, b)
}
