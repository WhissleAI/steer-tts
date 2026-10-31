import Foundation

/// How a sentence of a document is performed (plain, unit tested): a heading slower with a pause after it;
/// dialogue (words in quotation marks) apart from the narration, in the dialogue voice and the line's own mood
/// ("Run!" excited, "I'm so sorry" gentle), narration in the reader's voice; words the document set in italics or bold
/// stressed (engine markup *word*). Scare quotes ("so-called") stay narration. Ported from Narration.java.
public enum Narration {
    /// Private-use markers the text pipeline leaves in document text: emphasis on / off, and a heading (first character
    /// of the paragraph). Strings of one character (Java chars), so they concatenate into text and patterns directly.
    public static let EM_ON = "\u{E000}", EM_OFF = "\u{E001}", HEAD = "\u{E002}"
    public static let HEADING = Expression.Mood("heading", nil, "a heading: slower", ("rate", -0.3), ("f0_iqr", 0.3), ("loud", 0.2))

    public struct Seg {
        public let text: String          // engine text, markup allowed (*word*)
        public let dialogue: Bool, heading: Bool
        public let mood: Expression.Mood
        public let pauseAfterMs: Int

        public init(text: String, dialogue: Bool, heading: Bool, mood: Expression.Mood, pauseAfterMs: Int) {
            self.text = text
            self.dialogue = dialogue
            self.heading = heading
            self.mood = mood
            self.pauseAfterMs = pauseAfterMs
        }
    }

    /// line: the paragraph's reading from Cast (who speaks, how, the scene), or nil. Dialogue gets the speaker's
    /// delivery (whispered, shouted…) or, without one, the mood of its own words; narration gets the scene's mood.
    public static func segments(_ sentence: String, _ line: Cast.Line? = nil) -> [Seg] {
        var out: [Seg] = []
        if sentence.isEmpty { return out }
        if (sentence as NSString).hasPrefix(HEAD) {
            out.append(Seg(text: markup(sentence.jsub(1)), dialogue: false, heading: true, mood: HEADING, pauseAfterMs: 650))
            return out
        }
        let scene = line?.scene ?? Expression.NEUTRAL
        var at = 0
        for m in Cast.QUOTE.all(sentence) {
            let inner = Cast.inner(m)
            if !isDialogue(inner) { continue }
            narration(&out, sentence.jsub(at, m.start()), scene)
            let own = Expression.of(display(inner))
            let mood: Expression.Mood
            if let d = line?.delivery {
                mood = d.tag == nil ? (own.neutral ? d : own.plus(d.why, pairs: d.pairs)) : d
            } else {
                mood = own
            }
            out.append(Seg(text: markup(inner), dialogue: true, heading: false, mood: mood, pauseAfterMs: 120))
            at = m.end()
        }
        narration(&out, sentence.jsub(at), scene)
        return out
    }

    private static func narration(_ out: inout [Seg], _ s: String, _ scene: Expression.Mood) {
        let t = jtrim(s)
        if t.replaceAll("[!-/:-@\\[-`{-~\\s“”‘’—–]", "").isEmpty { return }   // only a comma or a dash between quotes
        // (Java's \\p{Punct} is ASCII punctuation; ICU's would be Unicode P, so the ASCII set is spelled out)
        out.append(Seg(text: markup(t), dialogue: false, heading: false, mood: scene, pauseAfterMs: 120))
    }

    /// Spoken words, not a scare quote or a title: three words or more, or a line ending in punctuation.
    public static func isDialogue(_ inner: String) -> Bool {
        let t = jtrim(inner)
        return t.split(regex: "\\s+").count >= 3 || t.matches(".*[.!?,]$")
    }

    /// Engine text: characters the engine would read as markup ([ ] < > *) made plain, then emphasis markers as *...*
    /// when they pair up in this piece (a sentence split inside an italic run just loses the stress).
    public static func markup(_ s: String) -> String {
        var t = s.replacingOccurrences(of: "[", with: "(").replacingOccurrences(of: "]", with: ")")
            .replacingOccurrences(of: "<", with: " ").replacingOccurrences(of: ">", with: " ")
            .replacingOccurrences(of: "*", with: " ")
        let on = count(t, EM_ON), off = count(t, EM_OFF)
        if on == off && on > 0 {
            t = t.replacingOccurrences(of: EM_ON, with: "*").replacingOccurrences(of: EM_OFF, with: "*")
                .replaceAll("\\*\\s*\\*", " ")
        } else {
            t = strip(t)
        }
        return jtrim(t.replaceAll("\\s{2,}", " "))
    }

    private static func count(_ s: String, _ c: String) -> Int {
        let u = c.utf16.first!
        return s.utf16.reduce(0) { $0 + ($1 == u ? 1 : 0) }
    }

    /// For the screen: the markers removed.
    public static func display(_ s: String) -> String { jtrim(strip(s)) }

    public static func strip(_ s: String) -> String {
        if !s.utf16.contains(where: { $0 >= 0xE000 && $0 <= 0xE002 }) { return s }
        return s.replacingOccurrences(of: EM_ON, with: "").replacingOccurrences(of: EM_OFF, with: "")
            .replacingOccurrences(of: HEAD, with: "")
    }
}

/// Java's String.trim(): code units <= U+0020 off both ends.
fileprivate func jtrim(_ s: String) -> String {
    let u = Array(s.utf16)
    var a = 0, b = u.count
    while a < b && u[a] <= 0x20 { a += 1 }
    while b > a && u[b - 1] <= 0x20 { b -= 1 }
    return a == 0 && b == u.count ? s : s.jsub(a, b)
}
