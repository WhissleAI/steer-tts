import Foundation

/// One thing the assistant says, as parts that may sound different: "Sam on WhatsApp" in the assistant's own voice,
/// then Sam's message in Sam's voice and the message's mood ("I'm so sorry..." gently), then a short pause. The
/// Announcer renders each part with the user's style, adjusted for the situation (ContextStyler) and then the part's
/// mood and voice.
public final class Speech {
    public final class Part {
        public let text: String
        public let mood: Expression.Mood
        public let voice: String?        // nil = the user's voice
        public let pauseAfterMs: Int

        public init(_ text: String, _ mood: Expression.Mood?, _ voice: String?, _ pauseAfterMs: Int) {
            self.text = text
            self.mood = mood ?? Expression.NEUTRAL
            self.voice = voice
            self.pauseAfterMs = pauseAfterMs
        }
    }

    public var parts: [Part] = []

    public init() {}

    public static func of(_ text: String?) -> Speech { Speech().add(text) }

    @discardableResult
    public func add(_ text: String?) -> Speech { add(text, nil, nil, 0) }

    @discardableResult
    public func add(_ text: String?, _ mood: Expression.Mood?) -> Speech { add(text, mood, nil, 0) }

    @discardableResult
    public func add(_ text: String?, _ mood: Expression.Mood?, _ voice: String?, _ pauseAfterMs: Int) -> Speech {
        if let text = text, !jtrim(text).isEmpty { parts.append(Part(jtrim(text), mood, voice, pauseAfterMs)) }
        return self
    }

    @discardableResult
    public func pause(_ ms: Int) -> Speech {
        if let p = parts.popLast() {
            parts.append(Part(p.text, p.mood, p.voice, p.pauseAfterMs + ms))
        }
        return self
    }

    public var isEmpty: Bool { parts.isEmpty }

    /// All the words, for the screen and the log.
    public func text() -> String {
        var b = ""
        for p in parts { b += (b.isEmpty ? "" : " ") + p.text }
        return b
    }

    /// The strongest mood in it (for the now-speaking card).
    public func mainMood() -> Expression.Mood {
        for p in parts where !p.mood.neutral { return p.mood }
        return Expression.NEUTRAL
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
