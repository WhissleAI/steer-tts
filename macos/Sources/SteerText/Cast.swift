import Foundation

/// Who speaks in a story, and how (plain, unit tested). Read once per book, paragraph by paragraph, the way a
/// reader follows it: "…,” said Tom / Tom said / she whispered / said the Queen name the speaker of a paragraph's
/// dialogue; "he" and "she" stand for the last man or woman named; in an exchange with no names, the two speakers take
/// turns. Each character gets a voice of their own (Voices), the speech verb says how the line is delivered (whispered,
/// shouted, sobbed, laughed), and a light mood is read from the narration (a tense scene, a sad one, a happy one).
/// Ported from Cast.java.
public final class Cast {
    /// One paragraph: its speaker (nil when unknown or no dialogue), how the line is said, and the scene's mood.
    public struct Line {
        public let speaker: String?, gender: String?
        public let delivery: Expression.Mood?, scene: Expression.Mood

        public init(speaker: String?, gender: String?, delivery: Expression.Mood?, scene: Expression.Mood) {
            self.speaker = speaker
            self.gender = gender
            self.delivery = delivery
            self.scene = scene
        }
    }

    public private(set) var lines: [Line] = []
    /// The characters met so far, in order (Java's LinkedHashMap key order), and their gender where known:
    /// character -> "female" / "male" (absent = unknown).
    public private(set) var characters: [String] = []
    public private(set) var gender: [String: String] = [:]

    private init() {}

    // ---------------------------------------------------------------- dialogue
    /// Curly double quotes, straight double quotes, and curly single quotes (an apostrophe inside a word is not a close).
    // straight single quotes too ('Now, my dears,' said old Mrs. Rabbit): an apostrophe between letters (don't,
    // McGregor's) is never a quote mark
    public static let QUOTE = rx("“([^”]+)”|\"([^\"]+)\"|‘((?:[^’]|’(?=\\p{L}))+)’(?!\\p{L})"
        + "|(?<![\\p{L}\\d])'((?:[^']|'(?=\\p{L}))+?)'(?!\\p{L})")

    public static func hasDialogue(_ p: String) -> Bool {
        for m in QUOTE.all(p) where Narration.isDialogue(inner(m)) { return true }
        return false
    }

    public static func inner(_ m: JMatch) -> String {
        for g in 1...4 { if let s = m.group(g) { return s } }
        return ""
    }

    /// The paragraph with its quoted words blanked out: what the narrator says around the dialogue.
    public static func outside(_ p: String) -> String {
        QUOTE.replaceAll(p) { m in Narration.isDialogue(inner(m)) ? " ¶ " : m.group()! }
    }

    // ---------------------------------------------------------------- speech verbs
    private static let WHISPER = Expression.Mood("whispered", "whisper", "whispered")
    private static let SHOUT = Expression.Mood("shouted", "loud", "shouted: loud", ("loud", 1.4), ("rate", 0.3), ("f0_med", 0.6), ("f0_iqr", 0.8))
    private static let MUTTER = Expression.Mood("muttered", "irritation", "muttered: low", ("loud", -0.8), ("f0_med", -0.4), ("rate", 0.2))
    private static let STAMMER = Expression.Mood("stammered", "hesitant", "hesitant", ("rate", -0.4), ("pause", 0.6))
    private static let GASP = Expression.Mood("gasped", "amazement", "surprised", ("f0_med", 0.8), ("f0_iqr", 1.0))
    private static let PLEAD = Expression.Mood("pleaded", "distress", "pleading", ("f0_med", 0.4), ("f0_iqr", 0.6), ("rate", -0.2))
    private static let SIGH = Expression.Mood("sighed", "tired", "tired", ("rate", -0.4), ("f0_iqr", -0.5), ("loud", -0.4))
    private static let GROWL = Expression.Mood("growled", "anger", "angry", ("loud", 0.8), ("f0_med", -0.6), ("f0_end", -0.8))
    private static let ASK = Expression.Mood("asked", nil, "a question", ("f0_end", 1.2))

    private static let VERBS: [(Expression.Mood?, String)] = [
        (WHISPER, "whisper(ed|s)?|murmur(ed|s)?|breathed"),
        (SHOUT, "shout(ed|s)?|yell(ed|s)?|scream(ed|s)?|roar(ed|s)?|bellow(ed|s)?|cried out|exclaim(ed|s)?|called out"),
        (Expression.SAD, "sob(bed|s)?|wept|weep(s)?|wail(ed|s)?"),
        (Expression.AMUSED, "laugh(ed|s)?|giggl(ed|es)|chuckl(ed|es)|grinn?ed"),
        (GROWL, "growl(ed|s)?|snarl(ed|s)?|snapp?ed|hiss(ed|es)?|thunder(ed|s)?"),
        (MUTTER, "mutter(ed|s)?|grumbl(ed|es)|mumbl(ed|es)"),
        (STAMMER, "stammer(ed|s)?|stutter(ed|s)?|faltered"),
        (GASP, "gasp(ed|s)?"),
        (PLEAD, "plead(ed|s)?|pled|begg?ed|implor(ed|es)"),
        (SIGH, "sigh(ed|s)?"),
        (ASK, "ask(ed|s)?|enquir(ed|es)|inquir(ed|es)|wonder(ed|s)?"),
        (nil, "said|says|say|replied|replies|answered|added|continued|went on|told|remarked|repeated|agreed|explained|"
            + "began|called|cried|insisted|announced|declared|suggested|interrupted|thought|admitted|observed"),
    ]
    // no groups of their own
    private static let VERB = "(?:" + VERBS.map { $0.1.replacingOccurrences(of: "(", with: "(?:") }.joined(separator: "|") + ")"
    private static let VERB_PATTERNS: [NSRegularExpression] = VERBS.map { rx("(?i)^(?:" + $0.1 + ")$") }

    public static func delivery(_ verb: String?) -> Expression.Mood? {
        guard let verb = verb else { return nil }
        let v = jtrim(verb.lowercased())
        for i in 0..<VERBS.count where VERB_PATTERNS[i].whole(v) { return VERBS[i].0 }
        return nil
    }

    // a name: one to three capitalised words, or "the" + one or two capitalised words ("the Queen", "the March Hare")
    private static let NAME = "((?:the |The )?(?:Mr\\.|Mrs\\.|Ms\\.|Miss|Dr\\.|Sir|Lady|Lord|Aunt|Uncle|Captain|King|Queen|Prince|Princess)?\\s?"
        + "\\p{Lu}[\\p{L}’'-]+(?: \\p{Lu}[\\p{L}’'-]+){0,2})"
    private static let ADJ = "(?:(?:old|young|little|poor|dear|kind|good|wise|tall|small|big)\\s+)?"   // "said old Mrs. Rabbit"
    private static let WHO = ADJ + "(" + String(NAME.dropFirst().dropLast()) + "|he|she|He|She|I)"
    // after the quote: "”, said Tom" / "” said the Queen" / "” Tom said" / "” she whispered"
    private static let AFTER_VS = rx("^\\s*[,.!?—–-]?\\s*(" + VERB + ")\\s+" + WHO + "(?=[\\s,.;!?]|$)")
    private static let AFTER_SV = rx("^\\s*[,.!?—–-]?\\s*" + WHO + "\\s+(?:\\w+ly\\s+)?(" + VERB + ")\\b")
    // an unnamed speaker: "said our strange visitor", "said a voice": someone, for turn-taking
    private static let AFTER_SOMEONE = rx("^\\s*[,.!?—–-]?\\s*(" + VERB + ")\\s+(?:our|the|a|an|his|her|my|their|old|young)\\s+\\p{Ll}")
    // before the quote: "Tom said, “" / "She whispered: “"
    private static let BEFORE_SV = rx(WHO + "\\s+(?:\\w+ly\\s+)?(" + VERB + ")[^.!?]{0,40}$")

    private static let NOT_NAMES: Set<String> = [
        "The", "Then", "But", "And", "It", "They", "We", "You", "There", "This", "That", "What", "When", "Where", "Why",
        "How", "Yes", "No", "Oh", "Well", "So", "If", "As", "At", "In", "On", "Now", "Here", "Just", "All", "One",
        "His", "Her", "Its", "Our", "Their", "My", "Your", "A", "An", "Nobody", "Everyone", "Someone", "Somebody"]

    private static let FEMALE: Set<String> = [
        "alice", "mary", "anna", "emma", "elizabeth", "jane", "lucy", "sarah", "sophie", "grace", "mia", "olivia", "ava",
        "lily", "ella", "hannah", "kate", "katie", "laura", "rose", "ruby", "amy", "beth", "meg", "jo", "wendy", "dorothy",
        "susan", "margaret", "helen", "clara", "heidi", "matilda", "hermione", "ginny", "luna", "molly", "petunia", "anne",
        "diana", "marianne", "eleanor", "catherine", "isabella", "harriet", "fanny", "emily", "charlotte", "maria",
        "nora", "zoe", "chloe", "julia", "rachel", "nancy", "betty", "linda", "karen", "lisa", "amelia", "isla",
        "mother", "mum", "mom", "mama", "grandma", "granny", "sister", "aunt", "queen", "princess", "duchess", "lady",
        "girl", "woman", "witch", "mrs", "miss", "ms", "madam", "wife", "daughter", "hen", "mrs."]
    private static let MALE: Set<String> = [
        "tom", "john", "james", "peter", "harry", "ron", "jack", "sam", "ben", "george", "william", "henry", "charles",
        "edward", "robert", "david", "michael", "paul", "mark", "luke", "matthew", "oliver", "noah", "leo", "max", "finn",
        "arthur", "frank", "fred", "bob", "jim", "joe", "huck", "pip", "darcy", "bingley", "heathcliff",
        "sherlock", "watson", "holmes", "frodo", "bilbo", "gandalf", "aragorn", "merry", "pippin", "dumbledore",
        "snape", "hagrid", "draco", "neville", "father", "dad", "papa", "grandpa", "grandfather", "brother", "uncle",
        "king", "prince", "duke", "lord", "sir", "boy", "man", "wizard", "mr", "mr.", "husband", "son", "captain",
        "rabbit", "hatter", "caterpillar", "mouse", "dodo", "gryphon", "knave"]

    /// "female", "male" or nil for a name, from titles, family words and common first names.
    public static func genderOf(_ name: String) -> String? {
        for w in name.lowercased().replacingOccurrences(of: "the ", with: "").split(regex: "[\\s.]+") {
            if FEMALE.contains(w) { return "female" }
            if MALE.contains(w) { return "male" }
        }
        return nil
    }

    /// "the Queen" -> "Queen", "the Rabbit’s" -> "Rabbit", "Mr. Hall" stays: one key per character.
    public static func canonical(_ name: String) -> String {
        jtrim(name.replaceFirst("^(the|The)\\s+", "").replaceFirst("[’']s$", ""))
    }

    /// The character a name refers to: an earlier one whose name contains this one or is contained in it ("Rabbit" and
    /// "White Rabbit", "Mock" and "Mock Turtle") is the same character; otherwise a new one.
    private func resolve(_ name: String) -> String {
        let words = Set(name.split(regex: " "))
        for known in characters {
            let kw = Set(known.split(regex: " "))
            if kw.isSuperset(of: words) || words.isSuperset(of: kw) { return known }
        }
        return name
    }

    /// Java's gender.putIfAbsent(name, g): a new character is remembered, a known gender is never overwritten.
    private func remember(_ name: String, _ g: String?) {
        if !characters.contains(name) { characters.append(name) }
        if gender[name] == nil, let g = g { gender[name] = g }
    }

    // ---------------------------------------------------------------- scenes
    private static let TENSE = Expression.Mood("tense", "mysterious", "a tense scene", ("rate", -0.15), ("loud", -0.25), ("f0_iqr", -0.2))
    private static let SOMBRE = Expression.Mood("sombre", "sadness", "a sad scene", ("rate", -0.25), ("f0_med", -0.25), ("f0_iqr", -0.3))
    private static let BRIGHT = Expression.Mood("bright", "contentment", "a happy scene", ("f0_iqr", 0.35), ("f0_med", 0.15))
    private static let ACTION = Expression.Mood("action", nil, "action", ("rate", 0.3), ("loud", 0.2))
    private static let P_TENSE = rx("(?i)\\b(dark(ness)?|shadow|creep|crept|silence|silent|whisper|fear|afraid|terrif|horror|dread|ghost|cold|trembl|shiver|eerie|strange noise|footsteps)\\w*")
    private static let P_SOMBRE = rx("(?i)\\b(tears|wept|weeping|grief|griev|funeral|died|death|dead|mourn|lonely|alone|sorrow|sad)\\w*")
    private static let P_BRIGHT = rx("(?i)\\b(laugh|smil|joy|happ|delight|sunshine|cheer|merr|danced|sang|wonderful)\\w*")
    private static let P_ACTION = rx("(?i)\\b(ran|run|rushed|dashed|leapt|jumped|crash|suddenly|chased|fled|raced|grabbed|burst)\\w*")

    /// A light mood for narration, from the paragraph's words (two cues of a kind; the strongest kind wins).
    public static func scene(_ p: String) -> Expression.Mood {
        let t = P_TENSE.all(p).count, s = P_SOMBRE.all(p).count, b = P_BRIGHT.all(p).count, a = P_ACTION.all(p).count
        let mx = max(max(t, s), max(b, a))
        if mx < 2 { return Expression.NEUTRAL }
        return mx == s ? SOMBRE : mx == t ? TENSE : mx == a ? ACTION : BRIGHT
    }

    // ---------------------------------------------------------------- the reading
    public static func analyze(_ paragraphs: [String]) -> Cast {
        let c = Cast()
        var lastFemale: String? = nil, lastMale: String? = nil, last: String? = nil, beforeLast: String? = nil
        var prevDialogue = false, prevPrevDialogue = false
        c.lines.reserveCapacity(paragraphs.count)
        for raw in paragraphs {
            let p = Narration.display(raw)
            let scene = scene(p)
            if !hasDialogue(p) {
                // names in narration are remembered for "he" / "she" later
                for n in namesIn(p) {
                    let g = c.genderFor(n)
                    if g == "female" { lastFemale = c.resolve(n) } else if g == "male" { lastMale = c.resolve(n) }
                }
                c.lines.append(Line(speaker: nil, gender: nil, delivery: nil, scene: scene))
                prevPrevDialogue = prevDialogue
                prevDialogue = false
                continue
            }
            var who: String? = nil, verb: String? = nil
            var searchFrom = 0
            for q in QUOTE.all(p) {
                if who != nil { break }
                if !Narration.isDialogue(inner(q)) { continue }
                let after = p.jsub(q.end())
                if let m = AFTER_VS.find(after) { verb = m.group(1); who = m.group(2) }
                else if let m = AFTER_SOMEONE.find(after) { verb = m.group(1); who = "?" }
                else if let m = AFTER_SV.find(after) { who = m.group(1); verb = m.group(2) }
                else if let m = BEFORE_SV.find(p.jsub(searchFrom, q.start())) { who = m.group(1); verb = m.group(2) }
                searchFrom = q.end()
            }
            var speaker: String? = nil
            if let wh = who {
                let w = jtrim(wh)
                if w == "~" { /* speaker already found in the narration */ }
                else if w == "?" { speaker = nil }   // someone unnamed
                else if w.lowercased() == "she" { speaker = lastFemale }
                else if w.lowercased() == "he" { speaker = lastMale }
                else if w == "I" { speaker = "Narrator" }
                else if !NOT_NAMES.contains(w.split(regex: " ")[0]) { speaker = c.resolve(canonical(w)) }
            }
            if who == nil {   // no "said X": the narration around the quote names the speaker, or "she" / "he" does
                let narr = outside(p)
                var best = Int.max
                for known in c.characters {
                    if let km = c.namePattern(known).find(narr), km.start() < best {
                        best = km.start()
                        speaker = known
                    }
                }
                if speaker == nil, let pm = rx("\\b(she|he|She|He)\\b").find(narr) {
                    speaker = pm.group(1)!.lowercased() == "she" ? lastFemale : lastMale
                }
                if speaker != nil { who = "~" }   // found in the narration: no turn-taking guess below
            }
            if who == nil && prevDialogue {
                if prevPrevDialogue, let bl = beforeLast, bl != last {
                    speaker = bl == "?" ? nil : bl   // a back-and-forth without names: turns
                } else if let l = last, l != "?" {
                    speaker = l   // no exchange going on: the last speaker goes on (a monologue over paragraphs)
                }
            }
            var g: String? = nil
            if let sp = speaker, sp != "Narrator" {
                g = c.genderFor(sp)
                if g == nil, let w = who?.lowercased(), w == "she" || w == "he" { g = w == "she" ? "female" : "male" }
                c.remember(sp, g)
                if g == "female" { lastFemale = sp } else if g == "male" { lastMale = sp }
            }
            let turn = speaker ?? "?"   // the unnamed count as one more person in the exchange
            if turn != last {
                beforeLast = last
                last = turn
            }
            c.lines.append(Line(speaker: speaker, gender: g, delivery: delivery(verb), scene: scene))
            prevPrevDialogue = prevDialogue
            prevDialogue = true
        }
        return c
    }

    private var namePatterns: [String: NSRegularExpression] = [:]

    private func namePattern(_ name: String) -> NSRegularExpression {
        if let p = namePatterns[name] { return p }
        let p = try! NSRegularExpression(pattern: "\\b" + NSRegularExpression.escapedPattern(for: name) + "\\b")
        namePatterns[name] = p
        return p
    }

    private func genderFor(_ name: String) -> String? { gender[name] ?? Cast.genderOf(name) }

    private static let NAME_IN_TEXT = rx("(?<![.!?]\\s)(?<!^)\\b((?:(?:Mr|Mrs|Ms|Dr)\\.|Miss|Sir|Lady|Lord)\\s)?(\\p{Lu}[a-z]{2,})\\b")
    private static let TITLES: Set<String> = ["Mr", "Mrs", "Ms", "Miss", "Dr", "Sir", "Lady", "Lord"]

    /// Capitalised words that are not the first word of a sentence: likely names.
    private static func namesIn(_ p: String) -> [String] {
        var out: [String] = []
        for m in NAME_IN_TEXT.all(p) {
            let n = m.group(2)!
            if NOT_NAMES.contains(n) || TITLES.contains(n) { continue }
            out.append((m.group(1) ?? "") + n)   // "Mrs. Hudson", not "Mrs"
        }
        return out
    }

    // ---------------------------------------------------------------- voices
    private var voices: [String: String] = [:]
    private let lock = NSLock()

    /// The voice of a character: one of the catalogue voices of their gender (either when unknown), never the
    /// narrator's, and different from the characters before them as long as there are voices to go round.
    public func voiceFor(_ speaker: String?, narrator: String?, female: [String], male: [String]) -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard let speaker = speaker else { return nil }
        if speaker == "Narrator" { return narrator }
        if let v = voices[speaker] { return v }
        let g = gender[speaker]
        var pool = g == "male" ? male : female
        if g == nil { pool += male }
        if let n = narrator, let i = pool.firstIndex(of: n) { pool.remove(at: i) }   // List.remove: the first one
        let taken = Set(voices.values)
        pool.removeAll { taken.contains($0) }
        if pool.isEmpty { pool = g == "male" ? male : female }
        var h: Int32 = 0
        for u in speaker.utf16 { h = 31 &* h &+ Int32(javaLower(u)) }
        let n = Int32(pool.count)
        let v = pool[Int(((h % n) + n) % n)]   // Math.floorMod
        voices[speaker] = v
        return v
    }
}

/// Character.toLowerCase(char): the simple lower-case mapping of one UTF-16 unit.
fileprivate func javaLower(_ u: UInt16) -> UInt16 {
    guard let s = Unicode.Scalar(u) else { return u }
    let l = s.properties.lowercaseMapping.unicodeScalars
    guard l.count == 1, let f = l.first, f.value <= 0xFFFF else { return u }
    return UInt16(f.value)
}

/// Java's String.trim(): code units <= U+0020 off both ends.
fileprivate func jtrim(_ s: String) -> String {
    let u = Array(s.utf16)
    var a = 0, b = u.count
    while a < b && u[a] <= 0x20 { a += 1 }
    while b > a && u[b - 1] <= 0x20 { b -= 1 }
    return a == 0 && b == u.count ? s : s.jsub(a, b)
}
