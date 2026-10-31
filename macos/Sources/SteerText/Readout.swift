import Foundation

/// What the assistant says for a message or a call, as a Speech (plain Swift, unit tested): the framing ("Sam on
/// WhatsApp") in the assistant's voice, then the message in its own mood, and in the sender's voice when people have
/// voices of their own. A message that is only emoji is described ("Sam sent a thumbs up"). Calls get more urgent when
/// the same caller rings again within a quarter of an hour.
public enum Readout {
    public static let MAX_CHARS = 300

    /// person: the sender (nil if unknown); group: the conversation's title when it is a group; app: the app's name;
    /// texts: the new messages, oldest first.
    public static func message(_ person: String?, _ group: String?, _ app: String, _ texts: [String],
                               _ personVoices: Bool, _ ownVoice: String?) -> Speech {
        var person = person, group = group
        let sp = Speech()
        var body = ""
        var onlyEmoji: String? = nil
        for t in texts {
            let d = Expression.describeEmoji(t)
            if d != nil && texts.count == 1 { onlyEmoji = d }
            else if d == nil { body += (body.isEmpty ? "" : " ") + sentence(t) }
        }
        if person != nil && Spoken.unspoken(person) != nil { person = "Someone" }   // a name the voice can't read
        if group != nil && Spoken.unspoken(group) != nil { group = nil }
        let who = person == nil ? app : group != nil ? person! + " in " + group! : person!
        if let e = onlyEmoji { return sp.add(who + (person == nil ? ": " : " sent ") + e + ".") }
        let said = cap(body)
        if said.isEmpty { return sp }
        if let lang = Spoken.unspoken(said) {   // say that it came, instead of a framing followed by silence
            return sp.add((person == nil ? "A message on " + app : who + " sent a message on " + app)
                + (lang == "another language" ? ", in a language I can't read yet." : ", in " + lang + ". I can't read " + lang + " yet."))
        }
        let framing = person == nil ? app + "." : who + " on " + app + "."
        sp.add(framing, nil, nil, 180)
        // the mood is read from the original words (emoji, "lol" and capitals carry it), the words spoken are cleaned
        var raw = ""
        for t in texts { raw += t + " " }
        let voice = personVoices && person != nil ? Voices.forPerson(person!, own: ownVoice) : nil
        sp.add(said, Expression.of(raw), voice, 0)
        return sp
    }

    /// New messages of one conversation, oldest first, as (sender, text) pairs: each run from one person gets its own
    /// framing and voice ("Sam in Family on WhatsApp. …  Then Mom. …"), so nobody's words are read as someone else's.
    public static func conversation(_ msgs: [(String, String)], _ group: String?, _ app: String,
                                    _ personVoices: Bool, _ ownVoice: String?) -> Speech {
        let sp = Speech()
        var i = 0
        var first = true
        while i < msgs.count {
            let who = msgs[i].0
            var run: [String] = []
            while i < msgs.count && msgs[i].0 == who { run.append(msgs[i].1); i += 1 }
            let part = message(who, group, app, run, personVoices, ownVoice)
            if !first, let f = part.parts.first {   // later senders: "Then Mom." instead of the full framing again
                let framing = (f.text as NSString).hasSuffix(" on " + app + ".") ? "Then " + who + "." : f.text
                part.parts[0] = Speech.Part(framing, f.mood, f.voice, f.pauseAfterMs)
            }
            sp.parts.append(contentsOf: part.parts)
            first = false
        }
        return sp
    }

    /// A notification that is not a conversation: its title (a subject, a headline) and text.
    public static func notice(_ title: String?, _ text: String?, _ app: String) -> Speech {
        let sp = Speech()
        let t = title.map { Expression.speakable($0) }, b = text.map { cap(sentence($0)) }
        if (t == nil || t!.isEmpty) && (b == nil || b!.isEmpty) { return sp }
        sp.add(t == nil || t!.isEmpty ? app + "." : t! + ", on " + app + ".", nil, nil, 150)
        if let b = b, !b.isEmpty { sp.add(b, Expression.of(text), nil, 0) }
        return sp
    }

    /// One message as at least one sentence, in words a voice can say.
    public static func sentence(_ text: String) -> String {
        let s = Expression.speakable(text)
        if s.isEmpty { return s }
        let c = s.jchar(s.jlength - 1)
        return c == 0x2E || c == 0x21 || c == 0x3F ? s : s + "."   // . ! ?
    }

    public static func cap(_ s0: String) -> String {
        let s = jtrim(s0.replaceAll("\\s+", " "))
        if s.jlength <= MAX_CHARS { return s }
        let cut = TCChar.lastIndexOf(s as NSString, " ", MAX_CHARS)
        return s.jsub(0, cut > 0 ? cut : MAX_CHARS) + ", and more."
    }

    // ---------------------------------------------------------------- other things worth hearing
    /// "You missed a call from Mom."
    public static func missedCall(_ who: String?) -> Speech {
        Speech().add("You missed a call from " + (who == nil || jtrim(who!).isEmpty ? "an unknown number" : jtrim(who!)) + ".",
                     Expression.CONFIDENT)
    }

    /// An alarm, a timer or a reminder: "Reminder: call the dentist." / "Timer: time's up." The title says what it is,
    /// the text adds the detail; a clock's bare "Alarm 7:00 AM" is read as it is.
    public static func reminder(_ title: String?, _ text: String?, _ app: String) -> Speech {
        let t = clean(title), b = clean(text)
        let what = t.isEmpty ? app : t
        let said = b.isEmpty || b.lowercased() == what.lowercased() ? what : what + ": " + b
        let low = said.lowercased()
        let head = low.hasPrefix("reminder") || low.hasPrefix("timer") || low.hasPrefix("alarm") ? "" : "Reminder: "
        return Speech().add(sentence(head + said), Expression.CONFIDENT.plus("a reminder: clear", ("rate", -0.1)))
    }

    /// A delivery or ride update: "Uber: Your driver is arriving."
    public static func update(_ title: String?, _ text: String?, _ app: String) -> Speech {
        let t = clean(title), b = clean(text)
        let sp = Speech().add(app + ".", nil, nil, 150)
        let said = t.isEmpty ? b : b.isEmpty || (b as NSString).hasPrefix(t) ? t : t + ". " + b
        return sp.add(cap(sentence(said)), Expression.of(said))
    }

    /// "Email from Priya. Contract signed. It starts: The signed copy is attached…" The subject, then the first sentence
    /// of the preview the notification itself shows (never more). sender: a name, or an address ("billing@acme.com" is
    /// said "Acme"). preview: the notification's expanded text (often "subject\nbody"), may be nil.
    public static func email(_ sender: String?, _ subject: String?, _ preview: String? = nil) -> Speech {
        var who = clean(senderName(sender))
        let sub = clean(subject)
        if who.isEmpty { who = "someone" }
        if let many = rx("(?i)^([0-9]+) new (messages|emails)$").find(who) {
            return Speech().add("You have " + many.group(1)! + " new emails.", nil, nil, 0)
        }
        let sp = Speech().add("Email from " + who + ".", nil, nil, 150)
        if !sub.isEmpty { sp.add(cap(sentence(sub)), Expression.of(sub), nil, 120) }
        let first = firstSentence(preview, subject)
        if !first.isEmpty { sp.add(first, Expression.of(first), nil, 0) }
        return sp
    }

    /// "Priya Sharma <priya@x.com>" -> "Priya Sharma"; "no-reply@accounts.google.com" -> "Google".
    public static func senderName(_ s: String?) -> String {
        guard let s = s else { return "" }
        let t = jtrim(s.replaceAll("<[^>]*>", "").replacingOccurrences(of: "\"", with: ""))
        // Java's \w is ASCII-only; ICU's is Unicode, so the class is spelled out
        let m = rx("^[A-Za-z0-9_.+-]+@([A-Za-z0-9_-]+\\.)*?([A-Za-z0-9_-]+)\\.[a-z]{2,}(\\.[a-z]{2})?$").find(t.lowercased())
        if let m = m, var org = m.group(2) {
            if org.matches("gmail|yahoo|outlook|hotmail|icloud|proton|protonmail") {
                let at = (t as NSString).range(of: "@").location
                org = t.jsub(0, at).replaceAll("[._-]+", " ")
            }
            var b = ""
            for w in jtrim(org).split(regex: " +") where !w.isEmpty {
                b += (b.isEmpty ? "" : " ") + w.jsub(0, 1).uppercased() + w.jsub(1)
            }
            return b
        }
        return t
    }

    /// The first sentence of an email preview, without the subject line it repeats, at most about 140 characters.
    public static func firstSentence(_ preview: String?, _ subject: String?) -> String {
        guard let preview = preview else { return "" }
        var p = jtrim(preview)
        if let subject = subject, !jtrim(subject).isEmpty, (p as NSString).hasPrefix(jtrim(subject)) { p = p.jsub(jtrim(subject).jlength) }
        p = jtrim(Expression.speakable(p.replaceAll("\\s+", " ")))
        if p.jlength < 12 { return "" }
        let sp = TextClean.spans(p)
        var f = sp.isEmpty ? p : p.jsub(sp[0].lowerBound, sp[0].upperBound)
        if f.jlength > 140 {
            let cut = TCChar.lastIndexOf(f as NSString, " ", 140)
            f = f.jsub(0, cut > 40 ? cut : 140) + "…"
        }
        return f
    }

    /// "Battery at 15 percent. Time to charge." / "Fully charged."
    public static func battery(_ level: Int, _ charging: Bool) -> Speech {
        if charging && level >= 100 { return Speech().add("Your phone is fully charged.", Expression.HAPPY) }
        return Speech().add("Battery at \(level) percent. Time to charge.", Expression.CONFIDENT.plus("low battery: clear", ("loud", 0.3)))
    }

    private static func clean(_ s: String?) -> String { s == nil ? "" : jtrim(Expression.speakable(s!)) }

    // ---------------------------------------------------------------- calls
    private static let REPEAT_WINDOW: Int64 = 15 * 60_000
    private static let ringsLock = NSLock()
    private static var rings: [String: [Int64]] = [:]

    /// How many times this caller has rung within the window, this call included. now: milliseconds.
    public static func ringCount(_ caller: String, _ now: Int64) -> Int {
        ringsLock.lock()
        defer { ringsLock.unlock() }
        let k = caller.lowercased()
        var d = rings[k] ?? []
        while let f = d.first, now - f > REPEAT_WINDOW { d.removeFirst() }
        d.append(now)
        rings[k] = d
        return d.count
    }

    /// "Incoming call from Mom." / "Mom is calling again." / "Mom is calling for the third time. It may be urgent."
    public static func call(_ caller: String, _ via: String?, _ count: Int) -> Speech {
        let how = via == nil ? "" : via! + " "
        if count <= 1 { return Speech().add("Incoming " + how + "call from " + caller + ".", Expression.CONFIDENT) }
        if count == 2 {
            return Speech().add(caller + " is calling again.", Expression.CONFIDENT.plus("again: more present", ("loud", 0.5), ("f0_med", 0.3)))
        }
        return Speech().add(caller + " is calling for the " + ordinal(count) + " time. It may be urgent.", Expression.URGENT)
    }

    private static func ordinal(_ n: Int) -> String {
        let w = ["", "first", "second", "third", "fourth", "fifth", "sixth"]
        return n < w.count ? w[n] : "\(n)th"
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
