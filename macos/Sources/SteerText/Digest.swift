import Foundation

/// Messages that arrived while they could not be read out (no headphones, Do Not Disturb), for "catch me up". The
/// words are held in memory only, never written to storage; only the messages' keys are saved (in `kept.keys` in the
/// directory given, if any), so after a restart the caller rebuilds the list from the notifications still showing
/// (restore). A message leaves when its notification is dismissed or opened (the user has seen it), when it has been
/// caught up on, or after twelve hours.
///
/// Android keeps one static digest and finds its file through a Context; here the caller owns an instance and passes
/// the directory (nil keeps nothing on disk). Thread-safe.
public final class Digest {
    public struct Item {
        public let key: String, person: String, app: String, text: String
        public let time: Int64   // milliseconds since 1970, when it was kept

        public init(key: String, person: String, app: String, text: String, time: Int64) {
            self.key = key
            self.person = person
            self.app = app
            self.text = text
            self.time = time
        }
    }

    private static let MAX_AGE: Int64 = 12 * 3600_000
    private static let MAX_ITEMS = 40

    private let lock = NSLock()
    private var items: [Item] = []
    private let store: URL?
    private var saved: [String: Int64] = [:]   // key -> when it was kept
    private let clock: () -> Int64

    /// Loads the saved keys (what was kept before a restart, for restore). directory: where `kept.keys` lives, nil for
    /// memory only; clock: the time in milliseconds (tests pass their own).
    public init(directory: URL? = nil,
                clock: @escaping () -> Int64 = { Int64((Date().timeIntervalSince1970 * 1000).rounded(.down)) }) {
        self.clock = clock
        store = directory?.appendingPathComponent("kept.keys")
        guard let store = store, let s = try? String(contentsOf: store, encoding: .utf8) else { return }   // nothing kept
        for line in s.components(separatedBy: "\n") {
            let tab = (line as NSString).range(of: "\t", options: .backwards).location
            if tab != NSNotFound && tab > 0, let t = Int64(line.jsub(tab + 1)) { saved[line.jsub(0, tab)] = t }
        }
    }

    /// Keys kept before a restart and not restored yet.
    public func savedKeys() -> Set<String> {
        lock.lock(); defer { lock.unlock() }
        return Set(saved.keys)
    }

    /// A message kept before a restart, read again from its notification (keeps its first time, not added twice).
    public func restore(_ key: String, _ person: String, _ app: String, _ text: String) {
        lock.lock(); defer { lock.unlock() }
        guard let t = saved[key] else { return }
        for i in items where i.key == key && i.text == text { return }
        items.append(Item(key: key, person: person, app: app, text: text, time: t))
    }

    /// After a restore: forget keys whose notifications are gone (seen or dismissed while the app was not running).
    public func forgetExcept(_ active: Set<String>) {
        lock.lock(); defer { lock.unlock() }
        saved = saved.filter { active.contains($0.key) }
        persist()
    }

    /// Only keys and times are written, never the words.
    private func persist() {
        guard let store = store else { return }
        var out = ""
        for (k, t) in saved { out += k + "\t" + String(t) + "\n" }
        try? FileManager.default.createDirectory(at: store.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? out.write(to: store, atomically: true, encoding: .utf8)
    }

    public func add(_ key: String, _ person: String, _ app: String, _ text: String) {
        lock.lock(); defer { lock.unlock() }
        let now = clock()
        items.append(Item(key: key, person: person, app: app, text: text, time: now))
        while items.count > Digest.MAX_ITEMS { items.removeFirst() }
        if saved[key] == nil {
            saved[key] = now
            persist()
        }
    }

    /// The notification is gone: the user has seen or dismissed it.
    public func removeKey(_ key: String) {
        lock.lock(); defer { lock.unlock() }
        items.removeAll { $0.key == key }
        if saved.removeValue(forKey: key) != nil { persist() }
    }

    public func size() -> Int {
        lock.lock(); defer { lock.unlock() }
        prune()
        return items.count
    }

    public func clear() {
        lock.lock(); defer { lock.unlock() }
        items.removeAll()
        saved.removeAll()
        persist()
    }

    private func prune() {
        let old = clock() - Digest.MAX_AGE
        items.removeAll { $0.time < old }
        let before = saved.count
        saved = saved.filter { $0.value >= old }
        if saved.count != before { persist() }
    }

    /// "You have 4 new messages, from Sam and Mom." then each person's messages, newest last, in their mood (and voice,
    /// when people have voices of their own). Empties the digest.
    public func take(_ personVoices: Bool, _ ownVoice: String?) -> Speech {
        lock.lock(); defer { lock.unlock() }
        prune()
        let sp = Speech()
        if items.isEmpty { return sp.add("You're all caught up. No new messages.") }
        var order: [String] = []                 // people in the order they first wrote
        var byPerson: [String: [Item]] = [:]
        for i in items {
            if byPerson[i.person] == nil { order.append(i.person) }
            byPerson[i.person, default: []].append(i)
        }
        let n = items.count
        sp.add((n == 1 ? "You have one new message" : "You have \(n) new messages") + ", from " + Digest.names(order) + ".",
               Expression.CONFIDENT, nil, 350)
        var people = 0
        for who in order {
            people += 1
            if people > 6 {
                sp.add("And more from \(order.count - 6) others.")
                break
            }
            let l = byPerson[who]!
            let last = l[max(0, l.count - 3)...]
            var said = ""
            for i in last { said += (said.isEmpty ? "" : " ") + Readout.sentence(i.text) }
            sp.add((l.count > last.count ? "The latest from " : "From ") + who + ".", nil, nil, 150)
            sp.add(said, Expression.of(said), personVoices ? Voices.forPerson(who, own: ownVoice) : nil, 400)
        }
        items.removeAll()
        saved.removeAll()
        persist()
        return sp
    }

    /// "Sam", "Sam and Mom", "Sam, Mom and Alex", "Sam, Mom and 3 others".
    public static func names(_ names: [String]) -> String {
        if names.count == 1 { return names[0] }
        if names.count <= 3 { return names.dropLast().joined(separator: ", ") + " and " + names[names.count - 1] }
        return names[0] + ", " + names[1] + " and \(names.count - 2) others"
    }
}
