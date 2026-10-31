import Foundation

/// Where the voice library keeps the user's favourites (Android: the "voices" SharedPreferences). UserDefaults
/// conforms; tests use an in-memory one.
public protocol VoicesStore: AnyObject {
    func stringSet(forKey key: String) -> Set<String>
    func setStringSet(_ value: Set<String>, forKey key: String)
}

extension UserDefaults: VoicesStore {
    public func stringSet(forKey key: String) -> Set<String> { Set(stringArray(forKey: key) ?? []) }
    public func setStringSet(_ value: Set<String>, forKey key: String) { set(value.sorted(), forKey: key) }
}

/// A VoicesStore in memory (tests, previews).
public final class MemoryVoicesStore: VoicesStore {
    private var sets: [String: Set<String>] = [:]
    public init() {}
    public func stringSet(forKey key: String) -> Set<String> { sets[key] ?? [] }
    public func setStringSet(_ value: Set<String>, forKey key: String) { sets[key] = value }
}

/// The voice library: every voice of the model with a human name and a plain description from its measured pitch
/// register and pace (the model's voices.json), a featured set, and the user's favourites. Names are assigned in voice
/// order from fixed lists, so a voice keeps its name across installs. Ported from Voices.java (the singleton over an
/// Android Context becomes an instance made from the voices.json URL and a store).
public final class Voices {
    public struct Voice {
        public let id: String, gender: String, pitch: String, pace: String, name: String

        public init(id: String, gender: String, pitch: String, pace: String, name: String) {
            self.id = id
            self.gender = gender
            self.pitch = pitch
            self.pace = pace
            self.name = name
        }

        /// "Deep, relaxed" from the measured register and pace.
        public func describe() -> String {
            let p = pitch == "low" ? "Deep" : pitch == "high" ? "Bright" : "Warm"
            let s = pace == "slow" ? "relaxed" : pace == "fast" ? "brisk" : "steady"
            return p + ", " + s
        }

        /// "Female" / "Male" (Java's gender()).
        public var genderLabel: String { gender == "female" ? "Female" : "Male" }
    }

    // featured: both defaults, then a spread of registers and paces
    public static let FEATURED = ["voice_1688", "voice_2500", "voice_0010", "voice_1469", "voice_0185", "voice_0413",
                                  "voice_1226", "voice_0213"]

    private static let FEMALE = ["Ava", "Mia", "Zoe", "Lily", "Emma", "Nora", "Ruby", "Iris", "Maya", "Leah",
        "Clara", "Ella", "Grace", "Hazel", "Ivy", "Jade", "Kate", "Lena", "Luna", "Mila", "Nina", "Olive", "Paige",
        "Quinn", "Rose", "Sara", "Tessa", "Vera", "Willow", "Yara", "Abby", "Beth", "Cora", "Dana", "Eva", "Faith",
        "Gemma", "Hope", "Isla", "June", "Kira", "Lucy", "Molly", "Nadia", "Opal", "Piper", "Rhea", "Sofia", "Tara",
        "Una", "Violet", "Wren", "Alice", "Brooke", "Chloe", "Daisy", "Eden", "Fiona", "Gwen", "Holly", "Ingrid",
        "Joy", "Kylie", "Laura", "Megan", "Naomi", "Olivia", "Penny", "Riley", "Stella", "Tina", "Uma", "Val",
        "Wendy", "Amber", "Bella", "Celia", "Diana", "Elise", "Freya", "Greta", "Heidi", "Irene", "Jenna", "Kelly",
        "Lara", "Martha", "Nell", "Orla", "Phoebe", "Rachel", "Sienna", "Thea", "Ursula", "Vivian", "Zara",
        "Anna", "Bonnie", "Carmen", "Della", "Erin", "Flora", "Gloria", "Hannah", "Isabel"]
    private static let MALE = ["Leo", "Max", "Sam", "Noah", "Eli", "Owen", "Jack", "Theo", "Finn", "Luke",
        "Adam", "Ben", "Cole", "Dan", "Evan", "Felix", "Gus", "Hugo", "Ian", "Jude", "Kai", "Liam", "Milo", "Nate",
        "Oscar", "Paul", "Reed", "Seth", "Tom", "Wade", "Alex", "Blake", "Carl", "Dean", "Ezra", "Frank", "Grant",
        "Hank", "Isaac", "Jake", "Kyle", "Lance", "Mark", "Neil", "Otto", "Pete", "Ray", "Scott", "Troy", "Vince",
        "Will", "Aaron", "Brian", "Chris", "Derek", "Eric", "Fred", "Gary", "Henry", "Ivan", "Jason", "Kevin",
        "Leon", "Mason", "Nick", "Omar", "Peter", "Ross", "Simon", "Tyler", "Victor", "Wes", "Andy", "Bruce",
        "Caleb", "David", "Edward", "Floyd", "George", "Harry", "Jasper", "Keith", "Louis", "Martin", "Nolan",
        "Oliver", "Philip", "Ralph", "Steve", "Tony", "Walter", "Arthur", "Bryce", "Connor", "Dylan", "Elliot",
        "Gavin", "Harvey", "Jonah", "Kurt", "Logan", "Miles", "Nigel", "Patrick", "Roman", "Silas", "Toby"]

    private var order: [Voice] = []
    private var byId: [String: Voice] = [:]
    private let store: VoicesStore

    /// The library from the model's voices.json ({"voices": [{"name", "gender", "pitch_register", "pace"}, ...]}).
    /// A file that cannot be read gives an empty library (isEmpty: the caller may retry).
    public convenience init(voicesJSON url: URL, store: VoicesStore) {
        self.init(data: (try? Data(contentsOf: url)) ?? Data(), store: store)
    }

    public init(data: Data, store: VoicesStore) {
        self.store = store
        guard let o = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let a = o["voices"] as? [[String: Any]] else { return }
        var nf = 0, nm = 0
        for v in a {
            guard let id = v["name"] as? String else { break }   // getString throws in Java: the load stops there
            let g = v["gender"] as? String ?? "female"
            let name: String
            if g == "female" { name = Voices.FEMALE[nf % Voices.FEMALE.count]; nf += 1 }
            else { name = Voices.MALE[nm % Voices.MALE.count]; nm += 1 }
            let voice = Voice(id: id, gender: g, pitch: v["pitch_register"] as? String ?? "mid",
                              pace: v["pace"] as? String ?? "average", name: name)
            if byId[id] == nil { order.append(voice) } else if let i = order.firstIndex(where: { $0.id == id }) { order[i] = voice }
            byId[id] = voice
        }
    }

    public var isEmpty: Bool { byId.isEmpty }

    public func all() -> [Voice] { order }

    public func voice(_ id: String) -> Voice? { byId[id] }

    /// A voice id, "female" / "male" (the defaults) or nil, as the name people see ("Ava").
    public func nameOf(_ id: String?) -> String {
        var id = id ?? Voices.FEATURED[0]
        if id == "female" { id = Voices.FEATURED[0] } else if id == "male" { id = Voices.FEATURED[1] }
        return byId[id]?.name ?? id
    }

    public func featured() -> [Voice] { Voices.FEATURED.compactMap { byId[$0] } }

    public func favourites() -> Set<String> { store.stringSet(forKey: "fav") }

    public func isFavourite(_ id: String) -> Bool { favourites().contains(id) }

    public func toggleFavourite(_ id: String) {
        var s = favourites()
        if s.remove(id) == nil { s.insert(id) }
        store.setStringSet(s, forKey: "fav")
    }

    public static func featuredIds() -> [String] { FEATURED }

    /// A voice of their own for a person whose messages are read out: always the same for the same name, one of the
    /// featured voices, never the user's own (so the assistant's framing and the message are told apart).
    public static func forPerson(_ name: String, own: String?) -> String {
        let ownId = own == nil || own == "female" ? FEATURED[0] : own == "male" ? FEATURED[1] : own!
        let pool = FEATURED.filter { $0 != ownId }
        let key = jtrim(name).lowercased()
        var h: Int32 = 0
        for u in key.utf16 { h = 31 &* h &+ Int32(u) }   // String.hashCode, spelled out: stable
        let n = Int32(pool.count)
        return pool[Int(((h % n) + n) % n)]   // Math.floorMod
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
