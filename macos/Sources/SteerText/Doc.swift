import Foundation

/// Something to listen to: a title, its clean paragraphs (TextClean), the sentences they split into, and where the
/// listener is. Stored as one JSON file per document in a library directory, so the reading list, progress and resume
/// survive restarts; the original file is not kept. Ported from Doc.java: the Android Context's files/library becomes
/// a directory URL passed to each storage call; the files and their JSON fields are the same (plus an optional
/// "author", which Android folds into the title).
public final class Doc {
    public let id: String, title: String, kind: String, source: String
    /// The author, kept apart from the title ("" when unknown). Android has none: its title is "Title · Author".
    public let author: String
    public let paragraphs: [String]
    public let sentences: [String]
    public let spans: [[PdfLayout.Span]]?   // PDFs: where each paragraph's lines are on the pages (nil otherwise)
    public let regions: [[[Float]]]?        // PDFs: per sentence, boxes [page, l, t, r, b] as page fractions
    public let paraOf: [Int]                // the paragraph each sentence is in
    public let added: Int64                 // milliseconds since 1970, as Java's currentTimeMillis
    public var pos: Int                     // the sentence to read next
    public var opened: Int64
    public var done = false                 // heard to the end (pos is back at 0)
    public var isTransient = false          // read once and not kept (a screen being read): never saved, no position file

    private var cast_: Cast?
    private let castLock = NSLock()

    /// Who speaks where, and how (Cast.analyze on first use: call off the main thread for a long book).
    public func cast() -> Cast {
        castLock.lock()
        defer { castLock.unlock() }
        if let c = cast_ { return c }
        let c = Cast.analyze(paragraphs)
        cast_ = c
        return c
    }

    private init(id: String, title: String, author: String, kind: String, source: String, paragraphs: [String],
                 spans: [[PdfLayout.Span]]?, added: Int64, pos: Int, opened: Int64) {
        self.id = id
        self.title = title
        self.author = author
        self.kind = kind
        self.source = source
        self.paragraphs = paragraphs
        let sp = spans != nil && spans!.count == paragraphs.count ? spans : nil
        self.spans = sp
        var sentences: [String] = []
        var regions: [[[Float]]]? = sp != nil ? [] : nil
        var para: [Int] = []
        for (k, p) in paragraphs.enumerated() {
            for r in TextClean.spans(p) {
                sentences.append(p.jsub(r.lowerBound, r.upperBound))
                para.append(k)
                if regions != nil { regions!.append(Doc.boxes(sp![k], r.lowerBound, r.upperBound)) }
            }
        }
        self.sentences = sentences
        self.regions = regions
        self.paraOf = para
        self.added = added
        self.pos = max(0, min(pos, max(0, sentences.count - 1)))
        self.opened = opened
    }

    /// A new document (Java's Doc.create; author is new). source: where it came from, nil or "" for typed text.
    public convenience init(title: String, author: String? = nil, kind: String, source: String?, paragraphs: [String]) {
        let now = Doc.now()
        self.init(id: UUID().uuidString.lowercased(), title: title, author: author ?? "", kind: kind, source: source ?? "",
                  paragraphs: paragraphs, spans: nil, added: now, pos: 0, opened: now)
    }

    public static func create(title: String, kind: String, source: String?, paragraphs: [String]) -> Doc {
        Doc(title: title, kind: kind, source: source, paragraphs: paragraphs)
    }

    /// From what Extract made of a file, a page or some text.
    public static func create(from e: ExtractedDoc) -> Doc {
        Doc(title: e.title, author: e.author, kind: e.kind, source: e.source, paragraphs: e.paragraphs)
    }

    /// Something to read now and not keep (the text of a screen).
    public static func createTransient(title: String, paragraphs: [String]) -> Doc {
        let now = Doc.now()
        let d = Doc(id: "screen-\(now)", title: title, author: "", kind: "screen", source: "", paragraphs: paragraphs,
                    spans: nil, added: now, pos: 0, opened: now)
        d.isTransient = true
        return d
    }

    public static func createPdf(id: String, title: String, source: String, paragraphs: [String], spans: [[PdfLayout.Span]]) -> Doc {
        let now = Doc.now()
        return Doc(id: id, title: title, author: "", kind: "pdf", source: source, paragraphs: paragraphs, spans: spans,
                   added: now, pos: 0, opened: now)
    }

    static func now() -> Int64 { Int64((Date().timeIntervalSince1970 * 1000).rounded(.down)) }

    /// The title as shown: "Title · Author" when there is an author.
    public var displayTitle: String { author.isEmpty ? title : title + " · " + author }

    /// The part of each line a sentence covers (characters [a, b) of its paragraph), as page boxes.
    public static func boxes(_ lines: [PdfLayout.Span], _ a: Int, _ b: Int) -> [[Float]] {
        var out: [[Float]] = []
        for l in lines {
            let s = max(a, l.start), e = min(b, l.end)
            if e <= s { continue }
            let n = Float(max(1, l.end - l.start)), w = l.r - l.l
            out.append([Float(l.page), l.l + w * Float(s - l.start) / n, l.t, l.l + w * Float(e - l.start) / n, l.b])
        }
        return out
    }

    /// The first page this sentence is on (PDFs), or -1.
    public func pageOf(_ sentence: Int) -> Int {
        guard let r = regions, sentence >= 0, sentence < r.count, let first = r[sentence].first else { return -1 }
        return Int(first[0])
    }

    public func progress() -> Float { done ? 1 : sentences.isEmpty ? 0 : min(1, Float(pos) / Float(sentences.count)) }

    /// What the reading list shows, without the text: read from a small <id>.meta file and the position file, so listing
    /// the library never parses whole books (that would block the screen with a long list).
    public struct Meta {
        public let id: String, title: String, kind: String
        public let added: Int64
        public var author = ""
        public var source = ""   // where it came from (a link, a file, a Gutenberg book): the same source is the same document
        public var opened: Int64 = 0
        public var pos = 0, count: Int
        public var chars: Int64
        public var done = false

        public init(id: String, title: String, kind: String, added: Int64, count: Int, chars: Int64) {
            self.id = id
            self.title = title
            self.kind = kind
            self.added = added
            self.count = count
            self.chars = chars
        }

        public var displayTitle: String { author.isEmpty ? title : title + " · " + author }

        public func progress() -> Float { done ? 1 : count == 0 ? 0 : min(1, Float(pos) / Float(count)) }

        public func minutesLeft(_ pace: Float) -> Int {
            let left = Float(chars) * (1 - progress())
            return max(1, javaRound(left / 15 / max(0.5, pace) / 60))
        }
    }

    public func meta() -> Meta {
        var chars: Int64 = 0
        for x in sentences { chars += Int64(x.jlength) }
        var m = Meta(id: id, title: title, kind: kind, added: added, count: sentences.count, chars: chars)
        m.author = author
        m.source = source
        m.opened = opened
        m.pos = pos
        m.done = done
        return m
    }

    private static func writeMeta(_ dir: URL, _ m: Meta) {
        var o: [String: Any] = ["title": m.title, "kind": m.kind, "added": m.added, "count": m.count,
                                "chars": m.chars, "source": m.source]
        if !m.author.isEmpty { o["author"] = m.author }
        write(o, to: dir.appendingPathComponent(m.id + ".meta"))
    }

    /// The library, most recently opened first, as metadata (cheap).
    public static func list(in library: URL) -> [Meta] {
        let dir = ensure(library)
        var out: [Meta] = []
        for id in ids(dir) {
            var m: Meta? = nil
            if let o = readJson(dir.appendingPathComponent(id + ".meta")), let title = o["title"] as? String {
                var x = Meta(id: id, title: title, kind: o["kind"] as? String ?? "text", added: int64(o["added"]),
                             count: Int(int64(o["count"])), chars: int64(o["chars"]))
                x.source = o["source"] as? String ?? ""
                x.author = o["author"] as? String ?? ""
                m = x
            }
            if m == nil {   // a document saved before metadata files existed: read it once
                guard let d = load(id, in: dir) else { continue }
                m = d.meta()
                writeMeta(dir, m!)
            }
            readPos(dir, &m!)
            out.append(m!)
        }
        return stableSorted(out) { $0.opened > $1.opened }
    }

    /// "<pos> <opened> <done>" from <id>.pos; fields stop at the first that does not parse, as Java's.
    private static func readPos(_ dir: URL, _ id: String) -> (pos: Int?, opened: Int64?, done: Bool?) {
        guard let d = try? Data(contentsOf: dir.appendingPathComponent(id + ".pos")),
              let s = String(data: d, encoding: .utf8) else { return (nil, nil, nil) }
        let p = s.trimmingCharacters(in: .whitespacesAndNewlines).components(separatedBy: " ")
        guard let pos = Int(p[0]) else { return (nil, nil, nil) }
        var opened: Int64? = nil
        if p.count > 1 {
            guard let o = Int64(p[1]) else { return (pos, nil, nil) }
            opened = o
        }
        return (pos, opened, p.count > 2 ? p[2] == "1" : nil)
    }

    private static func readPos(_ dir: URL, _ m: inout Meta) {
        let r = readPos(dir, m.id)
        if let p = r.pos { m.pos = p }
        if let o = r.opened { m.opened = o }
        if let d = r.done { m.done = d }
    }

    /// The reading queue: documents not finished, the oldest added first (what plays next after one ends).
    public static func upNext(in library: URL, except exceptId: String?) -> [Meta] {
        stableSorted(list(in: library).filter { !$0.done && $0.id != exceptId }) { $0.added < $1.added }
    }

    /// Rough listening time left, from the characters still to read (about 15 characters a second at 1x).
    public func minutesLeft(_ pace: Float) -> Int {
        var chars: Int64 = 0
        if pos < sentences.count { for i in pos..<sentences.count { chars += Int64(sentences[i].jlength) } }
        return max(1, javaRound(Float(chars) / 15 / max(0.5, pace) / 60))
    }

    // ---------------------------------------------------------------- store
    private static func ensure(_ dir: URL) -> URL {
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static func ids(_ dir: URL) -> [String] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        return names.filter { $0.hasSuffix(".json") }.map { String($0.dropLast(5)) }
    }

    /// Saves this document unless the list already has it, and returns the id to open. The same book, link or file opens
    /// the copy already there, where it was left; the same copied or typed text replaces its older copy (heard again
    /// from the top) instead of piling up.
    @discardableResult
    public func saveOrExisting(in library: URL) -> String {
        let mine = meta()
        for m in Doc.list(in: library) {
            let same = !mine.source.isEmpty
                ? mine.source == m.source
                    || (m.source.isEmpty && m.kind == kind && m.displayTitle == displayTitle)   // saved before sources were kept
                : m.kind == kind && m.displayTitle == displayTitle && m.chars == mine.chars
            if !same { continue }
            if kind != "text" { return m.id }
            Doc.delete(m.id, in: library)
            break
        }
        save(in: library)
        return id
    }

    /// The id of a document in the list that came from this source (or has this title, for older copies), or nil.
    public static func findSource(_ source: String, titlePrefix: String?, in library: URL) -> String? {
        for m in list(in: library)
            where source == m.source || (m.source.isEmpty && titlePrefix != nil && m.displayTitle.hasPrefix(titlePrefix!)) {
            return m.id
        }
        return nil
    }

    /// The document as <id>.json (atomically) and its <id>.meta.
    public func save(in library: URL) {
        let dir = Doc.ensure(library)
        var o: [String: Any] = ["id": id, "title": title, "kind": kind, "source": source, "added": added, "pos": pos,
                                "opened": opened, "paragraphs": paragraphs]
        if !author.isEmpty { o["author"] = author }
        if let spans = spans {
            o["spans"] = spans.map { ps in
                ps.map { x -> [Any] in [x.page, x.start, x.end, Double(x.l), Double(x.t), Double(x.r), Double(x.b)] }
            }
        }
        Doc.write(o, to: dir.appendingPathComponent(id + ".json"))
        Doc.writeMeta(dir, meta())
    }

    /// Just the position (cheap, called as reading moves on).
    public func savePos(in library: URL) {
        if isTransient { return }
        let dir = Doc.ensure(library)
        try? Data("\(pos) \(opened) \(done ? 1 : 0)".utf8).write(to: dir.appendingPathComponent(id + ".pos"), options: .atomic)
    }

    public static func load(_ id: String, in library: URL) -> Doc? {
        let dir = library
        guard let o = readJson(dir.appendingPathComponent(id + ".json")),
              let a = o["paragraphs"] as? [Any], let docId = o["id"] as? String, let title = o["title"] as? String
        else { return nil }
        var paras: [String] = []
        for p in a {
            guard let s = p as? String else { return nil }
            paras.append(s)
        }
        var pos = Int(int64(o["pos"]))
        var opened = int64(o["opened"])
        var done = false
        let r = readPos(dir, id)
        if let p = r.pos { pos = p }
        if let x = r.opened { opened = x }
        if let d = r.done { done = d }
        var spans: [[PdfLayout.Span]]? = nil
        if let sa = o["spans"] as? [Any] {
            var all: [[PdfLayout.Span]] = []
            for pa in sa {
                guard let pa = pa as? [Any] else { return nil }
                var ps: [PdfLayout.Span] = []
                for x in pa {
                    guard let x = x as? [NSNumber], x.count >= 7 else { return nil }
                    ps.append(PdfLayout.Span(page: x[0].intValue, start: x[1].intValue, end: x[2].intValue,
                                             l: x[3].floatValue, t: x[4].floatValue, r: x[5].floatValue, b: x[6].floatValue))
                }
                all.append(ps)
            }
            spans = all
        }
        let d = Doc(id: docId, title: title, author: o["author"] as? String ?? "", kind: o["kind"] as? String ?? "text",
                    source: o["source"] as? String ?? "", paragraphs: paras, spans: spans, added: int64(o["added"]),
                    pos: pos, opened: opened)
        d.done = done
        return d
    }

    /// Removes the document's files (Android also drops its PDF text cache, which the Mac app keeps elsewhere if at all).
    public static func delete(_ id: String, in library: URL) {
        for ext in [".json", ".pos", ".meta"] {
            try? FileManager.default.removeItem(at: library.appendingPathComponent(id + ext))
        }
    }

    /// Everything in the library, most recently opened first (light: titles and progress only need the full load).
    public static func all(in library: URL) -> [Doc] {
        let dir = ensure(library)
        return stableSorted(ids(dir).compactMap { load($0, in: dir) }) { $0.opened > $1.opened }
    }

    // ---------------------------------------------------------------- JSON helpers
    private static func write(_ o: [String: Any], to url: URL) {
        guard let d = try? JSONSerialization.data(withJSONObject: o, options: [.withoutEscapingSlashes]) else { return }
        try? d.write(to: url, options: .atomic)   // a temporary file renamed over it, as Android's .tmp
    }

    private static func readJson(_ url: URL) -> [String: Any]? {
        guard let d = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: d)) as? [String: Any]
    }
}

/// org.json's optLong: a number (or numeric string) else 0.
fileprivate func int64(_ v: Any?) -> Int64 {
    switch v {
    case let n as NSNumber: return n.int64Value
    case let s as String: return Int64(s) ?? Int64(Double(s) ?? 0)
    default: return 0
    }
}

/// Math.round(float): half up.
fileprivate func javaRound(_ x: Float) -> Int { Int((x + 0.5).rounded(.down)) }

/// A stable sort, as Java's List.sort.
fileprivate func stableSorted<T>(_ a: [T], by less: (T, T) -> Bool) -> [T] {
    a.enumerated().sorted { less($0.element, $1.element) || (!less($1.element, $0.element) && $0.offset < $1.offset) }
        .map { $0.element }
}
