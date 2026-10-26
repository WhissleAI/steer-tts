import Foundation
import SteerText

/// The reading list: Doc's storage (one JSON file per document, positions apart) with the list the sidebar shows, most
/// recently opened first. Full documents are loaded when opened and kept while the app runs.
@MainActor
final class Library: ObservableObject {
    let dir: URL
    @Published private(set) var docs: [Doc.Meta] = []
    private var loaded: [String: Doc] = [:]

    init(dir: URL) {
        self.dir = dir
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        reload()
    }

    func reload() {
        docs = Doc.list(in: dir).sorted { max($0.opened, $0.added) > max($1.opened, $1.added) }
    }

    func doc(_ id: String) -> Doc? {
        if let d = loaded[id] { return d }
        let d = Doc.load(id, in: dir)
        loaded[id] = d
        return d
    }

    /// Saves the reading position (cheap; called as reading moves on).
    func save(_ d: Doc) {
        d.savePos(in: dir)
        if let i = docs.firstIndex(where: { $0.id == d.id }) {
            var m = docs[i]
            m.pos = d.pos
            m.done = d.done
            m.opened = d.opened
            docs[i] = m
        }
    }

    /// Adds a document unless the list already has it (the same book, link or file); returns the one to open.
    func saveOrExisting(_ d: Doc) -> Doc {
        let id = d.saveOrExisting(in: dir)
        if id == d.id { loaded[id] = d }
        reload()
        return doc(id) ?? d
    }

    func find(source: String, titlePrefix: String?) -> Doc? {
        Doc.findSource(source, titlePrefix: titlePrefix, in: dir).flatMap { doc($0) }
    }

    func delete(_ id: String) {
        Doc.delete(id, in: dir)
        try? FileManager.default.removeItem(at: pdfURL(id))
        loaded[id] = nil
        reload()
    }

    /// PDFs keep their file, so the reader can show the pages with the line being read.
    func pdfURL(_ id: String) -> URL { dir.appendingPathComponent(id + ".pdf") }

    func copyPdf(_ src: URL, for d: Doc) {
        try? FileManager.default.removeItem(at: pdfURL(d.id))
        try? FileManager.default.copyItem(at: src, to: pdfURL(d.id))
    }

    /// What plays next after the current document: unfinished documents, not snippets.
    func next(after id: String?) -> Doc? {
        docs.first { $0.id != id && !$0.done && $0.kind != "text" }.flatMap { doc($0.id) }
    }
}
