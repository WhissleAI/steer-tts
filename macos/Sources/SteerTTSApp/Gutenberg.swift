import Foundation
import SteerText

/// Free books from Project Gutenberg (70,000+ public-domain books): shelves and search from its OPDS catalogue, as in
/// the Android app (FreeBooks). A book is downloaded as an EPUB and read through Extract.
enum Gutenberg {
    struct Book: Identifiable, Hashable {
        let id: String
        let title: String
        let author: String
        var source: String { "https://www.gutenberg.org/ebooks/" + id }
    }

    static let shelves: [(name: String, query: String)] = [
        ("Popular", ""), ("Children's", "bookshelf/20"), ("Fairy tales", "bookshelf/216"), ("Adventure", "s.adventure l.en"),
        ("Mystery", "s.detective l.en"), ("Science fiction", "s.science fiction l.en"), ("Poetry", "s.poetry l.en"),
        ("Humour", "s.humor l.en"), ("Classics", "s.fiction l.en"),
    ]

    /// One page (25 books) of a shelf or a search, most downloaded first.
    static func page(_ query: String, start: Int = 1) async throws -> [Book] {
        let url: String
        if query.hasPrefix("bookshelf/") {
            url = "https://www.gutenberg.org/ebooks/\(query).opds?sort_order=downloads&start_index=\(start)"
        } else {
            let q = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
            url = "https://www.gutenberg.org/ebooks/search.opds/?sort_order=downloads&start_index=\(start)"
                + (query.isEmpty ? "" : "&query=" + q)
        }
        return parse(try await get(url, max: 4 << 20))
    }

    /// The book's EPUB (without images, else with them).
    static func epub(_ b: Book) async throws -> Data {
        do {
            return try await get(b.source + ".epub.noimages", max: 64 << 20)
        } catch {
            return try await get(b.source + ".epub3.images", max: 64 << 20)
        }
    }

    private static let otherLanguage = "\\((German|Dutch|French|Spanish|Italian|Portuguese|Finnish|Swedish|"
        + "Danish|Norwegian|Esperanto|Latin|Hungarian|Polish|Czech|Greek|Russian|Chinese|Japanese|Tagalog|Catalan|Welsh|Icelandic|"
        + "Afrikaans|Frisian|Galician|Romanian|Bulgarian|Slovenian|Serbian|Croatian|Hebrew|Arabic)\\)\\s*$"

    /// The books in an OPDS page (entries that are not books, such as "sort by", have no numeric id).
    static func parse(_ xml: Data) -> [Book] {
        let s = String(decoding: xml, as: UTF8.self)
        var out: [Book] = []
        for e in rx("(?s)<entry>(.*?)</entry>").all(s) {
            let x = e.group(1) ?? ""
            guard let id = rx("/ebooks/(\\d+)\\.opds").find(x)?.group(1),
                  let t = rx("(?s)<title>(.*?)</title>").find(x)?.group(1) else { continue }
            let title = Extract.decode(t).replaceAll("\\s+", " ").trimmed
            var author = (rx("(?s)<content type=\"text\">(.*?)</content>").find(x)?.group(1)).map {
                Extract.decode($0).replaceAll("\\s+", " ").trimmed
            } ?? ""
            if author.matches("\\d+ downloads?") { author = "" }   // bookshelves put the download count where the author goes
            if rx(otherLanguage).find(title) != nil { continue }    // the voice reads English
            out.append(Book(id: id, title: title, author: author))
        }
        return out
    }

    private static func get(_ url: String, max: Int) async throws -> Data {
        var req = URLRequest(url: URL(string: url)!, timeoutInterval: 30)
        req.setValue("SteerTTS (Mac; free books from Project Gutenberg)", forHTTPHeaderField: "User-Agent")
        let (data, resp) = try await URLSession.shared.data(for: req)
        if let h = resp as? HTTPURLResponse, h.statusCode >= 400 {
            throw Extract.Unsupported("Project Gutenberg answered \(h.statusCode). Try again in a moment.")
        }
        if data.count > max { throw Extract.Unsupported("This book is too large to download.") }
        return data
    }
}
