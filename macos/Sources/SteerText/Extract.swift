import Foundation

/// What Extract makes of a file, a page or some text: the paragraphs to read and what to call them. Storage (Doc)
/// builds its record from this.
public struct ExtractedDoc: Equatable {
    /// The document's own title (EPUB/DOCX metadata, an HTML <title>, a page's og:title), else the file name without
    /// its extension, else the first words of the text.
    public var title: String
    /// The EPUB's dc:creator, when it has one (other formats: nil).
    public var author: String?
    /// Clean paragraphs (TextClean), which may carry Narration's emphasis and heading markers (U+E000, U+E001, U+E002).
    public var paragraphs: [String]
    /// "text", "web", "html", "markdown", "epub" or "docx" (the app makes "pdf" documents itself, from PdfLayout).
    public var kind: String
    /// Where it came from: the file or page URL as a string, nil for pasted text.
    public var source: String?

    public init(title: String, author: String? = nil, paragraphs: [String], kind: String, source: String?) {
        self.title = title
        self.author = author
        self.paragraphs = paragraphs
        self.kind = kind
        self.source = source
    }

    /// The title as the Android app stores it: "Title · Author" when there is an author.
    public var displayTitle: String {
        if let a = author, !a.isEmpty { return title + " · " + a }
        return title
    }
}

/// A file, a page or plain text -> an ExtractedDoc to listen to. Formats: EPUB (the spine's XHTML chapters, in reading
/// order), DOCX (the body's paragraphs), HTML, Markdown and plain text; a fetched web page's article is read through
/// Article. Everything goes through TextClean. PDFs are not read here: the app reads the page's lines and boxes
/// (PDFKit) and hands them to PdfLayout.
///
/// Ported from the Android app's Extract.java; the ContentResolver/Uri parts are replaced by file URLs and Data.
public enum Extract {
    public static let mimeTypes = ["application/pdf", "application/epub+zip",
                                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain",
                                   "text/html", "text/markdown", "text/x-markdown"]

    /// A failure with a message for the user ("This EPUB has no text to read.").
    public struct Unsupported: Error, LocalizedError, Equatable {
        public let message: String
        public init(_ message: String) { self.message = message }
        public var errorDescription: String? { message }
    }

    public static func fromText(_ text: String, title: String? = nil) -> ExtractedDoc {
        let paras = TextClean.clean(text)
        return ExtractedDoc(title: title ?? firstWords(paras), paragraphs: paras, kind: "text", source: nil)
    }

    /// A fetched web page (the app fetches; Article.isLink/link(in:) tell it what): its article, one paragraph per line.
    public static func fromHTML(_ html: String, baseURL: URL?) -> ExtractedDoc {
        fromArticle(Article.extract(html, url: baseURL), source: baseURL?.absoluteString)
    }

    /// An Article as a document (Extract.fromUrl in the Java, after the fetch).
    public static func fromArticle(_ a: Article, source: String?) -> ExtractedDoc {
        var paras: [String] = []
        for p in a.text.components(separatedBy: "\n") where !TCChar.trim(p).isEmpty { paras.append(TextClean.speakable(TCChar.trim(p))) }
        return ExtractedDoc(title: a.title ?? firstWords(paras), paragraphs: paras, kind: "web", source: source)
    }

    /// A file on disk, by its extension (and an optional MIME type). PDFs throw Unsupported: the app reads them.
    public static func document(fromFile url: URL, mime: String? = nil) throws -> ExtractedDoc {
        if let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize, size > 200 << 20 {
            throw Unsupported("This file is too large to read.")
        }
        let name = url.lastPathComponent.isEmpty ? "Document" : url.lastPathComponent
        let ext = (name as NSString).pathExtension.lowercased()
        if mime == "application/pdf" || ext == "pdf" { throw Unsupported("PDFs are read from their pages (PdfLayout).") }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return try document(data: data, name: name, mime: mime, source: url.absoluteString)
    }

    /// A file's bytes, with its name (for the extension and the fallback title) and an optional MIME type.
    public static func document(data: Data, name: String, mime: String? = nil, source: String? = nil) throws -> ExtractedDoc {
        let dot = (name as NSString).range(of: ".", options: .backwards).location
        let ext = dot != NSNotFound ? name.jsub(dot + 1).lowercased() : ""
        let title = dot != NSNotFound ? name.jsub(0, dot) : name
        if mime == "application/pdf" || ext == "pdf" { throw Unsupported("PDFs are read from their pages (PdfLayout).") }
        if data.count > 200 << 20 { throw Unsupported("This file is too large to read.") }
        if mime == "application/epub+zip" || ext == "epub" { return try epub(data, fallbackTitle: title, source: source) }
        if ext == "docx" || (mime?.contains("wordprocessingml") ?? false) { return try docx(data, fallbackTitle: title, source: source) }
        let text = String(decoding: data, as: UTF8.self)
        if ext == "html" || ext == "htm" || ext == "xhtml" || (mime?.contains("html") ?? false) {
            let t = htmlTitle(text)
            return ExtractedDoc(title: t ?? title, paragraphs: TextClean.clean(htmlToText(text)), kind: "html", source: source)
        }
        if ext == "md" || ext == "markdown" || (mime?.contains("markdown") ?? false) {
            return ExtractedDoc(title: title, paragraphs: TextClean.clean(markdownToText(text)), kind: "markdown", source: source)
        }
        if let m = mime, !m.hasPrefix("text/"), ext != "", ext != "txt" {
            throw Unsupported("SteerTTS can read PDF, EPUB, Word (.docx), web pages and text files.")
        }
        return ExtractedDoc(title: title, paragraphs: TextClean.clean(text), kind: "text", source: source)
    }

    // ---------------------------------------------------------------- EPUB
    /// An EPUB: container.xml -> the OPF package -> the spine's chapters in order, each read as HTML; the book's
    /// wrapping (Gutenberg licence, contents, title page lines) handled by TextClean.bookBody. Title and author from
    /// the OPF's metadata. A damaged ZIP throws ZipArchive.Failure.
    public static func epub(_ data: Data, fallbackTitle: String, source: String?) throws -> ExtractedDoc {
        let files = try unzip(data)
        let container = str(files["META-INF/container.xml"])
        guard let opfPath = first(container, "full-path=\"([^\"]+)\"") else { throw Unsupported("This EPUB could not be opened.") }
        let opf = str(files[opfPath])
        let slash = (opfPath as NSString).range(of: "/", options: .backwards).location
        let base = slash != NSNotFound ? opfPath.jsub(0, slash + 1) : ""
        var manifest: [String: String] = [:]
        for item in rx("<item\\b[^>]*>").all(opf) {
            let tag = item.group() ?? ""
            if let id = first(tag, "\\bid=\"([^\"]+)\""), let href = first(tag, "\\bhref=\"([^\"]+)\"") { manifest[id] = href }
        }
        var paras: [String] = []
        for ref in rx("<itemref\\b[^>]*idref=\"([^\"]+)\"").all(opf) {
            guard let href = manifest[ref.group(1) ?? ""] else { continue }
            if let x = files[resolve(base, href.removingPercentEncoding ?? href)] {
                paras.append(contentsOf: TextClean.clean(htmlToText(str(x))))
            }
        }
        paras = TextClean.bookBody(paras)
        if paras.isEmpty { throw Unsupported("This EPUB has no text to read.") }
        let title = clean(first(opf, "<dc:title[^>]*>([\\s\\S]*?)</dc:title>"))
        let author = clean(first(opf, "<dc:creator[^>]*>([\\s\\S]*?)</dc:creator>"))
        let t = title.map { $0.isEmpty ? fallbackTitle : $0 } ?? fallbackTitle
        return ExtractedDoc(title: t, author: author?.isEmpty == false ? author : nil, paragraphs: paras, kind: "epub", source: source)
    }

    private static func resolve(_ base: String, _ href: String) -> String {
        let p = base + href
        var out: [String] = []
        for s in p.components(separatedBy: "/") {
            if s == ".." { if !out.isEmpty { out.removeLast() } }
            else if s != "." && !s.isEmpty { out.append(s) }
        }
        let r = out.joined(separator: "/")
        if let hash = r.firstIndex(of: "#") { return String(r[..<hash]) }
        return r
    }

    // ---------------------------------------------------------------- DOCX
    /// A Word file: the paragraphs of word/document.xml (runs' text, tabs and breaks as spaces), title from
    /// docProps/core.xml. A damaged ZIP throws ZipArchive.Failure.
    public static func docx(_ data: Data, fallbackTitle: String, source: String?) throws -> ExtractedDoc {
        let files = try unzip(data)
        let doc = str(files["word/document.xml"])
        if doc.isEmpty { throw Unsupported("This Word file could not be opened.") }
        var text = ""
        for p in doc.components(separatedBy: "</w:p>") {
            var para = ""
            for m in rx("<w:t(?:\\s[^>]*)?>([^<]*)</w:t>|<w:tab/>|<w:br/>").all(p) {
                para += m.group(1).map(decode) ?? " "
            }
            if !TCChar.trim(para).isEmpty { text += para + "\n\n" }
        }
        let title = clean(first(str(files["docProps/core.xml"]), "<dc:title>([\\s\\S]*?)</dc:title>"))
        return ExtractedDoc(title: title.map { $0.isEmpty ? fallbackTitle : $0 } ?? fallbackTitle,
                            paragraphs: TextClean.clean(text), kind: "docx", source: source)
    }

    // ---------------------------------------------------------------- HTML / Markdown
    /// Visible text of an HTML document, one block element per paragraph.
    public static func htmlToText(_ html: String) -> String {
        var h = html.replaceAll("(?is)<(script|style|noscript|svg|head|nav|header|footer|aside|form|figure|sup)\\b.*?</\\1>", " ")
        h = contentsRemoved(h)
        // pictures: a drop cap's letter is kept ("I" + "t is…"), a caption or alt text is not read
        h = h.replaceAll("(?is)<img\\b[^>]*\\balt=\"([A-Za-z])\"[^>]*>", "$1").replaceAll("(?is)<img\\b[^>]*>", "")
        h = rx("(?is)<span\\b[^>]*\\bid=\"img_[^\"]*\"[^>]*>(.*?)</span>").replaceAll(h) { m in
            let g = TCChar.trim(m.group(1) ?? "")
            return g.jlength == 1 ? g : ""
        }
        h = h.replaceAll("(?is)<span\\b[^>]*class=\"[^\"]*pageno[^\"]*\"[^>]*>.*?</span>", "")
        h = h.replaceAll("(?i)<br\\s*/?>", "\n")
        // italics and bold become stress markers, a heading starts with a heading marker (Narration reads both)
        h = h.replaceAll("(?i)<(em|i|strong|b)\\b[^>]*>", TextMarkers.on)
        h = h.replaceAll("(?i)</(em|i|strong|b)\\s*>", TextMarkers.off)
        h = h.replaceAll("(?i)<h[1-6]\\b[^>]*>", "\n\n" + TextMarkers.head)
        h = h.replaceAll("(?i)</?(p|div|h[1-6]|li|blockquote|section|article|tr|dt|dd|pre|title|table|ul|ol|dl|hr|caption|figcaption|center|body)\\b[^>]*>", "\n\n")
        h = h.replaceAll("(?i)</?(td|th)\\b[^>]*>", " ")
        h = h.replaceAll("(?s)<[^>]*>", "")   // inline tags (span, a, small caps) join their text: “<span>He</span>” is “He”
        let t = decode(h).replaceAll("[ \\t]+", " ").replaceAll("\\n[ \\t]+", "\n").replaceAll("\\n{3,}", "\n\n")
        // a drop cap set apart from its word: "I" / "t is a little remarkable" -> "It is a little remarkable"
        return t.replaceAll("(?m)^([“\"‘']?[A-Z])[ \\t]*\\n+[ \\t]*(?=[a-z])", "$1")
    }

    private static let pageLink =
        "(?is)<a\\b[^>]*href=\"[^\"]*#[^\"]*\"[^>]*>\\s*(?:page\\s*)?[0-9ivxlcdmIVXLCDM]{1,8}\\.?\\s*</a>"
    private static let internalLink = "(?is)<a\\b[^>]*href=\"[^\"]*#[^\"]*\"[^>]*>.*?</a>"

    /// A table of contents or list of illustrations, which nobody wants read aloud: table rows that link to a page
    /// ("A Gleam of Sunshine … 249"), and paragraphs or list items that are only links within the book.
    public static func contentsRemoved(_ html: String) -> String {
        let h = rx("(?is)<tr\\b[^>]*>.*?</tr>").replaceAll(html) { m in
            let g = m.group() ?? ""
            return rx(pageLink).find(g) != nil ? "" : g
        }
        return rx("(?is)<(p|li|dt|dd)\\b[^>]*>(.*?)</\\1>").replaceAll(h) { m in
            let whole = m.group() ?? "", inner = m.group(2) ?? ""
            if rx(internalLink).find(inner) == nil { return whole }
            let rest = rx(internalLink).replaceAll(inner, "").replaceAll("<[^>]*>", "").replaceAll("&[a-z]+;|[\\s.,:;—–-]", "")
            return rest.jlength <= 2 ? "" : whole
        }
    }

    private static func htmlTitle(_ html: String) -> String? { clean(first(html, "(?is)<title[^>]*>(.*?)</title>")) }

    public static func markdownToText(_ md: String) -> String {
        var t = md.replaceAll("(?s)```.*?```", " ")
        t = t.replaceAll("!\\[[^]]*]\\([^)]*\\)", "")                    // images
        t = t.replaceAll("\\[([^]]+)]\\([^)]*\\)", "$1")                 // links -> their text
        t = t.replaceAll("(?m)^\\s{0,3}#{1,6}\\s*(.*?)\\s*#*$", "\n" + TextMarkers.head + "$1.\n")  // headings
        t = t.replaceAll("(\\*\\*|__)(\\S(?:.*?\\S)?)\\1", TextMarkers.on + "$2" + TextMarkers.off)  // **bold**
        t = t.replaceAll("(?<![\\w*])([*_])(\\S(?:.*?\\S)?)\\1(?![\\w*])", TextMarkers.on + "$2" + TextMarkers.off)  // *italic*
        t = t.replaceAll("(?m)^\\s*>\\s?", "")
        t = t.replaceAll("(\\*\\*|__|\\*|_|`)", "")
        t = t.replaceAll("(?m)^\\s*([-*+]|\\d+\\.)\\s+", "• ")
        return t
    }

    // ---------------------------------------------------------------- helpers
    /// The archive's text entries by name (pictures, fonts and audio are skipped), each entry and the whole capped: a
    /// large illustrated book, or a zip bomb, must not run the machine out of memory.
    private static func unzip(_ data: Data) throws -> [String: Data] {
        let zip = try ZipArchive(data: data)
        var out: [String: Data] = [:]
        var total = 0
        for e in zip.entries {
            let name = e.name.lowercased()
            if e.isDirectory || !name.matches(".*\\.(xhtml|html|htm|xml|opf|ncx|rels)$") { continue }
            if e.uncompressedSize > 16 << 20 || total + e.uncompressedSize > 96 << 20 {
                throw Unsupported("This file is too large to open.")
            }
            let b = try zip.contents(of: e, limit: 16 << 20)
            total += b.count
            out[e.name] = b
        }
        return out
    }

    private static func str(_ b: Data?) -> String { b.map { String(decoding: $0, as: UTF8.self) } ?? "" }

    private static func first(_ s: String?, _ re: String) -> String? {
        guard let s = s else { return nil }
        return rx(re).find(s)?.group(1)
    }

    private static func clean(_ s: String?) -> String? {
        guard let s = s else { return nil }
        return TCChar.trim(decode(s.replaceAll("<[^>]*>", " ")).replaceAll("\\s+", " "))
    }

    /// HTML entities -> characters: the common named ones and every numeric one (an invalid one becomes a space).
    public static func decode(_ str: String) -> String {
        let s = str.replacingOccurrences(of: "&nbsp;", with: " ").replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<").replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"").replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&apos;", with: "'").replacingOccurrences(of: "&rsquo;", with: "’")
            .replacingOccurrences(of: "&lsquo;", with: "‘").replacingOccurrences(of: "&ldquo;", with: "“")
            .replacingOccurrences(of: "&rdquo;", with: "”").replacingOccurrences(of: "&mdash;", with: "—")
            .replacingOccurrences(of: "&ndash;", with: "–").replacingOccurrences(of: "&hellip;", with: "…")
        return htmlNumericEntities(s)
    }

    private static func firstWords(_ paras: [String]) -> String {
        let s = paras.first ?? "Text"
        if s.jlength <= 48 { return s }
        let sp = TCChar.lastIndexOf(s as NSString, " ", 48)
        return s.jsub(0, sp > 0 ? sp : 48) + "…"
    }
}

/// "&#233;" and "&#xE9;" -> "é"; a number that is not a code point (or not a number in its base) becomes a space, as
/// Integer.parseInt / Character.toChars fail in the Java. Shared by Extract and Article.
func htmlNumericEntities(_ s: String) -> String {
    rx("&#(x?)([0-9a-fA-F]+);").replaceAll(s) { m in
        let hex = !(m.group(1) ?? "").isEmpty
        guard let v = Int(m.group(2) ?? "", radix: hex ? 16 : 10), v <= 0x10FFFF else { return " " }
        if let u = Unicode.Scalar(v) { return String(Character(u)) }
        return String(utf16CodeUnits: [unichar(v)], count: 1)   // a lone surrogate, as Java's toChars gives
    }
}
