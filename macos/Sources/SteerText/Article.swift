import Foundation

/// What to read for shared text: the text itself, or, when it is (little more than) a link, that page's article: its
/// title and paragraphs (inside <article> when the page has one), without scripts, menus, headers and footers. A rough
/// extraction, not a reader mode, but enough for news and blog pages.
///
/// Ported from the Android app's Article.java. Fetching is the app's job: link(in:) says what to fetch (at most
/// maxBytes of it), text(of:contentType:) decodes the body, extract(_:url:) reads the article.
public struct Article: Equatable {
    private static let urlPattern = "https?://\\S+"
    /// How much of a page to download, and how much of its article to keep.
    public static let maxBytes = 3 << 20, maxChars = 20000

    public let title: String?
    /// The title (as a sentence) and the article's paragraphs, one per line.
    public let text: String
    /// The page the article came from, when the caller gave it.
    public let url: URL?

    public init(title: String?, text: String, url: URL? = nil) {
        self.title = title
        self.text = text
        self.url = url
    }

    /// Shared text that is a link with at most a short caption ("Interesting read https://...").
    public static func isLink(_ s: String) -> Bool {
        guard let m = rx(urlPattern).find(s) else { return false }
        return s.jlength - (m.group() ?? "").jlength < 100
    }

    /// The link to fetch for shared text, or nil when it is not a link (read the text itself). Plain http becomes
    /// https, as on Android (App Transport Security blocks plain http on the Mac too; most sites serve both).
    public static func link(in shared: String) -> URL? {
        guard isLink(shared), let m = rx(urlPattern).find(shared), let u = m.group() else { return nil }
        return URL(string: u.replaceFirst("^http://", "https://"))
    }

    /// A response body as text, in the charset its Content-Type names (UTF-8 otherwise, or when it is unknown).
    public static func text(of body: Data, contentType: String?) -> String {
        var enc = String.Encoding.utf8
        if let ct = contentType, let name = rx("charset=([\\w-]+)").find(ct)?.group(1) {
            let cf = CFStringConvertIANACharSetNameToEncoding(name as CFString)
            if cf != kCFStringEncodingInvalidId { enc = String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(cf)) }
        }
        return String(data: body, encoding: enc) ?? String(decoding: body, as: UTF8.self)
    }

    /// The article in a page's HTML.
    public static func extract(_ html: String, url: URL? = nil) -> Article {
        let h = html.replaceAll("(?is)<(script|style|noscript|svg|nav|header|footer|aside|form|figure)\\b.*?</\\1>", " ")
        var title = first(h, "(?is)<meta[^>]+property=[\"']og:title[\"'][^>]+content=[\"']([^\"']+)")
        if title == nil { title = first(h, "(?is)<title[^>]*>(.*?)</title>") }
        var body = paragraphs(first(h, "(?is)<article\\b[^>]*>(.*?)</article>"))
        if body.jlength < 500 { body = paragraphs(h) }
        if body.jlength > maxChars {
            let cut = TCChar.lastIndexOf(body as NSString, ". ", maxChars)
            body = body.jsub(0, cut > 0 ? cut + 1 : maxChars)
        }
        title = title.map(clean)
        if body.isEmpty {
            return Article(title: title, text: title.map { $0 + ". I couldn't find an article on that page." }
                ?? "I couldn't find an article on that page.", url: url)
        }
        return Article(title: title, text: (title.map { $0 + ".\n" } ?? "") + body, url: url)
    }

    private static func paragraphs(_ h: String?) -> String {
        guard let h = h else { return "" }
        var b = ""
        for m in rx("(?is)<p\\b[^>]*>(.*?)</p>").all(h) {
            let p = clean(m.group(1) ?? "")
            if p.jlength >= 40 { b += p + "\n" }
        }
        return TCChar.trim(b)
    }

    private static func first(_ h: String?, _ re: String) -> String? {
        guard let h = h else { return nil }
        return rx(re).find(h)?.group(1)
    }

    private static func clean(_ str: String) -> String {
        let s = str.replaceAll("(?s)<[^>]*>", " ").replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: "&amp;", with: "&").replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">").replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'").replacingOccurrences(of: "&apos;", with: "'")
            .replacingOccurrences(of: "&rsquo;", with: "'").replacingOccurrences(of: "&lsquo;", with: "'")
            .replacingOccurrences(of: "&ldquo;", with: "\"").replacingOccurrences(of: "&rdquo;", with: "\"")
            .replacingOccurrences(of: "&mdash;", with: ", ").replacingOccurrences(of: "&ndash;", with: "-")
            .replacingOccurrences(of: "&hellip;", with: "...")
        return TCChar.trim(htmlNumericEntities(s).replaceAll("\\s+", " "))
    }
}
