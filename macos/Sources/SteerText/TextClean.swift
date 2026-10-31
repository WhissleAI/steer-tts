import Foundation

/// Document text -> speakable sentences. Text pulled out of PDFs, e-books and web pages carries layout that should not
/// be read aloud: lines wrapped mid-sentence, words hyphenated across lines, page numbers, running headers and footers,
/// ligature characters, citation markers, long URLs. clean() removes that layout; sentences() then splits the text where
/// a reader would pause: at sentence ends, but not after abbreviations ("Dr.", "e.g.", "U.S."), initials or decimals,
/// and long sentences are split again at a semicolon, colon or comma so the first sound never waits on a paragraph.
///
/// Ported from the Android app's TextClean.java. All offsets are UTF-16 code units (NSString / Java indices).
public enum TextClean {
    private static let ligatures: [(String, String)] = [
        ("\u{FB00}", "ff"), ("\u{FB01}", "fi"), ("\u{FB02}", "fl"), ("\u{FB03}", "ffi"), ("\u{FB04}", "ffl"),
        ("\u{FB05}", "st"), ("\u{FB06}", "st"),
    ]

    /// Pages of text (one string per PDF page; for other sources a single "page") -> clean paragraphs.
    public static func clean(_ pages: [String]) -> [String] {
        let lines = pages.map { splitLines(normalizeChars($0)) }
        let running = runningLines(lines)
        var paras: [String] = []
        let cur = NSMutableString()
        for page in lines {
            for raw in page {
                let ln = TCChar.trim(raw)
                if ln.isEmpty {                              // a blank line ends a paragraph
                    flush(cur, &paras)
                    continue
                }
                if isPageNumber(ln) || running.contains(key(ln)) { continue }
                if cur.length > 0 {
                    let last = cur.character(at: cur.length - 1)
                    if last == 0x2D /* - */ && cur.length > 1 && TCChar.isLetter(cur.character(at: cur.length - 2))
                        && TCChar.isLowerCase(ln.jchar(0)) {
                        cur.deleteCharacters(in: NSRange(location: cur.length - 1, length: 1))   // "exam-" + "ple" -> "example"
                    } else if isBullet(ln) || (endsSentence(cur) && looksLikeHeading(ln)) {
                        flush(cur, &paras)
                    } else {
                        cur.append(" ")
                    }
                }
                cur.append(isBullet(ln) ? ln.replaceFirst("^[•·▪◦‣\\-–*]\\s*", "") : ln)
            }
            // a page break inside a sentence continues it; after a sentence end it starts a new paragraph
            if cur.length > 0 && endsSentence(cur) { flush(cur, &paras) }
        }
        flush(cur, &paras)
        var out: [String] = []
        for p in paras {
            let s = speakable(p)
            if !s.isEmpty { out.append(s) }
        }
        return out
    }

    public static func clean(_ text: String) -> [String] { clean([text]) }

    private static func flush(_ cur: NSMutableString, _ paras: inout [String]) {
        let s = TCChar.trim(cur as String)
        if !s.isEmpty { paras.append(s) }
        cur.setString("")
    }

    public static func normalizeChars(_ str: String) -> String {
        var s = str
        for (k, v) in ligatures { s = s.replacingOccurrences(of: k, with: v) }
        return s.replacingOccurrences(of: "\u{00AD}", with: "")             // soft hyphen
            .replacingOccurrences(of: "\u{00A0}", with: " ").replacingOccurrences(of: "\u{2009}", with: " ")
            .replacingOccurrences(of: "\u{202F}", with: " ")
            .replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
            .replacingOccurrences(of: "\u{0C}", with: "\n")
    }

    // Swift treats "\r\n" as one Character, so split on UTF-16 units to keep Java's split("\n", -1) exactly.
    private static func splitLines(_ s: String) -> [String] {
        (s as NSString).components(separatedBy: "\n")
    }

    private static let pageNo =
        "(?i)^(page\\s+)?\\d{1,4}(\\s*(/|of)\\s*\\d{1,4})?$|^[-–—]\\s*\\d{1,4}\\s*[-–—]$|^[ivxlcdm]{1,6}$"

    private static func isPageNumber(_ ln: String) -> Bool { ln.matches(pageNo) }

    private static func key(_ ln: String) -> String { ln.replaceAll("\\d+", "#").lowercased() }

    /// Lines that repeat at the top or bottom of most pages (running headers, footers), digits ignored.
    private static func runningLines(_ pages: [[String]]) -> Set<String> {
        var out = Set<String>()
        if pages.count < 3 { return out }
        var n: [String: Int] = [:]
        for page in pages {
            var edge: [String] = []
            var nonEmpty: [String] = []
            for l in page where !TCChar.trim(l).isEmpty { nonEmpty.append(TCChar.trim(l)) }
            for i in 0..<min(2, nonEmpty.count) { edge.append(nonEmpty[i]) }
            for i in max(0, nonEmpty.count - 2)..<nonEmpty.count { edge.append(nonEmpty[i]) }
            for l in Set(edge) where l.jlength < 90 { n[key(l), default: 0] += 1 }
        }
        for (k, v) in n where v >= max(3, pages.count / 2) { out.insert(k) }
        return out
    }

    private static func isBullet(_ ln: String) -> Bool { ln.matches("^[•·▪◦‣]\\s*.+|^[-–*]\\s+\\S.*") }

    private static func endsSentence(_ ns: NSString) -> Bool {
        var i = ns.length - 1
        while i >= 0 {
            let c = ns.character(at: i)
            if c == 0x22 || c == 0x27 || c == 0x29 || c == 0x201D || c == 0x2019 || c == 0x5D { i -= 1; continue }
            return c == 0x2E || c == 0x21 || c == 0x3F || c == 0x3A
        }
        return false
    }

    /// A short line without a final full stop, mostly capitalised words: a heading.
    private static func looksLikeHeading(_ ln: String) -> Bool {
        if ln.jlength > 70 || endsSentence(ln as NSString) { return false }
        let w = ln.split(regex: "\\s+")
        var caps = 0
        for x in w where !x.isEmpty && TCChar.isUpperCase(x.jchar(0)) { caps += 1 }
        return w.count <= 10 && caps * 2 >= w.count
    }

    private static let url = "(?i)\\b(?:https?://|www\\.)([\\w.-]+\\.[a-z]{2,})(?:/[^\\s)\\]]*)?"
    private static let cite = "\\s?\\[(\\d{1,3}([,–-]\\s?\\d{1,3})*|citation needed|[a-z])\\]"

    /// One paragraph, ready for the voice: links read as their site, citation markers gone, spaces tidy.
    public static func speakable(_ p: String) -> String {
        let b = rx(url).replaceAll(p) { m in (m.group(1) ?? "").replaceFirst("^www\\.", "") }
        var s = b.replaceAll(cite, "")
        s = TCChar.trim(s.replaceAll("\\s+([,.;:!?])", "$1").replaceAll("\\s{2,}", " "))
        return s
    }

    /// A book's story without its wrapping: Project Gutenberg's licence header and footer (between its "*** START OF"
    /// and "*** END OF" lines) and a table of contents (a "Contents" heading followed by short lines, up to the next
    /// heading or long paragraph).
    public static func bookBody(_ paras: [String]) -> [String] {
        var from = 0, to = paras.count
        for i in 0..<paras.count {
            let p = paras[i].uppercased()
            if p.contains("*** START OF THE PROJECT GUTENBERG") || p.contains("*** START OF THIS PROJECT GUTENBERG") { from = i + 1 }
            if (p.contains("*** END OF THE PROJECT GUTENBERG") || p.contains("*** END OF THIS PROJECT GUTENBERG")) && i >= from {
                to = i
                break
            }
        }
        var out: [String] = []
        var i = from
        while i < to {
            let p = paras[i], plain = TCChar.trim(TextMarkers.display(p))
            if plain.matches("(?i)(table of )?contents\\.?|(list of )?(illustrations|plates|figures)\\.?") {   // skip the table of contents
                var j = i + 1
                // contents entries are short lines; the story starts at the first real heading or a paragraph
                while j < to && !paras[j].hasPrefix(TextMarkers.head)
                        && TextMarkers.display(paras[j]).jlength < 80 && j - i < 200 { j += 1 }
                i = j
                continue
            }
            i += 1
            if plain.isEmpty { continue }
            out.append(p)
        }
        return titlePage(out)
    }

    /// A title page is set one short line at a time ("PRIDE." / "and" / "PREJUDICE" / "by" / "Jane Austen,"); read
    /// line by line each would be its own sentence with a pause. Before the first real paragraph, runs of three or more
    /// short lines are read as one.
    public static func titlePage(_ paras: [String]) -> [String] {
        var out: [String] = []
        var i = 0
        let n = paras.count
        while i < n && TextMarkers.display(paras[i]).jlength < 120 {
            var j = i
            while j < n && TCChar.trim(TextMarkers.display(paras[j])).jlength < 45 && !heading(paras[j]) { j += 1 }
            if j - i >= 3 {
                var b = ""
                for k in i..<j {
                    let x = TCChar.trim(TextMarkers.display(paras[k]))
                    if !b.isEmpty { b += " " }
                    b += x
                }
                out.append(b)
                i = j
            } else {
                out.append(paras[i])
                i += 1
            }
        }
        while i < n { out.append(paras[i]); i += 1 }
        return out
    }

    private static func heading(_ p: String) -> Bool {
        p.hasPrefix(TextMarkers.head)
            || TCChar.trim(TextMarkers.display(p)).matches("(?i)(chapter|book|part|stave|volume|act)\\b.*|[IVXLC]+\\.?")
    }

    // ---------------------------------------------------------------- sentences
    private static let abbrev: Set<String> = [
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "e.g", "i.e", "fig", "figs", "no", "nos",
        "vol", "vols", "pp", "p", "ch", "sec", "ed", "eds", "inc", "ltd", "co", "corp", "dept", "est", "approx", "u.s",
        "u.k", "a.m", "p.m", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun", "gen", "gov", "sen", "rep", "rev", "capt", "col", "lt",
        "sgt", "ave", "blvd", "rd", "al", "cf", "ibid", "op", "ca",
    ]
    private static let maxLength = 240    // characters: split longer sentences at a pause

    /// Clean paragraphs -> sentences.
    public static func sentences(_ paragraphs: [String]) -> [String] {
        var out: [String] = []
        for p in paragraphs { for sp in spans(p) { out.append(p.jsub(sp.lowerBound, sp.upperBound)) } }
        return out
    }

    /// Where each sentence of a paragraph is: [start, end) ranges in UTF-16 code units, trimmed, in order.
    public static func spans(_ p: String) -> [Range<Int>] {
        let ns = p as NSString
        var out: [Range<Int>] = []
        for s in splitParagraph(ns) { out.append(contentsOf: splitLong(ns, s.lowerBound, s.upperBound)) }
        return out
    }

    private static func trimmed(_ p: NSString, _ a0: Int, _ b0: Int) -> Range<Int> {
        var a = a0, b = b0
        while a < b && TCChar.isWhitespace(p.character(at: a)) { a += 1 }
        while b > a && TCChar.isWhitespace(p.character(at: b - 1)) { b -= 1 }
        return a..<b
    }

    private static let closers: Set<unichar> = Set(".!?\"'”’)]".utf16)
    private static let openers: Set<unichar> = Set("\"'“‘(".utf16)

    private static func splitParagraph(_ p: NSString) -> [Range<Int>] {
        var out: [Range<Int>] = []
        var start = 0
        let n = p.length
        var i = 0
        while i < n {
            defer { i += 1 }
            let c = p.character(at: i)
            if c != 0x2E && c != 0x21 && c != 0x3F { continue }
            var j = i + 1
            while j < n && (closers.contains(p.character(at: j)) || p.character(at: j) == TextMarkers.emOff) { j += 1 }
            if j < n && !TCChar.isWhitespace(p.character(at: j)) { continue }          // "3.5", "e.g.x", "U.S.A"
            var k = j
            while k < n && TCChar.isWhitespace(p.character(at: k)) { k += 1 }
            if insideQuote(p, start, j) { continue }   // "Run! They're coming!" stays one line of dialogue
            var look = k   // past emphasis and heading markers, to the first real character
            while look < n && (p.character(at: look) == TextMarkers.emOn || p.character(at: look) == TextMarkers.headChar) { look += 1 }
            if look < n {
                let next = p.character(at: look)
                if !(TCChar.isUpperCase(next) || TCChar.isDigit(next) || openers.contains(next)) { continue }
            }
            if c == 0x2E {
                var w = i - 1
                while w >= start && !TCChar.isWhitespace(p.character(at: w)) { w -= 1 }
                let word = p.substring(with: NSRange(location: w + 1, length: i - w - 1)).lowercased()
                    .replaceAll("^[(\"'“‘]+", "")
                if abbrev.contains(word) || word.matches("[a-z]") { continue }    // "Dr.", "J. Smith"
            }
            let t = trimmed(p, start, j)
            if t.upperBound > t.lowerBound { out.append(t) }
            start = k
            i = k - 1
        }
        let t = trimmed(p, start, n)
        if t.upperBound > t.lowerBound { out.append(t) }
        return out
    }

    /// Whether a quotation opened in p[from, to) is still open at to (curly quotes counted, straight ones paired).
    private static func insideQuote(_ p: NSString, _ from: Int, _ to: Int) -> Bool {
        var curly = 0, straight = 0, single = 0
        var i = from
        while i < to {
            let c = p.character(at: i)
            let letterBefore = i > 0 && TCChar.isLetter(p.character(at: i - 1))
            let letterAfter = i + 1 < p.length && TCChar.isLetter(p.character(at: i + 1))
            if c == 0x201C || (c == 0x2018 && !letterBefore) { curly += 1 }
            else if c == 0x201D || (c == 0x2019 && curly > 0 && !letterAfter) { curly -= 1 }   // "don’t" is an apostrophe
            else if c == 0x22 { straight += 1 }
            else if c == 0x27 && !letterBefore && letterAfter { single += 1 }                  // 'Now, my dears,'
            else if c == 0x27 && single > 0 && !letterAfter { single -= 1 }
            i += 1
        }
        return curly > 0 || straight % 2 == 1 || single > 0   // (single quotes: ‘…’ opened and not yet closed count too)
    }

    /// A sentence longer than maxLength, split again at a pause (; : — ,), as character ranges of p.
    private static func splitLong(_ p: NSString, _ a0: Int, _ b0: Int) -> [Range<Int>] {
        var out: [Range<Int>] = []
        var a = a0, b = b0
        while b - a > maxLength {
            let s = p.substring(with: NSRange(location: a, length: b - a)) as NSString
            var cut = -1
            for sep in ["; ", ": ", " — ", ", "] {
                let at = TCChar.lastIndexOf(s, sep, maxLength)
                if at > maxLength / 3 {
                    cut = at + (sep as NSString).length - 1
                    break
                }
            }
            if cut < 0 { cut = TCChar.lastIndexOf(s, " ", maxLength) }
            if cut <= 0 { break }
            let head = trimmed(p, a, a + cut + 1)
            if head.upperBound > head.lowerBound { out.append(head) }
            let rest = trimmed(p, a + cut + 1, b)
            a = rest.lowerBound
            b = rest.upperBound
        }
        if b > a { out.append(a..<b) }
        return out
    }
}

/// The markup characters Narration reads (Narration.EM_ON / EM_OFF / HEAD in the Java: private-use code points), kept
/// here so the text pipeline does not depend on Narration; Narration.display() is mirrored by display().
enum TextMarkers {
    static let emOn: unichar = 0xE000, emOff: unichar = 0xE001, headChar: unichar = 0xE002
    static let on = "\u{E000}", off = "\u{E001}", head = "\u{E002}"

    /// Text without its emphasis and heading markers, trimmed (Narration.display).
    static func display(_ s: String) -> String {
        guard s.contains(on) || s.contains(off) || s.contains(head) else { return TCChar.trim(s) }
        return TCChar.trim(s.replacingOccurrences(of: on, with: "").replacingOccurrences(of: off, with: "")
            .replacingOccurrences(of: head, with: ""))
    }
}

/// java.lang.Character and String helpers over UTF-16 code units, so the ports decide exactly as the Java does
/// (a lone surrogate is no letter, digit or space, as in Java).
enum TCChar {
    private static func scalar(_ c: unichar) -> Unicode.Scalar? { Unicode.Scalar(c) }

    static func isLetter(_ c: unichar) -> Bool {
        guard let s = scalar(c) else { return false }
        switch s.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return false
        }
    }

    static func isUpperCase(_ c: unichar) -> Bool { scalar(c)?.properties.isUppercase ?? false }
    static func isLowerCase(_ c: unichar) -> Bool { scalar(c)?.properties.isLowercase ?? false }
    static func isDigit(_ c: unichar) -> Bool { scalar(c)?.properties.generalCategory == .decimalNumber }

    /// Character.isWhitespace: Unicode space separators except the no-break ones, line and paragraph separators,
    /// and the ASCII controls \t \n \u{0B} \f \r and \u{1C}-\u{1F}.
    static func isWhitespace(_ c: unichar) -> Bool {
        if (0x09...0x0D).contains(c) || (0x1C...0x1F).contains(c) { return true }
        if c == 0x00A0 || c == 0x2007 || c == 0x202F { return false }
        guard let s = scalar(c) else { return false }
        switch s.properties.generalCategory {
        case .spaceSeparator, .lineSeparator, .paragraphSeparator: return true
        default: return false
        }
    }

    /// Java's String.trim(): code units up to U+0020 removed from both ends.
    static func trim(_ str: String) -> String {
        let s = str as NSString
        var a = 0, b = s.length
        while a < b && s.character(at: a) <= 0x20 { a += 1 }
        while b > a && s.character(at: b - 1) <= 0x20 { b -= 1 }
        return a == 0 && b == s.length ? str : s.substring(with: NSRange(location: a, length: b - a))
    }

    /// Java's s.lastIndexOf(str, fromIndex): the last occurrence starting at or before fromIndex, or -1.
    static func lastIndexOf(_ s: NSString, _ str: String, _ fromIndex: Int) -> Int {
        let len = (str as NSString).length
        let end = min(s.length, fromIndex + len)
        if end < len { return -1 }
        let r = s.range(of: str, options: [.backwards, .literal], range: NSRange(location: 0, length: end))
        return r.location == NSNotFound ? -1 : r.location
    }
}
