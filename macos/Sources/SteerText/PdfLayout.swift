import Foundation

/// Reading order of a PDF from where its lines sit on the page (plain Swift, unit tested; the app feeds it the lines and
/// their boxes from PDFKit). Running headers, footers and page numbers are recognised by position and repetition and
/// dropped; a page in two columns is read column by column (a title or abstract across the top first); lines are
/// joined into paragraphs by spacing, indentation and size; a line set clearly larger than the text is a heading; a word
/// hyphenated at a line end is joined. Every paragraph keeps the boxes of its lines, so the reader can show the very
/// line being read on the page.
///
/// Coordinates: page points with the origin at the TOP-LEFT of the page and y growing DOWN (as Android's PdfRenderer).
/// PDFKit's page space has its origin at the bottom-left with y up, so the app converts each line's bounds:
/// t = pageHeight - maxY, b = pageHeight - minY (and l = minX - cropBox.minX, r = maxX - cropBox.minX).
///
/// Ported from the Android app's PdfLayout.java; arithmetic is in Float, as the Java's.
public enum PdfLayout {
    /// A line as the PDF has it: page index, text, box in page points (top-left origin, y down), and the page size.
    public struct Line: Equatable {
        public let page: Int
        public let text: String
        public let l: Float, t: Float, r: Float, b: Float, pw: Float, ph: Float

        public init(page: Int, text: String, l: Float, t: Float, r: Float, b: Float, pw: Float, ph: Float) {
            self.page = page
            self.text = text
            self.l = l
            self.t = t
            self.r = r
            self.b = b
            self.pw = pw
            self.ph = ph
        }

        var h: Float { b - t }
        var w: Float { r - l }
    }

    /// Where part of a paragraph is on a page: characters [start, end) of the paragraph's raw text (UTF-16 offsets),
    /// box as fractions of the page (top-left origin, y down).
    public struct Span: Equatable, Codable {
        public let page: Int, start: Int, end: Int
        public let l: Float, t: Float, r: Float, b: Float

        public init(page: Int, start: Int, end: Int, l: Float, t: Float, r: Float, b: Float) {
            self.page = page
            self.start = start
            self.end = end
            self.l = l
            self.t = t
            self.r = r
            self.b = b
        }
    }

    public struct Para: Equatable {
        /// The paragraph's text, without the heading marker; span offsets index into it.
        public var raw: String = ""
        public var spans: [Span] = []
        public var heading = false

        public init(raw: String = "", spans: [Span] = [], heading: Bool = false) {
            self.raw = raw
            self.spans = spans
            self.heading = heading
        }

        /// The text to read: a heading starts with Narration's heading marker (U+E002), as the Java Para.text().
        public var text: String { (heading ? TextMarkers.head : "") + raw }
    }

    private static let pageNo =
        "(?i)^(page\\s+)?\\d{1,4}(\\s*(/|of)\\s*\\d{1,4})?$|^[-–—]\\s*\\d{1,4}\\s*[-–—]$|^[ivxlcdm]{1,6}$"
    private static let bullet = "^([•·▪◦‣●○■□–-]|\\d{1,2}[.)]|[a-z][.)])\\s+.*"

    public static func layout(_ all: [Line]) -> [Para] {
        // ---- 1. running headers, footers, page numbers
        var pages = 0
        for ln in all { pages = max(pages, ln.page + 1) }
        var edgeCount: [String: Int] = [:]
        for ln in all where edge(ln) { edgeCount[key(ln), default: 0] += 1 }
        var body: [Line] = []
        for ln in all {
            let t = TCChar.trim(ln.text)
            if t.isEmpty { continue }
            if edge(ln) && (t.matches(pageNo) || (pages >= 2 && edgeCount[key(ln)]! >= max(2, pages / 2))) { continue }
            body.append(ln)
        }
        let median = medianHeight(body)
        body.removeAll { ln in   // small print at the edge: a header
            edge(ln) && ln.h < median * 0.85 && TCChar.trim(ln.text).split(regex: "\\s+").count <= 12
        }
        // ---- 2. reading order, page by page
        var ordered: [Line] = []
        for p in 0..<pages { ordered.append(contentsOf: readingOrder(body.filter { $0.page == p })) }
        // ---- 3. paragraphs
        var out: [Para] = []
        var cur: Para?
        let text = NSMutableString()   // cur's text, edited in UTF-16 units
        var prev: Line?
        for ln in ordered {
            let t = TCChar.trim(ln.text).replaceAll("\\s+", " ")
            let heading = isHeading(ln, t, median, prev)
            if cur == nil || prev == nil || breaksBefore(prev!, ln, t, median, heading, cur!.heading) {
                if var c = cur, text.length > 0 { c.raw = text as String; out.append(c) }
                cur = Para(heading: heading)
                text.setString("")
            }
            var start = text.length
            if start > 0 {
                let last = text.character(at: start - 1)
                if last == 0x2D && start > 1 && TCChar.isLetter(text.character(at: start - 2)) && !t.isEmpty
                    && TCChar.isLowerCase(t.jchar(0)) {
                    text.deleteCharacters(in: NSRange(location: start - 1, length: 1))   // "exam-" + "ple"
                    start -= 1
                } else {
                    text.append(" ")
                    start += 1
                }
            }
            text.append(t)
            cur!.spans.append(Span(page: ln.page, start: start, end: text.length, l: ln.l / ln.pw, t: ln.t / ln.ph,
                                   r: ln.r / ln.pw, b: ln.b / ln.ph))
            prev = ln
        }
        if var c = cur, text.length > 0 { c.raw = text as String; out.append(c) }
        return out
    }

    private static func edge(_ ln: Line) -> Bool { ln.b < ln.ph * 0.09 || ln.t > ln.ph * 0.91 }

    private static func key(_ ln: Line) -> String {
        (ln.t < ln.ph / 2 ? "top:" : "bottom:") + TCChar.trim(ln.text).replaceAll("\\d+", "#").lowercased()
    }

    private static func medianHeight(_ lines: [Line]) -> Float {
        if lines.isEmpty { return 10 }
        let h = lines.map { $0.h }.sorted()
        return h[h.count / 2]
    }

    /// Two columns when enough lines sit wholly on each side of the page's middle and few cross it; then: lines across
    /// the page above the columns, the left column, the right column, lines across the page below. Otherwise top to
    /// bottom, with lines that share a row (a role and its dates) left to right.
    public static func readingOrder(_ page: [Line]) -> [Line] {
        guard let firstLine = page.first else { return [] }
        let pw = firstLine.pw, mid = pw / 2
        var left: [Line] = [], right: [Line] = [], across: [Line] = []
        for ln in page {
            if ln.r <= mid + pw * 0.02 { left.append(ln) }
            else if ln.l >= mid - pw * 0.02 { right.append(ln) }
            else { across.append(ln) }
        }
        let columns = left.count >= 6 && right.count >= 6 && across.count * 4 <= page.count && wide(left, pw) && wide(right, pw)
        if !columns {
            let all = page.sorted { a, b in sameRow(a, b) ? a.l < b.l : a.t < b.t }
            return mergeRows(all)
        }
        var colTop = Float.greatestFiniteMagnitude, colBottom: Float = 0
        for ln in left { colTop = min(colTop, ln.t); colBottom = max(colBottom, ln.b) }
        for ln in right { colTop = min(colTop, ln.t); colBottom = max(colBottom, ln.b) }
        var above: [Line] = [], below: [Line] = []
        for ln in across { if ln.t < colTop + 1 { above.append(ln) } else { below.append(ln) } }
        let byTop: (Line, Line) -> Bool = { $0.t < $1.t }
        return above.sorted(by: byTop) + left.sorted(by: byTop) + right.sorted(by: byTop) + below.sorted(by: byTop)
    }

    /// A column of real text: most of its lines are wider than a quarter of the page (not a column of dates).
    private static func wide(_ col: [Line], _ pw: Float) -> Bool {
        var n = 0
        for ln in col where ln.w > pw * 0.25 { n += 1 }
        return n * 2 >= col.count
    }

    private static func sameRow(_ a: Line, _ b: Line) -> Bool {
        let overlap = min(a.b, b.b) - max(a.t, b.t)
        return overlap > 0.6 * min(a.h, b.h)
    }

    /// Lines on one row (a job title on the left, its dates on the right) become one line, read left to right.
    private static func mergeRows(_ sorted: [Line]) -> [Line] {
        var out: [Line] = []
        for ln in sorted {
            if let p = out.last, p.page == ln.page && sameRow(p, ln) && ln.l > p.r {
                let sep = TCChar.trim(p.text).matches(".*[,.;:!?]$") ? " " : ", "
                out[out.count - 1] = Line(page: p.page, text: TCChar.trim(p.text) + sep + TCChar.trim(ln.text), l: p.l,
                                          t: min(p.t, ln.t), r: ln.r, b: max(p.b, ln.b), pw: p.pw, ph: p.ph)
                continue
            }
            out.append(ln)
        }
        return out
    }

    private static let numbered = "^(\\d+(\\.\\d+)*\\.?|[IVX]+\\.|Chapter \\w+|Part \\w+)\\s+\\p{Lu}.*"

    /// A heading: short, not a sentence, and either clearly larger than the text, numbered like a section ("2 Method",
    /// "3.1 Data"), or a title-case line set apart by space above ("Skills"). Box heights alone mislead: a word without
    /// descenders measures smaller.
    private static func isHeading(_ ln: Line, _ t: String, _ median: Float, _ prev: Line?) -> Bool {
        let w = t.split(regex: " ")
        if t.jlength > 90 || w.count > 12 { return false }
        if t.matches(".*[.,;:]$") && !t.matches(numbered) { return false }
        if ln.h >= median * 1.25 { return true }
        if w.count <= 8 && t.matches(numbered) { return true }
        let apart = prev == nil || prev!.page != ln.page || ln.t - prev!.b > median * 1.1
        var caps = 0
        for x in w where !x.isEmpty && TCChar.isUpperCase(x.jchar(0)) { caps += 1 }
        return apart && w.count <= 5 && caps * 2 >= w.count && !t.matches(".*\\d{4}.*") && !t.contains("@")
    }

    private static func breaksBefore(_ prev: Line, _ ln: Line, _ t: String, _ median: Float, _ heading: Bool,
                                     _ prevHeading: Bool) -> Bool {
        let ended = TCChar.trim(prev.text).matches(".*[.!?:”\"]$")
        if prev.page != ln.page {
            // a page break inside a sentence continues the paragraph
            return heading || prevHeading || ended
        }
        if heading != prevHeading { return true }
        if heading { return abs(ln.h - prev.h) > median * 0.2 || ln.t - prev.b > median * 0.8 }
        let gap = ln.t - prev.b
        if gap > max(median * 0.75, prev.h * 0.6) { return true }          // a blank line between paragraphs
        if ln.t < prev.t - median { return ended }                          // a new column: continues mid-sentence
        if t.matches(bullet) { return true }
        if ended && ln.l > prev.l + median * 0.9 { return true }            // a first-line indent
        return ended && prev.r < ln.r - prev.pw * 0.12 && ln.w > prev.w      // a short last line
    }
}
