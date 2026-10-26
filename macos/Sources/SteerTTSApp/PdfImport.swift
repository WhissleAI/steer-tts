import Foundation
import PDFKit
import SteerText

/// A PDF's lines with their boxes, from PDFKit, handed to PdfLayout for the reading order (running headers, footers and
/// page numbers dropped, columns read in order, paragraphs joined, headings found), as the Android app's PdfText does
/// with PdfRenderer. Boxes are converted to PdfLayout's convention: points, origin top-left, y down.
enum PdfImport {
    struct Result {
        var title: String
        var author: String?
        var paragraphs: [String] = []
        var spans: [[PdfLayout.Span]] = []
    }

    static func read(_ url: URL) throws -> Result {
        guard let doc = PDFDocument(url: url) else {
            throw Extract.Unsupported("This PDF could not be opened.")
        }
        if doc.isLocked { throw Extract.Unsupported("This PDF is protected with a password.") }
        var lines: [PdfLayout.Line] = []
        for p in 0..<doc.pageCount {
            guard let page = doc.page(at: p) else { continue }
            let box = page.bounds(for: .mediaBox)
            guard let all = page.selection(for: box) else { continue }
            for sel in all.selectionsByLine() {
                let t = TextClean.normalizeChars(sel.string ?? "").replacingOccurrences(of: "\r", with: " ").trimmed
                if t.isEmpty { continue }
                let r = sel.bounds(for: page)
                if r.isEmpty { continue }
                let top = box.maxY - r.maxY, bottom = box.maxY - r.minY
                lines.append(PdfLayout.Line(page: p, text: t, l: Float(r.minX - box.minX), t: Float(top),
                                            r: Float(r.maxX - box.minX), b: Float(bottom),
                                            pw: Float(box.width), ph: Float(box.height)))
            }
        }
        let attrs = doc.documentAttributes ?? [:]
        let fallback = url.deletingPathExtension().lastPathComponent
        var out = Result(title: (attrs[PDFDocumentAttribute.titleAttribute] as? String)?.trimmed.nonEmpty ?? fallback,
                         author: (attrs[PDFDocumentAttribute.authorAttribute] as? String)?.trimmed.nonEmpty)
        for para in PdfLayout.layout(lines) {
            let shift = para.heading ? 1 : 0   // the heading marker before the text
            out.paragraphs.append(para.text)
            out.spans.append(para.spans.map {
                PdfLayout.Span(page: $0.page, start: $0.start + shift, end: $0.end + shift, l: $0.l, t: $0.t, r: $0.r, b: $0.b)
            })
        }
        if out.paragraphs.isEmpty {
            throw Extract.Unsupported("This PDF has no text to read. It may be scanned pages (pictures of text).")
        }
        return out
    }
}

extension String {
    var nonEmpty: String? { isEmpty ? nil : self }
}
