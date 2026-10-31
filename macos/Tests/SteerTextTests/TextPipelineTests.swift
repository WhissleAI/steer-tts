import XCTest
@testable import SteerText

/// Document text to sentences, files and pages to documents, PDF reading order: the tests of SpeechLogicTest.java that
/// exercise TextClean, Extract, Article and PdfLayout, plus the Mac's own ZIP reader, EPUB/DOCX paths and PDF layout.
final class TextPipelineTests: XCTestCase {
    // ---------------------------------------------------------------- from SpeechLogicTest.java
    func testSentencesKeepMarkersAndAbbreviations() {
        let s = TextClean.sentences(["Dr. Smith arrived. " + Narration.EM_ON + "Nobody" + Narration.EM_OFF + " saw him. Then he left."])
        XCTAssertEqual(3, s.count)
        XCTAssertEqual("Dr. Smith arrived.", s[0])
        XCTAssertEqual("Nobody saw him.", Narration.display(s[1]))
    }

    func testDialogueIsNotSplitInsideQuotes() {
        let s = TextClean.sentences(["“Run! They're coming!” she shouted. It was quiet."])
        XCTAssertEqual(2, s.count)
        XCTAssertEqual("“Run! They're coming!” she shouted.", s[0])
        XCTAssertTrue(Narration.segments(s[0])[0].dialogue)
        let st = TextClean.sentences(["\"Stop. Now.\" He left."])
        XCTAssertEqual(2, st.count)
    }

    func testQuotesOfEveryKindSentences() {   // the TextClean half of quotesOfEveryKind
        let s = TextClean.sentences(["‘Run! They’re coming!’ she cried. Nobody moved."])
        XCTAssertEqual(2, s.count)
        let d = Narration.segments(s[0])[0]
        XCTAssertTrue(d.dialogue)
        XCTAssertEqual("Run! They’re coming!", d.text)
    }

    func testHtmlKeepsEmphasisAndHeadings() {
        let t = Extract.htmlToText("<h1>One</h1><p>It was <em>cold</em>.</p>")
        XCTAssertTrue(t.contains(Narration.HEAD + "One"))
        XCTAssertTrue(t.contains(Narration.EM_ON + "cold" + Narration.EM_OFF))
    }

    func testBookMarkupReadsCleanly() {
        let html = "<h2>LIST OF ILLUSTRATIONS.</h2><table>"
            + "<tr><td>“<span class=\"smcap\">He gathered herbs</span>”</td><td><a href=\"b.html#Page_213\">213</a></td></tr>"
            + "<tr><td>A Gleam of Sunshine</td><td><a href=\"b.html#Page_249\">249</a></td></tr></table>"
            + "<p><a href=\"c.html#ch1\">CHAPTER I. Down the Rabbit-Hole</a></p>"
            + "<p class=\"figcenter\"><span title=\"The Custom House\" id=\"img_images_illu_001.jpg\">The Custom House</span></p>"
            + "<div class=\"initial\"><span title=\"I\" id=\"img_images_initial_i.jpg\">I</span></div>"
            + "<p><span class=\"firstwords\">t</span> is a little remarkable, said “<span>he</span>.”</p>"
        let text = Extract.htmlToText(html)
        XCTAssertFalse(text.contains("249") || text.contains("Gleam") || text.contains("Rabbit-Hole") || text.contains("Custom House"), text)
        XCTAssertTrue(text.contains("It is a little remarkable, said “he.”"), text)
    }

    // ---------------------------------------------------------------- TextClean
    func testCleanJoinsLinesHyphensAndDropsPageNumbers() {
        let paras = TextClean.clean("The exam-\nple was wrapped\nacross lines.\n\n12\n\nSee www.example.com/x for more [3].")
        XCTAssertEqual(["The example was wrapped across lines.", "See example.com for more."], paras)
    }

    func testRunningHeadersAreDropped() {
        let words = ["alpha", "beta", "gamma", "delta"]
        let pages = (0..<4).map { "My Book\n\nThe \(words[$0]) page begins\nand goes on\nto the \(words[$0]) end.\n\nPage \($0 + 1)" }
        XCTAssertEqual(words.map { "The \($0) page begins and goes on to the \($0) end." }, TextClean.clean(pages))
    }

    func testLongSentencesSplitAtAPause() {
        let long = String(repeating: "word ", count: 60) + "and then; " + String(repeating: "more ", count: 20) + "end."
        let s = TextClean.sentences([long])
        XCTAssertGreaterThan(s.count, 1)
        XCTAssertTrue(s.allSatisfy { $0.jlength <= 240 })
    }

    func testBookBodySkipsGutenbergWrappingAndContents() {
        let paras = ["The Project Gutenberg eBook", "*** START OF THE PROJECT GUTENBERG EBOOK X ***", "CONTENTS",
                     "Chapter I", "Chapter II", Narration.HEAD + "CHAPTER I.",
                     "It was a long story that went on for well over a few words to be a real paragraph of the book.",
                     "*** END OF THE PROJECT GUTENBERG EBOOK X ***", "licence"]
        XCTAssertEqual([Narration.HEAD + "CHAPTER I.",
                        "It was a long story that went on for well over a few words to be a real paragraph of the book."],
                       TextClean.bookBody(paras))
        XCTAssertEqual(["PRIDE. and PREJUDICE by Jane Austen,", Narration.HEAD + "Chapter 1"],
                       TextClean.titlePage(["PRIDE.", "and", "PREJUDICE", "by", "Jane Austen,", Narration.HEAD + "Chapter 1"]))
    }

    // ---------------------------------------------------------------- Markdown, Article
    func testMarkdown() {
        let t = Extract.markdownToText("# Title\nSome **bold** and *it* with [a link](http://x).\n- item")
        XCTAssertTrue(t.contains(Narration.HEAD + "Title."))
        XCTAssertTrue(t.contains(Narration.EM_ON + "bold" + Narration.EM_OFF))
        XCTAssertTrue(t.contains(Narration.EM_ON + "it" + Narration.EM_OFF))
        XCTAssertTrue(t.contains("a link."))
        XCTAssertTrue(t.contains("• item"))
    }

    func testArticleExtraction() {
        XCTAssertTrue(Article.isLink("Interesting read http://example.com/a"))
        XCTAssertFalse(Article.isLink("no link here"))
        XCTAssertEqual(URL(string: "https://example.com/a"), Article.link(in: "look http://example.com/a"))
        let para = "<p>This is a paragraph long enough to count as article text, &amp; it goes on.</p>"
        let html = "<html><head><title>T &amp; U</title></head><body><nav><p>Menu menu menu menu menu menu menu menu menu</p></nav>"
            + "<article>" + String(repeating: para, count: 8) + "</article><footer>x</footer></body></html>"
        let a = Article.extract(html, url: URL(string: "https://example.com/a"))
        XCTAssertEqual("T & U", a.title)
        XCTAssertTrue(a.text.hasPrefix("T & U.\nThis is a paragraph"))
        XCTAssertFalse(a.text.contains("Menu"))
        let d = Extract.fromHTML(html, baseURL: URL(string: "https://example.com/a"))
        XCTAssertEqual("web", d.kind)
        XCTAssertEqual(9, d.paragraphs.count)
        XCTAssertEqual("https://example.com/a", d.source)
    }

    // ---------------------------------------------------------------- ZIP, EPUB, DOCX
    /// Files written to a temp dir and zipped with /usr/bin/zip: `stored` names are added uncompressed (-0), the rest
    /// deflated.
    private func makeZip(_ files: [(String, String)], stored: Set<String> = []) throws -> Data {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("steertext-zip-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        for (name, text) in files {
            let u = dir.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: u.deletingLastPathComponent(), withIntermediateDirectories: true)
            try text.write(to: u, atomically: true, encoding: .utf8)
        }
        let out = dir.appendingPathComponent("out.zip")
        func zip(_ args: [String]) throws {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
            p.currentDirectoryURL = dir
            p.arguments = ["-q", "-X", out.path] + args
            try p.run()
            p.waitUntilExit()
            XCTAssertEqual(0, p.terminationStatus)
        }
        let s = files.map { $0.0 }.filter { stored.contains($0) }, d = files.map { $0.0 }.filter { !stored.contains($0) }
        if !s.isEmpty { try zip(["-0"] + s) }
        if !d.isEmpty { try zip(["-9"] + d) }
        return try Data(contentsOf: out)
    }

    func testZipReaderStoredAndDeflated() throws {
        let big = String(repeating: "All work and no play makes Jack a dull boy. ", count: 500)
        let data = try makeZip([("plain.txt", "stored text"), ("dir/big.txt", big)], stored: ["plain.txt"])
        let zip = try ZipArchive(data: data)
        XCTAssertEqual(["plain.txt", "dir/big.txt"], zip.entries.map { $0.name })
        XCTAssertEqual(0, zip.entry(named: "plain.txt")?.method)
        XCTAssertEqual(8, zip.entry(named: "dir/big.txt")?.method)
        XCTAssertEqual("stored text", String(decoding: try zip.contents(named: "plain.txt")!, as: UTF8.self))
        XCTAssertEqual(big, String(decoding: try zip.contents(named: "dir/big.txt")!, as: UTF8.self))
        XCTAssertThrowsError(try zip.contents(named: "dir/big.txt", limit: 100))
        XCTAssertNil(try zip.contents(named: "missing"))
        // a slice of a larger buffer (non-zero start index) reads the same
        let padded = Data([1, 2, 3]) + data
        XCTAssertEqual(2, try ZipArchive(data: padded[3...]).entries.count)
        XCTAssertThrowsError(try ZipArchive(data: Data("not a zip at all, just some text".utf8)))
        var cut = data
        cut.removeLast(30)
        XCTAssertThrowsError(try ZipArchive(data: cut))
    }

    func testEpubInSpineOrderWithTitleAndAuthor() throws {
        let container = """
        <?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
        <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>
        """
        let opf = """
        <?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0">
        <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>The Test &amp; Book</dc:title>
        <dc:creator opf:role="aut">Ann Author</dc:creator></metadata>
        <manifest><item id="toc" href="text/toc.xhtml" media-type="application/xhtml+xml"/>
        <item id="c1" href="text/chapter%201.xhtml" media-type="application/xhtml+xml"/>
        <item id="c2" href="text/c2.xhtml#start" media-type="application/xhtml+xml"/>
        <item id="pic" href="images/p.jpg" media-type="image/jpeg"/></manifest>
        <spine><itemref idref="toc"/><itemref idref="c2"/><itemref idref="c1"/></spine></package>
        """
        let toc = "<html><body><h2>Contents</h2><p><a href=\"chapter%201.xhtml#c1\">Chapter One</a></p>"
            + "<p><a href=\"c2.xhtml#c2\">Chapter Two</a></p></body></html>"
        let c1 = "<html><head><title>x</title></head><body><h2>Chapter One</h2>"
            + "<p><img src=\"../images/i.jpg\" alt=\"T\"/>he first chapter, in <i>italics</i>, comes second in the spine.</p></body></html>"
        let c2 = "<html><body><h2>Chapter Two</h2><p>The second chapter is read first. It has two sentences.</p></body></html>"
        let data = try makeZip([("mimetype", "application/epub+zip"), ("META-INF/container.xml", container),
                                ("OEBPS/content.opf", opf), ("OEBPS/text/toc.xhtml", toc),
                                ("OEBPS/text/chapter 1.xhtml", c1), ("OEBPS/text/c2.xhtml", c2)], stored: ["mimetype"])
        let d = try Extract.epub(data, fallbackTitle: "file", source: "file:///b.epub")
        XCTAssertEqual("The Test & Book", d.title)
        XCTAssertEqual("Ann Author", d.author)
        XCTAssertEqual("The Test & Book · Ann Author", d.displayTitle)
        XCTAssertEqual("epub", d.kind)
        XCTAssertEqual([Narration.HEAD + "Chapter Two", "The second chapter is read first. It has two sentences.",
                        Narration.HEAD + "Chapter One",
                        "The first chapter, in " + Narration.EM_ON + "italics" + Narration.EM_OFF + ", comes second in the spine."],
                       d.paragraphs)
        // through the file API too
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("steertext-\(UUID().uuidString).epub")
        try data.write(to: u)
        defer { try? FileManager.default.removeItem(at: u) }
        XCTAssertEqual(d.paragraphs, try Extract.document(fromFile: u).paragraphs)
    }

    func testDocxParagraphs() throws {
        let body = """
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
        <w:p><w:r><w:t>First</w:t></w:r><w:r><w:t xml:space="preserve"> paragraph &amp; more.</w:t></w:r></w:p>
        <w:p><w:r><w:t>Second</w:t><w:tab/><w:t>one.</w:t></w:r></w:p>
        <w:p></w:p>
        <w:p><w:r><w:t>Third.</w:t></w:r></w:p></w:body></w:document>
        """
        let core = "<cp:coreProperties xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:title>Memo</dc:title></cp:coreProperties>"
        let data = try makeZip([("[Content_Types].xml", "<Types/>"), ("word/document.xml", body), ("docProps/core.xml", core)])
        let d = try Extract.document(data: data, name: "memo.docx")
        XCTAssertEqual("Memo", d.title)
        XCTAssertEqual("docx", d.kind)
        XCTAssertEqual(["First paragraph & more.", "Second one.", "Third."], d.paragraphs)
        let untitled = try Extract.docx(try makeZip([("word/document.xml", body)]), fallbackTitle: "memo", source: nil)
        XCTAssertEqual("memo", untitled.title)
    }

    func testPlainTextAndHtmlFiles() throws {
        let t = try Extract.document(data: Data("Hello there.\n\nSecond.".utf8), name: "notes.txt")
        XCTAssertEqual(["Hello there.", "Second."], t.paragraphs)
        XCTAssertEqual("notes", t.title)
        let h = try Extract.document(data: Data("<title>Page</title><p>Hi &#233;t&#xE9;.</p>".utf8), name: "a.html")
        XCTAssertEqual("Page", h.title)
        XCTAssertEqual(["Page", "Hi été."], h.paragraphs)
        XCTAssertThrowsError(try Extract.document(data: Data(), name: "x.pdf"))
        XCTAssertThrowsError(try Extract.document(data: Data([0]), name: "x.bin", mime: "application/octet-stream"))
        XCTAssertEqual("Hello there.", Extract.fromText("Hello there.").title)
    }

    // ---------------------------------------------------------------- PdfLayout
    private func line(_ page: Int, _ text: String, _ l: Float, _ t: Float, _ r: Float, h: Float = 10) -> PdfLayout.Line {
        PdfLayout.Line(page: page, text: text, l: l, t: t, r: r, b: t + h, pw: 600, ph: 800)
    }

    func testPdfTwoColumnsTitleFirst() {
        var lines = [line(0, "A Study of Things", 150, 90, 450, h: 18)]
        for i in 0..<8 {
            let y = Float(150 + i * 12)
            lines.append(line(0, i == 7 ? "left ends here." : "left column line \(i) goes on", 50, y, 280))
            lines.append(line(0, i == 7 ? "right ends here." : "right column line \(i) goes on", 320, y, 550))
        }
        let paras = PdfLayout.layout(lines.shuffled())
        XCTAssertEqual(3, paras.count)
        XCTAssertTrue(paras[0].heading)
        XCTAssertEqual(Narration.HEAD + "A Study of Things", paras[0].text)
        XCTAssertTrue(paras[1].raw.hasPrefix("left column line 0 goes on left column line 1"))
        XCTAssertTrue(paras[1].raw.hasSuffix("left ends here."))
        XCTAssertTrue(paras[2].raw.hasPrefix("right column line 0"))
        XCTAssertEqual(8, paras[1].spans.count)
        let s0 = paras[1].spans[0]
        XCTAssertEqual(0, s0.start)
        XCTAssertEqual("left column line 0 goes on".jlength, s0.end)
        XCTAssertEqual(50 / 600, s0.l, accuracy: 1e-6)
        XCTAssertEqual(150 / 800, s0.t, accuracy: 1e-6)
    }

    func testPdfHeadersFootersAndPageNumbersDropped() {
        var lines: [PdfLayout.Line] = []
        for p in 0..<3 {
            lines.append(line(p, "Journal of Tests, Vol. 3", 50, 20, 300))     // running header
            lines.append(line(p, "\(p + 1)", 290, 760, 310))                   // page number
            lines.append(line(p, "Body text on page \(p + 1) that con-", 50, 100, 550))
            lines.append(line(p, "tinues on the next line.", 50, 112, 400))
        }
        let paras = PdfLayout.layout(lines)
        XCTAssertEqual(["Body text on page 1 that continues on the next line.",
                        "Body text on page 2 that continues on the next line.",
                        "Body text on page 3 that continues on the next line."], paras.map { $0.text })
        // as the Java: a line's span ends where its text (with the hyphen) ended; the next starts where the hyphen was
        XCTAssertEqual("Body text on page 1 that con-".jlength, paras[0].spans[0].end)
        XCTAssertEqual(paras[0].spans[0].end - 1, paras[0].spans[1].start)
    }

    func testPdfRowsMergeAndParagraphGaps() {
        let lines = [line(0, "Engineer", 50, 100, 150), line(0, "2019 – 2023", 450, 100, 550),
                     line(0, "Built things for a long time at a company that made things.", 50, 140, 550),
                     line(0, "Another paragraph after a blank line, with words.", 50, 175, 550)]
        let paras = PdfLayout.layout(lines)
        XCTAssertEqual("Engineer, 2019 – 2023", paras[0].raw)
        XCTAssertEqual(3, paras.count)
    }

    /// End-to-end on a real book: STEERTEXT_EPUB=/path/to/book.epub swift test --filter TextPipelineTests/testRealEpub
    /// prints its first paragraphs (skipped when the variable is unset; no download in the test).
    func testRealEpub() throws {
        guard let path = ProcessInfo.processInfo.environment["STEERTEXT_EPUB"] else { throw XCTSkip("STEERTEXT_EPUB not set") }
        let d = try Extract.document(fromFile: URL(fileURLWithPath: path))
        print("TITLE: \(d.displayTitle) (\(d.paragraphs.count) paragraphs, \(TextClean.sentences(d.paragraphs).count) sentences)")
        for p in d.paragraphs.prefix(8) { print("PARA: " + Narration.display(p).prefix(300)) }
        XCTAssertFalse(d.paragraphs.prefix(10).contains { Narration.display($0).range(of: "contents", options: .caseInsensitive) != nil })
    }
}
