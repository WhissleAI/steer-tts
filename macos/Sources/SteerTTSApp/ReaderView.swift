import SteerText
import SwiftUI

/// A document as text to read along: headings set as headings, italics kept, the sentence being heard highlighted
/// and kept in view while reading moves on (unless the reader scrolls away; "Back to Reading" returns). Click any
/// sentence to read from there.
struct ReaderView: View {
    let doc: Doc
    @EnvironmentObject var app: AppModel
    @EnvironmentObject var narrator: Narrator
    @AppStorage("readingSize") private var size = 19.0
    @State private var paragraphs: [[Int]] = []
    @State private var follow = true

    private var isCurrent: Bool { narrator.doc?.id == doc.id }
    private var heard: Int { isCurrent ? narrator.heard : doc.pos }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    header.padding(.bottom, 28)
                    LazyVStack(alignment: .leading, spacing: size * 0.85) {
                        ForEach(paragraphs.indices, id: \.self) { p in
                            let idx = paragraphs[p]
                            ParagraphView(doc: doc, sentences: idx, current: idx.contains(heard) ? heard : nil, size: size)
                                .equatable()
                                .id(p)
                        }
                    }
                }
                .padding(.horizontal, 48).padding(.vertical, 36)
                .frame(maxWidth: 760, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
            .environment(\.openURL, OpenURLAction { url in
                if url.scheme == "steertts", let i = Int(url.lastPathComponent) {
                    follow = true
                    if isCurrent && narrator.playing { narrator.seek(i) } else { narrator.play(doc, from: i) }
                }
                return .handled
            })
            .onAppear {
                paragraphs = group(doc)
                DispatchQueue.main.async { scroll(proxy, animated: false) }
            }
            .onChange(of: narrator.heard) { _, _ in if isCurrent && follow { scroll(proxy, animated: true) } }
            .overlay(alignment: .bottomTrailing) {
                if !follow && isCurrent {
                    Button { follow = true; scroll(proxy, animated: true) } label: {
                        Label("Back to Reading", systemImage: "arrow.down.to.line")
                    }
                    .controlSize(.large)
                    .padding(16)
                }
            }
            .onScrollWheel { follow = false }
        }
        .toolbar {
            ToolbarItemGroup {
                Button { size = max(14, size - 1) } label: { Image(systemName: "textformat.size.smaller") }.help("Smaller text")
                Button { size = min(30, size + 1) } label: { Image(systemName: "textformat.size.larger") }.help("Larger text")
                Button { Exporter.shared.export(doc) } label: { Image(systemName: "square.and.arrow.down") }
                    .help("Save as audio (an audiobook file)")
            }
        }
        .navigationTitle(doc.title)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(doc.title).font(.system(size: size * 1.6, weight: .bold, design: .serif)).textSelection(.enabled)
            if !doc.author.isEmpty { Text(doc.author).font(.system(size: size * 0.95, design: .serif)).foregroundStyle(.secondary) }
            HStack(spacing: 12) {
                Button {
                    if isCurrent { narrator.toggle() } else { narrator.play(doc) }
                } label: {
                    Label(isCurrent && narrator.playing ? "Pause" : (doc.pos > 0 && !doc.done ? "Resume" : "Play"),
                          systemImage: isCurrent && narrator.playing ? "pause.fill" : "play.fill")
                        .frame(minWidth: 80)
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                Text(meta).font(.callout).foregroundStyle(.secondary)
            }
            .padding(.top, 6)
        }
    }

    private var meta: String {
        let m = doc.minutesLeft(narrator.speed * app.style.pace)
        let time = m >= 60 ? "\(m / 60) h \(m % 60) min" : "\(max(1, m)) min"
        let kind = ["epub": "Book", "pdf": "PDF", "web": "Web page", "docx": "Word document", "markdown": "Markdown"][doc.kind] ?? "Text"
        return "\(kind) · \(time)\(doc.pos > 0 && !doc.done ? " left" : "")"
    }

    private func scroll(_ proxy: ScrollViewProxy, animated: Bool) {
        guard heard < doc.paraOf.count else { return }
        let p = doc.paraOf[heard]
        if animated { withAnimation(.easeInOut(duration: 0.35)) { proxy.scrollTo(p, anchor: UnitPoint(x: 0.5, y: 0.08)) } }
        else { proxy.scrollTo(p, anchor: UnitPoint(x: 0.5, y: 0.08)) }
    }

    /// Sentence indices per paragraph.
    private func group(_ d: Doc) -> [[Int]] {
        var out: [[Int]] = Array(repeating: [], count: d.paragraphs.count)
        for (i, p) in d.paraOf.enumerated() where p < out.count { out[p].append(i) }
        return out.filter { !$0.isEmpty }
    }
}

/// One paragraph: its sentences as one text, each a link to read from there; the one being heard highlighted.
struct ParagraphView: View, Equatable {
    let doc: Doc
    let sentences: [Int]
    let current: Int?
    let size: Double

    static func == (a: ParagraphView, b: ParagraphView) -> Bool {
        a.doc.id == b.doc.id && a.sentences == b.sentences && a.current == b.current && a.size == b.size
    }

    var body: some View {
        let heading = sentences.first.map { doc.sentences[$0].contains(Narration.HEAD) } ?? false
        Text(attributed(heading: heading))
            .font(heading ? .system(size: size * 1.25, weight: .semibold, design: .serif) : Theme.reading(size))
            .lineSpacing(size * 0.3)
            .tint(.primary)
            .padding(.top, heading ? size * 0.8 : 0)
            .frame(maxWidth: .infinity, alignment: .leading)
            .textSelection(.enabled)
    }

    private func attributed(heading: Bool) -> AttributedString {
        var out = AttributedString()
        for (k, i) in sentences.enumerated() {
            var s = Self.styled(doc.sentences[i])
            s.link = URL(string: "steertts://sentence/\(i)")
            if i == current {
                s.backgroundColor = Theme.accentSoft
                s.foregroundColor = Theme.accent
            }
            if k > 0 { out += AttributedString(" ") }
            out += s
        }
        return out
    }

    /// The sentence as shown: markers out, emphasis in italics.
    static func styled(_ raw: String) -> AttributedString {
        var out = AttributedString()
        var italic = false
        var buf = ""
        func flush() {
            guard !buf.isEmpty else { return }
            var a = AttributedString(buf)
            if italic { a.inlinePresentationIntent = .emphasized }
            out += a
            buf = ""
        }
        for ch in raw {
            switch String(ch) {
            case Narration.EM_ON: flush(); italic = true
            case Narration.EM_OFF: flush(); italic = false
            case Narration.HEAD: break
            default: buf.append(ch)
            }
        }
        flush()
        return out
    }
}

extension View {
    /// Calls f when the mouse wheel or trackpad scrolls over this view (the reader stops following the voice).
    func onScrollWheel(_ f: @escaping () -> Void) -> some View { background(ScrollWheelWatcher(onScroll: f)) }
}

private struct ScrollWheelWatcher: NSViewRepresentable {
    let onScroll: () -> Void

    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        context.coordinator.monitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak v] e in
            if let v, let w = v.window, e.window == w, v.bounds.contains(v.convert(e.locationInWindow, from: nil)),
               e.phase == .began || e.phase == .changed || e.momentumPhase == [] && e.phase == [] {
                onScroll()
            }
            return e
        }
        return v
    }

    func updateNSView(_ v: NSView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var monitor: Any?
        deinit { if let m = monitor { NSEvent.removeMonitor(m) } }
    }
}
