import SteerText
import SwiftUI

/// The reading list beside the document being read. The list: most recently opened first, with progress and time
/// left; the reader: the text, the sentence being heard highlighted and followed; click a sentence to read from there.
struct LibraryView: View {
    @EnvironmentObject var app: AppModel
    @EnvironmentObject var library: Library
    @EnvironmentObject var narrator: Narrator
    @State private var selected: String?

    var body: some View {
        if library.docs.isEmpty {
            EmptyLibrary()
        } else {
            HSplitView {
                DocList(selected: $selected)
                    .frame(minWidth: 230, idealWidth: 270, maxWidth: 320)
                Group {
                    if let id = selected ?? narrator.doc?.id ?? library.docs.first?.id, let d = library.doc(id) {
                        ReaderView(doc: d).id(d.id)
                    } else {
                        Text("Choose something to read").foregroundStyle(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }
                .frame(minWidth: 420, maxWidth: .infinity, maxHeight: .infinity)
            }
            .onChange(of: narrator.doc?.id) { _, id in if let id { selected = id } }
            .toolbar { AddMenu() }
        }
    }
}

/// "+" in the toolbar: the four ways to add something to hear.
struct AddMenu: ToolbarContent {
    var body: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            Menu {
                Button("Open File…", systemImage: "doc") { Panels.openFiles() }
                Button("Open Link…", systemImage: "link") { Panels.openLink() }
                Button("From Clipboard", systemImage: "doc.on.clipboard") { ReadAnywhere.readClipboard() }
                Button("Type Text…", systemImage: "character.cursor.ibeam") { Panels.typeText() }
            } label: {
                Label("Add", systemImage: "plus")
            }
            .help("Add something to listen to")
        }
    }
}

struct DocList: View {
    @EnvironmentObject var app: AppModel
    @EnvironmentObject var library: Library
    @EnvironmentObject var narrator: Narrator
    @Binding var selected: String?

    var body: some View {
        List(selection: $selected) {
            ForEach(library.docs, id: \.id) { m in
                DocRow(meta: m, playing: narrator.doc?.id == m.id && narrator.playing, pace: narrator.speed * app.style.pace)
                    .tag(m.id)
                    .contextMenu {
                        Button("Play") { if let d = library.doc(m.id) { narrator.play(d) } }
                        Button("Play from the Start") { if let d = library.doc(m.id) { narrator.play(d, from: 0) } }
                        Button("Export Audio…") { if let d = library.doc(m.id) { Exporter.shared.export(d) } }
                        if !m.source.isEmpty, m.source.hasPrefix("http") {
                            Button("Open Original") { if let u = URL(string: m.source) { NSWorkspace.shared.open(u) } }
                        }
                        Divider()
                        Button("Remove from List", role: .destructive) { remove(m.id) }
                    }
            }
            .onDelete { idx in for i in idx { remove(library.docs[i].id) } }
        }
        .listStyle(.inset)
        .onDeleteCommand { if let s = selected { remove(s) } }
    }

    private func remove(_ id: String) {
        if narrator.doc?.id == id { narrator.stop() }
        if selected == id { selected = nil }
        library.delete(id)
    }
}

struct DocRow: View {
    let meta: Doc.Meta
    let playing: Bool
    let pace: Float

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 15))
                .foregroundStyle(Theme.accent)
                .frame(width: 30, height: 30)
                .background(Theme.accentSoft, in: RoundedRectangle(cornerRadius: 7))
            VStack(alignment: .leading, spacing: 3) {
                Text(meta.title).font(.body.weight(.medium)).lineLimit(2)
                if !meta.author.isEmpty { Text(meta.author).font(.callout).foregroundStyle(.secondary).lineLimit(1) }
                HStack(spacing: 6) {
                    if playing { Image(systemName: "speaker.wave.2.fill").foregroundStyle(Theme.accent) }
                    ProgressView(value: Double(meta.progress())).controlSize(.mini).frame(width: 60)
                    Text(detail).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private var icon: String {
        switch meta.kind {
        case "epub": return "book.closed"
        case "pdf": return "doc.richtext"
        case "web": return "globe"
        case "docx": return "doc.text"
        default: return "text.alignleft"
        }
    }

    private var detail: String {
        if meta.done { return "Finished" }
        let m = meta.minutesLeft(pace)
        let left = m >= 60 ? "\(m / 60) h \(m % 60) min" : "\(max(1, m)) min"
        return meta.pos == 0 ? left : left + " left"
    }
}

struct EmptyLibrary: View {
    @EnvironmentObject var app: AppModel

    var body: some View {
        VStack(spacing: 22) {
            AppBadge(size: 84)
            VStack(spacing: 6) {
                Text("Listen to anything").font(.largeTitle.weight(.semibold))
                Text("Books, PDFs, articles and notes, read aloud by a voice that runs on this Mac.")
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 12) {
                AddTile(title: "Open a File", detail: "PDF, EPUB, Word, text", icon: "doc") { Panels.openFiles() }
                AddTile(title: "A Web Page", detail: "Paste a link", icon: "link") { Panels.openLink() }
                AddTile(title: "From Clipboard", detail: "What you copied", icon: "doc.on.clipboard") { ReadAnywhere.readClipboard() }
                AddTile(title: "Type Text", detail: "Or paste it", icon: "character.cursor.ibeam") { Panels.typeText() }
            }
            Button("Or pick one of 70,000 free books") { app.section = .freeBooks }
                .buttonStyle(.plain).foregroundStyle(Theme.accent)
            Text("You can also drop files here, or select text in any app and press ⌃⌥R.")
                .font(.callout).foregroundStyle(.tertiary)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .toolbar { AddMenu() }
    }
}

struct AddTile: View {
    let title: String, detail: String, icon: String
    let action: () -> Void
    @State private var hover = false

    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon).font(.system(size: 22)).foregroundStyle(Theme.accent).frame(height: 28)
                Text(title).font(.headline)
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
            .frame(width: 140, height: 104)
            .background(hover ? Theme.accentSoft : Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(.quaternary))
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
    }
}
