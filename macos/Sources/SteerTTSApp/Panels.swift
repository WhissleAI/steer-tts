import AppKit
import SteerEngine
import SteerText
import SwiftUI
import UniformTypeIdentifiers

/// Adding things to hear: the open panel, and the link and text sheets (shown by MainView).
@MainActor
enum Panels {
    static let types: [UTType] = [.pdf, .epub, .html, .plainText, .text,
                                  UTType("org.openxmlformats.wordprocessingml.document")!,
                                  UTType("net.daringfireball.markdown") ?? .plainText]

    static func openFiles() {
        let p = NSOpenPanel()
        p.allowedContentTypes = types
        p.allowsMultipleSelection = true
        p.message = "Choose a book, a PDF, a Word document, a web page or text to listen to."
        guard p.runModal() == .OK else { return }
        for (i, u) in p.urls.enumerated() { AppModel.shared.open(file: u, play: i == 0) }
        AppModel.shared.section = .library
        showMain()
    }

    static func openLink() { showMain(); AppModel.shared.sheet = .link }
    static func typeText() { showMain(); AppModel.shared.sheet = .text }

    static func showMain() {
        NSApp.activate(ignoringOtherApps: true)
        if let w = NSApp.windows.first(where: { $0.identifier?.rawValue == "main" || $0.title == "SteerTTS" && $0.canBecomeMain }) {
            w.makeKeyAndOrderFront(nil)
        }
    }
}

struct LinkSheet: View {
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var link = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Listen to a web page").font(.title3.weight(.semibold))
            Text("Paste a link to an article, a blog post or a book. SteerTTS reads the article itself, without menus and ads.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            TextField("https://", text: $link).textFieldStyle(.roundedBorder).onSubmit(go)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Listen", action: go).keyboardShortcut(.defaultAction).disabled(url == nil)
            }
        }
        .padding(20)
        .frame(width: 460)
        .onAppear {
            if let s = NSPasteboard.general.string(forType: .string), Article.isLink(s) { link = s.trimmingCharacters(in: .whitespacesAndNewlines) }
        }
    }

    private var url: URL? {
        let s = link.trimmingCharacters(in: .whitespacesAndNewlines)
        let u = URL(string: s.contains("://") ? s : "https://" + s)
        return u?.host != nil && s.contains(".") ? u : nil
    }

    private func go() {
        guard let url else { return }
        dismiss()
        app.section = .library
        app.open(link: url)
    }
}

struct TextSheet: View {
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var text = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Type or paste text").font(.title3.weight(.semibold))
            TextEditor(text: $text)
                .font(.body)
                .frame(minHeight: 220)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
            HStack {
                Text("Times, dates, amounts and links are read the way people say them.").font(.callout).foregroundStyle(.secondary)
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Listen") {
                    dismiss()
                    app.section = .library
                    app.read(text: text)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 560)
    }
}

/// "Export Audio…": the whole document, performed as in the player, into an .m4a (or .wav) file. Runs on an engine of
/// its own, so listening goes on meanwhile; on this Mac a novel takes a few minutes.
@MainActor
final class Exporter: ObservableObject {
    static let shared = Exporter()
    @Published private(set) var title: String?
    @Published private(set) var fraction = 0.0
    private var cancelled = false

    func export(_ d: Doc) {
        guard title == nil else { return }
        let p = NSSavePanel()
        p.allowedContentTypes = [.mpeg4Audio, .wav]
        p.nameFieldStringValue = d.title.replacingOccurrences(of: "/", with: "-") + ".m4a"
        p.message = "The whole document, read in your voice and style. AAC (.m4a) is about 30 MB per hour; WAV is uncompressed."
        guard p.runModal() == .OK, let url = p.url else { return }
        export(d, to: url)
    }

    func export(_ d: Doc, to url: URL) {
        let app = AppModel.shared
        let style = app.style, speed = app.narrator.speed, dialogue = app.narrator.dialogueVoice, voices = app.voices
        let modelDir = app.modelDir
        title = d.title
        fraction = 0
        cancelled = false
        Task.detached(priority: .utility) { [weak self] in
            do {
                let engine = try Engine(modelDir: modelDir, threads: 4)
                let writer = try AudioFileWriter(url: url)
                let perf = Performer(doc: d, style: style, speed: speed, dialogueVoices: dialogue, voices: voices, engine: engine)
                for i in 0..<d.sentences.count {
                    if await self?.cancelled ?? true { try? FileManager.default.removeItem(at: url); break }
                    if let a = try perf.render(i, with: engine) { try writer.write(a) }
                    if i % 10 == 0 { await self?.progress(Double(i + 1) / Double(d.sentences.count)) }
                }
                await self?.done(url)
            } catch {
                await self?.failed(error)
            }
        }
    }

    func cancel() { cancelled = true }

    private func progress(_ f: Double) { fraction = f }

    private func done(_ url: URL) {
        let wasCancelled = cancelled
        title = nil
        if !wasCancelled { NSWorkspace.shared.activateFileViewerSelecting([url]) }
    }

    private func failed(_ e: Error) {
        title = nil
        AppModel.shared.problem = "The audio could not be saved: \(e.localizedDescription)"
    }
}
