import AppKit
import Foundation
import SteerEngine
import SteerText

/// The app's state: the engine (on its own queue), the voice library, the user's style, the reading list and the
/// narrator. Everything lives in ~/Library/Application Support/SteerTTS: the reading list as one JSON file per
/// document (library/), the style as style.json. Nothing is sent anywhere: the voice runs on this Mac.
@MainActor
final class AppModel: ObservableObject {
    static let shared = AppModel()

    let support: URL
    let modelDir: URL
    let voices: Voices
    let library: Library
    private(set) var narrator: Narrator!

    /// The user's base style (voice, style tag, controls, speed), used for everything the app says.
    @Published var style: Style { didSet { style.save(to: support.appendingPathComponent("style.json")) } }
    @Published var section: Section? = .library
    @Published var opening: String?            // "Downloading Alice's Adventures…", while something is being added
    @Published var problem: String?            // shown as an alert

    @Published var sheet: Sheet?

    enum Section: Hashable { case library, freeBooks, assistant, voices, anywhere }
    enum Sheet: String, Identifiable { case link, text; var id: String { rawValue } }

    private let engineQueue = DispatchQueue(label: "steertts.engine", qos: .userInitiated)

    private init() {
        let fm = FileManager.default
        support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("SteerTTS")
        try? fm.createDirectory(at: support, withIntermediateDirectories: true)
        // the model inside the app; next to the sources when run from `swift run`
        let bundled = Bundle.main.resourceURL!.appendingPathComponent("model")
        modelDir = fm.fileExists(atPath: bundled.appendingPathComponent("config.json").path) ? bundled
            : URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .appendingPathComponent("../../../c/models/steer-tts-edge").standardized
        voices = Voices(voicesJSON: modelDir.appendingPathComponent("voices.json"), store: UserDefaults.standard)
        style = Style.load(from: support.appendingPathComponent("style.json"))
        library = Library(dir: support.appendingPathComponent("library"))
        narrator = Narrator(app: self)
        narrator.restoreLast()
    }

    /// Runs f on the engine's queue with the engine (loaded on first use, in a few milliseconds).
    nonisolated func onEngine(_ f: @escaping (Engine) -> Void) {
        engineQueue.async { [self] in
            do {
                let e = try loadedEngine()
                f(e)
            } catch {
                DispatchQueue.main.async { self.problem = "The voice could not start: \(error.localizedDescription)" }
            }
        }
    }

    private nonisolated func loadedEngine() throws -> Engine {
        if let e = engineBox.engine { return e }
        let e = try Engine(modelDir: modelDir, threads: 4)
        engineBox.engine = e
        return e
    }

    private nonisolated let engineBox = EngineBox()

    /// The voice's name for people ("Philip"), from the voice library.
    func voiceName(_ id: String?) -> String { voices.nameOf(engineVoice(id)) }

    /// The voice id a style's voice means: "female" / "male" / nil are the featured defaults (as on Android).
    func engineVoice(_ v: String?) -> String {
        if v == nil || v == "female" { return Voices.FEATURED[0] }
        if v == "male" { return Voices.FEATURED[1] }
        return v!
    }

    // ---------------------------------------------------------------- adding things to hear

    /// Text from anywhere (a selection, the clipboard, typed): into the reading list as a snippet, and read now.
    func read(text: String, title: String? = nil, play: Bool = true) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        if let url = URL(string: t), url.scheme?.hasPrefix("http") == true, !t.contains(" ") {
            open(link: url, play: play)
            return
        }
        let e = Extract.fromText(t, title: title)
        add(e, play: play)
    }

    /// A file: PDF, EPUB, Word, HTML, Markdown or text.
    func open(file url: URL, play: Bool = true) {
        opening = "Opening \(url.lastPathComponent)…"
        Task.detached(priority: .userInitiated) {
            do {
                if url.pathExtension.lowercased() == "pdf" {
                    let r = try PdfImport.read(url)
                    await MainActor.run {
                        let d = Doc.createPdf(id: UUID().uuidString.lowercased(), title: r.title, source: url.path,
                                              paragraphs: r.paragraphs, spans: r.spans)
                        self.library.copyPdf(url, for: d)
                        self.finishAdding(d, play: play)
                    }
                } else {
                    let e = try Extract.document(fromFile: url)
                    await MainActor.run { self.add(e, play: play) }
                }
            } catch {
                await MainActor.run { self.failedAdding(error) }
            }
        }
    }

    /// A web page: its article.
    func open(link url: URL, play: Bool = true) {
        opening = "Reading \(url.host ?? "the page")…"
        Task.detached(priority: .userInitiated) {
            do {
                var req = URLRequest(url: url, timeoutInterval: 20)
                req.setValue("Mozilla/5.0 (Macintosh) SteerTTS", forHTTPHeaderField: "User-Agent")
                let (data, resp) = try await URLSession.shared.data(for: req)
                let type = (resp as? HTTPURLResponse)?.value(forHTTPHeaderField: "Content-Type") ?? ""
                let e: ExtractedDoc
                if type.contains("pdf") || url.pathExtension.lowercased() == "pdf" {
                    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".pdf")
                    try data.write(to: tmp)
                    await MainActor.run { self.open(file: tmp, play: play) }
                    return
                } else if type.contains("epub") || url.pathExtension.lowercased() == "epub" {
                    e = try Extract.epub(data, fallbackTitle: url.lastPathComponent, source: url.absoluteString)
                } else {
                    e = Extract.fromArticle(Article.extract(Article.text(of: data, contentType: type), url: url), source: url.absoluteString)
                }
                await MainActor.run { self.add(e, play: play) }
            } catch {
                await MainActor.run { self.failedAdding(error) }
            }
        }
    }

    /// A free book from Project Gutenberg: the copy already in the list, or downloaded.
    func open(book b: Gutenberg.Book) {
        if let have = library.find(source: b.source, titlePrefix: b.title) {
            section = .library
            narrator.play(have)
            return
        }
        opening = "Downloading \(b.title)…"
        Task.detached(priority: .userInitiated) {
            do {
                let data = try await Gutenberg.epub(b)
                let e = try Extract.epub(data, fallbackTitle: b.title, source: b.source)
                await MainActor.run { self.section = .library; self.add(e, play: true) }
            } catch {
                await MainActor.run { self.failedAdding(error) }
            }
        }
    }

    private func add(_ e: ExtractedDoc, play: Bool) {
        let d = Doc.create(from: e)
        finishAdding(d, play: play)
    }

    private func finishAdding(_ d: Doc, play: Bool) {
        opening = nil
        guard !d.sentences.isEmpty else {
            problem = "There is nothing to read in this."
            return
        }
        let saved = library.saveOrExisting(d)
        if play { narrator.play(saved) }
    }

    private func failedAdding(_ error: Error) {
        opening = nil
        problem = (error as? Extract.Unsupported)?.message ?? error.localizedDescription
    }
}

/// The engine, owned by the engine queue.
final class EngineBox: @unchecked Sendable {
    var engine: Engine?
}
