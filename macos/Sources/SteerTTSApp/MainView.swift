import AVKit
import SteerText
import SwiftUI

/// The window: a sidebar (Reading List, Free Books, Voices, Read Anywhere) and the section; the player bar along the
/// bottom whenever something is open.
struct MainView: View {
    @EnvironmentObject var app: AppModel
    @EnvironmentObject var narrator: Narrator
    @ObservedObject private var exporter = Exporter.shared

    var body: some View {
        NavigationSplitView {
            List(selection: $app.section) {
                Section("Listen") {
                    Label("Reading List", systemImage: "books.vertical").tag(AppModel.Section.library)
                    Label("Free Books", systemImage: "building.columns").tag(AppModel.Section.freeBooks)
                }
                Section("SteerTTS") {
                    Label("Voices", systemImage: "person.wave.2").tag(AppModel.Section.voices)
                    Label("Read Anywhere", systemImage: "text.viewfinder").tag(AppModel.Section.anywhere)
                }
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 200, max: 260)
        } detail: {
            Group {
                switch app.section ?? .library {
                case .library: LibraryView()
                case .freeBooks: FreeBooksView()
                case .voices: VoicesView()
                case .anywhere: AnywhereView()
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                VStack(spacing: 0) {
                    if let t = exporter.title { ExportBar(title: t, fraction: exporter.fraction) }
                    if narrator.doc != nil { TransportBar() }
                }
            }
        }
        .tint(Theme.accent)
        .sheet(item: $app.sheet) { s in
            switch s {
            case .link: LinkSheet()
            case .text: TextSheet()
            }
        }
        .alert("SteerTTS", isPresented: Binding(get: { app.problem != nil }, set: { if !$0 { app.problem = nil } })) {
            Button("OK") { app.problem = nil }
        } message: {
            Text(app.problem ?? "")
        }
        .overlay(alignment: .top) {
            if let o = app.opening {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(o)
                }
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(.regularMaterial, in: Capsule())
                .padding(.top, 10)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.2), value: app.opening)
        .onDrop(of: [.fileURL, .url, .plainText], isTargeted: nil) { providers in
            Drops.handle(providers)
            return true
        }
    }
}

/// Files, links and text dropped on the window.
enum Drops {
    @MainActor static func handle(_ providers: [NSItemProvider]) {
        for p in providers {
            if p.hasItemConformingToTypeIdentifier("public.file-url") {
                _ = p.loadObject(ofClass: URL.self) { u, _ in
                    if let u { Task { @MainActor in AppModel.shared.section = .library; AppModel.shared.open(file: u) } }
                }
            } else if p.canLoadObject(ofClass: URL.self) {
                _ = p.loadObject(ofClass: URL.self) { u, _ in
                    if let u { Task { @MainActor in AppModel.shared.section = .library; AppModel.shared.open(link: u) } }
                }
            } else if p.canLoadObject(ofClass: String.self) {
                _ = p.loadObject(ofClass: String.self) { s, _ in
                    if let s { Task { @MainActor in AppModel.shared.section = .library; AppModel.shared.read(text: s) } }
                }
            }
        }
    }
}

// MARK: - The player bar

struct TransportBar: View {
    @EnvironmentObject var app: AppModel
    @EnvironmentObject var narrator: Narrator
    @State private var scrub: Double?

    var body: some View {
        if let d = narrator.doc {
            HStack(spacing: 18) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(d.title).font(.headline).lineLimit(1)
                    Text(status(d)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                .frame(width: 210, alignment: .leading)

                HStack(spacing: 14) {
                    Button { narrator.previousParagraph() } label: { Image(systemName: "backward.fill") }
                        .help("Previous paragraph (⌘←)")
                    Button { narrator.toggle() } label: {
                        Image(systemName: narrator.playing ? "pause.circle.fill" : "play.circle.fill")
                            .font(.system(size: 34)).foregroundStyle(Theme.accent)
                    }
                    .help(narrator.playing ? "Pause (Space)" : "Play (Space)")
                    Button { narrator.nextParagraph() } label: { Image(systemName: "forward.fill") }
                        .help("Next paragraph (⌘→)")
                }
                .buttonStyle(.plain)
                .font(.title3)

                VStack(spacing: 2) {
                    Slider(value: Binding(get: { scrub ?? Double(narrator.heard) },
                                          set: { scrub = $0 }),
                           in: 0...Double(max(1, d.sentences.count - 1))) { editing in
                        if !editing, let s = scrub { narrator.seek(Int(s)); scrub = nil }
                    }
                    .controlSize(.small)
                    HStack {
                        Text("\(Int(narrator.progress * 100))%")
                        Spacer()
                        Text(timeLeft)
                    }
                    .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                }
                .frame(minWidth: 160)

                Menu {
                    ForEach([0.75, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0], id: \.self) { s in
                        Button { narrator.speed = Float(s) } label: {
                            Text(speedLabel(Float(s)) + (abs(narrator.speed - Float(s)) < 0.01 ? "  ✓" : ""))
                        }
                    }
                } label: {
                    Text(speedLabel(narrator.speed)).monospacedDigit()
                }
                .menuStyle(.borderlessButton).fixedSize()
                .help("Reading speed (⌘[ and ⌘])")

                VoiceMenu()

                Menu {
                    Button("Off") { narrator.sleep(after: nil) }
                    ForEach([5, 10, 15, 30, 45, 60], id: \.self) { m in Button("In \(m) minutes") { narrator.sleep(after: m) } }
                } label: {
                    Image(systemName: narrator.sleepAt == nil ? "moon.zzz" : "moon.zzz.fill")
                }
                .menuStyle(.borderlessButton).fixedSize()
                .help(narrator.sleepAt.map { "Stops at \($0.formatted(date: .omitted, time: .shortened))" } ?? "Sleep timer")

                RoutePicker(player: narrator.player.player)
                    .frame(width: 26, height: 22)
                    .help("Play on a HomePod, Apple TV or AirPlay speaker")
            }
            .padding(.horizontal, 18).padding(.vertical, 10)
            .background(.bar)
            .overlay(alignment: .top) { Divider() }
        }
    }

    private func status(_ d: Doc) -> String {
        if let p = narrator.problem { return p }
        if narrator.preparing && narrator.playing { return "Getting the voice ready…" }
        return d.author.isEmpty ? app.voiceName(app.style.voice) + " reading" : d.author
    }

    private var timeLeft: String {
        let s = Int(narrator.secondsLeft)
        if s >= 3600 { return "\(s / 3600) h \(s % 3600 / 60) min left" }
        return s >= 60 ? "\(s / 60) min left" : "\(s) s left"
    }

    private func speedLabel(_ s: Float) -> String {
        String(format: s == Float(Int(s)) ? "%.0f×" : (s * 100).truncatingRemainder(dividingBy: 10) == 0 ? "%.1f×" : "%.2f×", s)
    }
}

/// The reader's voice, from the featured voices (the full library is on the Voices page).
struct VoiceMenu: View {
    @EnvironmentObject var app: AppModel

    var body: some View {
        Menu {
            ForEach(app.voices.featured(), id: \.id) { v in
                Button {
                    let s = app.style.copy()
                    s.voice = v.id
                    app.style = s
                    app.narrator.restyle()
                } label: {
                    Text("\(v.name) · \(v.genderLabel), \(v.describe().lowercased())" + (app.style.voice == v.id ? "  ✓" : ""))
                }
            }
            Divider()
            Button("More Voices…") { app.section = .voices }
        } label: {
            Label(app.voiceName(app.style.voice), systemImage: "person.wave.2")
        }
        .menuStyle(.borderlessButton).fixedSize()
        .help("The reading voice")
    }
}

/// AirPlay: the system's route picker for the player.
struct RoutePicker: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVRoutePickerView {
        let v = AVRoutePickerView()
        v.player = player
        v.isRoutePickerButtonBordered = false
        return v
    }

    func updateNSView(_ v: AVRoutePickerView, context: Context) { v.player = player }
}

struct ExportBar: View {
    let title: String
    let fraction: Double

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "waveform.badge.plus").foregroundStyle(Theme.accent)
            Text("Saving “\(title)” as audio").lineLimit(1)
            ProgressView(value: fraction).frame(maxWidth: 260)
            Text("\(Int(fraction * 100))%").monospacedDigit().foregroundStyle(.secondary)
            Spacer()
            Button("Cancel") { Exporter.shared.cancel() }
        }
        .font(.callout)
        .padding(.horizontal, 18).padding(.vertical, 8)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }
}
