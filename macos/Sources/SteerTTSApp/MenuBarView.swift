import ServiceManagement
import SwiftUI

/// The menu bar item: what is playing with its controls, the ways to read from other apps, and the reading list.
struct MenuBarView: View {
    @EnvironmentObject var app: AppModel
    @EnvironmentObject var narrator: Narrator
    @EnvironmentObject var library: Library
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let d = narrator.doc {
                VStack(alignment: .leading, spacing: 2) {
                    Text(d.title).font(.headline).lineLimit(2)
                    Text(narrator.preparing && narrator.playing ? "Getting the voice ready…" : "\(Int(narrator.progress * 100))% · \(app.voiceName(app.style.voice))")
                        .font(.caption).foregroundStyle(.secondary)
                }
                HStack(spacing: 22) {
                    Spacer()
                    Button { narrator.previousParagraph() } label: { Image(systemName: "backward.fill") }
                    Button { narrator.toggle() } label: {
                        Image(systemName: narrator.playing ? "pause.circle.fill" : "play.circle.fill").font(.system(size: 30))
                            .foregroundStyle(Theme.accent)
                    }
                    Button { narrator.nextParagraph() } label: { Image(systemName: "forward.fill") }
                    Spacer()
                }
                .buttonStyle(.plain).font(.title3)
                Divider()
            } else {
                HStack(spacing: 10) {
                    AppBadge(size: 30)
                    VStack(alignment: .leading) {
                        Text("SteerTTS").font(.headline)
                        Text("Nothing playing").font(.caption).foregroundStyle(.secondary)
                    }
                }
                Divider()
            }

            MenuRow(title: "Read Selection", keys: "⌃⌥R", icon: "text.cursor") { ReadAnywhere.readSelection() }
            MenuRow(title: "Read Clipboard", keys: "⌃⌥V", icon: "doc.on.clipboard") { ReadAnywhere.readClipboard() }

            if !library.docs.isEmpty {
                Divider()
                Text("Reading list").font(.caption).foregroundStyle(.secondary)
                ForEach(library.docs.prefix(4), id: \.id) { m in
                    MenuRow(title: m.title, keys: m.done ? "Finished" : "\(Int(m.progress() * 100))%", icon: "book") {
                        if let d = library.doc(m.id) { narrator.play(d) }
                    }
                }
            }

            Divider()
            HStack {
                Button("Open SteerTTS") { openWindow(id: "main"); NSApp.activate(ignoringOtherApps: true) }
                Spacer()
                Button("Quit") { NSApp.terminate(nil) }
            }
            .buttonStyle(.link)
        }
        .padding(14)
        .frame(width: 300)
    }
}

struct MenuRow: View {
    let title: String, keys: String, icon: String
    let action: () -> Void
    @State private var hover = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: icon).frame(width: 18).foregroundStyle(Theme.accent)
                Text(title).lineLimit(1)
                Spacer()
                Text(keys).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 6).padding(.vertical, 4)
            .background(hover ? Theme.accentSoft : .clear, in: RoundedRectangle(cornerRadius: 5))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
    }
}

/// Settings (⌘,): starting with the Mac, reading, and about.
struct SettingsView: View {
    @EnvironmentObject var app: AppModel
    @EnvironmentObject var narrator: Narrator
    @AppStorage("autoNext") private var autoNext = true
    @AppStorage("readingSize") private var size = 19.0
    @State private var atLogin = SMAppService.mainApp.status == .enabled

    var body: some View {
        TabView {
            Form {
                Toggle("Open at login (in the menu bar)", isOn: $atLogin)
                    .onChange(of: atLogin) { _, on in
                        do { if on { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() } }
                        catch { atLogin = SMAppService.mainApp.status == .enabled }
                    }
                Toggle("Play the next document in the reading list", isOn: $autoNext)
                Toggle("Dialogue in a second voice, characters in voices of their own", isOn: $narrator.dialogueVoice)
                Slider(value: $size, in: 14...30, step: 1) { Text("Reading text size") }
            }
            .formStyle(.grouped)
            .tabItem { Label("General", systemImage: "gearshape") }

            VStack(spacing: 12) {
                AppBadge(size: 72)
                Text("SteerTTS").font(.title2.weight(.semibold))
                Text("Version \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev")")
                    .foregroundStyle(.secondary)
                Text("A voice that runs on this Mac: books, articles and anything you select, read aloud with no account and nothing sent anywhere.")
                    .multilineTextAlignment(.center).foregroundStyle(.secondary).frame(maxWidth: 360)
                Link("whissle.ai", destination: URL(string: "https://whissle.ai")!)
            }
            .padding(24)
            .tabItem { Label("About", systemImage: "info.circle") }
        }
        .frame(width: 480, height: 300)
    }
}
