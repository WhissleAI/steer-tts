import SwiftUI

/// Read Anywhere: every way SteerTTS reads from other apps, each with what it needs and whether it is ready.
struct AnywhereView: View {
    @EnvironmentObject var app: AppModel
    @EnvironmentObject var narrator: Narrator
    @AppStorage("menuBarItem") private var menuBarItem = true
    @AppStorage("shortcuts") private var shortcuts = true
    @State private var trusted = Selection.trusted
    private let tick = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Read Anywhere").font(.title2.weight(.semibold))
                    Text("Hear text from any app on this Mac, in your voice. Everything is read on this Mac; nothing is sent anywhere.")
                        .foregroundStyle(.secondary)
                }
                .padding(.bottom, 4)

                Feature(icon: "text.cursor", title: "Read what you select", keys: "⌃⌥R",
                        detail: "Select text in any app (a web page, a mail, a document) and press Control-Option-R.",
                        state: !shortcuts ? .off("Shortcuts are off") : trusted ? .ready : .needs("Needs Accessibility access")) {
                    if !trusted {
                        Button("Allow…") {
                            Selection.askForAccess()
                            NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!)
                        }
                    }
                }

                Feature(icon: "contextualmenu.and.cursorarrow", title: "Services menu", keys: nil,
                        detail: "Right-click selected text and choose Services > Read with SteerTTS, or Add to SteerTTS Reading List. Works everywhere, no permission needed. You can give it your own shortcut in Keyboard settings.",
                        state: .ready) {
                    Button("Keyboard Shortcuts…") {
                        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.Keyboard-Settings.extension?Shortcuts")!)
                    }
                }

                Feature(icon: "doc.on.clipboard", title: "Read what you copied", keys: "⌃⌥V",
                        detail: "Copy text or a link anywhere and press Control-Option-V. A link is read as its article.",
                        state: shortcuts ? .ready : .off("Shortcuts are off")) {
                    Button("Read Clipboard Now") { ReadAnywhere.readClipboard() }
                }

                Feature(icon: "playpause", title: "Play and pause from anywhere", keys: "⌃⌥Space",
                        detail: "Or use the play key, AirPods and Control Center: SteerTTS shows there like any player.",
                        state: shortcuts ? .ready : .off("Shortcuts are off")) {
                    Toggle("Shortcuts", isOn: $shortcuts).toggleStyle(.switch).labelsHidden()
                        .onChange(of: shortcuts) { _, _ in (NSApp.delegate as? AppDelegate)?.registerHotkeys() }
                }

                Feature(icon: "menubar.rectangle", title: "Menu bar", keys: nil,
                        detail: "What is playing, the controls and Read Selection, from the menu bar, with the window closed.",
                        state: menuBarItem ? .ready : .off("Hidden")) {
                    Toggle("Menu bar", isOn: $menuBarItem).toggleStyle(.switch).labelsHidden()
                }

                Feature(icon: "hifispeaker.2", title: "Speakers in your home", keys: nil,
                        detail: "Send the voice to a HomePod, an Apple TV, or any AirPlay speaker or TV, with the AirPlay button in the player.",
                        state: .ready) {
                    RoutePicker(player: narrator.player.player).frame(width: 28, height: 24)
                }

                Feature(icon: "waveform.badge.plus", title: "Save as audio", keys: "⇧⌘E",
                        detail: "Turn any book or article into an audio file for your phone, the car or a walk, read in your voice and style.",
                        state: .ready) {
                    Button("Export…") { if let d = narrator.doc { Exporter.shared.export(d) } }.disabled(narrator.doc == nil)
                }
            }
            .padding(28)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .onReceive(tick) { _ in trusted = Selection.trusted }
    }
}

struct Feature<Trailing: View>: View {
    enum State { case ready, needs(String), off(String) }
    let icon: String, title: String, keys: String?, detail: String
    let state: State
    @ViewBuilder let trailing: () -> Trailing

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: icon).font(.system(size: 18)).foregroundStyle(Theme.accent)
                .frame(width: 38, height: 38).background(Theme.accentSoft, in: RoundedRectangle(cornerRadius: 9))
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 8) {
                    Text(title).font(.headline)
                    if let k = keys {
                        Text(k).font(.callout.monospaced()).padding(.horizontal, 6).padding(.vertical, 1)
                            .background(.quaternary, in: RoundedRectangle(cornerRadius: 4))
                    }
                    badge
                }
                Text(detail).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 12)
            trailing()
        }
        .padding(16)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(.quaternary))
    }

    @ViewBuilder private var badge: some View {
        switch state {
        case .ready: EmptyView()
        case .needs(let s): Text(s).font(.caption.weight(.medium)).foregroundStyle(.orange)
        case .off(let s): Text(s).font(.caption).foregroundStyle(.secondary)
        }
    }
}
