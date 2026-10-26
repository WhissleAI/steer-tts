import AppKit
import SwiftUI

/// SteerTTS for the Mac: a voice that runs on this Mac, for anything you want to hear. The main window (the reading
/// list and player, free books, voices, read anywhere), a menu bar item for reading from any app, the Services menu
/// ("Read with SteerTTS" on any selected text) and a global shortcut. Settings behind ⌘,.
@main
struct SteerTTSApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @ObservedObject private var app = AppModel.shared
    @AppStorage("menuBarItem") private var menuBarItem = true

    var body: some Scene {
        Window("SteerTTS", id: "main") {
            MainView()
                .environmentObject(app)
                .environmentObject(app.narrator)
                .environmentObject(app.library)
                .frame(minWidth: 820, minHeight: 520)
        }
        .defaultSize(width: 1120, height: 740)
        .commands { AppCommands(app: app, narrator: app.narrator) }

        Settings {
            SettingsView()
                .environmentObject(app)
                .environmentObject(app.narrator)
        }

        MenuBarExtra(isInserted: $menuBarItem) {
            MenuBarView()
                .environmentObject(app)
                .environmentObject(app.narrator)
                .environmentObject(app.library)
        } label: {
            Image(nsImage: MenuBarIcon.image)
        }
        .menuBarExtraStyle(.window)
    }
}

/// Files opened from Finder or dropped on the Dock icon, the Services menu, and the global shortcuts.
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var hotkeys: [Hotkey] = []

    func applicationDidFinishLaunching(_ n: Notification) {
        NSApp.servicesProvider = ServiceProvider()
        NSUpdateDynamicServices()
        MainActor.assumeIsolated { registerHotkeys(); DevHooks.install(); Assistant.shared.start() }
    }

    func application(_ sender: NSApplication, open urls: [URL]) {
        MainActor.assumeIsolated {
            for u in urls { u.isFileURL ? AppModel.shared.open(file: u) : AppModel.shared.open(link: u) }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    @MainActor
    func registerHotkeys() {
        hotkeys = []
        guard UserDefaults.standard.object(forKey: "shortcuts") as? Bool ?? true else { return }
        // ⌃⌥R reads the selection in any app, ⌃⌥V what was copied, ⌃⌥Space plays or pauses
        if let h = Hotkey(keyCode: 0x0F, modifiers: 0x1000 | 0x0800, id: 1, action: { ReadAnywhere.readSelection() }) { hotkeys.append(h) }
        if let h = Hotkey(keyCode: 0x09, modifiers: 0x1000 | 0x0800, id: 2, action: { ReadAnywhere.readClipboard() }) { hotkeys.append(h) }
        if let h = Hotkey(keyCode: 0x31, modifiers: 0x1000 | 0x0800, id: 3, action: { AppModel.shared.narrator.toggle() }) { hotkeys.append(h) }
    }
}

/// "Read with SteerTTS" and "Add to SteerTTS Reading List" in every app's Services menu (right-click on selected text).
final class ServiceProvider: NSObject {
    @objc func readSelection(_ pb: NSPasteboard, userData: String?, error: AutoreleasingUnsafeMutablePointer<NSString?>) {
        guard let s = pb.string(forType: .string) else { return }
        MainActor.assumeIsolated { AppModel.shared.read(text: s) }
    }

    @objc func addSelection(_ pb: NSPasteboard, userData: String?, error: AutoreleasingUnsafeMutablePointer<NSString?>) {
        guard let s = pb.string(forType: .string) else { return }
        MainActor.assumeIsolated { AppModel.shared.read(text: s, play: false) }
    }

    @objc func readFiles(_ pb: NSPasteboard, userData: String?, error: AutoreleasingUnsafeMutablePointer<NSString?>) {
        let urls = pb.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
        MainActor.assumeIsolated { for u in urls { AppModel.shared.open(file: u) } }
    }
}

/// Reading from any app: the selection (through the accessibility API) or the clipboard.
@MainActor
enum ReadAnywhere {
    static func readSelection() {
        guard Selection.trusted else {
            Selection.askForAccess()
            AppModel.shared.problem = "To read what you select in other apps, allow SteerTTS in System Settings > Privacy & Security > Accessibility. Until then, select text and choose Services > Read with SteerTTS."
            return
        }
        Selection.read { s in
            guard let s, !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                NSSound.beep()
                return
            }
            AppModel.shared.read(text: s)
        }
    }

    static func readClipboard() {
        guard let s = NSPasteboard.general.string(forType: .string), !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            NSSound.beep()
            return
        }
        AppModel.shared.read(text: s)
    }
}

/// The menu bar item's icon: the mark as a template image (it follows the menu bar's light or dark look).
enum MenuBarIcon {
    @MainActor static let image: NSImage = {
        let r = ImageRenderer(content: Mark(mono: true).frame(width: 20, height: 16))
        r.scale = 2
        let img = r.nsImage ?? NSImage()
        img.isTemplate = true
        img.size = NSSize(width: 20, height: 16)
        return img
    }()
}

struct AppCommands: Commands {
    let app: AppModel
    @ObservedObject var narrator: Narrator

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("Open…") { Panels.openFiles() }.keyboardShortcut("o")
            Button("Open Link…") { Panels.openLink() }.keyboardShortcut("l")
            Button("New from Clipboard") { ReadAnywhere.readClipboard() }.keyboardShortcut("v", modifiers: [.command, .shift])
            Button("New from Text…") { Panels.typeText() }.keyboardShortcut("n")
            Divider()
            Button("Export Audio…") { if let d = narrator.doc { Exporter.shared.export(d) } }
                .keyboardShortcut("e", modifiers: [.command, .shift])
                .disabled(narrator.doc == nil)
        }
        CommandMenu("Controls") {
            Button(narrator.playing ? "Pause" : "Play") { narrator.toggle() }.keyboardShortcut(.space, modifiers: [])
            Button("Next Paragraph") { narrator.nextParagraph() }.keyboardShortcut(.rightArrow, modifiers: .command)
            Button("Previous Paragraph") { narrator.previousParagraph() }.keyboardShortcut(.leftArrow, modifiers: .command)
            Button("Next Sentence") { narrator.skip(1) }.keyboardShortcut(.rightArrow, modifiers: [.command, .option])
            Button("Previous Sentence") { narrator.skip(-1) }.keyboardShortcut(.leftArrow, modifiers: [.command, .option])
            Divider()
            Button("Faster") { narrator.speed = min(2.5, narrator.speed + 0.1) }.keyboardShortcut("]")
            Button("Slower") { narrator.speed = max(0.6, narrator.speed - 0.1) }.keyboardShortcut("[")
            Button("Normal Speed") { narrator.speed = 1 }.keyboardShortcut("0")
            Divider()
            Button("Read Selection in Any App") { ReadAnywhere.readSelection() }
            Button("Stop") { narrator.stop() }.keyboardShortcut(".")
        }
    }
}
