import AppKit
import Foundation

/// Developer hooks, as the Android debug build's adb hooks: only when the app is started with STEERTTS_DEV=1
/// (`open --env STEERTTS_DEV=1 dist/SteerTTS.app`). Commands arrive as distributed notifications named
/// ai.whissle.steertts.dev with the command in the object string (tools/dev.sh sends them):
///   section:library|freeBooks|voices|anywhere   open:<file path>   link:<url>   text:<text>
///   play   pause   seek:<sentence>   snapshot:<png path> (the main window, drawn by the app itself)
///   log (prints the narrator's state to stdout)
@MainActor
enum DevHooks {
    static func install() {
        guard ProcessInfo.processInfo.environment["STEERTTS_DEV"] == "1" else { return }
        DistributedNotificationCenter.default().addObserver(forName: Notification.Name("ai.whissle.steertts.dev"),
                                                            object: nil, queue: .main) { n in
            guard let cmd = n.object as? String else { return }
            MainActor.assumeIsolated { run(cmd) }
        }
        print("dev hooks on")
    }

    static func run(_ cmd: String) {
        let app = AppModel.shared
        let parts = cmd.split(separator: ":", maxSplits: 1).map(String.init)
        let arg = parts.count > 1 ? parts[1] : ""
        switch parts[0] {
        case "section":
            app.section = ["library": .library, "freeBooks": .freeBooks, "assistant": .assistant, "voices": .voices,
                           "anywhere": .anywhere][arg] ?? .library
        case "open": app.open(file: URL(fileURLWithPath: arg))
        case "link": if let u = URL(string: arg) { app.open(link: u) }
        case "text": app.read(text: arg)
        case "play": if !app.narrator.playing { app.narrator.toggle() }
        case "pause": app.narrator.pause()
        case "seek": app.narrator.seek(Int(arg) ?? 0)
        case "snapshot": snapshot(to: arg)
        case "sample": Assistant.shared.sample()
        case "notes": print("notes: access=\(Notifications.shared.access) \(Notifications.shared.debugState)")
        case "allowapp": AssistantPrefs.shared.apps.insert(arg)
        case "looking": AssistantPrefs.shared.whileLooking = arg == "1"
        case "mute": Announcer.shared.volume = 0; app.narrator.player.volume = 0
        case "unmute": Announcer.shared.volume = 1; app.narrator.player.volume = 1
        case "recent": for h in Assistant.shared.recent.prefix(8) { print("recent: \(h.app) | \(h.who) | \(h.outcome)") }
        case "said": print("saying: \(Announcer.shared.speaking ?? "-")")
        case "export": if let d = app.narrator.doc { Exporter.shared.export(d, to: URL(fileURLWithPath: arg)) }
        case "exportlog": print("export: \(Exporter.shared.title ?? "-") \(Int(Exporter.shared.fraction * 100))%")
        case "log":
            let n: Narrator = app.narrator
            print("narrator: doc=\(n.doc?.title ?? "-") heard=\(n.heard) playing=\(n.playing) preparing=\(n.preparing) problem=\(n.problem ?? "-") app.problem=\(app.problem ?? "-")")
        default: print("unknown dev command \(cmd)")
        }
        fflush(stdout)
    }

    /// The main window as the window server shows it (an app may capture its own windows without permission).
    private static func snapshot(to path: String) {
        guard let w = NSApp.windows.first(where: { $0.isVisible && $0.frame.width > 400 }),
              let img = CGWindowListCreateImage(.null, .optionIncludingWindow, CGWindowID(w.windowNumber), [.boundsIgnoreFraming])
        else { print("no window"); return }
        try? NSBitmapImageRep(cgImage: img).representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
        print("snapshot \(path)")
    }
}
