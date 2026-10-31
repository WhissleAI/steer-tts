import AppKit
import CoreAudio
import Foundation

/// The assistant's settings, and whether speaking out loud is fine right now (Android's AssistantPrefs.mayReadNow,
/// for a Mac): not while the microphone is in use (a call or a meeting), not in a Focus, not in quiet hours; while
/// you are away (screen locked, or no keyboard or mouse for a few minutes) things wait for "welcome back".
@MainActor
final class AssistantPrefs: ObservableObject {
    static let shared = AssistantPrefs()
    private let d = UserDefaults.standard

    @Published var on: Bool { didSet { d.set(on, forKey: "a.on") } }
    @Published var readText: Bool { didSet { d.set(readText, forKey: "a.readText") } }       // the words, not only who
    @Published var personVoices: Bool { didSet { d.set(personVoices, forKey: "a.personVoices") } }
    @Published var claude: Bool { didSet { d.set(claude, forKey: "a.claude") } }
    @Published var whileLooking: Bool { didSet { d.set(whileLooking, forKey: "a.whileLooking") } }   // also when you are in that window
    @Published var commands: Bool { didSet { d.set(commands, forKey: "a.commands") } }
    @Published var commandSeconds: Double { didSet { d.set(commandSeconds, forKey: "a.commandSeconds") } }
    @Published var meetings: Bool { didSet { d.set(meetings, forKey: "a.meetings") } }
    @Published var downloads: Bool { didSet { d.set(downloads, forKey: "a.downloads") } }
    @Published var battery: Bool { didSet { d.set(battery, forKey: "a.battery") } }
    @Published var devices: Bool { didSet { d.set(devices, forKey: "a.devices") } }
    @Published var welcomeBack: Bool { didSet { d.set(welcomeBack, forKey: "a.welcomeBack") } }
    @Published var quietInCalls: Bool { didSet { d.set(quietInCalls, forKey: "a.quietInCalls") } }
    @Published var respectFocus: Bool { didSet { d.set(respectFocus, forKey: "a.respectFocus") } }
    @Published var quietHours: Bool { didSet { d.set(quietHours, forKey: "a.quietHours") } }
    @Published var quietFrom: Int { didSet { d.set(quietFrom, forKey: "a.quietFrom") } }    // hour, 0-23
    @Published var quietTo: Int { didSet { d.set(quietTo, forKey: "a.quietTo") } }
    /// Apps whose notifications are read (bundle ids). Messaging apps are on by default.
    @Published var apps: Set<String> { didSet { d.set(Array(apps), forKey: "a.apps") } }

    static let defaultApps: Set<String> = [
        "net.whatsapp.WhatsApp", "com.apple.MobileSMS", "com.apple.iChat", "com.apple.FaceTime", "com.apple.TelephonyUtilities",
        "com.tinyspeck.slackmacgap", "ru.keepcoder.Telegram", "org.whispersystems.signal-desktop", "com.hnc.Discord",
        "com.microsoft.teams2", "us.zoom.xos", "com.anthropic.claudefordesktop", "com.apple.reminders",
    ]

    private init() {
        let ud = UserDefaults.standard
        func b(_ k: String, _ def: Bool) -> Bool { ud.object(forKey: k) as? Bool ?? def }
        on = b("a.on", true)
        readText = b("a.readText", true)
        personVoices = b("a.personVoices", true)
        claude = b("a.claude", true)
        whileLooking = b("a.whileLooking", false)
        commands = b("a.commands", true)
        commandSeconds = ud.object(forKey: "a.commandSeconds") as? Double ?? 20
        meetings = b("a.meetings", true)
        downloads = b("a.downloads", true)
        battery = b("a.battery", true)
        devices = b("a.devices", true)
        welcomeBack = b("a.welcomeBack", true)
        quietInCalls = b("a.quietInCalls", true)
        respectFocus = b("a.respectFocus", true)
        quietHours = b("a.quietHours", false)
        quietFrom = ud.object(forKey: "a.quietFrom") as? Int ?? 22
        quietTo = ud.object(forKey: "a.quietTo") as? Int ?? 7
        apps = Set(ud.stringArray(forKey: "a.apps") ?? Array(AssistantPrefs.defaultApps))
    }

    func appOn(_ bundleID: String) -> Bool {
        apps.contains(bundleID) || apps.contains { $0.caseInsensitiveCompare(bundleID) == .orderedSame }
    }

    // ---------------------------------------------------------------- may it speak now?

    /// nil when speaking is fine; else why not ("you're in a call", "Focus is on", "quiet hours", "you're away").
    /// Urgent things (a call ringing) ignore only quiet hours' and away's waiting, never a call or a Focus.
    func whyNotNow(urgent: Bool = false) -> String? {
        if !on { return "the assistant is off" }
        if quietInCalls && Machine.micInUse() { return "you're in a call" }
        if respectFocus && Machine.focusOn() { return "Focus is on" }
        if urgent { return nil }
        if quietHours && inQuietHours() { return "quiet hours" }
        if Machine.away { return "you're away" }
        return nil
    }

    private func inQuietHours() -> Bool {
        let h = Calendar.current.component(.hour, from: Date())
        return quietFrom <= quietTo ? (h >= quietFrom && h < quietTo) : (h >= quietFrom || h < quietTo)
    }
}

/// What the Mac is doing: the microphone in use, a Focus on, the screen locked, the person away from the keyboard,
/// the app in front.
@MainActor
enum Machine {
    static var locked = false
    static let awaySeconds: Double = 300

    /// Away: the screen is locked, or nothing was typed or clicked for five minutes.
    static var away: Bool { locked || idleSeconds > awaySeconds }

    static var idleSeconds: Double {
        CGEventSource.secondsSinceLastEventType(.combinedSessionState, eventType: CGEventType(rawValue: ~0)!)
    }

    /// A call or a meeting app is using the microphone: FaceTime and iPhone calls, Zoom, Teams, Slack huddles,
    /// WhatsApp, Discord... (per process, macOS 14+). The microphone alone says little: macOS's own speech service
    /// listens all the time for "Hey Siri" and dictation, and a browser tab may be dictating.
    nonisolated static func micInUse() -> Bool { !callApps().isEmpty }

    nonisolated static let callAppIDs = ["facetime", "telephonyutilities", "callservicesd", "avconferenced", "zoom", "teams",
                                         "webex", "slack", "discord", "whatsapp", "telegram", "signal", "skype", "gotomeeting",
                                         "bluejeans", "ringcentral", "dialpad", "aircall", "around", "tuple", "pop.app"]

    /// The bundle ids of call and meeting apps using the microphone now.
    nonisolated static func callApps() -> [String] {
        var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyProcessObjectList,
                                              mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size) == noErr, size > 0 else { return [] }
        var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids) == noErr else { return [] }
        var out: [String] = []
        for id in ids {
            var a = AudioObjectPropertyAddress(mSelector: kAudioProcessPropertyIsRunningInput,
                                               mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
            var running = UInt32(0)
            var s = UInt32(MemoryLayout<UInt32>.size)
            guard AudioObjectGetPropertyData(id, &a, 0, nil, &s, &running) == noErr, running != 0 else { continue }
            a.mSelector = kAudioProcessPropertyBundleID
            var b: Unmanaged<CFString>?
            s = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
            guard AudioObjectGetPropertyData(id, &a, 0, nil, &s, &b) == noErr, let bid = b?.takeRetainedValue() as String? else { continue }
            let l = bid.lowercased()
            if callAppIDs.contains(where: { l.contains($0) }) { out.append(bid) }
        }
        return out
    }

    /// A Focus (Do Not Disturb, Work, Sleep…) is on: macOS keeps its active assertions in a file only readable with
    /// Full Disk Access (which reading notifications needs anyway).
    nonisolated static func focusOn() -> Bool {
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/DoNotDisturb/DB/Assertions.json")
        guard let data = try? Data(contentsOf: url),
              let o = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let records = o["data"] as? [[String: Any]] else { return false }
        return records.contains { ($0["storeAssertionRecords"] as? [Any])?.isEmpty == false }
    }

    /// The app in front, and whether the person is using it right now (typed or clicked in the last 20 seconds).
    static func looking(at bundleIDs: Set<String>) -> Bool {
        guard let front = NSWorkspace.shared.frontmostApplication?.bundleIdentifier else { return false }
        return bundleIDs.contains(front) && idleSeconds < 20
    }

    static let terminals: Set<String> = ["com.apple.Terminal", "com.googlecode.iterm2", "com.microsoft.VSCode", "com.mitchellh.ghostty",
                                         "dev.warp.Warp-Stable", "com.todesktop.230313mzl4w4u92", "dev.zed.Zed", "net.kovidgoyal.kitty",
                                         "com.github.wez.wezterm", "co.zeit.hyper", "com.jetbrains.intellij", "com.apple.dt.Xcode"]
}
