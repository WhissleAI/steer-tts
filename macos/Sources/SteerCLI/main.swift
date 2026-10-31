import AppKit
import Foundation

// steer: tells SteerTTS (the app) that something happened, so it can say so. Inside the app at
// SteerTTS.app/Contents/MacOS/steer (named so it cannot clash with SteerTTS on a case-insensitive disk). Starts the app in the background if it is not running. Never fails loudly:
// a hook or a prompt must not break because the voice is unavailable.
//
//   steer say "The deploy is live" [--mood happy|sad|urgent|calm|excited]
//   steer hook                     Claude Code hook: the hook's JSON on stdin (Notification, Stop, ...)
//   steer command --status 0 --seconds 95 -- make test     a shell command finished (the zsh integration)

let args = Array(CommandLine.arguments.dropFirst())
var event: [String: Any] = ["cwd": FileManager.default.currentDirectoryPath, "time": Date().timeIntervalSince1970]

func value(_ flag: String) -> String? {
    guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
    return args[i + 1]
}

switch args.first {
case "say":
    event["kind"] = "say"
    event["text"] = args.dropFirst().first { !$0.hasPrefix("--") } ?? ""
    event["mood"] = value("--mood")
case "hook":
    event["kind"] = "claude"
    let input = FileHandle.standardInput.readDataToEndOfFile()
    event["hook"] = (try? JSONSerialization.jsonObject(with: input)) ?? [:]
    event["terminal"] = ProcessInfo.processInfo.environment["TERM_PROGRAM"] ?? ""
case "command":
    event["kind"] = "command"
    event["status"] = Int(value("--status") ?? "0") ?? 0
    event["seconds"] = Double(value("--seconds") ?? "0") ?? 0
    if let i = args.firstIndex(of: "--") { event["command"] = args[(i + 1)...].joined(separator: " ") }
    event["terminal"] = ProcessInfo.processInfo.environment["TERM_PROGRAM"] ?? ""
default:
    FileHandle.standardError.write(Data("""
    usage: steer say TEXT [--mood happy|sad|urgent|calm|excited]
           steer hook                      (Claude Code hooks: JSON on stdin)
           steer command --status N --seconds S -- COMMAND

    """.utf8))
    exit(args.isEmpty ? 0 : 2)
}

let bundleID = "ai.whissle.steertts"
if NSRunningApplication.runningApplications(withBundleIdentifier: bundleID).isEmpty {
    // Contents/MacOS/steer -> the app bundle
    let app = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let target = app.pathExtension == "app" ? app : NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID)
    if let target {
        let cfg = NSWorkspace.OpenConfiguration()
        cfg.activates = false
        cfg.hides = true
        let done = DispatchSemaphore(value: 0)
        NSWorkspace.shared.openApplication(at: target, configuration: cfg) { _, _ in done.signal() }
        _ = done.wait(timeout: .now() + 5)
        Thread.sleep(forTimeInterval: 1.5)   // let it start listening
    }
}

if let data = try? JSONSerialization.data(withJSONObject: event), let json = String(data: data, encoding: .utf8) {
    DistributedNotificationCenter.default().postNotificationName(Notification.Name("ai.whissle.steertts.event"),
                                                                 object: json, userInfo: nil, deliverImmediately: true)
}
exit(0)
