import AppKit
import Foundation
import SteerText

private typealias Mood = SteerText.Expression.Mood

/// The moods of things that happen at a desk, on top of Android's message moods.
enum ClaudeEvents {
    /// Needs you: clear, a little louder, the end rising like a question.
    static let attentive = SteerText.Expression.Mood("attentive", "interest", "needs you: clear, rising", ("f0_end", 0.8), ("loud", 0.4), ("rate", 0.2))
    /// Done: bright and sure.
    static let done = SteerText.Expression.Mood("done", "confidence", "done: bright, sure", ("f0_iqr", 0.6), ("f0_med", 0.3), ("loud", 0.2))
    /// Failed: lower, slower, let down (not alarmed: a red build is not an emergency).
    static let failed = SteerText.Expression.Mood("failed", "disappointment", "failed: let down", ("f0_med", -0.5), ("rate", -0.2), ("f0_iqr", -0.4))
}

/// What the `steer` command sends (Contents/MacOS/steer): Claude Code's hooks, the shell integration's finished
/// commands, and `steer say` from any script.
@MainActor
enum SteerEvents {
    static func handle(_ json: String) {
        guard let data = json.data(using: .utf8), let e = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        switch e["kind"] as? String {
        case "say": say(e)
        case "claude": claude(e)
        case "command": command(e)
        default: break
        }
    }

    // ---------------------------------------------------------------- steer say

    private static func say(_ e: [String: Any]) {
        let text = e["text"] as? String ?? ""
        let moods: [String: Mood] = ["happy": SteerText.Expression.HAPPY, "sad": SteerText.Expression.SAD, "urgent": SteerText.Expression.URGENT,
                                     "calm": SteerText.Expression.CALM, "excited": SteerText.Expression.EXCITED,
                                     "done": ClaudeEvents.done, "failed": ClaudeEvents.failed, "attentive": ClaudeEvents.attentive]
        let mood = (e["mood"] as? String).flatMap { moods[$0] } ?? SteerText.Expression.of(text)
        Assistant.shared.handle(.init(kind: .say, app: "Script", bundleID: nil, who: "", speech: Speech().add(text, mood),
                                      away: text))
    }

    // ---------------------------------------------------------------- Claude Code

    /// Notification ("Claude needs your permission to use Bash", "Claude is waiting for your input") and Stop (the
    /// turn ended: what it did, from the last thing it wrote).
    private static func claude(_ e: [String: Any]) {
        let prefs = AssistantPrefs.shared
        guard prefs.claude, let hook = e["hook"] as? [String: Any] else { return }
        let project = ((hook["cwd"] as? String) ?? (e["cwd"] as? String) ?? "").split(separator: "/").last.map(String.init) ?? ""
        let place = project.isEmpty ? "" : " in \(spoken(project))"
        if !prefs.whileLooking && Machine.looking(at: Machine.terminals) { return }   // you are watching it
        let s = Speech()
        var away: String
        switch hook["hook_event_name"] as? String {
        case "Notification":
            let msg = hook["message"] as? String ?? ""
            if let m = rx("(?i)permission to use (.+)$").find(msg) {
                s.add("Claude needs your permission\(place), to use \(m.group(1)!.trimmingCharacters(in: .punctuationCharacters)).",
                      ClaudeEvents.attentive)
                away = "Claude asked for permission\(place)."
            } else {
                s.add("Claude is waiting for you\(place).", ClaudeEvents.attentive)
                away = "Claude was waiting for you\(place)."
            }
        case "Stop":
            let summary = (hook["transcript_path"] as? String).flatMap(lastWords) ?? ""
            if summary.hasSuffix("?") {
                s.add("Claude has a question\(place).", ClaudeEvents.attentive, nil, 150)
                if prefs.readText { s.add(summary, ClaudeEvents.attentive) }
                away = "Claude had a question\(place)."
            } else {
                s.add("Claude is done\(place).", ClaudeEvents.done, nil, 150)
                if prefs.readText && !summary.isEmpty { s.add(summary, ClaudeEvents.done) }
                away = "Claude finished\(place)."
            }
        default:
            return
        }
        Assistant.shared.handle(.init(kind: .claude, app: "Claude Code", bundleID: nil, who: project, speech: s, away: away))
    }

    /// The first sentence of the last thing Claude wrote in this session (the transcript's last assistant text),
    /// without Markdown, at most about 200 characters.
    static func lastWords(_ path: String) -> String? {
        guard let h = FileHandle(forReadingAtPath: path) else { return nil }
        defer { try? h.close() }
        let size = (try? h.seekToEnd()) ?? 0
        try? h.seek(toOffset: size > 400_000 ? size - 400_000 : 0)
        guard let data = try? h.readToEnd(), let tail = String(data: data, encoding: .utf8) else { return nil }
        for line in tail.split(separator: "\n").reversed() {
            guard line.contains("\"assistant\""), let o = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                  o["type"] as? String == "assistant", let msg = o["message"] as? [String: Any],
                  let content = msg["content"] as? [[String: Any]] else { continue }
            let text = content.compactMap { $0["type"] as? String == "text" ? $0["text"] as? String : nil }.joined(separator: "\n")
            if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
            return summary(text)
        }
        return nil
    }

    /// Markdown out (code, emphasis, links, headings, bullets), then the first sentence.
    static func summary(_ text: String) -> String {
        var t = text.replaceAll("(?s)```.*?```", " ")
        t = t.replaceAll("`([^`]*)`", "$1").replaceAll("\\*\\*?([^*]+)\\*\\*?", "$1").replaceAll("\\[([^\\]]+)\\]\\([^)]*\\)", "$1")
        t = t.replaceAll("(?m)^\\s*(#+|[-*•]|\\d+\\.)\\s+", "").replaceAll("\\s+", " ").trimmed
        let first = TextClean.sentences([t]).first ?? t
        return first.count > 220 ? String(first.prefix(200)).replaceAll("\\s+\\S*$", "") + "…" : first
    }

    /// A folder name as words: "steer-tts" → "steer tts", "live_assist" → "live assist".
    static func spoken(_ name: String) -> String { name.replaceAll("[-_.]+", " ") }

    // ---------------------------------------------------------------- the shell: a long command finished

    private static func command(_ e: [String: Any]) {
        let prefs = AssistantPrefs.shared
        guard prefs.commands else { return }
        let seconds = e["seconds"] as? Double ?? 0
        guard seconds >= prefs.commandSeconds else { return }
        if !prefs.whileLooking && Machine.looking(at: Machine.terminals) { return }
        let cmd = (e["command"] as? String ?? "").trimmed
        let ok = (e["status"] as? Int ?? 0) == 0
        let what = name(of: cmd)
        let after = duration(seconds)
        let line: String
        if ok {
            line = "\(Readout.cap(what)) finished, after \(after)."
        } else {
            line = what.hasSuffix("tests") ? "The tests failed, after \(after)." : "\(Readout.cap(what)) failed, after \(after)."
        }
        let s = Speech().add(line, ok ? ClaudeEvents.done : ClaudeEvents.failed)
        Assistant.shared.handle(.init(kind: .command, app: "Terminal", bundleID: nil, who: what, speech: s,
                                      away: line.replaceAll(", after [^.]+\\.", ".")))
    }

    /// What people would call a command: "the build", "the tests", "the push", "the install", else its program.
    static func name(of cmd: String) -> String {
        let c = cmd.lowercased()
        let rules: [(String, String)] = [
            ("\\b(test|tests|pytest|jest|vitest|rspec|ctest|xcodebuild test|swift test|go test|cargo test|npm (run )?test)\\b", "the tests"),
            ("\\b(build|make|cmake|xcodebuild|gradlew?|assemble|compile|tsc|webpack|vite build|cargo build|swift build)\\b", "the build"),
            ("\\bgit (push)\\b", "the push"), ("\\bgit (pull|fetch|clone)\\b", "the git \\1"),
            ("\\b(npm|pnpm|yarn|pip3?|brew|pod|bundle|cargo) (install|i|add|update|upgrade)\\b", "the install"),
            ("\\bdocker (build|compose)\\b", "the docker \\1"), ("\\b(deploy|gcloud run deploy|kubectl apply|terraform apply)\\b", "the deploy"),
            ("\\b(rsync|scp|curl|wget|aws s3|gsutil)\\b", "the transfer"), ("\\b(python3?|node|ruby|go run|cargo run|swift run)\\b", "your script"),
        ]
        for (re, phrase) in rules { if let m = rx(re).find(c) {
            return phrase.contains("\\1") ? phrase.replacingOccurrences(of: "\\1", with: m.group(1) ?? "") : phrase
        } }
        let prog = c.split(separator: " ").first.map { String($0).split(separator: "/").last.map(String.init) ?? "" } ?? ""
        return prog.isEmpty ? "your command" : "your \(prog) command"
    }

    static func duration(_ s: Double) -> String {
        let n = Int(s.rounded())
        if n < 60 { return "\(n) seconds" }
        if n < 3600 { let m = Int((s / 60).rounded()); return m == 1 ? "a minute" : "\(m) minutes" }
        let h = n / 3600, m = n % 3600 / 60
        return (h == 1 ? "an hour" : "\(h) hours") + (m > 0 ? " and \(m) minutes" : "")
    }
}
