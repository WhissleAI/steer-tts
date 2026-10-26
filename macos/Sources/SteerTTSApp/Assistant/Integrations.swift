import Foundation

/// Connecting other tools to the assistant, each on the person's own click and reversible: Claude Code (two hooks in
/// ~/.claude/settings.json that run `steer hook`) and the shell (a few lines of zsh, sourced from ~/.zshrc, that run
/// `steer command` when a long command finishes). Files are backed up before the first change.
enum Integrations {
    /// The `steer` command inside this app (or next to the binary, when run from the build folder).
    static var steer: String {
        Bundle.main.bundleURL.pathExtension == "app"
            ? Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/steer").path
            : URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("steer").path
    }

    private static let home = FileManager.default.homeDirectoryForCurrentUser

    // ---------------------------------------------------------------- Claude Code

    static var claudeSettings: URL { home.appendingPathComponent(".claude/settings.json") }
    private static let events = ["Notification", "Stop"]

    static var claudeConnected: Bool {
        guard let s = try? String(contentsOf: claudeSettings, encoding: .utf8) else { return false }
        return s.contains("/steer\\\" hook") || s.contains("/steer hook") || s.contains("steer\" hook")
    }

    static func connectClaude() throws {
        var o = try readJSON(claudeSettings)
        backup(claudeSettings)
        var hooks = o["hooks"] as? [String: Any] ?? [:]
        let command = "\"\(steer)\" hook"
        for e in events {
            var list = hooks[e] as? [[String: Any]] ?? []
            list.removeAll(where: isOurs)
            list.append(["hooks": [["type": "command", "command": command]]])
            hooks[e] = list
        }
        o["hooks"] = hooks
        try writeJSON(o, claudeSettings)
    }

    static func disconnectClaude() throws {
        var o = try readJSON(claudeSettings)
        guard var hooks = o["hooks"] as? [String: Any] else { return }
        for e in events {
            var list = hooks[e] as? [[String: Any]] ?? []
            list.removeAll(where: isOurs)
            hooks[e] = list.isEmpty ? nil : list
        }
        o["hooks"] = hooks.isEmpty ? nil : hooks
        try writeJSON(o, claudeSettings)
    }

    private static func isOurs(_ entry: [String: Any]) -> Bool {
        (entry["hooks"] as? [[String: Any]] ?? []).contains { ($0["command"] as? String)?.contains("/steer\" hook") == true }
    }

    // ---------------------------------------------------------------- zsh

    static var zshrc: URL { home.appendingPathComponent(".zshrc") }
    static var zshFile: URL { home.appendingPathComponent(".steertts.zsh") }
    private static let sourceLine = "[ -f ~/.steertts.zsh ] && source ~/.steertts.zsh   # SteerTTS: say when a long command finishes"

    static var shellConnected: Bool {
        ((try? String(contentsOf: zshrc, encoding: .utf8)) ?? "").contains(".steertts.zsh") && FileManager.default.fileExists(atPath: zshFile.path)
    }

    static func connectShell() throws {
        try zshScript.write(to: zshFile, atomically: true, encoding: .utf8)
        var rc = (try? String(contentsOf: zshrc, encoding: .utf8)) ?? ""
        if !rc.contains(".steertts.zsh") {
            backup(zshrc)
            if !rc.isEmpty && !rc.hasSuffix("\n") { rc += "\n" }
            rc += sourceLine + "\n"
            try rc.write(to: zshrc, atomically: true, encoding: .utf8)
        }
    }

    static func disconnectShell() throws {
        if var rc = try? String(contentsOf: zshrc, encoding: .utf8) {
            rc = rc.split(separator: "\n", omittingEmptySubsequences: false).filter { !$0.contains(".steertts.zsh") }.joined(separator: "\n")
            try rc.write(to: zshrc, atomically: true, encoding: .utf8)
        }
        try? FileManager.default.removeItem(at: zshFile)
    }

    /// The zsh side: the time each command starts; when the prompt returns, a command that ran long enough (and is
    /// not an interactive program: an editor, a pager, ssh, claude itself) is reported with its exit status.
    static var zshScript: String {
        """
        # SteerTTS: says when a long command finishes ("The build finished, after 4 minutes"; "The tests failed").
        # Installed and removed by SteerTTS > Assistant > Terminal commands. The app decides how long counts as long.
        zmodload zsh/datetime 2>/dev/null
        typeset -g __steer_cmd="" __steer_start=0
        __steer_preexec() { __steer_cmd="$1"; __steer_start=$EPOCHREALTIME }
        __steer_precmd() {
          local st=$?
          [[ -z "$__steer_cmd" ]] && return
          local secs=$(( EPOCHREALTIME - __steer_start ))
          local first=${${(z)__steer_cmd}[1]}
          case "$first" in
            vim|nvim|vi|nano|emacs|less|more|man|ssh|mosh|top|htop|btop|tmux|screen|claude|python|python3|ipython|node|irb|psql|mysql|sqlite3|watch|tail|fzf|lazygit|tig) ;;
            *) if (( secs >= 10 )); then
                 ( "\(steer)" command --status $st --seconds $secs -- "$__steer_cmd" >/dev/null 2>&1 & )
               fi ;;
          esac
          __steer_cmd=""
        }
        autoload -Uz add-zsh-hook
        add-zsh-hook preexec __steer_preexec
        add-zsh-hook precmd __steer_precmd

        """
    }

    // ---------------------------------------------------------------- files

    private static func readJSON(_ url: URL) throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: url.path) else { return [:] }
        let data = try Data(contentsOf: url)
        if data.isEmpty { return [:] }
        guard let o = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile, userInfo: [NSLocalizedDescriptionKey: "\(url.path) is not a JSON object."])
        }
        return o
    }

    private static func writeJSON(_ o: [String: Any], _ url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: o, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: url, options: .atomic)
    }

    /// <file>.steertts-backup, once (the state before SteerTTS first touched it).
    private static func backup(_ url: URL) {
        let b = url.appendingPathExtension("steertts-backup")
        if FileManager.default.fileExists(atPath: url.path) && !FileManager.default.fileExists(atPath: b.path) {
            try? FileManager.default.copyItem(at: url, to: b)
        }
    }
}
