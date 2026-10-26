import AppKit
import ApplicationServices
import Foundation
import SQLite3
import SteerText

/// New notifications from the apps you picked, as they arrive: macOS keeps every delivered notification in a small
/// database (~/Library/Group Containers/group.com.apple.usernoted/db2/db) that only apps with Full Disk Access may
/// read. Watched by its write-ahead log; each new record is read once (its app, title, subtitle and body), phrased as
/// Android's Readout does (a message in its mood and the sender's voice, a call urgent, an email by who and how it
/// starts) and handed to the Assistant. Nothing is stored: the last record id seen, nothing else.
@MainActor
final class Notifications: ObservableObject {
    static let shared = Notifications()

    @Published private(set) var access = true                 // false: Full Disk Access is needed
    @Published private(set) var seenApps: [(id: String, name: String)] = []   // apps that have posted, for the app list

    private let dir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Group Containers/group.com.apple.usernoted/db2")
    private var lastID: Int64 = -1
    private var lastStamp: Date?
    private var timer: Timer?
    private var names: [String: String] = [:]

    var debugState: String { "lastID=\(lastID) timer=\(timer != nil) apps=\(seenApps.count) polls=\(polls) queries=\(queries) error=\(lastError)" }
    private var polls = 0, queries = 0
    private var lastError = "-"

    func start() {
        guard timer == nil else { return }
        LiveBanners.shared.start()
        lastID = (try? maxID()) ?? -1   // only what arrives from now on
        refreshApps()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in MainActor.assumeIsolated { self.poll() } }
    }

    /// Checks access again (after the person granted Full Disk Access in System Settings).
    func recheck() {
        lastID = (try? maxID()) ?? -1
        refreshApps()
    }

    private func poll() {
        polls += 1
        let wal = dir.appendingPathComponent("db-wal")
        let stamp = (try? FileManager.default.attributesOfItem(atPath: wal.path)[.modificationDate]) as? Date
        guard stamp != lastStamp || !access else { return }
        lastStamp = stamp
        queries += 1
        do {
            if lastID < 0 { lastID = try maxID(); return }
            for r in try records(after: lastID) {
                lastID = max(lastID, r.id)
                arrived(r)
            }
        } catch { lastError = "\(error)" }
    }

    // ---------------------------------------------------------------- reading the database

    struct Record {
        let id: Int64
        let app: String
        let title: String, subtitle: String, body: String
        let category: String
    }

    private func open() throws -> OpaquePointer {
        var db: OpaquePointer?
        let path = dir.appendingPathComponent("db").path
        guard sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK, let db else {
            if let db { sqlite3_close(db) }
            access = false
            throw CocoaError(.fileReadNoPermission)
        }
        return db
    }

    private func query(_ sql: String, _ row: (OpaquePointer) -> Void) throws {
        let db = try open()
        defer { sqlite3_close(db) }
        var st: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK, let st else {
            access = false   // "authorization denied": no Full Disk Access
            throw CocoaError(.fileReadNoPermission)
        }
        defer { sqlite3_finalize(st) }
        access = true
        while sqlite3_step(st) == SQLITE_ROW { row(st) }
    }

    private func maxID() throws -> Int64 {
        var m: Int64 = 0
        try query("select coalesce(max(rec_id), 0) from record") { m = sqlite3_column_int64($0, 0) }
        return m
    }

    private func records(after id: Int64) throws -> [Record] {
        var out: [Record] = []
        try query("select r.rec_id, a.identifier, r.data from record r join app a on a.app_id = r.app_id where r.rec_id > \(id) order by r.rec_id") { st in
            let rid = sqlite3_column_int64(st, 0)
            let app = sqlite3_column_text(st, 1).map { String(cString: $0) } ?? ""
            guard let blob = sqlite3_column_blob(st, 2) else { return }
            let data = Data(bytes: blob, count: Int(sqlite3_column_bytes(st, 2)))
            guard let p = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
                  let req = p["req"] as? [String: Any] else { return }
            out.append(Record(id: rid, app: (p["app"] as? String) ?? app,
                              title: req["titl"] as? String ?? "", subtitle: req["subt"] as? String ?? "",
                              body: req["body"] as? String ?? "", category: req["cate"] as? String ?? ""))
        }
        return out
    }

    private func refreshApps() {
        var ids: [String] = []
        try? query("select distinct a.identifier from app a join record r on r.app_id = a.app_id") { st in
            if let t = sqlite3_column_text(st, 0) { ids.append(String(cString: t)) }
        }
        let known = Set(ids.map { $0.lowercased() })
        // the messaging apps that are installed, even before they have posted
        for b in AssistantPrefs.defaultApps where !known.contains(b.lowercased()) && NSWorkspace.shared.urlForApplication(withBundleIdentifier: b) != nil {
            ids.append(b)
        }
        seenApps = ids.filter { $0 != Bundle.main.bundleIdentifier && !$0.hasPrefix("_") }
            .map { ($0, name(of: $0)) }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    /// The app's name as people know it ("WhatsApp", "Messages").
    func name(of bundleID: String) -> String {
        if let n = names[bundleID] { return n }
        var n = bundleID.split(separator: ".").last.map(String.init) ?? bundleID
        if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) {
            n = FileManager.default.displayName(atPath: url.path).replacingOccurrences(of: ".app", with: "")
        }
        let fixes = ["claudefordesktop": "Claude", "slackmacgap": "Slack", "alertnotificationservice": "Chrome", "MobileSMS": "Messages"]
        n = fixes[n] ?? n
        names[bundleID] = n
        return n
    }

    // ---------------------------------------------------------------- what it becomes

    private static let callWords = "(?i)\\b(incoming (voice |video |audio )?call|is calling|calling you|wants to (facetime|call)|"
        + "facetime (audio|video)|video call|voice call|ringing)\\b"

    /// Notifications already said from their banner (app|title|body), so the history does not say them again.
    private var said: [String: Date] = [:]

    private func signature(_ r: Record) -> String {
        [r.app.lowercased(), r.title, r.body].joined(separator: "|").replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
    }

    func arrived(_ r: Record, live: Bool = false) {
        let prefs = AssistantPrefs.shared
        guard r.app != Bundle.main.bundleIdentifier, prefs.appOn(r.app) else { return }
        said = said.filter { $0.value.timeIntervalSinceNow > -600 }
        let sig = signature(r)
        if live { said[sig] = Date() } else if said[sig] != nil { return }
        let app = name(of: r.app)
        let lower = r.app.lowercased()
        let title = r.title.trimmingCharacters(in: .whitespacesAndNewlines)
        let body = r.body.trimmingCharacters(in: .whitespacesAndNewlines)
        let a = Assistant.shared
        let own = AppModel.shared.style.voice

        // a call: from FaceTime, Phone (Continuity), WhatsApp, Teams, Zoom... urgent, more so when it rings again
        let callText = [title, r.subtitle, body, r.category].joined(separator: " ")
        if lower.contains("facetime") || lower.contains("telephonyutilities") || rx(Self.callWords).find(callText) != nil
            || r.category.lowercased().contains("call") {
            let caller = callerName(title: title, body: body, app: app)
            let count = Readout.ringCount(caller, Date().millis)
            a.handle(.init(kind: .call, app: app, bundleID: r.app, who: caller,
                           speech: Readout.call(caller, app == "FaceTime" ? nil : app, count), urgent: true,
                           away: "\(caller) called on \(app)."))
            return
        }
        // the Claude app: waiting for you, or done
        if lower.contains("claudefordesktop") {
            let s = Speech()
            s.add(title.isEmpty ? "Claude needs you." : Readout.sentence(title), ClaudeEvents.attentive, nil, 120)
            if prefs.readText { s.add(Readout.sentence(body)) }
            a.handle(.init(kind: .claude, app: "Claude", bundleID: r.app, who: "", speech: s, away: "Claude asked for you."))
            return
        }
        // email
        if lower.contains("mail") || lower.contains("outlook") || lower.contains("spark") || lower.contains("superhuman") {
            let s = prefs.readText ? Readout.email(title, r.subtitle, body) : Readout.email(title, nil)
            a.handle(.init(kind: .message, app: app, bundleID: r.app, who: Readout.senderName(title), speech: s,
                           away: "an email from \(Readout.senderName(title))."))
            return
        }
        // reminders and calendars
        if lower.contains("reminders") || lower.contains("ical") || lower.contains("calendar") {
            a.handle(.init(kind: .system, app: app, bundleID: r.app, who: "", speech: Readout.reminder(title, body, app),
                           away: "a reminder: \(title)."))
            return
        }
        // messages: Messages, WhatsApp, Slack, Telegram, Signal, Discord, Teams...
        if Self.messaging(lower) {
            let (person, group) = split(title: title, subtitle: r.subtitle, body: body, app: lower)
            var text = body
            if group != nil, text.hasPrefix(person + ": ") { text = String(text.dropFirst(person.count + 2)) }
            let s = prefs.readText
                ? Readout.message(person, group, app, [text], prefs.personVoices, own)
                : Speech.of(group != nil ? "\(person) in \(group!), on \(app)." : "\(person) on \(app).")
            a.handle(.init(kind: .message, app: app, bundleID: r.app, who: person, speech: s, away: nil))
            return
        }
        // anything else from an app you picked: its title and text
        a.handle(.init(kind: .system, app: app, bundleID: r.app, who: title,
                       speech: prefs.readText ? Readout.notice(title, body, app) : Speech.of("\(app): \(title)"),
                       away: "\(app): \(title)."))
    }

    static func messaging(_ lower: String) -> Bool {
        ["whatsapp", "mobilesms", "ichat", "slack", "telegram", "signal", "discord", "teams", "messenger", "wechat", "line", "skype", "viber"]
            .contains { lower.contains($0) }
    }

    /// The person and the group a message is from: "Priya" / nil; Slack's "#design" title; WhatsApp's "Family: Priya".
    private func split(title: String, subtitle: String, body: String, app: String) -> (String, String?) {
        if !subtitle.isEmpty { return (subtitle, title) }                    // Messages groups: title the group, subtitle the sender
        if let m = rx("^(.+?)\\s*[:@]\\s*(.+)$").find(title), app.contains("whatsapp") {   // "Family: Priya", "Priya @ Family"
            return (m.group(2)!, m.group(1)!)
        }
        if let m = rx("^(.{1,40}?):\\s").find(body), title.hasPrefix("#") || app.contains("slack") {   // Slack: "#design" / "Priya: text"
            return (m.group(1)!, title)
        }
        return (title.isEmpty ? "Someone" : title, nil)
    }

    private func callerName(title: String, body: String, app: String) -> String {
        for s in [title, body] {
            if let m = rx("(?i)^(.+?)\\s+(is calling|wants to|calling)").find(s) { return m.group(1)! }
        }
        let t = title.replacingOccurrences(of: "(?i)^(incoming )?(facetime )?(voice |video |audio )?call( from)?:?\\s*", with: "",
                                           options: .regularExpression)
        return t.isEmpty ? "Someone" : t
    }
}


/// Notifications the moment they arrive. macOS writes a notification into its history only half a minute or so
/// later (fine for a message, too late for a call), but its notification daemon logs each one as it is created,
/// with the app's id (the words are private in the log). That line is the trigger; the words come from the banner on
/// screen, read through the accessibility API ("WhatsApp, Priya, Family, Running late, so sorry!"). With no banner
/// (a Focus hides them) or no accessibility access, a call is still said at once ("A call on FaceTime"), and
/// messages wait for the history.
@MainActor
final class LiveBanners {
    static let shared = LiveBanners()
    private var process: Process?
    private var seen: Set<String> = []

    func start() {
        guard process == nil else { return }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/log")
        p.arguments = ["stream", "--style", "ndjson", "--predicate",
                       "subsystem == \"com.apple.usernotificationsd\" AND category == \"NotificationsPipeline\" AND eventMessage CONTAINS \"Request: Starting\""]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        var buffer = Data()
        pipe.fileHandleForReading.readabilityHandler = { h in
            let chunk = h.availableData
            if chunk.isEmpty { return }
            buffer.append(chunk)
            while let nl = buffer.firstIndex(of: 0x0A) {
                let line = buffer[buffer.startIndex..<nl]
                buffer.removeSubrange(buffer.startIndex...nl)
                guard let o = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
                      let msg = o["eventMessage"] as? String,
                      let m = rx("\\[create, \\[id=([0-9A-Fa-f-]+), time=[^,]*, bundle=([^\\]]+)\\]").find(msg),
                      let id = m.group(1), let bundle = m.group(2) else { continue }
                DispatchQueue.main.async { MainActor.assumeIsolated { LiveBanners.shared.created(id: id, bundle: bundle) } }
            }
        }
        p.terminationHandler = { _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) { MainActor.assumeIsolated { LiveBanners.shared.restart() } }
        }
        do { try p.run(); process = p } catch {}
    }

    private func restart() { process = nil; start() }

    private func created(id: String, bundle: String) {
        guard !seen.contains(id) else { return }
        seen.insert(id)
        if seen.count > 500 { seen.removeAll() }
        guard bundle != Bundle.main.bundleIdentifier, AssistantPrefs.shared.appOn(bundle) else { return }
        read(bundle: bundle, tries: 4)
    }

    /// The banner's title, subtitle and text, a moment after it appears (it is drawn after the log line).
    private func read(bundle: String, tries: Int) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
            let n = Notifications.shared
            if let t = Self.banner(appName: n.name(of: bundle)) {
                n.arrived(.init(id: -1, app: bundle, title: t.title, subtitle: t.subtitle, body: t.body, category: ""), live: true)
            } else if tries > 1 {
                self.read(bundle: bundle, tries: tries - 1)
            } else if Self.isCallApp(bundle) {
                // no banner to read (a Focus, or no accessibility access): a call still cannot wait
                n.arrived(.init(id: -1, app: bundle, title: "", subtitle: "", body: "Incoming call", category: "call"), live: true)
            }
        }
    }

    static func isCallApp(_ bundle: String) -> Bool {
        let l = bundle.lowercased()
        return l.contains("facetime") || l.contains("telephonyutilities")
    }

    /// The newest banner of this app in Notification Center's windows: a group described "App, title, subtitle, text"
    /// whose static texts are those parts.
    static func banner(appName: String) -> (title: String, subtitle: String, body: String)? {
        guard AXIsProcessTrusted(),
              let nc = NSRunningApplication.runningApplications(withBundleIdentifier: "com.apple.notificationcenterui").first else { return nil }
        let root = AXUIElementCreateApplication(nc.processIdentifier)
        var found: [String]?
        func walk(_ e: AXUIElement, _ depth: Int) {
            if found != nil || depth > 12 { return }
            if let d = attr(e, kAXDescriptionAttribute) as? String, d.hasPrefix(appName + ",") {
                var texts: [String] = []
                collect(e, 0, &texts)
                if !texts.isEmpty { found = texts; return }
            }
            for c in (attr(e, kAXChildrenAttribute) as? [AXUIElement]) ?? [] { walk(c, depth + 1) }
        }
        walk(root, 0)
        guard let t = found else { return nil }
        switch t.count {
        case 1: return ("", "", t[0])
        case 2: return (t[0], "", t[1])
        default: return (t[0], t[1], t[2...].joined(separator: " "))
        }
    }

    private static func collect(_ e: AXUIElement, _ depth: Int, _ out: inout [String]) {
        if depth > 8 { return }
        if attr(e, kAXRoleAttribute) as? String == "AXStaticText", let v = attr(e, kAXValueAttribute) as? String,
           !v.isEmpty, !v.matches("(?i)now|\\d+[mh] ago|yesterday") { out.append(v) }
        for c in (attr(e, kAXChildrenAttribute) as? [AXUIElement]) ?? [] { collect(c, depth + 1, &out) }
    }

    private static func attr(_ e: AXUIElement, _ name: String) -> CFTypeRef? {
        var v: CFTypeRef?
        return AXUIElementCopyAttributeValue(e, name as CFString, &v) == .success ? v : nil
    }
}
