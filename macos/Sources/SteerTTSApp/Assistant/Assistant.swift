import AppKit
import Foundation
import SteerText

/// What happens on this Mac, said out loud: messages and calls from the apps you pick (from macOS's own notification
/// history), Claude needing you or finishing (Claude Code hooks, the Claude app's notifications), long terminal
/// commands finishing, downloads, a meeting about to start, the battery running low, AirPods connecting. Each is said
/// in a delivery that fits it (a call urgent, a failed build let down, a finished task bright, late at night soft),
/// unless now is not the time (a call, a Focus, quiet hours), and while you are away it waits: "Welcome back. While
/// you were away…".
@MainActor
final class Assistant: ObservableObject {
    static let shared = Assistant()

    /// Something worth saying, before it is phrased.
    struct Event {
        enum Kind: String { case message, call, claude, command, download, meeting, battery, device, system, say }
        let kind: Kind
        let app: String                 // "WhatsApp", "Claude Code", "Terminal"
        let bundleID: String?
        let who: String                 // the person, project or file it is about ("" if none)
        let speech: Speech
        var urgent = false
        var away: String?               // the line for "while you were away" ("Priya sent 2 messages on WhatsApp")
    }

    struct Heard: Identifiable {
        let id = UUID()
        let time = Date()
        let app: String, who: String, outcome: String
        let spoken: Bool
    }

    @Published private(set) var recent: [Heard] = []    // what was done with the last events (never the words)
    @Published private(set) var waiting: [Event] = []    // kept while away, for welcome back

    private let prefs = AssistantPrefs.shared
    private var wasAway = false
    private var awayTimer: Timer?

    private init() {
        let dnc = DistributedNotificationCenter.default()
        dnc.addObserver(forName: Notification.Name("com.apple.screenIsLocked"), object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { Machine.locked = true; self.wasAway = true }
        }
        dnc.addObserver(forName: Notification.Name("com.apple.screenIsUnlocked"), object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { Machine.locked = false; self.back(after: 2) }
        }
        // the `steer` command (Claude Code hooks, the shell integration, scripts)
        dnc.addObserver(forName: Notification.Name("ai.whissle.steertts.event"), object: nil, queue: .main) { n in
            guard let json = n.object as? String else { return }
            MainActor.assumeIsolated { SteerEvents.handle(json) }
        }
        // back at the keyboard after a while without it
        awayTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { _ in
            MainActor.assumeIsolated {
                if Machine.away { self.wasAway = true } else if self.wasAway && !Machine.locked { self.back(after: 0.5) }
            }
        }
    }

    /// Starts the sources (notifications, system events, the calendar).
    func start() {
        Notifications.shared.start()
        SystemEvents.shared.start()
        Meetings.shared.start()
    }

    // ---------------------------------------------------------------- deciding

    func handle(_ e: Event) {
        let why = prefs.whyNotNow(urgent: e.urgent)
        if why == nil {
            Announcer.shared.say(e.speech, about: e.bundleID ?? e.kind.rawValue, interrupt: e.kind == .call)
            log(e, "spoken", spoken: true)
        } else if why == "you're away" || why == "quiet hours", e.away != nil || e.kind == .message, prefs.welcomeBack {
            waiting.append(e)
            log(e, "kept for when you're back", spoken: false)
        } else {
            log(e, "not read: \(why!)", spoken: false)
        }
    }

    private func log(_ e: Event, _ outcome: String, spoken: Bool) {
        recent.insert(Heard(app: e.app, who: e.who, outcome: outcome, spoken: spoken), at: 0)
        if recent.count > 30 { recent.removeLast(recent.count - 30) }
    }

    // ---------------------------------------------------------------- welcome back

    private func back(after seconds: Double) {
        guard wasAway else { return }
        wasAway = false
        guard !waiting.isEmpty, prefs.welcomeBack else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { self.catchUp(greeting: true) }
    }

    /// "Welcome back. While you were away: Priya sent 2 messages on WhatsApp. Claude finished in steer-tts. …"
    func catchUp(greeting: Bool = false) {
        guard !waiting.isEmpty else {
            if !greeting { Announcer.shared.say(Speech.of("Nothing new while you were away.")) }
            return
        }
        guard prefs.whyNotNow() == nil || !greeting else { return }   // still in a call or a Focus: keep them
        let items = waiting
        waiting.removeAll()
        let s = Speech()
        s.add(greeting ? "Welcome back. While you were away:" : "While you were away:", SteerText.Expression.CALM).pause(250)
        // messages grouped by person and app; everything else one line each, in order
        var counts: [String: (who: String, app: String, n: Int)] = [:]
        var order: [String] = []
        for e in items where e.kind == .message {
            let k = e.who + "|" + e.app
            if counts[k] == nil { order.append(k); counts[k] = (e.who, e.app, 0) }
            counts[k]!.n += 1
        }
        for k in order {
            let c = counts[k]!
            let who = c.who.isEmpty ? "Someone" : c.who
            s.add(c.n == 1 ? "\(who) sent a message on \(c.app)." : "\(who) sent \(c.n) messages on \(c.app).").pause(150)
        }
        for e in items where e.kind != .message { if let a = e.away { s.add(a).pause(150) } }
        Announcer.shared.say(s, about: "catch-up")
        recent.insert(Heard(app: "SteerTTS", who: "", outcome: "caught up on \(items.count)", spoken: true), at: 0)
    }

    // ---------------------------------------------------------------- a sample, for the Assistant page

    func sample() {
        let s = Speech()
        s.add("Priya on WhatsApp.", nil, nil, 150)
        s.add("Running ten minutes late, so sorry!", SteerText.Expression.of("Running ten minutes late, so sorry!"),
              prefs.personVoices ? Voices.forPerson("Priya", own: AppModel.shared.style.voice) : nil, 400)
        s.add("Claude is done in steer-tts.", nil, nil, 120)
        s.add("The Mac app builds and all tests pass.", SteerText.Expression.HAPPY)
        Announcer.shared.say(s, about: "sample")
    }
}
