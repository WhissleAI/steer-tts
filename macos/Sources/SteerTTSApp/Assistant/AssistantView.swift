import SteerText
import SwiftUI

/// The Assistant page: what SteerTTS says about this Mac, each source with what it needs and whether it is ready,
/// when it stays quiet, and what it did with the last things that happened (never the words).
struct AssistantView: View {
    @ObservedObject private var prefs = AssistantPrefs.shared
    @ObservedObject private var assistant = Assistant.shared
    @ObservedObject private var notes = Notifications.shared
    @ObservedObject private var announcer = Announcer.shared
    @State private var claudeOn = Integrations.claudeConnected
    @State private var shellOn = Integrations.shellConnected
    @State private var calendarOn = Meetings.shared.allowed
    @State private var quietNow: String?
    @State private var problem: String?
    private let tick = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                messages
                claude
                terminal
                Feature(icon: "calendar", title: "Meetings", keys: nil,
                        detail: "A few minutes before a meeting: “In 5 minutes: design review with Priya, on Zoom.”",
                        state: !calendarOn ? .needs("Needs your calendar") : prefs.meetings ? .ready : .off("Off")) {
                    if calendarOn { Toggle("", isOn: $prefs.meetings).toggleStyle(.switch).labelsHidden() }
                    else { Button("Allow…") { Meetings.shared.requestAccess { calendarOn = $0 } } }
                }
                Feature(icon: "macbook", title: "This Mac", keys: nil,
                        detail: "Downloads finishing, the battery running low or full, drives and AirPods connecting. Connecting headphones also catches you up.",
                        state: .ready) {
                    VStack(alignment: .trailing, spacing: 4) {
                        Toggle("Downloads", isOn: $prefs.downloads)
                        Toggle("Battery", isOn: $prefs.battery)
                        Toggle("Devices", isOn: $prefs.devices)
                    }
                    .toggleStyle(.checkbox)
                }
                quiet
                recent
            }
            .padding(28)
            .frame(maxWidth: 780, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .onReceive(tick) { _ in refresh() }
        .onAppear(perform: refresh)
        .alert("SteerTTS", isPresented: Binding(get: { problem != nil }, set: { if !$0 { problem = nil } })) {
            Button("OK") { problem = nil }
        } message: { Text(problem ?? "") }
    }

    private func refresh() {
        quietNow = prefs.whyNotNow()
        claudeOn = Integrations.claudeConnected
        shellOn = Integrations.shellConnected
        calendarOn = Meetings.shared.allowed
    }

    // ---------------------------------------------------------------- sections

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text("Assistant").font(.title2.weight(.semibold))
                Spacer()
                Toggle("Speak what happens", isOn: $prefs.on).toggleStyle(.switch)
            }
            Text("SteerTTS tells you what happens on this Mac while you work: messages and calls, Claude needing you or finishing, long commands, meetings. Each in a voice that fits it. Everything stays on this Mac.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 10) {
                Button { Assistant.shared.sample() } label: { Label("Hear an Example", systemImage: "play.fill") }
                Button { Assistant.shared.catchUp() } label: {
                    Label(assistant.waiting.isEmpty ? "Catch Me Up" : "Catch Me Up · \(assistant.waiting.count)", systemImage: "tray.full")
                }
                .disabled(assistant.waiting.isEmpty)
                if announcer.speaking != nil { Button("Stop") { Announcer.shared.stop() } }
                Spacer()
                if let q = quietNow, prefs.on {
                    Label("Quiet now: \(q)", systemImage: "moon.fill").font(.callout).foregroundStyle(.secondary)
                }
            }
            if let s = announcer.speaking {
                Text("Saying: \(s)").font(.callout).foregroundStyle(Theme.accent).lineLimit(2)
            }
        }
    }

    private var messages: some View {
        Feature(icon: "message", title: "Messages and calls", keys: nil,
                detail: "WhatsApp, Messages, FaceTime and phone calls from your iPhone, Slack, Telegram, Teams… as they arrive: “Priya on WhatsApp: running late, so sorry!”, said gently and in a voice of her own. Calls are said at once, more urgently when they ring again.",
                state: notes.access ? .ready : .needs("Needs Full Disk Access"),
                footer: notes.access ? AnyView(appOptions) : nil) {
            if !notes.access {
                VStack(alignment: .trailing, spacing: 6) {
                    Button("Allow…") {
                        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles")!)
                    }
                    Button("Check Again") { Notifications.shared.recheck() }.buttonStyle(.link)
                }
            }
        }
    }

    private var appOptions: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 18) {
                Toggle("Read the words, not only who", isOn: $prefs.readText)
                Toggle("A voice for each person", isOn: $prefs.personVoices)
            }
            .toggleStyle(.checkbox)
            Text("Apps").font(.callout.weight(.medium)).foregroundStyle(.secondary)
            ChosenApps()
        }
    }

    private var claude: some View {
        Feature(icon: "sparkles", title: "Claude", keys: nil,
                detail: "When Claude needs your permission or your answer, and when it is done, with what it did in a sentence: “Claude is done in steer tts. The Mac app builds and all tests pass.” Claude Code through its hooks; the Claude app through its notifications. Not while you are looking at that terminal.",
                state: !prefs.claude ? .off("Off") : claudeOn ? .ready : .needs("Claude Code not connected")) {
            VStack(alignment: .trailing, spacing: 6) {
                Button(claudeOn ? "Disconnect Claude Code" : "Connect Claude Code") {
                    do { try claudeOn ? Integrations.disconnectClaude() : Integrations.connectClaude(); refresh() }
                    catch { problem = "Claude Code's settings could not be changed: \(error.localizedDescription)" }
                }
                Toggle("Even when I'm looking", isOn: $prefs.whileLooking).toggleStyle(.checkbox).font(.callout)
            }
        }
    }

    private var terminal: some View {
        Feature(icon: "terminal", title: "Long commands", keys: nil,
                detail: "When a command that ran a while finishes: “The build finished, after 4 minutes”, brightly; “The tests failed”, let down. Works in Terminal, iTerm, VS Code and any zsh. Scripts can speak too: steer say \"Deploy is live\" --mood happy",
                state: !prefs.commands ? .off("Off") : shellOn ? .ready : .needs("Terminal not connected")) {
            VStack(alignment: .trailing, spacing: 6) {
                Button(shellOn ? "Disconnect" : "Connect zsh") {
                    do { try shellOn ? Integrations.disconnectShell() : Integrations.connectShell(); refresh() }
                    catch { problem = "~/.zshrc could not be changed: \(error.localizedDescription)" }
                }
                HStack(spacing: 4) {
                    Text("Longer than").font(.callout)
                    Picker("", selection: $prefs.commandSeconds) {
                        ForEach([10.0, 20, 30, 60, 120, 300], id: \.self) { s in Text(s < 60 ? "\(Int(s)) s" : "\(Int(s / 60)) min").tag(s) }
                    }
                    .labelsHidden().fixedSize()
                }
            }
        }
    }

    private var quiet: some View {
        Feature(icon: "moon", title: "When to stay quiet", keys: nil,
                detail: "Never over a call or a meeting (whenever an app is using the microphone), and not in a Focus. While you are away (screen locked, or five minutes without the keyboard), things wait, and you hear them when you are back.",
                state: .ready) {
            VStack(alignment: .trailing, spacing: 4) {
                Toggle("In calls", isOn: $prefs.quietInCalls)
                Toggle("In a Focus", isOn: $prefs.respectFocus)
                Toggle("Welcome back", isOn: $prefs.welcomeBack)
                HStack(spacing: 4) {
                    Toggle("Quiet hours", isOn: $prefs.quietHours)
                    Picker("", selection: $prefs.quietFrom) { ForEach(0..<24, id: \.self) { Text(hour($0)).tag($0) } }
                        .labelsHidden().fixedSize().disabled(!prefs.quietHours)
                    Text("to")
                    Picker("", selection: $prefs.quietTo) { ForEach(0..<24, id: \.self) { Text(hour($0)).tag($0) } }
                        .labelsHidden().fixedSize().disabled(!prefs.quietHours)
                }
            }
            .toggleStyle(.checkbox)
        }
    }

    private var recent: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recently").font(.headline)
            if assistant.recent.isEmpty {
                Text("Nothing yet. What the assistant says, keeps or skips shows here, with why (never the words).")
                    .foregroundStyle(.secondary)
            }
            ForEach(assistant.recent.prefix(12)) { h in
                HStack(spacing: 10) {
                    Image(systemName: h.spoken ? "speaker.wave.2.fill" : "speaker.slash")
                        .foregroundStyle(h.spoken ? Theme.accent : .secondary).frame(width: 18)
                    Text(h.who.isEmpty ? h.app : "\(h.app) · \(h.who)").lineLimit(1)
                    Spacer()
                    Text(h.outcome).foregroundStyle(.secondary)
                    Text(h.time.formatted(date: .omitted, time: .shortened)).font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                }
                .font(.callout)
            }
        }
        .padding(.top, 6)
    }

    private func hour(_ h: Int) -> String {
        var c = DateComponents()
        c.hour = h
        return Calendar.current.date(from: c).map { $0.formatted(date: .omitted, time: .shortened) } ?? "\(h)"
    }
}
