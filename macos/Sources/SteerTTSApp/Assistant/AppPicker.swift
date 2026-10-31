import AppKit
import SwiftUI

/// The apps installed on this Mac (Applications, Utilities, ~/Applications and Apple's own), for choosing which
/// apps' notifications the assistant reads.
struct InstalledApp: Identifiable, Hashable {
    let id: String          // bundle id
    let name: String
    let url: URL
    var icon: NSImage { NSWorkspace.shared.icon(forFile: url.path) }

    static func all() -> [InstalledApp] {
        let fm = FileManager.default
        let dirs = ["/Applications", "/Applications/Utilities", "/System/Applications", "/System/Applications/Utilities",
                    fm.homeDirectoryForCurrentUser.appendingPathComponent("Applications").path]
        var out: [String: InstalledApp] = [:]
        for d in dirs {
            for name in (try? fm.contentsOfDirectory(atPath: d)) ?? [] where name.hasSuffix(".app") {
                let url = URL(fileURLWithPath: d).appendingPathComponent(name)
                guard let b = Bundle(url: url), let id = b.bundleIdentifier, id != Bundle.main.bundleIdentifier else { continue }
                let shown = (b.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String)
                    ?? fm.displayName(atPath: url.path).replacingOccurrences(of: ".app", with: "")
                out[id] = InstalledApp(id: id, name: shown, url: url)
            }
        }
        return out.values.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }
}

/// The apps whose notifications are read, as icons with their names, and "Add App…".
struct ChosenApps: View {
    @ObservedObject private var prefs = AssistantPrefs.shared
    @State private var picking = false
    @State private var installed: [InstalledApp] = []

    var body: some View {
        Flow(spacing: 6) {
            ForEach(chosen, id: \.id) { a in
                HStack(spacing: 6) {
                    Image(nsImage: a.icon).resizable().frame(width: 18, height: 18)
                    Text(a.name)
                    Button {
                        prefs.apps = prefs.apps.filter { $0.caseInsensitiveCompare(a.id) != .orderedSame }
                    } label: { Image(systemName: "xmark").font(.caption2.weight(.bold)) }
                    .buttonStyle(.plain).foregroundStyle(.secondary)
                    .help("Stop reading \(a.name)")
                }
                .font(.callout)
                .padding(.leading, 7).padding(.trailing, 9).padding(.vertical, 4)
                .background(Theme.accentSoft, in: Capsule())
                .overlay(Capsule().stroke(Theme.accent.opacity(0.6)))
            }
            Button { picking = true } label: { Label("Add App…", systemImage: "plus") }
                .buttonStyle(Chip(on: false))
        }
        .onAppear { if installed.isEmpty { installed = InstalledApp.all() } }
        .sheet(isPresented: $picking) { AppPickerSheet(installed: installed) }
    }

    /// The chosen apps that are installed (an app removed from the Mac drops out of the list).
    private var chosen: [InstalledApp] {
        installed.filter { prefs.appOn($0.id) }
    }
}

struct AppPickerSheet: View {
    let installed: [InstalledApp]
    @ObservedObject private var prefs = AssistantPrefs.shared
    @ObservedObject private var notes = Notifications.shared
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Which apps should SteerTTS read?").font(.title3.weight(.semibold))
            Text("Their notifications are said as they arrive: messages in the sender's voice and mood, calls at once. Apps that have sent notifications on this Mac come first.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            TextField("Search apps", text: $query).textFieldStyle(.roundedBorder)
            List {
                let posted = Set(notes.seenApps.map { $0.id.lowercased() })
                let shown = installed.filter { query.isEmpty || $0.name.localizedCaseInsensitiveContains(query) }
                let first = shown.filter { posted.contains($0.id.lowercased()) }
                if !first.isEmpty { Section("Sends notifications") { ForEach(first) { row($0) } } }
                Section(first.isEmpty ? "Apps" : "Other apps") { ForEach(shown.filter { !posted.contains($0.id.lowercased()) }) { row($0) } }
            }
            .listStyle(.inset)
            .frame(height: 380)
            HStack {
                Text("\(installed.filter { prefs.appOn($0.id) }.count) apps chosen").foregroundStyle(.secondary)
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 480)
    }

    private func row(_ a: InstalledApp) -> some View {
        let on = prefs.appOn(a.id)
        return HStack(spacing: 10) {
            Image(nsImage: a.icon).resizable().frame(width: 24, height: 24)
            Text(a.name)
            Spacer()
            Toggle("", isOn: Binding(get: { on }, set: { v in
                if v { prefs.apps.insert(a.id) } else { prefs.apps = prefs.apps.filter { $0.caseInsensitiveCompare(a.id) != .orderedSame } }
            }))
            .toggleStyle(.switch).labelsHidden().controlSize(.small)
        }
    }
}
