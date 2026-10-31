import SteerEngine
import SteerText
import SwiftUI

/// Short things said in the app's voice: voice and style previews. Its own player, so a preview never disturbs the
/// reading position (reading pauses while it plays).
@MainActor
final class Speaker {
    static let shared = Speaker()
    private let player = SentencePlayer()

    func say(_ text: String, style: Style) {
        let app = AppModel.shared
        if app.narrator.playing { app.narrator.pause() }
        let token = player.flush()
        player.play()
        let st = style.copy()
        app.onEngine { [player] engine in
            let voice = engine.resolve(voice: st.voice)
            try? engine.stream(text, voice: voice, tag: st.tagOrNull(), ctrl: st.ctrl(), pace: st.pace) { a, i in
                player.enqueue(a, mark: i, token: token)
            }
        }
    }

    func stop() { player.flush(); player.pause() }
}

/// The Voices page: pick a voice (named, described, previewed, favourited, filtered), a style, the speed and the
/// delivery, as on Android. The four main controls are the ones the model follows reliably; the rest sit under
/// Advanced. Every change is the user's base style, used for everything SteerTTS reads.
struct VoicesView: View {
    @EnvironmentObject var app: AppModel
    @State private var filter = "All"
    @State private var sample = "It's 10:30 on Tuesday, and the train to Boston leaves in 25 minutes. Don't forget your umbrella!"
    @State private var advanced = false
    @State private var favs: Set<String> = []

    private static let main: [(String, String, String, String)] = [
        ("f0_med", "Pitch", "Lower", "Higher"), ("f0_iqr", "Expressiveness", "Flatter", "Livelier"),
        ("loud", "Loudness", "Softer", "Louder"), ("f0_end", "Endings", "Fall", "Rise")]
    private static let more: [(String, String, String, String)] = [
        ("rate", "Speaking rate", "Slower", "Faster"), ("tilt", "Brightness", "Darker", "Brighter"),
        ("h1h2", "Breathiness", "Pressed", "Breathy"), ("centroid", "Timbre", "Duller", "Sharper"),
        ("pause", "Pauses", "Fewer", "More"), ("voiced", "Voicing", "Less", "More"),
        ("flat", "Steadiness", "Varied", "Steady"), ("cpps", "Clarity", "Rougher", "Clearer")]
    private static let filters = ["All", "Female", "Male", "Favourites", "Deep", "Warm", "Bright", "Relaxed", "Brisk"]

    var body: some View {
        HSplitView {
            voiceList.frame(minWidth: 280, idealWidth: 330, maxWidth: 420)
            ScrollView { controls.padding(28).frame(maxWidth: 640, alignment: .leading) }
                .frame(minWidth: 380, maxWidth: .infinity)
        }
        .onAppear { favs = app.voices.favourites() }
    }

    // ---------------------------------------------------------------- voices

    private var voiceList: some View {
        VStack(spacing: 0) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(Self.filters, id: \.self) { f in
                        Button(f == "Favourites" ? "★ Favourites" : f) { filter = f }
                            .buttonStyle(Chip(on: filter == f))
                    }
                }
                .padding(10)
            }
            Divider()
            List {
                if filter == "All" {
                    Section("Featured") { ForEach(app.voices.featured(), id: \.id) { row($0) } }
                    Section("All voices") { ForEach(shown.filter { !Voices.FEATURED.contains($0.id) }, id: \.id) { row($0) } }
                } else {
                    ForEach(shown, id: \.id) { row($0) }
                }
            }
            .listStyle(.inset)
        }
    }

    private var shown: [Voices.Voice] {
        app.voices.all().filter { v in
            switch filter {
            case "Female": return v.gender == "female"
            case "Male": return v.gender == "male"
            case "Favourites": return favs.contains(v.id)
            case "Deep": return v.pitch == "low"
            case "Bright": return v.pitch == "high"
            case "Warm": return v.pitch != "low" && v.pitch != "high"
            case "Relaxed": return v.pace == "slow"
            case "Brisk": return v.pace == "fast"
            default: return true
            }
        }
    }

    private func row(_ v: Voices.Voice) -> some View {
        let on = (app.engineVoice(app.style.voice)) == v.id
        return HStack(spacing: 10) {
            Image(systemName: on ? "checkmark.circle.fill" : "circle").foregroundStyle(on ? Theme.accent : .secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(v.name).font(.body.weight(on ? .semibold : .regular))
                Text("\(v.genderLabel) · \(v.describe())").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Button { app.voices.toggleFavourite(v.id); favs = app.voices.favourites() } label: {
                Image(systemName: favs.contains(v.id) ? "star.fill" : "star")
                    .foregroundStyle(favs.contains(v.id) ? Color.yellow : Color.secondary)
            }
            .buttonStyle(.plain)
            .help(favs.contains(v.id) ? "Remove from favourites" : "Add to favourites")
        }
        .contentShape(Rectangle())
        .onTapGesture { change { $0.voice = v.id } }
        .padding(.vertical, 2)
    }

    // ---------------------------------------------------------------- style and delivery

    private var controls: some View {
        VStack(alignment: .leading, spacing: 22) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Try it").font(.headline)
                HStack(alignment: .top) {
                    TextField("Something to hear", text: $sample, axis: .vertical)
                        .lineLimit(2...4).textFieldStyle(.roundedBorder)
                    Button { Speaker.shared.say(sample, style: app.style) } label: { Label("Play", systemImage: "play.fill") }
                        .buttonStyle(.borderedProminent)
                }
                Text("\(app.voiceName(app.style.voice)) · \(app.style.preset)").font(.callout).foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Style").font(.headline)
                FlowChips(items: Style.STYLES, selected: app.style.preset) { s in change { $0.applyPreset(s) } }
            }

            VStack(alignment: .leading, spacing: 14) {
                Text("Delivery").font(.headline)
                SliderRow(label: "Speed", lo: "Slower", hi: "Faster", value: Double(app.style.pace),
                          range: Double(Style.MIN_PACE)...Double(Style.MAX_PACE), shown: String(format: "%.2f×", app.style.pace)) { v in
                    change { $0.pace = Float(v) }
                }
                ForEach(Self.main, id: \.0) { c in control(c) }
                DisclosureGroup("Advanced", isExpanded: $advanced) {
                    VStack(spacing: 14) { ForEach(Self.more, id: \.0) { c in control(c) } }.padding(.top, 10)
                }
                HStack {
                    Spacer()
                    Button("Reset Delivery") { change { s in s.z = [Float](repeating: 0, count: Style.CONTROLS.count); s.pace = Style.DEFAULT_PACE } }
                }
            }
        }
    }

    private func control(_ c: (String, String, String, String)) -> some View {
        let i = Style.CONTROLS.firstIndex(of: c.0)!
        return SliderRow(label: c.1, lo: c.2, hi: c.3, value: Double(app.style.z[i]), range: -Double(Style.LIMIT)...Double(Style.LIMIT),
                         shown: app.style.z[i] == 0 ? "Natural" : String(format: "%+.1f", app.style.z[i])) { v in
            change { $0.z[i] = Float((v * 10).rounded() / 10) }
        }
    }

    /// Changes the base style, re-renders what the reader has queued and previews the change.
    private func change(_ f: (Style) -> Void) {
        let s = app.style.copy()
        f(s)
        app.style = s
        app.narrator.restyle()
        if !app.narrator.playing { Speaker.shared.say(sample, style: s) }
    }
}

struct SliderRow: View {
    let label: String, lo: String, hi: String
    let value: Double
    let range: ClosedRange<Double>
    let shown: String
    let commit: (Double) -> Void
    @State private var live: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                Spacer()
                Text(shown).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: Binding(get: { live ?? value }, set: { live = $0 }), in: range) {
                Text(label)
            } minimumValueLabel: {
                Text(lo).font(.caption).foregroundStyle(.secondary)
            } maximumValueLabel: {
                Text(hi).font(.caption).foregroundStyle(.secondary)
            } onEditingChanged: { editing in
                if !editing, let v = live { commit(v); live = nil }
            }
            .labelsHidden()
        }
    }
}

struct Chip: ButtonStyle {
    let on: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.callout)
            .padding(.horizontal, 11).padding(.vertical, 5)
            .foregroundStyle(on ? Theme.accent : .primary)
            .background(on ? Theme.accentSoft : Color(nsColor: .controlBackgroundColor), in: Capsule())
            .overlay(Capsule().stroke(on ? Theme.accent : Color(nsColor: .separatorColor)))
            .opacity(configuration.isPressed ? 0.7 : 1)
    }
}

/// Chips that wrap onto as many rows as they need.
struct FlowChips: View {
    let items: [String]
    let selected: String
    let pick: (String) -> Void

    var body: some View {
        Flow(spacing: 6) {
            ForEach(items, id: \.self) { s in Button(s) { pick(s) }.buttonStyle(Chip(on: s == selected)) }
        }
    }
}

struct Flow: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(proposal.width ?? .infinity, subviews)
        return CGSize(width: proposal.width ?? rows.width, height: rows.height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowH: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x + s.width > bounds.maxX && x > bounds.minX { x = bounds.minX; y += rowH + spacing; rowH = 0 }
            v.place(at: CGPoint(x: x, y: y), proposal: .unspecified)
            x += s.width + spacing
            rowH = max(rowH, s.height)
        }
    }

    private func arrange(_ width: CGFloat, _ subviews: Subviews) -> (width: CGFloat, height: CGFloat) {
        var x: CGFloat = 0, y: CGFloat = 0, rowH: CGFloat = 0, maxW: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x + s.width > width && x > 0 { x = 0; y += rowH + spacing; rowH = 0 }
            x += s.width + spacing
            maxW = max(maxW, x)
            rowH = max(rowH, s.height)
        }
        return (maxW, y + rowH)
    }
}
