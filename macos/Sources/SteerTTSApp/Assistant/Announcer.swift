import Foundation
import SteerEngine
import SteerText

/// The assistant's voice on the Mac, as Android's Announcer: what it says is a Speech, parts that may sound different
/// ("Priya on WhatsApp" in the assistant's voice, then her message in its mood and her voice). Each part is rendered
/// with the user's style, adjusted for the situation (ContextStyler: alerts brighter, questions rising, late at night
/// softer) and then the part's own mood and voice. Requests queue; an interrupting one (a call) drops what is queued or
/// playing. A book being read pauses while the assistant speaks and goes on after. What could not be said now
/// (you're away, in a call, in a Focus) waits for "welcome back".
@MainActor
final class Announcer: ObservableObject {
    static let shared = Announcer()

    @Published private(set) var speaking: String?
    private let player = SentencePlayer()
    private var queue: [(Speech, String?)] = []   // (what, the app or kind it is about, for ContextStyler)
    private var busy = false
    private var resumeReading = false
    private let gate = Gate()
    private var gen = 0

    private init() {
        player.onDrained = { [weak self] in self?.next() }
    }

    /// Says s now (or after what is being said); interrupt: a call, it goes first and cuts the rest.
    func say(_ s: Speech, about kind: String? = nil, interrupt: Bool = false) {
        guard !s.isEmpty else { return }
        if interrupt {
            queue.removeAll()
            gen += 1
            gate.set(gen)
            player.flush()
            busy = false
        }
        queue.append((s, kind))
        if !busy { next() }
    }

    /// Volume of the assistant's voice, 0 ... 1.
    var volume: Float {
        get { player.volume }
        set { player.volume = newValue }
    }

    func stop() {
        queue.removeAll()
        gen += 1
        gate.set(gen)
        player.flush()
        player.pause()
        finish()
    }

    private func next() {
        guard !queue.isEmpty else { finish(); return }
        let (s, kind) = queue.removeFirst()
        let narrator = AppModel.shared.narrator!
        if narrator.playing {
            narrator.pause()
            resumeReading = true
        }
        busy = true
        speaking = s.text()
        let g = gen
        let token = player.flush()
        player.play()
        let app = AppModel.shared
        let base = app.style.copy()
        let device = ContextStyler.DeviceState.now(quietRinger: Machine.focusOn())
        let player = self.player, gate = self.gate
        app.onEngine { engine in
            // all parts first, then one item: an announcement is short, and the queue must not run dry halfway
            let own = engine.resolve(voice: base.voice)
            var all: [Float] = []
            for p in s.parts {
                guard gate.current(g) else { return }
                let st = Performer.moodStyle(ContextStyler.decide(base, text: p.text, caller: kind, device: device), p.mood)
                let voice = p.voice.flatMap { engine.gender(of: $0) != nil ? $0 : nil } ?? own
                guard var y = try? engine.synthesize(SteerText.Expression.bmp(p.text), voice: voice, tag: st.tagOrNull(),
                                                     ctrl: st.ctrl(), pace: st.pace, markup: false) else { continue }
                if p.pauseAfterMs > 0 { y += [Float](repeating: 0, count: Engine.sampleRate * p.pauseAfterMs / 1000) }
                all += y
            }
            all += [Float](repeating: 0, count: Engine.sampleRate / 3)   // a short breath after each announcement
            if gate.current(g) { player.enqueue(all, mark: 0, token: token) }
        }
    }

    private func finish() {
        busy = false
        speaking = nil
        if resumeReading {
            resumeReading = false
            AppModel.shared.narrator.toggle()
        }
    }
}
