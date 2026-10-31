import AVFoundation
import Foundation
import MediaPlayer
import SteerEngine
import SteerText

/// Reads a Doc aloud, sentence by sentence, with the controls a listener expects: play, pause, read from a sentence,
/// skip back and forward, speed, a sleep timer; when a document ends the next in the reading list starts (one-off
/// snippets excepted). Each sentence is performed as on Android (Narration): headings slower, dialogue in a second
/// voice and its own mood, characters in voices of their own (Cast), italics stressed. Sentences are rendered a few
/// seconds ahead of playback on the engine's own queue; the item being heard tells the screen which sentence that is.
/// Position is saved as it goes. Media keys, Control Center and AirPlay come with the player (SentencePlayer).
@MainActor
final class Narrator: ObservableObject {
    @Published private(set) var doc: Doc?
    @Published private(set) var heard = 0            // the sentence being heard (or next to be heard)
    @Published private(set) var playing = false
    @Published private(set) var preparing = false    // play pressed, no sound yet
    @Published var problem: String?
    @Published var speed: Float = UserDefaults.standard.object(forKey: "speed") as? Float ?? 1 {
        didSet { UserDefaults.standard.set(speed, forKey: "speed"); if playing { renderFrom(heard) } }
    }
    @Published var dialogueVoice = UserDefaults.standard.object(forKey: "dialogueVoice") as? Bool ?? true {
        didSet { UserDefaults.standard.set(dialogueVoice, forKey: "dialogueVoice"); if playing { renderFrom(heard) } }
    }
    @Published private(set) var sleepAt: Date?

    let player = SentencePlayer()
    private unowned let app: AppModel
    private var gen = 0                       // bumped on every new render; a render for an older one stops
    private let gate = Gate()                 // gen, readable from the engine queue
    private var sleepTimer: Timer?
    private static let ahead = 6.0            // seconds rendered ahead of what is heard

    init(app: AppModel) {
        self.app = app
        player.onMark = { [weak self] i in self?.heardSentence(i) }
        player.onDrained = { [weak self] in self?.drained() }
        NowPlaying.setUp(self)
    }

    /// At launch: the document last listened to, paused where it was left, so the player is ready to resume.
    func restoreLast() {
        guard doc == nil, let m = app.library.docs.first(where: { $0.opened > $0.added || $0.pos > 0 }),
              let d = app.library.doc(m.id) else { return }
        doc = d
        heard = d.done ? 0 : min(d.pos, max(0, d.sentences.count - 1))
        NowPlaying.update(self)
    }

    // ---------------------------------------------------------------- controls

    /// Opens a document and plays it from where it was left (or from sentence `from`).
    func play(_ d: Doc, from: Int? = nil) {
        if doc?.id != d.id { savePosition() }
        doc = d
        heard = max(0, min(from ?? (d.done ? 0 : d.pos), max(0, d.sentences.count - 1)))
        d.opened = Date().millis
        app.library.save(d)
        start()
    }

    func toggle() {
        if playing { pause() } else if doc != nil { start() } else if let m = app.library.docs.first, let d = app.library.doc(m.id) { play(d) }
    }

    func pause() {
        bump()
        _ = player.flush()
        player.pause()
        playing = false
        preparing = false
        savePosition()
        NowPlaying.update(self)
    }

    /// Reads from sentence i.
    func seek(_ i: Int) {
        guard let d = doc else { return }
        heard = max(0, min(i, d.sentences.count - 1))
        savePosition()
        if playing { renderFrom(heard) } else { NowPlaying.update(self) }
    }

    func skip(_ n: Int) { seek(heard + n) }

    /// The voice or style changed: what is queued is re-rendered from the sentence being heard.
    func restyle() { if playing { renderFrom(heard) } }

    /// Back to the start of the paragraph (or the one before, when already at its start).
    func previousParagraph() {
        guard let d = doc, !d.paraOf.isEmpty else { return }
        let p = d.paraOf[min(heard, d.paraOf.count - 1)]
        var i = heard
        while i > 0 && d.paraOf[i - 1] == p { i -= 1 }
        if i == heard && i > 0 {
            let q = d.paraOf[i - 1]
            i -= 1
            while i > 0 && d.paraOf[i - 1] == q { i -= 1 }
        }
        seek(i)
    }

    func nextParagraph() {
        guard let d = doc, !d.paraOf.isEmpty else { return }
        let p = d.paraOf[min(heard, d.paraOf.count - 1)]
        var i = heard
        while i < d.paraOf.count && d.paraOf[i] == p { i += 1 }
        seek(min(i, d.sentences.count - 1))
    }

    func stop() {
        pause()
        doc = nil
        NowPlaying.update(self)
    }

    /// Stops reading after `minutes` (nil: off; 0: at the end of the chapter, i.e. the current document).
    func sleep(after minutes: Int?) {
        sleepTimer?.invalidate()
        sleepTimer = nil
        sleepAt = nil
        guard let m = minutes, m > 0 else { return }
        let at = Date().addingTimeInterval(TimeInterval(m * 60))
        sleepAt = at
        sleepTimer = Timer.scheduledTimer(withTimeInterval: TimeInterval(m * 60), repeats: false) { [weak self] _ in
            Task { @MainActor in self?.pause(); self?.sleepAt = nil }
        }
    }

    /// Seconds of listening left in the document at the current speed (from the words still to read).
    var secondsLeft: Double {
        guard let d = doc else { return 0 }
        var chars = 0
        for i in heard..<d.sentences.count { chars += d.sentences[i].count }
        return Double(chars) / 15.0 / Double(max(0.5, speed * app.style.pace))
    }

    var progress: Double {
        guard let d = doc, d.sentences.count > 1 else { return 0 }
        return Double(heard) / Double(d.sentences.count - 1)
    }

    // ---------------------------------------------------------------- playback

    private func start() {
        guard doc != nil else { return }
        playing = true
        problem = nil
        player.play()
        renderFrom(heard)
    }

    private func bump() {
        gen += 1
        gate.set(gen)
    }

    private func renderFrom(_ start: Int) {
        guard let d = doc else { return }
        bump()
        let g = gen
        preparing = true
        let token = player.flush()
        player.play()
        NowPlaying.update(self)
        let style = app.style, speed = self.speed, second = dialogueVoice, voices = app.voices
        let player = self.player, gate = self.gate
        app.onEngine { [weak self] engine in
            do {
                let perf = Performer(doc: d, style: style, speed: speed, dialogueVoices: second, voices: voices, engine: engine)
                for i in start..<d.sentences.count {
                    while player.pendingSeconds > Narrator.ahead && gate.current(g) { usleep(40_000) }
                    guard gate.current(g), let audio = try perf.render(i, with: engine), gate.current(g) else { return }
                    player.enqueue(audio, mark: i, token: token)
                }
                if gate.current(g) { player.enqueue([Float](repeating: 0, count: Engine.sampleRate / 4), mark: -2, token: token) }
            } catch {
                let message = error.localizedDescription
                DispatchQueue.main.async { self?.failed(g, message) }
            }
        }
    }

    /// The voice could not render: back to paused with a message, instead of a Pause button over silence.
    private func failed(_ g: Int, _ message: String) {
        guard gen == g else { return }
        pause()
        problem = "The voice could not read this: \(message)"
    }

    private func heardSentence(_ i: Int) {
        preparing = false
        guard let d = doc else { return }
        if i == -2 { finished(); return }
        guard i >= 0 && i < d.sentences.count else { return }
        heard = i
        if i % 5 == 0 { savePosition() }
        NowPlaying.update(self)
    }

    private func drained() {
        // the queue ran dry: rendering is behind (rare on a Mac) or the document has ended (the end marker plays first)
        if playing && preparing == false { preparing = true }
    }

    private func finished() {
        guard let d = doc else { return }
        d.pos = 0
        d.done = true
        app.library.save(d)
        playing = false
        preparing = false
        player.pause()
        NowPlaying.update(self)
        // the next document in the reading list (not one-off snippets: copied or typed text plays only when asked)
        if UserDefaults.standard.object(forKey: "autoNext") as? Bool ?? true, let next = app.library.next(after: d.id) {
            play(next)
        }
    }

    private func savePosition() {
        guard let d = doc else { return }
        d.pos = heard
        if heard > 0 { d.done = false }
        app.library.save(d)
    }
}

/// The narrator's render generation, shared with the engine queue.
final class Gate: @unchecked Sendable {
    private let lock = NSLock()
    private var g = 0
    func set(_ v: Int) { lock.lock(); g = v; lock.unlock() }
    func current(_ v: Int) -> Bool { lock.lock(); defer { lock.unlock() }; return g == v }
}

extension Date {
    var millis: Int64 { Int64(timeIntervalSince1970 * 1000) }
}

/// The system's Now Playing (Control Center, the menu bar's media item, media keys, AirPods): what is being read and
/// play / pause / previous / next.
@MainActor
enum NowPlaying {
    static func setUp(_ n: Narrator) {
        let c = MPRemoteCommandCenter.shared()
        c.togglePlayPauseCommand.addTarget { _ in Task { @MainActor in n.toggle() }; return .success }
        c.playCommand.addTarget { _ in Task { @MainActor in if !n.playing { n.toggle() } }; return .success }
        c.pauseCommand.addTarget { _ in Task { @MainActor in n.pause() }; return .success }
        c.nextTrackCommand.addTarget { _ in Task { @MainActor in n.nextParagraph() }; return .success }
        c.previousTrackCommand.addTarget { _ in Task { @MainActor in n.previousParagraph() }; return .success }
        c.skipForwardCommand.isEnabled = false
        c.skipBackwardCommand.isEnabled = false
    }

    static func update(_ n: Narrator) {
        let center = MPNowPlayingInfoCenter.default()
        guard let d = n.doc else {
            center.nowPlayingInfo = nil
            center.playbackState = .stopped
            return
        }
        let total = n.secondsLeft / max(0.001, 1 - n.progress)
        center.nowPlayingInfo = [
            MPMediaItemPropertyTitle: d.title,
            MPMediaItemPropertyArtist: d.author.isEmpty ? "SteerTTS" : d.author,
            MPMediaItemPropertyPlaybackDuration: total,
            MPNowPlayingInfoPropertyElapsedPlaybackTime: total - n.secondsLeft,
            MPNowPlayingInfoPropertyPlaybackRate: n.playing ? 1.0 : 0.0,
        ]
        center.playbackState = n.playing ? .playing : .paused
    }
}
