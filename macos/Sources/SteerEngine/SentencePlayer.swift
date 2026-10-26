import AVFoundation
import Foundation

/// Plays rendered speech as a queue of short items, one per sentence (or announcement), through AVQueuePlayer, so the
/// system's AirPlay picker can send it to a HomePod, an Apple TV or any AirPlay speaker, and the item being heard tells
/// the screen which sentence that is. Rendering runs ahead on another thread and calls enqueue(); each item is a small
/// WAV file in the caches folder, deleted once heard. A flush() (seek, new speed, stop) drops everything queued; audio
/// from a render that started before it is ignored (the token).
public final class SentencePlayer: NSObject {
    public let player = AVQueuePlayer()

    /// Main thread: the item with this mark started (a sentence index, or -1 for an intro).
    public var onMark: ((Int) -> Void)?
    /// Main thread: everything queued has been heard (rendering may still be catching up, or the end was reached).
    public var onDrained: (() -> Void)?

    private struct Queued {
        let item: AVPlayerItem
        let mark: Int
        let url: URL
        let seconds: Double
    }

    private var queued: [Queued] = []   // main thread
    private let lock = NSLock()
    private var token = 0
    private var pending = 0.0           // seconds queued and not yet finished (lock)
    private var wantsPlay = false
    private let dir: URL
    private var observation: NSKeyValueObservation?

    public override init() {
        dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SteerTTS/audio", isDirectory: true)
        try? FileManager.default.removeItem(at: dir)   // left from a previous run
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        super.init()
        player.actionAtItemEnd = .advance
        observation = player.observe(\.currentItem, options: [.new]) { [weak self] p, _ in
            DispatchQueue.main.async { self?.currentChanged() }
        }
        NotificationCenter.default.addObserver(self, selector: #selector(itemEnded(_:)),
                                               name: AVPlayerItem.didPlayToEndTimeNotification, object: nil)
    }

    /// The current token, for enqueue: audio rendered for an older one is dropped.
    public var currentToken: Int { lock.lock(); defer { lock.unlock() }; return token }

    /// Seconds of audio queued ahead (any thread).
    public var pendingSeconds: Double { lock.lock(); defer { lock.unlock() }; return pending }

    /// Queues audio (any thread). Returns false if the token is stale (a flush happened since).
    @discardableResult
    public func enqueue(_ samples: [Float], mark: Int, token t: Int) -> Bool {
        guard t == currentToken, !samples.isEmpty else { return false }
        let url = dir.appendingPathComponent(UUID().uuidString + ".wav")
        do { try wavData(samples).write(to: url) } catch { return false }
        let seconds = Double(samples.count) / Double(Engine.sampleRate)
        lock.lock()
        let fresh = t == token
        if fresh { pending += seconds }
        lock.unlock()
        guard fresh else { try? FileManager.default.removeItem(at: url); return false }
        DispatchQueue.main.async { [self] in
            guard t == currentToken else { try? FileManager.default.removeItem(at: url); return }
            let item = AVPlayerItem(url: url)
            queued.append(Queued(item: item, mark: mark, url: url, seconds: seconds))
            player.insert(item, after: nil)
            if wantsPlay && player.rate == 0 { player.play() }
        }
        return true
    }

    /// Drops everything queued and playing; returns the new token. Main thread.
    @discardableResult
    public func flush() -> Int {
        lock.lock()
        token += 1
        pending = 0
        let t = token
        lock.unlock()
        player.removeAllItems()
        for q in queued { try? FileManager.default.removeItem(at: q.url) }
        queued.removeAll()
        return t
    }

    public func play() {
        wantsPlay = true
        player.play()
    }

    public func pause() {
        wantsPlay = false
        player.pause()
    }

    public var isPlaying: Bool { wantsPlay }

    /// Volume of the voice, 0 ... 1.
    public var volume: Float {
        get { player.volume }
        set { player.volume = newValue }
    }

    private func currentChanged() {
        guard let cur = player.currentItem, let q = queued.first(where: { $0.item === cur }) else { return }
        onMark?(q.mark)
    }

    @objc private func itemEnded(_ n: Notification) {
        guard let item = n.object as? AVPlayerItem else { return }
        DispatchQueue.main.async { [self] in
            guard let i = queued.firstIndex(where: { $0.item === item }) else { return }
            let q = queued.remove(at: i)
            try? FileManager.default.removeItem(at: q.url)
            lock.lock()
            pending = max(0, pending - q.seconds)
            lock.unlock()
            if queued.isEmpty { onDrained?() }
        }
    }
}
