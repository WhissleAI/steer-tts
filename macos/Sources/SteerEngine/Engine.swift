import CSteerTTS
import Foundation
import SteerText

/// The SteerTTS edge engine (the C library, linked in). Calls are serialized on the instance, so a voice query from one
/// thread waits for a render on another; cancel() is the one call that does not wait (it is how a stream is cut short).
/// Every text is said through Spoken first: numbers, times, dates, amounts, Roman numerals, links and accented letters
/// in words, which the engine's own front end reads only in part.
public final class Engine {
    public static let sampleRate = 24000

    /// The 12 prosody controls, in the order Style uses (z-scores, about -3 .. 3 in practice).
    public static let controls = ["rate", "loud", "f0_med", "f0_iqr", "f0_end", "pause", "tilt", "voiced", "flat",
                                  "centroid", "cpps", "h1h2"]

    public struct Failure: LocalizedError {
        public let message: String
        public var errorDescription: String? { message }
    }

    private var h: OpaquePointer?
    private let lock = NSLock()
    private let cancelLock = NSLock()
    private var cancelled = false

    /// threads: worker threads for one render (0 = one per CPU). Loading takes a few milliseconds (the weights are
    /// memory-mapped).
    public init(modelDir: URL, threads: Int = 4) throws {
        var err = [CChar](repeating: 0, count: 256)
        guard let e = steertts_create(modelDir.path, Int32(threads), &err, err.count) else {
            throw Failure(message: String(cString: err))
        }
        h = e
    }

    deinit { if let h { steertts_destroy(h) } }

    // ---------------------------------------------------------------- the model

    public var voices: [String] {
        locked { h in (0..<Int(steertts_num_voices(h))).compactMap { str(steertts_voice(h, Int32($0))) } } ?? []
    }

    /// "female", "male" or nil.
    public func gender(of voice: String) -> String? {
        locked { h -> String? in
            let n = Int(steertts_num_voices(h))
            for i in 0..<n where str(steertts_voice(h, Int32(i))) == voice { return str(steertts_voice_gender(h, Int32(i))) }
            return nil
        } ?? nil
    }

    /// The default voice of "female" / "male".
    public func voice(ofGender g: String) -> String? { locked { h in str(steertts_gender_voice(h, g)) } ?? nil }

    public var defaultVoice: String? { locked { h in str(steertts_default_voice(h)) } ?? nil }

    /// Style tags ("none", "happiness", "whisper", ...).
    public var tags: [String] {
        locked { h in (0..<Int(steertts_num_tags(h))).compactMap { str(steertts_tag(h, Int32($0))) } } ?? []
    }

    /// A voice name the engine has: female / male / nil pick that gender's default, an unknown name the female one.
    public func resolve(voice v: String?) -> String? {
        if v == nil || v == "female" { return voice(ofGender: "female") }
        if v == "male" { return voice(ofGender: "male") }
        return gender(of: v!) != nil ? v : voice(ofGender: "female")
    }

    // ---------------------------------------------------------------- rendering

    /// One render. markup: parse *word*, [pause 300ms] and <whisper>..</whisper> in text. ctrl: "rate=1,loud=-0.5".
    /// nil if cancel() was called since the last resetCancel().
    public func synthesize(_ text: String, voice: String? = nil, tag: String? = nil, ctrl: String? = nil,
                           pace: Float = 1, markup: Bool = false) throws -> [Float]? {
        let say = Spoken.say(text, markup: markup)
        return try locked { h -> [Float]? in
            if isCancelled { return nil }
            var o = steertts_options()
            steertts_default_options(&o)
            o.pace = pace
            o.markup = markup ? 1 : 0
            var err = [CChar](repeating: 0, count: 256)
            var audio: UnsafeMutablePointer<Float>?
            var n = 0
            let rc = withOptional(voice) { v in withOptional(tag) { t in withOptional(ctrl) { c -> Int32 in
                o.voice = v
                o.tag = t
                o.ctrl = c
                return steertts_synthesize(h, say, &o, &audio, &n, &err, err.count)
            } } }
            guard rc == 0, let audio else { throw Failure(message: String(cString: err)) }
            defer { steertts_free(audio) }
            return Array(UnsafeBufferPointer(start: audio, count: n))
        } ?? nil
    }

    /// Streams text sentence by sentence (markup parsed). chunk returns false to stop. Returns after the last chunk,
    /// or after the current sentence once cancel() is called.
    public func stream(_ text: String, voice: String? = nil, tag: String? = nil, ctrl: String? = nil, pace: Float = 1,
                       chunk: @escaping ([Float], _ index: Int) -> Bool) throws {
        let say = Spoken.say(text, markup: true)
        final class Box {
            let f: ([Float], Int) -> Bool
            let engine: Engine
            init(_ f: @escaping ([Float], Int) -> Bool, _ e: Engine) { self.f = f; engine = e }
        }
        let box = Unmanaged.passRetained(Box(chunk, self))
        defer { box.release() }
        try locked { h in
            var o = steertts_options()
            steertts_default_options(&o)
            o.pace = pace
            o.markup = 1
            var err = [CChar](repeating: 0, count: 256)
            let rc = withOptional(voice) { v in withOptional(tag) { t in withOptional(ctrl) { c -> Int32 in
                o.voice = v
                o.tag = t
                o.ctrl = c
                return steertts_stream_ex(h, say, &o, { audio, n, info, user in
                    let b = Unmanaged<Box>.fromOpaque(user!).takeUnretainedValue()
                    if b.engine.isCancelled { return 1 }
                    let a = Array(UnsafeBufferPointer(start: audio, count: n))
                    return b.f(a, Int(info?.pointee.index ?? 0)) ? 0 : 1
                }, box.toOpaque(), &err, err.count)
            } } }
            if rc != 0 { throw Failure(message: String(cString: err)) }
        }
    }

    /// Stops a running stream() from another thread (it returns after the current sentence), and makes the next
    /// synthesize() return nil, until resetCancel().
    public func cancel() { cancelLock.lock(); cancelled = true; cancelLock.unlock() }

    /// Forget an earlier cancel(): call at the start of a new request.
    public func resetCancel() { cancelLock.lock(); cancelled = false; cancelLock.unlock() }

    private var isCancelled: Bool { cancelLock.lock(); defer { cancelLock.unlock() }; return cancelled }

    // ---------------------------------------------------------------- helpers

    private func locked<T>(_ f: (OpaquePointer) throws -> T) rethrows -> T? {
        lock.lock()
        defer { lock.unlock() }
        guard let h else { return nil }
        return try f(h)
    }
}

private func str(_ p: UnsafePointer<CChar>?) -> String? { p.map { String(cString: $0) } }

/// Runs f with a C string for s, or NULL.
private func withOptional<T>(_ s: String?, _ f: (UnsafePointer<CChar>?) throws -> T) rethrows -> T {
    guard let s else { return try f(nil) }
    return try s.withCString { try f($0) }
}

/// The voice as 16-bit PCM WAV bytes (24 kHz mono), for a file or a speaker that fetches it.
public func wavData(_ samples: [Float], sampleRate: Int = Engine.sampleRate) -> Data {
    var d = Data()
    func put32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
    func put16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
    let bytes = UInt32(samples.count * 2)
    d.append(contentsOf: Array("RIFF".utf8)); put32(36 + bytes); d.append(contentsOf: Array("WAVEfmt ".utf8))
    put32(16); put16(1); put16(1); put32(UInt32(sampleRate)); put32(UInt32(sampleRate * 2)); put16(2); put16(16)
    d.append(contentsOf: Array("data".utf8)); put32(bytes)
    d.reserveCapacity(d.count + samples.count * 2)
    for s in samples { put16(UInt16(bitPattern: Int16(max(-1, min(1, s)) * 32767))) }
    return d
}
