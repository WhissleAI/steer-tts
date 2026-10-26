import Foundation

/// A delivery style: voice, style tag and the 12 controls (z-scores against the voice's own neutral speech, CONTROLS
/// order; 0 = left to the model). The user's base style is saved as a small JSON file (the same format as the Android
/// app's style.json), read before every system utterance, so a change in the app applies to the next one.
/// Ported from Style.java; the Android Context and files dir are replaced by an explicit file URL.
public final class Style {
    /// The 12 prosody controls, in the order Style uses (z-scores, about -3 .. 3 in practice); SteerTTS.CONTROLS on Android.
    public static let CONTROLS = ["rate", "loud", "f0_med", "f0_iqr", "f0_end", "pause", "tilt", "voiced",
                                  "flat", "centroid", "cpps", "h1h2"]
    public static let LIMIT: Float = 3   // beyond +-3 is outside the model's well-tested range

    public var voice: String?                 // a voice name, "female" / "male", or nil for the female default
    public var tag: String = "none"
    public var z = [Float](repeating: 0, count: Style.CONTROLS.count)
    public var contextual = true              // let ContextStyler adjust system utterances
    public var pace: Float = Style.DEFAULT_PACE   // speaking speed (1 = the voice's own; the default voice is a slow speaker)
    public var preset = "Natural"             // the Voices screen's style chip (the tag and controls hold the delivery itself)
    public static let DEFAULT_PACE: Float = 1.1, MIN_PACE: Float = 0.7, MAX_PACE: Float = 1.6
    public var smooth = false                 // "Smoother sound": the Smooth filter on every voice
    public var reason = ""                    // why a context style looks the way it does (shown in the app)

    public init() {}

    public func copy() -> Style {
        let s = Style()
        s.voice = voice
        s.tag = tag
        s.z = z
        s.contextual = contextual
        s.pace = pace
        s.preset = preset
        s.smooth = smooth
        s.reason = reason
        return s
    }

    /// One-tap deliveries for the "Say it" chips: a style tag plus the controls the model follows reliably (most emotion
    /// tags alone change little; voice-quality tags such as whisper act strongly).
    public static let PRESETS = ["Happy", "Whisper", "Excited", "Sad", "Calm"]

    /// The voice's styles on the Voices screen: Natural (no style) plus the deliveries that clearly work.
    public static let STYLES = ["Natural", "Calm", "Warm", "Happy", "Excited", "Serious", "Sad", "Whisper"]

    /// The style, as the user's base: the controls reset, then the preset's tag and controls; the name is kept.
    public func applyPreset(_ name: String) {
        z = [Float](repeating: 0, count: z.count)
        tag = "none"
        let p = preset(name)
        tag = p.tag
        z = p.z
        preset = name
    }

    public func preset(_ name: String) -> Style {
        let s = copy()
        switch name {
        case "Natural": s.tag = "none"
        case "Warm": s.tag = "contentment"; s.add("f0_iqr", 0.6); s.add("h1h2", 0.6); s.add("rate", -0.2)
        case "Serious": s.tag = "confidence"; s.add("f0_med", -0.6); s.add("f0_iqr", -0.5); s.add("loud", 0.4); s.add("f0_end", -0.8)
        case "Calm": s.tag = "serenity"; s.add("rate", -0.4); s.add("f0_iqr", -0.6); s.add("loud", -0.4)
        case "Happy": s.tag = "happiness"; s.add("f0_iqr", 1.3); s.add("f0_med", 0.7); s.add("rate", 0.4); s.add("tilt", 0.5)
        case "Excited": s.tag = "excited"; s.add("rate", 0.9); s.add("loud", 1.1); s.add("f0_iqr", 1.6); s.add("f0_med", 1.0)
        case "Sad": s.tag = "sadness"; s.add("rate", -0.8); s.add("f0_med", -0.9); s.add("f0_iqr", -0.9); s.add("loud", -0.8)
        case "Whisper": s.tag = "whisper"
        default: break
        }
        s.contextual = false   // the chip decides the delivery
        s.reason = name.lowercased() + ": " + (s.tag == "none" ? "" : s.tag + " + ") + "controls"
        return s
    }

    public func add(_ control: String, _ dz: Float) {
        for i in 0..<z.count where Style.CONTROLS[i] == control { z[i] = Style.clamp(z[i] + dz) }
    }

    /// A mood's shifts on top (Expression.Mood.ctrl), in its order.
    public func add(_ mood: Expression.Mood) {
        for (k, v) in mood.pairs { add(k, v) }
    }

    public static func clamp(_ v: Float) -> Float { max(-LIMIT, min(LIMIT, v)) }

    /// "rate=1.20,loud=-0.50" with the controls that are set, or nil.
    public func ctrl() -> String? {
        var parts: [String] = []
        for i in 0..<z.count where abs(z[i]) >= 0.05 {
            parts.append(Style.CONTROLS[i] + "=" + javaFixed2(z[i]))
        }
        return parts.isEmpty ? nil : parts.joined(separator: ",")
    }

    public func tagOrNull() -> String? { tag == "none" ? nil : tag }

    // ---------------------------------------------------------------- JSON (the Android style.json format)
    /// {"voice": string|null, "tag", "z": [12 numbers], "contextual", "pace", "preset", "smooth", "reason"}.
    public func toJson() -> [String: Any] {
        [
            "voice": voice.map { $0 as Any } ?? NSNull(),
            "tag": tag,
            "z": z.map { Double($0) },
            "contextual": contextual,
            "pace": Double(pace),
            "preset": preset,
            "smooth": smooth,
            "reason": reason,
        ]
    }

    public func jsonData() -> Data {
        (try? JSONSerialization.data(withJSONObject: toJson(), options: [.sortedKeys])) ?? Data("{}".utf8)
    }

    public static func fromJson(_ o: [String: Any]) -> Style {
        let s = Style()
        s.voice = o["voice"] is NSNull ? nil : optString(o["voice"])
        s.tag = optString(o["tag"]) ?? "none"
        if let a = o["z"] as? [Any] {
            for i in 0..<min(a.count, s.z.count) { s.z[i] = clamp(Float(optDouble(a[i]) ?? 0)) }
        }
        s.contextual = optBool(o["contextual"]) ?? true
        s.pace = max(MIN_PACE, min(MAX_PACE, Float(optDouble(o["pace"]) ?? Double(DEFAULT_PACE))))
        s.preset = optString(o["preset"]) ?? "Natural"
        s.smooth = optBool(o["smooth"]) ?? false
        s.reason = optString(o["reason"]) ?? ""
        return s
    }

    public static func fromJson(data: Data) -> Style? {
        guard let o = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        return fromJson(o)
    }

    /// The style saved at url (Android: files/style.json), or the defaults when there is none or it does not parse.
    public static func load(from url: URL) -> Style {
        guard let d = try? Data(contentsOf: url), let s = fromJson(data: d) else { return Style() }
        return s
    }

    /// Saves atomically (a temporary file renamed over it): a reader never sees half a file.
    public func save(to url: URL) {
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? jsonData().write(to: url, options: .atomic)
    }
}

// org.json's opt* coercions, as far as they matter here
fileprivate func optString(_ v: Any?) -> String? {
    switch v {
    case let s as String: return s
    case nil, is NSNull: return nil
    case let n as NSNumber: return n.stringValue
    default: return "\(v!)"
    }
}

fileprivate func optDouble(_ v: Any?) -> Double? {
    switch v {
    case let n as NSNumber: return n.doubleValue
    case let s as String: return Double(s.trimmingCharacters(in: .whitespaces))
    default: return nil
    }
}

fileprivate func optBool(_ v: Any?) -> Bool? {
    switch v {
    case let n as NSNumber where CFGetTypeID(n) == CFBooleanGetTypeID(): return n.boolValue
    case let s as String where s.lowercased() == "true": return true
    case let s as String where s.lowercased() == "false": return false
    default: return nil
    }
}

/// String.format(Locale.US, "%.2f", float) as Java does it: the float's shortest decimal form (Float.toString), rounded
/// half up, not the binary value rounded half even as printf would ("0.125" -> "0.13").
fileprivate func javaFixed2(_ v: Float) -> String {
    guard v.isFinite, var d = Decimal(string: "\(v)") else { return String(format: "%.2f", Double(v)) }
    var r = Decimal()
    NSDecimalRound(&r, &d, 2, .plain)
    let f = NumberFormatter()
    f.locale = Locale(identifier: "en_US_POSIX")
    f.minimumFractionDigits = 2
    f.maximumFractionDigits = 2
    f.minimumIntegerDigits = 1
    f.usesGroupingSeparator = false
    return f.string(from: r as NSDecimalNumber) ?? String(format: "%.2f", Double(v))
}
