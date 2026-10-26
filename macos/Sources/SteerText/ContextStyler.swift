import Foundation

/// Adjusts the user's base style to what is being said and the device's state: alerts faster, louder and brighter;
/// questions rising at the end; good news livelier; navigation and screen-reader speech brisk and clear; late at night,
/// or with the ringer off, softer (not for alerts; never slower: the default voice is already a slow speaker).
/// Adjustments add to the base controls (z-scores), so the user's own settings stay the starting point. The rules lean
/// on loud, f0_med, rate, tilt and h1h2: the controls the model follows reliably (docs/request.md).
/// Ported from ContextStyler.java: only the pure rules; the device state (hour, ringer, battery) is passed in.
public enum ContextStyler {
    /// What the device is doing, as the Android version read it from the phone.
    public struct DeviceState {
        public var hour: Int              // 0-23, local time
        public var quietRinger: Bool      // ringer silent or vibrate (on a Mac: e.g. Focus / Do Not Disturb)
        public var lowBattery: Bool       // under 15 % and not charging

        public init(hour: Int, quietRinger: Bool = false, lowBattery: Bool = false) {
            self.hour = hour
            self.quietRinger = quietRinger
            self.lowBattery = lowBattery
        }

        /// Now, with the ringer and battery as given.
        public static func now(quietRinger: Bool = false, lowBattery: Bool = false) -> DeviceState {
            DeviceState(hour: Calendar.current.component(.hour, from: Date()), quietRinger: quietRinger, lowBattery: lowBattery)
        }
    }

    // other apps' words (system voice): only clear alerts and clear good news; ordinary UI words ("done", "error",
    // "timer", "security") are left alone, they are labels a screen reader reads all day
    private static let ALERT = rx("\\b(urgent|warning|emergency|alarm|critical|danger|fraud alert|low battery|battery (is )?low|wake up|incoming call)\\b")
    private static let GOOD = rx("\\b(congratulations|congrats|great news|happy birthday|well done)\\b")

    public static let MAPS = "com.google.android.apps.maps", WAZE = "com.waze",
        TALKBACK = "com.google.android.marvin.talkback"
    // caller may also be one of the Announcer's sources: "calendar", "read-aloud", or the package a message is from

    /// The style for one utterance. caller: the requesting app (nil if unknown). textCues false: only the situation
    /// (the requesting app, time of day, ringer, battery); the words are left to the caller's own moods (parts that
    /// carry Expression moods, which would otherwise count twice). By default a screen reader's labels carry no mood.
    public static func decide(_ base: Style, text: String, caller: String?, device: DeviceState,
                              textCues: Bool? = nil) -> Style {
        let textCues = textCues ?? (caller != TALKBACK)
        let s = base.copy()
        var why: [String] = []
        if !base.contextual {
            s.reason = "your settings (context styling off)"
            return s
        }
        let t = text.lowercased()
        let trimmed = jtrim(text)
        var alert = textCues && (ALERT.find(t) != nil || trimmed.hasSuffix("!!"))
        if let c = caller, c.contains("deskclock") || c.contains("alarm") { alert = true }
        if alert {
            s.add("rate", 0.8)
            s.add("loud", 1.2)
            s.add("f0_med", 0.5)
            s.add("tilt", 0.8)
            s.add("pause", -0.5)
            why.append("alert: faster, louder, brighter")
        } else if textCues && GOOD.find(t) != nil {
            s.add("f0_iqr", 1.2)
            s.add("f0_med", 0.4)
            s.add("loud", 0.3)
            why.append("good news: livelier")
        }
        if textCues && trimmed.hasSuffix("?") {
            s.add("f0_end", 1.5)
            why.append("question: rising end")
        }
        let navigation = caller == MAPS || caller == WAZE
        if navigation {
            s.add("rate", 0.3)
            s.add("loud", 0.6)
            why.append("navigation: clear")
        } else if caller == TALKBACK {
            s.add("rate", 1.0)
            why.append("screen reader: brisk")
        }
        if !alert && !navigation {   // directions while driving stay clear at night and on silent
            if device.hour >= 22 || device.hour < 7 {
                s.add("loud", -1.2)
                s.add("h1h2", 0.6)
                why.append("night: softer")
            } else if device.quietRinger {
                s.add("loud", -0.8)
                why.append("ringer off: softer")
            }
            if device.lowBattery {
                s.add("rate", -0.3)
                why.append("battery low: calmer")
            }
        }
        s.reason = why.isEmpty ? "neutral: your settings" : why.joined(separator: "; ")
        return s
    }
}

/// Java's String.trim(): code units <= U+0020 off both ends.
fileprivate func jtrim(_ s: String) -> String {
    let u = Array(s.utf16)
    var a = 0, b = u.count
    while a < b && u[a] <= 0x20 { a += 1 }
    while b > a && u[b - 1] <= 0x20 { b -= 1 }
    return a == 0 && b == u.count ? s : s.jsub(a, b)
}
