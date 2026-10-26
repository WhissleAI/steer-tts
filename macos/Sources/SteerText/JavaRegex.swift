import Foundation

// Java's regex idioms over NSRegularExpression (ICU, whose syntax is close to java.util.regex), so the Android
// app's plain-Java text code ports line for line. Indices are UTF-16 offsets, as in Java and NSString.

private let cacheLock = NSLock()
private var cache: [String: NSRegularExpression] = [:]

/// A compiled pattern, cached by its source. A bad pattern is a programming error, so it traps.
public func rx(_ pattern: String) -> NSRegularExpression {
    cacheLock.lock()
    defer { cacheLock.unlock() }
    if let r = cache[pattern] { return r }
    let r = try! NSRegularExpression(pattern: pattern)
    cache[pattern] = r
    return r
}

/// One match, with Java's Matcher accessors.
public struct JMatch {
    public let result: NSTextCheckingResult
    let source: NSString

    /// Group i's text, or nil when it did not take part (Java's group(i) == null).
    public func group(_ i: Int = 0) -> String? {
        guard i < result.numberOfRanges else { return nil }
        let r = result.range(at: i)
        return r.location == NSNotFound ? nil : source.substring(with: r)
    }

    public func group(_ name: String) -> String? {
        let r = result.range(withName: name)
        return r.location == NSNotFound ? nil : source.substring(with: r)
    }

    public func start(_ i: Int = 0) -> Int { result.range(at: i).location }
    public func end(_ i: Int = 0) -> Int { NSMaxRange(result.range(at: i)) }
}

public extension NSRegularExpression {
    /// Matcher.find() from the start.
    func find(_ s: String, from: Int = 0) -> JMatch? {
        let ns = s as NSString
        guard from <= ns.length,
              let m = firstMatch(in: s, range: NSRange(location: from, length: ns.length - from)) else { return nil }
        return JMatch(result: m, source: ns)
    }

    /// Every match, in order (a Matcher.find() loop).
    func all(_ s: String) -> [JMatch] {
        let ns = s as NSString
        return matches(in: s, range: NSRange(location: 0, length: ns.length)).map { JMatch(result: $0, source: ns) }
    }

    /// Matcher.matches(): the whole string.
    func whole(_ s: String) -> Bool {
        let ns = s as NSString
        guard let m = firstMatch(in: s, options: [.anchored], range: NSRange(location: 0, length: ns.length)) else { return false }
        return m.range.length == ns.length
    }

    /// Matcher.lookingAt(): a match at the start.
    func atStart(_ s: String) -> JMatch? {
        let ns = s as NSString
        guard let m = firstMatch(in: s, options: [.anchored], range: NSRange(location: 0, length: ns.length)) else { return nil }
        return JMatch(result: m, source: ns)
    }

    /// replaceAll with a template ($1, $2; \$ for a dollar sign), as Java.
    func replaceAll(_ s: String, _ template: String) -> String {
        stringByReplacingMatches(in: s, range: NSRange(location: 0, length: (s as NSString).length), withTemplate: template)
    }

    /// replaceFirst with a template.
    func replaceFirst(_ s: String, _ template: String) -> String {
        guard let m = find(s) else { return s }
        let ns = s as NSString
        let rep = replacementString(for: m.result, in: s, offset: 0, template: template)
        return ns.replacingCharacters(in: m.result.range, with: rep)
    }

    /// Matcher.replaceAll(Function<MatchResult, String>): the replacement is used literally.
    func replaceAll(_ s: String, _ f: (JMatch) -> String) -> String {
        let ns = s as NSString
        var out = ""
        var last = 0
        for m in all(s) {
            out += ns.substring(with: NSRange(location: last, length: m.start() - last))
            out += f(m)
            last = m.end()
        }
        out += ns.substring(from: last)
        return out
    }

    /// Pattern.split(s) with Java's rules: trailing empty strings removed; a zero-width match at the start does not
    /// produce a leading empty string.
    func split(_ s: String) -> [String] {
        let ns = s as NSString
        var parts: [String] = []
        var last = 0
        for m in all(s) {
            if m.end() == 0 { continue }
            parts.append(ns.substring(with: NSRange(location: last, length: m.start() - last)))
            last = m.end()
        }
        parts.append(ns.substring(from: last))
        while let l = parts.last, l.isEmpty, parts.count > 0 { parts.removeLast() }
        return parts.isEmpty && ns.length == 0 ? [""] : parts
    }
}

public extension String {
    /// String.matches(regex): the whole string.
    func matches(_ pattern: String) -> Bool { rx(pattern).whole(self) }
    /// String.replaceAll(regex, template).
    func replaceAll(_ pattern: String, _ template: String) -> String { rx(pattern).replaceAll(self, template) }
    /// String.replaceFirst(regex, template).
    func replaceFirst(_ pattern: String, _ template: String) -> String { rx(pattern).replaceFirst(self, template) }
    /// String.split(regex), Java's rules.
    func split(regex pattern: String) -> [String] { rx(pattern).split(self) }
    /// Java's String.trim() is ASCII-only; this is the Unicode-aware strip() most ports want.
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
    /// Length in UTF-16 code units, as Java's length().
    var jlength: Int { (self as NSString).length }
    /// substring(begin, end) in UTF-16 offsets, as Java.
    func jsub(_ begin: Int, _ end: Int? = nil) -> String {
        let ns = self as NSString
        let e = end ?? ns.length
        return ns.substring(with: NSRange(location: begin, length: e - begin))
    }
    /// charAt in UTF-16 code units.
    func jchar(_ i: Int) -> unichar { (self as NSString).character(at: i) }
}
