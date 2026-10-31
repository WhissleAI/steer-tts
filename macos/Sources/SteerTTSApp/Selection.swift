import AppKit
import ApplicationServices

/// The text selected in the app in front, for "Read selection" from anywhere. Read through the accessibility API
/// (the user allows SteerTTS once, in System Settings > Privacy & Security > Accessibility); apps that do not expose
/// their selection that way (some web views, Electron apps) get a copy command instead, with the clipboard put back
/// as it was afterwards.
enum Selection {
    static var trusted: Bool { AXIsProcessTrusted() }

    /// Shows the system prompt that leads to the Accessibility setting.
    static func askForAccess() {
        let opts = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(opts)
    }

    /// The selection, or nil (no access, nothing selected).
    static func read(completion: @escaping (String?) -> Void) {
        guard trusted else { completion(nil); return }
        if let s = viaAccessibility(), !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            completion(s)
            return
        }
        viaCopy(completion: completion)
    }

    private static func viaAccessibility() -> String? {
        let system = AXUIElementCreateSystemWide()
        var focused: CFTypeRef?
        guard AXUIElementCopyAttributeValue(system, kAXFocusedUIElementAttribute as CFString, &focused) == .success,
              let el = focused else { return nil }
        var sel: CFTypeRef?
        guard AXUIElementCopyAttributeValue(el as! AXUIElement, kAXSelectedTextAttribute as CFString, &sel) == .success else {
            return nil
        }
        return sel as? String
    }

    /// Sends ⌘C to the app in front and reads what it copied; the clipboard is restored after.
    private static func viaCopy(completion: @escaping (String?) -> Void) {
        let pb = NSPasteboard.general
        let saved = pb.pasteboardItems?.map { item -> NSPasteboardItem in
            let copy = NSPasteboardItem()
            for t in item.types { if let d = item.data(forType: t) { copy.setData(d, forType: t) } }
            return copy
        } ?? []
        let before = pb.changeCount
        let src = CGEventSource(stateID: .combinedSessionState)
        let down = CGEvent(keyboardEventSource: src, virtualKey: 0x08, keyDown: true)   // C
        let up = CGEvent(keyboardEventSource: src, virtualKey: 0x08, keyDown: false)
        down?.flags = .maskCommand
        up?.flags = .maskCommand
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
            let text = pb.changeCount != before ? pb.string(forType: .string) : nil
            if pb.changeCount != before {
                pb.clearContents()
                if !saved.isEmpty { pb.writeObjects(saved) }
            }
            completion(text)
        }
    }
}
