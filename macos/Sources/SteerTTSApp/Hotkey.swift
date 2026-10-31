import AppKit
import Carbon.HIToolbox

/// A system-wide keyboard shortcut (Carbon's RegisterEventHotKey: works in every app and needs no permission).
final class Hotkey {
    private static var handlers: [UInt32: () -> Void] = [:]
    private static var installed = false
    private var ref: EventHotKeyRef?
    private let id: UInt32

    /// keyCode: a kVK_ constant; modifiers: Carbon's cmdKey, optionKey, controlKey, shiftKey.
    init?(keyCode: Int, modifiers: Int, id: UInt32, action: @escaping () -> Void) {
        self.id = id
        Hotkey.install()
        Hotkey.handlers[id] = action
        let hk = EventHotKeyID(signature: OSType(0x5354_5453), id: id)   // 'STTS'
        guard RegisterEventHotKey(UInt32(keyCode), UInt32(modifiers), hk, GetApplicationEventTarget(), 0, &ref) == noErr else {
            return nil
        }
    }

    deinit {
        if let ref { UnregisterEventHotKey(ref) }
        Hotkey.handlers[id] = nil
    }

    private static func install() {
        guard !installed else { return }
        installed = true
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, event, _ in
            var hk = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID), nil,
                              MemoryLayout<EventHotKeyID>.size, nil, &hk)
            DispatchQueue.main.async { Hotkey.handlers[hk.id]?() }
            return noErr
        }, 1, &spec, nil, nil)
    }
}
