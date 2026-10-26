import AppKit
import CoreAudio
import DiskArbitration
import EventKit
import Foundation
import IOKit.ps
import SteerText

/// Things the Mac itself does that are worth a word: a download finished, the battery running low or full, a drive
/// connected or ejected, AirPods or headphones connected (then anything that waited is caught up on, as on Android).
@MainActor
final class SystemEvents {
    static let shared = SystemEvents()
    private var downloads: DispatchSourceFileSystemObject?
    private var known: Set<String> = []
    private var lastBattery: (level: Int, charging: Bool)?
    private var batterySource: CFRunLoopSource?
    private var outputName = ""

    func start() {
        watchDownloads()
        watchBattery()
        watchDrives()
        watchOutput()
    }

    private func say(_ kind: Assistant.Event.Kind, _ app: String, _ s: Speech, who: String = "", away: String? = nil, urgent: Bool = false) {
        Assistant.shared.handle(.init(kind: kind, app: app, bundleID: nil, who: who, speech: s, urgent: urgent, away: away))
    }

    // ---------------------------------------------------------------- downloads

    private var downloadsDir: URL { FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0] }

    private func watchDownloads() {
        known = Set((try? FileManager.default.contentsOfDirectory(atPath: downloadsDir.path)) ?? [])
        let fd = open(downloadsDir.path, O_EVTONLY)
        guard fd >= 0 else { return }
        let src = DispatchSource.makeFileSystemObjectSource(fileDescriptor: fd, eventMask: .write, queue: .main)
        src.setEventHandler { [weak self] in MainActor.assumeIsolated { self?.downloadsChanged() } }
        src.setCancelHandler { close(fd) }
        src.resume()
        downloads = src
    }

    private static let partial = "(?i)\\.(crdownload|download|part|partial|tmp|opdownload)$"

    private func downloadsChanged() {
        let now = Set((try? FileManager.default.contentsOfDirectory(atPath: downloadsDir.path)) ?? [])
        let fresh = now.subtracting(known).filter { !$0.hasPrefix(".") && rx(Self.partial).find($0) == nil }
        known = now
        for name in fresh {
            // wait until the file stops growing (a browser renames it into place, a copy may still be writing)
            settle(downloadsDir.appendingPathComponent(name), last: -1)
        }
    }

    private func settle(_ url: URL, last: Int64) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            guard AssistantPrefs.shared.downloads,
                  let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? Int64 else { return }
            if size != last { self.settle(url, last: size); return }
            let name = url.deletingPathExtension().lastPathComponent.replaceAll("[-_]+", " ").replaceAll("\\s*\\(\\d+\\)$", "")
            let kind = Self.kindName(url.pathExtension)
            let line = "Downloaded \(name)\(kind.isEmpty ? "" : ", \(kind)")."
            self.say(.download, "Downloads", Speech().add(line, SteerText.Expression.CALM), who: url.lastPathComponent, away: line)
        }
    }

    static func kindName(_ ext: String) -> String {
        ["pdf": "a PDF", "zip": "a zip file", "dmg": "a disk image", "pkg": "an installer", "png": "an image", "jpg": "a photo",
         "jpeg": "a photo", "heic": "a photo", "mp4": "a video", "mov": "a video", "mp3": "audio", "m4a": "audio", "csv": "a spreadsheet",
         "xlsx": "a spreadsheet", "docx": "a Word document", "pptx": "a presentation", "epub": "a book", "txt": "text"][ext.lowercased()] ?? ""
    }

    // ---------------------------------------------------------------- battery

    private func watchBattery() {
        let ctx = Unmanaged.passUnretained(self).toOpaque()
        guard let src = IOPSNotificationCreateRunLoopSource({ ctx in
            guard let ctx else { return }
            let me = Unmanaged<SystemEvents>.fromOpaque(ctx).takeUnretainedValue()
            MainActor.assumeIsolated { me.batteryChanged() }
        }, ctx)?.takeRetainedValue() else { return }
        CFRunLoopAddSource(CFRunLoopGetMain(), src, .defaultMode)
        batterySource = src
        lastBattery = Self.battery()
    }

    static func battery() -> (level: Int, charging: Bool)? {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let list = IOPSCopyPowerSourcesList(info)?.takeRetainedValue() as? [CFTypeRef] else { return nil }
        for ps in list {
            guard let d = IOPSGetPowerSourceDescription(info, ps)?.takeUnretainedValue() as? [String: Any],
                  d[kIOPSTypeKey] as? String == kIOPSInternalBatteryType,
                  let cur = d[kIOPSCurrentCapacityKey] as? Int, let max = d[kIOPSMaxCapacityKey] as? Int, max > 0 else { continue }
            let onAC = d[kIOPSPowerSourceStateKey] as? String == kIOPSACPowerValue
            return (cur * 100 / max, onAC)
        }
        return nil
    }

    private func batteryChanged() {
        guard AssistantPrefs.shared.battery, let now = Self.battery() else { return }
        defer { lastBattery = now }
        guard let was = lastBattery else { return }
        if !now.charging {
            for mark in [20, 10, 5] where was.level > mark && now.level <= mark {
                say(.battery, "Battery", Readout.battery(now.level, false), away: "The battery got down to \(now.level) percent.",
                    urgent: mark <= 10)
                return
            }
        } else if was.level < 100 && now.level >= 100 {
            say(.battery, "Battery", Speech().add("The battery is fully charged.", SteerText.Expression.CALM))
        }
    }

    // ---------------------------------------------------------------- drives

    private func watchDrives() {
        let nc = NSWorkspace.shared.notificationCenter
        nc.addObserver(forName: NSWorkspace.didMountNotification, object: nil, queue: .main) { n in
            guard let url = n.userInfo?[NSWorkspace.volumeURLUserInfoKey] as? URL else { return }
            MainActor.assumeIsolated { self.drive(url, mounted: true) }
        }
        nc.addObserver(forName: NSWorkspace.didUnmountNotification, object: nil, queue: .main) { n in
            guard let url = n.userInfo?[NSWorkspace.volumeURLUserInfoKey] as? URL else { return }
            MainActor.assumeIsolated { self.drive(url, mounted: false) }
        }
    }

    private func drive(_ url: URL, mounted: Bool) {
        guard AssistantPrefs.shared.devices else { return }
        let v = try? url.resourceValues(forKeys: [.volumeIsInternalKey, .volumeLocalizedNameKey, .volumeIsRemovableKey])
        if v?.volumeIsInternal == true { return }
        let name = v?.volumeLocalizedName ?? url.lastPathComponent
        // disk images (an installer opened from Downloads) mount too: worth a word only for a real drive
        if mounted && Self.isDiskImage(url) { return }
        say(.device, "Drives", Speech().add(mounted ? "\(name) is connected." : "\(name) was ejected.", SteerText.Expression.CALM))
    }

    static func isDiskImage(_ url: URL) -> Bool {
        guard let session = DASessionCreate(kCFAllocatorDefault),
              let disk = DADiskCreateFromVolumePath(kCFAllocatorDefault, session, url as CFURL),
              let d = DADiskCopyDescription(disk) as? [String: Any] else { return false }
        return (d[kDADiskDescriptionDeviceModelKey as String] as? String)?.contains("Disk Image") == true
    }

    // ---------------------------------------------------------------- headphones

    private func watchOutput() {
        outputName = Self.outputDevice()?.name ?? ""
        var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultOutputDevice,
                                              mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &addr, .main) { _, _ in
            MainActor.assumeIsolated { self.outputChanged() }
        }
    }

    private func outputChanged() {
        guard let d = Self.outputDevice(), d.name != outputName else { return }
        outputName = d.name
        guard AssistantPrefs.shared.devices, d.bluetooth || d.name.lowercased().contains("headphone") else { return }
        say(.device, "Sound", Speech().add("\(d.name) connected.", SteerText.Expression.CALM))
        // headphones on: hear what waited, as Android's catch-up on connecting headphones
        if !Assistant.shared.waiting.isEmpty {
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { Assistant.shared.catchUp() }
        }
    }

    static func outputDevice() -> (name: String, bluetooth: Bool)? {
        var dev = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultOutputDevice,
                                              mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &dev) == noErr else { return nil }
        var name: Unmanaged<CFString>?
        size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        addr.mSelector = kAudioObjectPropertyName
        guard AudioObjectGetPropertyData(dev, &addr, 0, nil, &size, &name) == noErr, let n = name?.takeRetainedValue() else { return nil }
        var transport = UInt32(0)
        size = UInt32(MemoryLayout<UInt32>.size)
        addr.mSelector = kAudioDevicePropertyTransportType
        AudioObjectGetPropertyData(dev, &addr, 0, nil, &size, &transport)
        return (n as String, transport == kAudioDeviceTransportTypeBluetooth || transport == kAudioDeviceTransportTypeBluetoothLE)
    }
}

/// Calendar events a few minutes before they start, as Android's MeetingAnnouncer: "In 5 minutes: standup with Arun,
/// on Google Meet." From the Mac's calendars (permission asked when turned on); all-day and declined events skipped.
@MainActor
final class Meetings {
    static let shared = Meetings()
    private let store = EKEventStore()
    private var said: Set<String> = []
    private var timer: Timer?

    var allowed: Bool { EKEventStore.authorizationStatus(for: .event) == .fullAccess }

    func start() {
        guard timer == nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { _ in MainActor.assumeIsolated { self.check() } }
    }

    func requestAccess(_ done: @escaping (Bool) -> Void) {
        store.requestFullAccessToEvents { ok, _ in DispatchQueue.main.async { done(ok) } }
    }

    private func check() {
        guard AssistantPrefs.shared.meetings, allowed else { return }
        let now = Date()
        let pred = store.predicateForEvents(withStart: now, end: now.addingTimeInterval(6 * 60), calendars: nil)
        for e in store.events(matching: pred) where !e.isAllDay {
            let key = (e.eventIdentifier ?? e.title ?? "") + "\(e.startDate.timeIntervalSince1970)"
            let mins = Int((e.startDate.timeIntervalSince(now) / 60).rounded())
            guard !said.contains(key), mins <= 5, mins >= 0,
                  e.attendees?.first(where: { $0.isCurrentUser })?.participantStatus != .declined, e.status != .canceled else { continue }
            said.insert(key)
            let s = Speech().add(Self.phrase(e, minutes: mins), mins <= 1 ? SteerText.Expression.URGENT : ClaudeEvents.attentive)
            Assistant.shared.handle(.init(kind: .meeting, app: "Calendar", bundleID: nil, who: e.title ?? "", speech: s, urgent: mins <= 1,
                                          away: nil))
        }
    }

    static func phrase(_ e: EKEvent, minutes: Int) -> String {
        let when = minutes <= 0 ? "Starting now" : minutes == 1 ? "In a minute" : "In \(minutes) minutes"
        var s = "\(when): \(e.title ?? "an event")"
        let others = (e.attendees ?? []).filter { !$0.isCurrentUser }.compactMap { $0.name?.split(separator: " ").first.map(String.init) }
        if others.count == 1 { s += " with \(others[0])" } else if others.count == 2 { s += " with \(others[0]) and \(others[1])" }
        let place = [e.location ?? "", e.url?.absoluteString ?? "", e.notes ?? ""].joined(separator: " ").lowercased()
        if place.contains("zoom.us") { s += ", on Zoom" } else if place.contains("meet.google") { s += ", on Google Meet" }
        else if place.contains("teams.microsoft") { s += ", on Teams" }
        else if let l = e.location, !l.isEmpty, !l.contains("http") { s += ", in \(l)" }
        return s + "."
    }
}
