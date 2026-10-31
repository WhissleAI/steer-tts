import AppKit
import SwiftUI

/// The app's look: Whissle's green as the accent (the Android app's palette, light and dark), the system's own
/// materials and controls everywhere else, and New York (the system serif) for reading.
enum Theme {
    static let accent = Color(light: 0x1A6A55, dark: 0x6FD8B8)
    static let accentSoft = Color(light: 0xE1F1EB, dark: 0x1D3A31)
    static let red = Color(light: 0xF14E52, dark: 0xFF7A7E)
    static let mint = Color(light: 0xACEDDC, dark: 0xACEDDC)

    /// Book text: the system serif at the reader's chosen size.
    static func reading(_ size: CGFloat) -> Font { .system(size: size, design: .serif) }
}

extension Color {
    /// A colour with a light and a dark value, following the window's appearance.
    init(light: UInt32, dark: UInt32) {
        self.init(nsColor: NSColor(name: nil) { a in
            let v = a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
            return NSColor(srgbRed: CGFloat(v >> 16 & 255) / 255, green: CGFloat(v >> 8 & 255) / 255,
                           blue: CGFloat(v & 255) / 255, alpha: 1)
        })
    }
}

/// The SteerTTS mark (the pill waveform with the voice in red and the mint dot), drawn in SwiftUI for the menu bar,
/// the empty states and the about box.
struct Mark: View {
    var mono = false

    var body: some View {
        Canvas { ctx, size in
            let k = min(size.width, size.height) / 62
            let ox = (size.width - 62 * k) / 2, oy = (size.height - 62 * k) / 2
            func pill(_ cx: CGFloat, _ top: CGFloat, _ bottom: CGFloat, _ c: Color) {
                let r = CGRect(x: ox + (cx - 3.42) * k, y: oy + top * k, width: 6.84 * k, height: (bottom - top) * k)
                ctx.fill(Path(roundedRect: r, cornerRadius: 3.42 * k), with: .color(c))
            }
            // the 108-unit Android canvas, cropped to the mark (x 23 .. 85, y 32 .. 71, centred)
            let w = mono ? Color.primary : Color.white
            for (cx, t, b) in [(27.0, 52.95, 61.05), (36, 49.35, 64.65), (45, 44.85, 69.15), (63, 44.85, 69.15),
                               (72, 49.35, 64.65), (81, 52.95, 61.05)] as [(CGFloat, CGFloat, CGFloat)] {
                pill(cx - 23, t - 20.5, b - 20.5, w)
            }
            pill(54 - 23, 43.5 - 20.5, 70.5 - 20.5, mono ? .primary : Theme.red)
            let c = CGPoint(x: ox + (54 - 23) * k, y: oy + (36.12 - 20.5) * k), r = 3.78 * k
            ctx.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                     with: .color(mono ? .primary : Theme.mint))
        }
    }
}

/// The mark on its green rounded square, as in the app icon.
struct AppBadge: View {
    var size: CGFloat = 64

    var body: some View {
        RoundedRectangle(cornerRadius: size * 0.225, style: .continuous)
            .fill(LinearGradient(colors: [Color(light: 0x268A71, dark: 0x268A71), Color(light: 0x124E3F, dark: 0x124E3F)],
                                 startPoint: .top, endPoint: .bottom))
            .overlay(Mark().padding(size * 0.2))
            .frame(width: size, height: size)
    }
}
