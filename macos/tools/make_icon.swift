// Makes Resources/AppIcon.icns: the Android launcher mark (res/drawable/ic_launcher_foreground.xml, drawn here as
// vectors so every size is sharp) on its Whissle green gradient, in a macOS rounded square (824 of 1024 px, as Apple's
// icon grid) with a soft shadow. Run from macos/: swift tools/make_icon.swift
import AppKit

func rgb(_ hex: UInt32) -> NSColor {
    NSColor(srgbRed: CGFloat(hex >> 16 & 255) / 255, green: CGFloat(hex >> 8 & 255) / 255, blue: CGFloat(hex & 255) / 255, alpha: 1)
}

/// The mark on the 108-unit adaptive-icon canvas (y down), mapped into box.
func drawMark(in box: NSRect) {
    let k = box.width / 108
    func pt(_ x: CGFloat, _ y: CGFloat) -> NSPoint { NSPoint(x: box.minX + x * k, y: box.maxY - y * k) }
    func pill(_ cx: CGFloat, _ top: CGFloat, _ bottom: CGFloat, _ color: NSColor) {
        let r = 3.42 * k
        let a = pt(cx - 3.42, bottom), b = pt(cx + 3.42, top)
        color.setFill()
        NSBezierPath(roundedRect: NSRect(x: a.x, y: a.y, width: b.x - a.x, height: b.y - a.y), xRadius: r, yRadius: r).fill()
    }
    let white = rgb(0xFFFFFF)
    for (cx, top, bottom) in [(27.0, 52.95, 61.05), (36, 49.35, 64.65), (45, 44.85, 69.15), (63, 44.85, 69.15),
                              (72, 49.35, 64.65), (81, 52.95, 61.05)] {
        pill(cx, top, bottom, white)
    }
    pill(54, 43.5, 70.5, rgb(0xF14E52))
    let c = pt(54, 36.12), r = 3.78 * k
    rgb(0xACEDDC).setFill()
    NSBezierPath(ovalIn: NSRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)).fill()
}
let set = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("AppIcon.iconset")
try? FileManager.default.removeItem(at: set)
try! FileManager.default.createDirectory(at: set, withIntermediateDirectories: true)

func render(_ px: Int) -> Data {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px, bitsPerSample: 8, samplesPerPixel: 4,
                               hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let s = CGFloat(px) / 1024
    let box = NSRect(x: 100 * s, y: 100 * s + 10 * s, width: 824 * s, height: 824 * s)
    let path = NSBezierPath(roundedRect: box, xRadius: 185 * s, yRadius: 185 * s)
    let shadow = NSShadow()
    shadow.shadowColor = NSColor.black.withAlphaComponent(0.3)
    shadow.shadowBlurRadius = 18 * s
    shadow.shadowOffset = NSSize(width: 0, height: -8 * s)
    NSGraphicsContext.saveGraphicsState()
    shadow.set()
    rgb(0x1D6B58).setFill()
    path.fill()
    NSGraphicsContext.restoreGraphicsState()
    path.addClip()
    NSGradient(starting: rgb(0x268A71), ending: rgb(0x124E3F))!.draw(in: box, angle: -90)
    drawMark(in: box)
    NSGraphicsContext.restoreGraphicsState()
    return rep.representation(using: .png, properties: [:])!
}

for size in [16, 32, 128, 256, 512] {
    try! render(size).write(to: set.appendingPathComponent("icon_\(size)x\(size).png"))
    try! render(size * 2).write(to: set.appendingPathComponent("icon_\(size)x\(size)@2x.png"))
}
let p = Process()
p.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
p.arguments = ["-c", "icns", set.path, "-o", "Resources/AppIcon.icns"]
try! p.run()
p.waitUntilExit()
print(p.terminationStatus == 0 ? "Resources/AppIcon.icns" : "iconutil failed")
