// swift-tools-version:5.9
// SteerTTS for the Mac: the C engine (built from ../c/src through the symlinks in Sources/CSteerTTS), the text
// pipeline in plain Swift (SteerText, unit tested), the engine and playback (SteerEngine) and the app (SteerTTSApp).
// Build the app bundle with ./build.sh; run the tests with `swift test`.
import PackageDescription

let package = Package(
    name: "SteerTTSMac",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "SteerTTS", targets: ["SteerTTSApp"])],
    targets: [
        .target(
            name: "CSteerTTS",
            path: "Sources/CSteerTTS",
            cSettings: [
                .define("STEERTTS_BUILD"),
                .headerSearchPath("src"),
                // as the CMake build: no FMA contraction, so renders round exactly as the reference does
                .unsafeFlags(["-O2", "-ffp-contract=off", "-Wno-unused-parameter"]),
            ]),
        .target(name: "SteerText", path: "Sources/SteerText"),
        .target(name: "SteerEngine", dependencies: ["CSteerTTS", "SteerText"], path: "Sources/SteerEngine"),
        .executableTarget(name: "SteerTTSApp", dependencies: ["SteerEngine", "SteerText"], path: "Sources/SteerTTSApp"),
        .testTarget(name: "SteerTextTests", dependencies: ["SteerText"], path: "Tests/SteerTextTests"),
    ]
)
