import AVFoundation
import Foundation

/// Writes the voice to an audio file as it is rendered: AAC in .m4a (an audiobook, about 1 MB per 10 minutes) or
/// 16-bit WAV. Chunks are 24 kHz mono float, as the engine renders them.
public final class AudioFileWriter {
    private let file: AVAudioFile
    private let format = AVAudioFormat(standardFormatWithSampleRate: Double(Engine.sampleRate), channels: 1)!
    public private(set) var seconds = 0.0

    public init(url: URL) throws {
        try? FileManager.default.removeItem(at: url)
        let settings: [String: Any]
        if url.pathExtension.lowercased() == "wav" {
            settings = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: Engine.sampleRate, AVNumberOfChannelsKey: 1,
                        AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false]
        } else {
            settings = [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: Engine.sampleRate, AVNumberOfChannelsKey: 1,
                        AVEncoderBitRateKey: 64000]
        }
        file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    }

    public func write(_ samples: [Float]) throws {
        guard !samples.isEmpty,
              let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        buf.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { buf.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        try file.write(from: buf)
        seconds += Double(samples.count) / Double(Engine.sampleRate)
    }

    public func silence(ms: Int) throws { try write([Float](repeating: 0, count: Engine.sampleRate * ms / 1000)) }
}
