import Compression
import Foundation

/// A small reader for ZIP archives (EPUB and DOCX files are ZIPs; macOS has no public unzip API). It reads the central
/// directory at the end of the file, then each entry's data through its local header: stored entries as they are,
/// deflated ones through the Compression framework (COMPRESSION_ZLIB is raw deflate, as ZIP stores it). ZIP64 archives
/// (more than 65,535 entries or 4 GB) are read through their ZIP64 end record and extra fields; encrypted entries,
/// split archives and other compression methods throw.
public struct ZipArchive {
    public enum Failure: Error, Equatable {
        case notAZip            // no end-of-central-directory record
        case corrupt(String)    // a record that does not hold together
        case unsupported(String)
        case tooLarge(String)   // an entry larger than the caller's limit
    }

    public struct Entry: Equatable {
        public let name: String
        /// 0 stored, 8 deflate.
        public let method: Int
        public let flags: Int
        public let compressedSize: Int
        public let uncompressedSize: Int
        let localHeaderOffset: Int

        public var isDirectory: Bool { name.hasSuffix("/") }
    }

    public let entries: [Entry]
    private let data: Data

    public init(data: Data) throws {
        // a Data slice keeps its parent's indices; rebase so offsets are from 0
        self.data = data.startIndex == 0 ? data : Data(data)
        self.entries = try ZipArchive.centralDirectory(self.data)
    }

    public init(contentsOf url: URL) throws {
        try self.init(data: Data(contentsOf: url, options: .mappedIfSafe))
    }

    /// The first entry with this exact name.
    public func entry(named name: String) -> Entry? { entries.first { $0.name == name } }

    /// An entry's bytes, uncompressed; throws tooLarge past `limit` bytes (checked before anything is inflated).
    public func contents(of e: Entry, limit: Int = Int.max) throws -> Data {
        if e.flags & 1 != 0 { throw Failure.unsupported("encrypted entry \(e.name)") }
        if e.uncompressedSize > limit { throw Failure.tooLarge(e.name) }
        let o = e.localHeaderOffset
        guard o >= 0, o + 30 <= data.count, u32(data, o) == 0x0403_4B50 else { throw Failure.corrupt("local header of \(e.name)") }
        let start = o + 30 + u16(data, o + 26) + u16(data, o + 28)
        guard e.compressedSize >= 0, start + e.compressedSize <= data.count else { throw Failure.corrupt("data of \(e.name)") }
        let raw = data.subdata(in: start..<(start + e.compressedSize))
        switch e.method {
        case 0:
            return raw
        case 8:
            if e.uncompressedSize == 0 { return Data() }
            var out = Data(count: e.uncompressedSize)
            let n = out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) -> Int in
                raw.withUnsafeBytes { (src: UnsafeRawBufferPointer) -> Int in
                    guard let d = dst.bindMemory(to: UInt8.self).baseAddress,
                          let s = src.bindMemory(to: UInt8.self).baseAddress else { return 0 }
                    return compression_decode_buffer(d, e.uncompressedSize, s, raw.count, nil, COMPRESSION_ZLIB)
                }
            }
            guard n == e.uncompressedSize else { throw Failure.corrupt("deflate data of \(e.name)") }
            return out
        default:
            throw Failure.unsupported("compression method \(e.method) (\(e.name))")
        }
    }

    /// The bytes of the entry with this name, or nil when there is none.
    public func contents(named name: String, limit: Int = Int.max) throws -> Data? {
        guard let e = entry(named: name) else { return nil }
        return try contents(of: e, limit: limit)
    }

    // ---------------------------------------------------------------- the central directory
    private static func centralDirectory(_ d: Data) throws -> [Entry] {
        // the end record is 22 bytes plus a comment of up to 65,535, so search back that far for its signature
        guard d.count >= 22 else { throw Failure.notAZip }
        var eocd = -1
        var i = d.count - 22
        let lowest = max(0, d.count - 22 - 65_535)
        while i >= lowest {
            if u32(d, i) == 0x0605_4B50 { eocd = i; break }
            i -= 1
        }
        guard eocd >= 0 else { throw Failure.notAZip }
        if u16(d, eocd + 4) != 0 || u16(d, eocd + 6) != 0 { throw Failure.unsupported("split archive") }
        var count = u16(d, eocd + 10)
        var size = u32(d, eocd + 12)
        var offset = u32(d, eocd + 16)
        if count == 0xFFFF || size == 0xFFFF_FFFF || offset == 0xFFFF_FFFF {
            // ZIP64: a locator just before the end record points at the ZIP64 end record
            let loc = eocd - 20
            guard loc >= 0, u32(d, loc) == 0x0706_4B50 else { throw Failure.corrupt("ZIP64 locator") }
            let rec = try int(u64(d, loc + 8))
            guard rec >= 0, rec + 56 <= d.count, u32(d, rec) == 0x0606_4B50 else { throw Failure.corrupt("ZIP64 end record") }
            count = try int(u64(d, rec + 32))
            size = try int(u64(d, rec + 40))
            offset = try int(u64(d, rec + 48))
        }
        guard offset >= 0, size >= 0, offset + size <= d.count else { throw Failure.corrupt("central directory bounds") }
        var out: [Entry] = []
        out.reserveCapacity(min(count, 100_000))
        var p = offset
        for _ in 0..<count {
            guard p + 46 <= d.count, u32(d, p) == 0x0201_4B50 else { throw Failure.corrupt("central directory entry") }
            let flags = u16(d, p + 8), method = u16(d, p + 10)
            var csize = u32(d, p + 20), usize = u32(d, p + 24)
            let nameLen = u16(d, p + 28), extraLen = u16(d, p + 30), commentLen = u16(d, p + 32)
            var local = u32(d, p + 42)
            guard p + 46 + nameLen + extraLen + commentLen <= d.count else { throw Failure.corrupt("central directory entry") }
            let nameBytes = d.subdata(in: (p + 46)..<(p + 46 + nameLen))
            // bit 11: UTF-8 names; older tools wrote code page 437, read here as Latin-1 when it is not valid UTF-8
            let name = String(data: nameBytes, encoding: .utf8) ?? String(data: nameBytes, encoding: .isoLatin1) ?? ""
            if usize == 0xFFFF_FFFF || csize == 0xFFFF_FFFF || local == 0xFFFF_FFFF {
                // the ZIP64 extra field (id 1) holds, in order, only the values that overflowed
                var x = p + 46 + nameLen
                let xEnd = x + extraLen
                while x + 4 <= xEnd {
                    let id = u16(d, x), len = u16(d, x + 2)
                    if id == 1 {
                        var f = x + 4
                        if usize == 0xFFFF_FFFF, f + 8 <= x + 4 + len { usize = try int(u64(d, f)); f += 8 }
                        if csize == 0xFFFF_FFFF, f + 8 <= x + 4 + len { csize = try int(u64(d, f)); f += 8 }
                        if local == 0xFFFF_FFFF, f + 8 <= x + 4 + len { local = try int(u64(d, f)) }
                        break
                    }
                    x += 4 + len
                }
            }
            out.append(Entry(name: flags & 0x800 != 0 ? String(decoding: nameBytes, as: UTF8.self) : name, method: method,
                             flags: flags, compressedSize: csize, uncompressedSize: usize, localHeaderOffset: local))
            p += 46 + nameLen + extraLen + commentLen
        }
        return out
    }

    private static func int(_ v: UInt64) throws -> Int {
        guard v <= UInt64(Int.max) else { throw Failure.corrupt("ZIP64 value out of range") }
        return Int(v)
    }
}

// little-endian reads at byte offsets of a 0-based Data
private func u16(_ d: Data, _ o: Int) -> Int { Int(d[o]) | Int(d[o + 1]) << 8 }
private func u32(_ d: Data, _ o: Int) -> Int { u16(d, o) | u16(d, o + 2) << 16 }
private func u64(_ d: Data, _ o: Int) -> UInt64 { UInt64(u32(d, o)) | UInt64(u32(d, o + 4)) << 32 }
