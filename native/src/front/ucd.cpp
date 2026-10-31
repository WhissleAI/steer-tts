#include "ucd.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace steertts {
namespace detail {

u32str utf8_decode(const std::string& s) {
  u32str out;
  out.reserve(s.size());
  size_t i = 0, n = s.size();
  auto cont = [&](size_t k) { return k < n && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80; };
  while (i < n) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    char32_t cp = 0xFFFD;
    size_t len = 1;
    if (c < 0x80) {
      cp = c;
    } else if ((c & 0xE0) == 0xC0 && cont(i + 1)) {
      cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F);
      len = 2;
      if (cp < 0x80) cp = 0xFFFD;
    } else if ((c & 0xF0) == 0xE0 && cont(i + 1) && cont(i + 2)) {
      cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
      len = 3;
      if (cp < 0x800 || (cp >= 0xD800 && cp < 0xE000)) cp = 0xFFFD;
    } else if ((c & 0xF8) == 0xF0 && cont(i + 1) && cont(i + 2) && cont(i + 3)) {
      cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
      len = 4;
      if (cp < 0x10000 || cp > 0x10FFFF) cp = 0xFFFD;
    }
    out.push_back(cp);
    i += len;
  }
  return out;
}

void utf8_append(std::string& out, char32_t c) {
  if (c < 0x80) {
    out.push_back(static_cast<char>(c));
  } else if (c < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (c >> 6)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  } else if (c < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (c >> 12)));
    out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (c >> 18)));
    out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  }
}

std::string utf8_encode(const u32str& s) {
  std::string out;
  out.reserve(s.size());
  for (char32_t c : s) utf8_append(out, c);
  return out;
}

namespace {
struct Reader {
  std::string b;
  size_t o = 0;
  uint32_t u32() {
    if (o + 4 > b.size()) throw std::runtime_error("steertts front: truncated asset");
    uint32_t v;
    std::memcpy(&v, b.data() + o, 4);  // assets are little endian; so are all supported targets
    o += 4;
    return v;
  }
  uint8_t u8() {
    if (o + 1 > b.size()) throw std::runtime_error("steertts front: truncated asset");
    return static_cast<uint8_t>(b[o++]);
  }
  std::string bytes(size_t n) {
    if (o + n > b.size()) throw std::runtime_error("steertts front: truncated asset");
    std::string r = b.substr(o, n);
    o += n;
    return r;
  }
};
}  // namespace

void Ucd::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("steertts front: cannot open " + path);
  Reader r{std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>())};
  if (r.bytes(4) != "STUC" || r.u32() != 1) throw std::runtime_error("steertts front: bad unicode.bin");
  for (uint32_t n = r.u32(), k = 0; k < n; ++k) {
    char32_t lo = r.u32(), hi = r.u32();
    word_.push_back({lo, hi});
  }
  for (uint32_t n = r.u32(), k = 0; k < n; ++k) {
    char32_t c = r.u32();
    if (c >= 128) space_.push_back(c);
  }
  for (uint32_t n = r.u32(), k = 0; k < n; ++k) {
    DR d;
    d.lo = r.u32();
    d.hi = r.u32();
    d.v0 = r.u32();
    digit_.push_back(d);
  }
  for (uint32_t n = r.u32(), k = 0; k < n; ++k) {
    char32_t c = r.u32();
    std::string s = r.bytes(r.u8());
    if (c < 128)
      ascii_g2p_[c] = s;
    else
      g2p_[c] = s;
  }
  for (uint32_t n = r.u32(), k = 0; k < n; ++k) {
    char32_t c = r.u32(), a = r.u32();
    fold_.push_back({c, a});
  }
  for (uint32_t n = r.u32(), k = 0; k < n; ++k) {
    char32_t c = r.u32();
    lower_[c] = utf8_decode(r.bytes(r.u8()));
  }
}

bool Ucd::in_ranges(const std::vector<std::pair<char32_t, char32_t>>& r, char32_t c) {
  auto it = std::upper_bound(r.begin(), r.end(), c,
                             [](char32_t v, const std::pair<char32_t, char32_t>& p) { return v < p.first; });
  if (it == r.begin()) return false;
  --it;
  return c <= it->second;
}

int Ucd::digit(char32_t c) const {
  if (c >= '0' && c <= '9') return static_cast<int>(c - '0');
  if (c < 128) return -1;
  for (auto& d : digit_)
    if (c >= d.lo && c <= d.hi) return static_cast<int>(d.v0 + (c - d.lo));
  return -1;
}

u32str Ucd::lower_small(const u32str& s) const {
  u32str out;
  for (char32_t c : s) {
    if (c >= 'A' && c <= 'Z') {
      out.push_back(c + 32);
    } else if (c < 128) {
      out.push_back(c);
    } else {
      auto it = lower_.find(c);
      if (it != lower_.end())
        out += it->second;
      else
        out.push_back(c);  // callers only pass ascii / fold_ code points
    }
  }
  return out;
}

const std::string& Ucd::g2p_clean(char32_t c) const {
  if (c < 128) return ascii_g2p_[c];
  auto it = g2p_.find(c);
  return it == g2p_.end() ? empty_ : it->second;
}

}  // namespace detail
}  // namespace steertts
