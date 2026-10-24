// Minimal JSON reader for the package metadata (config.json, voices.json, safetensors headers).
// Objects keep insertion order. Numbers are doubles. Throws std::runtime_error on malformed input.
#pragma once
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace steertts {
namespace json {

struct Value {
  enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;

  bool has(const std::string& k) const {
    for (auto& kv : obj) if (kv.first == k) return true;
    return false;
  }
  const Value& operator[](const std::string& k) const {
    for (auto& kv : obj) if (kv.first == k) return kv.second;
    throw std::runtime_error("json: missing key '" + k + "'");
  }
  const Value& operator[](size_t i) const { return arr.at(i); }
  size_t size() const { return kind == Arr ? arr.size() : obj.size(); }
};

class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}
  Value parse() {
    Value v = value();
    ws();
    if (i_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  const std::string& s_;
  size_t i_ = 0;

  [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json: ") + m + " at " + std::to_string(i_)); }
  void ws() { while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_; }
  char peek() { ws(); if (i_ >= s_.size()) fail("unexpected end"); return s_[i_]; }
  bool lit(const char* w) {
    size_t n = std::char_traits<char>::length(w);
    if (s_.compare(i_, n, w) == 0) { i_ += n; return true; }
    return false;
  }
  static void utf8(std::string& o, unsigned cp) {
    if (cp < 0x80) o += char(cp);
    else if (cp < 0x800) { o += char(0xC0 | (cp >> 6)); o += char(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { o += char(0xE0 | (cp >> 12)); o += char(0x80 | ((cp >> 6) & 63)); o += char(0x80 | (cp & 63)); }
    else { o += char(0xF0 | (cp >> 18)); o += char(0x80 | ((cp >> 12) & 63)); o += char(0x80 | ((cp >> 6) & 63)); o += char(0x80 | (cp & 63)); }
  }
  unsigned hex4() {
    if (i_ + 4 > s_.size()) fail("bad \\u escape");
    unsigned v = std::stoul(s_.substr(i_, 4), nullptr, 16);
    i_ += 4;
    return v;
  }
  std::string string() {
    if (s_[i_] != '"') fail("expected string");
    ++i_;
    std::string o;
    while (true) {
      if (i_ >= s_.size()) fail("unterminated string");
      char c = s_[i_++];
      if (c == '"') break;
      if (c != '\\') { o += c; continue; }
      char e = s_[i_++];
      switch (e) {
        case '"': o += '"'; break;
        case '\\': o += '\\'; break;
        case '/': o += '/'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'n': o += '\n'; break;
        case 'r': o += '\r'; break;
        case 't': o += '\t'; break;
        case 'u': {
          unsigned cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && s_.compare(i_, 2, "\\u") == 0) {
            i_ += 2;
            unsigned lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          utf8(o, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
    return o;
  }
  Value value() {
    Value v;
    char c = peek();
    if (c == '{') {
      v.kind = Value::Obj;
      ++i_;
      if (peek() == '}') { ++i_; return v; }
      while (true) {
        ws();
        std::string k = string();
        if (peek() != ':') fail("expected ':'");
        ++i_;
        v.obj.emplace_back(std::move(k), value());
        char d = peek();
        ++i_;
        if (d == '}') break;
        if (d != ',') fail("expected ',' or '}'");
      }
    } else if (c == '[') {
      v.kind = Value::Arr;
      ++i_;
      if (peek() == ']') { ++i_; return v; }
      while (true) {
        v.arr.push_back(value());
        char d = peek();
        ++i_;
        if (d == ']') break;
        if (d != ',') fail("expected ',' or ']'");
      }
    } else if (c == '"') {
      v.kind = Value::Str;
      v.str = string();
    } else if (lit("true")) { v.kind = Value::Bool; v.b = true; }
    else if (lit("false")) { v.kind = Value::Bool; }
    else if (lit("null")) { v.kind = Value::Null; }
    else {
      const char* p = s_.c_str() + i_;
      char* end = nullptr;
      v.num = std::strtod(p, &end);
      if (end == p) fail("bad value");
      v.kind = Value::Num;
      i_ += size_t(end - p);
    }
    return v;
  }
};

inline Value parse(const std::string& s) { return Parser(s).parse(); }

}  // namespace json
}  // namespace steertts
