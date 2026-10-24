// SteerTTS edge text front end (C++17, standard library only).
//
// Bit-exact port of the Python edge front end (pkg/steertts_edge: text.normalize / text.to_phonemes /
// markup.MarkupSynth.parse with its edge_front backends), verified against tests/front/golden.jsonl.
// Assets: native/assets/front (built by tools/build_native_assets.py from pkg/front).
//
// Symbol table (118 ids, == text.SYMBOLS):
//   0 <pad>, 1 <sp> (word boundary), 2..8 punctuation , . ? ! ; : -   (only , . ? ! - can actually occur),
//   9..77 ARPAbet phones (AA0 = 9 ... ZH = 77), 78..117 sound-event tokens ([laugh] = 78 ... [clock] = 104,
//   [reserved0..12] = 105..117; the acoustic model's embedding has all 118 rows).
// "Nothing to say": Python's synthesiser (ort.py mel()) returns no audio when no id >= SYM2ID["AA0"] (= 9) is
// present, e.g. "", "...", "!!" or text the G2P drops entirely. Event tokens are >= 9 too, so event-only input like
// "[laugh]" (ids 1 78 1) IS rendered. Use Frontend::has_speech() for exactly that test.
//
// Errors: inputs that make Python raise make these functions throw, with Python's message:
//   ValueError -> std::invalid_argument, OverflowError -> std::overflow_error, IndexError -> std::out_of_range.
//   (e.g. a 103+ digit number, `$,5`, `[pause 1.2.3]`, `<style prompt="...">`.) Throwing is part of parity.
//
// Thread safety: a Frontend is immutable after construction except for an internal G2P cache guarded by a mutex;
// to_phonemes / normalize / parse_markup may be called concurrently.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace steertts {

struct Phonemes {
  std::vector<int64_t> ids;       // symbol ids; starts with <sp>, ends with <sp> after the last word (if any)
  std::vector<int32_t> word_idx;  // per id: index of the word the phone belongs to, -1 for <sp>/punct/events
};

// Per-word prosody shift in markup.PROSODY user units. The acoustic code (ort.py mel()) applies, for every token
// whose word_idx == key:  pitch += float32(pitch / 3.0)   (PITCH_SCALE, compute in double, then round)
//                         energy += float32(energy / 10.0)
//                         dur *= float32(dur)
// pitch/energy here are exact (sums of 2.5/3/-3 and 4/6/-6); dur is the product of the active factors
// (emph 1.2, slow 1.4, fast 0.75) multiplied in PROSODY order, rounded to float. (Python multiplies them in
// frozenset iteration order, which varies with PYTHONHASHSEED and can change the last bit of the double; the
// float32 value the model sees is the same.)
struct WordShift {
  float pitch = 0, energy = 0, dur = 1;
};

struct Segment {
  std::string text;  // whitespace-collapsed segment text (feed to Frontend::to_phonemes); empty for pauses
  std::string tag;   // style tag ("none" = no style)
  std::vector<std::pair<std::string, float>> ctrl;  // controls, sorted by key (Python: dict(sorted(items)))
  std::map<int, WordShift> word_shift;  // key = word index in to_phonemes(text).word_idx (same numbering)
  int pause_ms = 0;  // > 0: silence of that length instead of speech (text empty)
  std::string event;  // always empty for well-formed input (kept for parity with Python's Segment)

  // Python quirk replicated for parity: a text piece that is exactly "PAUSE" / "EVENT" (e.g. "Stop. PAUSE")
  // becomes a Segment whose pause_ms / event field holds the *style tuple* instead of a number / name. In C++ that
  // is pause_ms == kPauseIsStyleTuple, or event == kEventIsStyleTuple. Python's stream() would crash on the first
  // (np.zeros(int(tuple))) and render an empty-text segment for the second; an engine should skip both.
  static constexpr int kPauseIsStyleTuple = -1;
  static constexpr const char* kEventIsStyleTuple = "<style-tuple>";
};

class Frontend {
 public:
  explicit Frontend(const std::string& asset_dir);
  ~Frontend();
  Frontend(const Frontend&) = delete;
  Frontend& operator=(const Frontend&) = delete;

  Phonemes to_phonemes(const std::string& utf8) const;   // == text.to_phonemes(...)[:2]
  std::string normalize(const std::string& utf8) const;  // == text.normalize
  const std::vector<std::string>& symbols() const;       // == text.SYMBOLS
  int symbol_id(const std::string& s) const;             // == SYM2ID[s], -1 if absent
  bool has_speech(const Phonemes& p) const;              // any id >= SYM2ID["AA0"] (events count)

  static constexpr int kPad = 0, kSp = 1, kFirstPhone = 9, kNumSymbols = 118;

  struct Impl;
  const Impl& impl() const { return *impl_; }

 private:
  std::unique_ptr<Impl> impl_;
};

// == MarkupSynth(tts).parse(text, prompt=None, tag=default_tag, ctrl=default_ctrl) where tts.tag2i has the keys
// known_tags (config.json "tags"). default_tag "" == None. default_ctrl: insertion-ordered pairs (later duplicates
// win, like a dict). Segments are returned in playback order; render each non-pause segment with its tag / ctrl /
// word_shift, insert pause_ms of silence for pause segments.
//
// Streaming rule of MarkupSynth.stream() (to be done by the engine): pause segment -> int(pause_ms * 24000 / 1000)
// zero samples; speech segment -> rendered audio (empty if the segment has no phones, see has_speech); if it has
// more than 2*240 samples, multiply the first 240 samples by linspace(0, 1, 240) (float32: k/239) and the last 240
// by the reversed ramp (10 ms fades at 24 kHz); concatenate segments in order with nothing in between.
std::vector<Segment> parse_markup(const Frontend& fe, const std::string& text, const std::string& default_tag,
                                  const std::vector<std::pair<std::string, float>>& default_ctrl,
                                  const std::vector<std::string>& known_tags);

}  // namespace steertts
