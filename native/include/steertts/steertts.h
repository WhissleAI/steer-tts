// SteerTTS edge, native runtime: text -> 24 kHz mono float32 audio, on ONNX Runtime.
//
//   steertts::Engine tts("path/to/steer-tts-edge");          // package dir (config.json, voices.*, onnx/, native/)
//   steertts::Request r;
//   r.text = "I can't believe it worked.";
//   r.tag = "happiness";
//   r.ctrl = {{"rate", 1.0f}};
//   std::vector<float> wav = tts.synthesize(r);
//   tts.stream(r, [](const steertts::Chunk& c) { play(c.audio); return true; });   // sentence by sentence
//
// Same rendering path as the Python runtimes (steertts_edge.EdgeTTS / OnnxTTS): text front end -> encoder graph ->
// duration rounding + per-word shifts + length regulation -> decoder graph -> vocoder graph -> iSTFT.
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "steertts/front.h"

namespace steertts {

constexpr int kSampleRate = 24000;

struct Request {
  std::string text;
  std::string voice;                                  // empty: the package default voice
  std::string tag = "none";                           // style tag (Engine::tags())
  std::vector<std::pair<std::string, float>> ctrl;    // control z-values in [-5, 5] (Engine::controls())
  float pace = 1.0f;                                  // global tempo divisor (>1 faster), does not change style
  float guidance = -1.0f;                             // steering scale; < 0: the engine default (1.0)
  std::map<int, WordShift> word_shift;                // per-word edits (semitones / dB / duration factor); plain text only
  bool markup = false;                                // parse inline markup (*emph*, [pause 400ms], <slow>, <whisper> ...)
  std::vector<float> spk_vec;                         // optional 512-d host-computed speaker vector (clone); overrides voice
};

struct Chunk {
  std::vector<float> audio;   // this segment's samples (10 ms fades at both ends, as the Python stream())
  Segment segment;            // what was rendered (text, tag, ctrl, pause)
  double synth_ms = 0;        // wall time to render it
};

struct VoiceInfo {
  std::string name, gender, pitch_register, pace;
  double f0_median_hz = 0;
};

struct Options {
  int threads = 1;            // ONNX Runtime intra-op threads
  float guidance = 1.0f;      // default steering scale
};

class Engine {
 public:
  explicit Engine(const std::string& package_dir, Options opt = {});
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Whole request in one pass (markup=false), or every markup segment concatenated (markup=true).
  std::vector<float> synthesize(const Request& r);
  // Markup segments / sentences one by one (always parses markup, as Python's stream()); return false to stop.
  void stream(const Request& r, const std::function<bool(const Chunk&)>& on_chunk);
  // The mel spectrogram only ([100, T] row-major in out, returns T; 0 = nothing to say).
  int mel(const Request& r, std::vector<float>& out);

  const std::vector<std::string>& tags() const;
  const std::vector<std::string>& controls() const;
  const std::vector<VoiceInfo>& voices() const;
  const std::string& default_voice() const;
  const Frontend& frontend() const;

  static void save_wav(const std::vector<float>& audio, const std::string& path);   // 16-bit PCM, 24 kHz

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace steertts
