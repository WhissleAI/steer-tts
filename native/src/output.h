// Output level: the last stage between the vocoder and the caller. Same stage as steertts_edge/output.py (and the C
// library's st_output.c, the ESP32 firmware): config.json "output" {target_lufs, true_peak_db, knee_db, gain_db,
// voice_gain_db}. "auto": the voice's calibrated gain (neutral speech at target_lufs), a soft-knee tanh limiter
// (ceiling true_peak_db) and, for a whole clip, a 4x-oversampled true-peak guard; streamed chunks get the fixed gain +
// limiter only (no look-ahead, no added latency). "clip": a whole clip normalised to target_lufs. "off": raw output.
#pragma once
#include <map>
#include <string>
#include <vector>

#include "json.h"

namespace steertts {
namespace output {

enum Level { kOff = 0, kAuto = 1, kClip = 2 };
Level parse_level(const std::string& name);   // throws std::invalid_argument (the Python message)
void check_gain(float gain_db);                // NaN = calibrated; else [-6, 18] or throws

struct Stage {
  float target_lufs = -17.0f, true_peak_db = -1.0f, knee_db = -3.0f, gain_db = 0.0f;
  std::map<std::string, float> voice_gain_db;
  void load(const json::Value& cfg);
  float gain(const std::string& voice) const;    // empty / uncalibrated voice: the fallback gain
  void chunk(std::vector<float>& y, Level level, const std::string& voice, float gain_db) const;
  void clip(std::vector<float>& y, Level level, const std::string& voice, float gain_db, int sr) const;
};

void soft_limit(std::vector<float>& y, float knee_db, float ceiling_db);
double true_peak(const std::vector<float>& y);
double lufs(const std::vector<float>& y, int sr);

}  // namespace output
}  // namespace steertts
