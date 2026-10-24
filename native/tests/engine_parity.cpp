// engine_parity <package_dir> <golden_dir>: renders every request in <golden_dir>/cases.jsonl (written by
// tools/make_engine_golden.py from the Python ONNX runtime) and compares with the reference audio.
// Pass: same length and SNR >= 40 dB for every case (float op order differs, so not bit-exact; the Python ONNX and
// PyTorch runtimes agree with each other at the same level).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "../src/json.h"
#include "steertts/steertts.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: engine_parity <package_dir> <golden_dir>\n");
    return 2;
  }
  steertts::Engine tts(argv[1]);
  std::ifstream f(std::string(argv[2]) + "/cases.jsonl");
  std::string line;
  int n = 0, fail = 0;
  double worst = 1e9;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    auto c = steertts::json::parse(line);
    steertts::Request r;
    r.text = c["text"].str;
    if (c.has("voice")) r.voice = c["voice"].str;
    if (c.has("tag")) r.tag = c["tag"].str;
    if (c.has("ctrl")) for (auto& kv : c["ctrl"].obj) r.ctrl.emplace_back(kv.first, float(kv.second.num));
    if (c.has("pace")) r.pace = float(c["pace"].num);
    if (c.has("guidance")) r.guidance = float(c["guidance"].num);
    if (c.has("markup")) r.markup = c["markup"].b;
    if (c.has("word_shift"))
      for (auto& kv : c["word_shift"].obj) {
        steertts::WordShift ws;
        if (kv.second.has("pitch")) ws.pitch = float(kv.second["pitch"].num);
        if (kv.second.has("energy")) ws.energy = float(kv.second["energy"].num);
        if (kv.second.has("dur")) ws.dur = float(kv.second["dur"].num);
        r.word_shift[std::stoi(kv.first)] = ws;
      }
    std::vector<float> got = tts.synthesize(r);
    std::ifstream rf(std::string(argv[2]) + "/" + c["out"].str, std::ios::binary);
    std::vector<float> ref(size_t(c["n"].num));
    rf.read(reinterpret_cast<char*>(ref.data()), std::streamsize(ref.size() * 4));
    ++n;
    if (got.size() != ref.size()) {
      std::printf("FAIL length %zu vs %zu  %.50s\n", got.size(), ref.size(), r.text.c_str());
      ++fail;
      continue;
    }
    double sig = 0, err = 0, mx = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      sig += double(ref[i]) * ref[i];
      double d = double(got[i]) - ref[i];
      err += d * d;
      mx = std::max(mx, std::fabs(d));
    }
    double snr = ref.empty() ? 999 : (err == 0 ? 999 : 10 * std::log10(sig / err));
    worst = std::min(worst, snr);
    bool ok = snr >= 40;
    fail += !ok;
    std::printf("%s len %7zu  maxabs %.2e  SNR %6.1f dB  %.50s\n", ok ? "ok  " : "FAIL", ref.size(), mx, snr, r.text.c_str());
  }
  std::printf("%d cases, %d failed, worst SNR %.1f dB\n", n, fail, worst);
  return fail ? 1 : 0;
}
