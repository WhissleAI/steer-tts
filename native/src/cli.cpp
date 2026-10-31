// steertts-edge (native): the same command line as the Python package's `steertts-edge`.
//
//   steertts-edge "Hello there." -o hello.wav
//   steertts-edge "I can't believe it worked." --voice voice_0010 --tag happiness --ctrl rate=1,f0_iqr=1.5 -o happy.wav
//   steertts-edge "Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow>" --markup -o markup.wav
//   steertts-edge "First sentence. Second one." --stream -o - | aplay -f S16_LE -r 24000 -c 1     # raw PCM as rendered
//   steertts-edge --list voices | tags | controls
//   steertts-edge --bench                                                                        # speed + memory
//
// The package directory is --model, else $STEERTTS_EDGE_MODEL, else the current directory.
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include <onnxruntime_c_api.h>

#include "steertts/steertts.h"

using steertts::Engine;
using steertts::Request;

namespace {

const char* kUsage =
    "usage: steertts-edge [TEXT | -] [-o OUT.wav | -o -] [--model DIR] [--voice NAME] [--tag TAG]\n"
    "                     [--ctrl name=z,...] [--rate Z] [--pace X] [--guidance S] [--markup] [--stream]\n"
    "                     [--threads N] [--list voices|tags|controls] [--bench]\n"
    "\n"
    "  TEXT          text to speak ('-' reads stdin)\n"
    "  -o OUT        output WAV (16-bit, 24 kHz; default out.wav). '-' writes raw 16-bit PCM to stdout\n"
    "  --markup      parse inline markup (*emphasis*, [pause 300ms], <slow>, <whisper> ...)\n"
    "  --stream      render sentence by sentence (implies --markup), writing each chunk as soon as it is ready\n"
    "  --guidance S  style guidance scale (1.0 = off)\n"
    "  --pace X      global duration divisor (1.2 = 20% faster)\n"
    "  --threads N   ONNX Runtime threads (default 1)\n"
    "  --bench       render 20 fixed sentences, print real-time factor, latency and peak memory\n"
    "  --bench-json  the same, as one JSON line (edge/tools/bench_speed.py --runtime native)\n";

double peak_rss_mb() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss / 1024.0;
}

double rss_mb() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmRSS:", 0) == 0) return std::atof(line.c_str() + 6) / 1024.0;
  return 0;
}

std::vector<std::pair<std::string, float>> parse_ctrl(const std::string& s) {
  std::vector<std::pair<std::string, float>> out;
  std::stringstream ss(s);
  std::string kv;
  while (std::getline(ss, kv, ',')) {
    if (kv.empty()) continue;
    size_t eq = kv.find('=');
    if (eq == std::string::npos) throw std::invalid_argument("--ctrl expects name=value pairs, got '" + kv + "'");
    out.emplace_back(kv.substr(0, eq), std::stof(kv.substr(eq + 1)));
  }
  return out;
}

void write_pcm(const std::vector<float>& y) {
  std::vector<int16_t> pcm(y.size());
  for (size_t i = 0; i < y.size(); ++i) {
    float v = std::min(1.0f, std::max(-1.0f, y[i])) * 32767.0f;
    pcm[i] = int16_t(std::nearbyint(v));
  }
  std::fwrite(pcm.data(), 2, pcm.size(), stdout);
  std::fflush(stdout);
}

// Same sentences and probes as edge/tools/bench_speed.py, so the numbers compare across runtimes.
const char* kBenchTexts[] = {
    "The weather was calm when we left the harbour this morning.",
    "Please remember to bring the documents to the meeting on Thursday.",
    "I never thought a small garden could grow so many tomatoes.",
    "She opened the letter slowly and read it twice before speaking.",
    "Our train was delayed by forty minutes because of the snow.",
    "He laughed, shook his head, and walked back into the kitchen.",
    "Thanks for calling. How can I help you today?",
    "The museum opens at nine and closes at half past five.",
    "If you turn left at the bakery, you will see the station.",
    "Nobody expected the quiet student to win the debate.",
    "Could you send me the report before lunch?",
    "The river had risen overnight, and the old bridge was closed.",
    "We should probably leave now if we want to catch the last bus.",
    "Her grandmother kept every letter in a wooden box under the bed.",
    "That was the best soup I have had in years.",
    "The engineers tested the new design for three weeks before approving it.",
    "Wait, did you hear that noise outside?",
    "A small dog followed us all the way home from the park.",
    "The price of coffee went up again this month.",
    "When the lights came back on, everyone started clapping."};
const char* kShort = "Hello, how are you?";

double since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int bench(Engine& tts, double load_s, int threads, bool as_json) {
  Request r;
  for (int i = 0; i < 2; ++i) { r.text = kBenchTexts[i]; tts.synthesize(r); }   // warm-up
  std::vector<double> ttfa, first, per;
  for (int i = 0; i < 5; ++i) {
    r.text = kShort;
    auto t0 = std::chrono::steady_clock::now();
    tts.synthesize(r);
    ttfa.push_back(since(t0));
  }
  for (int i = 0; i < 5; ++i) {
    Request s;
    s.text = std::string(kShort) + " And a second sentence follows.";
    auto t0 = std::chrono::steady_clock::now();
    tts.stream(s, [&](const steertts::Chunk&) { first.push_back(since(t0)); return false; });
  }
  double synth = 0, audio = 0;
  for (const char* t : kBenchTexts) {
    r.text = t;
    auto t0 = std::chrono::steady_clock::now();
    auto y = tts.synthesize(r);
    double d = since(t0), a = double(y.size()) / steertts::kSampleRate;
    synth += d;
    audio += a;
    per.push_back(d / a);
  }
  std::sort(ttfa.begin(), ttfa.end());
  std::sort(first.begin(), first.end());
  std::sort(per.begin(), per.end());
  if (as_json)
    std::printf("{\"threads\": %d, \"load_s\": %.3f, \"rtf\": %.4f, \"rtf_median\": %.4f, \"audio_s\": %.2f, "
                "\"synth_s\": %.3f, \"ttfa_short_s\": %.3f, \"ttfa_stream_first_chunk_s\": %.3f, \"peak_rss_mb\": %.1f, "
                "\"runtime\": \"native\", \"version\": \"onnxruntime %s\"}\n",
                threads, load_s, synth / audio, per[per.size() / 2], audio, synth, ttfa[2], first[2], peak_rss_mb(),
                OrtGetApiBase()->GetVersionString());
  else
    std::printf("threads %d: load %.2f s, RTF %.4f (median %.4f) over %.1f s of audio, short sentence %.0f ms, "
                "first streamed chunk %.0f ms, peak RSS %.0f MB\n",
                threads, load_s, synth / audio, per[per.size() / 2], audio, ttfa[2] * 1000, first[2] * 1000, peak_rss_mb());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string text, out = "out.wav", model, list, ctrl_s;
  Request r;
  steertts::Options opt;
  bool stream = false, do_bench = false, bench_json = false, have_text = false;
  float rate = 0;
  bool have_rate = false;
  try {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "-h" || a == "--help") { std::fputs(kUsage, stdout); return 0; }
      else if (a == "-o" || a == "--out") out = next();
      else if (a == "--model") model = next();
      else if (a == "--voice") r.voice = next();
      else if (a == "--tag") r.tag = next();
      else if (a == "--ctrl") ctrl_s = next();
      else if (a == "--rate") { rate = std::stof(next()); have_rate = true; }
      else if (a == "--pace") r.pace = std::stof(next());
      else if (a == "--guidance") r.guidance = std::stof(next());
      else if (a == "--markup") r.markup = true;
      else if (a == "--stream") stream = true;
      else if (a == "--threads") opt.threads = std::stoi(next());
      else if (a == "--list") list = next();
      else if (a == "--bench") do_bench = true;
      else if (a == "--bench-json") { do_bench = true; bench_json = true; }
      else if (!a.empty() && a[0] == '-' && a != "-") throw std::invalid_argument("unknown option " + a);
      else { text = a; have_text = true; }
    }
    if (model.empty()) model = std::getenv("STEERTTS_EDGE_MODEL") ? std::getenv("STEERTTS_EDGE_MODEL") : ".";
    auto t0 = std::chrono::steady_clock::now();
    Engine tts(model, opt);
    double load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!list.empty()) {
      if (list == "voices")
        for (auto& v : tts.voices()) std::printf("%s\t%s\t%s pitch\t%s\t%.1f Hz\n", v.name.c_str(), v.gender.c_str(),
                                                 v.pitch_register.c_str(), v.pace.c_str(), v.f0_median_hz);
      else if (list == "tags") for (auto& t : tts.tags()) std::printf("%s\n", t.c_str());
      else if (list == "controls") for (auto& c : tts.controls()) std::printf("%s\n", c.c_str());
      else throw std::invalid_argument("--list takes voices, tags or controls");
      return 0;
    }
    if (do_bench) return bench(tts, load, opt.threads, bench_json);
    if (!have_text) { std::fputs(kUsage, stderr); return 2; }
    if (text == "-") text.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
    r.text = text;
    r.ctrl = parse_ctrl(ctrl_s);
    if (have_rate) r.ctrl.emplace_back("rate", rate);
    auto t1 = std::chrono::steady_clock::now();
    std::vector<float> wav;
    double first_ms = -1;
    if (stream) {
      tts.stream(r, [&](const steertts::Chunk& c) {
        if (first_ms < 0) first_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        if (out == "-") write_pcm(c.audio);
        else wav.insert(wav.end(), c.audio.begin(), c.audio.end());
        return true;
      });
    } else {
      wav = tts.synthesize(r);
      if (out == "-") write_pcm(wav);
    }
    double synth = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    if (out != "-") Engine::save_wav(wav, out);
    double dur = double(wav.size()) / steertts::kSampleRate;
    if (out != "-") {
      std::fprintf(stderr, "%s: %.2f s audio, load %.2f s, synth %.2f s", out.c_str(), dur, load, synth);
      if (dur > 0) std::fprintf(stderr, " (RTF %.3f)", synth / dur);
      if (first_ms >= 0) std::fprintf(stderr, ", first chunk %.0f ms", first_ms);
      std::fprintf(stderr, ", peak RSS %.0f MB\n", peak_rss_mb());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "steertts-edge: %s\n", e.what());
    return 2;
  }
  return 0;
}
