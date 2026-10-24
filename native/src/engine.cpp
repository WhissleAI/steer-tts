// SteerTTS edge native engine (see include/steertts/steertts.h). Mirrors steertts_edge/ort.py (OnnxTTS) line by line;
// the numbered comments point at the matching step of model.infer in the Python package.
#include "steertts/steertts.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "dsp.h"
#include "json.h"

namespace steertts {

namespace {

const std::vector<std::string> kControls = {"rate", "loud", "f0_med", "f0_iqr", "f0_end", "pause",
                                            "tilt", "voiced", "flat", "centroid", "cpps", "h1h2"};
constexpr float kCtrlLo = -5.0f, kCtrlHi = 5.0f;
constexpr float kPitchScale = 3.0f;   // word_shift pitch is given in semitones, the model predicts st / 3
constexpr int kChunk = 64;            // graphs take sequences zero-padded to a multiple of the sequence-mixing chunk

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

float half_to_float(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
  if (exp == 0) {
    if (man == 0) bits = sign;
    else {  // subnormal
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) { man <<= 1; --exp; }
      man &= 0x3FF;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) bits = sign | 0x7F800000 | (man << 13);
  else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// safetensors: u64 header length, JSON header {name: {dtype, shape, data_offsets}}, raw little-endian data.
std::map<std::string, std::vector<float>> load_safetensors(const std::string& path) {
  std::string raw = read_file(path);
  if (raw.size() < 8) throw std::runtime_error(path + ": not a safetensors file");
  uint64_t n;
  std::memcpy(&n, raw.data(), 8);
  json::Value hdr = json::parse(raw.substr(8, n));
  const char* base = raw.data() + 8 + n;
  std::map<std::string, std::vector<float>> out;
  for (auto& kv : hdr.obj) {
    if (kv.first == "__metadata__") continue;
    const json::Value& t = kv.second;
    size_t a = size_t(t["data_offsets"][0].num), b = size_t(t["data_offsets"][1].num);
    std::vector<float> v;
    if (t["dtype"].str == "F32") {
      v.resize((b - a) / 4);
      std::memcpy(v.data(), base + a, b - a);
    } else if (t["dtype"].str == "F16") {
      v.resize((b - a) / 2);
      for (size_t i = 0; i < v.size(); ++i) {
        uint16_t h;
        std::memcpy(&h, base + a + 2 * i, 2);
        v[i] = half_to_float(h);
      }
    } else {
      throw std::runtime_error(path + ": unsupported dtype " + t["dtype"].str);
    }
    out[kv.first] = std::move(v);
  }
  return out;
}

size_t padded(size_t n) { return (n + kChunk - 1) / kChunk * kChunk; }

Ort::Env& env() {
  static Ort::Env e(ORT_LOGGING_LEVEL_WARNING, "steertts");
  return e;
}

}  // namespace

struct Engine::Impl {
  Options opt;
  std::string dir;
  std::vector<std::string> tags;
  std::map<std::string, int> tag2i;
  std::vector<VoiceInfo> voices;
  std::map<std::string, int> voice2i;
  std::string default_voice;
  std::vector<float> slots;            // [V, 64] voice conditioning slots
  std::vector<float> spk_w, spk_b;     // spk_proj [64, 512], [64]
  int n_sym = 0, aa0 = 0;
  std::unique_ptr<Frontend> front;
  Ort::SessionOptions so;
  std::unique_ptr<Ort::Session> enc, dec, voc;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  void check(const Request& r, bool markup) const {
    if (markup && !r.word_shift.empty())
      throw std::invalid_argument("word_shift does not combine with markup / stream(); use inline markup "
                                  "(*word*, <slow>, <high> ...) there");
    if (!tag2i.count(r.tag)) {
      std::string known;
      for (auto& t : tags) known += (known.empty() ? "" : ", ") + t;
      throw std::invalid_argument("unknown tag '" + r.tag + "'; known tags: " + known);
    }
    if (r.spk_vec.empty() && !r.voice.empty() && !voice2i.count(r.voice))
      throw std::invalid_argument("unknown voice '" + r.voice + "'; this edge build ships " +
                                  std::to_string(voices.size()) + " voices");
    if (!r.spk_vec.empty() && r.spk_vec.size() != spk_w.size() / 64)
      throw std::invalid_argument("spk_vec must have " + std::to_string(spk_w.size() / 64) + " values");
  }

  void control_arrays(const std::vector<std::pair<std::string, float>>& ctrl, std::array<float, 12>& c,
                      std::array<float, 12>& m) const {
    c.fill(0);
    m.fill(0);
    for (auto& kv : ctrl) {
      auto it = std::find(kControls.begin(), kControls.end(), kv.first);
      if (it == kControls.end()) throw std::invalid_argument("unknown control '" + kv.first + "'");
      if (!(kv.second >= kCtrlLo && kv.second <= kCtrlHi))
        throw std::invalid_argument("ctrl." + kv.first + "=" + std::to_string(kv.second) + " is outside (-5.0, 5.0)");
      size_t i = size_t(it - kControls.begin());
      c[i] = std::isfinite(kv.second) ? kv.second : 0.0f;   // nan_to_num
      m[i] = 1.0f;
    }
  }

  struct EncOut {
    std::vector<float> g, h, logdur, pitch, energy;   // h [Np,192]; the others [Np] (zero beyond the real length)
  };

  EncOut run_enc(const std::vector<int64_t>& ids_padded, int64_t n, const float* spk, int64_t tag,
                 const std::array<float, 12>& c, const std::array<float, 12>& m) {
    int64_t Np = int64_t(ids_padded.size());
    std::array<int64_t, 2> s_ids{1, Np}, s_spk{1, 64}, s_c{1, 12};
    std::array<int64_t, 1> s1{1};
    std::vector<Ort::Value> in;
    in.push_back(Ort::Value::CreateTensor<int64_t>(mem, const_cast<int64_t*>(ids_padded.data()), ids_padded.size(), s_ids.data(), 2));
    in.push_back(Ort::Value::CreateTensor<int64_t>(mem, &n, 1, s1.data(), 1));
    in.push_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(spk), 64, s_spk.data(), 2));
    in.push_back(Ort::Value::CreateTensor<int64_t>(mem, &tag, 1, s1.data(), 1));
    in.push_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(c.data()), 12, s_c.data(), 2));
    in.push_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(m.data()), 12, s_c.data(), 2));
    const char* in_names[] = {"ids", "n", "spk", "tag", "ctrl", "cmask"};
    const char* out_names[] = {"g", "h", "logdur", "pitch", "energy"};
    auto out = enc->Run(Ort::RunOptions{nullptr}, in_names, in.data(), in.size(), out_names, 5);
    auto vec = [](Ort::Value& v) {
      size_t k = v.GetTensorTypeAndShapeInfo().GetElementCount();
      const float* p = v.GetTensorData<float>();
      return std::vector<float>(p, p + k);
    };
    return EncOut{vec(out[0]), vec(out[1]), vec(out[2]), vec(out[3]), vec(out[4])};
  }

  // -> mel [100, T] row-major (only the T real frames)
  std::vector<float> run_dec(const std::vector<float>& h, const std::vector<float>& pitch, const std::vector<float>& energy,
                             const std::vector<int64_t>& tok_padded, int64_t T, const std::vector<float>& g) {
    int64_t Np = int64_t(pitch.size()), Tp = int64_t(tok_padded.size());
    std::array<int64_t, 3> s_h{1, Np, 192};
    std::array<int64_t, 2> s_p{1, Np}, s_t{1, Tp}, s_g{1, 128};
    std::array<int64_t, 1> s1{1};
    std::vector<Ort::Value> in;
    in.push_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(h.data()), h.size(), s_h.data(), 3));
    in.push_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(pitch.data()), pitch.size(), s_p.data(), 2));
    in.push_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(energy.data()), energy.size(), s_p.data(), 2));
    in.push_back(Ort::Value::CreateTensor<int64_t>(mem, const_cast<int64_t*>(tok_padded.data()), tok_padded.size(), s_t.data(), 2));
    in.push_back(Ort::Value::CreateTensor<int64_t>(mem, &T, 1, s1.data(), 1));
    in.push_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(g.data()), g.size(), s_g.data(), 2));
    const char* in_names[] = {"h", "pitch", "energy", "tok", "nf", "g"};
    const char* out_names[] = {"mel"};
    auto out = dec->Run(Ort::RunOptions{nullptr}, in_names, in.data(), in.size(), out_names, 1);
    const float* p = out[0].GetTensorData<float>();   // [1, 100, Tp]
    std::vector<float> mel(size_t(100 * T));
    for (int64_t b = 0; b < 100; ++b) std::memcpy(&mel[size_t(b * T)], p + b * Tp, size_t(T) * 4);
    return mel;
  }

  // model.infer (batch 1) + EdgeTTS.mel; returns T (0 = nothing to say)
  int mel(const std::string& text, const std::string& voice_in, const std::string& tag,
          const std::vector<std::pair<std::string, float>>& ctrl, float pace, const std::map<int, WordShift>& word_shift,
          float guidance, const std::vector<float>& spk_vec, std::vector<float>& mel_out) {
    Phonemes ph = front->to_phonemes(text);
    bool speech = false;
    for (auto i : ph.ids) speech |= i >= aa0;
    if (!speech) return 0;
    std::vector<int64_t> ids;
    std::vector<int32_t> widx;
    for (size_t k = 0; k < ph.ids.size(); ++k)
      if (ph.ids[k] < n_sym) { ids.push_back(ph.ids[k]); widx.push_back(ph.word_idx[k]); }
    const size_t N = ids.size(), Np = padded(N);
    std::array<float, 12> c, m;
    control_arrays(ctrl, c, m);
    std::vector<float> spk(64);
    if (!spk_vec.empty()) {
      size_t D = spk_vec.size();
      for (size_t o = 0; o < 64; ++o) {
        float a = spk_b[o];
        for (size_t i = 0; i < D; ++i) a += spk_vec[i] * spk_w[o * D + i];
        spk[o] = a;
      }
    } else {
      const std::string& v = voice_in.empty() ? default_voice : voice_in;
      auto it = voice2i.find(v);
      if (it == voice2i.end()) throw std::invalid_argument("unknown voice '" + v + "'");
      std::copy(&slots[size_t(it->second) * 64], &slots[size_t(it->second) * 64] + 64, spk.begin());
    }
    int64_t tg = tag2i.count(tag) ? tag2i.at(tag) : 0;
    float s = guidance < 0 ? opt.guidance : guidance;
    bool guided = s != 1.0f && (!ctrl.empty() || tag != "none");
    std::vector<int64_t> ids_p(Np, 0);
    std::copy(ids.begin(), ids.end(), ids_p.begin());
    // 1. encoder + variance predictors (and the unconditioned pass for guidance)
    EncOut e = run_enc(ids_p, int64_t(N), spk.data(), tg, c, m);
    EncOut e0;
    if (guided) {
      std::array<float, 12> z{};
      e0 = run_enc(ids_p, int64_t(N), spk.data(), 0, z, z);
      for (size_t k = 0; k < N; ++k) {
        e.logdur[k] = e0.logdur[k] + s * (e.logdur[k] - e0.logdur[k]);
        e.pitch[k] = e0.pitch[k] + s * (e.pitch[k] - e0.pitch[k]);
        e.energy[k] = e0.energy[k] + s * (e.energy[k] - e0.energy[k]);
      }
    }
    // 2. durations: (exp(logdur) - 1).clamp_min(0) / pace, per-word shifts, round half to even, first token >= 1
    std::vector<float> dur(N);
    for (size_t k = 0; k < N; ++k) dur[k] = std::max(std::exp(e.logdur[k]) - 1.0f, 0.0f) / pace;
    for (auto& kv : word_shift)
      for (size_t k = 0; k < N; ++k)
        if (widx[k] == kv.first) {
          e.pitch[k] += kv.second.pitch / kPitchScale;
          e.energy[k] += kv.second.energy / 10.0f;
          dur[k] *= kv.second.dur;
        }
    std::vector<int64_t> d(N);
    int64_t T = 0;
    for (size_t k = 0; k < N; ++k) {
      d[k] = std::max<int64_t>(int64_t(std::nearbyint(dur[k])), 0);   // FE_TONEAREST: half to even, as torch.round
      if (k == 0) d[k] = std::max<int64_t>(d[k], 1);
      T += d[k];
    }
    // 3. length regulation: frame -> phone index
    std::vector<int64_t> tok(padded(size_t(T)), 0);
    size_t f = 0;
    for (size_t k = 0; k < N; ++k)
      for (int64_t r = 0; r < d[k]; ++r) tok[f++] = int64_t(k);
    // 4. decoder (twice for guidance: out = out0 + s * (out - out0))
    mel_out = run_dec(e.h, e.pitch, e.energy, tok, T, e.g);
    if (guided) {
      std::vector<float> m0 = run_dec(e0.h, e.pitch, e.energy, tok, T, e0.g);
      for (size_t i = 0; i < mel_out.size(); ++i) mel_out[i] = m0[i] + s * (mel_out[i] - m0[i]);
    }
    return int(T);
  }

  std::vector<float> render(const std::string& text, const std::string& voice, const std::string& tag,
                            const std::vector<std::pair<std::string, float>>& ctrl, float pace,
                            const std::map<int, WordShift>& ws, float guidance, const std::vector<float>& spk_vec) {
    std::vector<float> m;
    int T = mel(text, voice, tag, ctrl, pace, ws, guidance, spk_vec, m);
    if (T == 0) return {};
    std::array<int64_t, 3> shp{1, 100, T};
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, m.data(), m.size(), shp.data(), 3);
    const char* in_names[] = {"mel"};
    const char* out_names[] = {"re", "im"};
    auto out = voc->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 2);
    return dsp::istft(out[0].GetTensorData<float>(), out[1].GetTensorData<float>(), T);
  }
};

Engine::Engine(const std::string& package_dir, Options opt) : p_(new Impl) {
  Impl& p = *p_;
  p.opt = opt;
  p.dir = package_dir;
  json::Value cfg = json::parse(read_file(package_dir + "/config.json"));
  for (auto& t : cfg["tags"].arr) {
    p.tag2i[t.str] = int(p.tags.size());
    p.tags.push_back(t.str);
  }
  p.n_sym = int(cfg["acoustic"]["arch"]["n_symbols"].num);
  json::Value vj = json::parse(read_file(package_dir + "/voices.json"));
  for (auto& v : vj["voices"].arr) {
    VoiceInfo vi;
    vi.name = v["name"].str;
    if (v.has("gender")) vi.gender = v["gender"].str;
    if (v.has("pitch_register")) vi.pitch_register = v["pitch_register"].str;
    if (v.has("pace")) vi.pace = v["pace"].str;
    if (v.has("f0_median_hz")) vi.f0_median_hz = v["f0_median_hz"].num;
    p.voice2i[vi.name] = int(p.voices.size());
    p.voices.push_back(vi);
  }
  p.default_voice = cfg.has("default_voice") ? cfg["default_voice"].str : p.voices.at(0).name;
  auto vt = load_safetensors(package_dir + "/voices.safetensors");
  p.slots = std::move(vt.at("s"));
  if (p.slots.size() != p.voices.size() * 64) throw std::runtime_error("voices.safetensors does not match voices.json");
  auto sp = load_safetensors(package_dir + "/onnx/spkproj.safetensors");
  p.spk_w = std::move(sp.at("w"));
  p.spk_b = std::move(sp.at("b"));
  p.front.reset(new Frontend(package_dir + "/native/front"));
  const auto& syms = p.front->symbols();
  p.aa0 = int(std::find(syms.begin(), syms.end(), "AA0") - syms.begin());
  p.so.SetIntraOpNumThreads(opt.threads);
  p.so.SetInterOpNumThreads(1);
  p.so.DisableCpuMemArena();   // give activation memory back after each call
  p.so.AddConfigEntry("session.disable_prepacking", "1");   // no second, pre-packed weight copy
  p.so.DisableMemPattern();    // no pre-planned block for all intermediates (+24 MB peak with it)
  p.so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  auto mk = [&](const char* f) {
    std::string path = package_dir + "/onnx/" + f;
    return std::unique_ptr<Ort::Session>(new Ort::Session(env(), path.c_str(), p.so));
  };
  p.enc = mk("enc.onnx");
  p.dec = mk("dec.onnx");
  p.voc = mk("voc.onnx");
}

Engine::~Engine() = default;

std::vector<float> Engine::synthesize(const Request& r) {
  p_->check(r, r.markup);
  if (!r.markup)
    return p_->render(r.text, r.voice, r.tag, r.ctrl, r.pace, r.word_shift, r.guidance, r.spk_vec);
  std::vector<float> out;
  bool any = false;
  stream(r, [&](const Chunk& c) {
    out.insert(out.end(), c.audio.begin(), c.audio.end());
    any = true;
    return true;
  });
  if (!any) out.assign(1, 0.0f);   // MarkupSynth.render: np.zeros(1) when there is no segment
  return out;
}

void Engine::stream(const Request& r, const std::function<bool(const Chunk&)>& on_chunk) {
  Request q = r;
  q.word_shift.clear();
  p_->check(r, true);
  const int fade = int(0.01 * kSampleRate);
  for (Segment& seg : parse_markup(*p_->front, r.text, r.tag, r.ctrl, p_->tags)) {
    // Python parity quirk (see front.h): a text piece that is literally "PAUSE" crashes Python's stream(); skip it.
    // A literal "EVENT" piece renders empty text there, i.e. an empty chunk, which the code below reproduces.
    if (seg.pause_ms == Segment::kPauseIsStyleTuple) continue;
    if (seg.event == Segment::kEventIsStyleTuple) seg.text.clear();
    auto t0 = std::chrono::steady_clock::now();
    Chunk c;
    if (seg.pause_ms > 0) {
      c.audio.assign(size_t(double(seg.pause_ms) * kSampleRate / 1000.0), 0.0f);
    } else {
      c.audio = p_->render(seg.text, r.voice, seg.tag, seg.ctrl, r.pace, seg.word_shift, r.guidance, r.spk_vec);
      dsp::fade_ends(c.audio, fade);
    }
    c.synth_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    c.segment = std::move(seg);
    if (!on_chunk(c)) return;
  }
}

int Engine::mel(const Request& r, std::vector<float>& out) {
  p_->check(r, false);
  return p_->mel(r.text, r.voice, r.tag, r.ctrl, r.pace, r.word_shift, r.guidance, r.spk_vec, out);
}

const std::vector<std::string>& Engine::tags() const { return p_->tags; }
const std::vector<std::string>& Engine::controls() const { return kControls; }
const std::vector<VoiceInfo>& Engine::voices() const { return p_->voices; }
const std::string& Engine::default_voice() const { return p_->default_voice; }
const Frontend& Engine::frontend() const { return *p_->front; }

void Engine::save_wav(const std::vector<float>& audio, const std::string& path) { dsp::save_wav(audio, kSampleRate, path); }

}  // namespace steertts
