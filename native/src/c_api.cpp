#include "steertts/steertts_c.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#include "steertts/steertts.h"

struct steertts_engine {
  steertts::Engine engine;
  steertts_engine(const char* dir, steertts::Options o) : engine(dir, o) {}
};

namespace {

void set_err(char* err, size_t len, const char* msg) {
  if (err && len) {
    std::strncpy(err, msg, len - 1);
    err[len - 1] = 0;
  }
}

steertts::Request make_request(const char* text, const steertts_options* o) {
  steertts::Request r;
  r.text = text ? text : "";
  if (!o) return r;
  if (o->voice) r.voice = o->voice;
  if (o->tag) r.tag = o->tag;
  if (o->ctrl) {
    std::stringstream ss(o->ctrl);
    std::string kv;
    while (std::getline(ss, kv, ',')) {
      if (kv.empty()) continue;
      size_t eq = kv.find('=');
      if (eq == std::string::npos) throw std::invalid_argument("ctrl expects name=value pairs, got '" + kv + "'");
      r.ctrl.emplace_back(kv.substr(0, eq), std::stof(kv.substr(eq + 1)));
    }
  }
  r.pace = o->pace;
  r.guidance = o->guidance;
  r.markup = o->markup != 0;
  return r;
}

}  // namespace

extern "C" {

steertts_engine* steertts_create(const char* package_dir, int threads, char* err, size_t err_len) {
  try {
    steertts::Options o;
    o.threads = threads > 0 ? threads : 1;
    return new steertts_engine(package_dir, o);
  } catch (const std::exception& e) {
    set_err(err, err_len, e.what());
    return nullptr;
  }
}

void steertts_destroy(steertts_engine* e) { delete e; }

void steertts_default_options(steertts_options* o) {
  o->voice = nullptr;
  o->tag = nullptr;
  o->ctrl = nullptr;
  o->pace = 1.0f;
  o->guidance = -1.0f;
  o->markup = 0;
}

int steertts_sample_rate(void) { return steertts::kSampleRate; }

int steertts_synthesize(steertts_engine* e, const char* text, const steertts_options* o, float** audio, size_t* n,
                        char* err, size_t err_len) {
  try {
    std::vector<float> y = e->engine.synthesize(make_request(text, o));
    *n = y.size();
    *audio = static_cast<float*>(std::malloc(std::max<size_t>(1, y.size()) * sizeof(float)));
    if (!*audio) throw std::bad_alloc();
    if (!y.empty()) std::memcpy(*audio, y.data(), y.size() * sizeof(float));
    return 0;
  } catch (const std::exception& ex) {
    set_err(err, err_len, ex.what());
    return 1;
  }
}

void steertts_free_audio(float* audio) { std::free(audio); }

int steertts_stream(steertts_engine* e, const char* text, const steertts_options* o, steertts_chunk_fn fn, void* user,
                    char* err, size_t err_len) {
  try {
    e->engine.stream(make_request(text, o), [&](const steertts::Chunk& c) {
      return fn(c.audio.data(), c.audio.size(), user) == 0;
    });
    return 0;
  } catch (const std::exception& ex) {
    set_err(err, err_len, ex.what());
    return 1;
  }
}

int steertts_save_wav(const float* audio, size_t n, const char* path) {
  try {
    steertts::Engine::save_wav(std::vector<float>(audio, audio + n), path);
    return 0;
  } catch (...) {
    return 1;
  }
}

}  // extern "C"
