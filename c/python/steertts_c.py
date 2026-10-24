"""Python binding for the stand-alone C library (ctypes, no compilation; numpy optional).

    from steertts_c import Engine
    tts = Engine("steer-tts-edge", threads=2)                       # or a v1 dir; precision="int8" for v1 int8
    wav = tts("Are you coming with us tonight?", ctrl={"f0_end": 2, "f0_iqr": 1.5})   # float32 numpy array, 24 kHz
    wav = tts.synthesize({"text": "It is over.", "tag": "sadness", "guidance": 1.5})    # a full JSON-style request
    for chunk, info in tts.stream("First sentence. <whisper>Second.</whisper>"):
        ...
    tts.save(wav, "out.wav")

The library is found from $STEERTTS_LIB, else libsteertts.so / .dylib / steertts.dll next to this file, in
../build, or on the system library path.
"""
import ctypes
import json
import os
import sys
from ctypes import POINTER, c_char_p, c_float, c_int, c_int64, c_size_t, c_void_p
from pathlib import Path

SAMPLE_RATE = 24000


def _load():
    names = {"darwin": ["libsteertts.dylib"], "win32": ["steertts.dll", "libsteertts.dll"]}.get(sys.platform,
                                                                                               ["libsteertts.so"])
    here = Path(__file__).resolve().parent
    cands = [os.environ["STEERTTS_LIB"]] if os.environ.get("STEERTTS_LIB") else []
    for d in (here, here.parent / "build", here.parent / "build" / "Release"):
        cands += [str(d / n) for n in names]
    cands += names
    for c in cands:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    raise OSError("libsteertts not found: build c/ (cmake -B build && cmake --build build) or set STEERTTS_LIB")


class _Config(ctypes.Structure):
    _fields_ = [("threads", c_int), ("precision", c_char_p), ("flow", c_int), ("flow_steps", c_int),
                ("flow_temp", c_float), ("guidance", c_float), ("assets_dir", c_char_p)]


class _ChunkInfo(ctypes.Structure):
    _fields_ = [("text", c_char_p), ("tag", c_char_p), ("ctrl", c_char_p), ("pause_ms", c_int), ("index", c_int),
                ("synth_ms", ctypes.c_double)]


_CHUNK_FN = ctypes.CFUNCTYPE(c_int, POINTER(c_float), c_size_t, POINTER(_ChunkInfo), c_void_p)
_lib = None


def _api():
    global _lib
    if _lib is None:
        L = _load()
        L.steertts_default_config.argtypes = [POINTER(_Config)]
        L.steertts_create_ex.argtypes = [c_char_p, POINTER(_Config), c_char_p, c_size_t]
        L.steertts_create_ex.restype = c_void_p
        L.steertts_destroy.argtypes = [c_void_p]
        L.steertts_synthesize_json.argtypes = [c_void_p, c_char_p, POINTER(POINTER(c_float)), POINTER(c_size_t),
                                               c_char_p, c_size_t]
        L.steertts_stream_json.argtypes = [c_void_p, c_char_p, _CHUNK_FN, c_void_p, c_char_p, c_size_t]
        L.steertts_resolve_json.argtypes = [c_void_p, c_char_p, POINTER(c_char_p), c_char_p, c_size_t]
        L.steertts_map_prompt.argtypes = [c_void_p, c_char_p, POINTER(c_char_p), c_char_p, c_size_t]
        L.steertts_speaker_vector.argtypes = [c_void_p, c_char_p, POINTER(c_float), c_size_t, c_int, POINTER(c_float),
                                              c_char_p, c_size_t]
        L.steertts_free.argtypes = [c_void_p]
        L.steertts_save_wav.argtypes = [POINTER(c_float), c_size_t, c_char_p]
        for f in ("steertts_model_type", "steertts_voice", "steertts_tag", "steertts_default_voice",
                  "steertts_version", "steertts_control"):
            getattr(L, f).restype = c_char_p
        L.steertts_model_type.argtypes = L.steertts_default_voice.argtypes = [c_void_p]
        L.steertts_voice.argtypes = L.steertts_tag.argtypes = [c_void_p, c_int]
        L.steertts_num_voices.argtypes = L.steertts_num_tags.argtypes = [c_void_p]
        L.steertts_control.argtypes = [c_int]
        L.steertts_has.argtypes = [c_void_p, c_char_p]
        _lib = L
    return _lib


def _json(req):
    """Request dict -> JSON bytes; numpy arrays / scalars (e.g. spk_vec) become lists / numbers."""
    def conv(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if isinstance(o, os.PathLike):
            return os.fspath(o)
        raise TypeError(f"{type(o).__name__} is not JSON serializable")
    return json.dumps(req, default=conv).encode()


def _array(ptr, n):
    try:
        import numpy as np
        return np.ctypeslib.as_array(ptr, shape=(n,)).copy() if n else np.zeros(0, np.float32)
    except ImportError:
        return [ptr[i] for i in range(n)]


class SteerTTSError(ValueError):
    pass


class Engine:
    sample_rate = SAMPLE_RATE

    def __init__(self, model_dir=None, threads=1, precision="fp32", flow=True, flow_steps=10, flow_temp=0.4,
                 guidance=1.0, assets_dir=None):
        L = _api()
        c = _Config()
        L.steertts_default_config(ctypes.byref(c))
        c.threads, c.precision, c.flow, c.flow_steps = threads, precision.encode(), int(flow), flow_steps
        c.flow_temp, c.guidance = flow_temp, guidance
        c.assets_dir = os.fsencode(str(assets_dir)) if assets_dir else None
        err = ctypes.create_string_buffer(1024)
        md = os.fsencode(str(model_dir)) if model_dir else None   # None: $STEERTTS_MODEL
        self._e = L.steertts_create_ex(md, ctypes.byref(c), err, len(err))
        if not self._e:
            raise SteerTTSError(err.value.decode())
        self.model_type = L.steertts_model_type(self._e).decode()
        self.voices = [L.steertts_voice(self._e, i).decode() for i in range(L.steertts_num_voices(self._e))]
        self.tags = [L.steertts_tag(self._e, i).decode() for i in range(L.steertts_num_tags(self._e))]
        self.controls = [L.steertts_control(i).decode() for i in range(12)]
        self.default_voice = L.steertts_default_voice(self._e).decode()

    def close(self):
        if getattr(self, "_e", None):
            _api().steertts_destroy(self._e)
            self._e = None

    __del__ = close

    def has(self, feature):
        """'prompt', 'clone' or 'flow'."""
        return bool(_api().steertts_has(self._e, feature.encode()))

    def synthesize(self, request=None, **kw):
        """A request dict (request.schema.json fields) and/or keyword fields -> float32 waveform at 24 kHz."""
        req = _json({**(request or {}), **{k: v for k, v in kw.items() if v is not None}})
        out, n, err = POINTER(c_float)(), c_size_t(), ctypes.create_string_buffer(1024)
        if _api().steertts_synthesize_json(self._e, req, ctypes.byref(out), ctypes.byref(n), err, len(err)):
            raise SteerTTSError(err.value.decode())
        try:
            return _array(out, n.value)
        finally:
            _api().steertts_free(out)

    def __call__(self, text, **kw):
        return self.synthesize(dict(kw, text=text))

    def stream(self, text=None, request=None, **kw):
        """Yields (audio, info) per sentence / pause / style span; info: text, tag, ctrl, pause_ms, index, synth_ms.
        (The whole text is rendered before the generator returns its first item; use stream_cb for true
        incremental playback.)"""
        chunks = []
        self.stream_cb(lambda a, i: chunks.append((a, i)), text, request, **kw)
        yield from chunks

    def stream_cb(self, fn, text=None, request=None, **kw):
        """fn(audio, info) is called as each chunk is rendered; return True from fn to stop."""
        if text is not None:
            kw["text"] = text
        req = _json({**(request or {}), **{k: v for k, v in kw.items() if v is not None}})
        failure = []

        def cb(ptr, n, info, _):
            try:
                i = info.contents
                d = dict(text=i.text.decode(), tag=i.tag.decode(), ctrl=i.ctrl.decode(), pause_ms=i.pause_ms,
                         index=i.index, synth_ms=i.synth_ms)
                return 1 if fn(_array(ptr, n), d) else 0
            except BaseException as e:   # stop the stream and re-raise in the caller
                failure.append(e)
                return 1

        err = ctypes.create_string_buffer(1024)
        rc = _api().steertts_stream_json(self._e, req, _CHUNK_FN(cb), None, err, len(err))
        if failure:
            raise failure[0]
        if rc:
            raise SteerTTSError(err.value.decode())

    def resolve(self, request=None, **kw):
        req = _json({**(request or {}), **kw})
        out, err = c_char_p(), ctypes.create_string_buffer(1024)
        if _api().steertts_resolve_json(self._e, req, ctypes.byref(out), err, len(err)):
            raise SteerTTSError(err.value.decode())
        s = out.value.decode()
        _api().steertts_free(out)
        return json.loads(s)

    def speaker_vector(self, wav_path):
        """v1: the 512-d x-vector of a WAV file (usable as spk_vec with either model)."""
        v, err = (c_float * 512)(), ctypes.create_string_buffer(1024)
        if _api().steertts_speaker_vector(self._e, os.fsencode(str(wav_path)), None, 0, 0, v, err, len(err)):
            raise SteerTTSError(err.value.decode())
        return list(v)

    @staticmethod
    def save(wav, path):
        """16-bit PCM WAV, 24 kHz mono."""
        if hasattr(wav, "astype"):
            arr = wav.astype("float32", copy=False).ravel()   # keep a reference while C reads it
            buf, n = arr.ctypes.data_as(POINTER(c_float)), len(arr)
        else:
            n = len(wav)
            buf = (c_float * n)(*wav)
        if _api().steertts_save_wav(buf, n, os.fsencode(str(path))):
            raise OSError(f"cannot write {path}")
