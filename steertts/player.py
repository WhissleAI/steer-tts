"""Real-time playback for the CLI: chunks are played as they are rendered, while the next one is synthesised.

Backends, in order: sounddevice (pip install "steertts[play]"; bundles PortAudio on macOS / Windows, needs
libportaudio2 on Linux), then ffplay (ffmpeg), then aplay (ALSA)."""
import queue
import shutil
import subprocess
import threading

import numpy as np


class Player:
    def __init__(self, sr):
        self.sr, self.q, self.err = sr, queue.Queue(), None
        self._open()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _open(self):
        try:
            import sounddevice as sd
            self.sd = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32")
            self.sd.start()
            self.proc, self.name = None, "sounddevice"
            return
        except Exception as e:  # not installed, or no PortAudio / output device
            first = e
        self.sd = None
        for name, cmd in (("ffplay", ["ffplay", "-loglevel", "error", "-nodisp", "-autoexit", "-f", "f32le",
                                      "-ar", str(self.sr), "-ac", "1", "-i", "-"]),
                          ("aplay", ["aplay", "-q", "-f", "FLOAT_LE", "-r", str(self.sr), "-c", "1"])):
            if shutil.which(name):
                self.proc, self.name = subprocess.Popen(cmd, stdin=subprocess.PIPE), name
                return
        raise RuntimeError(f"no audio output: install sounddevice (pip install \"steertts[play]\"), ffmpeg or "
                           f"alsa-utils, or write a file with -o instead ({first})")

    def _run(self):
        try:
            while (y := self.q.get()) is not None:
                y = np.ascontiguousarray(y, np.float32)
                if self.sd is not None:
                    self.sd.write(y[:, None])
                else:
                    self.proc.stdin.write(y.tobytes())
                    self.proc.stdin.flush()
        except Exception as e:  # device unplugged, player closed by the user
            self.err = e

    def play(self, y):
        self.q.put(y)

    def close(self):
        """Blocks until everything queued has been played."""
        self.q.put(None)
        self.t.join()
        if self.sd is not None:
            self.sd.stop(); self.sd.close()
        else:
            self.proc.stdin.close(); self.proc.wait()
