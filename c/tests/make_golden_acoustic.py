"""Golden data for c/tests/test_acoustic.c: inputs + reference outputs of the PyTorch acoustic models and vocoders.

    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=<repo>:<repo>/edge python c/tests/make_golden_acoustic.py \
        [--v1 <c_native/v1 dir>] [--edge <edge package dir>] [--out c/tests/golden/acoustic]

Writes one safetensors file per case (<model>_<case>.safetensors: ids, widx, speaker input, tag, ctrl, cmask, guidance
/ base flag, pace, per-word shifts in model units, reference g / g0, durations, mel [T, 100] time-major) and
vocoder.safetensors (a reference mel and the v1 Vocos fp32 / int8 and edge vocoder waveforms). The inputs follow
pipeline._render / EdgeTTS.mel exactly (ids from steertts.text.to_phonemes, word_shift user units -> model units).
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

V1 = "../models/steer-tts"
EDGE = "../models/steer-tts-edge"
SHORT = "Well, I never thought the red boat would make it."
LONG = ("On the morning of the fourteenth, the committee gathered in the old library to review the proposal. "
        "Nobody expected the discussion to last until midnight, but the questions kept coming.")
CASES = {
    # name: text, voice index (None = default voice), tag, ctrl, guidance, pace, word_shift (user units)
    "plain": (SHORT, None, "none", {}, 1.0, 1.0, None),
    "steer": (SHORT, 7, "happiness", {"rate": 1.0, "f0_iqr": 0.8}, 1.5, 0.8,
              {1: {"pitch": 2.0, "energy": 3.0, "dur": 1.3}, 3: {"dur": 0.7}, 6: {"pitch": -1.5}}),
    "long": (LONG, 120, "none", {"f0_med": 0.5}, 1.0, 1.0, None),
}
PLAN = {"v1": ["plain", "steer", "long"], "v1int8": ["steer"], "edge": ["steer", "long"]}
VOC_FRAMES = 160


def inputs(case, n_sym, tags, n_ctrl, controls):
    from steertts.config import PITCH_SCALE
    from steertts.text import to_phonemes
    text, vi, tag, ctrl, gd, pace, ws = case
    ids, widx, _ = to_phonemes(text)
    keep = [k for k, i in enumerate(ids) if i < n_sym]
    ids, widx = [ids[k] for k in keep], [widx[k] for k in keep]
    c = torch.zeros(1, n_ctrl)
    m = torch.zeros(1, n_ctrl)
    for k, v in ctrl.items():
        c[0, controls.index(k)], m[0, controls.index(k)] = float(v), 1.0
    tg = torch.tensor([tags.index(tag)])
    n_words = max(widx) + 1
    shifts = np.zeros((n_words, 3), np.float32)
    shifts[:, 2] = 1.0
    wsm = None
    if ws:
        wsm = {int(w): dict(pitch=s.get("pitch", 0) / PITCH_SCALE, energy=s.get("energy", 0) / 10.0,
                            dur=s.get("dur", 1.0)) for w, s in ws.items()}
        for w, s in wsm.items():   # torch applies python-float scalars to float32 tensors as float32
            shifts[w] = [np.float32(s["pitch"]), np.float32(s["energy"]), np.float32(s["dur"])]
    kw = {}
    if gd != 1.0 and (ctrl or tag != "none"):
        kw = dict(guidance=gd, base=(torch.zeros_like(tg), torch.zeros_like(c), torch.zeros_like(m)))
    return ids, widx, c, m, tg, wsm, shifts, kw, pace


def run_case(model, case, spk, spk_vec, tags, controls):
    ids, widx, c, m, tg, wsm, shifts, kw, pace = inputs(case, model.emb.num_embeddings, tags, len(controls), controls)
    ids_t, widx_t = torch.tensor([ids]), torch.tensor([widx])
    with torch.no_grad():
        mel, aux = model.infer(ids_t, spk, tg, c, m, pace=pace, word_idx=widx_t, word_shift=wsm, spk_vec=spk_vec, **kw)
        g0 = model.condition(spk, *kw["base"], spk_vec) if kw else torch.zeros_like(aux["g"])
    out = dict(ids=np.array(ids, np.int32), widx=np.array(widx, np.int32), tag=np.array([int(tg)], np.int32),
               ctrl=c[0].numpy(), cmask=m[0].numpy(), guidance=np.array([kw.get("guidance", 1.0)], np.float32),
               base=np.array([1 if kw else 0], np.int32), pace=np.array([pace], np.float32), shifts=shifts,
               g=aux["g"][0].numpy(), g0=g0[0].numpy(), dur=aux["dur"][0].numpy().astype(np.int32),
               mel=np.ascontiguousarray(mel[0].T.numpy().astype(np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=V1)
    ap.add_argument("--edge", default=EDGE)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "golden" / "acoustic"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from steertts import SteerTTS
    from steertts.config import CONTROL_DIMS
    from steertts.quant import round_int8_
    from steertts_edge import EdgeTTS

    ref_mel = None
    v1_vocos = {}
    for prec, key in (("fp32", "v1"), ("int8", "v1int8")):
        tts = SteerTTS(a.v1, device="cpu", flow=False, precision=prec)
        spk = torch.tensor([tts.n_speakers])
        for cname in PLAN[key]:
            case = CASES[cname]
            voice = tts.default_voice if case[1] is None else tts.voices[case[1]]
            sv = torch.from_numpy(np.asarray(tts.spk_ref[voice], np.float32))[None]
            o = run_case(tts.model, case, spk, sv, tts.tags, CONTROL_DIMS)
            o["spk_vec"] = sv[0].numpy()
            save_file(o, str(out / f"{key}_{cname}.safetensors"))
            print(f"{key}_{cname}: N={len(o['ids'])} T={o['mel'].shape[0]} voice={voice}")
            if key == "v1" and cname == "plain":
                ref_mel = o["mel"][:VOC_FRAMES]
        v1_vocos[prec] = tts.vocos
    edge = EdgeTTS(a.edge)
    for cname in PLAN["edge"]:
        case = CASES[cname]
        vi = edge._vi[edge.default_voice] if case[1] is None else case[1]
        s = edge._s[vi][None]
        o = run_case(edge.model, case, s, None, edge.tags, CONTROL_DIMS)
        o["spk64"] = s[0].numpy()
        o["voice_row"] = np.array([vi], np.int32)
        save_file(o, str(out / f"edge_{cname}.safetensors"))
        print(f"edge_{cname}: N={len(o['ids'])} T={o['mel'].shape[0]} voice row={vi}")

    mel_t = torch.from_numpy(ref_mel.T.copy())[None]
    with torch.no_grad():
        voc = dict(mel=np.ascontiguousarray(ref_mel),
                   wav_v1=v1_vocos["fp32"].decode(mel_t)[0].numpy(),
                   wav_v1int8=v1_vocos["int8"].decode(mel_t)[0].numpy(),
                   wav_edge=edge.vocoder.decode(mel_t)[0].numpy())
    # the int8 Vocos above must be the round_int8_ rounding of the fp32 one (what export_vocos.py writes)
    chk = round_int8_(copy.deepcopy(v1_vocos["fp32"]))
    with torch.no_grad():
        assert torch.equal(chk.decode(mel_t)[0], torch.from_numpy(voc["wav_v1int8"]))
    save_file({k: v.astype(np.float32) for k, v in voc.items()}, str(out / "vocoder.safetensors"))
    print(f"vocoder: T={VOC_FRAMES} samples={len(voc['wav_v1'])}")
    total = sum(f.stat().st_size for f in out.glob("*.safetensors"))
    print(f"golden size {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
