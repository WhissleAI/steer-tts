"""Replace fp32 MatMul weights in the exported graphs with ONNX Runtime MatMulNBits nodes holding the package's own
int4 / int8 codes (exact: the same q and scales as acoustic/vocoder.safetensors, no re-quantization).

    python pack_onnx.py <pkg_dir> <fp32_dir> <out_dir>

int4 tensors (groups of g along the flattened input dim, code + 8 in a nibble, one fp16 scale per group) map 1:1
onto MatMulNBits bits=4, block_size=g, default zero point 8. int8 tensors (one scale per output row) map onto
bits=8 with the row scale repeated per block and code + 128 (default zero point 128). MatMul weights whose source
tensor is not quantized (fp16 in the package) are stored as they are.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from safetensors.numpy import load_file
from safetensors import safe_open

PKG, SRC, OUT = (Path(a) for a in sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)


def packed_tensors(fn):
    with safe_open(str(fn), "np") as f:
        lay = json.loads(f.metadata()["steertts_quant"])["layout"]
    t = load_file(str(fn))
    out = {}
    for k, L in lay.items():
        if L["s"] not in ("int4", "int8"):
            continue
        q, s = t[k + "::q"], t[k + "::s"].astype(np.float32)
        shape = L["shape"]
        O, K = shape[0], int(np.prod(shape[1:]))
        if L["s"] == "int8":
            w = (q.astype(np.float32) * s.reshape(-1, *([1] * (q.ndim - 1)))).reshape(O, K)
            out[k] = dict(kind="int8", w=w, q=q.reshape(O, K).astype(np.int16), s=s.reshape(O))
        else:
            g, Kp = L["g"], L["Kp"]
            codes = np.stack([q & 15, q >> 4], 1).reshape(-1)[: O * Kp].reshape(O, Kp)   # stored code = value + 8
            w = ((codes.astype(np.float32) - 8).reshape(O, Kp // g, g) * s.reshape(O, Kp // g, 1)).reshape(O, Kp)[:, :K]
            out[k] = dict(kind="int4", w=w, codes=codes, s=s.reshape(O, Kp // g), g=g, Kp=Kp)
    return out


def key(a):
    a = np.ascontiguousarray(a, np.float32)
    return a.shape, hashlib.sha1(a.tobytes()).hexdigest()


def nbits_inputs(P, K, N):
    """-> (B uint8 [N, nblk, blob], scales float32 [N*nblk], bits, block)."""
    if P["kind"] == "int4":
        g, Kp = P["g"], P["Kp"]
        codes = P["codes"].astype(np.uint8)                                  # [N, Kp]
        B = (codes[:, 0::2] | (codes[:, 1::2] << 4)).reshape(N, Kp // g, g // 2)
        return B, P["s"].reshape(-1).astype(np.float32), 4, g
    block = 32
    nblk = -(-K // block)
    codes = np.full((N, nblk * block), 128, np.uint8)                        # padding: code 128 = value 0
    codes[:, :K] = (P["q"] + 128).astype(np.uint8)
    return codes.reshape(N, nblk, block), np.repeat(P["s"], nblk).astype(np.float32), 8, block


def compact(m):
    """Drop what inference never reads (exporter metadata, stack traces, doc strings) and shorten internal names:
    ~1.8 MB of the 6 MB encoder graph was node metadata. The shape annotations (value_info) stay: without them ONNX
    Runtime re-runs shape inference at every load (3x slower session creation for +0.1 MB)."""
    keep = {i.name for i in m.graph.input} | {o.name for o in m.graph.output}
    ren, k = {}, [0]

    def short(x):
        if not x or x in keep:
            return x
        if x not in ren:
            ren[x] = f"t{k[0]:x}"
            k[0] += 1
        return ren[x]

    for n in m.graph.node:
        n.doc_string = ""
        del n.metadata_props[:]
        n.name = ""
        n.input[:] = [short(x) for x in n.input]
        n.output[:] = [short(x) for x in n.output]
    for i in m.graph.initializer:
        i.name = short(i.name)
        i.doc_string = ""
    # fp32 initializers whose values are exactly fp16 (MatMulNBits scales, the package's fp16 tensors) are stored as
    # fp16 + Cast; ONNX Runtime folds the Cast when the session loads, so the weights it runs with are bit-identical
    casts, inits = [], []
    for i in m.graph.initializer:
        a = numpy_helper.to_array(i)
        if a.dtype == np.float32 and a.size >= 64 and np.array_equal(a.astype(np.float16).astype(np.float32), a):
            h = numpy_helper.from_array(a.astype(np.float16), i.name + "h")
            inits.append(h)
            casts.append(helper.make_node("Cast", [h.name], [i.name], to=onnx.TensorProto.FLOAT))
        else:
            inits.append(i)
    del m.graph.initializer[:]
    m.graph.initializer.extend(inits)
    nodes = casts + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    for v in m.graph.value_info:
        v.name = short(v.name)
        v.doc_string = ""
        del v.metadata_props[:]
    for v in list(m.graph.input) + list(m.graph.output):
        v.doc_string = ""
        del v.metadata_props[:]
    del m.metadata_props[:]
    del m.graph.metadata_props[:]
    m.doc_string = ""
    m.graph.doc_string = ""
    m.producer_name, m.producer_version = "steertts-edge", ""
    return m


def main():
    index = {}
    for f in ("acoustic.safetensors", "vocoder.safetensors"):
        for name, P in packed_tensors(PKG / f).items():
            index[key(P["w"].T)] = (name, P)          # MatMul B = Linear weight transposed
            index[("W",) + key(P["w"])] = (name, P)   # Gemm transB=1 B / Conv weight flattened to [O, C*k]
    for g in ("enc", "dec", "voc"):
        m = onnx.load(str(SRC / f"{g}.onnx"))
        inits = {i.name: i for i in m.graph.initializer}
        nodes, n_rep, n_keep, drop = [], 0, 0, set()
        uid = [0]

        def fresh(tag):
            uid[0] += 1
            return f"{n.name}_{tag}{uid[0]}"

        def nbits(a, W_KN, P, out):
            K, N = W_KN.shape
            B, S, bits, block = nbits_inputs(P, K, N)
            bn, sn = fresh("q"), fresh("s")
            m.graph.initializer.extend([numpy_helper.from_array(B, bn), numpy_helper.from_array(S, sn)])
            return helper.make_node("MatMulNBits", [a, bn, sn], [out], name=fresh("nbits"), domain="com.microsoft",
                                    K=K, N=N, bits=bits, block_size=block, accuracy_level=0)

        def const(arr, tag):
            nm = fresh(tag)
            m.graph.initializer.append(numpy_helper.from_array(np.asarray(arr), nm))
            return nm

        for n in m.graph.node:
            att = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            if n.op_type == "MatMul" and n.input[1] in inits:
                W = numpy_helper.to_array(inits[n.input[1]])
                hit = index.get(key(W)) if W.ndim == 2 else None
                if hit:
                    nodes.append(nbits(n.input[0], W, hit[1], n.output[0])); drop.add(n.input[1]); n_rep += 1
                    continue
                n_keep += 1
            elif (n.op_type == "Gemm" and n.input[1] in inits and att.get("transB", 0) == 1 and att.get("transA", 0) == 0
                  and att.get("alpha", 1.0) == 1.0 and att.get("beta", 1.0) == 1.0):
                W = numpy_helper.to_array(inits[n.input[1]])                 # [N, K]
                hit = index.get(("W",) + key(W))
                if hit:
                    mm = fresh("mm")
                    nodes.append(nbits(n.input[0], W.T, hit[1], mm if len(n.input) > 2 else n.output[0]))
                    if len(n.input) > 2:
                        nodes.append(helper.make_node("Add", [mm, n.input[2]], [n.output[0]], name=fresh("bias")))
                    drop.add(n.input[1]); n_rep += 1
                    continue
            elif n.op_type == "Conv" and n.input[1] in inits and att.get("group", 1) == 1:
                W = numpy_helper.to_array(inits[n.input[1]])                 # [O, C, k]
                O, Cin, k = W.shape if W.ndim == 3 else (0, 0, 0)
                hit = index.get(("W",) + key(W.reshape(O, -1))) if W.ndim == 3 else None
                pads = att.get("pads", [0, 0])
                if (hit and att.get("strides", [1]) == [1] and att.get("dilations", [1]) == [1]
                        and pads == [k // 2, k // 2] and k % 2 == 1):
                    # im2col: X [1,C,T] -> pad T by k//2 -> k shifted slices stacked on a new last axis [1,C,T,k]
                    # -> [1,T,C,k] -> [1,T,C*k] (channel-major, tap-minor = the packed groups' flattening) -> MatMulNBits
                    xp = fresh("pad")
                    nodes.append(helper.make_node("Pad", [n.input[0], const(np.array([0, 0, k // 2, 0, 0, k // 2], np.int64), "pp")],
                                                  [xp], name=fresh("padn")))
                    taps = []
                    for j in range(k):
                        end = -(k - 1 - j) if j < k - 1 else np.iinfo(np.int64).max
                        sl = fresh("sl")
                        nodes.append(helper.make_node("Slice", [xp, const(np.array([j], np.int64), "st"),
                                                                const(np.array([end], np.int64), "en"),
                                                                const(np.array([2], np.int64), "ax")], [sl], name=fresh("sln")))
                        us = fresh("us")
                        nodes.append(helper.make_node("Unsqueeze", [sl, const(np.array([3], np.int64), "ua")], [us], name=fresh("usn")))
                        taps.append(us)
                    cat, tr, rs, mm = fresh("cat"), fresh("tr"), fresh("rs"), fresh("mm")
                    nodes.append(helper.make_node("Concat", taps, [cat], axis=3, name=fresh("catn")))
                    nodes.append(helper.make_node("Transpose", [cat], [tr], perm=[0, 2, 1, 3], name=fresh("trn")))
                    nodes.append(helper.make_node("Reshape", [tr, const(np.array([0, 0, Cin * k], np.int64), "rsh")], [rs], name=fresh("rsn")))
                    nodes.append(nbits(rs, W.reshape(O, -1).T, hit[1], mm))
                    if len(n.input) > 2:
                        mb = fresh("mb")
                        nodes.append(helper.make_node("Add", [mm, n.input[2]], [mb], name=fresh("bias")))
                        mm = mb
                    nodes.append(helper.make_node("Transpose", [mm], [n.output[0]], perm=[0, 2, 1], name=fresh("tro")))
                    drop.add(n.input[1]); n_rep += 1
                    continue
            nodes.append(n)
        still_used = {x for n in nodes for x in n.input}
        keep_inits = [i for i in m.graph.initializer if i.name not in drop or i.name in still_used]
        del m.graph.node[:]
        m.graph.node.extend(nodes)
        del m.graph.initializer[:]
        m.graph.initializer.extend(keep_inits)
        if not any(o.domain == "com.microsoft" for o in m.opset_import):
            m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
        onnx.save(compact(m), str(OUT / f"{g}.onnx"))
        print(f"{g}: {n_rep} MatMul/Gemm/Conv -> MatMulNBits, {n_keep} MatMul kept fp32, {(OUT / f'{g}.onnx').stat().st_size:,} B")
    shutil.copy(SRC / "spkproj.safetensors", OUT / "spkproj.safetensors")


if __name__ == "__main__":
    main()
