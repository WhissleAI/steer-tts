"""seqmix.chunked_mix written as batched matmuls, for ONNX export: ORT runs multi-operand
Einsum generically (materialising (B,C,H,L,L,R,K) and running slowly); MatMul hits its GEMM kernels.
Input length must be a multiple of `chunk` (the export pads outside the graph)."""
import torch


def mix_matmul(a, keys, vals, qs, chunk: int = 64):
    B, T, H, N, K = keys.shape
    P, R = vals.shape[-2], qs.shape[-1]
    L = chunk
    C = T // L
    la = torch.log(a.clamp_min(1e-30)).reshape(B, C, L, H).permute(0, 1, 3, 2)          # (B,C,H,L)
    cl = torch.cumsum(la, 3)                                                                  # log decay from chunk start
    Bc = keys.reshape(B, C, L, H, N, K).permute(0, 1, 3, 4, 2, 5).reshape(B, C, H, N, L * K)  # (B,C,H,N,L*K)
    xc = vals.reshape(B, C, L, H, P, K).permute(0, 1, 3, 2, 5, 4).reshape(B, C, H, L * K, P)  # (B,C,H,L*K,P)
    Cc = qs.reshape(B, C, L, H, N, R).permute(0, 1, 3, 2, 5, 4).reshape(B, C, H, L * R, N)  # (B,C,H,L*R,N)
    tril = torch.tril(torch.ones(L, L, dtype=torch.bool))
    Ldec = torch.exp((cl.unsqueeze(-1) - cl.unsqueeze(-2)).masked_fill(~tril, float("-inf")))  # (B,C,H,L(t),L(s))
    G = (Cc @ Bc).reshape(B, C, H, L, R, L, K) * Ldec.reshape(B, C, H, L, 1, L, 1)
    y = G.reshape(B, C, H, L * R, L * K) @ xc                                                # intra-chunk (B,C,H,L*R,P)
    w = torch.exp(cl[..., -1:] - cl)                                                          # (B,C,H,L)
    U = (Bc.reshape(B, C, H, N, L, K) * w.reshape(B, C, H, 1, L, 1)).reshape(B, C, H, N, L * K) @ xc  # (B,C,H,N,P)
    tot = cl[..., -1]                                                                         # (B,C,H)
    ct = torch.cumsum(tot, 1)
    ctm1 = ct - tot
    stril = torch.tril(torch.ones(C, C, dtype=torch.bool), -1)
    Dc = torch.exp((ctm1.permute(0, 2, 1).unsqueeze(-1) - ct.permute(0, 2, 1).unsqueeze(-2)).masked_fill(~stril, float("-inf")))  # (B,H,C,C)
    g0 = (Dc @ U.permute(0, 2, 1, 3, 4).reshape(B, H, C, N * P)).reshape(B, H, C, N, P).permute(0, 2, 1, 3, 4)  # (B,C,H,N,P)
    yi = (Cc @ g0).reshape(B, C, H, L, R, P) * torch.exp(cl).reshape(B, C, H, L, 1, 1)        # inter-chunk
    y = y.reshape(B, C, H, L, R, P) + yi
    return y.permute(0, 1, 3, 2, 4, 5).reshape(B, T, H, R, P)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, sys.argv[1])
    from steertts_edge.seqmix import chunked_mix
    torch.manual_seed(0)
    for (Bn, T, H, N, R) in [(1, 128, 6, 32, 1), (2, 320, 4, 16, 2), (1, 64, 3, 8, 1)]:
        a = torch.rand(Bn, T, H, dtype=torch.float64) * 0.5 + 0.5
        Bf = torch.randn(Bn, T, H, N, 2 * R, dtype=torch.float64)
        xf = torch.randn(Bn, T, H, 32, 2 * R, dtype=torch.float64)
        Cf = torch.randn(Bn, T, H, N, R, dtype=torch.float64)
        ref, got = chunked_mix(a, Bf, xf, Cf, 64), mix_matmul(a, Bf, xf, Cf, 64)
        print((Bn, T, H, N, R), "max abs diff %.2e" % float((ref - got).abs().max()), "ref scale %.1f" % float(ref.abs().max()))
