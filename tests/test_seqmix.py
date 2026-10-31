"""The chunked sequence-mixing path equals the step-by-step recurrence (float64), for any chunk size and length."""
import pytest
import torch

from steertts.seqmix import SeqMixLayer, mix_forward


def _rotate(h, ang):  # h (..., N, P), ang (..., N/2)
    a, b = h[..., 0::2, :], h[..., 1::2, :]
    cs, sn = torch.cos(ang)[..., None], torch.sin(ang)[..., None]
    return torch.stack((cs * a - sn * b, sn * a + cs * b), -2).flatten(-3, -2)


def recurrence(layer, u):
    """h_t = Rot_t(a_t h_{t-1} + b_t S_{t-1}) + c_t S_t, S_t = k_t v_t^T, y_t = q_t^T h_t, one step at a time."""
    nu = layer.norm(u)
    a, b, c, k, q, v, ang = layer.coeffs(nu)
    S = k @ v.transpose(-1, -2)
    B, T, H = a.shape
    h, ys = torch.zeros_like(S[:, 0]), []
    for t in range(T):
        pre = a[:, t, :, None, None] * h
        if t:
            pre = pre + b[:, t, :, None, None] * S[:, t - 1]
        h = _rotate(pre, ang[:, t]) + c[:, t, :, None, None] * S[:, t]
        ys.append((q[:, t].transpose(-1, -2) @ h).reshape(B, -1))
    return layer.gate_and_proj(torch.stack(ys, 1), nu)


@pytest.mark.parametrize("rank", [1, 2])
def test_chunked_equals_recurrence(rank):
    torch.manual_seed(0)
    lay = SeqMixLayer(32, d_state=16, rank=rank, head_dim=8).double()
    with torch.no_grad():
        for p in lay.parameters():
            p.add_(0.3 * torch.randn_like(p))
        for T, chunk in ((40, 8), (40, 16), (37, 16), (130, 64)):
            u = torch.randn(2, T, 32, dtype=torch.float64)
            assert float((mix_forward(lay, u, chunk) - recurrence(lay, u)).abs().max()) < 1e-9


def test_causal():
    torch.manual_seed(1)
    lay = SeqMixLayer(32, d_state=16, head_dim=8).double()
    u = torch.randn(1, 40, 32, dtype=torch.float64)
    with torch.no_grad():
        y = mix_forward(lay, u, 16)
        y2 = mix_forward(lay, torch.cat([u, 50 * torch.randn(1, 9, 32, dtype=torch.float64)], 1), 16)
    assert float((y2[:, :40] - y).abs().max()) < 1e-9
