"""Vocos-style vocoder (ConvNeXt backbone + iSTFT head) without the `vocos` package: the same modules and
operations as vocos.Vocos(VocosBackbone, ISTFTHead), so its state dict loads unchanged (feature_extractor.* is not
needed at inference). Input: 100-band natural-log mel at 24 kHz, hop 256; output: waveform."""
import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, inter):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, inter)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(inter, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r = x
        x = self.norm(self.dwconv(x).transpose(1, 2))
        x = self.gamma * self.pwconv2(self.act(self.pwconv1(x)))
        return r + x.transpose(1, 2)


class Backbone(nn.Module):
    def __init__(self, n_mels, dim, inter, layers):
        super().__init__()
        self.embed = nn.Conv1d(n_mels, dim, 7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList([ConvNeXtBlock(dim, inter) for _ in range(layers)])
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
        for b in self.convnext:
            x = b(x)
        return self.final_layer_norm(x.transpose(1, 2))


class ISTFTHeadOut(nn.Module):
    def __init__(self, dim, n_fft):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)


class Vocoder(nn.Module):
    def __init__(self, dim, inter, layers, n_mels=100, n_fft=1024, hop=256, padding="center"):
        super().__init__()
        self.backbone = Backbone(n_mels, dim, inter, layers)
        self.head = ISTFTHeadOut(dim, n_fft)
        self.n_fft, self.hop, self.padding = n_fft, hop, padding
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def istft(self, S):
        if self.padding == "center":
            return torch.istft(S, self.n_fft, self.hop, self.n_fft, self.window, center=True)
        pad = (self.n_fft - self.hop) // 2   # vocos ISTFT(padding="same")
        T = S.shape[-1]
        y = torch.fft.irfft(S, self.n_fft, dim=1, norm="backward") * self.window[None, :, None]
        size = (T - 1) * self.hop + self.n_fft
        fold = lambda z: torch.nn.functional.fold(z, output_size=(1, size), kernel_size=(1, self.n_fft),
                                                  stride=(1, self.hop))
        y = fold(y)[:, 0, 0, pad:-pad]
        env = fold(self.window.square().expand(1, T, -1).transpose(1, 2)).squeeze()[pad:-pad]
        return y / env

    @torch.no_grad()
    def decode(self, mel):
        x = self.head.out(self.backbone(mel)).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        return self.istft(mag * (torch.cos(p) + 1j * torch.sin(p)))
