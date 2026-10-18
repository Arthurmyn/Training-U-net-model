"""A standard U-Net (Ronneberger et al. 2015) with batch normalisation and padded convolutions."""
from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    """(Conv 3x3 - BatchNorm - ReLU) twice."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """U-Net with 4 down/up steps. Input (B, 3, H, W) with H, W divisible by 16; output logits (B, 1, H, W)."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, n_classes: int = 1) -> None:
        super().__init__()
        c = base_channels
        self.enc = nn.ModuleList([DoubleConv(in_channels, c), DoubleConv(c, 2 * c), DoubleConv(2 * c, 4 * c), DoubleConv(4 * c, 8 * c)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(8 * c, 16 * c)
        self.up = nn.ModuleList([nn.ConvTranspose2d(16 * c, 8 * c, 2, stride=2), nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2),
                                 nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2), nn.ConvTranspose2d(2 * c, c, 2, stride=2)])
        self.dec = nn.ModuleList([DoubleConv(16 * c, 8 * c), DoubleConv(8 * c, 4 * c), DoubleConv(4 * c, 2 * c), DoubleConv(2 * c, c)])
        self.head = nn.Conv2d(c, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
