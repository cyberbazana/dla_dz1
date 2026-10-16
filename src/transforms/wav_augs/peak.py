import torch
from torch import Tensor, nn

class PeakNormalize(nn.Module):
    def __init__(self, peak, eps):
        super().__init__()
        self.peak = peak
        self.eps = eps

    def forward(self, audio: Tensor) -> Tensor:
        max_amplitude = audio.abs().amax()
        if max_amplitude < self.eps:
            return audio
        return audio * (self.peak / max_amplitude)