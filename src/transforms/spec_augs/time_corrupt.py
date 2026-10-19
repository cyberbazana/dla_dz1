import random

import torch
from torch import Tensor, nn


class TimeCorrupt(nn.Module):
    def __init__(self, num_masks=2, p=0.4):
        super().__init__()
        self.num_masks = num_masks
        self.p = p

    def __call__(self, spectrogram):
        if random.random() >= self.p:
            return spectrogram
        T = spectrogram.shape[-1]
        max_width = int((T - 1) * 0.02)
        result = spectrogram.clone()

        for _ in range(self.num_masks):
            width = random.randint(0, max_width)
            if width == 0:
                continue
            st = random.randint(0, T - width)
            result[..., st : st + width] = 0.0

        return result
