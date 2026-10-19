import random

import torchaudio
from torch import Tensor, nn


class SpeedChanger(nn.Module):
    def __init__(self, sample_rate=16000, speeds=[0.85, 0.9, 1.0, 1.1, 1.15], p=0.4):
        super().__init__()
        self.sample_rate = sample_rate
        self.speeds = speeds
        self.p = p

    def __call__(self, audio):
        if random.random() >= self.p:
            return audio
        speed = random.choice(self.speeds)
        if abs(speed - 1.0) < 1e-6:
            return audio
        input_rate = max(1, int(round(self.sample_rate * speed)))
        return torchaudio.functional.resample(
            audio, orig_freq=input_rate, new_freq=self.sample_rate
        )
