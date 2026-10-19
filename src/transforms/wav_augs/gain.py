import torch_audiomentations
from torch import Tensor, nn


class Gain(nn.Module):
    def __init__(self, min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.4):
        super().__init__()
        self._aug = torch_audiomentations.Gain(min_gain_in_db, max_gain_in_db, p=p)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
