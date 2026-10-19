from torch import Tensor, nn
from torch_audiomentations import AddColoredNoise


class AddNoise(nn.Module):
    def __init__(self, min_snr_in_db=10.0, max_snr_in_db=30.0, p=0.4):
        super().__init__()
        self._aug = AddColoredNoise(
            min_snr_in_db=min_snr_in_db,
            max_snr_in_db=max_snr_in_db,
            min_f_decay=0.0,
            max_f_decay=0.0,
            p=p,
            sample_rate=16000,
        )

    def __call__(self, data):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
