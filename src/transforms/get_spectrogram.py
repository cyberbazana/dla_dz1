import torch
import torchaudio
from torch import Tensor, nn


class Spectrogram(nn.Module):
    def __init__(self, sample_rate = 16000, n_fft = 512, hop_length = 160, n_mels = 80, win_length = 400, f_min = 0.0, f_max = None):
        super().__init__()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels
        )

    def forward(self, waveform):
        waveform = waveform.squeeze(0)
        mel_power = self.mel(waveform) 
        log_mel = torch.log(mel_power.clamp_min(1e-6))
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True).clamp_min(1e-5)
        return (log_mel - mean) / std