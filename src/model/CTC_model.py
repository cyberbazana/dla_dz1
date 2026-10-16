from torch import nn
from src.model.base_model import BaselineModel
import torch.nn.functional as F



class CTCModel(BaselineModel):

    def __init__(self, encoder, n_tokens):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.output_dim, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **kwargs):
        x, output_lengths = self.encoder(spectrogram, spectrogram_length)
        logits = self.classifier(x)
        log_probs = F.log_softmax(logits, dim=-1)

        return {
            "logits": logits,
            "log_probs": log_probs,
            "log_probs_length": output_lengths,
        }