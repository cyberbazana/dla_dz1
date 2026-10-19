from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_dist

# lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class CERMetric(BaseMetric):
    def __init__(self, text_encoder, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.decoder = decoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        total_err = 0
        total_chars = 0
        for probs, length, target_text in zip(log_probs, log_probs_length, text):
            length = int(length.item())
            pred_tokens = self.decoder.decode(probs[:length])
            pred_text = self.text_encoder.decode(pred_tokens)

            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.normalize_text(pred_text)

            target_chars = list(target_text)
            pred_chars = list(pred_text)
            total_err += calc_dist(target_chars, pred_chars)
            total_chars += len(target_chars)

        return total_err / max(total_chars, 1), total_chars
