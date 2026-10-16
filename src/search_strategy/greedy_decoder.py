import torch
from src.search_strategy.base_decoder import BaseDecoder



class GreedyDecoder(BaseDecoder):
    def __init__(self, blank_id: int = 0):
        super().__init__(blank_id=blank_id)

    def decode(self, log_probs):
        frame_ids = log_probs.argmax(dim=-1).detach().cpu().tolist()

        result = []
        previous = None

        for token_id in frame_ids:
            if token_id != self.blank_id and token_id != previous:
                result.append(token_id)
            previous = token_id

        return tuple(result)
