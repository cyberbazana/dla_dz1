import torch

class BaseDecoder:
    def __init__(self, blank_id = 0):
        self.blank_id = blank_id

    def decode(self, log_probs):
        raise NotImplementedError("Decode method must be implemented in subclasses.")
