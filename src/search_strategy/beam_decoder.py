import numpy as np
import torch
from collections import defaultdict
from src.search_strategy.base_decoder import BaseDecoder

class BeamDecoder(BaseDecoder):
    def __init__(self, beam_size: int = 3, blank_id: int = 0):
        super().__init__(blank_id=blank_id)
        self.beam_size = beam_size

    def decode(self, log_probs):
        
        #log_probs: [T, C] log prob.
        # p_blanc, p_nonblank
        beams = {(): (0.0, -np.inf)}
        for log_prob in log_probs:
            log_prob = log_prob.detach().cpu().numpy()
            next_beams = defaultdict(lambda: (-np.inf, -np.inf))
            for prefix, (p_blank, p_nonblank) in beams.items():
                p_total = np.logaddexp(p_blank, p_nonblank)
                for token_id, token_score in enumerate(log_prob):
                    if token_id == self.blank_id:
                        old_blank, old_nonblank = next_beams[prefix]
                        next_beams[prefix] = (np.logaddexp(old_blank, p_total + token_score), old_nonblank)
                    else:
                        new_prefix = prefix + (token_id,)
                        if len(prefix) > 0 and prefix[-1] == token_id:
                            old_blank, old_nonblank = next_beams[prefix]
                            next_beams[prefix] = (old_blank, np.logaddexp(old_nonblank, p_nonblank + token_score))
                            
                            old_blank_new, old_nonblank_new = next_beams[new_prefix]
                            next_beams[new_prefix] = (old_blank_new, np.logaddexp(old_nonblank_new, p_blank + token_score))
                        else:
                            old_blank_new, old_nonblank_new = next_beams[new_prefix]
                            next_beams[new_prefix] = (old_blank_new, np.logaddexp(old_nonblank_new, p_total + token_score))
                    

            beams = dict(sorted(next_beams.items(), key=lambda item: np.logaddexp(*item[1]), reverse=True)[: self.beam_size])
        best_prefix = max(beams, key=lambda prefix: np.logaddexp(*beams[prefix]))
        return best_prefix