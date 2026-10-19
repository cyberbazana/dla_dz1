import gzip
import shutil
import urllib.request
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase

import numpy as np
from pyctcdecode import build_ctcdecoder

from src.search_strategy.base_decoder import BaseDecoder

LM_URL = "http://openslr.elda.org/resources/11/3-gram.pruned.3e-7.arpa.gz"


class LMDecoder(BaseDecoder):
    def __init__(
        self,
        beam_size,
        alpha,
        beta,
        blank_id=0.0,
        lm_path="data/lm/3-gram.pruned.3e-7.arpa",
        lm_url=LM_URL,
    ):
        super().__init__(blank_id=blank_id)
        self.beam_size = beam_size
        self.alpha = alpha
        self.beta = beta
        self.vocab = [""] + list(ascii_lowercase + " ")
        self.lm_vocab = [""] + list(ascii_uppercase + " ")
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        lm_path_obj = Path(lm_path).expanduser().resolve()
        if not lm_path_obj.exists():
            self._download_lm(lm_url=lm_url, output_path=lm_path_obj)

        self.lm_path = str(lm_path_obj)
        self.decoder = build_ctcdecoder(
            labels=self.lm_vocab,
            kenlm_model_path=self.lm_path,
            alpha=self.alpha,
            beta=self.beta,
        )

    @staticmethod
    def _download_lm(lm_url, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gz_path = output_path.with_suffix(output_path.suffix + ".gz")
        print(f"KenLM model not found. Downloading from:\n  {lm_url}")
        try:
            urllib.request.urlretrieve(lm_url, gz_path)
            print(f"Unpacking KenLM model to:\n  {output_path}")
            with gzip.open(gz_path, "rb") as src, output_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the LibriSpeech KenLM model automatically. "
                f"Download {lm_url} manually, unpack it, and place it at "
                f"{output_path}."
            ) from exc
        finally:
            gz_path.unlink(missing_ok=True)

    def decode_text(self, log_probs):
        logits = log_probs.detach().float().cpu().numpy()
        return self.decoder.decode(logits, beam_width=self.beam_size).strip().lower()

    def decode(self, log_probs):
        text = self.decode_text(log_probs)
        return tuple(self.char2ind[ch] for ch in text if ch in self.char2ind)
