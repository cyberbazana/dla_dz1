import argparse
from pathlib import Path

from src.metrics.utils import calc_dist
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def read_text(path):
    return path.read_text(encoding="utf-8").strip()


def normalize_text(text):
    return CTCTextEncoder.normalize_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()

    refs = {p.name: p for p in args.target.glob("*.txt")}
    preds = {p.name: p for p in args.predictions.glob("*.txt")}

    missing = refs.keys() - preds.keys()
    if missing:
        parser.error(f"No predictions for: {', '.join(missing[:10])}")
    if not refs:
        parser.error(f"No files in {args.ground_truth}")

    total_word_err = 0
    total_words = 0
    total_char_err = 0
    total_chars = 0

    for filename, ref_path in refs.items():
        target = normalize_text(read_text(refs[filename]))
        predicted = normalize_text(read_text(preds[filename]))

        target_words = target.split()
        predicted_words = predicted.split()

        total_word_err += calc_dist(target_words, predicted_words)
        total_words += len(target_words)

        target_chars = list(target)
        predicted_chars = list(predicted)
        total_char_err += calc_dist(target_chars, predicted_chars)
        total_chars += len(target_chars)

    wer = total_word_err / max(total_words, 1)
    cer = total_char_err / max(total_chars, 1)

    print(f"WER: {wer}")
    print(f"CER: {cer}")


if __name__ == "__main__":
    main()
