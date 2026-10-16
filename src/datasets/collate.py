import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    audio = pad_sequence([item["audio"].squeeze(0) for item in dataset_items], batch_first=True,
                         padding_value=0.0).unsqueeze(1)  # [B, 1, T]

    spectrograms = [item["spectrogram"].transpose(0, 1) for item in dataset_items]
    spectrogram = pad_sequence(spectrograms, batch_first=True, padding_value=0.0).transpose(1, 2)  # [B, F, T]

    text_encoded = pad_sequence([item["text_encoded"].reshape(-1) for item in dataset_items],
                        batch_first=True, padding_value=0)

    return {
        "audio": audio,
        "spectrogram": spectrogram,
        "text": [item["text"] for item in dataset_items],
        "text_encoded": text_encoded,
        "audio_path": [item["audio_path"] for item in dataset_items],
        "audio_length": torch.tensor([item["audio"].shape[-1] for item in dataset_items], dtype=torch.long),
        "spectrogram_length": torch.tensor([item["spectrogram"].shape[-1] for item in dataset_items], dtype=torch.long),
        "text_encoded_length": torch.tensor([item["text_encoded"].numel() for item in dataset_items], dtype=torch.long)    
    }
