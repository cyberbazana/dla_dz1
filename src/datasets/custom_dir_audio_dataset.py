from pathlib import Path

import soundfile as sf

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        root = Path(data_dir)
        audio_dir = root / "audio"
        transcription_dir = root / "transcriptions"
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                entry["text"] = ""
                info = sf.info(str(path))
                entry["audio_len"] = info.frames / info.samplerate
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()

            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
