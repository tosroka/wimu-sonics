from typing import Optional
import numpy as np
from pathlib import Path

import torch
from wimu_sonics.augmentation import load_audio
import soundfile as sf

class FakeAudioDataset:
    """Dataset similar to the one from sonics repo.
    
    The point is to process the data and avoid writing to temp files. 
    Iterating over the dataset yields augmented audio that are loaded on the go."""

    def __init__(self, dataset_path: Path, aug_function: callable =None, params: dict=None, limit_files: int=None, max_len: int =None, save_to : Optional[Path]=None):
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
        if not dataset_path.is_dir():
            raise NotADirectoryError(f"Dataset path {dataset_path} is not a directory.")
        if aug_function is not None and not callable(aug_function):
            raise ValueError(f"Augmentation function {aug_function} is not callable.")
        
        self.aug_function = aug_function
        self.params = params if params is not None else {}
        self.dataset_path = dataset_path
        self.limit_files = limit_files
        self.max_len = max_len
        self.save_to = save_to / dataset_path.name if save_to is not None else None
        if self.save_to is not None:
            if not self.save_to.exists():
                self.save_to.mkdir(parents=True, exist_ok=True)
            elif not self.save_to.is_dir():
                raise NotADirectoryError(f"Save path {self.save_to} is not a directory.")

    @staticmethod
    def crop_or_pad(audio, max_len, random_sampling=True):
        audio_len = audio.shape[0]
        if random_sampling:
            diff_len = abs(max_len - audio_len)
            if audio_len < max_len:
                pad1 = np.random.randint(0, diff_len)
                pad2 = diff_len - pad1
                audio = np.pad(audio, (pad1, pad2), mode="constant")
            elif audio_len > max_len:
                idx = np.random.randint(0, diff_len)
                audio = audio[idx : (idx + max_len)]
        else:
            if audio_len < max_len:
                audio = np.pad(audio, (0, max_len - audio_len), mode="constant")
            elif audio_len > max_len:
                # Crop from 3/4 of the audio (of the chunk remaning after subtracting max_len!!!)
                # eq: l = (3x + t + x) => idx = 3x = (l - t) / 4 * 3
                idx = int((audio_len - max_len) / 4 * 3)
                audio = audio[idx : (idx + max_len)]
        return audio

    def __iter__(self):
        files = self.dataset_path.glob("*.*")
        if self.limit_files is not None:
            files = list(files)[:self.limit_files]
        for audio_file in files:
            audio, sample_rate = load_audio(audio_file)
            augmented_audio = self.aug_function(audio, **self.params)
            if self.max_len is not None:
                augmented_audio = FakeAudioDataset.crop_or_pad(augmented_audio, self.max_len, random_sampling=False)
            if self.save_to:
                audio_file_temp = self.save_to / audio_file.name
                sf.write(audio_file_temp, augmented_audio, sample_rate, format='WAV')
            yield {"audio":torch.from_numpy(augmented_audio),"target": [0]} # Dummy target because we only want to predict