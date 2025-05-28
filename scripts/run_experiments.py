"""Based on input yaml file, run experiments with different augmentations and save results."""
from typing import Optional
import yaml
import sys
from sonics.utils.dataset import AudioDataset
import make_predictions
import argparse
from wimu_sonics.augmentation import augumentation_methods, load_audio
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
import soundfile as sf
import pandas as pd
import torch
import numpy as np

DATASETS = Path("data/examples")
model_time = 120


class FakeAudioDataset:
    """ The point is to process the data and avoid writing to temp files. 
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

    def crop_or_pad(self, audio, max_len, random_sampling=True):
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
                # Crop from the beginning
                # audio = audio[:max_len]

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
            augmented_audio = self.aug_function(audio, sample_rate, **self.params)
            if self.max_len is not None:
                augmented_audio = self.crop_or_pad(augmented_audio, self.max_len, random_sampling=False)
            if self.save_to:
                audio_file_temp = self.save_to / audio_file.name
                sf.write(audio_file_temp, augmented_audio, sample_rate, format='WAV')
            yield {"audio":torch.from_numpy(augmented_audio),"target": [0]} # Dummy target because we only want to predict

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def find_datasets(base_path: Path):
    datasets = []
    for dataset in base_path.iterdir():
        if dataset.is_dir():
            datasets.append(dataset)
    return datasets

def augment_and_predict_notemp(dataset, aug_function, params):
    if not params:
        params = {}
    fake_dataset = FakeAudioDataset(dataset, aug_function=aug_function, params=params, max_len=16000*model_time)#, save_to=Path("results/500percent"))
    prediction = make_predictions.get_predictions_local(fake_dataset)
    return prediction

def augment_and_predict_with_dataset(dataset, aug_function, params):
    if not params:
        params = {}
    with TemporaryDirectory(delete=False) as temp_dir:
        print("Temporary directory created:", temp_dir)
        for audio_file in list(dataset.glob("*.*")):
            print(f"Processing {audio_file.name} with {aug_function.__name__} augmentation")
            audio, sample_rate = load_audio(audio_file)
            saved_audio = audio.copy()
            augmented_audio = aug_function(audio, sample_rate, **params)
            print("is augumented the same:", (augmented_audio == saved_audio).all())

            audio_file_temp = Path(temp_dir) / audio_file.name
            sf.write(audio_file_temp, augmented_audio, sample_rate, format='WAV')

        all_temp_files = list(Path(temp_dir).glob("*.*"))
        print(all_temp_files)
        audio_dataset = AudioDataset(all_temp_files, labels=[0]*len(all_temp_files), random_sampling=False, max_len=16000*model_time)
        print("Is from dataset the same:", (audio_dataset[0]['audio'] == torch.from_numpy(saved_audio).float()).all())
        prediction = make_predictions.get_predictions_local(audio_dataset)
        return prediction

def run_experiments(config_path):
    config = load_config(config_path)
    datasets = find_datasets(DATASETS)
    for dataset in datasets:
        print("Found dataset:", dataset.name, len(list(dataset.glob("*.*"))), "files")
    
    print("")
    print("Loaded configuration:")
    for aug in config['augmentations']:
        print(aug)
        print(aug["name"], aug["aug_function"], aug["params"])
        f = augumentation_methods[aug["aug_function"]]
        params = aug.get("params", {})
        all_datasets = {}
        for dataset in datasets:
            print(f"Running {aug['name']} on dataset {dataset.name}")
            #dataset_preds = augment_and_predict_with_dataset(dataset, f, params)
            dataset_preds = augment_and_predict_notemp(dataset, f, params)
            all_datasets[dataset.name] = dataset_preds
        # fix uneven 
        max_len = max(len(v) for v in all_datasets.values())
        padded_data = {k: v + [None] * (max_len - len(v)) for k, v in all_datasets.items()}
        df = pd.DataFrame(padded_data)
        print(df)
        output_file = Path("results") / f"{aug['name']}_results.csv"
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with different audio augmentations.")
    parser.add_argument('config_path', type=str, nargs='?', default="configs/paper.yaml", help='Path to the YAML configuration file.')
    args = parser.parse_args()

    run_experiments(args.config_path)
    