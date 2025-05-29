"""Based on input yaml file, run experiments with different augmentations and save results."""
from wimu_sonics.dataset import FakeAudioDataset
import yaml
import sys
import make_predictions
import argparse
from wimu_sonics.augmentation import augmentation_methods, load_audio
from wimu_sonics.special_augmentation import special_augmentation_methods
from pathlib import Path
import pandas as pd

DATASETS = Path("data/examples")
model_time = 120

augmentation_methods.update(special_augmentation_methods)

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

def augment_and_predict_notemp(dataset, aug_function, params, experiment_name: str):
    if not params:
        params = {}
    fake_dataset = FakeAudioDataset(dataset, aug_function=aug_function, params=params, max_len=16000*model_time, save_to=Path(f"results/{experiment_name}"))
    prediction = make_predictions.get_predictions_local(fake_dataset)
    return prediction

# def augment_and_predict_with_dataset(dataset, aug_function, params):
#     if not params:
#         params = {}
#     with TemporaryDirectory(delete=False) as temp_dir:
#         print("Temporary directory created:", temp_dir)
#         for audio_file in list(dataset.glob("*.*")):
#             print(f"Processing {audio_file.name} with {aug_function.__name__} augmentation")
#             audio, sample_rate = load_audio(audio_file)
#             saved_audio = audio.copy()
#             augmented_audio = aug_function(audio, sample_rate, **params)
#             print("is augumented the same:", (augmented_audio == saved_audio).all())

#             audio_file_temp = Path(temp_dir) / audio_file.name
#             sf.write(audio_file_temp, augmented_audio, sample_rate, format='WAV')

#         all_temp_files = list(Path(temp_dir).glob("*.*"))
#         print(all_temp_files)
#         audio_dataset = AudioDataset(all_temp_files, labels=[0]*len(all_temp_files), random_sampling=False, max_len=16000*model_time)
#         print("Is from dataset the same:", (audio_dataset[0]['audio'] == torch.from_numpy(saved_audio).float()).all())
#         prediction = make_predictions.get_predictions_local(audio_dataset)
#         return prediction

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
        f = augmentation_methods[aug["aug_function"]]
        params = aug.get("params", {})
        all_datasets = {}
        for dataset in datasets:
            print(f"Running {aug['name']} on dataset {dataset.name}")
            dataset_preds = augment_and_predict_notemp(dataset, f, params, aug["name"])
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
    