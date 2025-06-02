"""Run the sonics model in different ways."""

from typing import Literal
import torch
from pathlib import Path
import pandas as pd
from sonics.utils.dataset import AudioDataset

from wimu_sonics.prediciton_methods.local_predict import get_predictions_local
from wimu_sonics.prediciton_methods.huggingface_api_predict import get_predictions_hf
from wimu_sonics.prediciton_methods.local_torch_predict import get_predictions_torch
model_time = 120 # it should be possible to extract this from a model instance

test_assets = Path("test/assets")

def run_experiment(path: Path, type: Literal['torch','hf_local','hf_api']='torch'):
    results = {}
    with torch.no_grad():
        for folder in path.iterdir():
            print("running",folder.name)
            all_audio = list(folder.glob("*"))
            if type=='hf_local':
                dataset = AudioDataset(all_audio, labels=[0]*10, random_sampling=False, max_len=16000*model_time)
                r = get_predictions_local(dataset)
            elif type=='hf_api':
                r = get_predictions_hf(all_audio)
            elif type=='torch':
                dataset = AudioDataset(all_audio, labels=[0]*10, random_sampling=False, max_len=16000*model_time)
                r = get_predictions_torch(dataset)
            results[folder.name] = r

    print(results)
    df = pd.DataFrame(results)
    return df

def test_local_torch():
    df = run_experiment(test_assets, 'torch')
    print("Torch results:")
    print(df)

def test_local_hf():
    df = run_experiment(test_assets, 'hf_local')
    print("Local hf results:")
    print(df)

def test_api_hf():
    df = run_experiment(test_assets, 'hf_api')
    print("API hf results:")
    print(df)