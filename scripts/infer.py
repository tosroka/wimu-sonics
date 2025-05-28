from pathlib import Path
import librosa
import torch
import pandas as pd

from sonics import HFAudioClassifier
from sonics.utils.dataset import AudioDataset

from sonics.models.model import AudioClassifier

from sonics.utils.config import dict2cfg
import yaml
import numpy as np

model = HFAudioClassifier.from_pretrained("awsaf49/sonics-spectttra-alpha-120s")
max_time = model.config.audio.max_time
torch_model = "models/pytorch_model.bin"
cfg_path = "test_config.yaml"
device = "cuda:1"

from gradio_client import Client, handle_file
def get_predictions_hf(audio_files, client: Client):

    results = []
    for file in audio_files:
        while True:
            try:
                result = client.predict(
                        audio_file=handle_file(file),
                        model_type="SpecTTTra-α",
                        duration="120s",
                        api_name="/predict"
                )
                print(result)
                results.append(result["label"])
                break
            except:
                pass
    return results

def get_middle_chunk(audio):
    chunk_samples = int(max_time * 16000)
    total_chunks = len(audio) // chunk_samples
    middle_chunk_idx = total_chunks // 2

    # Extract middle chunk
    start = middle_chunk_idx * chunk_samples
    end = start + chunk_samples
    chunk = audio[start:end]

    if len(chunk) < chunk_samples:
        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
    return chunk

def get_predictions_local_hf(audio_files: list[Path]) ->list[bool]:
    predictions = []
    for audio in audio_files:
        audio, sr = librosa.load(audio, sr=16000)
        chunk = get_middle_chunk(audio)
        chunk = torch.from_numpy(chunk)
        raw = model(chunk[None,:])
        pred = torch.nn.functional.sigmoid(raw) # if higher, then it's fake
        predictions.append("Fake" if pred.item()>0.5 else "Real")
        print(pred.item())
    return predictions

def get_predictions_local_torch(audio_files: list[Path]) ->list[bool]:
    """Mostly taken from test.py"""
    import torch

    with open(cfg_path,"rb") as f:
        dict_ = yaml.safe_load(f)
    cfg = dict2cfg(dict_)

    d = torch.device(device)
    with open(torch_model, "rb") as f:
        weights = torch.load(f)
    model = AudioClassifier(cfg)
    model.eval()
    model.load_state_dict(weights)
    model.to(d)

    predictions = []
    for audio in audio_files:
        audio, sr = librosa.load(audio, sr=16000)
        chunk = get_middle_chunk(audio)
        chunk = torch.from_numpy(chunk).to(device)
        raw = model(chunk[None,:])
        pred = torch.nn.functional.sigmoid(raw) # if higher, then it's fake
        predictions.append("Fake" if pred.item()>0.5 else "Real")
        print(pred.item())
    return predictions
        
def run_experiment(local):
    if not local:
        client = Client("awsaf49/sonics-fake-song-detection")
    results = {}
    with torch.no_grad():
        for folder in Path("data/examples").iterdir():
            print("running",folder.name)
            all_audio = list(folder.glob("*.*"))[:1]
            if local:
                result = get_predictions_local_hf(all_audio)
            else:
                result = get_predictions_hf(all_audio, client)
            if len(result):
                results[folder.name] = result

    print(results)
    df = pd.DataFrame(results)
    print("True if fake")

    return df

df = run_experiment(local=True)

for c in df.columns:
    print("---- %s ---" % c)
    for label,counts in df[c].value_counts().items():
        print(label,counts)
