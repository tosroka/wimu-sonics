
from sonics.models.model import AudioClassifier

from sonics.utils.config import dict2cfg

import torch

from sonics.models.hf_model import HFAudioClassifier
from sonics.utils.dataset import AudioDataset

import librosa

import yaml

model = HFAudioClassifier.from_pretrained("awsaf49/sonics-spectttra-alpha-120s")
max_time = model.config.audio.max_time
torch_model = "pytorch_model.bin" # download this if you want to use
cfg_path = "sonics/configs/spectttra_f1t3-120s.yaml"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_predictions_torch(audio_files: AudioDataset) ->list[bool]:
    """Mostly taken from test.py"""
    import torch

    with open(cfg_path,"rb") as f:
        dict_ = yaml.safe_load(f)
    cfg = dict2cfg(dict_)

    with open(torch_model, "rb") as f:
        weights = torch.load(f)
    model = AudioClassifier(cfg)
    model.eval()
    model.load_state_dict(weights)
    model.to(device)

    predictions = []
    for sample in audio_files:
        print(sample)
        waveform = sample['audio']
        chunk = waveform.to(device)
        raw = model(chunk[None,:])
        pred = torch.nn.functional.sigmoid(raw)
        predictions.append(pred.item())
    return predictions