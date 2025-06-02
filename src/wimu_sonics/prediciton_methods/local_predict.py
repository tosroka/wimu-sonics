from sonics import HFAudioClassifier
from sonics.utils.dataset import AudioDataset
import torch

model = None

def get_predictions_local(audio_files: AudioDataset) ->list[bool]:
    global model # not pretty
    if not model:
        model = HFAudioClassifier.from_pretrained("awsaf49/sonics-spectttra-alpha-120s")

    predictions = []
    for audio in audio_files:
        print(audio)
        raw = model(audio["audio"][None,:])
        tf = torch.nn.functional.sigmoid(raw) # >0.5 if higher, then it's fake
        predictions.append(tf.item())
    return predictions