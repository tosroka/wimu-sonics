from gradio_client import Client, handle_file
from sonics import HFAudioClassifier
import torch
from pathlib import Path
import pandas as pd
from sonics.utils.dataset import AudioDataset

# TODO: load only if local inference
# model = HFAudioClassifier.from_pretrained("awsaf49/sonics-spectttra-alpha-120s")

def get_predictions_hf(audio_files, client: Client):
    
    results = []
    for file in audio_files:
        result = client.predict(
                audio_file=handle_file(file),
                model_type="SpecTTTra-α",
                duration="120s",
                api_name="/predict"
        )
        results.append(result["label"])
    return results

def get_predictions_local(audio_files) ->list[bool]:
    predictions = []
    for audio in audio_files:
        print(audio)
        # raw = model(audio["audio"][None,:])
        # tf = torch.nn.functional.sigmoid(raw) # >0.5 if higher, then it's fake
        # predictions.append(tf.item())
    return predictions

def run_experiment(local):
    if not local:
        client = Client("awsaf49/sonics-fake-song-detection")
    results = {}
    with torch.no_grad():
        for folder in Path("../data/examples").iterdir():
            print("running",folder.name)
            all_audio = list(folder.glob("*"))[:1]
            if local:
                dataset = AudioDataset(all_audio, labels=[0]*10, random_sampling=False, max_len=16000*model_time)
                result = get_predictions_local(dataset)
            else:
                result = get_predictions_hf(all_audio, client)
            results[folder.name] = result

    print(results)
    df = pd.DataFrame(results)
    print("True if fake")
    return df


if __name__ == "__main__":
    run_experiment(None)
