from gradio_client import Client, handle_file
import numpy as np

client = None

def get_middle_chunk(audio):
    """This function would reproduce api responses if exact match is needed.
    Locally we pick different part of the waveform."""
    chunk_samples = int(120 * 16000)
    total_chunks = len(audio) // chunk_samples
    middle_chunk_idx = total_chunks // 2

    # Extract middle chunk
    start = middle_chunk_idx * chunk_samples
    end = start + chunk_samples
    chunk = audio[start:end]

    if len(chunk) < chunk_samples:
        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
    return chunk

def get_predictions_hf(audio_files): 
    global client # again not pretty
    if not client:
        client = Client("awsaf49/sonics-fake-song-detection")
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