from gradio_client import Client, handle_file

client = None

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