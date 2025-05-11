from pathlib import Path
import librosa
import soundfile as sf
from pydub import AudioSegment

def resample_audio(input_path, output_path, target_sr, output_format="wav"):
    # Load audio with librosa (automatically resamples)
    audio, sr = librosa.load(str(input_path), sr=target_sr)

    if output_format == "wav":
        sf.write(str(output_path.with_suffix(".wav")), audio, target_sr)

    elif output_format == "mp3":
        # Save to a temporary wav file first
        temp_wav = output_path.with_suffix(".wav")
        sf.write(str(temp_wav), audio, target_sr)

        # Convert wav to mp3 using pydub
        sound = AudioSegment.from_wav(str(temp_wav))
        sound.export(str(output_path.with_suffix(".mp3")), format="mp3")

        # Remove temporary wav
        temp_wav.unlink()
    else:
        raise ValueError(f"Unsupported format: {output_format}")

def process_directory(input_dir, output_dir, target_sr, fmt = "mp3"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for input_file in input_dir.rglob("*.*"):
        print(input_file)
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing {input_file} -> {output_file.with_suffix("."+fmt)}")
        resample_audio(input_file, output_file, target_sr, fmt)
        print("done")

if __name__ == "__main__":
    target_sample_rate = 16000
    input_directory = "data/examples" 
    output_directory = f"data/examples_{target_sample_rate}"
    fmt = "mp3"

    process_directory(input_directory, output_directory, target_sample_rate, fmt)