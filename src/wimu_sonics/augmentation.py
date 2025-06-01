import librosa
import numpy as np
import soundfile as sf
from audiomentations import Compose, BandStopFilter, TimeMask, AddGaussianNoise, TanhDistortion, BandPassFilter, BitCrush, \
                            Gain, PitchShift, TimeStretch, AddShortNoises, PolarityInversion, Aliasing, GainTransition, \
                            HighPassFilter, LowPassFilter, HighShelfFilter, LowShelfFilter, Limiter, Mp3Compression, SevenBandParametricEQ
from scipy.signal import fftconvolve
from typing import Literal
import io
import subprocess

def load_audio(audio, sr=16000):
    y, sr = librosa.load(audio, sr=sr)
    return y, sr

def no_augment(audio, sample_rate):
    return audio

def apply_frequency_masking(audio, sample_rate, center_freq, bandwidth_fraction, rolloff=6):
    augmenter = Compose([
        BandStopFilter(
            min_center_freq=center_freq,
            max_center_freq=center_freq,
            min_bandwidth_fraction=bandwidth_fraction,
            max_bandwidth_fraction=bandwidth_fraction,
            min_rolloff=rolloff,
            max_rolloff=rolloff,
            p=1.0
        )
    ])
    augmented = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented

def apply_time_masking(audio, sample_rate, min_band_part, max_band_part):
    augmenter = Compose([
        TimeMask(min_band_part=min_band_part, max_band_part=max_band_part, fade_duration=0.1, p=1.0)
    ])
    augmented = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented

def apply_mixup(audio1, audio2, proportion=0.5):
    min_len = min(len(audio1), len(audio2))
    audio1, audio2 = audio1[:min_len], audio2[:min_len]

    mixed_audio = proportion * audio1 + (1 - proportion) * audio2
    return mixed_audio

def apply_volume_increase(audio, sample_rate, db_gain):
    augmenter = Compose([
        Gain(min_gain_db=db_gain, max_gain_db=db_gain, p=1.0)
    ])
    augmented = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented

def apply_speed_increase(audio, sample_rate, rate):
    augmenter = Compose([
        TimeStretch(min_rate=rate, max_rate=rate, p=1.0, leave_length_unchanged=False)
    ])
    augmented = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented

def apply_pitch_shift(audio, sample_rate, semitones):
    augmenter = Compose([
        PitchShift(min_semitones=semitones, max_semitones=semitones, p=1.0)
    ])
    augmented = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented

def apply_white_noise(audio, sample_rate, amplitude):
    augmenter = Compose([
        AddGaussianNoise(min_amplitude=amplitude, max_amplitude=amplitude, p=1.0),
    ])
    augmented_audio = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_audio

def apply_tanh_distortion(audio, sample_rate, min_distortion=0.01, max_distortion=0.7, p=1.0):
    augmenter = Compose([
        TanhDistortion(
            min_distortion=min_distortion,
            max_distortion=max_distortion,
            p=p
        )
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound

def apply_band_pass_filter(audio, sample_rate, min_center_freq=100.0, max_center_freq=6000.0, p=1.0):
    augmenter = Compose([
        BandPassFilter(min_center_freq=min_center_freq, max_center_freq=max_center_freq, p=p)
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound

def apply_bit_crush(audio, sample_rate, bit_depth, p=1.0):
    augmenter = Compose([
        BitCrush(min_bit_depth=bit_depth, max_bit_depth=bit_depth, p=p)
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound

def apply_vibrato(audio, sample_rate, vibrato_freq=5, vibrato_magnitude=0.003):
    t = np.arange(len(audio)) / sample_rate
    modulation = vibrato_magnitude * np.sin(2 * np.pi * vibrato_freq * t)
    indices = np.arange(len(audio)) + modulation * sample_rate
    
    indices = np.clip(indices, 0, len(audio) - 1)
    vibrato_audio = np.interp(indices, np.arange(len(audio)), audio)
    
    return vibrato_audio

def apply_reverb(audio, sample_rate, reverb_decay=0.5, reverb_delay_ms=50):
    delay_samples = int(sample_rate * reverb_delay_ms / 1000)
    impulse_response = np.zeros(delay_samples + 1)
    impulse_response[0] = 1.0
    impulse_response[-1] = reverb_decay
    
    reverbed = fftconvolve(audio, impulse_response, mode='full')[:len(audio)]
    
    reverbed = reverbed / np.max(np.abs(reverbed))
    return reverbed

def apply_short_noise(audio, sample_rate, noise_folder_path, min_snr_db=3.0, max_snr_db=30.0, min_time_between_sounds=2.0, max_time_between_sounds=8.0, p=1.0):
    augmenter = Compose([
        AddShortNoises(
        sounds_path=noise_folder_path,
        min_snr_db=min_snr_db,
        max_snr_db=max_snr_db,
        noise_rms="relative_to_whole_input",
        min_time_between_sounds=min_time_between_sounds,
        max_time_between_sounds=max_time_between_sounds,
        noise_transform=PolarityInversion(),
        p=p
    )
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound


def apply_aliasing(audio, sample_rate, target_sample_rate):
    augmenter = Compose([
        Aliasing(min_sample_rate=target_sample_rate, max_sample_rate=target_sample_rate, p=1)
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound

def apply_gain_transition(audio, sample_rate, min_gain_db=-6.0, max_gain_db=6.0, p=1.0):
    augmenter = Compose([
        GainTransition(min_gain_db=min_gain_db, max_gain_db=max_gain_db, p=p)
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound


def apply_high_pass_filter(audio, sample_rate, cutoff_freq):
    augmenter = Compose([
        HighPassFilter(min_cutoff_freq=cutoff_freq, max_cutoff_freq=cutoff_freq, p=1)
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound

def apply_low_pass_filter(audio, sample_rate, cutoff_freq):
    augmenter = Compose([
        LowPassFilter(min_cutoff_freq=cutoff_freq, max_cutoff_freq=cutoff_freq, p=1)
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound

def apply_high_shelf_filter(audio, sample_rate, min_gain_db=-12.0, max_gain_db=12.0, min_cutoff_freq=3000.0, max_cutoff_freq=6000.0, p=1.0):
    augmenter = Compose([
        HighShelfFilter(
            min_gain_db=min_gain_db,
            max_gain_db=max_gain_db,
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            p=p
        )
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound

def apply_low_shelf_filter(audio, sample_rate, min_gain_db=-12.0, max_gain_db=12.0, min_cutoff_freq=100.0, max_cutoff_freq=500.0, p=1.0):
    augmenter = Compose([
        LowShelfFilter(
            min_gain_db=min_gain_db,
            max_gain_db=max_gain_db,
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            p=p
        )
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound

def apply_limiter(audio, sample_rate, threshold_db=-1.0, p=1.0):
    augmenter = Compose([
        Limiter(threshold_db=threshold_db, p=p)
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound

def apply_mp3_compression(audio, sample_rate, bitrate=8):
    augmenter = Compose([
        Mp3Compression(min_bitrate=bitrate, max_bitrate=bitrate, p=1)
    ])
    augmented_audio = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_audio

def apply_EQ(audio, sample_rate, db_gain):
    augmenter = Compose([
        SevenBandParametricEQ(db_gain, db_gain,p=1)
    ])
    augmented_audio = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_audio

# Compression

def decompress_ogg_to_audio(ogg_data: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Decompress OGG (Vorbis) binary data to NumPy waveform and sample rate.

    Args:
        ogg_data (np.ndarray): Compressed audio in OGG format as uint8 array

    Returns:
        tuple: (audio waveform as np.ndarray, sample rate)
    """
    process = subprocess.run(
        ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', 'pipe:1'],
        input=ogg_data.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    wav_data = io.BytesIO(process.stdout)
    audio, sr = sf.read(wav_data)
    return audio

# TODO: ONLY OGG BECAUSE BROKEN
def compress_audio_codec(audio: np.ndarray, sample_rate: int,
    codec: Literal[
        "mp3",
        "aac",
        "g723_1",
        "mp2",
        "opus",
        "vorbis",
    ] = "mp3"
    ):

    raw_audio = io.BytesIO()
    sf.write(raw_audio, audio, sample_rate, format='WAV')
    raw_audio.seek(0)

    # Step 2: Call ffmpeg subprocess to compress to OGG
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',                # Overwrite output if needed
        '-i', 'pipe:0',      # Input from stdin
        '-f', codec,         # Output format
        '-acodec', 'libvorbis',  # Use Vorbis codec
        'pipe:1'             # Output to stdout
    ]

    try:
        result = subprocess.run(
            ffmpeg_cmd,
            input=raw_audio.read(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("FFmpeg error:", e.stderr.decode())
        raise

    # Step 3: Convert output to numpy array
    ogg_bytes = result.stdout
    compressed = np.frombuffer(ogg_bytes, dtype=np.uint8)
    return decompress_ogg_to_audio(ogg_data=compressed)


augmentation_methods = {
    f.__qualname__: f for f in [
        apply_frequency_masking,
        apply_time_masking,
        apply_mixup,
        apply_volume_increase,
        apply_speed_increase,
        apply_pitch_shift,
        apply_white_noise,
        apply_tanh_distortion,
        apply_band_pass_filter,
        apply_bit_crush,
        apply_vibrato,
        apply_reverb,
        apply_short_noise,
        apply_aliasing,
        apply_gain_transition,
        apply_high_pass_filter,
        apply_low_pass_filter,
        apply_high_shelf_filter,
        apply_low_shelf_filter,
        apply_limiter,
        apply_mp3_compression,
        apply_EQ,
        compress_audio_codec,
        no_augment
    ]
}