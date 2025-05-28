import librosa
import numpy as np
import soundfile as sf
from audiomentations import Compose, BandStopFilter, TimeMask, AddGaussianNoise, TanhDistortion, BandPassFilter, BitCrush, \
                            Gain, PitchShift, TimeStretch, AddShortNoises, PolarityInversion, Aliasing, GainTransition, \
                            HighPassFilter, LowPassFilter, HighShelfFilter, LowShelfFilter, Limiter, Mp3Compression
from scipy.signal import fftconvolve
import subprocess
from typing import Literal

def load_audio(audio):
    y, sr = librosa.load(audio, sr=16000)
    return y, sr

def no_augment(audio, sample_rate):
    return audio

def apply_frequency_masking(audio, sample_rate):
    augmenter = Compose([
        BandStopFilter(
            min_center_freq=200.0,
            max_center_freq=4000.0,
            min_bandwidth_fraction=0.5,
            max_bandwidth_fraction=1.99,
            p=1.0
        )
    ])
    augmented = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented

def apply_time_masking(audio, sample_rate):
    augmenter = Compose([
        TimeMask(min_band_part=0.05, max_band_part=0.2, p=1.0)
    ])
    augmented = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented

def apply_mixup(audio1, audio2, proportion=0.5):
    min_len = min(len(audio1), len(audio2))
    audio1, audio2 = audio1[:min_len], audio2[:min_len]

    mixed_audio = proportion * audio1 + (1 - proportion) * audio2
    return mixed_audio

def apply_volume_increase(audio, sample_rate, min_gain_in_db=6.0, max_gain_in_db=6.0):
    augmenter = Compose([
        Gain(min_gain_in_db=min_gain_in_db, max_gain_in_db=max_gain_in_db, p=1.0)
    ])
    augmented = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented

def apply_speed_increase(audio, sample_rate, min_rate=1.1, max_rate=1.3):
    augmenter = Compose([
        TimeStretch(min_rate=min_rate, max_rate=max_rate, p=1.0, leave_length_unchanged=False)
    ])
    augmented = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented

def apply_pitch_shift(audio, sample_rate, min_semitones=-4, max_semitones=4):
    augmenter = Compose([
        PitchShift(min_semitones=min_semitones, max_semitones=max_semitones, p=1.0)
    ])
    augmented = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented

def apply_white_noise(audio, sample_rate, min_amplitude=0.001, max_amplitude=0.015):
    augmenter = Compose([
        AddGaussianNoise(min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=1.0),
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

def apply_bit_crush(audio, sample_rate, min_bit_depth=5, max_bit_depth=14, p=1.0):
    augmenter = Compose([
        BitCrush(min_bit_depth=min_bit_depth, max_bit_depth=max_bit_depth, p=p)
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


def apply_aliasing(audio, sample_rate, min_sample_rate=8000, max_sample_rate=30000, p=1.0):
    augmenter = Compose([
        Aliasing(min_sample_rate=min_sample_rate, max_sample_rate=max_sample_rate, p=p)
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound



def apply_gain_transition(audio, sample_rate, min_gain_db=-6.0, max_gain_db=6.0, p=1.0):
    augmenter = Compose([
        GainTransition(min_gain_db=min_gain_db, max_gain_db=max_gain_db, p=p)
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound


def apply_high_pass_filter(audio, sample_rate, min_cutoff_freq=100.0, max_cutoff_freq=800.0, p=1.0):
    augmenter = Compose([
        HighPassFilter(min_cutoff_freq=min_cutoff_freq, max_cutoff_freq=max_cutoff_freq, p=p)
    ])
    augmented_sound = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_sound

def apply_low_pass_filter(audio, sample_rate, min_cutoff_freq=300.0, max_cutoff_freq=3000.0, p=1.0):
    augmenter = Compose([
        LowPassFilter(min_cutoff_freq=min_cutoff_freq, max_cutoff_freq=max_cutoff_freq, p=p)
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

def apply_mp3_compression(audio, sample_rate, min_bitrate=8, max_bitrate=64, p=1.0):
    augmenter = Compose([
        Mp3Compression(min_bitrate=min_bitrate, max_bitrate=max_bitrate, p=p)
    ])
    augmented_audio = augmenter(samples=audio, sample_rate=sample_rate)
    return augmented_audio


def save_to_audio(wave, sample_rate, filename):
    sf.write(filename, wave, sr=sample_rate)


# Compression


def compress_audio_ffmpeg(input_path: str, output_path: str,
    codec: Literal[
    "mp3",
    # "aac", # Przynajmniej mi działa tylko mp3 a pozostałe tworzą puste pliki
    # "g723_1",
    # "mp2",
    # "opus",
    ] = "mp3"
    ):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:a", codec,
        output_path
    ]
    subprocess.run(cmd, check=True)

augumentation_methods = {
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
        no_augment
    ]
}