import numpy as np
import scipy.signal
from wimu_sonics.augmentation import load_audio

def replace_spectrogram(x: np.ndarray, source_audio: str, sample_rate: float, band_start: float, band_stop: float, n_fft=2048, hop_length=512):
    """Replace a frequency band in the spectrogram of `x` with the corresponding band from `source_audio`."""
    # stft
    source_audio, sr = load_audio(source_audio,sr=sample_rate)
    f, t, Zxx_x = scipy.signal.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    _, _, Zxx_source = scipy.signal.stft(source_audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    band_indices = np.where((f >= band_start) & (f < band_stop))[0]

    # make sure source spectrogram has enough time frames
    min_time_frames = min(Zxx_x.shape[1], Zxx_source.shape[1])
    Zxx_source = Zxx_source[:, :min_time_frames]

    # Replace frequency band
    Zxx_x[band_indices, :min_time_frames] = Zxx_source[band_indices, :]

    # Inverse STFT
    _, x_modified = scipy.signal.istft(Zxx_x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Trim or pad to match original length
    x_modified = x_modified[:len(x)]
    if len(x_modified) < len(x):
        x_modified = np.pad(x_modified, (0, len(x) - len(x_modified)))

    return x_modified


special_augmentation_methods = {
    f.__qualname__: f for f in [replace_spectrogram]}
