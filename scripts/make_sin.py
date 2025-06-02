import numpy as np
import soundfile as sf
from argparse import ArgumentParser
import wimu_sonics.data.load_data as wimu
import os


sin_out = wimu.get_data_dir() / "examples" / "sin"


if __name__ == "__main__":
    os.makedirs(sin_out, exist_ok=True)

    parser = ArgumentParser("Generate sin")
    parser.add_argument("n", type=int, help="Amount of random freqs from 200Hz to 4kHz")
    args = parser.parse_args()

    sr = 22050
    duration = 10
    freqs = np.random.randint(200, 4000, size=args.n)

    a = 0.25

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    y = a * sum([np.sin(2 * np.pi * freq * t) for freq in freqs])

    i = wimu.get_last_number(sin_out)

    sf.write(sin_out / f"{i}.wav", y, sr)
