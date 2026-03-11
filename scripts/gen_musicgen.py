from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

import wimu_sonics.data.load_data as wimu
from argparse import ArgumentParser
import os

musicgen_in = wimu.get_musicgen()
musicgen_out = wimu.get_data_dir() / "examples" / "musicgen"


def read_prompts(start_idx):
    prompts = []
    for file in os.listdir(musicgen_in):
        number = int(file.split(".")[0])
        if number < start_idx:
            continue
        with open(musicgen_in / file, "r") as f:
            prompts.append(f.read())
    return prompts


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "n",
        type=int,
        help="First prompt number from which musicgen starts generate music",
    )
    args = parser.parse_args()

    prompts = read_prompts(args.n)

    model = MusicGen.get_pretrained("facebook/musicgen-melody")
    model.set_generation_params(duration=100)

    for idx, prompt in enumerate(prompts):
        waves = model.generate([prompt])
        wav = waves[0]
        audio_write(
            musicgen_out / f"{args.n + idx}",
            wav.cpu(),
            model.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )
