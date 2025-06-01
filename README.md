# EVALUATING FAKE MUSIC DETECTION PERFORMANCE UNDER AUDIO AUGMENTATIONS

With the rapid advancement of generative audio models, distinguishing between human composed and generated music is becoming increasingly challenging. As a response, models for detecting fake music has been proposed. In this work, we explore the robustness of such systems under audio augmentations.

To evaluate model generalization, we constructed a dataset consisting of both real and synthetic music generated using several systems. We then apply a range of audio transformations and analyze how they affect classification accuracy. Our work identifies flaws in chosen model, underlining the difficulty of constructing such solutions.

## Recreating the experiments

1. Preparing the dataset: Download songs listed in `real_music.txt` and save them to `data/examples/real`. Or use any collection of around 20 genuine songs.
2. Setting up the environment: run the following commands (using venv is recommended)
```sh
git clone https://github.com/awsaf49/sonics
pip3 install -e .
pip3 install ./sonics
```
3. Run:
```sh
python scripts/run_experiments.py configs/paper.yaml
python scripts/run_experiments.py configs/paper_randoms.yaml
```

Output is stored in `results`, all augmented datasets are also saved as well (~35 GiB in total), you can disable this with `--no-save_datasets` parameter

4. To generate figures, run:

```sh
python3 reports/make_augmentations_heatmap.py
```

## Expanding the dataset

You can place new datasets directly in the `examples` directory. To generate more songs, you can enter scripts and run from this directory
```sh
cd scripts
./gen_data.sh <model1> <number1 <model2> <number2>
```
where `model` is yue or musicgen and `number` is the number of new songs to generate. Before doing this, make sure [yue](https://github.com/multimodal-art-projection/YuE) is installed correctly in parallel to `<project root>` and [musicgen](https://github.com/facebookresearch/audiocraft).