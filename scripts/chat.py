from openai import OpenAI
from openai.resources import chat
import wimu_sonics.data.load_data as wimu
from api_key import KEY, KEY2
import os
from argparse import ArgumentParser
from pathlib import Path


MODEL = "deepseek/deepseek-r1:free"  # free słowo kluczowe, żeby nie bulić 6/10000 $


class LimitExceeded(Exception):
    def __init__(self):
        super().__init__("Przekroczono limit 50 zapytań")


model_prompts = {
    "yue": """Please can you generate lyrics and key words which will describe songs lyrics?\
Every lyrics should have at least 2 verses and 2 choruses (after verse should be chorus). Key words should describe for example\
genre, instrument, mood, gender, timbre and etc. Can you give answer like '[lyrics] empty line [key words in 1 line without ,]'?\
Also in lyrics every chorus start with [chorus] and every verse with [verse] without any number next to it.
""",
    "musicgen": """Return a short prompt for musicgen model to generate a random song in pop genre. Return only the prompt, which is a textual description of the desired song. Don't write any other text. For example, your full response might look like 'bossa nova with soft piano and saxophone'.""",
}

musicgen_out = wimu.get_musicgen()
lyrics_path = wimu.get_lyrics()
genre_path = wimu.get_genre()


def ask(chat: chat.Chat, content: str):
    response = chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": content}]
    )
    if not response.choices:
        raise LimitExceeded()
    return response.choices[0].message.content


def save_answer(answer: str, number_of_data: int):
    answer = answer.split("\n")
    answer = [line if line.strip() else "\n" for line in answer]
    lyrics, genre = "\n".join(answer[:-2]), answer[-1]
    if not os.path.exists(lyrics_path):
        os.makedirs(lyrics_path)
    with open(lyrics_path / f"{number_of_data}.txt", "w") as f:
        f.write(lyrics)
    if not os.path.exists(genre_path):
        os.makedirs(genre_path)
    with open(genre_path / f"{number_of_data}.txt", "w") as f:
        f.write(genre)


def save_answer_musicgen(answer: str, i: int, out_path: Path):
    if not os.path.exists(out_path):
        out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / str(i)).with_suffix(".txt").open("w") as f:
        f.write(answer)


if __name__ == "__main__":
    parser = ArgumentParser("Generate prompts for models using deepseek")
    parser.add_argument("n", type=int, help="Amount of samples to generate")
    parser.add_argument(
        "model",
        choices=["yue", "musicgen"],
        help="Generate prompt for Yue or musicgen",
        default="yue",
    )

    args = parser.parse_args()

    base_url = "https://openrouter.ai/api/v1"
    clients = [OpenAI(api_key=key, base_url=base_url) for key in [KEY, KEY2]]
    prompt = model_prompts[args.model]
    answer = None
    clients = iter(clients)
    client = next(clients)
    i = 0
    last_num_yue = wimu.get_last_number(genre_path)
    last_num_musicgen = wimu.get_last_number(musicgen_out)
    while i < args.n:
        try:
            answer = ask(chat=client.chat, content=prompt)
        except LimitExceeded:
            try:
                client = next(clients)
            except StopIteration:
                raise RuntimeError("No more clients available")
            continue

        # print("Generated:", answer)
        if args.model == "yue":
            save_answer(answer, last_num_yue + i)
        elif args.model == "musicgen":
            save_answer_musicgen(answer, last_num_musicgen + i, musicgen_out)
        i += 1
    
    print(last_num_yue if args.model == "yue" else last_num_musicgen)
