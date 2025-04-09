from openai import OpenAI
from openai.resources import chat
import sys
import wimu_sonics.data.load_data as l
from api_key import KEY
import os


MODEL = "deepseek/deepseek-r1:free"  # free słowo kluczowe, żeby nie bulić 6/10000 $


def ask(chat: chat.Chat, content: str):
    response = chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": content}
        ]
    )
    return response.choices[0].message.content


def save_answer(answer: str, number_of_data: int, model: str):
    answer = answer.split('\n')
    answer = [line if line.strip() else '\n' for line in answer]
    lyrics, genre = '\n'.join(answer[:-2]), answer[-1]
    lyrics_path = l.get_lyrics(model)
    genre_path = l.get_genre(model)
    if not os.path.exists(lyrics_path):
        os.makedirs(lyrics_path)
    with open(lyrics_path  / f'lyrics_{number_of_data}.txt', 'w') as f:
        f.write(lyrics)
    if not os.path.exists(genre_path):
        os.makedirs(genre_path)
    with open(genre_path / f'genre_{number_of_data}.txt', 'w') as f:
        f.write(genre)


if __name__ == "__main__":
    client = OpenAI(
        api_key=KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    try:
        n = int(sys.argv[1])
    except Exception:
        print("Give number to generate lyrics and genres for [n] songs")
        sys.exit(1)
    try:
        model = sys.argv[2]
    except Exception:
        print("Give model name for which you want to generate data")
        sys.exit(1)
    for i in range(n):
        question = f"Please can you generate lyrics and key words which will describe songs lyrics?\
Every lyrics should have at least 2 verses and 2 choruses (after verse should be chorus). Key words should describe for example\
genre, instrument, mood, gender, timbre and etc. Can you give answer like '[lyrics] empty line [key words in 1 line without ,]'?\
Also in lyrics every chorus start with [chorus] and every verse with [verse] without any number next to it."
        answer = ask(chat=client.chat, content=question)
        save_answer(answer, i, model=model)
