from openai import OpenAI
from openai.resources import chat
import sys
import wimu_sonics.data.load_data as l
from scripts.token import KEY


MODEL = "deepseek/deepseek-r1:free"  # free słowo kluczowe, żeby nie bulić 6/10000 $


def ask(chat: chat.Chat, content: str):
    response = chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": content}
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    client = OpenAI(
        api_key=KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    try:
        n = int(sys.argv[1])
    except Exception:
        print("Give number to generate lyrics and genres")
        sys.exit(1)
    
    for i in range(n):
        question = f"Please can you generate lyrics and key words which will describe songs lyrics?\
Every lyrics should have at least 2 verses and 2 choruses (after verse should be chorus). Key words should describe for example\
who is singing, how, mood of song, genre etc. Can you give answer like '[lyrics] empty line [key words in 1 line without ,]'?\
Also in lyrics every chorus start with [chorus] and every verse with [verse] without any number next to it."
        answer = ask(chat=client.chat, content=question)
        print(answer)
