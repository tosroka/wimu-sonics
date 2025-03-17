from yt_dlp import YoutubeDL


def download_audio(filename: str, yt_id: str):
    url = f'https://www.youtube.com/watch?v={yt_id}'
    with YoutubeDL({'format': 'bestaudio', 'outtmpl': f'{filename}.mp3'}) as video:
        video.download(url)


if __name__ == "__main__":
    download_audio('real_00001', 'gl1aHhXnN1k')
