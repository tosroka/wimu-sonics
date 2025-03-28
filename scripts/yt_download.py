from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


def download_audio(filename: str, yt_id: str):
    url = f'https://www.youtube.com/watch?v={yt_id}'
    try:
        with YoutubeDL({'format': 'bestaudio', 'outtmpl': f'{filename}.mp3'}) as video:
            video.download(url)
    except DownloadError as e:
        print(f"{yt_id} not available")



if __name__ == "__main__":
    download_audio('real_00001', 'gl1aahXnN1k')
