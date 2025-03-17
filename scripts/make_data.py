""" This script makes the datasets.
"""
import logging
from huggingface_hub import snapshot_download
from yt_download import download_audio
import csv
import os
import zipfile

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# %%
# Make raw data
logger.info("Making raw data.")

logger.info("Download data from huggingface.")
snapshot_download(repo_id="awsaf49/sonics", repo_type="dataset", local_dir='./data/raw')

logger.info("Extract fake songs.")
fake_folder_path = './data/raw/fake_songs'
for file in os.listdir(fake_folder_path):
    if file.endswith(".zip"):
        file_path = os.path.join(fake_folder_path, file)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('./data/raw/')
        os.remove(file_path)

logger.info("Download real songs from youtube.")
real_folder_path = "./data/raw/real_songs"
os.makedirs(real_folder_path, exist_ok=True)
with open('./data/raw/real_songs.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        filename = row['filename']
        yt_id = row['youtube_id']
        download_audio(f"{real_folder_path}/{filename}", yt_id)

# %%
# Make intermediate data
logger.info("Making intermediate data.")
# %%
# Make processed data
logger.info("Making processed data.")
# %%
# Indicate that it completed
logger.info("Done. I can't beleive that actually worked!")
