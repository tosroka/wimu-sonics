""" This module helps locate directories in the main project directory.
"""
from pathlib import Path
import os

# path for the project directory
project_dir = Path(__file__).resolve().parents[3]


def get_data_dir() -> Path:
    """
    Gets raw directory path.
    """
    return project_dir / 'data'


def get_results_dir() -> Path:
    """
    Gets results directory path.
    """
    return project_dir / 'results'


def get_reports_dir() -> Path:
    """
    Gets reports directory path.
    """
    return project_dir / 'reports'


def get_references_dir() -> Path:
    """
    Gets references directory path.
    """
    return project_dir / 'references'


def get_lyrics() -> Path:
    return get_data_dir() / 'prompt_egs' / 'lyrics'  # -> ...wimu-sonics/data/prompt_egs/lyrics


def get_genre() -> Path:
    return get_data_dir() / 'prompt_egs' / 'genre'  # -> ...wimu-sonics/data/prompt_egs/genre


def get_YuE() -> Path:
    return get_data_dir() / 'YuE'  # -> ...wimu-sonics/data/YuE


def get_musicgen() -> Path:
    return get_data_dir() / 'musicgen'  # -> ...wimu-sonics/data/musicgen


def get_last_number(path: Path) -> int:
    max_num = -1
    for file in os.listdir(path):
        try:
            num = int(file.split('.')[0])
        except:
            num = -1
        max_num = max(max_num, num)
    return max_num + 1
