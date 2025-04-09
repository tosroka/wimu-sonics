""" This module helps locate directories in the main project directory.
"""
from pathlib import Path

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


def get_data() -> Path:
    return project_dir / 'data'  # -> ...wimu-sonics/data


def get_lyrics() -> Path:
    return get_data() / 'prompt_egs' / 'lyrics'  # -> ...wimu-sonics/data/prompt_egs/lyrics


def get_genre() -> Path:
    return get_data() / 'prompt_egs' / 'genre'  # -> ...wimu-sonics/data/prompt_egs/genre


def get_YuE() -> Path:
    return get_data() / 'YuE'  # -> ...wimu-sonics/data/YuE

