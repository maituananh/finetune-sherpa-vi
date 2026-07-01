"""Application config — loaded once from .env at startup.

Environment variables:
    YOUTUBE_PLAYLISTS   JSON array of playlist URLs, e.g.:
                        '["https://www.youtube.com/playlist?list=PL..."]'
    TRANSCRIPT_LANGS    Comma-separated language codes to try in order.
                        Default: "vi,en"
    TRANSCRIPT_OUTPUT_DIR  Directory for .trans.txt files. Default: ./transcripts
    LOCK_FILE           Path to the video-id-locked.txt file.
                        Default: ./video-id-locked.txt
    LOG_LEVEL           Logging level. Default: INFO
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env file from project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Config:
    """Immutable application configuration."""

    playlists: List[str]
    """List of YouTube playlist URLs to process."""

    transcript_langs: List[str]
    """Language codes to try when fetching transcripts (priority order)."""

    output_dir: Path
    """Directory where .trans.txt files will be written."""

    lock_file: Path
    """Path to video-id-locked.txt tracking already-processed video IDs."""

    log_level: str
    """Logging level (DEBUG, INFO, WARNING, ERROR)."""


def load_config() -> Config:
    """Read environment variables and return a validated Config instance."""
    raw_playlists = os.getenv("YOUTUBE_PLAYLISTS", "[]")
    try:
        playlists: List[str] = json.loads(raw_playlists)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"YOUTUBE_PLAYLISTS must be a valid JSON array of strings. Got: {raw_playlists!r}"
        ) from exc

    if not isinstance(playlists, list) or not all(isinstance(p, str) for p in playlists):
        raise ValueError("YOUTUBE_PLAYLISTS must be a JSON array of strings.")

    raw_langs = os.getenv("TRANSCRIPT_LANGS", "vi,en")
    transcript_langs = [lang.strip() for lang in raw_langs.split(",") if lang.strip()]

    output_dir = Path(os.getenv("TRANSCRIPT_OUTPUT_DIR", "./transcripts"))
    lock_file = Path(os.getenv("LOCK_FILE", "./video-id-locked.txt"))
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    return Config(
        playlists=playlists,
        transcript_langs=transcript_langs,
        output_dir=output_dir,
        lock_file=lock_file,
        log_level=log_level,
    )
