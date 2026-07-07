"""Application config — loaded once from .env at startup.

Environment variables:
    YOUTUBE_PLAYLISTS   JSON array of playlist URLs, e.g.:
                        '["https://www.youtube.com/playlist?list=PL..."]'
    YOUTUBE_PLAYLIST_VIDEO_LIMIT
                        Maximum number of videos to take from each playlist.
                        Default: null/empty (take all). 0 also means take all.
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

    playlist_video_limit: int | None = None
    """Max videos to take from each playlist; ``None`` means take all."""


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

    raw_playlist_video_limit = os.getenv("YOUTUBE_PLAYLIST_VIDEO_LIMIT")
    playlist_video_limit: int | None
    if raw_playlist_video_limit is None:
        playlist_video_limit = None
    else:
        normalized_playlist_video_limit = raw_playlist_video_limit.strip()
        if normalized_playlist_video_limit == "":
            playlist_video_limit = None
        else:
            try:
                parsed_playlist_video_limit = int(normalized_playlist_video_limit)
            except ValueError as exc:
                raise ValueError(
                    "YOUTUBE_PLAYLIST_VIDEO_LIMIT must be an integer or empty/null."
                ) from exc

            if parsed_playlist_video_limit < 0:
                raise ValueError("YOUTUBE_PLAYLIST_VIDEO_LIMIT must be >= 0.")

            playlist_video_limit = (
                None if parsed_playlist_video_limit == 0 else parsed_playlist_video_limit
            )

    raw_langs = os.getenv("TRANSCRIPT_LANGS", "vi,en")
    transcript_langs = [lang.strip() for lang in raw_langs.split(",") if lang.strip()]

    output_dir = Path(os.getenv("TRANSCRIPT_OUTPUT_DIR", "./transcripts"))
    lock_file = Path(os.getenv("LOCK_FILE", "./video-id-locked.txt"))
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    return Config(
        playlists=playlists,
        playlist_video_limit=playlist_video_limit,
        transcript_langs=transcript_langs,
        output_dir=output_dir,
        lock_file=lock_file,
        log_level=log_level,
    )
