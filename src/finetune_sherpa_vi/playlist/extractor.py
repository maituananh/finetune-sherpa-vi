"""Playlist extractor — retrieves all video IDs from a YouTube playlist.

Uses yt-dlp in metadata-only mode (no download) to extract the list of
video entries from a playlist URL.

Why yt-dlp instead of youtube_dl:
    youtube_dl (https://pypi.org/project/youtube_dl/) is the original library
    but has had almost no maintenance since 2021 and regularly breaks on YouTube.
    yt-dlp is the actively-maintained fork with an identical API surface.
    This module tries yt-dlp first, then falls back to youtube_dl, so either
    package satisfies the import — install at least one:
        pip install yt-dlp        ← recommended
        pip install youtube_dl    ← original (may break on modern YouTube)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from urllib.parse import parse_qs, urlparse

from finetune_sherpa_vi.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class VideoEntry:
    """Minimal metadata for a single video in a playlist."""

    video_id: str
    title: str


class PlaylistExtractor:
    """Extracts video IDs from one or more YouTube playlist URLs.

    Uses ``extract_flat=True`` which is fast: it retrieves only the list of
    entries without fetching each video's individual page.
    """

    _YDL_OPTS = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,   # metadata only — no per-video page fetch
        "skip_download": True,
        "ignoreerrors": True,   # skip unavailable / private videos silently
    }

    def __init__(self, video_limit: int | None = None) -> None:
        self._video_limit = video_limit

    def extract(self, playlist_url: str) -> List[VideoEntry]:
        """Return a list of :class:`VideoEntry` for every video in *playlist_url*.

        Raises:
            ImportError: if neither yt-dlp nor youtube_dl is installed.
            RuntimeError: if extraction returns no data at all.
        """
        self._validate_playlist_url(playlist_url)
        YoutubeDL = self._import_ydl()

        log.info("[Playlist] Extracting video list from: %s", playlist_url)

        with YoutubeDL(self._YDL_OPTS) as ydl:
            info = ydl.extract_info(playlist_url, download=False)

        if info is None:
            raise RuntimeError(
                f"Extractor returned nothing for URL: {playlist_url}"
            )

        entries = info.get("entries") or []
        videos: List[VideoEntry] = []

        for entry in entries:
            if entry is None:
                continue  # deleted / private video in the middle of a playlist
            vid_id: str = entry.get("id") or ""
            title: str = entry.get("title") or "(no title)"
            if vid_id:
                videos.append(VideoEntry(video_id=vid_id, title=title))

            if self._video_limit is not None and len(videos) >= self._video_limit:
                break

        if self._video_limit is None:
            log.info("[Playlist] Found %d videos", len(videos))
        else:
            log.info(
                "[Playlist] Found %d videos (limit=%d)",
                len(videos),
                self._video_limit,
            )
        return videos

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_playlist_url(playlist_url: str) -> None:
        """Validate that *playlist_url* looks like a YouTube playlist URL."""
        parsed = urlparse(playlist_url)
        hostname = (parsed.hostname or "").lower()

        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                "Playlist URL must start with http:// or https:// and point to YouTube."
            )

        if not hostname or "youtube.com" not in hostname:
            raise ValueError(
                "Playlist URL must be a YouTube URL (e.g. https://www.youtube.com/playlist?list=...)."
            )

        query = parse_qs(parsed.query)
        playlist_ids = [value.strip() for value in query.get("list", []) if value.strip()]
        if not playlist_ids:
            raise ValueError(
                "Playlist URL must contain a non-empty 'list' query parameter."
            )

    @staticmethod
    def _import_ydl():
        """Try yt-dlp first (actively maintained), fall back to youtube_dl."""
        try:
            from yt_dlp import YoutubeDL  # noqa: PLC0415
            log.debug("[Playlist] Using yt-dlp backend")
            return YoutubeDL
        except ImportError:
            pass
        try:
            from youtube_dl import YoutubeDL  # noqa: PLC0415
            log.debug("[Playlist] Using youtube_dl backend (yt-dlp preferred)")
            return YoutubeDL
        except ImportError as exc:
            raise ImportError(
                "No YouTube downloader found. Install one:\n"
                "  pip install yt-dlp      (recommended)\n"
                "  pip install youtube_dl  (legacy)"
            ) from exc
