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

    def extract(self, playlist_url: str) -> List[VideoEntry]:
        """Return a list of :class:`VideoEntry` for every video in *playlist_url*.

        Raises:
            ImportError: if neither yt-dlp nor youtube_dl is installed.
            RuntimeError: if extraction returns no data at all.
        """
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

        log.info("[Playlist] Found %d videos", len(videos))
        return videos

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

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
