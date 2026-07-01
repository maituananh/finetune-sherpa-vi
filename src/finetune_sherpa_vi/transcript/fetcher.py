"""Transcript fetcher — retrieves captions for a single YouTube video.

Uses youtube-transcript-api (https://pypi.org/project/youtube-transcript-api/).
Tries languages in priority order (default: Vietnamese → English).

Segment format returned by the API:
    [
        {"text": "Xin chào",  "start": 0.0,  "duration": 2.5},
        {"text": "Hôm nay",   "start": 2.5,  "duration": 3.0},
        ...
    ]
Only the ``text`` field is used; timing data is discarded.
"""

from __future__ import annotations

from typing import List, Optional

from finetune_sherpa_vi.utils.logger import get_logger

log = get_logger(__name__)


class TranscriptFetcher:
    """Fetches the transcript for a YouTube video.

    Parameters:
        languages: Language codes tried in order. E.g. ['vi', 'en'].
    """

    def __init__(self, languages: List[str]) -> None:
        self._languages = languages
        self._api = self._import_api()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch(self, video_id: str) -> Optional[List[str]]:
        """Fetch transcript segments for *video_id*.

        Returns:
            List of text strings (one per segment), or ``None`` if no
            transcript is available / the video is inaccessible.
        """
        from youtube_transcript_api import (  # noqa: PLC0415
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )

        log.info("[Transcript] Fetching for video: %s (langs: %s)", video_id, self._languages)

        try:
            transcript = self._api.fetch(video_id, languages=self._languages)
            raw = transcript.to_raw_data()
            segments = [seg["text"] for seg in raw if seg.get("text", "").strip()]
            log.info("[Transcript] OK — %d segments retrieved", len(segments))
            return segments

        except TranscriptsDisabled:
            log.warning("[Transcript] SKIP %s — transcripts are disabled by uploader", video_id)
        except NoTranscriptFound:
            log.warning(
                "[Transcript] SKIP %s — no transcript found for languages %s",
                video_id,
                self._languages,
            )
        except VideoUnavailable:
            log.warning("[Transcript] SKIP %s — video unavailable (private/deleted)", video_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("[Transcript] SKIP %s — unexpected error: %s", video_id, exc)

        return None

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _import_api():
        """Import YouTubeTranscriptApi or raise a helpful error."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # noqa: PLC0415
            return YouTubeTranscriptApi()
        except ImportError as exc:
            raise ImportError(
                "youtube-transcript-api is not installed.\n"
                "Run: pip install youtube-transcript-api"
            ) from exc
