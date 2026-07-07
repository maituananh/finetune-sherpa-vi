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

from dataclasses import dataclass
from typing import List, Optional

from finetune_sherpa_vi.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class TranscriptFetchResult:
    """Outcome of a transcript fetch attempt."""

    segments: Optional[List[str]]
    reason: Optional[str] = None


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

    def fetch(self, video_id: str) -> TranscriptFetchResult:
        """Fetch transcript segments for *video_id*.

        Returns:
            Fetch result containing transcript segments and failure reason.
        """
        from youtube_transcript_api import (  # noqa: PLC0415
            AgeRestricted,
            CouldNotRetrieveTranscript,
            InvalidVideoId,
            NoTranscriptFound,
            RequestBlocked,
            TranscriptsDisabled,
            VideoUnavailable,
            VideoUnplayable,
        )

        log.info("[Transcript] Fetching for video: %s (langs: %s)", video_id, self._languages)

        try:
            transcript = self._api.fetch(video_id, languages=self._languages)
            raw = transcript.to_raw_data()
            segments = [seg["text"] for seg in raw if seg.get("text", "").strip()]
            if not segments:
                reason = "Transcript API returned no non-empty segments."
                log.warning("[Transcript] SKIP %s — %s", video_id, reason)
                return TranscriptFetchResult(segments=None, reason=reason)
            log.info("[Transcript] OK — %d segments retrieved", len(segments))
            return TranscriptFetchResult(segments=segments, reason=None)

        except TranscriptsDisabled:
            reason = "Transcripts are disabled by the uploader."
        except NoTranscriptFound:
            reason = f"No transcript found for languages {self._languages}."
        except VideoUnavailable:
            reason = "Video unavailable (private, deleted, or removed)."
        except VideoUnplayable:
            reason = "Video is unplayable."
        except AgeRestricted:
            reason = "Video is age-restricted."
        except InvalidVideoId:
            reason = "Video ID is invalid."
        except RequestBlocked as exc:
            reason = f"Transcript request blocked by YouTube: {exc}"
        except CouldNotRetrieveTranscript as exc:
            reason = f"Could not retrieve transcript: {exc}"
        except Exception as exc:  # noqa: BLE001
            reason = f"Unexpected error: {exc}"

        log.warning("[Transcript] SKIP %s — %s", video_id, reason)
        return TranscriptFetchResult(segments=None, reason=reason)

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
