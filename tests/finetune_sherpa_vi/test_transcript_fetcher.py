from __future__ import annotations

from youtube_transcript_api._errors import RequestBlocked, TranscriptsDisabled

from finetune_sherpa_vi.transcript.fetcher import TranscriptFetchResult, TranscriptFetcher


class _FakeTranscript:
    def __init__(self, raw_data):
        self._raw_data = raw_data

    def to_raw_data(self):
        return self._raw_data


def test_fetch_returns_segments_when_transcript_exists(monkeypatch) -> None:
    fetcher = TranscriptFetcher(languages=["vi", "en"])
    monkeypatch.setattr(
        fetcher,
        "_api",
        type(
            "FakeApi",
            (),
            {"fetch": staticmethod(lambda _video_id, languages: _FakeTranscript([{"text": "A"}]))},
        )(),
    )

    result = fetcher.fetch("video-1")

    assert result == TranscriptFetchResult(segments=["A"], reason=None)


def test_fetch_returns_reason_when_transcript_is_empty(monkeypatch) -> None:
    fetcher = TranscriptFetcher(languages=["vi", "en"])
    monkeypatch.setattr(
        fetcher,
        "_api",
        type(
            "FakeApi",
            (),
            {
                "fetch": staticmethod(
                    lambda _video_id, languages: _FakeTranscript([{"text": "   "}, {"text": ""}])
                )
            },
        )(),
    )

    result = fetcher.fetch("video-2")

    assert result.segments is None
    assert result.reason == "Transcript API returned no non-empty segments."


def test_fetch_returns_reason_for_transcript_api_errors(monkeypatch) -> None:
    fetcher = TranscriptFetcher(languages=["vi", "en"])

    def _raise(_video_id, languages):
        raise TranscriptsDisabled("video-3")

    monkeypatch.setattr(fetcher, "_api", type("FakeApi", (), {"fetch": staticmethod(_raise)})())

    result = fetcher.fetch("video-3")

    assert result.segments is None
    assert result.reason == "Transcripts are disabled by the uploader."


def test_fetch_returns_reason_for_blocked_requests(monkeypatch) -> None:
    fetcher = TranscriptFetcher(languages=["vi", "en"])

    def _raise(_video_id, languages):
        raise RequestBlocked("video-4")

    monkeypatch.setattr(fetcher, "_api", type("FakeApi", (), {"fetch": staticmethod(_raise)})())

    result = fetcher.fetch("video-4")

    assert result.segments is None
    assert "Transcript request blocked by YouTube" in (result.reason or "")
