from __future__ import annotations

import pytest

from finetune_sherpa_vi.playlist.extractor import PlaylistExtractor


def test_extract_respects_video_limit(monkeypatch) -> None:
    class FakeYoutubeDL:
        def __init__(self, _opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, _playlist_url, download=False):
            return {
                "entries": [
                    {"id": "a", "title": "Video A"},
                    {"id": "b", "title": "Video B"},
                    {"id": "c", "title": "Video C"},
                ]
            }

    monkeypatch.setattr(PlaylistExtractor, "_import_ydl", staticmethod(lambda: FakeYoutubeDL))

    videos = PlaylistExtractor(video_limit=2).extract(
        "https://www.youtube.com/playlist?list=PL123456"
    )

    assert [video.video_id for video in videos] == ["a", "b"]


def test_extract_takes_all_when_video_limit_is_none(monkeypatch) -> None:
    class FakeYoutubeDL:
        def __init__(self, _opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, _playlist_url, download=False):
            return {
                "entries": [
                    {"id": "a", "title": "Video A"},
                    {"id": "b", "title": "Video B"},
                ]
            }

    monkeypatch.setattr(PlaylistExtractor, "_import_ydl", staticmethod(lambda: FakeYoutubeDL))

    videos = PlaylistExtractor(video_limit=None).extract(
        "https://www.youtube.com/playlist?list=PL123456"
    )

    assert [video.video_id for video in videos] == ["a", "b"]


@pytest.mark.parametrize(
    ("playlist_url", "expected_message"),
    [
        (
            "https://vimeo.com/123",
            "Playlist URL must be a YouTube URL",
        ),
        (
            "https://www.youtube.com/watch?v=abc",
            "Playlist URL must contain a non-empty 'list' query parameter.",
        ),
        (
            "not-a-url",
            "Playlist URL must start with http:// or https://",
        ),
    ],
)
def test_extract_rejects_non_youtube_playlist_urls(
    playlist_url: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        PlaylistExtractor().extract(playlist_url)


def test_extract_accepts_youtube_watch_url_with_list(monkeypatch) -> None:
    class FakeYoutubeDL:
        def __init__(self, _opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, _playlist_url, download=False):
            return {"entries": [{"id": "a", "title": "Video A"}]}

    monkeypatch.setattr(PlaylistExtractor, "_import_ydl", staticmethod(lambda: FakeYoutubeDL))

    videos = PlaylistExtractor().extract(
        "https://www.youtube.com/watch?v=abc123&list=PL123456"
    )

    assert [video.video_id for video in videos] == ["a"]
