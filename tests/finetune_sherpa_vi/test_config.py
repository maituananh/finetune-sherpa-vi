from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from finetune_sherpa_vi.config import load_config


def test_load_config_defaults_playlist_limit_to_none() -> None:
    with patch.dict(
        os.environ,
        {
            "YOUTUBE_PLAYLISTS": '["https://example.com/playlist"]',
        },
        clear=True,
    ):
        cfg = load_config()

    assert cfg.playlist_video_limit is None


def test_load_config_treats_zero_playlist_limit_as_all() -> None:
    with patch.dict(
        os.environ,
        {
            "YOUTUBE_PLAYLISTS": '["https://example.com/playlist"]',
            "YOUTUBE_PLAYLIST_VIDEO_LIMIT": "0",
        },
        clear=True,
    ):
        cfg = load_config()

    assert cfg.playlist_video_limit is None


def test_load_config_parses_positive_playlist_limit() -> None:
    with patch.dict(
        os.environ,
        {
            "YOUTUBE_PLAYLISTS": '["https://example.com/playlist"]',
            "YOUTUBE_PLAYLIST_VIDEO_LIMIT": "7",
        },
        clear=True,
    ):
        cfg = load_config()

    assert cfg.playlist_video_limit == 7


@pytest.mark.parametrize("raw_value", ["abc", "-1"])
def test_load_config_rejects_invalid_playlist_limit(raw_value: str) -> None:
    with patch.dict(
        os.environ,
        {
            "YOUTUBE_PLAYLISTS": '["https://example.com/playlist"]',
            "YOUTUBE_PLAYLIST_VIDEO_LIMIT": raw_value,
        },
        clear=True,
    ):
        with pytest.raises(ValueError):
            load_config()
